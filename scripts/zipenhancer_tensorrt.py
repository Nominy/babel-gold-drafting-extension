"""Private TensorRT 10/11 ZipEnhancer experiment; never edits the packaged graph.

Requires CUDA PyTorch and TensorRT's Python bindings/runtime/ONNX parser. TensorRT
11 FP16 additionally requires onnx and onnxconverter-common: strongly typed
networks take their precision from a separate converted graph, not builder flags.
TensorRT 10 FP16 limits half compute to Conv/MatMul and keeps every floating
layer output FP32. TensorRT 11 conversion preserves full normalization in FP32.
Builds allow a 3 GiB workspace; execution uses a dedicated stream with caller
event dependencies.
The caller owns numerical admission and must not capture this adapter externally.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import threading
import time


MODEL_SHA256 = "2f18c8f7ff10a2702d6243ce1230db9e73e6804dd6cd7b20d8e191ee06924016"
SHAPE = (1, 201, 641)
INPUTS = ("noisy_mag", "noisy_pha")
OUTPUTS = ("amp_g", "pha_g")
WORKSPACE_BYTES = 3 * 1024**3
CACHE_SCHEMA = 2
NORMALIZATION_POLICY_VERSION = 1


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _require(owner, *names):
    missing = [name for name in names if not hasattr(owner, name)]
    if missing:
        raise RuntimeError(f"Unsupported TensorRT API on {type(owner).__name__}: missing {missing}")


def _atomic_write(path, data):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _progress_monitor(trt, progress):
    _require(trt, "IProgressMonitor")

    class BuildProgress(trt.IProgressMonitor):
        def __init__(self):
            super().__init__()
            self.lock = threading.Lock()
            self.phases = {}
            self.last_report = 0.0
            self.started = time.perf_counter()
            self.error = None

        def report(self, event, phase, details, force=False):
            now = time.perf_counter()
            if self.error is not None or (not force and now - self.last_report < 1.0):
                return
            self.last_report = now
            try:
                progress(f"TensorRT build {event}: {phase}",
                         {"phase": phase, "event": event, "buildElapsedSeconds": now - self.started, **details})
            except Exception as error:
                # Do not let a Python exception disappear across a C++ callback.
                # step_complete cancels the build; the owner then re-raises it.
                self.error = error

        def phase_start(self, phase_name, parent_phase, num_steps):
            with self.lock:
                details = {"parentPhase": parent_phase, "totalSteps": int(num_steps)}
                self.phases[phase_name] = details
                self.report("phase_start", phase_name, details, force=not parent_phase)

        def step_complete(self, phase_name, step):
            with self.lock:
                self.report("step_complete", phase_name,
                            {**self.phases.get(phase_name, {}), "step": int(step)})
                return self.error is None

        def phase_finish(self, phase_name):
            with self.lock:
                details = self.phases.pop(phase_name, {})
                self.report("phase_finish", phase_name, details, force=not details.get("parentPhase"))

    return BuildProgress()


class BackendNetwork:
    def __init__(self, torch, model_path: Path, precision: str, cache_dir: Path, progress):
        started = time.perf_counter()
        if precision not in ("fp32", "fp16"):
            raise ValueError(f"Unsupported TensorRT precision: {precision}")
        model_path = Path(model_path).resolve(strict=True)
        graph = model_path.read_bytes()
        if _sha256(graph) != MODEL_SHA256:
            raise ValueError("Packaged ZipEnhancer ONNX SHA256 mismatch; refusing TensorRT build")
        cache_root = Path(cache_dir).resolve()
        if cache_root == model_path.parent or model_path.parent in cache_root.parents:
            raise ValueError("TensorRT cache must be outside the packaged models directory")
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT requires CUDA; CPU inference is not supported")

        import tensorrt as trt

        major = int(trt.__version__.split(".", 1)[0])
        if major not in (10, 11):
            raise RuntimeError(f"TensorRT 10 or 11 is required, got {trt.__version__}")
        _require(trt, "Logger", "Builder", "OnnxParser", "Runtime", "BuilderFlag", "MemoryPoolType",
                 "TensorIOMode", "TensorLocation", "TensorFormat", "float32", "init_libnvinfer_plugins")
        _require(trt.BuilderFlag, "TF32")
        _require(trt.MemoryPoolType, "WORKSPACE")
        if major == 10 and precision == "fp16":
            _require(trt.BuilderFlag, "FP16", "OBEY_PRECISION_CONSTRAINTS")

        self.torch, self.trt = torch, trt
        self.device = torch.device("cuda", torch.cuda.current_device())
        self._stream = torch.cuda.Stream(device=self.device)
        self._closed = False
        self.context = self.engine = self.runtime = None
        self.logger = trt.Logger(trt.Logger.WARNING)
        if not trt.init_libnvinfer_plugins(self.logger, ""):
            raise RuntimeError("TensorRT plugin initialization failed")
        properties = torch.cuda.get_device_properties(self.device)
        dependencies = {"torch": str(torch.__version__), "tensorrt": str(trt.__version__)}
        conversion_seconds = 0.0
        conversion = None
        policy = "strongly-typed-fp32" if major == 11 else "weakly-typed-fp32"
        progress("TensorRT setup starting", {"precision": precision, "tensorrt": str(trt.__version__),
                                               "device": properties.name})
        if precision == "fp16" and major == 11:
            import onnx
            from onnxconverter_common import float16

            dependencies.update({"onnx": onnx.__version__,
                                 "onnxconverter-common": importlib.metadata.version("onnxconverter-common")})
            # Zero/infinity disable the converter's extra small/large-value clipping;
            # weight conversion uses the actual IEEE FP16 rounding/overflow rules.
            conversion = {"keep_io_types": True, "disable_shape_infer": False,
                          "min_positive_val": 0.0, "max_finite_val": "infinity",
                          "op_block_list": "converter-default"}
            progress("Converting private strongly typed FP16 ONNX graph", {"dependencies": dependencies})
            conversion_started = time.perf_counter()
            source_model = onnx.load_model_from_string(graph)
            scopes = {node.name.rsplit("/", 1)[0] for node in source_model.graph.node
                      if node.op_type == "ReduceMean"}
            if len(scopes) != 24:
                raise ValueError(f"Expected 24 normalization scopes in packaged ONNX, got {len(scopes)}")
            blocked = sorted(node.name for node in source_model.graph.node
                             if node.name.rsplit("/", 1)[0] in scopes)
            conversion["fp32NormalizationNodes"] = blocked
            model = float16.convert_float_to_float16(
                source_model, keep_io_types=True, node_block_list=blocked,
                min_positive_val=0.0, max_finite_val=float("inf"), disable_shape_infer=False,
            )
            onnx.checker.check_model(model)
            graph = model.SerializeToString()
            del model, source_model
            conversion_seconds = time.perf_counter() - conversion_started
            policy = "strongly-typed-fp16-fp32-normalization-v1"
        elif precision == "fp16":
            policy = "weakly-typed-fp16-linear-only-fp32-outputs-v3"

        # Parse before cache lookup so identity includes the actual parsed layer
        # names, output types and full normalization scope coverage.
        with torch.cuda.device(self.device):
            builder, network, parser, parse_seconds = self._parse(graph, progress)
            constraints = self._normalization_constraints(network, major, precision)
            linear_constraints = (self._linear_fp16_constraints(network)
                                  if major == 10 and precision == "fp16" else None)
            if linear_constraints is not None:
                progress("Applying TensorRT controlled FP16 compute policy",
                         {key: value for key, value in linear_constraints.items() if key != "layers"})

        identity = {
            "schema": CACHE_SCHEMA, "sourceSha256": MODEL_SHA256, "graphSha256": _sha256(graph),
            "precision": precision, "precisionPolicy": policy, "tensorrt": str(trt.__version__),
            "cuda": str(torch.version.cuda), "deviceName": properties.name,
            "computeCapability": [properties.major, properties.minor],
            "workspaceBytes": WORKSPACE_BYTES, "builderOptimizationLevel": 3, "tf32": False,
            "shape": list(SHAPE), "inputs": list(INPUTS), "outputs": list(OUTPUTS),
            "conversion": conversion, "dependencies": dependencies,
            "normalizationConstraints": constraints,
        }
        if linear_constraints is not None:
            identity["precisionConstraints"] = linear_constraints
        key = _sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode("utf-8"))
        directory = cache_root / "tensorrt" / key
        directory.mkdir(parents=True, exist_ok=True)
        engine_path, manifest_path = directory / "engine.plan", directory / "manifest.json"
        converted_path = directory / "graph.fp16.onnx"
        build_seconds = 0.0
        cached = engine_path.exists() or manifest_path.exists() or converted_path.exists()
        if cached:
            if not engine_path.is_file() or not manifest_path.is_file():
                raise ValueError(f"Incomplete TensorRT cache: {directory}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("identity") != identity or manifest.get("cacheKey") != key:
                raise ValueError(f"TensorRT cache identity mismatch: {manifest_path}")
            plan = engine_path.read_bytes()
            if manifest.get("engineSha256") != _sha256(plan):
                raise ValueError(f"TensorRT engine cache SHA256 mismatch: {engine_path}")
            if conversion is not None:
                if not converted_path.is_file() or _sha256(converted_path.read_bytes()) != identity["graphSha256"]:
                    raise ValueError(f"TensorRT converted graph cache SHA256 mismatch: {converted_path}")
            progress("Loading verified TensorRT engine cache", {"cacheKey": key, "engineSha256": _sha256(plan)})
        else:
            with torch.cuda.device(self.device):
                plan, build_seconds = self._build(builder, network, parser, major, precision, parse_seconds, progress)
            manifest = {"identity": identity, "cacheKey": key, "engineSha256": _sha256(plan),
                        "parseSeconds": parse_seconds, "buildSeconds": build_seconds,
                        "conversionSeconds": conversion_seconds}
            if conversion is not None:
                _atomic_write(converted_path, graph)
            _atomic_write(engine_path, plan)
            _atomic_write(manifest_path, (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode("utf-8"))
        # The parser owns parsed weights; retain it until engine construction ends.
        del parser, network, builder

        with torch.cuda.device(self.device):
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(plan)
            if self.engine is None:
                raise RuntimeError("TensorRT engine deserialization failed")
            self._validate_engine()
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("TensorRT execution context creation failed")
            _require(self.context, "set_tensor_address", "execute_async_v3", "get_tensor_shape", "get_tensor_strides")
            for name in INPUTS + OUTPUTS:
                if tuple(self.context.get_tensor_shape(name)) != SHAPE:
                    raise RuntimeError(f"TensorRT context shape mismatch for {name}")
                if tuple(self.context.get_tensor_strides(name)) != (201 * 641, 641, 1):
                    raise RuntimeError(f"TensorRT context requires unsupported strides for {name}")
        self._details = {
            "provider": "TensorRT", "precision": precision, "precisionPolicy": policy,
            "cacheHit": cached, "cacheKey": key, "cacheDirectory": str(directory),
            "sourceSha256": MODEL_SHA256, "graphSha256": identity["graphSha256"],
            "engineSha256": manifest["engineSha256"], "dependencies": dependencies,
            "cuda": identity["cuda"], "deviceName": properties.name,
            "computeCapability": identity["computeCapability"], "tf32": False,
            "workspaceBytes": WORKSPACE_BYTES, "builderOptimizationLevel": 3,
            "normalizationConstraints": constraints, "executionStream": "dedicated-nondefault",
            "conversionSeconds": conversion_seconds, "parseSeconds": parse_seconds,
            "buildSeconds": build_seconds, "cachedBuildSeconds": manifest["buildSeconds"],
            "setupSeconds": time.perf_counter() - started,
            "api": {"major": major, "execution": "set_tensor_address/execute_async_v3",
                    "networkTyping": "strong" if major == 11 else "weak",
                    "io": "CUDA linear FP32 fixed [1,201,641]", "externalGraphCapture": False},
        }
        if linear_constraints is not None:
            self._details["precisionConstraints"] = linear_constraints
        progress("TensorRT engine ready", dict(self._details))

    def _parse(self, graph, progress):
        trt = self.trt
        builder = trt.Builder(self.logger)
        _require(builder, "create_network", "create_builder_config", "build_serialized_network")
        # TensorRT 10 is always explicit-batch; 11 additionally always strongly typed.
        network = builder.create_network(0)
        if network is None:
            raise RuntimeError("TensorRT network creation failed")
        parser = trt.OnnxParser(network, self.logger)
        progress("Parsing verified private TensorRT ONNX graph", {"graphSha256": _sha256(graph)})
        started = time.perf_counter()
        if not parser.parse(graph):
            errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
            raise RuntimeError(f"TensorRT ONNX parsing failed:\n{errors}")
        parse_seconds = time.perf_counter() - started
        for names, count, getter in ((INPUTS, network.num_inputs, network.get_input),
                                      (OUTPUTS, network.num_outputs, network.get_output)):
            tensors = [getter(index) for index in range(count)]
            if count != len(names) or {tensor.name for tensor in tensors} != set(names):
                raise ValueError("TensorRT parsed graph I/O names differ from the packaged model contract")
            for tensor in tensors:
                if tuple(tensor.shape) != SHAPE or tensor.dtype != trt.float32:
                    raise ValueError(f"TensorRT parsed graph must expose fixed FP32 I/O: {tensor.name}")
        return builder, network, parser, parse_seconds

    def _normalization_constraints(self, network, major, precision):
        trt = self.trt
        _require(trt, "LayerType")
        _require(trt.LayerType, "REDUCE", "ELEMENTWISE", "UNARY")
        layers = [network.get_layer(index) for index in range(network.num_layers)]
        scopes = set()
        for layer in layers:
            # get_layer returns generic ILayer in TensorRT 10.16. The verified
            # graph's original ReduceMean names identify these scopes without
            # accessing IReduceLayer-only attributes on that generic wrapper.
            if layer.type == trt.LayerType.REDUCE:
                names = [layer.name, *(layer.get_output(index).name for index in range(layer.num_outputs))]
                for name in names:
                    if name.startswith("/model/") and "/ReduceMean" in name:
                        scopes.add(name.rsplit("/", 1)[0])
        if len(scopes) != 24:
            raise ValueError(f"TensorRT parser exposed {len(scopes)} normalization scopes; expected all 24")
        arithmetic = {trt.LayerType.REDUCE, trt.LayerType.ELEMENTWISE, trt.LayerType.UNARY}
        constrained = []
        covered = set()
        apply = major == 10 and precision == "fp16"
        for layer in layers:
            outputs = [layer.get_output(index) for index in range(layer.num_outputs)]
            names = [layer.name, *(tensor.name for tensor in outputs)]
            matched = {scope for scope in scopes if any(name.startswith(scope + "/") for name in names)}
            if not matched or layer.type not in arithmetic:
                continue
            floating = [(index, tensor) for index, tensor in enumerate(outputs)
                        if tensor.dtype in (trt.float32, trt.float16)]
            if not floating:
                continue
            # TRT10's complete floating-layer policy is applied separately.
            covered.update(matched)
            constrained.append({"name": layer.name, "outputs": [tensor.name for _, tensor in floating]})
        if covered != scopes:
            raise ValueError(f"TensorRT normalization precision policy missed scopes: {sorted(scopes - covered)}")
        return {"version": NORMALIZATION_POLICY_VERSION,
                "mode": "OBEY_PRECISION_CONSTRAINTS" if apply else "graph-FP32-normalization",
                "computeDtype": "float32", "outputDtype": "float32",
                "scopeCount": len(scopes), "scopes": sorted(scopes),
                "layerCount": len(constrained), "layers": sorted(constrained, key=lambda layer: layer["name"])}

    def _linear_fp16_constraints(self, network):
        trt = self.trt
        _require(trt.LayerType, "MATRIX_MULTIPLY", "CONVOLUTION")
        half_compute = {trt.LayerType.MATRIX_MULTIPLY, trt.LayerType.CONVOLUTION}
        layers = []
        half_count = full_count = output_count = 0
        by_type = {}
        for index in range(network.num_layers):
            layer = network.get_layer(index)
            outputs = [(output_index, layer.get_output(output_index))
                       for output_index in range(layer.num_outputs)]
            floating = [(output_index, tensor) for output_index, tensor in outputs
                        if tensor.dtype in (trt.float32, trt.float16)]
            if not floating:
                continue
            use_half = layer.type in half_compute
            layer.precision = trt.float16 if use_half else trt.float32
            for output_index, _ in floating:
                layer.set_output_type(output_index, trt.float32)
            half_count += int(use_half)
            full_count += int(not use_half)
            output_count += len(floating)
            layer_type = str(layer.type)
            by_type[layer_type] = by_type.get(layer_type, 0) + 1
            layers.append({"name": layer.name, "type": layer_type,
                           "computeDtype": "float16" if use_half else "float32",
                           "outputDtype": "float32",
                           "outputs": [tensor.name for _, tensor in floating]})
        if not half_count or not full_count:
            raise ValueError("Controlled TensorRT FP16 policy requires both linear and non-linear floating layers")
        return {"version": 3, "mode": "OBEY_PRECISION_CONSTRAINTS",
                "fp16ComputeTypes": ["MATRIX_MULTIPLY", "CONVOLUTION"],
                "otherFloatingComputeDtype": "float32", "allFloatingOutputsDtype": "float32",
                "fp16ComputeLayerCount": half_count, "fp32ComputeLayerCount": full_count,
                "floatingOutputCount": output_count, "layerCount": len(layers),
                "layerTypeCounts": by_type, "layers": sorted(layers, key=lambda entry: entry["name"])}

    def _build(self, builder, network, parser, major, precision, parse_seconds, progress):
        trt = self.trt
        config = builder.create_builder_config()
        if config is None:
            raise RuntimeError("TensorRT builder configuration creation failed")
        _require(config, "set_memory_pool_limit", "clear_flag", "builder_optimization_level",
                 "profile_stream", "progress_monitor")
        self._monitor = _progress_monitor(trt, progress)
        config.progress_monitor = self._monitor
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_BYTES)
        config.builder_optimization_level = 3
        config.clear_flag(trt.BuilderFlag.TF32)
        if major == 10 and precision == "fp16":
            _require(config, "set_flag")
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        config.profile_stream = self._stream.cuda_stream
        progress("Building TensorRT engine", {"precision": precision, "workspaceBytes": WORKSPACE_BYTES,
                                               "builderOptimizationLevel": 3, "parseSeconds": parse_seconds,
                                               "stronglyTyped": major == 11, "tf32": False})
        started = time.perf_counter()
        serialized = builder.build_serialized_network(network, config)
        build_seconds = time.perf_counter() - started
        if self._monitor.error is not None:
            raise self._monitor.error
        if serialized is None:
            errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
            raise RuntimeError(f"TensorRT engine build failed; inspect TensorRT logger output.\n{errors}")
        plan = bytes(serialized)
        progress("TensorRT engine build complete", {"buildSeconds": build_seconds, "engineBytes": len(plan)})
        return plan, build_seconds

    def _validate_engine(self):
        trt, engine = self.trt, self.engine
        _require(engine, "num_io_tensors", "get_tensor_name", "get_tensor_mode", "get_tensor_dtype",
                 "get_tensor_shape", "get_tensor_location", "get_tensor_format", "get_tensor_vectorized_dim",
                 "create_execution_context")
        names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
        if len(names) != 4 or set(names) != set(INPUTS + OUTPUTS):
            raise ValueError(f"TensorRT engine I/O names mismatch: {names}")
        for name in names:
            mode = trt.TensorIOMode.INPUT if name in INPUTS else trt.TensorIOMode.OUTPUT
            if (engine.get_tensor_mode(name) != mode or engine.get_tensor_dtype(name) != trt.float32
                    or tuple(engine.get_tensor_shape(name)) != SHAPE
                    or engine.get_tensor_location(name) != trt.TensorLocation.DEVICE
                    or engine.get_tensor_format(name) != trt.TensorFormat.LINEAR
                    or engine.get_tensor_vectorized_dim(name) != -1):
                raise ValueError(f"TensorRT engine violates fixed linear CUDA FP32 I/O contract: {name}")

    def __call__(self, magnitude, phase):
        if self._closed:
            raise RuntimeError("TensorRT backend has already been finished")
        torch = self.torch
        for name, tensor in zip(INPUTS, (magnitude, phase)):
            if (not isinstance(tensor, torch.Tensor) or tensor.device != self.device
                    or tensor.dtype != torch.float32 or tuple(tensor.shape) != SHAPE):
                raise ValueError(f"{name} must be a CUDA FP32 tensor on {self.device} with shape {SHAPE}")
        with torch.cuda.device(self.device):
            caller_stream = torch.cuda.current_stream(self.device)
            inputs = (magnitude.contiguous(), phase.contiguous())
            outputs = tuple(torch.empty(SHAPE, device=self.device, dtype=torch.float32) for _ in OUTPUTS)
            # Event dependencies preserve caller ordering without the default
            # stream enqueue synchronization or any per-call host synchronization.
            self._stream.wait_stream(caller_stream)
            with torch.cuda.stream(self._stream):
                for name, tensor in zip(INPUTS + OUTPUTS, inputs + outputs):
                    tensor.record_stream(self._stream)
                    if not self.context.set_tensor_address(name, tensor.data_ptr()):
                        raise RuntimeError(f"TensorRT rejected tensor address for {name}")
                if not self.context.execute_async_v3(stream_handle=self._stream.cuda_stream):
                    raise RuntimeError("TensorRT execute_async_v3 failed")
            caller_stream.wait_stream(self._stream)
        # Each call has distinct output storage; later inference cannot overwrite it.
        return outputs

    def finish(self) -> dict:
        if not self._closed:
            self._stream.synchronize()
            self.context = None
            self.engine = None
            self.runtime = None
            self.logger = None
            self._monitor = None
            self._stream = None
            self._closed = True
        return dict(self._details)
