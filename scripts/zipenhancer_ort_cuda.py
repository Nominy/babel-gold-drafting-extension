"""Benchmark-only CUDA EP adapter; never edits the packaged ONNX model.

Dependencies are imported at construction: torch, numpy, onnxruntime-gpu;
FP16 additionally needs onnx and either onnxconverter-common or ORT's bundled
transformers.float16 converter. No inference, installation, or CPU fallback is
performed at import time. Calls must use the construction CUDA stream, device 0,
and FP32 [1, 201, 641] tensors (strided inputs are densified on the GPU).
Each result owns fresh CUDA storage; no audio or tensor data is copied to CPU.

Set ZIPENHANCER_ORT_VERBOSE=1 before running the benchmark CLI to log CUDA
kernel rejection names during strict session construction. This does not enable
CPU fallback or run a second diagnostic session.
"""

from collections import deque
import hashlib
import importlib
import json
import os
from pathlib import Path
import time
import uuid


MODEL_SHA256 = "2f18c8f7ff10a2702d6243ce1230db9e73e6804dd6cd7b20d8e191ee06924016"
SHAPE = (1, 201, 641)
INPUTS = ("noisy_mag", "noisy_pha")
OUTPUTS = ("amp_g", "pha_g")
# Block complete exporter scopes around reductions as well as normalization ops:
# keeping only ReduceMean in FP32 would still square centered values in FP16.
SENSITIVE_OPS = (
    "BatchNormalization", "InstanceNormalization", "LayerNormalization",
    "GroupNormalization", "LpNormalization", "Softmax", "LogSoftmax",
    "ReduceMean", "ReduceSum", "ReduceSumSquare", "ReduceL1", "ReduceL2",
    "ReduceLogSum", "ReduceLogSumExp", "ReduceMax", "ReduceMin", "ReduceProd",
    "Sqrt", "Pow", "Reciprocal",
)


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _preload_runtime(torch, ort, handles):
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    cuda_directory = None
    if os.name == "nt":
        # ORT 1.23's wheel requires CUDA 12, even alongside torch built for CUDA 13.
        cuda_dlls = ("cublasLt64_12.dll", "cublas64_12.dll", "cufft64_11.dll", "cudart64_12.dll")
        candidates = [torch_lib]
        candidates.extend(Path(value) / "bin" for key, value in sorted(os.environ.items())
                          if key == "CUDA_HOME" or key.startswith("CUDA_PATH"))
        cuda_directory = next((path for path in candidates
                               if all((path / name).is_file() for name in cuda_dlls)), None)
        if hasattr(os, "add_dll_directory"):
            for path in dict.fromkeys(path for path in (cuda_directory, torch_lib) if path is not None):
                handles.append(os.add_dll_directory(str(path)))
    if hasattr(ort, "preload_dlls"):
        # CUDA and cuDNN need separate search roots when torch is CUDA-13 based.
        ort.preload_dlls(cuda=True, cudnn=False, directory=str(cuda_directory) if cuda_directory else None)
        ort.preload_dlls(cuda=False, cudnn=True, msvc=False, directory=str(torch_lib))
    return {"cudaSearchDirectory": str(cuda_directory) if cuda_directory else "ORT default search",
            "cudnnSearchDirectory": str(torch_lib), "preloadApiAvailable": hasattr(ort, "preload_dlls")}


def _sort_graph(graph):
    """ORT's converter appends boundary casts; restore valid ONNX DAG order."""
    nodes = list(graph.node)
    producers = {name: index for index, node in enumerate(nodes) for name in node.output if name}
    consumers = [[] for _ in nodes]
    pending = []
    for index, node in enumerate(nodes):
        dependencies = {producers[name] for name in node.input if name in producers}
        pending.append(len(dependencies))
        for producer in dependencies:
            consumers[producer].append(index)
    ready = deque(index for index, count in enumerate(pending) if not count)
    ordered = []
    while ready:
        index = ready.popleft()
        ordered.append(nodes[index])
        for consumer in consumers[index]:
            pending[consumer] -= 1
            if not pending[consumer]:
                ready.append(consumer)
    if len(ordered) != len(nodes):
        raise ValueError("FP16 converter produced a cyclic graph")
    del graph.node[:]
    graph.node.extend(ordered)


def _fp16_model(model_path, cache_dir, ort, progress):
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("ORT FP16 conversion requires onnx; install onnx in the benchmark Python environment.") from exc
    converter = None
    failures = []
    for name in ("onnxruntime.transformers.float16", "onnxconverter_common.float16"):
        try:
            converter = importlib.import_module(name)
            break
        except ImportError as exc:
            failures.append(f"{name}: {exc}")
    if converter is None:
        raise RuntimeError("ORT FP16 requires an ONNX float16 converter. Install onnxconverter-common "
                           "in the benchmark Python environment. " + "; ".join(failures))
    converter_version = (importlib.import_module("onnxconverter_common").__version__
                         if converter.__name__.startswith("onnxconverter_common.") else ort.__version__)
    graph = onnx.load(str(model_path))
    scopes = set()
    for node in graph.graph.node:
        if node.op_type.startswith("Reduce") or "norm" in node.op_type.lower() or "norm" in node.name.lower():
            scope, separator, _ = node.name.rpartition("/")
            if separator and scope:
                scopes.add(scope + "/")
    blocked_nodes = sorted(node.name for node in graph.graph.node
                           if node.name and any(node.name.startswith(scope) for scope in scopes))
    blocked_ops = sorted(set(converter.DEFAULT_OP_BLOCK_LIST) | set(SENSITIVE_OPS))
    recipe = {
        "schema": 1, "sourceSha256": MODEL_SHA256, "precision": "fp16",
        "onnxVersion": onnx.__version__, "converter": converter.__name__,
        "converterVersion": converter_version, "converterSha256": _digest(converter.__file__),
        "keepIoTypes": True, "opBlockList": blocked_ops,
        "nodeBlockListSha256": hashlib.sha256("\n".join(blocked_nodes).encode()).hexdigest(),
        "blockedNodeCount": len(blocked_nodes), "minPositiveVal": 5.960464477539063e-08,
        "maxFiniteVal": 65504.0,
    }
    key = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    directory = Path(cache_dir).resolve() / "ort-cuda" / key
    directory.mkdir(parents=True, exist_ok=True)
    cached = directory / "model.fp16.onnx"
    manifest = directory / "conversion.json"
    if cached.resolve() == model_path.resolve():
        raise ValueError("FP16 cache must not overwrite the packaged graph")
    if cached.is_file() and manifest.is_file():
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        if metadata.get("recipe") == recipe and metadata.get("sha256") == _digest(cached):
            return cached, {**metadata, "cacheHit": True}
    progress("Converting private ONNX FP16 candidate", {"cacheKey": key, "blockedNodeCount": len(blocked_nodes)})
    started = time.perf_counter()
    converted = converter.convert_float_to_float16(
        graph, keep_io_types=True, op_block_list=blocked_ops, node_block_list=blocked_nodes,
        min_positive_val=recipe["minPositiveVal"], max_finite_val=recipe["maxFiniteVal"],
    )
    for values, names in ((converted.graph.input, INPUTS), (converted.graph.output, OUTPUTS)):
        if tuple(value.name for value in values) != names:
            raise ValueError("FP16 converter changed the ONNX boundary names")
        for value in values:
            tensor = value.type.tensor_type
            if tensor.elem_type != onnx.TensorProto.FLOAT or tuple(dim.dim_value for dim in tensor.shape.dim) != SHAPE:
                raise ValueError("FP16 converter changed the fixed FP32 graph boundary")
    half_weights = sum(value.data_type == onnx.TensorProto.FLOAT16 for value in converted.graph.initializer)
    if not half_weights:
        raise RuntimeError("FP16 conversion produced no FP16 weights; refusing a mislabeled FP32 candidate")
    _sort_graph(converted.graph)
    onnx.checker.check_model(converted)
    temporary = directory / f"{uuid.uuid4().hex}.onnx"
    temporary_manifest = directory / f"{uuid.uuid4().hex}.json"
    try:
        onnx.save(converted, str(temporary))
        metadata = {"recipe": recipe, "sha256": _digest(temporary), "fp16InitializerCount": half_weights,
                    "conversionSeconds": time.perf_counter() - started}
        temporary_manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        temporary.replace(cached)
        temporary_manifest.replace(manifest)
    finally:
        temporary.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
    return cached, {**metadata, "cacheHit": False}


class BackendNetwork:
    def __init__(self, torch, model_path: Path, precision: str, cache_dir: Path, progress):
        started = time.perf_counter()
        if precision not in ("fp32", "fp16"):
            raise ValueError(f"Unsupported ORT CUDA precision: {precision}")
        model_path = Path(model_path).resolve()
        if _digest(model_path) != MODEL_SHA256:
            raise ValueError("Packaged ZipEnhancer ONNX SHA256 mismatch")
        import numpy as np
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Install onnxruntime-gpu in the benchmark Python environment for CUDA EP.") from exc
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("ONNX Runtime CUDAExecutionProvider is unavailable; CPU fallback is forbidden")
        if not torch.cuda.is_available():
            raise RuntimeError("ORT CUDA requires a CUDA device; CPU fallback is forbidden")
        self.torch, self.np = torch, np
        self.session = None
        self.stream = torch.cuda.current_stream(device=0)
        self._dll_handles = []
        self._finished = False
        try:
            preload = _preload_runtime(torch, ort, self._dll_handles)
            conversion = None
            runtime_model = model_path
            if precision == "fp16":
                runtime_model, conversion = _fp16_model(model_path, cache_dir, ort, progress)
            options = ort.SessionOptions()
            verbose = os.environ.get("ZIPENHANCER_ORT_VERBOSE") == "1"
            options.logid = "zipenhancer-ort-cuda"
            if verbose:
                options.log_severity_level = 0
                options.log_verbosity_level = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            provider_options = {
                "device_id": "0", "user_compute_stream": str(self.stream.cuda_stream),
                "do_copy_in_default_stream": "1", "use_tf32": "0",
                "enable_cuda_graph": "0", "cudnn_conv_algo_search": "EXHAUSTIVE",
            }
            progress("Building optimized ORT CUDA session", {"precision": precision, "model": str(runtime_model),
                                                           "dllPreload": preload, "verbose": verbose})
            build_started = time.perf_counter()
            self.session = ort.InferenceSession(str(runtime_model), sess_options=options,
                                               providers=[("CUDAExecutionProvider", provider_options)])
            self.session.disable_fallback()
            providers = self.session.get_providers()
            if providers != ["CUDAExecutionProvider"]:
                raise RuntimeError(f"Unexpected ORT providers {providers}; CUDA-only placement is required")
            for values, names in ((self.session.get_inputs(), INPUTS), (self.session.get_outputs(), OUTPUTS)):
                if tuple(value.name for value in values) != names:
                    raise ValueError("Unexpected packaged ONNX input/output names")
                for value in values:
                    if value.type != "tensor(float)" or tuple(value.shape) != SHAPE:
                        raise ValueError(f"Unexpected ONNX boundary: {value.name} {value.type} {value.shape}")
            self.run_options = ort.RunOptions()
            # Stream ordering is supplied by PyTorch, without a host sync after each window.
            self.run_options.add_run_config_entry("disable_synchronize_execution_providers", "1")
            self._report = {
                "backend": "onnxruntime-cuda", "provider": "CUDAExecutionProvider", "precision": precision,
                "sourceModelSha256": MODEL_SHA256,
                "runtimeModelSha256": conversion["sha256"] if conversion else MODEL_SHA256,
                "runtimeModel": str(runtime_model), "conversion": conversion,
                "dependencies": {"torch": torch.__version__, "onnxruntime": ort.__version__, "numpy": np.__version__},
                "dllPreload": preload, "providers": providers,
                "requestedProviderOptions": provider_options, "providerOptions": self.session.get_provider_options(),
                "sessionOptions": {"graphOptimizationLevel": "ORT_ENABLE_ALL", "executionMode": "ORT_SEQUENTIAL",
                                   "session.disable_cpu_ep_fallback": "1", "verbose": verbose},
                "runOptions": {"disable_synchronize_execution_providers": "1"},
                "placement": {"cpuExecutionProviderFallbackDisabled": True, "pythonFallbackDisabled": True,
                              "nodeProfilingCollected": False},
                "buildSeconds": time.perf_counter() - build_started,
                "setupSeconds": time.perf_counter() - started,
                "limitations": ["Fixed batch 1, device 0, construction CUDA stream only; no parent graph capture.",
                                "FP16 is mixed precision with FP32 graph boundaries and protected reduction/normalization scopes.",
                                "Unsupported CUDA nodes fail session construction rather than execute on CPU.",
                                "Provider metadata and strict placement policy are reported, not a per-node profiling trace."],
            }
        except BaseException:
            self.session = None
            for handle in self._dll_handles:
                handle.close()
            self._dll_handles.clear()
            raise

    def __call__(self, magnitude, phase):
        if self._finished:
            raise RuntimeError("ORT CUDA backend is closed")
        if self.torch.cuda.current_stream(device=0).cuda_stream != self.stream.cuda_stream:
            raise RuntimeError("ORT CUDA must run on its construction PyTorch CUDA stream")
        for value in (magnitude, phase):
            if value.device.type != "cuda" or value.device.index != 0 or value.dtype != self.torch.float32:
                raise ValueError("ORT CUDA inputs must be FP32 CUDA device-0 tensors")
            if tuple(value.shape) != SHAPE:
                raise ValueError(f"ORT CUDA inputs must have shape {SHAPE}")
        # ORT pointer bindings are dense; only materialize genuinely strided inputs.
        magnitude, phase = magnitude.contiguous(), phase.contiguous()
        outputs = tuple(self.torch.empty(SHAPE, dtype=self.torch.float32, device="cuda:0") for _ in OUTPUTS)
        binding = self.session.io_binding()
        for name, value in zip(INPUTS, (magnitude, phase)):
            binding.bind_input(name, "cuda", 0, self.np.float32, SHAPE, value.data_ptr())
            # Inputs allocated on another stream must not be freed while ORT reads them.
            # The caller remains responsible for producer-stream readiness, as for torch ops.
            value.record_stream(self.stream)
        for name, value in zip(OUTPUTS, outputs):
            binding.bind_output(name, "cuda", 0, self.np.float32, SHAPE, value.data_ptr())
        self.session.run_with_iobinding(binding, self.run_options)
        return outputs

    def finish(self):
        if not self._finished:
            # Destruction cannot release ORT allocations while queued CUDA work uses them.
            try:
                self.stream.synchronize()
            finally:
                self.session = None
                for handle in self._dll_handles:
                    handle.close()
                self._dll_handles.clear()
                self._finished = True
        return self._report
