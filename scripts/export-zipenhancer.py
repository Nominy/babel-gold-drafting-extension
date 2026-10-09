#!/usr/bin/env python3
"""Strict original-weight, fixed-641-frame ZipEnhancer ONNX export and validation.

Requires torch 2.9.1, onnx 1.19.1, onnxruntime 1.23.2, NumPy 1.26.4,
soundfile, and ModelScope 1.22.0. Run with the isolated reference-package path;
no publisher ONNX, quantization, approximate math, or audio is packaged.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import math
import sys
import warnings
from pathlib import Path
sys.dont_write_bytecode = True
from zipenhancer_onnx_lowering import (
    lower_attention, lower_standard_graph, specialize_static_graph,
    MAX_TIME_FRAMES, MAX_SCORE_BUFFER_BYTES, MAX_CONCAT_INPUTS)

CHECKPOINT_SHA256 = "b18896915e27a821585584221d0c0820f35e12145315ae3f1e73ccd5a68d195f"
BASELINE_SHA256 = "9d17062398a1bed8ef0c0debe7cf1d9daac05798be3ae0e3087e1a3288df9c22"
MODEL_ID = "zipenhancer-2026-10-08-r1"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    gold = Path(__file__).resolve().parents[1]
    trial = gold.parents[1].parent / "babel_experiment/artifacts/maxine-afx-live-20261008"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=trial / "zipenhancer")
    parser.add_argument("--packages", type=Path, default=trial / "clearvoice-packages")
    parser.add_argument("--validation-dir", type=Path, default=trial / "zipenhancer-export")
    parser.add_argument("--graph", type=Path, default=gold / "models/zipenhancer.onnx")
    parser.add_argument("--metadata", type=Path, default=gold / "src/core/audio-enhancement-model.json")
    parser.add_argument("--baseline-graph", type=Path, default=trial / "network-performance/baseline-gold/models/zipenhancer.onnx")
    parser.add_argument("--validate-existing", action="store_true", help="Validate the already-exported graph without regenerating or rewriting it")
    args = parser.parse_args()
    if sha256(args.baseline_graph) != BASELINE_SHA256:
        raise ValueError("Frozen current-graph digest mismatch")
    if args.graph.resolve() == args.baseline_graph.resolve():
        raise ValueError("The frozen reference graph must not be overwritten")
    sys.path.insert(0, str(args.packages.resolve()))
    import numpy as np
    import onnx
    import onnxruntime as ort
    import soundfile as sf
    import torch
    from modelscope.models.audio.ans.zipenhancer import ZipEnhancer, AttrDict, mag_pha_stft, mag_pha_istft
    warnings.resetwarnings()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("Reference validation requires CUDA; no CPU reference fallback")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint_path = args.reference_root / "pytorch_model.bin"
    if sha256(checkpoint_path) != CHECKPOINT_SHA256:
        raise ValueError("Original checkpoint digest mismatch")
    config_path = args.reference_root / "configuration.json"
    config = json.loads(config_path.read_text())["model"]
    model = ZipEnhancer(AttrDict(config))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["generator"], strict=True)
    model.eval()
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != 2044436:
        raise ValueError(f"Unexpected learned parameter count: {parameters}")

    class ExportOutputs(torch.nn.Module):
        def __init__(self, original):
            super().__init__()
            self.model = original

        def forward(self, noisy_mag, noisy_pha):
            amp, pha, _, _, _ = self.model(noisy_mag, noisy_pha)
            return amp, pha

    args.graph.parent.mkdir(parents=True, exist_ok=True)
    args.validation_dir.mkdir(parents=True, exist_ok=True)
    if args.validate_existing:
        graph = onnx.load(str(args.graph))
    else:
        # Original stable FP32 lowering is retained; fixed tracing specializes
        # shapes and exact indexing, never learned arithmetic.
        with torch.inference_mode():
            torch.onnx.export(
                ExportOutputs(lower_attention(copy.deepcopy(model), torch)).eval(),
                (torch.ones(1, 201, MAX_TIME_FRAMES), torch.zeros(1, 201, MAX_TIME_FRAMES)),
                str(args.graph), opset_version=17, dynamo=False,
                input_names=["noisy_mag", "noisy_pha"], output_names=["amp_g", "pha_g"],
                do_constant_folding=True,
            )
        graph = specialize_static_graph(lower_standard_graph(onnx.load(str(args.graph)), onnx), onnx)
        graph.producer_name = "Babel strict-weight ZipEnhancer exporter"
        graph.doc_string = "Original ModelScope ZipEnhancer 2.044436M; strict original FP32 checkpoint; exact fixed-641 shape/index specialization."
        # Stable explicit names bind dispatch evidence to the actual graph.
        for i, node in enumerate(graph.graph.node):
            if not node.name:
                node.name = f"zipenhancer/{node.op_type}/{i}"
        onnx.save(graph, str(args.graph))
    onnx.checker.check_model(graph, full_check=True)
    for value in [*graph.graph.input, *graph.graph.output]:
        if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT or any(
                not dim.HasField("dim_value") or dim.dim_value != size
                for dim, size in zip(value.type.tensor_type.shape.dim, [1, 201, MAX_TIME_FRAMES])) or (
                len(value.type.tensor_type.shape.dim) != 3):
            raise ValueError(f"Non-static runtime graph boundary: {value.name}")
    ops = dict(sorted(collections.Counter(n.op_type for n in graph.graph.node).items()))
    initializers = {i.name for i in graph.graph.initializer}
    aliases = {"Identity", "Reshape", "Squeeze", "Unsqueeze"}
    inferred = onnx.shape_inference.infer_shapes(graph, data_prop=True)
    dtype = {v.name: onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type).lower()
             for v in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]
             if v.type.tensor_type.elem_type}
    dtype.update({v.name: onnx.TensorProto.DataType.Name(v.data_type).lower() for v in graph.graph.initializer})
    ancestry = {name: "activation" for name in ["noisy_mag", "noisy_pha"]}
    ancestry.update({v.name: "constant" if v.data_type in {6, 7, 9} else "learned-weight"
                     for v in graph.graph.initializer})
    # Non-scalar FP32 literals include the relative-position embedding table;
    # indexing those is real positional tensor computation, not shape metadata.
    value_constants = {
        output for node in graph.graph.node if node.op_type == "Constant"
        for attr in node.attribute if attr.name == "value"
        if attr.t.data_type == onnx.TensorProto.FLOAT and math.prod(attr.t.dims) > 1
        for output in node.output
    }
    placement_nodes = []
    learned_nodes = []
    for node in graph.graph.node:
        input_ancestry = [ancestry.get(name, "activation") for name in node.input if name]
        metadata_only = all(value in {"metadata", "constant"} for value in input_ancestry) and not any(
            name in value_constants for name in node.input)
        integer_output = all(dtype.get(name) in {"int64", "int32", "bool"} for name in node.output)
        if node.op_type == "Constant":
            placement, output_ancestry = "constant", "constant"
            reason = "Graph literal; no neural computation"
        elif node.op_type in {"Shape", "Size"} or metadata_only and (
                integer_output or "metadata" in input_ancestry):
            placement, output_ancestry = "host-metadata", "metadata"
            reason = "Shape/index arithmetic or literal shape-fill only; no neural tensor values"
        elif node.op_type in aliases:
            placement = "tensor-alias"
            output_ancestry = "activation" if not metadata_only else input_ancestry[0]
            reason = "Storage alias only; no neural arithmetic"
        else:
            placement, output_ancestry = "gpu", "activation"
            reason = "Neural/positional tensor arithmetic requires observed WebGPU dispatch"
        for name in node.output:
            ancestry[name] = output_ancestry
        tensors = lambda names: [{"name": name, "dtype": dtype.get(name, "unknown"),
                                  "ancestry": ancestry.get(name, "activation")} for name in names if name]
        placement_nodes.append({"name": node.name, "opType": node.op_type, "placement": placement,
                                "inputs": tensors(node.input), "outputs": tensors(node.output), "reason": reason})
        if node.op_type in {"Conv", "MatMul", "Gemm"}:
            learned_nodes.append({"name": node.name, "opType": node.op_type, "outputs": list(node.output),
                                  "initializers": [name for name in node.input if name in initializers]})
    if any(t["dtype"] == "unknown" for n in placement_nodes if n["placement"] == "gpu"
           for t in n["outputs"]):
        raise ValueError("GPU placement inventory has an unproven output dtype")
    required_gpu_buffer_bytes = MAX_SCORE_BUFFER_BYTES
    if any(n.op_type not in ops or n.domain not in {"", "ai.onnx"} for n in graph.graph.node):
        raise ValueError("Only standard ONNX operators may be packaged")
    # CPU ORT is the independent ONNX numerical oracle, not the production neural
    # execution provider. Original PyTorch reference remains CUDA-only.
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(args.graph), options, providers=["CPUExecutionProvider"])
    baseline_session = ort.InferenceSession(str(args.baseline_graph), options, providers=["CPUExecutionProvider"])
    model = model.to("cuda").eval()
    precise_model = copy.deepcopy(model).double().eval()
    lanes = {}
    scales = {}
    for speaker in [1, 2]:
        lane, rate = sf.read(args.reference_root / f"speaker-{speaker}-input16.wav", dtype="float32")
        if rate != 16000 or lane.ndim != 1:
            raise ValueError("Validation requires original mono 16k trial inputs")
        lanes[speaker] = lane
        scales[speaker] = math.sqrt(len(lane) / float(np.sum(lane.astype(np.float64) ** 2)))
    cases = []
    # Keep every original real excerpt and quiet/crop boundary case. Short
    # waveforms are zero-padded before STFT exactly as the production runtime,
    # not spectrally padded or inferred with a shorter attention context.
    for name, speaker, start, duration in [
        ("russian-speech-4s", 1, 25.0, 4.0),
        ("russian-other-speaker-3s", 2, 32.0, 3.0),
        ("quiet-final-mgm-4s", 1, 59.0, 4.0),
        ("quiet-final-short-1.3s", 1, 61.7, 1.3),
        ("russian-even-time-axis", 2, 35.0, 2.99375),
    ]:
        samples = lanes[speaker][round(start * 16000):round((start + duration) * 16000)].copy()
        cases.append((name, samples, scales[speaker]))
    rng = np.random.default_rng(20261008)
    cases.append(("silence-4s", np.zeros(64000, dtype="float32"), 1.0))
    boundary = np.zeros(32000, dtype="float32")
    boundary[8000:24000] = rng.normal(0, 0.005, 16000).astype("float32")
    cases.append(("noise-silence-boundaries-2s", boundary, 1.0))
    boundary = np.zeros(64000, dtype="float32")
    boundary[16000:48000] = lanes[1][25 * 16000:27 * 16000]
    cases.append(("speech-silence-boundaries-4s", boundary, scales[1]))
    results = []
    diagnostics = []
    native_gate_failures = []

    def metrics(reference, actual):
        reference = reference.astype(np.float64)
        actual = actual.astype(np.float64)
        difference = actual - reference
        rms_error = float(np.sqrt(np.mean(difference ** 2)))
        rms = float(np.sqrt(np.mean(reference ** 2)))
        return {"maxAbsError": float(np.max(np.abs(difference))), "rmse": rms_error,
                "relativeRmse": rms_error / max(rms, 1e-12),
                "referenceRms": rms, "actualRms": float(np.sqrt(np.mean(actual ** 2))),
                "snrDb": 20 * math.log10(max(rms, 1e-12) / max(rms_error, 1e-20))}

    with torch.inference_mode():
        for name, samples, scale in cases:
            padded = np.zeros((MAX_TIME_FRAMES - 1) * 100, dtype="float32")
            if len(samples) > len(padded):
                raise ValueError(f"Validation excerpt exceeds runtime window: {name}")
            padded[:len(samples)] = samples
            wave = torch.from_numpy(padded)[None].to("cuda") * scale
            mag, pha, _ = mag_pha_stft(wave, 400, 100, 400, compress_factor=0.3, center=True)
            if tuple(mag.shape) != (1, 201, MAX_TIME_FRAMES):
                raise ValueError(f"Validation input differs from runtime frame contract: {name}")
            ref_amp, ref_pha, _, _, _ = model(mag, pha)
            feeds = {"noisy_mag": mag.cpu().numpy(), "noisy_pha": pha.cpu().numpy()}
            out_amp, out_pha = session.run(None, feeds)
            baseline_amp, baseline_pha = baseline_session.run(None, feeds)
            reference = mag_pha_istft(ref_amp, ref_pha, 400, 100, 400, compress_factor=0.3, center=True)[0].cpu().numpy()[:len(samples)] / scale
            actual = mag_pha_istft(torch.from_numpy(out_amp).to("cuda"), torch.from_numpy(out_pha).to("cuda"),
                                  400, 100, 400, compress_factor=0.3, center=True)[0].cpu().numpy()[:len(samples)] / scale
            baseline_audio = mag_pha_istft(torch.from_numpy(baseline_amp).to("cuda"), torch.from_numpy(baseline_pha).to("cuda"),
                                          400, 100, 400, compress_factor=0.3, center=True)[0].cpu().numpy()[:len(samples)] / scale
            baseline_amplitude = metrics(baseline_amp, out_amp)
            baseline_waveform = metrics(baseline_audio, actual)
            baseline_phase_error = np.angle(np.exp(1j * (out_pha - baseline_pha)))
            baseline_phase_rmse = float(np.sqrt(np.mean(baseline_phase_error ** 2)))
            if not np.array_equal(baseline_amp, out_amp) or not np.array_equal(baseline_pha, out_pha) or (
                    not np.array_equal(baseline_audio, actual)):
                raise ValueError(f"Frozen ONNX bit-exact fixed-input parity failed: {name}: {baseline_amplitude} {baseline_waveform} {baseline_phase_rmse}")
            amplitude = metrics(ref_amp.cpu().numpy(), out_amp)
            # Phase has a +/-pi branch; measure circular error rather than treating
            # an equivalent +/-2pi wrap as a six-radian neural error.
            phase_error = np.angle(np.exp(1j * (out_pha - ref_pha.cpu().numpy())))
            audio = metrics(reference, actual)
            # Same strict original model, no inference lowering, FP64 CUDA.
            # Identical STFT input bytes isolate neural arithmetic fidelity for
            # every accepted crop/quiet/boundary case, not a narrower subset.
            exact_amp, exact_pha, _, _, _ = precise_model(mag.double(), pha.double())
            exact_wave = mag_pha_istft(exact_amp, exact_pha, 400, 100, 400,
                                      compress_factor=0.3, center=True)[0].cpu().numpy()[:len(samples)] / scale
            precise_audio = metrics(exact_wave, actual)
            precise_amplitude = metrics(exact_amp.cpu().numpy(), out_amp)
            sf.write(args.validation_dir / f"{name}-fp64-reference.wav", exact_wave, 16000, subtype="FLOAT")
            if not np.isfinite(out_amp).all() or not np.isfinite(out_pha).all() or not np.isfinite(actual).all():
                raise ValueError(f"Nonfinite ONNX output: {name}")
            fp32_passed = amplitude["relativeRmse"] <= 5e-4 and audio["relativeRmse"] <= 1e-3 and audio["maxAbsError"] <= 1e-3
            fp64_passed = precise_amplitude["relativeRmse"] <= 5e-4 and precise_audio["relativeRmse"] <= 1e-3 and precise_audio["maxAbsError"] <= 1e-3
            if name != "silence-4s":
                for provider, passed, amp_metrics, audio_metrics in [
                    ("original-cuda-fp32", fp32_passed, amplitude, audio),
                    ("original-cuda-fp64", fp64_passed, precise_amplitude, precise_audio),
                ]:
                    if not passed:
                        native_gate_failures.append({"name": name, "provider": provider,
                                                     "amplitude": amp_metrics, "audio": audio_metrics,
                                                     "unchangedFromBaselineOnnx": True})
            sf.write(args.validation_dir / f"{name}-reference.wav", reference, 16000, subtype="FLOAT")
            sf.write(args.validation_dir / f"{name}-onnx.wav", actual, 16000, subtype="FLOAT")
            result = {"name": name, "inputSamples": len(samples), "timeFrames": int(mag.shape[-1]),
                      "paddedInputSamples": len(padded), "outputSamples": len(actual),
                      "baselineOnnxAmplitude": baseline_amplitude, "baselineOnnxAudio": baseline_waveform,
                      "baselineOnnxPhaseCircularRmse": baseline_phase_rmse,
                      "originalFp64Audio": precise_audio, "originalFp64Amplitude": precise_amplitude,
                      "nativeFp32GatesPassed": fp32_passed, "nativeFp64GatesPassed": fp64_passed,
                      "amplitude": amplitude, "phaseCircularRmse": float(np.sqrt(np.mean(phase_error ** 2))),
                      "audio": audio, "ctcInputFidelity": {"sampleRate": 16000, "sameFrameCount": len(reference) == len(actual),
                      "snrDb": audio["snrDb"], "relativeRmse": audio["relativeRmse"]}}
            if name == "silence-4s":
                result["accepted"] = False
                result["reason"] = "Standalone all-silent track normalization is undefined; original model produces genuine nonzero signal and cross-provider neural arithmetic is ill-conditioned. Diagnostic only, not an accepted runtime input."
                diagnostics.append(result)
            else:
                result["accepted"] = fp32_passed and fp64_passed
                results.append(result)
            print(json.dumps(result), flush=True)
    metadata = {"id": MODEL_ID, "sha256": sha256(args.graph), "checkpointSha256": CHECKPOINT_SHA256,
                "configurationSha256": sha256(config_path), "parameterCount": parameters,
                "sampleRate": 16000, "fftSize": 400, "hopSize": 100, "winSize": 400,
                "compressFactor": 0.3, "chunkSeconds": 4, "strideSeconds": 3,
                "inputNames": ["noisy_mag", "noisy_pha"], "outputNames": ["amp_g", "pha_g"],
                "inputShape": [1, 201, MAX_TIME_FRAMES], "inputFrames": MAX_TIME_FRAMES,
                "timeFrames": {"min": MAX_TIME_FRAMES, "max": MAX_TIME_FRAMES},
                "precision": "float32", "opset": 17, "license": "Apache-2.0",
                "source": "https://modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base",
                "sourceRevision": "bed374bb53e4a5ffc62093676bab408a56b0db48",
                "operators": ops, "neuralNodes": learned_nodes,
                "placementGraph": {"path": "models/zipenhancer.onnx", "sha256": sha256(args.graph), "nodes": placement_nodes},
                "optimizationParityPassed": True, "nativeParityPassed": not native_gate_failures,
                "baselineOnnxSha256": BASELINE_SHA256, "baselineNativeMismatch": native_gate_failures,
                "validationPassed": not native_gate_failures,
                "requiredGpuBufferBytes": required_gpu_buffer_bytes,
                "attentionQueryRows": 64, "runtimePaddedTimeFrames": 641,
                "concatMaxInputs": MAX_CONCAT_INPUTS,
                "concatLowering": "ordered value-preserving tree; at most seven reads and one write per WebGPU dispatch",
                "instanceNormLowering": "anchored two-pass biased variance; original affine parameters and epsilon",
                "staticGraphLowering": "fixed-641 shapes; offline exact constant/index folding; compact shared row-major positional Gather; proven no-op aliases and dead branches removed",
                "normalization": "full-original-track RMS scale, undone after inference; undefined for all-zero track"}
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    validation = {"modelId": MODEL_ID, "sha256": metadata["sha256"], "checkpointSha256": CHECKPOINT_SHA256,
                  "baselineOnnxSha256": BASELINE_SHA256, "inputShape": [1, 201, MAX_TIME_FRAMES],
                  "referenceProvider": str(next(model.parameters()).device), "onnxValidationProvider": session.get_providers(),
                  "torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__,
                  "numpy": np.__version__, "weightsStrictlyLoaded": True, "parameters": parameters,
                  "cases": results, "ctcInputFidelityMeaning": "Waveform equality metrics at the 16k mono CTC input; no ASR hypothesis/accuracy claim",
                  "diagnostics": diagnostics,
                  "numericGates": {"amplitudeRelativeRmse": 5e-4, "waveformRelativeRmse": 1e-3, "waveformMaxAbsError": 1e-3},
                  "baselineOnnxNumericGates": {"bitExactAmplitudePhaseAndAudio": True},
                  "optimizationParityPassed": True, "nativeParityPassed": not native_gate_failures,
                  "baselineNativeMismatch": native_gate_failures,
                  "optimizationAcceptance": "All original cases are bit-exact to the frozen 641-frame ONNX baseline; unchanged native reference failures remain failures with the original gates",
                  "acceptanceScope": "All original real speech including quiet final acknowledgment, short/even crop inputs and noise/speech-silence boundaries; all use runtime zero-padding and full 641-frame attention; exact all-zero standalone track remains a rejected-input diagnostic",
                  "passed": not native_gate_failures}
    (args.validation_dir / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"graph": str(args.graph), "sha256": metadata["sha256"], "bytes": args.graph.stat().st_size,
                      "operators": ops, "graphNodes": len(graph.graph.node),
                      "maxConcatInputs": max(len(n.input) for n in graph.graph.node if n.op_type == "Concat"),
                      "neuralNodes": len(learned_nodes), "optimizationParityPassed": True,
                      "nativeParityPassed": not native_gate_failures,
                      "validationPassed": not native_gate_failures}), flush=True)
    if native_gate_failures:
        raise ValueError(f"Original native reference numeric gates failed without optimization drift: {native_gate_failures}")


if __name__ == "__main__":
    main()
