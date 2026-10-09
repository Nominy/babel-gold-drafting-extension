#!/usr/bin/env python3
"""Private original-weight CUDA experiment; never changes the accepted browser model.

Example (from the repository root, using the isolated experiment Python):
  python drafting/gold-drafting-extension/scripts/benchmark-zipenhancer-native.py \
    --candidate autocastfp16 --batch-size 4 --out D:/babel_experiment_runs/zipenhancer-native-speed-20261009

Every measured pass runs both complete recordings. END-TO-END starts with decoded
source PCM in RAM and includes resampling, normalization, window assembly, transfers,
original Torch STFT/ISTFT, ordered context cropping, clock restoration and in-memory
FLOAT/PCM16 WAV encoding. File reads/writes, numerical analysis and events are outside
that clock. NETWORK uses resident, real input spectra, never cached network outputs.
The reference is strict eager FP32, batch one, not the different-DSP browser output.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback

sys.dont_write_bytecode = True
CHECKPOINT_SHA256 = "b18896915e27a821585584221d0c0820f35e12145315ae3f1e73ccd5a68d195f"
RATE, WINDOW, STRIDE, EDGE = 16000, 64000, 48000, 8000
CANDIDATES = ("fp32eager", "tf32eager", "autocastfp16", "bf16")
DSP = {
    "version": 1, "rate": RATE, "windowSamples": WINDOW, "strideSamples": STRIDE,
    "guardSamples": RATE, "edgeSamples": EDGE,
    "resample": "scipy.signal.resample_poly float32, gcd factors, Kaiser 5.0, constant zero pad",
    "normalization": "per complete mono lane RMS using float64 energy; silent scale 1; undo after crop",
    "padding": "zero pad waveform through one-second guard, then complete four-second windows",
    "stft": {"nFft": 400, "hop": 100, "win": 400, "center": True,
             "padMode": "reflect", "compression": 0.3, "implementation": "original ModelScope torch STFT/ISTFT"},
    "overlap": "context-only ordered crop: first left=0, later left=8000, all right=56000",
    "output": "resample to original rate, truncate/zero-pad to exact source frames; FLOAT and PCM_16 WAV",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def event(out, kind, candidate, message=None, metrics=None):
    value = {"time": datetime.now(timezone.utc).isoformat(), "kind": kind, "candidate": candidate}
    if message is not None:
        value["message"] = message
    if metrics is not None:
        value["metrics"] = metrics
    with (out / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
    print(json.dumps(value, allow_nan=False), flush=True)


def arguments():
    repo = Path(__file__).resolve().parents[3]
    trial = repo.parent / "babel_experiment/artifacts/maxine-afx-live-20261008"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", choices=(*CANDIDATES, "all"), default="fp32eager")
    parser.add_argument("--out", type=Path, default=Path("D:/babel_experiment_runs/zipenhancer-native-speed-20261009"))
    parser.add_argument("--reference-root", type=Path, default=trial / "zipenhancer")
    parser.add_argument("--packages", type=Path, default=trial / "clearvoice-packages")
    parser.add_argument("--backend", choices=("torch", "ort", "tensorrt"), default="torch")
    parser.add_argument("--backend-model", type=Path, default=Path(__file__).resolve().parents[1] / "models/zipenhancer.onnx")
    parser.add_argument("--backend-packages", type=Path, help="Isolated optional backend packages; never installed into the existing environment")
    parser.add_argument("--audio", type=Path, action="append", help="Repeat exactly twice; defaults to private speaker-1/2.wav")
    parser.add_argument("--browser-wav", type=Path, action="append", help="Repeat twice to override available accepted browser references")
    parser.add_argument("--baseline-dir", type=Path, help="Reuse or create a verified strict baseline directory containing baseline.json")
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--max-batch-size", type=int, choices=(1, 2, 4, 8), default=4,
                        help="Safety bound; batch 8 requires explicit --max-batch-size 8")
    parser.add_argument("--repeats", type=int, default=3, help="At least three full warm passes per timing scope")
    parser.add_argument("--low-rank", type=float, help="Experimental feedforward SVD rank fraction in (0,1); baseline stays original")
    parser.add_argument("--fused-softmax", action="store_true", help="Replace multi-pass attention softmax with the native fused CUDA operation")
    parser.add_argument("--cudnn-tune", action="store_true", help="Benchmark convolution algorithms; allow nondeterministic choices in the candidate only")
    parser.add_argument("--channels-last", action="store_true", help="Store convolution weights in channels-last format")
    parser.add_argument("--triton-norm", action="store_true", help="Experimental anchored FP32-statistics CUDA instance normalization")
    parser.add_argument("--fp16-residuals", action="store_true", help="Experimental FP16 parameter/residual storage with FP32 nonlinear statistics")
    parser.add_argument("--tiled-attention-library", type=Path, help="Experimental controlled C++/CUDA all-key relative-attention DLL")
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="default")
    parser.add_argument("--cuda-graph", action="store_true", help="Fixed-size manual graph; can capture default compiled execution")
    parser.add_argument("--compile", action="store_true", help="torch.compile inductor, no eager fallback on compiler errors")
    args = parser.parse_args()
    if args.batch_size > args.max_batch_size or args.repeats < 3:
        parser.error("batch-size exceeds safety bound or repeats is below three")
    if args.low_rank is not None and not (math.isfinite(args.low_rank) and 0 < args.low_rank < 1):
        parser.error("--low-rank must be finite and strictly between zero and one")
    if args.compile and args.cuda_graph and args.compile_mode != "default":
        parser.error("Manual CUDA graphs require default compile mode, not Inductor graph trees")
    if args.fp16_residuals and args.candidate != "autocastfp16":
        parser.error("FP16 residual storage must be measured as an explicit autocastfp16 candidate")
    if args.tiled_attention_library and (args.backend != "torch" or args.candidate != "autocastfp16" or args.fused_softmax):
        parser.error("Tiled CUDA attention is an explicit Torch autocastfp16 candidate, not a separate softmax transform")
    if args.backend != "torch" and (args.batch_size != 1 or args.candidate not in ("fp32eager", "autocastfp16") or
                                   args.compile or args.cuda_graph or args.low_rank is not None or args.fused_softmax or
                                   args.cudnn_tune or args.channels_last or args.triton_norm or args.fp16_residuals):
        parser.error("ONNX/TensorRT adapters require fixed batch one, fp32eager/autocastfp16, and no PyTorch-only transforms")
    args.audio = args.audio or [repo / f"audio/.private/noisy-live-20261008/speaker-{n}.wav" for n in (1, 2)]
    args.browser_wav = args.browser_wav or [trial / f"feature-smoke-v4/speaker-{n}-browser-enhanced.wav" for n in (1, 2)]
    if len(args.audio) != 2 or len(args.browser_wav) != 2:
        parser.error("Supply exactly two audio paths and two browser WAV paths")
    args.out = args.out.resolve()
    if args.out == repo or repo in args.out.parents:
        parser.error("--out must be private and outside the repository")
    # The checkpoint/reference and browser artifact trees are immutable inputs.
    protected = [args.reference_root.resolve(), trial.resolve(), args.packages.resolve()]
    if any(args.out == root or root in args.out.parents for root in protected):
        parser.error("--out must not be inside an existing reference/artifact tree")
    return args


def configure_precision(torch, candidate):
    tf32 = candidate == "tf32eager"
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


class Network:
    def __init__(self, torch, model, candidate, batch, graph=False, compile_model=False, compile_mode="default"):
        self.torch, self.batch = torch, batch
        self.dtype = {"autocastfp16": torch.float16, "bf16": torch.bfloat16}.get(candidate)
        if self.dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA device does not support BF16; no precision fallback")
        self.model = torch.compile(model, backend="inductor", dynamic=False, mode=compile_mode) if compile_model else model
        self.graph = None
        self.static = None
        if graph:
            self.static = (torch.zeros(batch, 201, 641, device="cuda"), torch.zeros(batch, 201, 641, device="cuda"))
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.raw(*self.static)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.captured = self.raw(*self.static)
            torch.cuda.synchronize()

    def raw(self, magnitude, phase):
        context = self.torch.autocast("cuda", dtype=self.dtype, cache_enabled=False) if self.dtype else nullcontext()
        with context:
            result = self.model(magnitude, phase)
        return result[0], result[1]

    def __call__(self, magnitude, phase):
        if self.graph is None:
            return self.raw(magnitude, phase)
        self.static[0].copy_(magnitude)
        self.static[1].copy_(phase)
        self.graph.replay()
        # Never return storage that a subsequent graph replay can overwrite.
        return self.captured[0].clone(), self.captured[1].clone()


class Pipeline:
    def __init__(self, np, sf, torch, resample_poly, stft, istft, lanes, batch):
        self.np, self.sf, self.torch = np, sf, torch
        self.resample_poly, self.stft, self.istft = resample_poly, stft, istft
        self.lanes, self.batch = lanes, batch
        self.counts = [self.window_count(math.ceil(len(lane["audio"]) * RATE / lane["rate"])) for lane in lanes]
        self.schedule = [(lane, step * STRIDE) for step in range(max(self.counts))
                         for lane, count in enumerate(self.counts) if step < count]
        self.batches = [self.schedule[start:start + batch] for start in range(0, len(self.schedule), batch)]

    @staticmethod
    def window_count(frames):
        return 1 + max(0, math.ceil((frames + RATE - WINDOW) / STRIDE))

    def resample(self, samples, source, target):
        if source == target:
            return samples
        divisor = math.gcd(source, target)
        return self.resample_poly(samples, target // divisor, source // divisor,
                                  window=("kaiser", 5.0), padtype="constant").astype(self.np.float32, copy=False)

    def prepare(self):
        padded, scales, lengths = [], [], []
        for lane, count in zip(self.lanes, self.counts):
            samples = self.resample(lane["audio"], lane["rate"], RATE)
            double_samples = samples.astype(self.np.float64)
            energy = float(self.np.dot(double_samples, double_samples))
            if not math.isfinite(energy) or energy <= 0:
                raise ValueError("Original lane RMS normalization requires finite, nonzero source energy")
            scale = math.sqrt(len(samples) / energy)
            buffer = self.np.zeros(WINDOW + (count - 1) * STRIDE, dtype=self.np.float32)
            self.np.multiply(samples, scale, out=buffer[:len(samples)])
            padded.append(buffer)
            scales.append(scale)
            lengths.append(len(samples))
        return padded, scales, lengths

    def spectra(self, padded, items):
        # The last batch is zero-filled to the same shape for graph/compile.
        wave = self.np.zeros((self.batch, WINDOW), dtype=self.np.float32)
        for row, (lane, start) in enumerate(items):
            wave[row] = padded[lane][start:start + WINDOW]
        wave = self.torch.from_numpy(wave).to("cuda")
        magnitude, phase, _ = self.stft(wave, 400, 100, 400, compress_factor=0.3, center=True)
        return magnitude, phase

    def resident_spectra(self):
        padded, _, _ = self.prepare()
        return [self.spectra(padded, items) for items in self.batches]

    def run(self, network):
        torch, np = self.torch, self.np
        timers = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in self.batches]
        torch.cuda.synchronize()
        started = time.perf_counter()
        padded, scales, lengths = self.prepare()
        outputs = [np.empty(length, dtype=np.float32) for length in lengths]
        for items, (begin, end) in zip(self.batches, timers):
            magnitude, phase = self.spectra(padded, items)
            begin.record()
            amplitude, prediction = network(magnitude, phase)
            end.record()
            # DSP stays FP32 even when the network uses autocast.
            enhanced = self.istft(amplitude.float(), prediction.float(), 400, 100, 400,
                                  compress_factor=0.3, center=True).cpu().numpy()
            if enhanced.shape != (self.batch, WINDOW):
                raise RuntimeError(f"Unexpected ISTFT shape: {enhanced.shape}")
            for row, (lane, start) in enumerate(items):
                left = 0 if start == 0 else EDGE
                right = min(WINDOW - EDGE, lengths[lane] - start)
                if right > left:
                    outputs[lane][start + left:start + right] = enhanced[row, left:right] / scales[lane]
        restored, encoded = [], []
        for samples, lane in zip(outputs, self.lanes):
            samples = self.resample(samples, RATE, lane["rate"])
            frames = len(lane["audio"])
            samples = samples[:frames] if len(samples) >= frames else np.pad(samples, (0, frames - len(samples)))
            restored.append(samples)
            waves = {}
            for subtype in ("FLOAT", "PCM_16"):
                stream = io.BytesIO()
                self.sf.write(stream, samples, lane["rate"], format="WAV", subtype=subtype)
                waves[subtype] = stream.getvalue()
            encoded.append(waves)
        torch.cuda.synchronize()
        total = time.perf_counter() - started
        gpu = sum(begin.elapsed_time(end) for begin, end in timers) / 1000
        return {"totalSeconds": total, "networkSeconds": gpu, "dspSeconds": total - gpu}, restored, encoded

    def network_run(self, network, spectra):
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        for magnitude, phase in spectra:
            amplitude, prediction = network(magnitude, phase)
            # The calls execute every window; no output reuse or network cache.
            del amplitude, prediction
        self.torch.cuda.synchronize()
        return time.perf_counter() - started


def error_metrics(np, reference, actual):
    reference, actual = reference.astype(np.float64), actual.astype(np.float64)
    difference = actual - reference
    error = float(np.sqrt(np.mean(difference * difference)))
    rms = float(np.sqrt(np.mean(reference * reference)))
    actual_rms = float(np.sqrt(np.mean(actual * actual)))
    return {"relativeRmse": error / max(rms, 1e-12), "rmse": error,
            "maxAbsError": float(np.max(np.abs(difference))), "referenceRms": rms, "actualRms": actual_rms,
            "snrDb": 20 * math.log10(max(rms, 1e-12) / max(error, 1e-20)),
            "energyRatio": actual_rms ** 2 / max(rms ** 2, 1e-24)}


def comparison(np, reference, actual, rate):
    result = error_metrics(np, reference, actual)
    windows = [(start, min(start + rate, len(reference))) for start in range(0, len(reference), rate)]
    energies = [float(np.mean(reference[left:right].astype(np.float64) ** 2)) for left, right in windows]
    quiet_limit, speech_limit = np.quantile(energies, [0.2, 0.6])
    result["windowDefinition"] = "Nonoverlapping 1s source-clock windows; quiet <= FP32 energy p20, speech >= p60 (energy proxy, not VAD)"
    for name, selected in (("quietWindows", [i for i, e in enumerate(energies) if e <= quiet_limit]),
                           ("speechWindows", [i for i, e in enumerate(energies) if e >= speech_limit])):
        refs = np.concatenate([reference[slice(*windows[i])] for i in selected])
        values = np.concatenate([actual[slice(*windows[i])] for i in selected])
        result[name] = {"count": len(selected), **error_metrics(np, refs, values),
                        "windows": [{"startSeconds": windows[i][0] / rate,
                                     "endSeconds": windows[i][1] / rate,
                                     **error_metrics(np, reference[slice(*windows[i])], actual[slice(*windows[i])])}
                                    for i in selected]}
    return result


def save_waves(directory, encoded, lanes):
    artifacts = []
    for index, (waves, lane) in enumerate(zip(encoded, lanes), 1):
        files = {}
        for subtype, suffix in (("FLOAT", "float"), ("PCM_16", "pcm16")):
            file = directory / f"speaker-{index}-{suffix}.wav"
            file.write_bytes(waves[subtype])
            files[suffix] = {"file": file.name, "sha256": digest(file)}
        artifacts.append({"speaker": index, "frames": len(lane["audio"]), "sampleRate": lane["rate"], **files})
    return artifacts


def load_baseline(directory, identity, np, sf, lanes):
    report = json.loads((directory / "baseline.json").read_text(encoding="utf-8"))
    if report.get("identity") != identity or report.get("precision") != "fp32eager" or report.get("batchSize") != 1:
        raise ValueError("Baseline checkpoint/audio/config/DSP/implementation identity does not match")
    if report.get("status") != "complete" or report.get("cudaGraph") or report.get("compiled"):
        raise ValueError("Baseline is not a complete strict eager reference")
    artifacts = report.get("artifacts", [])
    if len(artifacts) != len(lanes):
        raise ValueError("Baseline lane count mismatch")
    values = []
    for artifact, lane in zip(artifacts, lanes):
        for subtype in ("float", "pcm16"):
            path = (directory / artifact[subtype]["file"]).resolve()
            if path.parent != directory.resolve() or digest(path) != artifact[subtype]["sha256"]:
                raise ValueError("Baseline WAV path/hash mismatch")
        samples, rate = sf.read(directory / artifact["float"]["file"], dtype="float32")
        if rate != lane["rate"] or samples.shape != lane["audio"].shape or not np.isfinite(samples).all():
            raise ValueError("Baseline source clock or sample values mismatch")
        values.append(samples)
    return values, report


def benchmark(args, candidate, batch, directory, model, modules, lanes, identity, baseline, browser, baseline_run=False):
    np, sf, torch, resample_poly, stft, istft = modules
    candidate_id = "baseline" if baseline_run else directory.name
    graph, compiled = (False, False) if baseline_run else (args.cuda_graph, args.compile)
    backend = "torch" if baseline_run else args.backend
    directory.mkdir(parents=True, exist_ok=False)
    report = {"schema": "babel-zipenhancer-native-v1", "candidate": candidate_id, "precision": candidate,
              "batchSize": batch, "cudaGraph": graph, "compiled": compiled, "identity": identity,
              "parameters": sum(value.numel() for value in model.parameters()),
              "modelChanges": None if baseline_run else getattr(args, "low_rank_report", None),
              "backend": backend,
              "status": "running", "timingContract": {
                  "scope": "both complete lanes, source PCM already decoded in host RAM; no disk I/O",
                  "networkWarm": "synchronized wall time across all resident spectra; graph includes copies/clones",
                  "networkWithinEndToEnd": "sum of CUDA-event intervals around model dispatch; includes graph copies/clones",
                  "dspSeconds": "END-TO-END minus model CUDA-event intervals; includes host orchestration and WAV encoding",
                  "firstCall": "first complete END-TO-END after candidate setup, before warm repetitions",
                  "graphSetup": "three side-stream warmups and one capture, if enabled, counted only in setup"}}
    network = None
    spectra = None
    original_softmax = None
    try:
        configure_precision(torch, candidate)
        if not baseline_run and args.cudnn_tune:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        if not baseline_run and args.fused_softmax:
            from zipenhancer_native_optimizations import install_fused_attention_softmax
            original_softmax = install_fused_attention_softmax()
        report["kernelOptions"] = {} if baseline_run else {
            "fusedSoftmax": args.fused_softmax, "cudnnTune": args.cudnn_tune,
            "channelsLast": args.channels_last, "compileMode": args.compile_mode,
            "implementationSha256": digest(Path(__file__).with_name("zipenhancer_native_optimizations.py"))}
        if not baseline_run:
            report["kernelOptions"]["tritonNorm"] = getattr(args, "triton_norm_report", None)
            report["kernelOptions"]["fp16Residuals"] = getattr(args, "fp16_residual_report", None)
            report["kernelOptions"]["tiledAttention"] = getattr(args, "tiled_attention_report", None)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        pipeline = Pipeline(*modules, lanes, batch)
        with torch.inference_mode():
            event(args.out, "status", candidate_id, "Preparing network execution", {"precision": candidate, "batchSize": batch, "compiled": compiled, "cudaGraph": graph})
            if backend == "torch":
                network = Network(torch, model, candidate, batch, graph, compiled, args.compile_mode)
            else:
                if args.backend_packages:
                    sys.path.insert(0, str(args.backend_packages.resolve()))
                if backend == "ort":
                    from zipenhancer_ort_cuda import BackendNetwork
                else:
                    from zipenhancer_tensorrt import BackendNetwork
                network = BackendNetwork(torch, args.backend_model, "fp16" if candidate == "autocastfp16" else "fp32",
                                         args.out / "backend-cache" / backend,
                                         lambda message, metrics: event(args.out, "status", candidate_id, message, metrics))
            torch.cuda.synchronize()
            report["candidateSetupSeconds"] = time.perf_counter() - started
            report["windowCounts"] = pipeline.counts
            report["realWindows"] = len(pipeline.schedule)
            report["executedWindows"] = len(pipeline.batches) * batch
            event(args.out, "status", candidate_id, "Running first complete END-TO-END pass", {
                "windowCounts": pipeline.counts, "realWindows": len(pipeline.schedule),
                "executedWindows": len(pipeline.batches) * batch, "batchSize": batch})
            report["firstCall"], _, _ = pipeline.run(network)
            event(args.out, "status", candidate_id, "First call completed", report["firstCall"])
            spectra = pipeline.resident_spectra()
            torch.cuda.synchronize()
            network_times = []
            for repeat in range(args.repeats):
                seconds = pipeline.network_run(network, spectra)
                network_times.append(seconds)
                event(args.out, "status", candidate_id, "Warm NETWORK completed", {
                    "scope": "network", "repeat": repeat + 1, "networkSeconds": seconds})
            del spectra
            spectra = None
            warm, consistency, last_values, last_encoded = [], [], None, None
            for repeat in range(args.repeats):
                timing, values, encoded = pipeline.run(network)
                if any(not np.isfinite(samples).all() for samples in values):
                    raise RuntimeError("Nonfinite waveform produced; candidate rejected")
                if last_values is not None:
                    consistency.append([error_metrics(np, previous, current) for previous, current in zip(last_values, values)])
                last_values, last_encoded = values, encoded
                warm.append(timing)
                event(args.out, "status", candidate_id, "Warm END-TO-END completed", {
                    "scope": "end-to-end", "repeat": repeat + 1, **timing})
        report["warmNetworkSeconds"] = network_times
        report["warmEndToEnd"] = warm
        report["warmRepeatNumericalChanges"] = consistency
        report["metrics"] = {key: statistics.median([sample[key] for sample in warm])
                             for key in ("totalSeconds", "networkSeconds", "dspSeconds")}
        report["metrics"]["standaloneNetworkSeconds"] = statistics.median(network_times)
        report["metrics"]["underOneSecondBothLanes"] = report["metrics"]["totalSeconds"] <= 1.0
        report["metrics"]["allWarmPassesUnderOneSecond"] = all(sample["totalSeconds"] <= 1.0 for sample in warm)
        report["peakAllocatedBytes"] = torch.cuda.max_memory_allocated()
        report["peakReservedBytes"] = torch.cuda.max_memory_reserved()
        report["artifacts"] = save_waves(directory, last_encoded, lanes)
        references = last_values if baseline_run else baseline
        report["numericalChanges"] = [{"speaker": index, "domain": "unquantized source-clock FLOAT waveform",
                                       **comparison(np, reference, actual, lane["rate"])}
                                      for index, (reference, actual, lane) in enumerate(zip(references, last_values, lanes), 1)]
        report["browserComparison"] = []
        for index, (browser_lane, actual, lane) in enumerate(zip(browser, last_values, lanes), 1):
            entry = {"speaker": index, "label": "original-native-vs-browser: different DSP, not a native precision gate"}
            if browser_lane.get("audio") is None:
                entry.update({key: value for key, value in browser_lane.items() if key != "audio"})
            else:
                entry.update({"status": "compared", "sha256": browser_lane["sha256"],
                              **comparison(np, browser_lane["audio"], actual, lane["rate"])})
            report["browserComparison"].append(entry)
        if backend != "torch":
            report["backendReport"] = network.finish()
            network = None
        report["status"] = "complete"
        write_json(directory / ("baseline.json" if baseline_run else "candidate.json"), report)
        event(args.out, "result", candidate_id, "Complete; timings cover both full recordings", {
            **report["metrics"], "precision": candidate, "batchSize": batch, "numericalChanges": [
                {"speaker": item["speaker"], **{key: item[key] for key in ("relativeRmse", "maxAbsError", "snrDb")}}
                for item in report["numericalChanges"]]})
        return last_values, report
    except Exception as error:
        report.update(status="failed", errorType=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        report["oom"] = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
        if backend != "torch" and network is not None:
            try:
                report["backendReport"] = network.finish()
            except Exception as cleanup_error:
                report["backendCleanupError"] = str(cleanup_error)
            network = None
        write_json(directory / ("baseline.json" if baseline_run else "candidate.json"), report)
        event(args.out, "error", candidate_id, f"{type(error).__name__}: {error}; no CPU or precision fallback", {"oom": report["oom"]})
        return None, report
    finally:
        if original_softmax is not None:
            from zipenhancer_native_optimizations import restore_attention_softmax
            restore_attention_softmax(original_softmax)
        network = None
        spectra = None
        gc.collect()
        torch.cuda.empty_cache()


def main():
    args = arguments()
    args.out.mkdir(parents=True, exist_ok=True)
    candidate_id = args.candidate
    cold_started = time.perf_counter()
    event(args.out, "status", candidate_id, "Cold setup: importing dependencies and verifying private inputs")
    try:
        import numpy as np
        import scipy
        from scipy.signal import resample_poly
        import soundfile as sf
        import torch
        # Keep environment NumPy/SciPy/Torch; use only the isolated ModelScope package tree.
        sys.path.insert(0, str(args.packages.resolve()))
        from modelscope.models.audio.ans.zipenhancer import ZipEnhancer, AttrDict, mag_pha_stft, mag_pha_istft
        if not torch.cuda.is_available():
            raise RuntimeError("Native experiment requires CUDA; no CPU fallback")
        torch.set_num_threads(4)
        configure_precision(torch, "fp32eager")
        checkpoint_path = args.reference_root / "pytorch_model.bin"
        if digest(checkpoint_path) != CHECKPOINT_SHA256:
            raise ValueError("Original checkpoint digest mismatch")
        config_path = args.reference_root / "configuration.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))["model"]
        model = ZipEnhancer(AttrDict(config))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["generator"], strict=True)
        del checkpoint
        parameters = sum(value.numel() for value in model.parameters())
        if parameters != 2044436:
            raise ValueError(f"Unexpected learned parameter count: {parameters}")
        model = model.to("cuda").eval()
        lanes, browser = [], []
        for path, browser_path in zip(args.audio, args.browser_wav):
            samples, rate = sf.read(path, dtype="float32")
            if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
                raise ValueError(f"Expected nonempty finite mono input: {path}")
            lanes.append({"audio": samples, "rate": rate, "path": str(path.resolve()), "sha256": digest(path)})
            if not browser_path.is_file():
                browser.append({"audio": None, "status": "unavailable", "path": str(browser_path)})
            else:
                reference, reference_rate = sf.read(browser_path, dtype="float32")
                if reference_rate != rate or reference.shape != samples.shape or not np.isfinite(reference).all():
                    browser.append({"audio": None, "status": "incompatible source clock/shape/values", "path": str(browser_path)})
                else:
                    browser.append({"audio": reference, "sha256": digest(browser_path)})
        source_root = args.packages / "modelscope/models/audio/ans"
        source_files = [source_root / "zipenhancer.py", *sorted((source_root / "zipenhancer_layers").rglob("*.py"))]
        identity = {"checkpointSha256": CHECKPOINT_SHA256, "configurationSha256": digest(config_path),
                    "audioSha256": [lane["sha256"] for lane in lanes], "dsp": DSP,
                    "runnerSha256": digest(Path(__file__)),
                    "nativeSourceSha256": {str(path.relative_to(source_root)): digest(path) for path in source_files},
                    "versions": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                                 "numpy": np.__version__, "scipy": scipy.__version__, "soundfile": sf.__version__},
                    "device": torch.cuda.get_device_name()}
        torch.cuda.synchronize()
        cold_seconds = time.perf_counter() - cold_started
        event(args.out, "status", candidate_id, "Cold setup completed", {
            "coldSetupSeconds": cold_seconds, "parameters": parameters,
            "laneDurationsSeconds": [len(lane["audio"]) / lane["rate"] for lane in lanes],
            "cudaFreeBytes": torch.cuda.mem_get_info()[0]})
        modules = (np, sf, torch, resample_poly, mag_pha_stft, mag_pha_istft)
        baseline_dir = args.baseline_dir.resolve() if args.baseline_dir else args.out / "baseline"
        if (baseline_dir / "baseline.json").is_file():
            baseline, baseline_report = load_baseline(baseline_dir, identity, np, sf, lanes)
            baseline_reused = True
            event(args.out, "status", "baseline", "Loaded verified strict FP32 audio; old timings are NOT current-run measurements")
        else:
            baseline, baseline_report = benchmark(args, "fp32eager", 1, baseline_dir, model, modules,
                                                  lanes, identity, None, browser, baseline_run=True)
            baseline_reused = False
            if baseline is None:
                raise RuntimeError("Strict FP32 baseline failed; candidates cannot be compared")
        # Only now may the experimental model change; the strict cache was already
        # generated/verified against the untouched checkpoint and DSP identity.
        candidate_identity = identity
        args.fp16_residual_report = None
        if args.fp16_residuals:
            from zipenhancer_fp16_residuals import use_half_residuals
            args.fp16_residual_report = {**use_half_residuals(model),
                                        "implementationSha256": digest(Path(__file__).with_name("zipenhancer_fp16_residuals.py"))}
        args.triton_norm_report = None
        if args.triton_norm:
            from zipenhancer_triton_norm import replace_instance_norm
            names = replace_instance_norm(model)
            args.triton_norm_report = {"layers": names, "implementationSha256": digest(Path(__file__).with_name("zipenhancer_triton_norm.py"))}
        args.tiled_attention_report = None
        if args.tiled_attention_library:
            from zipenhancer_tiled_model import install_tiled_attention
            args.tiled_attention_report = install_tiled_attention(model, args.tiled_attention_library)
        if args.channels_last:
            model = model.to(memory_format=torch.channels_last)
        args.low_rank_report = None
        if args.low_rank is not None:
            from zipenhancer_low_rank import factorize_projections
            event(args.out, "status", candidate_id, "Preparing experimental low-rank feedforward model; strict baseline is unchanged")
            started = time.perf_counter()
            model = model.cpu().eval()
            replacements = factorize_projections(model, args.low_rank)
            model = model.to("cuda").eval()
            torch.cuda.synchronize()
            args.low_rank_report = {"fraction": args.low_rank, "feedforwardOnly": True,
                                    "setupSeconds": time.perf_counter() - started,
                                    "replacements": replacements,
                                    "implementationSha256": digest(Path(__file__).with_name("zipenhancer_low_rank.py"))}
            candidate_identity = {**identity, "modelTransform": {
                key: value for key, value in args.low_rank_report.items() if key != "setupSeconds"}}
        summary = {"schema": "babel-zipenhancer-native-v1", "coldSetupSeconds": cold_seconds,
                   "identity": candidate_identity, "parameters": sum(value.numel() for value in model.parameters()),
                   "modelChanges": args.low_rank_report, "baselineDirectory": str(baseline_dir),
                   "baselineReused": baseline_reused, "baselineTimingsMeasuredThisRun": not baseline_reused,
                   "candidates": []}
        failed = False
        for candidate in CANDIDATES if args.candidate == "all" else (args.candidate,):
            mode = "compiled-graph" if args.cuda_graph and args.compile else "graph" if args.cuda_graph else "compile" if args.compile else "eager"
            name = f"{candidate}-b{args.batch_size}-{mode}"
            if args.backend != "torch":
                name = f"{args.backend}-{name}"
            if args.fused_softmax:
                name += "-softmax"
            if args.cudnn_tune:
                name += "-tuned"
            if args.channels_last:
                name += "-nhwc"
            if args.triton_norm:
                name += "-triton-norm"
            if args.fp16_residuals:
                name += "-fp16-residuals"
            if args.tiled_attention_library:
                name += "-cuda-tiled-attention"
            if args.compile and args.compile_mode != "default":
                name += f"-{args.compile_mode}"
            if args.low_rank is not None:
                name += f"-rank{args.low_rank:g}"
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            directory = args.out / f"{name}-{stamp}"
            _, report = benchmark(args, candidate, args.batch_size, directory, model, modules,
                                  lanes, candidate_identity, baseline, browser)
            summary["candidates"].append({"candidate": directory.name, "status": report["status"],
                                          "report": str(directory / "candidate.json"), "metrics": report.get("metrics"),
                                          "error": report.get("error")})
            failed = failed or report["status"] != "complete"
        summary["status"] = "failed" if failed else "complete"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        write_json(args.out / f"summary-{stamp}.json", summary)
        write_json(args.out / "finalsummary.json", summary)
        return 1 if failed else 0
    except Exception as error:
        failure = {"status": "failed", "candidate": candidate_id, "errorType": type(error).__name__,
                   "error": str(error), "traceback": traceback.format_exc(),
                   "coldElapsedSeconds": time.perf_counter() - cold_started}
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        write_json(args.out / f"failure-{stamp}.json", failure)
        event(args.out, "error", candidate_id, f"{type(error).__name__}: {error}; no fallback")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
