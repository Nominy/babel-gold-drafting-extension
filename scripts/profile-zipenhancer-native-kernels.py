#!/usr/bin/env python3
"""Nsight entrypoint: profile a real compiled ZipEnhancer window, not Python setup.

Run under ncu --profile-from-start off; cudaProfilerStart/Stop surrounds only
one warmed network execution. This script does not claim end-to-end timing.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
root = Path(__file__).resolve().parents[3]
trial = root.parent / 'babel_experiment/artifacts/maxine-afx-live-20261008'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--window-index', type=int, default=0)
parser.add_argument('--fp16-residuals', action='store_true')
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
sys.path.insert(0, str(trial / 'clearvoice-packages'))
from modelscope.models.audio.ans.zipenhancer import ZipEnhancer, AttrDict, mag_pha_stft, mag_pha_istft
spec = importlib.util.spec_from_file_location('native_bench', Path(__file__).with_name('benchmark-zipenhancer-native.py'))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
checkpoint_path = trial / 'zipenhancer/pytorch_model.bin'
if bench.digest(checkpoint_path) != bench.CHECKPOINT_SHA256:
    raise ValueError('Original checkpoint SHA mismatch')
torch.set_num_threads(4)
bench.configure_precision(torch, 'autocastfp16')
model = ZipEnhancer(AttrDict(json.loads((trial / 'zipenhancer/configuration.json').read_text())['model']))
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True)['generator'], strict=True)
model.cuda().eval()
if args.fp16_residuals:
    from zipenhancer_fp16_residuals import use_half_residuals
    use_half_residuals(model)
lanes = []
for index in (1, 2):
    path = root / f'audio/.private/noisy-live-20261008/speaker-{index}.wav'
    value, rate = sf.read(path, dtype='float32')
    lanes.append({'audio': value, 'rate': rate, 'sha256': bench.digest(path)})
with torch.inference_mode():
    pipeline = bench.Pipeline(np, sf, torch, resample_poly, mag_pha_stft, mag_pha_istft, lanes, 1)
    spectra = pipeline.resident_spectra()
    if not 0 <= args.window_index < len(spectra):
        raise ValueError('Window index outside the complete real recording pair')
    network = bench.Network(torch, model, 'autocastfp16', 1, graph=True, compile_model=True)
    magnitude, phase = spectra[args.window_index]
    for _ in range(3):
        network(magnitude, phase)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push('zipenhancer-inference')
    amp, pha = network(magnitude, phase)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    if not bool(torch.isfinite(amp).all() and torch.isfinite(pha).all()):
        raise RuntimeError('Profiled network output is nonfinite')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'checkpointSha256': bench.CHECKPOINT_SHA256,
        'audioSha256': [lane['sha256'] for lane in lanes], 'windowIndex': args.window_index,
        'window': pipeline.schedule[args.window_index], 'device': torch.cuda.get_device_name(),
        'compiled': True, 'cudaGraph': True, 'fp16Residuals': args.fp16_residuals,
        'magnitudeInputSha256': hashlib.sha256(magnitude.cpu().numpy().tobytes()).hexdigest(),
        'phaseInputSha256': hashlib.sha256(phase.cpu().numpy().tobytes()).hexdigest(),
        'outputFinite': True}, indent=2))
print('Real warmed inference profiling range complete')
