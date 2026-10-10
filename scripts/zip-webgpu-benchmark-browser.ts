/// <reference types="@webgpu/types" />
import { createZipWebGpuEngine } from '../src/core/zipenhancer-webgpu';
import type { ZipWebGpuPlan } from '../src/core/zipenhancer-webgpu-plan';
import { decodeEnhancementWav, zipStft, zipIstft, ZIP_CHUNK_SAMPLES } from '../src/core/audio-enhancement-dsp';
import { enhanceZipSamplesInWorker } from '../src/core/audio-enhancement-pipeline';

interface Options {
  capture: boolean;
  runs: number;
  audioCount: number;
}
const host = globalThis as typeof globalThis & { runZipWebBenchmark?: (options: Options) => Promise<unknown> };
const hash = async (bytes: Uint8Array<ArrayBuffer>) => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), value => value.toString(16).padStart(2, '0')).join('');
async function save(name: string, value: Uint8Array<ArrayBuffer> | string) {
  const bytes = typeof value === 'string' ? new TextEncoder().encode(value) : value;
  let offset = 0;
  do {
    const end = Math.min(bytes.byteLength, offset + 4 * 1024 * 1024);
    const response = await fetch(`/result/${name}?offset=${offset}&total=${bytes.byteLength}`, { method: 'POST', body: bytes.subarray(offset, end) });
    if (!response.ok) throw new Error(`Could not save browser evidence ${name}: ${await response.text()}`);
    const acknowledgement = await response.json();
    if (acknowledgement.nextOffset !== end || acknowledgement.total !== bytes.byteLength || acknowledgement.complete !== (end === bytes.byteLength)) throw new Error('Invalid artifact acknowledgement');
    offset = end;
  } while (offset < bytes.byteLength);
}
function compare(reference: Float32Array, actual: Float32Array, sampleRate: number) {
  if (reference.length !== actual.length) throw new Error('Browser enhancement changed the source frame count');
  let energy = 0, difference = 0, actualEnergy = 0, maxAbsError = 0;
  const windows: { energy: number; error: number; actual: number; length: number }[] = [];
  for (let start = 0; start < reference.length; start += sampleRate) {
    const end = Math.min(start + sampleRate, reference.length);
    let e = 0, d = 0, a = 0;
    for (let i = start; i < end; i++) {
      if (!Number.isFinite(actual[i])) throw new Error('Browser enhancement returned nonfinite PCM');
      e += reference[i] * reference[i]; a += actual[i] * actual[i];
      const delta = actual[i] - reference[i]; d += delta * delta; maxAbsError = Math.max(maxAbsError, Math.abs(delta));
    }
    energy += e; difference += d; actualEnergy += a; windows.push({ energy: e, error: d, actual: a, length: end - start });
  }
  const ordered = windows.map(w => w.energy / w.length).sort((a, b) => a - b);
  const quietLimit = ordered[Math.floor((ordered.length - 1) * .2)];
  const quiet = windows.filter(w => w.energy / w.length <= quietLimit);
  const quietEnergy = quiet.reduce((s, w) => s + w.energy, 0);
  return { relativeRmse: Math.sqrt(difference / Math.max(energy, 1e-24)), maxAbsError,
    energyRatio: actualEnergy / Math.max(energy, 1e-24),
    quietRelativeRmse: Math.sqrt(quiet.reduce((s, w) => s + w.error, 0) / Math.max(quietEnergy, 1e-24)),
    quietEnergyRatio: quiet.reduce((s, w) => s + w.actual, 0) / Math.max(quietEnergy, 1e-24) };
}

host.runZipWebBenchmark = async options => {
  const planResponse = await fetch('/plan'), weightResponse = await fetch('/weights');
  if (!planResponse.ok || !weightResponse.ok) throw new Error('Browser benchmark model assets are unavailable');
  const plan = await planResponse.json() as ZipWebGpuPlan;
  const weights = new Uint8Array(await weightResponse.arrayBuffer());
  const precision = plan.precision;
  const environment = { crossOriginIsolated, hardwareConcurrency: navigator.hardwareConcurrency };
  const sources = await Promise.all(Array.from({ length: options.audioCount }, async (_, i) => {
    const response = await fetch(`/audio/${i}`); if (!response.ok) throw new Error('Original audio unavailable');
    return response.arrayBuffer();
  }));
  const references = await Promise.all(sources.map(async (_, i) => {
    const response = await fetch(`/reference/${i}`); if (!response.ok) throw new Error('Accepted browser WAV reference unavailable');
    return decodeEnhancementWav(await response.arrayBuffer());
  }));
  let diagnostics: unknown[] = [];
  const started = performance.now();
  const engine = await createZipWebGpuEngine(plan, weights, { capture: options.capture, onDiagnostic: value => {
      if (!value.lastAudit) { diagnostics.push(value); return; }
      const { events, ...proof } = value.lastAudit;
      const operators: Record<string, { count: number; gpuMs: number }> = {};
      for (const event of events) {
        const group = operators[event.kernelType] ??= { count: 0, gpuMs: 0 };
        group.count++; group.gpuMs += (event.endTime - event.startTime) / 1e6;
      }
      diagnostics.push({ ...value, lastAudit: { ...proof, gpuMs: Object.values(operators).reduce((total, group) => total + group.gpuMs, 0), operators } });
    } });
  const setupMs = performance.now() - started;
  const results = [];
  try {
    for (let run = 0; run < options.runs; run++) {
      diagnostics = [];
      const start = performance.now(), progress: unknown[] = [], outputs = [];
      for (let lane = 0; lane < sources.length; lane++) {
        const source = decodeEnhancementWav(sources[lane]);
        const output = await enhanceZipSamplesInWorker(source.samples, source.sampleRate,
          (magnitude, phase) => engine.infer(magnitude, phase), {
            createWorker: () => new Worker('/dsp-worker.js'),
            onChunkCompleted: (completed, total) => {
              const completion = { lane, completed, total, elapsedMs: performance.now() - start };
              progress.push(completion);
              document.body.textContent = `WebGPU ${precision}: run ${run + 1}/${options.runs}, lane ${lane + 1}/${sources.length}, window ${completed}/${total}`;
            },
          });
        outputs.push(output);
      }
      const totalMs = performance.now() - start;
      const lanes = [];
      for (let lane = 0; lane < outputs.length; lane++) {
        const output = outputs[lane];
        await save(`run-${run}-speaker-${lane + 1}.wav`, output.bytes);
        const actual = decodeEnhancementWav(output.bytes.buffer);
        if (actual.sampleRate !== references[lane].sampleRate) throw new Error('Browser output sample rate differs from reference');
        lanes.push({ lane, frameCount: output.frameCount, sampleRate: output.sampleRate, sha256: await hash(output.bytes),
          comparison: compare(references[lane].samples, actual.samples, actual.sampleRate) });
      }
      const result = { run, label: run ? `warm-${run}` : 'cold', backend: 'webgpu', setupMs, totalMs,
        precision, capture: options.capture, environment, diagnostics, progress, lanes };
      results.push(result); await save(`run-${run}.json`, JSON.stringify(result, null, 2));
    }
    // Real neural regression cases, outside the timed runs. A non-silent lane
    // may contain a whole silent window; lane RMS validation cannot exclude it.
    const numericalChecks = [];
    for (const name of ['silent-window', 'quiet-window']) {
      const samples = new Float32Array(ZIP_CHUNK_SAMPLES);
      if (name === 'quiet-window') {
        for (let i = 0; i < samples.length; i++) samples[i] = 1e-7 * Math.sin(i / 19);
      }
      const input = zipStft(samples);
      const output = await engine.infer(input.magnitude, input.phase);
      if (output.magnitude.some(value => !Number.isFinite(value) || value < 0) ||
          output.phase.some(value => !Number.isFinite(value) || Math.abs(value) > Math.PI + 0.002)) {
        throw new Error(`WebGPU numerical regression: invalid ${name} magnitude/phase`);
      }
      const waveform = zipIstft(output.magnitude, output.phase, input.frames);
      if (waveform.some(value => !Number.isFinite(value))) {
        throw new Error(`WebGPU numerical regression: nonfinite ${name} reconstruction`);
      }
      numericalChecks.push({ name, frames: input.frames, finiteSpectrum: true, finiteWaveform: true });
    }
    await save('summary.json', JSON.stringify(results.map(result => ({ ...result, numericalChecks })), null, 2));
    const summary = results.map(({ label, backend, totalMs, lanes }) => ({ label, backend, totalMs, lanes, numericalChecks }));
    document.body.textContent = JSON.stringify(summary, null, 2);
    return summary;
  } finally { await engine.dispose(); }
};
