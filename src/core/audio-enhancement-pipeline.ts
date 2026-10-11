import { validateZipSpectrum, ZipInvalidSpectrumError, type ZipSpectrum } from './audio-enhancement-dsp';
import { createPackagedZipDspWorker, throwIfZipCancelled, ZipDspWorkerClient } from './audio-enhancement-worker-client';
import type { ZipDspWorkerPort } from './audio-enhancement-worker-protocol';

/** Recovery-enabled inference must not mutate or transfer inputs: FP32 uploads these exact same buffers. */
export type ZipNeuralInference = (magnitude: Float32Array, phase: Float32Array, frames: number) => Promise<{ magnitude: Float32Array; phase: Float32Array }>;
export interface ZipPipelineOptions {
  signal?: AbortSignal;
  createWorker?: () => ZipDspWorkerPort;
  /** A batch-owned worker can process consecutive lanes with fresh normalization/STFT state. */
  client?: ZipDspWorkerClient;
  onChunksReady?: (totalChunks: number) => void | Promise<void>;
  onChunkCompleted?: (completedChunks: number, totalChunks: number) => void | Promise<void>;
  onEncoding?: (totalChunks: number) => void | Promise<void>;
  /** Supplied only for a mixed-precision primary engine; never a CPU or original-audio fallback. */
  recoverNumericalFailure?: ZipNeuralInference;
  /** A latched FP32 engine is never retried as if it were still mixed precision. */
  canRecoverNumericalFailure?: () => boolean;
  onNumericalRecovery?: (index: number, totalChunks: number) => void | Promise<void>;
}
export interface ZipEnhancedLane {
  bytes: Uint8Array<ArrayBuffer>;
  sampleRate: number;
  frameCount: number;
  totalChunks: number;
}

async function inferWindow(
  spectrum: ZipSpectrum, infer: ZipNeuralInference, options: ZipPipelineOptions, index: number, totalChunks: number
): Promise<ZipSpectrum> {
  const { magnitude, phase, frames } = spectrum;
  // Inference/device/audit exceptions are fatal. Only validation of completed output can authorize recovery.
  let enhanced = await infer(magnitude, phase, frames);
  throwIfZipCancelled(options.signal);
  try {
    validateZipSpectrum(enhanced.magnitude, enhanced.phase, frames);
  } catch (error) {
    if (!(error instanceof ZipInvalidSpectrumError) || !options.recoverNumericalFailure ||
      options.canRecoverNumericalFailure?.() === false) throw error;
    await options.onNumericalRecovery?.(index, totalChunks);
    throwIfZipCancelled(options.signal);
    enhanced = await options.recoverNumericalFailure(magnitude, phase, frames);
    throwIfZipCancelled(options.signal);
    // Recovery is exactly once for this window; invalid FP32 output remains a terminal failure.
    validateZipSpectrum(enhanced.magnitude, enhanced.phase, frames);
  }
  return { ...enhanced, frames };
}

/** Consumes owned source PCM. Only audited neural results may enter this pipeline. */
export async function enhanceZipSamplesInWorker(
  source: Float32Array, sampleRate: number, infer: ZipNeuralInference, options: ZipPipelineOptions = {}
): Promise<ZipEnhancedLane> {
  throwIfZipCancelled(options.signal);
  const frameCount = source.length;
  const client = options.client ?? new ZipDspWorkerClient((options.createWorker ?? createPackagedZipDspWorker)(), options.signal);
  let succeeded = false;
  let neuralFlight: Promise<ZipSpectrum> | undefined;
  let lookahead: Promise<ZipSpectrum | undefined> | undefined;
  let reconstruction: Promise<void> | undefined;
  try {
    const { totalChunks } = await client.initialize(source, sampleRate);
    await options.onChunksReady?.(totalChunks);
    throwIfZipCancelled(options.signal);
    let spectrum = await client.nextSpectrum(0);
    let previous: ZipSpectrum | undefined;
    for (let index = 0; index < totalChunks; index++) {
      throwIfZipCancelled(options.signal);
      // Start the sole neural flight before posting prior ISTFT/copy and next STFT.
      neuralFlight = inferWindow(spectrum, infer, options, index, totalChunks);
      reconstruction = previous
        ? client.reconstruct(index - 1, previous).then(async (completedChunks) => {
          throwIfZipCancelled(options.signal);
          await options.onChunkCompleted?.(completedChunks, totalChunks);
          throwIfZipCancelled(options.signal);
        }) : Promise.resolve();
      lookahead = index + 1 < totalChunks ? client.nextSpectrum(index + 1) : Promise.resolve(undefined);
      const [, enhanced, next] = await Promise.all([reconstruction, neuralFlight, lookahead]);
      neuralFlight = undefined; reconstruction = undefined; lookahead = undefined;
      previous = enhanced;
      if (next) spectrum = next;
    }
    if (!previous) throw new Error('ZipEnhancer did not infer the complete source lane.');
    const completedChunks = await client.reconstruct(totalChunks - 1, previous);
    throwIfZipCancelled(options.signal);
    await options.onChunkCompleted?.(completedChunks, totalChunks);
    throwIfZipCancelled(options.signal);
    await options.onEncoding?.(totalChunks);
    throwIfZipCancelled(options.signal);
    const result = await client.finish();
    if (result.frameCount !== frameCount || result.sampleRate !== sampleRate || result.totalChunks !== totalChunks) {
      throw new Error('ZipEnhancer DSP did not preserve the complete source clock.');
    }
    throwIfZipCancelled(options.signal);
    succeeded = true;
    return { bytes: result.bytes, sampleRate, frameCount, totalChunks };
  } finally {
    // Reject speculative DSP requests immediately, but retain GPU/profiler admission
    // until the actual flight settles, including callback rejection and cancellation.
    if (!succeeded || !options.client) client.close();
    await Promise.allSettled([neuralFlight, lookahead, reconstruction]);
  }
}
