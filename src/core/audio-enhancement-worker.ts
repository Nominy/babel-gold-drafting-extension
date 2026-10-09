import { resamplePoly } from './c-denoise-acoustic';
import {
  createZipStftSequence, encodeEnhancementWav, zipIstft,
  ZIP_CHUNK_SAMPLES, ZIP_SAMPLE_RATE, ZIP_STRIDE_SAMPLES, type ZipSpectrum
} from './audio-enhancement-dsp';
import { parseZipDspRequest, zipSpectrumTransfers, type ZipDspResponse } from './audio-enhancement-worker-protocol';

/** One owned source lane, at most one speculative spectrum, no neural work. */
export function createZipDspWorkerHandler(post: (response: ZipDspResponse, transfer: ArrayBuffer[]) => void, close: () => void) {
  let lane: {
    samples: Float32Array; output: Float32Array; chunk: Float32Array; scale: number;
    sampleRate: number; frameCount: number; totalChunks: number; prepared: number; completed: number;
    stft: (samples: Float32Array) => ZipSpectrum;
  } | undefined;
  let stopped = false;
  let lastId = 0;
  return (message: unknown): void => {
    if (stopped) return;
    let id = 0;
    try {
      const request = parseZipDspRequest(message);
      id = request.id;
      if (id <= lastId) throw new Error('ZipEnhancer DSP requests must be strictly ordered.');
      lastId = id;
      if (request.type === 'initialize') {
        if (lane) throw new Error('ZipEnhancer DSP worker already owns a source lane.');
        // Preserve the source clock before any transferred source buffer is released.
        const { samples: source, sampleRate, frameCount } = request;
        if (source.length !== frameCount || frameCount < 1) throw new Error('ZipEnhancer DSP source clock is invalid.');
        const samples = resamplePoly(source, sampleRate, ZIP_SAMPLE_RATE);
        let power = 0;
        for (const value of samples) {
          if (!Number.isFinite(value)) throw new Error('ZipEnhancer source PCM is nonfinite.');
          power += value * value;
        }
        if (!(power > 0)) throw new Error('ZipEnhancer cannot enhance an all-zero source lane: the reference RMS normalization is undefined.');
        const scale = Math.sqrt(samples.length / power);
        const paddedLength = ZIP_CHUNK_SAMPLES + Math.max(0, Math.ceil((samples.length + ZIP_SAMPLE_RATE - ZIP_CHUNK_SAMPLES) / ZIP_STRIDE_SAMPLES)) * ZIP_STRIDE_SAMPLES;
        const totalChunks = 1 + (paddedLength - ZIP_CHUNK_SAMPLES) / ZIP_STRIDE_SAMPLES;
        lane = { samples, scale, sampleRate, frameCount, totalChunks, prepared: 0, completed: 0,
          output: new Float32Array(samples.length), chunk: new Float32Array(ZIP_CHUNK_SAMPLES), stft: createZipStftSequence() };
        post({ id, ok: true, type: 'ready', totalChunks }, []);
        return;
      }
      if (!lane) throw new Error('ZipEnhancer DSP source lane is unavailable.');
      if (request.type === 'spectrum') {
        if (request.index !== lane.prepared || request.index >= lane.totalChunks || lane.prepared > lane.completed + 1) {
          throw new Error('ZipEnhancer DSP spectrum exceeds the ordered one-window lookahead.');
        }
        const start = request.index * ZIP_STRIDE_SAMPLES;
        lane.chunk.fill(0);
        for (let index = 0; index < Math.min(lane.chunk.length, lane.samples.length - start); index++) {
          lane.chunk[index] = lane.samples[start + index] * lane.scale;
        }
        const spectrum = lane.stft(lane.chunk);
        lane.prepared++;
        post({ id, ok: true, type: 'spectrum', index: request.index, spectrum }, zipSpectrumTransfers(spectrum));
      } else if (request.type === 'reconstruct') {
        if (request.index !== lane.completed || request.index >= lane.prepared) throw new Error('ZipEnhancer DSP reconstruction is out of order.');
        const { magnitude, phase, frames } = request.spectrum;
        const waveform = zipIstft(magnitude, phase, frames), start = request.index * ZIP_STRIDE_SAMPLES;
        const edge = (ZIP_CHUNK_SAMPLES - ZIP_STRIDE_SAMPLES) / 2;
        const left = start === 0 ? 0 : edge, right = Math.min(ZIP_CHUNK_SAMPLES - edge, lane.samples.length - start);
        for (let index = left; index < right; index++) lane.output[start + index] = waveform[index] / lane.scale;
        lane.completed++;
        post({ id, ok: true, type: 'reconstructed', index: request.index, completedChunks: lane.completed }, []);
      } else {
        if (lane.completed !== lane.totalChunks || lane.prepared !== lane.totalChunks) throw new Error('ZipEnhancer DSP cannot publish a partial source lane.');
        const { sampleRate, frameCount, totalChunks } = lane;
        const waveform = resamplePoly(lane.output, ZIP_SAMPLE_RATE, sampleRate);
        const bytes = encodeEnhancementWav(waveform, sampleRate, frameCount);
        lane = undefined;
        post({ id, ok: true, type: 'finished', bytes, sampleRate, frameCount, totalChunks }, [bytes.buffer]);
      }
    } catch (error) {
      lane = undefined;
      stopped = true;
      post({ id, ok: false, error: { name: error instanceof Error ? error.name : 'Error', message: error instanceof Error ? error.message : String(error) } }, []);
      close();
    }
  };
}
