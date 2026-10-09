import type { ZipSpectrum } from './audio-enhancement-dsp';

export type ZipDspRequest =
  | { id: number; type: 'initialize'; samples: Float32Array; sampleRate: number; frameCount: number }
  | { id: number; type: 'spectrum'; index: number }
  | { id: number; type: 'reconstruct'; index: number; spectrum: ZipSpectrum }
  | { id: number; type: 'finish' };

export type ZipDspSuccess =
  | { id: number; ok: true; type: 'ready'; totalChunks: number }
  | { id: number; ok: true; type: 'spectrum'; index: number; spectrum: ZipSpectrum }
  | { id: number; ok: true; type: 'reconstructed'; index: number; completedChunks: number }
  | { id: number; ok: true; type: 'finished'; bytes: Uint8Array<ArrayBuffer>; sampleRate: number; frameCount: number; totalChunks: number };

export type ZipDspResponse = ZipDspSuccess | { id: number; ok: false; error: { name: string; message: string } };
export type ZipDspReply<Type extends ZipDspSuccess['type']> = Extract<ZipDspSuccess, { type: Type }>;

export interface ZipDspWorkerPort {
  postMessage(message: ZipDspRequest, transfer: ArrayBuffer[]): void;
  addEventListener(type: 'message' | 'error' | 'messageerror', listener: (event: Event) => void): void;
  removeEventListener(type: 'message' | 'error' | 'messageerror', listener: (event: Event) => void): void;
  terminate(): void;
}

/** Transfer only whole, exclusively owned views; ORT may otherwise return a subview. */
export function ownedZipArray(values: Float32Array): Float32Array<ArrayBuffer> {
  return values.buffer instanceof ArrayBuffer && values.byteOffset === 0 && values.byteLength === values.buffer.byteLength
    ? values as Float32Array<ArrayBuffer> : values.slice();
}

export function zipSpectrumTransfers(spectrum: ZipSpectrum): ArrayBuffer[] {
  const magnitude = spectrum.magnitude.buffer as ArrayBuffer, phase = spectrum.phase.buffer as ArrayBuffer;
  return magnitude === phase ? [magnitude] : [magnitude, phase];
}

function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Invalid ZipEnhancer DSP message.');
  return value as Record<string, unknown>;
}
function integer(value: unknown, minimum = 0): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < minimum) throw new Error('Invalid ZipEnhancer DSP clock/index.');
  return value;
}
function spectrum(value: unknown): ZipSpectrum {
  const data = object(value);
  if (!(data.magnitude instanceof Float32Array) || !(data.phase instanceof Float32Array)) throw new Error('Invalid ZipEnhancer DSP spectrum buffers.');
  return { magnitude: data.magnitude, phase: data.phase, frames: integer(data.frames, 1) };
}

export function parseZipDspRequest(value: unknown): ZipDspRequest {
  const data = object(value), id = integer(data.id, 1);
  switch (data.type) {
    case 'initialize':
      if (!(data.samples instanceof Float32Array)) throw new Error('Invalid ZipEnhancer DSP source buffer.');
      return { id, type: data.type, samples: data.samples, sampleRate: integer(data.sampleRate, 1), frameCount: integer(data.frameCount, 1) };
    case 'spectrum': return { id, type: data.type, index: integer(data.index) };
    case 'reconstruct': return { id, type: data.type, index: integer(data.index), spectrum: spectrum(data.spectrum) };
    case 'finish': return { id, type: data.type };
    default: throw new Error('Unknown ZipEnhancer DSP request.');
  }
}

export function parseZipDspResponse(value: unknown): ZipDspResponse {
  const data = object(value), id = integer(data.id);
  if (data.ok === false) {
    const error = object(data.error);
    if (typeof error.name !== 'string' || typeof error.message !== 'string') throw new Error('Invalid ZipEnhancer DSP error.');
    return { id, ok: false, error: { name: error.name, message: error.message } };
  }
  if (data.ok !== true) throw new Error('Invalid ZipEnhancer DSP acknowledgement.');
  switch (data.type) {
    case 'ready': return { id, ok: true, type: data.type, totalChunks: integer(data.totalChunks, 1) };
    case 'spectrum': return { id, ok: true, type: data.type, index: integer(data.index), spectrum: spectrum(data.spectrum) };
    case 'reconstructed': return { id, ok: true, type: data.type, index: integer(data.index), completedChunks: integer(data.completedChunks, 1) };
    case 'finished':
      if (!(data.bytes instanceof Uint8Array) || !(data.bytes.buffer instanceof ArrayBuffer) ||
        data.bytes.byteOffset !== 0 || data.bytes.byteLength !== data.bytes.buffer.byteLength) throw new Error('Invalid ZipEnhancer DSP WAV buffer.');
      return { id, ok: true, type: data.type, bytes: data.bytes as Uint8Array<ArrayBuffer>,
        sampleRate: integer(data.sampleRate, 1), frameCount: integer(data.frameCount, 1), totalChunks: integer(data.totalChunks, 1) };
    default: throw new Error('Unknown ZipEnhancer DSP response.');
  }
}
