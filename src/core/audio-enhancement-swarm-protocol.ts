import type { AudioEnhancementProgress, AudioEnhancementTrackMetadata } from '@nominy/babel-babel-runtime';
import { enhancementSha256 } from './audio-enhancement-cache';
import { readEnhancementWavHeader } from './audio-enhancement-dsp';
import { LOCAL_MODEL_AUDIO_CHUNK_BYTES } from './local-model-offscreen-protocol';

export const ENHANCEMENT_MODEL_ID = 'zipenhancer-webgpu-2026-10-09-r1';
export const ENHANCEMENT_SOURCE_GRAPH_SHA256 = '2f18c8f7ff10a2702d6243ce1230db9e73e6804dd6cd7b20d8e191ee06924016';
export const ENHANCEMENT_MAX_TRACK_BYTES = 240 * 1024 * 1024;
export const ENHANCEMENT_MAX_REQUEST_BYTES = 500 * 1024 * 1024;
export const ENHANCEMENT_MAX_SECONDS = 4 * 60 * 60;
export const ENHANCEMENT_CONTROL_BYTES = 64 * 1024;
const SHA256 = /^[a-f0-9]{64}$/;

export interface EnhancementModelDescriptor { id: string; sha256: string; sourceGraphSha256: string }
export interface EnhancementSwarmTrack {
  trackId: string;
  speakerKey: string;
  trackLabel: string;
  fieldName: 'audio:1' | 'audio:2';
  sourceSha256: string;
  sampleRate: number;
  frameCount: number;
}
export interface EnhancementSwarmPayload { taskId: string; model: EnhancementModelDescriptor; tracks: EnhancementSwarmTrack[] }
export type EnhancementWorkerProgress = Pick<AudioEnhancementProgress, 'phase' | 'trackId' | 'trackIndex' | 'trackCount' | 'completedChunks' | 'totalChunks'>;

export function swarmRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}
function text(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0 && value.length <= 256 && !/[\u0000-\u001f\u007f]/.test(value);
}
function positive(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0;
}
export function parseEnhancementModel(value: unknown): EnhancementModelDescriptor {
  if (!swarmRecord(value) || Object.keys(value).some(key => !['id', 'sha256', 'sourceGraphSha256'].includes(key)) ||
      value.id !== ENHANCEMENT_MODEL_ID || typeof value.sha256 !== 'string' || !SHA256.test(value.sha256) ||
      value.sourceGraphSha256 !== ENHANCEMENT_SOURCE_GRAPH_SHA256) throw new Error('Incompatible swarm enhancement model descriptor.');
  return value as unknown as EnhancementModelDescriptor;
}
export function sameEnhancementModel(left: EnhancementModelDescriptor, right: EnhancementModelDescriptor): boolean {
  return left.id === right.id && left.sha256 === right.sha256 && left.sourceGraphSha256 === right.sourceGraphSha256;
}
export function parseEnhancementPayload(value: unknown): EnhancementSwarmPayload {
  if (!swarmRecord(value) || Object.keys(value).some(key => !['taskId', 'model', 'tracks'].includes(key)) ||
      !text(value.taskId) || !Array.isArray(value.tracks) || value.tracks.length !== 2) throw new Error('Invalid swarm enhancement payload.');
  parseEnhancementModel(value.model);
  if (!value.tracks.every((track, index) => swarmRecord(track) &&
      Object.keys(track).every(key => ['trackId', 'speakerKey', 'trackLabel', 'fieldName', 'sourceSha256', 'sampleRate', 'frameCount'].includes(key)) &&
      text(track.trackId) && text(track.speakerKey) && text(track.trackLabel) && track.fieldName === `audio:${index + 1}` &&
      typeof track.sourceSha256 === 'string' && SHA256.test(track.sourceSha256) &&
      positive(track.sampleRate) && track.sampleRate <= 0xffffffff && positive(track.frameCount) && track.frameCount <= 0x7fffffff &&
      track.frameCount / track.sampleRate <= ENHANCEMENT_MAX_SECONDS &&
      44 + track.frameCount * 2 <= ENHANCEMENT_MAX_TRACK_BYTES) ||
      new Set(value.tracks.map(track => track.trackId)).size !== 2) throw new Error('Invalid swarm enhancement track identities or clocks.');
  return value as unknown as EnhancementSwarmPayload;
}
/** Validates the complete RIFF, not just the first twelve bytes. Originals stay byte-identical. */
export async function verifyEnhancementWav(bytes: ArrayBuffer, track: EnhancementSwarmTrack, expectedSha256: string, output = false): Promise<void> {
  if (bytes.byteLength > ENHANCEMENT_MAX_TRACK_BYTES || bytes.byteLength < 44) throw new Error('Swarm WAV exceeds the bounded audio size.');
  const header = readEnhancementWavHeader(bytes), view = new DataView(bytes);
  if (view.getUint32(4, true) !== bytes.byteLength - 8 || header.sampleRate !== track.sampleRate || header.frameCount !== track.frameCount) {
    throw new Error('Swarm WAV source clock or RIFF length mismatch.');
  }
  let formats = 0, data = 0, offset = 12;
  for (; offset + 8 <= bytes.byteLength;) {
    const id = view.getUint32(offset, false), size = view.getUint32(offset + 4, true);
    const next = offset + 8 + size + (size & 1);
    if (next > bytes.byteLength) throw new Error('Swarm WAV has a truncated chunk.');
    if (id === 0x666d7420) {
      formats++;
      if (size < 16 || view.getUint32(offset + 16, true) !== header.sampleRate * header.blockAlign ||
          view.getUint16(offset + 20, true) !== header.blockAlign) throw new Error('Swarm WAV has an invalid byte clock.');
    }
    if (id === 0x64617461) data++;
    offset = next;
  }
  if (offset !== bytes.byteLength || formats !== 1 || data !== 1) throw new Error('Swarm WAV has ambiguous or trailing chunks.');
  if (header.format === 3) {
    for (let position = header.dataOffset; position < header.dataOffset + header.frameCount * header.blockAlign; position += 4) {
      if (!Number.isFinite(view.getFloat32(position, true))) throw new Error('Swarm WAV contains nonfinite PCM.');
    }
  }
  if (output && (header.format !== 1 || header.channels !== 1 || header.bits !== 16 || header.dataOffset !== 44 ||
      bytes.byteLength !== 44 + track.frameCount * 2 || view.getUint32(16, true) !== 16)) throw new Error('Swarm output must be canonical PCM16 mono WAV.');
  if (await enhancementSha256(bytes) !== expectedSha256) throw new Error('Swarm WAV SHA-256 mismatch.');
}

export function parseEnhancementTrackMetadata(value: unknown, source: EnhancementSwarmTrack): AudioEnhancementTrackMetadata {
  if (!swarmRecord(value) || Object.keys(value).some(key => !['trackId', 'speakerKey', 'trackLabel', 'mimeType', 'sampleRate', 'frameCount', 'sourceSha256', 'wavSha256', 'totalBytes', 'chunkCount'].includes(key)) ||
      value.trackId !== source.trackId || value.speakerKey !== source.speakerKey || value.trackLabel !== source.trackLabel ||
      value.sourceSha256 !== source.sourceSha256 || value.sampleRate !== source.sampleRate || value.frameCount !== source.frameCount ||
      value.mimeType !== 'audio/wav' || typeof value.wavSha256 !== 'string' || !SHA256.test(value.wavSha256) ||
      value.totalBytes !== 44 + source.frameCount * 2 || value.chunkCount !== Math.ceil((value.totalBytes as number) / LOCAL_MODEL_AUDIO_CHUNK_BYTES)) {
    throw new Error('Swarm output metadata does not match the Original recording.');
  }
  return value as unknown as AudioEnhancementTrackMetadata;
}

export function parseEnhancementWorkerProgress(value: unknown, payload: EnhancementSwarmPayload, previous?: EnhancementWorkerProgress): EnhancementWorkerProgress {
  if (!swarmRecord(value) || Object.keys(value).some(key => !['phase', 'trackId', 'trackIndex', 'trackCount', 'completedChunks', 'totalChunks'].includes(key)) ||
      !['loading-model', 'enhancing', 'encoding'].includes(value.phase as string) || value.trackCount !== 2 ||
      !Number.isSafeInteger(value.trackIndex) || (value.trackIndex !== 0 && value.trackIndex !== 1) ||
      value.trackId !== payload.tracks[value.trackIndex as number].trackId ||
      !Number.isSafeInteger(value.completedChunks) || !Number.isSafeInteger(value.totalChunks) ||
      (value.completedChunks as number) < 0 || (value.totalChunks as number) < (value.completedChunks as number) ||
      (value.phase === 'encoding' && value.completedChunks !== value.totalChunks)) throw new Error('Invalid swarm worker progress.');
  const progress = value as unknown as EnhancementWorkerProgress;
  if (previous && (progress.trackIndex < previous.trackIndex || (progress.trackIndex === previous.trackIndex &&
      (progress.completedChunks < previous.completedChunks || (previous.totalChunks > 0 && progress.totalChunks !== previous.totalChunks) ||
       ['loading-model', 'enhancing', 'encoding'].indexOf(progress.phase) < ['loading-model', 'enhancing', 'encoding'].indexOf(previous.phase))))) {
    throw new Error('Swarm worker progress regressed.');
  }
  return progress;
}

/** Enforces limits while streaming, including responses without a trustworthy Content-Length. */
export async function readBoundedSwarmBlob(response: Response, maximum: number, signal?: AbortSignal): Promise<Blob> {
  signal?.throwIfAborted();
  const declared = response.headers.get('Content-Length');
  if (declared !== null && (!/^\d+$/.test(declared) || Number(declared) > maximum)) {
    await response.body?.cancel();
    throw new Error('Swarm response exceeds the bounded transfer size.');
  }
  if (!response.body) throw new Error('Swarm response body is missing.');
  const reader = response.body.getReader(), chunks: Uint8Array<ArrayBuffer>[] = [];
  let size = 0;
  const abort = () => { void reader.cancel(signal?.reason).catch(() => {}); };
  signal?.addEventListener('abort', abort, { once: true });
  try {
    for (;;) {
      signal?.throwIfAborted();
      const next = await reader.read();
      signal?.throwIfAborted();
      if (next.done) break;
      size += next.value.byteLength;
      if (size > maximum) throw new Error('Swarm response exceeds the bounded transfer size.');
      chunks.push(next.value as Uint8Array<ArrayBuffer>);
    }
    if (declared !== null && Number(declared) !== size) throw new Error('Swarm response length mismatch.');
    return new Blob(chunks, { type: response.headers.get('Content-Type') ?? 'application/octet-stream' });
  } catch (error) {
    await reader.cancel(error).catch(() => {});
    throw error;
  } finally {
    signal?.removeEventListener('abort', abort);
    reader.releaseLock();
  }
}
export async function readBoundedSwarmJson(response: Response, signal?: AbortSignal): Promise<unknown> {
  return JSON.parse(await (await readBoundedSwarmBlob(response, ENHANCEMENT_CONTROL_BYTES, signal)).text());
}
