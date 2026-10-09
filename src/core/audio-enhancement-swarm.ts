import type { AudioEnhancementProgress } from '@nominy/babel-babel-runtime';
import type { PreparedEnhancementSource } from './audio-enhancement-cache';
import type { EnhancedAudioTrack } from './audio-enhancement-runtime';
import { INFERENCE_HEADERS } from './inference-release';
import { normalizeL0CustomBaseUrl } from './settings';
import {
  ENHANCEMENT_CONTROL_BYTES, ENHANCEMENT_MAX_REQUEST_BYTES, parseEnhancementPayload,
  parseEnhancementTrackMetadata, parseEnhancementWorkerProgress, readBoundedSwarmBlob,
  readBoundedSwarmJson, swarmRecord, verifyEnhancementWav,
  type EnhancementModelDescriptor, type EnhancementSwarmPayload, type EnhancementWorkerProgress
} from './audio-enhancement-swarm-protocol';

const REQUEST_TIMEOUT_MS = 15 * 60_000;
const STATUS_TIMEOUT_MS = 15_000;
const POLL_MS = 1_000;

export interface EnhancementSwarmOptions {
  taskId: string;
  signal?: AbortSignal;
  onProgress?: (progress: AudioEnhancementProgress) => void | Promise<void>;
  /** Internal transport override for isolated coordinators; never sourced from page operations. */
  baseUrl?: string;
}

async function coordinatorBaseUrl(override?: string): Promise<string> {
  if (override !== undefined) {
    const endpoint = new URL(override);
    if (endpoint.protocol !== 'http:' && endpoint.protocol !== 'https:') throw new Error('The internal swarm coordinator override must be HTTP or HTTPS.');
    return normalizeL0CustomBaseUrl(override);
  }
  const settings: unknown = await chrome.runtime.sendMessage({ type: 'babel-l0-volunteer', target: 'background', action: 'settings' });
  if (!swarmRecord(settings)) throw new Error('Saved swarm coordinator settings are unavailable.');
  return normalizeL0CustomBaseUrl(settings.l0CustomBaseUrl);
}

async function waitForPoll(signal: AbortSignal): Promise<void> {
  signal.throwIfAborted();
  await new Promise<void>((resolve, reject) => {
    const abort = () => { clearTimeout(timer); reject(signal.reason); };
    const timer = setTimeout(() => { signal.removeEventListener('abort', abort); resolve(); }, POLL_MS);
    signal.addEventListener('abort', abort, { once: true });
  });
}

async function receiveTracks(response: Response, payload: EnhancementSwarmPayload, signal: AbortSignal): Promise<EnhancedAudioTrack[]> {
  if (!/^multipart\/form-data\s*;/i.test(response.headers.get('Content-Type') ?? '')) throw new Error('Swarm enhancement did not return binary multipart audio.');
  const maximum = payload.tracks.reduce((sum, track) => sum + 44 + track.frameCount * 2, ENHANCEMENT_CONTROL_BYTES);
  const blob = await readBoundedSwarmBlob(response, maximum, signal);
  const form = await new Response(blob, { headers: { 'Content-Type': response.headers.get('Content-Type')! } }).formData();
  signal.throwIfAborted();
  const fields = Array.from(form.keys());
  if (fields.length !== 3 || new Set(fields).size !== 3 || !fields.every(field => ['result', 'audio:1', 'audio:2'].includes(field))) {
    throw new Error('Swarm enhancement returned missing, duplicate, or unexpected audio fields.');
  }
  const resultText = form.get('result');
  if (typeof resultText !== 'string' || resultText.length > ENHANCEMENT_CONTROL_BYTES) throw new Error('Invalid swarm enhancement result metadata.');
  const result: unknown = JSON.parse(resultText);
  if (!swarmRecord(result) || result.ok !== true || result.provider !== 'swarm' || result.taskId !== payload.taskId ||
      result.model !== payload.model.id || result.modelSha256 !== payload.model.sha256 || !Array.isArray(result.tracks) || result.tracks.length !== 2) {
    throw new Error('Swarm enhancement returned a different task or model.');
  }
  const tracks: EnhancedAudioTrack[] = [];
  for (let index = 0; index < payload.tracks.length; index++) {
    const source = payload.tracks[index], metadata = parseEnhancementTrackMetadata(result.tracks[index], source);
    const part = form.get(source.fieldName);
    if (!(part instanceof Blob) || part.size !== metadata.totalBytes) throw new Error('Swarm enhanced WAV length mismatch.');
    const bytes = await part.arrayBuffer();
    await verifyEnhancementWav(bytes, source, metadata.wavSha256, true);
    signal.throwIfAborted();
    tracks.push({ metadata, bytes: new Uint8Array(bytes) });
  }
  return tracks;
}

/** One authenticated owner request; audio stays binary and every returned lane is independently verified. */
export async function enhanceAudioThroughSwarm(
  sources: PreparedEnhancementSource[], model: EnhancementModelDescriptor, options: EnhancementSwarmOptions
): Promise<EnhancedAudioTrack[]> {
  options.signal?.throwIfAborted();
  const payload = parseEnhancementPayload({ taskId: options.taskId, model, tracks: sources.map((source, index) => ({
    trackId: source.trackId, speakerKey: source.track.speakerKey ?? source.trackId,
    trackLabel: source.track.trackLabel ?? source.track.speakerKey ?? source.trackId,
    fieldName: `audio:${index + 1}`, sourceSha256: source.sourceSha256, sampleRate: source.sampleRate, frameCount: source.frameCount
  })) });
  if (sources.reduce((sum, source) => sum + source.bytes.byteLength, ENHANCEMENT_CONTROL_BYTES) > ENHANCEMENT_MAX_REQUEST_BYTES) {
    throw new Error('Original recordings exceed the swarm upload limit; Originals remain selected.');
  }
  const controller = new AbortController(), pollController = new AbortController();
  const timer = setTimeout(() => controller.abort(new Error('Swarm enhancement timed out; Originals remain selected. Try again when a compatible GPU volunteer is available.')), REQUEST_TIMEOUT_MS);
  const signal = options.signal ? AbortSignal.any([options.signal, controller.signal]) : controller.signal;
  const pollSignal = AbortSignal.any([signal, pollController.signal]);
  let polling: Promise<void> | undefined;
  let pollFailure: unknown;
  let receiving = false;
  try {
    const baseUrl = await coordinatorBaseUrl(options.baseUrl);
    signal.throwIfAborted();
    const form = new FormData();
    form.append('payload', JSON.stringify(payload));
    for (let index = 0; index < sources.length; index++) {
      await verifyEnhancementWav(sources[index].bytes, payload.tracks[index], sources[index].sourceSha256);
      signal.throwIfAborted();
      form.append(payload.tracks[index].fieldName, new Blob([sources[index].bytes], { type: 'audio/wav' }), `original-${index + 1}.wav`);
    }
    const requestId = crypto.randomUUID();
    const ownerToken = btoa(String.fromCharCode(...crypto.getRandomValues(new Uint8Array(32)))).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
    const headers = { ...INFERENCE_HEADERS, Authorization: `Bearer ${ownerToken}` };
    await options.onProgress?.({ phase: 'uploading', backend: 'swarm', backendReason: 'Uploading both Original recordings to the configured coordinator for a volunteer GPU.',
      trackId: payload.tracks[0].trackId, trackIndex: 0, trackCount: 2, completedChunks: 0, totalChunks: 0 });
    signal.throwIfAborted();
    const pending = fetch(`${baseUrl}/v1/enhance`, { method: 'POST', headers: { ...headers, 'X-Babel-Local-Engine': '1', 'X-Babel-Request-Id': requestId }, body: form, redirect: 'error', signal });
    polling = (async () => {
      let previous: EnhancementWorkerProgress | undefined;
      while (!pollSignal.aborted) {
        await waitForPoll(pollSignal);
        if (receiving) await options.onProgress?.({ phase: 'transferring', backend: 'swarm',
          trackId: payload.tracks[0].trackId, trackIndex: 0, trackCount: 2, completedChunks: 0, totalChunks: 0 });
        const response = await fetch(`${baseUrl}/v1/queue/${encodeURIComponent(requestId)}`, {
          // Revalidate ownership during downloads too; navigation must cancel a stalled transfer.
          headers, redirect: 'error', signal: AbortSignal.any([pollSignal, AbortSignal.timeout(STATUS_TIMEOUT_MS)])
        });
        if (response.status === 404) { await response.body?.cancel(); continue; }
        if (!response.ok) { await response.body?.cancel(); throw new Error(`Swarm status authorization or availability failed: HTTP ${response.status}`); }
        const status = await readBoundedSwarmJson(response, pollSignal);
        if (!swarmRecord(status) || status.requestId !== requestId || !['queued', 'running', 'completed'].includes(String(status.status))) throw new Error('Invalid swarm queue status.');
        if (receiving) continue;
        if (status.progress !== undefined) {
          const progress = parseEnhancementWorkerProgress(status.progress, payload, previous);
          await options.onProgress?.({ ...progress, backend: 'swarm' });
          previous = progress;
        } else if (!previous) {
          await options.onProgress?.({ phase: 'queued', backend: 'swarm', backendReason: 'Waiting for a compatible volunteer GPU; Original audio has been sent to the coordinator.',
            trackId: payload.tracks[0].trackId, trackIndex: 0, trackCount: 2, completedChunks: 0, totalChunks: 0 });
        }
      }
    })().catch(error => {
      if (!pollSignal.aborted) { pollFailure = error; controller.abort(error); }
    });
    const response = await pending;
    signal.throwIfAborted();
    if (!response.ok) {
      const error = await readBoundedSwarmJson(response, signal).catch(() => undefined);
      const detail = swarmRecord(error) && typeof error.detail === 'string' ? error.detail.slice(0, 500)
        : swarmRecord(error) && typeof error.error === 'string' ? error.error.slice(0, 500) : `HTTP ${response.status}`;
      throw new Error(`Swarm enhancement unavailable: ${detail}. Originals remain selected; enable a compatible GPU volunteer and try again.`);
    }
    receiving = true;
    await options.onProgress?.({ phase: 'transferring', backend: 'swarm',
      trackId: payload.tracks[0].trackId, trackIndex: 0, trackCount: 2, completedChunks: 0, totalChunks: 0 });
    return await receiveTracks(response, payload, signal);
  } catch (error) {
    controller.abort(error);
    throw pollFailure ?? error;
  } finally {
    clearTimeout(timer);
    pollController.abort();
    await polling;
  }
}
