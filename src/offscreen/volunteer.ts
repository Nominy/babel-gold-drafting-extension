import { assertL0WavAudio, type PreparedL0Track } from '../core/l0-client';
import { INFERENCE_RELEASE, INFERENCE_HEADERS, assertReleasedGraphs } from '../core/inference-release';
import { IS_DEV_C_DENOISE, isBrowserLocalMode, LOCAL_MODEL_BASE_URL, normalizeL0CustomBaseUrl, normalizeSettings } from '../core/settings';
import { getCachedBundleDescriptor } from '../core/local-model-bundle';
import type { CapturedAudioTrack, ExtensionSettings, L0DraftResponse, L0TimingResponse, TranscriptJob, TranscriptRow } from '../core/types';
import type { VolunteerStatus } from '../core/volunteer-protocol';
import { generateLocalL0DraftFromTiming, generateLocalL0Timing } from '../core/local-model-runtime';
import { parseL0TimingResponse } from '../core/l0-timing-client';
import { selectZipEnhancementBackend } from '../core/audio-enhancement-backend';
import { enhanceAudioTracks, getAudioEnhancementModel, type EnhancedAudioBatch } from '../core/audio-enhancement-runtime';
import {
  ENHANCEMENT_MAX_TRACK_BYTES, parseEnhancementModel, parseEnhancementPayload, parseEnhancementTrackMetadata,
  parseEnhancementWorkerProgress, readBoundedSwarmBlob, readBoundedSwarmJson, sameEnhancementModel, verifyEnhancementWav,
  type EnhancementModelDescriptor, type EnhancementSwarmPayload, type EnhancementWorkerProgress
} from '../core/audio-enhancement-swarm-protocol';

const SCHEMA = INFERENCE_RELEASE.bundleSchema;
const CONTROL_REQUEST_TIMEOUT_MS = 45_000;
const AUDIO_REQUEST_TIMEOUT_MS = 5 * 60_000;
// Draft leases carry complete word timings, unlike small registration/status
// messages. Match the coordinator's bounded 16 MiB worker JSON contract.
const LEASE_RESPONSE_BYTES = 16 * 1024 * 1024;
const IDLE_POLL_MS = 3_000;
const MAX_BACKOFF_MS = 30_000;

type Credentials = { workerId: string; token: string };
type LeaseBase = { jobId: string; leaseToken: string };
type TranscribeLease = LeaseBase & {
  operation: 'transcribe';
  payload: { taskId: string; tracks: Array<{ lane: string; fieldName: string }> };
  audio: Array<{ fieldName: string; url: string }>;
};
type DraftLease = LeaseBase & {
  operation: 'draft';
  payload: { taskId: string; timing: L0TimingResponse; options?: Record<string, unknown> };
  audio: [];
};
type EnhanceLease = LeaseBase & {
  operation: 'enhance';
  payload: EnhancementSwarmPayload;
  audio: Array<{ fieldName: string; url: string }>;
};
type Lease = TranscribeLease | DraftLease | EnhanceLease;

export interface VolunteerReadiness {
  transcribe: boolean;
  draft: boolean;
  enhancementModel?: EnhancementModelDescriptor;
}

export interface VolunteerDependencies {
  fetch: (url: string, init: RequestInit) => Promise<Response>;
  settings: () => Promise<ExtensionSettings>;
  ready: () => Promise<VolunteerReadiness>;
  runExclusive: <T>(action: () => Promise<T>) => Promise<T>;
  draft: typeof generateLocalL0DraftFromTiming;
  transcribe: typeof generateLocalL0Timing;
  enhance: typeof enhanceAudioTracks;
  wait: (ms: number, signal: AbortSignal) => Promise<void>;
}

export async function loadVolunteerSettings(): Promise<ExtensionSettings> {
  const settings: unknown = await chrome.runtime.sendMessage({
    type: 'babel-l0-volunteer', target: 'background', action: 'settings'
  });
  if (!record(settings) || typeof settings.localModelsEnabled !== 'boolean') {
    throw new Error('The background worker could not read saved local model settings.');
  }
  return normalizeSettings(settings);
}

export const defaultVolunteerDependencies: VolunteerDependencies = {
  fetch: (url, init) => fetch(url, init),
  settings: loadVolunteerSettings,
  ready: async () => {
    let asr = false;
    try {
      const bundle = await getCachedBundleDescriptor(LOCAL_MODEL_BASE_URL);
      if (bundle?.tested) { assertReleasedGraphs(bundle.files); asr = true; }
    } catch { /* An unavailable ASR bundle must not prevent packaged enhancement admission. */ }
    let enhancementModel: EnhancementModelDescriptor | undefined;
    try {
      if ((await selectZipEnhancementBackend()).backend === 'webgpu') enhancementModel = getAudioEnhancementModel();
    } catch { /* Enhancement admission is independent of the verified ASR bundle. */ }
    return { transcribe: asr, draft: asr, ...(enhancementModel ? { enhancementModel } : {}) };
  },
  runExclusive: (action) => action(),
  draft: generateLocalL0DraftFromTiming,
  transcribe: generateLocalL0Timing,
  enhance: enhanceAudioTracks,
  wait: (ms, signal) => new Promise<void>((resolve) => {
    if (signal.aborted) { resolve(); return; }
    const finish = () => {
      clearTimeout(timer);
      signal.removeEventListener('abort', finish);
      resolve();
    };
    const timer = setTimeout(finish, ms);
    signal.addEventListener('abort', finish, { once: true });
  })
};

function record(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function parseCredentials(value: unknown): Credentials {
  if (!record(value) || typeof value.workerId !== 'string' || !value.workerId ||
      typeof value.token !== 'string' || !value.token) throw new Error('Invalid worker registration response.');
  return { workerId: value.workerId, token: value.token };
}

export function parseLease(value: unknown): Lease {
  if (!record(value) || typeof value.jobId !== 'string' || !value.jobId ||
      typeof value.leaseToken !== 'string' || !value.leaseToken ||
      (value.operation !== 'draft' && value.operation !== 'transcribe' && value.operation !== 'enhance') ||
      !record(value.payload) || typeof value.payload.taskId !== 'string' || !value.payload.taskId ||
      !Array.isArray(value.audio)) throw new Error('Invalid volunteer lease.');
  if (value.operation === 'enhance') {
    const payload = parseEnhancementPayload(value.payload);
    if (value.audio.length !== 2 || !value.audio.every(entry => record(entry) &&
        typeof entry.fieldName === 'string' && payload.tracks.some(track => track.fieldName === entry.fieldName) &&
        entry.url === `/v1/jobs/${encodeURIComponent(value.jobId as string)}/audio/${encodeURIComponent(entry.fieldName)}`) ||
        new Set(value.audio.map(entry => entry.fieldName)).size !== 2) throw new Error('Invalid enhancement lease audio URLs.');
    return value as EnhanceLease;
  }
  if (value.operation === 'draft') {
    if (value.audio.length || (value.payload.options !== undefined && !record(value.payload.options))) {
      throw new Error('A punctuation lease cannot include audio.');
    }
    parseL0TimingResponse(value.payload.timing, value.payload.taskId);
    return value as DraftLease;
  }
  if (!Array.isArray(value.payload.tracks) || value.payload.tracks.length !== 2 ||
      value.audio.length !== 2 || value.payload.options !== undefined) throw new Error('Invalid transcription lease.');
  const tracks = value.payload.tracks;
  const audio = value.audio;
  if (!tracks.every((track) => record(track) && typeof track.lane === 'string' && !!track.lane &&
      (track.fieldName === 'audio:1' || track.fieldName === 'audio:2')) ||
      new Set(tracks.map((track) => track.lane)).size !== 2 ||
      new Set(tracks.map((track) => track.fieldName)).size !== 2 ||
      !audio.every((entry) => record(entry) && typeof entry.fieldName === 'string' &&
        typeof entry.url === 'string' && entry.url === `/v1/jobs/${encodeURIComponent(value.jobId as string)}/audio/${encodeURIComponent(entry.fieldName as string)}`) ||
      new Set(audio.map((entry) => entry.fieldName)).size !== 2 ||
      !tracks.every((track) => audio.some((entry) => entry.fieldName === track.fieldName))) {
    throw new Error('Invalid transcription lease tracks or audio URLs.');
  }
  return value as TranscribeLease;
}

function preserveRows(lease: DraftLease): TranscriptRow[] | undefined {
  const options = lease.payload.options;
  if (options === undefined) return undefined;
  if (Object.keys(options).some((key) => key !== 'preserveRows' && key !== 'preprocessing') ||
      (options.preprocessing !== undefined && options.preprocessing !== 'raw')) {
    throw new Error('Unsupported draft options for browser inference.');
  }
  if (options.preserveRows === undefined) return undefined;
  if (!Array.isArray(options.preserveRows) || !options.preserveRows.length ||
      !options.preserveRows.every((row) => record(row) && typeof row.rowId === 'string' && !!row.rowId &&
        typeof row.speakerKey === 'string' && lease.payload.timing.tracks.some((track) => track.lane === row.speakerKey) &&
        typeof row.startSeconds === 'number' && Number.isFinite(row.startSeconds) && row.startSeconds >= 0 &&
        typeof row.endSeconds === 'number' && Number.isFinite(row.endSeconds) && row.endSeconds > row.startSeconds &&
        typeof row.text === 'string' && Number.isSafeInteger(row.index) && (row.index as number) >= 0)) {
    throw new Error('Invalid preserveRows for browser inference.');
  }
  return options.preserveRows as TranscriptRow[];
}
async function request(dependencies: VolunteerDependencies, baseUrl: string, path: string, init: RequestInit, signal: AbortSignal): Promise<Response> {
  return dependencies.fetch(`${baseUrl}${path}`, {
    ...init,
    headers: { ...INFERENCE_HEADERS, ...init.headers },
    redirect: 'error',
    signal: AbortSignal.any([
      signal,
      AbortSignal.timeout(path.includes('/audio/') || init.body instanceof FormData ? AUDIO_REQUEST_TIMEOUT_MS : CONTROL_REQUEST_TIMEOUT_MS)
    ])
  });
}

async function executeLease(lease: Lease, settings: ExtensionSettings, readiness: VolunteerReadiness, credentials: Credentials,
  dependencies: VolunteerDependencies, baseUrl: string, signal: AbortSignal): Promise<L0DraftResponse | L0TimingResponse | EnhancedAudioBatch> {
  if (lease.operation === 'enhance') {
    signal.throwIfAborted();
    if (!readiness.enhancementModel || !sameEnhancementModel(readiness.enhancementModel, lease.payload.model)) {
      throw new Error('This worker is not admitted for the leased enhancement model.');
    }
    const tracks: CapturedAudioTrack[] = [];
    for (const track of lease.payload.tracks) {
      signal.throwIfAborted();
      const entry = lease.audio.find(audio => audio.fieldName === track.fieldName)!;
      const response = await request(dependencies, baseUrl, entry.url, { headers: { Authorization: `Bearer ${lease.leaseToken}` } }, signal);
      if (!response.ok) { await response.body?.cancel(); throw new Error(`Leased audio fetch failed: HTTP ${response.status}`); }
      const blob = await readBoundedSwarmBlob(response, ENHANCEMENT_MAX_TRACK_BYTES, signal);
      await verifyEnhancementWav(await blob.arrayBuffer(), track, track.sourceSha256);
      tracks.push({ trackId: track.trackId, speakerKey: track.speakerKey, trackLabel: track.trackLabel,
        source: 'volunteer-lease', mimeType: 'audio/wav', blob });
    }
    let previous: EnhancementWorkerProgress | undefined;
    return dependencies.runExclusive(async () => {
      signal.throwIfAborted();
      return dependencies.enhance(tracks, async progress => {
        if (!['loading-model', 'enhancing', 'encoding'].includes(progress.phase)) return;
        const next = parseEnhancementWorkerProgress({
          phase: progress.phase, trackId: progress.trackId, trackIndex: progress.trackIndex,
          trackCount: progress.trackCount, completedChunks: progress.completedChunks, totalChunks: progress.totalChunks
        }, lease.payload, previous);
        signal.throwIfAborted();
        const response = await request(dependencies, baseUrl, `/v1/jobs/${encodeURIComponent(lease.jobId)}/progress`, {
          method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${lease.leaseToken}` },
          body: JSON.stringify({ ...credentials, leaseToken: lease.leaseToken, progress: next })
        }, signal);
        await response.body?.cancel();
        if (!response.ok) throw new Error(`Enhancement lease progress rejected: HTTP ${response.status}`);
        previous = next;
      }, { localOnly: true, cache: false, taskId: lease.payload.taskId, signal });
    });
  }
  if (!readiness[lease.operation]) throw new Error('The verified ASR bundle is unavailable for this lease.');
  if (lease.operation === 'draft') {
    return dependencies.runExclusive(async () => {
      if (signal.aborted) throw new Error('Volunteer participation stopped.');
      return dependencies.draft(lease.payload.timing, preserveRows(lease));
    });
  }
  const prepared = await Promise.all(lease.payload.tracks.map(async (track): Promise<PreparedL0Track> => {
    const entry = lease.audio.find((audio) => audio.fieldName === track.fieldName)!;
    const response = await request(dependencies, baseUrl, entry.url, { headers: { Authorization: `Bearer ${lease.leaseToken}` } }, signal);
    if (!response.ok) throw new Error(`Leased audio fetch failed: HTTP ${response.status}`);
    const blob = await readBoundedSwarmBlob(response, ENHANCEMENT_MAX_TRACK_BYTES, signal);
    const audio: CapturedAudioTrack = {
      trackId: track.fieldName, speakerKey: track.lane, trackLabel: track.lane,
      source: 'volunteer-lease', blob, mimeType: 'audio/wav'
    };
    const preparedTrack: PreparedL0Track = {
      lane: track.lane, fieldName: track.fieldName as PreparedL0Track['fieldName'], audio
    };
    await assertL0WavAudio(preparedTrack);
    return preparedTrack;
  }));
  return dependencies.runExclusive(async () => {
    if (signal.aborted) throw new Error('Volunteer participation stopped.');
    const job: TranscriptJob = {
      jobId: lease.payload.taskId,
      rows: lease.payload.tracks.map((track, index) => ({
        rowId: `${lease.payload.taskId}:${index}`, speakerKey: track.lane,
        startSeconds: null, endSeconds: null, text: '', index
      }))
    };
    return dependencies.transcribe(settings, job, prepared.map((track) => track.audio), undefined, lease.payload.taskId);
  });
}

export function createVolunteer(dependencies: VolunteerDependencies = defaultVolunteerDependencies) {
  let controller: AbortController | null = null;
  let status: VolunteerStatus = { state: 'disabled' };
  let loop: Promise<void> | null = null;
  let restart = false;
  const getStatus = (): VolunteerStatus => ({ ...status });

  async function run(signal: AbortSignal): Promise<void> {
    let credentials: Credentials | null = null;
    let backoff = 2_000;
    let registrationKey = '';
    while (!signal.aborted) {
      try {
        const settings = await dependencies.settings();
        if (IS_DEV_C_DENOISE || !isBrowserLocalMode(settings)) {
          status = { state: 'disabled', detail: IS_DEV_C_DENOISE
            ? 'Own-task C-denoise WebGPU trial; the shared coordinator is not enabled.'
            : 'Local browser model settings are disabled.' };
          return;
        }
        if (!settings.volunteerInferenceEnabled) {
          status = { state: 'disabled', detail: 'Swarm participation is off; local models remain available for your own tasks.' };
          return;
        }
        const readiness = await dependencies.ready();
        const operations: string[] = [];
        if (readiness.transcribe) operations.push('transcribe');
        if (readiness.draft) operations.push('draft');
        if (readiness.enhancementModel) { parseEnhancementModel(readiness.enhancementModel); operations.push('enhance'); }
        if (!operations.length) {
          status = { state: 'disabled', detail: 'No verified local inference capability is available.' };
          return;
        }
        const baseUrl = normalizeL0CustomBaseUrl(settings.l0CustomBaseUrl);
        const registration = { modelBundleSchema: SCHEMA, protocolVersion: INFERENCE_RELEASE.protocolVersion, modelRelease: INFERENCE_RELEASE.id,
          maxLeaseBytes: LEASE_RESPONSE_BYTES, operations,
          ...(readiness.enhancementModel ? { enhancementModel: readiness.enhancementModel } : {}) };
        const nextKey = JSON.stringify([baseUrl, registration]);
        if (nextKey !== registrationKey) { credentials = null; registrationKey = nextKey; }
        if (!credentials) {
          status = { state: 'connecting' };
          const response = await request(dependencies, baseUrl, '/v1/workers/register', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(registration)
          }, signal);
          if (!response.ok) throw new Error(`Worker registration failed: HTTP ${response.status}`);
          credentials = parseCredentials(await readBoundedSwarmJson(response, signal));
        }
        if (signal.aborted) return;
        const response = await request(dependencies, baseUrl, '/v1/workers/lease', {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(credentials)
        }, signal);
        if (response.status === 401 || response.status === 403 || response.status === 404) {
          credentials = null;
          status = { state: 'connecting', detail: 'Re-registering with the coordinator.' };
          await dependencies.wait(2_000, signal);
          continue;
        }
        if (response.status === 204) {
          status = { state: 'connected' };
          backoff = 2_000;
          await dependencies.wait(IDLE_POLL_MS, signal);
          continue;
        }
        if (!response.ok) throw new Error(`Worker lease failed: HTTP ${response.status}`);
        const lease = parseLease(JSON.parse(await (await readBoundedSwarmBlob(response, LEASE_RESPONSE_BYTES, signal)).text()));
        status = { state: 'busy' };
        let completion: RequestInit;
        const leaseSignal = AbortSignal.any([signal, AbortSignal.timeout(8 * 60_000)]);
        try {
          const result = await executeLease(lease, settings, readiness, credentials, dependencies, baseUrl, leaseSignal);
          leaseSignal.throwIfAborted();
          if (lease.operation === 'enhance') {
            const batch = result as EnhancedAudioBatch;
            if (batch.provider !== 'browser-local' || batch.model !== lease.payload.model.id || batch.modelSha256 !== lease.payload.model.sha256 ||
                batch.tracks.length !== 2) throw new Error('The GPU worker returned a different enhancement model or incomplete pair.');
            const form = new FormData();
            for (let index = 0; index < batch.tracks.length; index++) {
              const track = batch.tracks[index], source = lease.payload.tracks[index];
              parseEnhancementTrackMetadata(track.metadata, source);
              const bytes = track.bytes.byteOffset === 0 && track.bytes.byteLength === track.bytes.buffer.byteLength
                ? track.bytes.buffer : track.bytes.slice().buffer;
              await verifyEnhancementWav(bytes, source, track.metadata.wavSha256, true);
              leaseSignal.throwIfAborted();
              form.append(source.fieldName, new Blob([bytes], { type: 'audio/wav' }), `enhanced-${index + 1}.wav`);
            }
            form.append('payload', JSON.stringify({ ...credentials, leaseToken: lease.leaseToken,
              result: { model: batch.model, modelSha256: batch.modelSha256, tracks: batch.tracks.map(track => track.metadata) } }));
            completion = { headers: { Authorization: `Bearer ${lease.leaseToken}` }, body: form };
          } else {
            completion = { headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ ...credentials, leaseToken: lease.leaseToken, result }) };
          }
        } catch (error) {
          completion = { headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ...credentials, leaseToken: lease.leaseToken,
              error: (error instanceof Error ? error.message : String(error)).slice(0, 1000) }) };
        }
        if (signal.aborted) return;
        const completed = await request(dependencies, baseUrl, `/v1/jobs/${encodeURIComponent(lease.jobId)}/complete`, {
          method: 'POST', ...completion
        }, signal);
        await completed.body?.cancel();
        if (completed.status === 401 || completed.status === 403 || completed.status === 404) {
          credentials = null;
          throw new Error(`Worker lease expired: HTTP ${completed.status}`);
        }
        if (!completed.ok) throw new Error(`Worker completion failed: HTTP ${completed.status}`);
        status = { state: 'connected' };
        backoff = 2_000;
        await dependencies.wait(1_000, signal);
      } catch (error) {
        if (signal.aborted) return;
        status = { state: 'error', detail: error instanceof Error ? error.message : String(error) };
        await dependencies.wait(backoff, signal);
        backoff = Math.min(backoff * 2, MAX_BACKOFF_MS);
      }
    }
  }

  function start(): VolunteerStatus {
    if (controller) return getStatus();
    if (loop) { restart = true; return { state: 'connecting' }; }
    controller = new AbortController();
    status = { state: 'connecting' };
    const signal = controller.signal;
    loop = run(signal).finally(() => {
      loop = null;
      controller = null;
      if (restart) { restart = false; start(); }
      else if (status.state !== 'disabled' && signal.aborted) status = { state: 'disabled' };
    });
    return getStatus();
  }
  function stop(): VolunteerStatus {
    restart = false;
    controller?.abort();
    status = { state: 'disabled' };
    return getStatus();
  }
  return { start, stop, getStatus };
}
