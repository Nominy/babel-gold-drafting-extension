import { assertL0WavAudio, type PreparedL0Track } from '../core/l0-client';
import { LOCAL_MODEL_BASE_URL, PUBLIC_L0_BASE_URL, normalizeSettings } from '../core/settings';
import { getCachedLocalModelFile } from '../core/local-model-bundle';
import type { CapturedAudioTrack, ExtensionSettings, L0DraftResponse, L0TimingResponse, TranscriptJob, TranscriptRow } from '../core/types';
import type { VolunteerStatus } from '../core/volunteer-protocol';
import { generateLocalL0Draft, generateLocalL0SegmentDraft, generateLocalL0Timing } from '../core/local-model-runtime';

const SCHEMA = 'babel-browser-model-bundle-v2';
const CONTROL_REQUEST_TIMEOUT_MS = 45_000;
const AUDIO_REQUEST_TIMEOUT_MS = 5 * 60_000;
const IDLE_POLL_MS = 3_000;
const MAX_BACKOFF_MS = 30_000;

type Credentials = { workerId: string; token: string };
type Lease = {
  jobId: string;
  leaseToken: string;
  operation: 'draft' | 'transcribe';
  payload: { taskId: string; tracks: Array<{ lane: string; fieldName: string }>; options?: Record<string, unknown> };
  audio: Array<{ fieldName: string; url: string }>;
};

export interface VolunteerDependencies {
  fetch: (url: string, init: RequestInit) => Promise<Response>;
  settings: () => Promise<ExtensionSettings>;
  ready: () => Promise<boolean>;
  runExclusive: <T>(action: () => Promise<T>) => Promise<T>;
  draft: typeof generateLocalL0Draft;
  segment: typeof generateLocalL0SegmentDraft;
  transcribe: typeof generateLocalL0Timing;
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
  ready: async () => Boolean(await getCachedLocalModelFile('asr/v3_ctc.yaml', LOCAL_MODEL_BASE_URL)),
  runExclusive: (action) => action(),
  draft: generateLocalL0Draft,
  segment: generateLocalL0SegmentDraft,
  transcribe: generateLocalL0Timing,
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
      (value.operation !== 'draft' && value.operation !== 'transcribe') ||
      !record(value.payload) || typeof value.payload.taskId !== 'string' || !value.payload.taskId ||
      !Array.isArray(value.payload.tracks) || value.payload.tracks.length !== 2 ||
      !Array.isArray(value.audio) || value.audio.length !== 2) throw new Error('Invalid volunteer lease.');
  const tracks = value.payload.tracks;
  const audio = value.audio;
  if (!tracks.every((track) => record(track) && typeof track.lane === 'string' && !!track.lane &&
      (track.fieldName === 'audio:1' || track.fieldName === 'audio:2')) ||
      new Set(tracks.map((track) => track.lane)).size !== 2 ||
      new Set(tracks.map((track) => track.fieldName)).size !== 2 ||
      !audio.every((entry) => record(entry) && typeof entry.fieldName === 'string' &&
        typeof entry.url === 'string' && entry.url === `/v1/jobs/${encodeURIComponent(value.jobId as string)}/audio/${encodeURIComponent(entry.fieldName as string)}`) ||
      new Set(audio.map((entry) => entry.fieldName)).size !== 2 ||
      !tracks.every((track) => audio.some((entry) => entry.fieldName === track.fieldName)) ||
      (value.payload.options !== undefined && !record(value.payload.options))) throw new Error('Invalid volunteer lease tracks or audio URLs.');
  return value as Lease;
}

function preserveRows(lease: Lease): TranscriptRow[] | null {
  const options = lease.payload.options;
  if (options === undefined) return null;
  if (Object.keys(options).some((key) => key !== 'preserveRows' && key !== 'preprocessing') ||
      (options.preprocessing !== undefined && options.preprocessing !== 'raw')) {
    throw new Error('Unsupported draft options for browser inference.');
  }
  if (options.preserveRows === undefined) return null;
  if (!Array.isArray(options.preserveRows) || !options.preserveRows.length ||
      !options.preserveRows.every((row) => record(row) && typeof row.rowId === 'string' && !!row.rowId &&
        typeof row.speakerKey === 'string' && lease.payload.tracks.some((track) => track.lane === row.speakerKey) &&
        typeof row.startSeconds === 'number' && Number.isFinite(row.startSeconds) && row.startSeconds >= 0 &&
        typeof row.endSeconds === 'number' && Number.isFinite(row.endSeconds) && row.endSeconds > row.startSeconds &&
        typeof row.text === 'string' && Number.isSafeInteger(row.index) && (row.index as number) >= 0)) {
    throw new Error('Invalid preserveRows for browser inference.');
  }
  return options.preserveRows as TranscriptRow[];
}

async function request(dependencies: VolunteerDependencies, path: string, init: RequestInit, signal: AbortSignal): Promise<Response> {
  return dependencies.fetch(`${PUBLIC_L0_BASE_URL}${path}`, {
    ...init,
    redirect: 'error',
    signal: AbortSignal.any([
      signal,
      AbortSignal.timeout(path.includes('/audio/') ? AUDIO_REQUEST_TIMEOUT_MS : CONTROL_REQUEST_TIMEOUT_MS)
    ])
  });
}

async function executeLease(lease: Lease, settings: ExtensionSettings, dependencies: VolunteerDependencies, signal: AbortSignal): Promise<L0DraftResponse | L0TimingResponse> {
  const prepared = await Promise.all(lease.payload.tracks.map(async (track): Promise<PreparedL0Track> => {
    const entry = lease.audio.find((audio) => audio.fieldName === track.fieldName)!;
    const response = await request(dependencies, entry.url, { headers: { Authorization: `Bearer ${lease.leaseToken}` } }, signal);
    if (!response.ok) throw new Error(`Leased audio fetch failed: HTTP ${response.status}`);
    const blob = await response.blob();
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
    if (lease.operation === 'transcribe') {
      if (lease.payload.options !== undefined) throw new Error('Unsupported transcription options for browser inference.');
      const timing = await dependencies.transcribe(settings, job, prepared.map((track) => track.audio));
      return {
        ...timing, taskId: lease.payload.taskId,
        tracks: timing.tracks.map((track) => ({
          ...track, tokens: track.tokens.map((token, index) => ({
            ...token, id: `${lease.payload.taskId}:${track.lane}:${index}`
          }))
        })),
        summary: { ...timing.summary, taskId: lease.payload.taskId }
      };
    }
    const rows = preserveRows(lease);
    if (!rows) return dependencies.draft(settings, job, prepared.map((track) => track.audio));
    const output: L0DraftResponse['rows'] = [];
    for (const row of rows) {
      const text = (await dependencies.segment(settings, lease.payload.taskId, row, prepared)).trim();
      if (!text) throw new Error(`Local segment returned no text for preserved row ${row.rowId}.`);
      output.push({ id: row.rowId, lane: row.speakerKey, startSeconds: row.startSeconds!, endSeconds: row.endSeconds!, text });
    }
    return {
      rows: output,
      summary: { taskId: lease.payload.taskId, trackCount: 2, rowCount: output.length, provider: 'browser-local' },
      models: { provider: 'browser-local' }
    };
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
    while (!signal.aborted) {
      try {
        const settings = await dependencies.settings();
        if (!settings.localModelsEnabled) {
          status = { state: 'disabled', detail: 'Saved local model settings are disabled.' };
          return;
        }
        if (!(await dependencies.ready())) {
          status = { state: 'disabled', detail: 'The verified model bundle is unavailable in the worker.' };
          return;
        }
        if (!credentials) {
          status = { state: 'connecting' };
          const response = await request(dependencies, '/v1/workers/register', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ modelBundleSchema: SCHEMA })
          }, signal);
          if (!response.ok) throw new Error(`Worker registration failed: HTTP ${response.status}`);
          credentials = parseCredentials(await response.json());
        }
        if (signal.aborted) return;
        const response = await request(dependencies, '/v1/workers/lease', {
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
        const lease = parseLease(await response.json());
        status = { state: 'busy' };
        let completion: { result: L0DraftResponse | L0TimingResponse } | { error: string };
        try {
          const result = await executeLease(lease, settings, dependencies, signal);
          completion = { result };
        } catch (error) {
          completion = { error: error instanceof Error ? error.message : String(error) };
        }
        if (signal.aborted) return;
        const completed = await request(dependencies, `/v1/jobs/${encodeURIComponent(lease.jobId)}/complete`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ...credentials, leaseToken: lease.leaseToken, ...completion })
        }, signal);
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
