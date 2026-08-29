import { normalizeL0CustomBaseUrl } from './settings';
import { assertL0WavAudio, createL0Payload, type PreparedL0Track } from './l0-client';
import { buildCanonicalTaskIdentity } from './transcript';
import type {
  CapturedAudioTrack,
  ExtensionSettings,
  L0TimingResponse,
  L0TimingToken,
  TranscriptJob
} from './types';

const L0_TIMING_PATH = '/v1/transcribe';
const L0_QUEUE_PATH = '/v1/queue';
const DEFAULT_QUEUE_POLL_INTERVAL_MS = 500;

export type L0TimingQueueStatus =
  | { requestId: string; status: 'preparing' }
  | { requestId: string; status: 'queued'; position: number; queuedCount: number }
  | { requestId: string; status: 'running'; position: 0; queuedCount: number }
  | { requestId: string; status: 'completed'; position: 0; queuedCount: number };

export interface L0TimingRequestCallbacks {
  onQueueStatus?: (status: L0TimingQueueStatus) => void;
  pollIntervalMs?: number;
  waitForPoll?: (signal: AbortSignal) => Promise<void>;
}

function normalizedLane(value: string | undefined): string {
  return typeof value === 'string' ? value.trim().toLocaleLowerCase() : '';
}

export function prepareL0TimingTracks(
  job: TranscriptJob,
  audioTracks: CapturedAudioTrack[]
): PreparedL0Track[] {
  const transcriptLaneByKey = new Map<string, string>();
  for (const row of job.rows) {
    const key = normalizedLane(row.speakerKey);
    if (key && !transcriptLaneByKey.has(key)) {
      transcriptLaneByKey.set(key, row.speakerKey.trim());
    }
  }
  if (transcriptLaneByKey.size === 0) {
    throw new Error('L0 transcription requires at least one transcript speaker lane.');
  }

  const candidates = audioTracks
    .map((audio) => {
      const identities = [audio.speakerKey, audio.trackLabel]
        .map((identity) => identity?.trim() || '')
        .filter((identity, index, all) => Boolean(identity) && all.indexOf(identity) === index);
      const matchedLane = identities
        .map((identity) => transcriptLaneByKey.get(normalizedLane(identity)))
        .find((lane): lane is string => Boolean(lane));
      return { audio, identities, matchedLane };
    })
    .filter((candidate) => candidate.identities.length > 0)
    .sort((left, right) => Number(Boolean(right.matchedLane)) - Number(Boolean(left.matchedLane)));

  const selected: PreparedL0Track[] = [];
  const selectedLanes = new Set<string>();
  for (const candidate of candidates) {
    const laneChoices = candidate.matchedLane
      ? [candidate.matchedLane, ...candidate.identities]
      : candidate.identities;
    const lane = laneChoices.find((choice) => !selectedLanes.has(normalizedLane(choice)));
    if (!lane) {
      continue;
    }
    selectedLanes.add(normalizedLane(lane));
    selected.push({
      lane,
      fieldName: selected.length === 0 ? 'audio:1' : 'audio:2',
      audio: candidate.audio
    });
    if (selected.length === 2) {
      return selected;
    }
  }
  throw new Error(`L0 transcription requires exactly two distinct captured speaker tracks; found ${selected.length}.`);
}


function isTimingToken(value: unknown): value is L0TimingToken {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return false;
  }
  if (!('id' in value) || !('text' in value) || !('startSeconds' in value) || !('endSeconds' in value)) {
    return false;
  }
  return (
    typeof value.id === 'string' &&
    Boolean(value.id) &&
    typeof value.text === 'string' &&
    Boolean(value.text) &&
    typeof value.startSeconds === 'number' &&
    Number.isFinite(value.startSeconds) &&
    value.startSeconds >= 0 &&
    typeof value.endSeconds === 'number' &&
    Number.isFinite(value.endSeconds) &&
    value.endSeconds > value.startSeconds
  );
}

export function parseL0TimingResponse(payload: unknown, expectedTaskId?: string): L0TimingResponse {
  if (
    !payload ||
    typeof payload !== 'object' ||
    Array.isArray(payload) ||
    !('taskId' in payload) ||
    !('tracks' in payload) ||
    !('summary' in payload) ||
    !('models' in payload) ||
    typeof payload.taskId !== 'string' ||
    !payload.taskId ||
    (expectedTaskId !== undefined && payload.taskId !== expectedTaskId) ||
    !Array.isArray(payload.tracks) ||
    payload.tracks.length !== 2 ||
    !payload.tracks.every(
      (track) =>
        track !== null &&
        typeof track === 'object' &&
        !Array.isArray(track) &&
        'lane' in track &&
        typeof track.lane === 'string' &&
        Boolean(track.lane) &&
        'tokens' in track &&
        Array.isArray(track.tokens) &&
        track.tokens.every(isTimingToken)
    ) ||
    !payload.summary ||
    typeof payload.summary !== 'object' ||
    Array.isArray(payload.summary) ||
    !payload.models ||
    typeof payload.models !== 'object' ||
    Array.isArray(payload.models)
  ) {
    throw new Error('L0 transcription endpoint returned an invalid timing response.');
  }
  return payload as unknown as L0TimingResponse;
}

async function parseResponsePayload(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) {
    return null;
  }
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function responseError(status: number, payload: unknown): string {
  if (payload && typeof payload === 'object' && !Array.isArray(payload)) {
    if ('error' in payload && typeof payload.error === 'string') {
      return payload.error;
    }
    if ('detail' in payload && typeof payload.detail === 'string') {
      return payload.detail;
    }
  }
  return `HTTP ${status}`;
}

export function getL0TimingEndpoint(settings: ExtensionSettings): string {
  return `${normalizeL0CustomBaseUrl(settings.l0CustomBaseUrl)}${L0_TIMING_PATH}`;
}

export function getL0QueueEndpoint(settings: ExtensionSettings, requestId: string): string {
  return `${normalizeL0CustomBaseUrl(settings.l0CustomBaseUrl)}${L0_QUEUE_PATH}/${encodeURIComponent(requestId)}`;
}

function createRequestId(): string {
  if (typeof crypto?.randomUUID !== 'function') {
    throw new Error('L0 transcription requires crypto.randomUUID support.');
  }
  return crypto.randomUUID();
}

function emitQueueStatus(callbacks: L0TimingRequestCallbacks, status: L0TimingQueueStatus): void {
  try {
    callbacks.onQueueStatus?.(status);
  } catch {
    // Queue presentation is observational and must never affect transcription.
  }
}

function parseQueueStatus(payload: unknown, requestId: string): L0TimingQueueStatus | null {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload) || !('requestId' in payload) || !('status' in payload)) {
    return null;
  }
  if (payload.requestId !== requestId || typeof payload.status !== 'string') {
    return null;
  }
  if (payload.status === 'queued') {
    if (
      !('position' in payload) ||
      !('queuedCount' in payload) ||
      typeof payload.position !== 'number' ||
      !Number.isInteger(payload.position) ||
      payload.position < 1 ||
      typeof payload.queuedCount !== 'number' ||
      !Number.isInteger(payload.queuedCount) ||
      payload.queuedCount < 0
    ) {
      return null;
    }
    return { requestId, status: 'queued', position: payload.position, queuedCount: payload.queuedCount };
  }
  if (payload.status === 'running' || payload.status === 'completed') {
    if (
      !('position' in payload) ||
      payload.position !== 0 ||
      !('queuedCount' in payload) ||
      typeof payload.queuedCount !== 'number' ||
      !Number.isInteger(payload.queuedCount) ||
      payload.queuedCount < 0
    ) {
      return null;
    }
    return { requestId, status: payload.status, position: 0, queuedCount: payload.queuedCount };
  }
  return null;
}

function waitForNextPoll(delayMs: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted || delayMs <= 0) {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    const timeoutId = setTimeout(resolve, delayMs);
    signal.addEventListener(
      'abort',
      () => {
        clearTimeout(timeoutId);
        resolve();
      },
      { once: true }
    );
  });
}

async function pollQueueStatus(
  settings: ExtensionSettings,
  requestId: string,
  callbacks: L0TimingRequestCallbacks,
  signal: AbortSignal
): Promise<void> {
  const endpoint = getL0QueueEndpoint(settings, requestId);
  while (!signal.aborted) {
    try {
      const response = await fetch(endpoint, {
        method: 'GET',
        headers: { Accept: 'application/json' },
        signal
      });
      if (signal.aborted) {
        return;
      }
      if (response.status === 404) {
        emitQueueStatus(callbacks, { requestId, status: 'preparing' });
      } else if (response.ok) {
        const status = parseQueueStatus(await parseResponsePayload(response), requestId);
        if (status) {
          emitQueueStatus(callbacks, status);
        }
      }
    } catch {
      // Polling is best-effort; POST success/failure remains authoritative.
    }
    if (callbacks.waitForPoll) {
      await callbacks.waitForPoll(signal);
    } else {
      await waitForNextPoll(callbacks.pollIntervalMs ?? DEFAULT_QUEUE_POLL_INTERVAL_MS, signal);
    }
  }
}

export async function generateL0Timing(
  settings: ExtensionSettings,
  job: TranscriptJob,
  audioTracks: CapturedAudioTrack[],
  callbacks: L0TimingRequestCallbacks = {}
): Promise<L0TimingResponse> {
  const tracks = prepareL0TimingTracks(job, audioTracks);
  await Promise.all(tracks.map(assertL0WavAudio));
  const taskId = buildCanonicalTaskIdentity(job);

  const body = new FormData();
  body.set('payload', JSON.stringify({ ...createL0Payload(job, tracks), taskId }));
  for (const track of tracks) {
    body.append(track.fieldName, track.audio.blob, `${track.fieldName.replace(':', '-')}.wav`);
  }

  const requestId = createRequestId();
  const queuePolling = new AbortController();
  const endpoint = getL0TimingEndpoint(settings);
  const responsePromise = fetch(endpoint, {
    method: 'POST',
    headers: {
      Accept: 'application/json',
      'X-Babel-Local-Engine': '1',
      'X-Babel-Request-Id': requestId
    },
    body
  });
  const pollingPromise = pollQueueStatus(settings, requestId, callbacks, queuePolling.signal);

  let response: Response;
  try {
    response = await responsePromise;
  } finally {
    queuePolling.abort();
    await pollingPromise;
  }
  const responsePayload = await parseResponsePayload(response);
  if (!response.ok) {
    throw new Error(`L0 transcription failed: ${responseError(response.status, responsePayload)}`);
  }
  return parseL0TimingResponse(responsePayload, taskId);
}
