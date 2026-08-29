import { normalizeL0CustomBaseUrl } from './settings';
import type {
  CapturedAudioTrack,
  ExtensionSettings,
  L0DraftPayload,
  L0DraftResponse,
  L0DraftRow,
  TranscriptJob
} from './types';

const L0_CUSTOM_PATH = '/v1/draft';
const WAV_HEADER_BYTES = 12;

export type PreparedL0Track = {
  lane: string;
  fieldName: 'audio:1' | 'audio:2';
  audio: CapturedAudioTrack;
};

function normalizedLane(value: string | undefined): string {
  return typeof value === 'string' ? value.trim().toLocaleLowerCase() : '';
}

function parseErrorMessage(status: number, payload: unknown): string {
  if (payload && typeof payload === 'object') {
    if ('error' in payload && typeof payload.error === 'string') {
      return payload.error;
    }
    if ('detail' in payload) {
      const detail = payload.detail;
      return typeof detail === 'string' ? detail : JSON.stringify(detail);
    }
  }
  if (typeof payload === 'string' && payload.trim()) {
    return `HTTP ${status}: ${payload.slice(0, 240)}`;
  }
  return `HTTP ${status}`;
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

function isCanonicalRow(value: unknown): value is L0DraftRow {
  return Boolean(
    value &&
      typeof value === 'object' &&
      'id' in value &&
      typeof value.id === 'string' &&
      value.id &&
      'lane' in value &&
      typeof value.lane === 'string' &&
      value.lane &&
      'startSeconds' in value &&
      typeof value.startSeconds === 'number' &&
      Number.isFinite(value.startSeconds) &&
      'endSeconds' in value &&
      typeof value.endSeconds === 'number' &&
      Number.isFinite(value.endSeconds) &&
      value.endSeconds > value.startSeconds &&
      'text' in value &&
      typeof value.text === 'string' &&
      value.text.trim()
  );
}

function parseL0DraftResponse(payload: unknown): L0DraftResponse {
  if (
    !payload ||
    typeof payload !== 'object' ||
    !('rows' in payload) ||
    !Array.isArray(payload.rows) ||
    payload.rows.length === 0 ||
    !payload.rows.every(isCanonicalRow) ||
    !('summary' in payload) ||
    !payload.summary ||
    typeof payload.summary !== 'object' ||
    !('models' in payload) ||
    !payload.models ||
    typeof payload.models !== 'object'
  ) {
    throw new Error('L0 drafting endpoint returned an invalid DraftResponse.');
  }
  return payload as L0DraftResponse;
}

export function prepareL0Tracks(job: TranscriptJob, audioTracks: CapturedAudioTrack[]): PreparedL0Track[] {
  const laneByKey = new Map<string, string>();
  for (const row of job.rows) {
    const key = normalizedLane(row.speakerKey);
    if (key && !laneByKey.has(key)) {
      laneByKey.set(key, row.speakerKey.trim());
    }
  }
  const lanes = Array.from(laneByKey.values());
  if (lanes.length !== 2) {
    throw new Error(`L0 drafting requires exactly two transcript speaker lanes; found ${lanes.length}.`);
  }

  const usedTracks = new Set<CapturedAudioTrack>();
  return lanes.map((lane, index) => {
    const key = normalizedLane(lane);
    const audio = audioTracks.find(
      (track) =>
        !usedTracks.has(track) &&
        (normalizedLane(track.speakerKey) === key || normalizedLane(track.trackLabel) === key)
    );
    if (!audio) {
      throw new Error(`L0 drafting could not capture a distinct audio track for speaker lane "${lane}".`);
    }
    usedTracks.add(audio);
    return {
      lane,
      fieldName: index === 0 ? 'audio:1' : 'audio:2',
      audio
    };
  });
}

export async function assertL0WavAudio(track: PreparedL0Track): Promise<void> {
  if (!track.audio.blob.size) {
    throw new Error(`L0 audio track for "${track.lane}" is empty.`);
  }
  const header = new Uint8Array(await track.audio.blob.slice(0, WAV_HEADER_BYTES).arrayBuffer());
  const isWav =
    header.length === WAV_HEADER_BYTES &&
    String.fromCharCode(...header.slice(0, 4)) === 'RIFF' &&
    String.fromCharCode(...header.slice(8, 12)) === 'WAVE';
  if (!isWav) {
    throw new Error(`L0 audio track for "${track.lane}" is not a WAV file.`);
  }
}
export function createL0Payload(job: TranscriptJob, tracks: PreparedL0Track[]): L0DraftPayload {
  return {
    taskId: job.jobId,
    tracks: [
      { lane: tracks[0].lane, fieldName: tracks[0].fieldName },
      { lane: tracks[1].lane, fieldName: tracks[1].fieldName }
    ]
  };
}

export async function generateL0SegmentDraft(
  settings: ExtensionSettings,
  taskId: string,
  row: TranscriptJob['rows'][number],
  tracks: PreparedL0Track[]
): Promise<string> {
  if (tracks.length !== 2 || !tracks.some((track) => track.lane === row.speakerKey)) {
    throw new Error('L0 segment drafting requires two audio tracks including the target speaker lane.');
  }
  await Promise.all(tracks.map(assertL0WavAudio));
  const body = new FormData();
  body.set('payload', JSON.stringify({
    taskId,
    tracks: tracks.map((track) => ({ lane: track.lane, fieldName: track.fieldName })),
    options: {
      preserveRows: [{
        rowId: row.rowId,
        speakerKey: row.speakerKey,
        startSeconds: row.startSeconds,
        endSeconds: row.endSeconds,
        text: '',
        index: 0
      }]
    }
  }));
  for (const track of tracks) {
    body.append(track.fieldName, track.audio.blob, `${track.fieldName.replace(':', '-')}.wav`);
  }

  const endpoint = getL0DraftEndpoint(settings);
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: {
      Accept: 'application/json',
      'X-Babel-Local-Engine': '1'
    },
    body
  });
  const responsePayload = await parseResponsePayload(response);
  if (!response.ok) {
    throw new Error(`L0 segment drafting failed: ${parseErrorMessage(response.status, responsePayload)}`);
  }
  const parsed = parseL0DraftResponse(responsePayload);
  const result = parsed.rows.find((candidate) => candidate.id === row.rowId);
  if (!result?.text.trim()) {
    throw new Error('L0 segment drafting response did not contain the target row.');
  }
  return result.text.trim();
}

export function getL0DraftEndpoint(settings: ExtensionSettings): string {
  return `${normalizeL0CustomBaseUrl(settings.l0CustomBaseUrl)}${L0_CUSTOM_PATH}`;
}

export async function generateL0Draft(
  settings: ExtensionSettings,
  job: TranscriptJob,
  audioTracks: CapturedAudioTrack[]
): Promise<L0DraftResponse> {
  const tracks = prepareL0Tracks(job, audioTracks);
  await Promise.all(tracks.map(assertL0WavAudio));
  const payload = createL0Payload(job, tracks);
  const body = new FormData();
  body.set('payload', JSON.stringify(payload));
  for (const track of tracks) {
    body.append(track.fieldName, track.audio.blob, `${track.fieldName.replace(':', '-')}.wav`);
  }

  const headers: Record<string, string> = {
    Accept: 'application/json',
    'X-Babel-Local-Engine': '1'
  };

  const endpoint = getL0DraftEndpoint(settings);
  let response: Response;
  try {
    response = await fetch(endpoint, { method: 'POST', headers, body });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Could not reach L0 drafting endpoint ${endpoint}: ${message}`);
  }
  const responsePayload = await parseResponsePayload(response);
  if (!response.ok) {
    throw new Error(`L0 drafting failed: ${parseErrorMessage(response.status, responsePayload)}`);
  }
  return parseL0DraftResponse(responsePayload);
}

