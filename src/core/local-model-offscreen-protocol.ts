import type {
  ExtensionSettings,
  L0DraftResponse,
  L0TimingResponse,
  TranscriptJob,
  TranscriptRow
} from './types';

export const LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE = 'babel-gold-drafting:local-model-offscreen';
export const LOCAL_MODEL_OFFSCREEN_VERSION = 3 as const;
export const LOCAL_MODEL_AUDIO_CHUNK_BYTES = 512 * 1024;
export const LOCAL_MODEL_MAX_BUFFERED_AUDIO_BYTES = 512 * 1024 * 1024;
export const LOCAL_MODEL_AUDIO_TRANSFER_STALE_MS = 10 * 60 * 1000;

export type LocalModelOperation = 'upload' | 'timing' | 'draft' | 'segment';
export type LocalModelMessageTarget = 'background' | 'offscreen';

export interface WireCapturedAudioTrack {
  trackId: string;
  speakerKey?: string;
  trackLabel?: string;
  source: string;
  audioTransferId: string;
  mimeType: string;
}

export interface WirePreparedL0Track {
  lane: string;
  fieldName: 'audio:1' | 'audio:2';
  audio: WireCapturedAudioTrack;
}

const BASE64_STRING_CHUNK_BYTES = 0x8000;
const BASE64_PATTERN = /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/;

function base64DecodedLength(base64: string): number {
  if (base64.length === 0) return 0;
  const padding = base64.endsWith('==') ? 2 : base64.endsWith('=') ? 1 : 0;
  return (base64.length / 4) * 3 - padding;
}

function isBoundedBase64Chunk(value: unknown): value is string {
  return (
    typeof value === 'string' &&
    value.length % 4 === 0 &&
    BASE64_PATTERN.test(value) &&
    base64DecodedLength(value) <= LOCAL_MODEL_AUDIO_CHUNK_BYTES
  );
}

export function encodeAudioChunk(bytes: Uint8Array): string {
  if (!(bytes instanceof Uint8Array) || bytes.byteLength > LOCAL_MODEL_AUDIO_CHUNK_BYTES) {
    throw new TypeError(`Audio chunks must be Uint8Array values no larger than ${LOCAL_MODEL_AUDIO_CHUNK_BYTES} bytes.`);
  }
  const nativeEncoder = (bytes as Uint8Array & { toBase64?: () => string }).toBase64;
  if (typeof nativeEncoder === 'function') return nativeEncoder.call(bytes);
  if (typeof globalThis.btoa !== 'function') {
    throw new Error('This environment cannot encode base64 audio.');
  }
  const chunks: string[] = [];
  for (let offset = 0; offset < bytes.length; offset += BASE64_STRING_CHUNK_BYTES) {
    chunks.push(String.fromCharCode(...bytes.subarray(offset, offset + BASE64_STRING_CHUNK_BYTES)));
  }
  return globalThis.btoa(chunks.join(''));
}

export function decodeAudioChunk(base64: string): Uint8Array<ArrayBuffer> {
  if (!isBoundedBase64Chunk(base64)) {
    throw new TypeError(`Audio chunks must be valid base64 values no larger than ${LOCAL_MODEL_AUDIO_CHUNK_BYTES} bytes.`);
  }
  const nativeDecoder = (
    Uint8Array as typeof Uint8Array & { fromBase64?: (value: string) => Uint8Array }
  ).fromBase64;
  if (typeof nativeDecoder === 'function') return nativeDecoder(base64) as Uint8Array<ArrayBuffer>;
  if (typeof globalThis.atob !== 'function') {
    throw new Error('This environment cannot decode base64 audio.');
  }
  const binary = globalThis.atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

interface LocalModelRequestBase {
  type: typeof LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE;
  version: typeof LOCAL_MODEL_OFFSCREEN_VERSION;
  target: LocalModelMessageTarget;
  requestId: string;
  operation: LocalModelOperation;
}

export interface LocalModelUploadRequest extends LocalModelRequestBase {
  operation: 'upload';
  transferId: string;
  chunkIndex: number;
  chunkCount: number;
  totalBytes: number;
  mimeType: string;
  dataBase64: string;
}

export interface LocalModelTimingRequest extends LocalModelRequestBase {
  operation: 'timing';
  settings: ExtensionSettings;
  job: TranscriptJob;
  audioTracks: WireCapturedAudioTrack[];
}

export interface LocalModelDraftRequest extends LocalModelRequestBase {
  operation: 'draft';
  settings: ExtensionSettings;
  job: TranscriptJob;
  audioTracks: WireCapturedAudioTrack[];
}

export interface LocalModelSegmentRequest extends LocalModelRequestBase {
  operation: 'segment';
  settings: ExtensionSettings;
  taskId: string;
  row: TranscriptRow;
  tracks: WirePreparedL0Track[];
}

export type LocalModelOffscreenRequest =
  | LocalModelUploadRequest
  | LocalModelTimingRequest
  | LocalModelDraftRequest
  | LocalModelSegmentRequest;

interface LocalModelResponseBase {
  type: typeof LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE;
  version: typeof LOCAL_MODEL_OFFSCREEN_VERSION;
  requestId: string;
  operation: LocalModelOperation;
}

export interface LocalModelUploadResult {
  transferId: string;
  nextChunkIndex: number;
  complete: boolean;
}

export interface LocalModelUploadSuccessResponse extends LocalModelResponseBase {
  ok: true;
  operation: 'upload';
  result: LocalModelUploadResult;
}

export interface LocalModelTimingSuccessResponse extends LocalModelResponseBase {
  ok: true;
  operation: 'timing';
  result: L0TimingResponse;
}

export interface LocalModelDraftSuccessResponse extends LocalModelResponseBase {
  ok: true;
  operation: 'draft';
  result: L0DraftResponse;
}

export interface LocalModelSegmentSuccessResponse extends LocalModelResponseBase {
  ok: true;
  operation: 'segment';
  result: string;
}

export type LocalModelSuccessResponse =
  | LocalModelUploadSuccessResponse
  | LocalModelTimingSuccessResponse
  | LocalModelDraftSuccessResponse
  | LocalModelSegmentSuccessResponse;

export type LocalModelErrorCode = 'invalid-request' | 'offscreen-unavailable' | 'inference-failed';

export interface LocalModelFailureResponse extends LocalModelResponseBase {
  ok: false;
  error: {
    code: LocalModelErrorCode;
    name: string;
    message: string;
  };
}

export type LocalModelOffscreenResponse = LocalModelSuccessResponse | LocalModelFailureResponse;

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function isFiniteNumberOrNull(value: unknown): value is number | null {
  return value === null || (typeof value === 'number' && Number.isFinite(value));
}

function isTranscriptRow(value: unknown): value is TranscriptRow {
  if (!isRecord(value)) return false;
  return (
    typeof value.rowId === 'string' &&
    typeof value.speakerKey === 'string' &&
    isFiniteNumberOrNull(value.startSeconds) &&
    isFiniteNumberOrNull(value.endSeconds) &&
    typeof value.text === 'string' &&
    Number.isInteger(value.index)
  );
}

function isTranscriptJob(value: unknown): value is TranscriptJob {
  return (
    isRecord(value) &&
    typeof value.jobId === 'string' &&
    Array.isArray(value.rows) &&
    value.rows.every(isTranscriptRow)
  );
}

function isWireCapturedAudioTrack(value: unknown): value is WireCapturedAudioTrack {
  return (
    isRecord(value) &&
    !('blob' in value) &&
    !('audioDataUrl' in value) &&
    typeof value.trackId === 'string' &&
    (value.speakerKey === undefined || typeof value.speakerKey === 'string') &&
    (value.trackLabel === undefined || typeof value.trackLabel === 'string') &&
    typeof value.source === 'string' &&
    typeof value.audioTransferId === 'string' &&
    value.audioTransferId.length > 0 &&
    typeof value.mimeType === 'string'
  );
}

function isWirePreparedTrack(value: unknown): value is WirePreparedL0Track {
  return (
    isRecord(value) &&
    typeof value.lane === 'string' &&
    (value.fieldName === 'audio:1' || value.fieldName === 'audio:2') &&
    isWireCapturedAudioTrack(value.audio)
  );
}

function isSettings(value: unknown): value is ExtensionSettings {
  return isRecord(value) && typeof value.localModelsEnabled === 'boolean';
}

function hasValidEnvelope(value: Record<string, unknown>): boolean {
  return (
    value.type === LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE &&
    value.version === LOCAL_MODEL_OFFSCREEN_VERSION &&
    (value.target === 'background' || value.target === 'offscreen') &&
    typeof value.requestId === 'string' &&
    value.requestId.length > 0 &&
    (value.operation === 'upload' ||
      value.operation === 'timing' ||
      value.operation === 'draft' ||
      value.operation === 'segment')
  );
}

function isUploadRequest(value: Record<string, unknown>): value is Record<string, unknown> & LocalModelUploadRequest {
  if (
    typeof value.transferId !== 'string' ||
    value.transferId.length === 0 ||
    !Number.isInteger(value.chunkIndex) ||
    !Number.isInteger(value.chunkCount) ||
    !Number.isSafeInteger(value.totalBytes) ||
    typeof value.mimeType !== 'string' ||
    !isBoundedBase64Chunk(value.dataBase64)
  ) {
    return false;
  }
  const chunkIndex = value.chunkIndex as number;
  const chunkCount = value.chunkCount as number;
  const totalBytes = value.totalBytes as number;
  const decodedBytes = base64DecodedLength(value.dataBase64);
  return (
    chunkIndex >= 0 &&
    chunkCount > 0 &&
    chunkIndex < chunkCount &&
    totalBytes >= 0 &&
    (totalBytes === 0
      ? chunkCount === 1 && chunkIndex === 0 && decodedBytes === 0
      : chunkCount <= totalBytes && decodedBytes > 0)
  );
}

export function isLocalModelOffscreenRequest(
  value: unknown,
  target?: LocalModelMessageTarget
): value is LocalModelOffscreenRequest {
  if (!isRecord(value) || !hasValidEnvelope(value) || (target !== undefined && value.target !== target)) {
    return false;
  }
  if (value.operation === 'upload') return isUploadRequest(value);
  if (!isSettings(value.settings)) return false;
  if (value.operation === 'segment') {
    return (
      typeof value.taskId === 'string' &&
      value.taskId.length > 0 &&
      isTranscriptRow(value.row) &&
      Array.isArray(value.tracks) &&
      value.tracks.every(isWirePreparedTrack)
    );
  }
  return (
    isTranscriptJob(value.job) &&
    Array.isArray(value.audioTracks) &&
    value.audioTracks.every(isWireCapturedAudioTrack)
  );
}

function isTimingResult(value: unknown): value is L0TimingResponse {
  if (
    !isRecord(value) ||
    typeof value.taskId !== 'string' ||
    !Array.isArray(value.tracks) ||
    !isRecord(value.summary) ||
    !isRecord(value.models)
  ) {
    return false;
  }
  return value.tracks.every(
    (track) =>
      isRecord(track) &&
      typeof track.lane === 'string' &&
      Array.isArray(track.tokens) &&
      track.tokens.every(
        (token) =>
          isRecord(token) &&
          typeof token.id === 'string' &&
          typeof token.text === 'string' &&
          typeof token.startSeconds === 'number' &&
          Number.isFinite(token.startSeconds) &&
          typeof token.endSeconds === 'number' &&
          Number.isFinite(token.endSeconds)
      )
  );
}

function isDraftResult(value: unknown): value is L0DraftResponse {
  if (
    !isRecord(value) ||
    !Array.isArray(value.rows) ||
    !isRecord(value.summary) ||
    !isRecord(value.models)
  ) {
    return false;
  }
  return value.rows.every(
    (row) =>
      isRecord(row) &&
      typeof row.id === 'string' &&
      typeof row.lane === 'string' &&
      typeof row.startSeconds === 'number' &&
      Number.isFinite(row.startSeconds) &&
      typeof row.endSeconds === 'number' &&
      Number.isFinite(row.endSeconds) &&
      typeof row.text === 'string'
  );
}

export function isLocalModelOffscreenResponse(
  value: unknown,
  request: LocalModelOffscreenRequest
): value is LocalModelOffscreenResponse {
  if (!isRecord(value)) return false;
  if (
    value.type !== LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE ||
    value.version !== LOCAL_MODEL_OFFSCREEN_VERSION ||
    value.requestId !== request.requestId ||
    value.operation !== request.operation ||
    typeof value.ok !== 'boolean'
  ) {
    return false;
  }
  if (!value.ok) {
    return (
      isRecord(value.error) &&
      (value.error.code === 'invalid-request' ||
        value.error.code === 'offscreen-unavailable' ||
        value.error.code === 'inference-failed') &&
      typeof value.error.name === 'string' &&
      typeof value.error.message === 'string' &&
      value.error.message.length > 0
    );
  }
  if (request.operation === 'upload') {
    return (
      isRecord(value.result) &&
      value.result.transferId === request.transferId &&
      value.result.nextChunkIndex === request.chunkIndex + 1 &&
      value.result.complete === (request.chunkIndex === request.chunkCount - 1)
    );
  }
  if (request.operation === 'timing') return isTimingResult(value.result);
  if (request.operation === 'draft') return isDraftResult(value.result);
  return typeof value.result === 'string';
}

export function toOffscreenRequest(request: LocalModelOffscreenRequest): LocalModelOffscreenRequest {
  return { ...request, target: 'offscreen' };
}

export function createLocalModelFailure(
  request: Pick<LocalModelOffscreenRequest, 'requestId' | 'operation'>,
  code: LocalModelErrorCode,
  error: unknown
): LocalModelFailureResponse {
  const source = error instanceof Error ? error : new Error(String(error));
  return {
    type: LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE,
    version: LOCAL_MODEL_OFFSCREEN_VERSION,
    requestId: request.requestId,
    operation: request.operation,
    ok: false,
    error: {
      code,
      name: source.name || 'Error',
      message: source.message || String(error)
    }
  };
}
