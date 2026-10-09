import { isAudioEnhancementProgress, type AudioEnhancementChunk, type AudioEnhancementProgress, type AudioEnhancementResult, type AudioEnhancementTrackMetadata } from '@nominy/babel-babel-runtime';
import type {
  ExtensionSettings,
  L0DraftResponse,
  L0TimingResponse,
  TranscriptJob,
  TranscriptRow
} from './types';

export const LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE = 'babel-gold-drafting:local-model-offscreen';
export const LOCAL_MODEL_OFFSCREEN_VERSION = 5 as const;
export const LOCAL_MODEL_AUDIO_CHUNK_BYTES = 512 * 1024;
export const LOCAL_MODEL_MAX_BUFFERED_AUDIO_BYTES = 512 * 1024 * 1024;
export const LOCAL_MODEL_AUDIO_TRANSFER_STALE_MS = 10 * 60 * 1000;

export type LocalModelOperation = 'upload' | 'timing' | 'draft' | 'segment' | 'enhanceAudio' | 'download' | 'release';
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
  taskId: string;
}

export interface LocalModelSegmentRequest extends LocalModelRequestBase {
  operation: 'segment';
  settings: ExtensionSettings;
  taskId: string;
  row: TranscriptRow;
  tracks: WirePreparedL0Track[];
}

export interface LocalModelEnhanceAudioRequest extends LocalModelRequestBase {
  operation: 'enhanceAudio';
  taskId: string;
  audioTracks: WireCapturedAudioTrack[];
}

export interface LocalModelDownloadRequest extends LocalModelRequestBase {
  operation: 'download';
  transferId: string;
  chunkIndex: number;
}

export interface LocalModelReleaseRequest extends LocalModelRequestBase {
  operation: 'release';
  transferIds: string[];
}

export type LocalModelOffscreenRequest =
  | LocalModelUploadRequest
  | LocalModelTimingRequest
  | LocalModelDraftRequest
  | LocalModelSegmentRequest
  | LocalModelEnhanceAudioRequest
  | LocalModelDownloadRequest
  | LocalModelReleaseRequest;

export interface LocalModelEnhancementProgressMessage {
  type: typeof LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE;
  version: typeof LOCAL_MODEL_OFFSCREEN_VERSION;
  target: 'background' | 'content';
  event: 'enhancement-progress';
  operation: 'enhanceAudio';
  requestId: string;
  taskId: string;
  progress: AudioEnhancementProgress;
}

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

export interface LocalModelEnhancedTrack extends AudioEnhancementTrackMetadata {
  audioTransferId: string;
}

export interface LocalModelEnhanceAudioResult extends Omit<AudioEnhancementResult, 'tracks'> {
  tracks: LocalModelEnhancedTrack[];
}

export interface LocalModelEnhanceAudioSuccessResponse extends LocalModelResponseBase {
  ok: true;
  operation: 'enhanceAudio';
  result: LocalModelEnhanceAudioResult;
}

export interface LocalModelDownloadSuccessResponse extends LocalModelResponseBase {
  ok: true;
  operation: 'download';
  result: AudioEnhancementChunk;
}

export interface LocalModelReleaseSuccessResponse extends LocalModelResponseBase {
  ok: true;
  operation: 'release';
  result: { released: true };
}

export type LocalModelSuccessResponse =
  | LocalModelUploadSuccessResponse
  | LocalModelTimingSuccessResponse
  | LocalModelDraftSuccessResponse
  | LocalModelSegmentSuccessResponse
  | LocalModelEnhanceAudioSuccessResponse
  | LocalModelDownloadSuccessResponse
  | LocalModelReleaseSuccessResponse;

export type LocalModelErrorCode = 'invalid-request' | 'offscreen-unavailable' | 'inference-failed' | 'timing-unavailable';

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
    typeof value.trackId === 'string' && value.trackId.length > 0 &&
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
      value.operation === 'segment' ||
      value.operation === 'enhanceAudio' ||
      value.operation === 'download' ||
      value.operation === 'release')
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
  if (value.operation === 'release') return Array.isArray(value.transferIds) &&
    value.transferIds.length <= 64 && value.transferIds.every((id) => typeof id === 'string' && id.length > 0);
  if (value.operation === 'download') return typeof value.transferId === 'string' && value.transferId.length > 0 &&
    Number.isSafeInteger(value.chunkIndex) && (value.chunkIndex as number) >= 0;
  if (value.operation === 'enhanceAudio') return typeof value.taskId === 'string' && value.taskId.length > 0 &&
    Array.isArray(value.audioTracks) && value.audioTracks.length > 0 && value.audioTracks.length <= 2 &&
    value.audioTracks.every(isWireCapturedAudioTrack) &&
    new Set(value.audioTracks.map((track) => track.trackId)).size === value.audioTracks.length;
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
  if (value.operation === 'draft') {
    return typeof value.taskId === 'string' && value.taskId.length > 0 && !('audioTracks' in value);
  }
  return (
    isTranscriptJob(value.job) &&
    Array.isArray(value.audioTracks) &&
    value.audioTracks.every(isWireCapturedAudioTrack)
  );
}

export function isLocalModelEnhancementProgress(
  value: unknown,
  target?: LocalModelEnhancementProgressMessage['target'],
  request?: LocalModelEnhanceAudioRequest
): value is LocalModelEnhancementProgressMessage {
  if (!isRecord(value) || value.type !== LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE ||
    value.version !== LOCAL_MODEL_OFFSCREEN_VERSION || value.event !== 'enhancement-progress' ||
    value.operation !== 'enhanceAudio' || (value.target !== 'background' && value.target !== 'content') ||
    (target !== undefined && value.target !== target) ||
    typeof value.requestId !== 'string' || !value.requestId ||
    typeof value.taskId !== 'string' || !value.taskId ||
    Object.keys(value).some((key) => !['type', 'version', 'target', 'event', 'operation', 'requestId', 'taskId', 'progress'].includes(key)) ||
    !isAudioEnhancementProgress(value.progress) ||
    Object.keys(value.progress).some((key) => !['phase', 'trackId', 'trackIndex', 'trackCount', 'completedChunks', 'totalChunks', 'backend', 'backendReason'].includes(key))) {
    return false;
  }
  return request === undefined || (
    value.requestId === request.requestId && value.taskId === request.taskId &&
    value.progress.trackCount === request.audioTracks.length &&
    value.progress.trackId === request.audioTracks[value.progress.trackIndex]?.trackId
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
      typeof track.pcmSha256 === 'string' &&
      /^[0-9a-f]{64}$/.test(track.pcmSha256) &&
      typeof track.sampleRate === 'number' &&
      Number.isSafeInteger(track.sampleRate) &&
      track.sampleRate > 0 &&
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
      ) &&
      Array.isArray(track.segments) &&
      track.segments.every(
        (segment) =>
          isRecord(segment) &&
          typeof segment.id === 'string' &&
          typeof segment.startSeconds === 'number' &&
          typeof segment.endSeconds === 'number' &&
          Number.isSafeInteger(segment.startSample) &&
          Number.isSafeInteger(segment.endSample) &&
          Number.isSafeInteger(segment.sampleRate)
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
        value.error.code === 'timing-unavailable' ||
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
  if (request.operation === 'release') return isRecord(value.result) && value.result.released === true;
  if (request.operation === 'download') {
    const chunk = value.result;
    return isRecord(chunk) && chunk.type === 'audio-chunk' && typeof chunk.trackId === 'string' &&
      chunk.chunkIndex === request.chunkIndex && Number.isSafeInteger(chunk.chunkCount) &&
      Number.isSafeInteger(chunk.totalBytes) && (chunk.totalBytes as number) > 0 &&
      chunk.chunkCount === Math.ceil((chunk.totalBytes as number) / LOCAL_MODEL_AUDIO_CHUNK_BYTES) &&
      (chunk.chunkCount as number) > request.chunkIndex && isBoundedBase64Chunk(chunk.dataBase64) &&
      base64DecodedLength(chunk.dataBase64) === Math.min(LOCAL_MODEL_AUDIO_CHUNK_BYTES,
        (chunk.totalBytes as number) - request.chunkIndex * LOCAL_MODEL_AUDIO_CHUNK_BYTES);
  }
  if (request.operation === 'enhanceAudio') {
    const result = value.result;
    return isRecord(result) && result.ok === true && (result.provider === 'browser-local' || result.provider === 'swarm') &&
      Object.keys(value).every((key) => ['type', 'version', 'requestId', 'operation', 'ok', 'result'].includes(key)) &&
      result.taskId === request.taskId && typeof result.model === 'string' && result.model.length > 0 &&
      typeof result.modelSha256 === 'string' && /^[a-f0-9]{64}$/.test(result.modelSha256) &&
      Object.keys(result).every((key) => ['ok', 'provider', 'taskId', 'model', 'modelSha256', 'tracks', 'cacheStatus', 'cacheMessage'].includes(key)) &&
      (result.cacheStatus === undefined || result.cacheStatus === 'hit' || result.cacheStatus === 'stored' || result.cacheStatus === 'unavailable') &&
      (result.cacheMessage === undefined || typeof result.cacheMessage === 'string' && result.cacheMessage.trim().length > 0) &&
      (result.cacheMessage === undefined || result.cacheStatus === 'unavailable') &&
      (result.cacheStatus !== 'unavailable' || typeof result.cacheMessage === 'string' && result.cacheMessage.trim().length > 0) &&
      Array.isArray(result.tracks) && result.tracks.length === request.audioTracks.length &&
      result.tracks.every((track, index) => isRecord(track) && track.trackId === request.audioTracks[index].trackId &&
        Object.keys(track).every((key) => ['trackId', 'speakerKey', 'trackLabel', 'mimeType', 'sampleRate', 'frameCount', 'sourceSha256', 'wavSha256', 'totalBytes', 'chunkCount', 'audioTransferId'].includes(key)) &&
        typeof track.speakerKey === 'string' && typeof track.trackLabel === 'string' && track.mimeType === 'audio/wav' &&
        Number.isSafeInteger(track.sampleRate) && (track.sampleRate as number) > 0 &&
        Number.isSafeInteger(track.frameCount) && (track.frameCount as number) > 0 &&
        typeof track.sourceSha256 === 'string' && /^[a-f0-9]{64}$/.test(track.sourceSha256) &&
        typeof track.wavSha256 === 'string' && /^[a-f0-9]{64}$/.test(track.wavSha256) &&
        track.totalBytes === 44 + (track.frameCount as number) * 2 &&
        track.chunkCount === Math.ceil((track.totalBytes as number) / LOCAL_MODEL_AUDIO_CHUNK_BYTES) &&
        typeof track.audioTransferId === 'string' && track.audioTransferId.length > 0);
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
