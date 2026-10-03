import { buildCanonicalTaskIdentity } from './transcript';
import { encodeAudioChunk, LOCAL_MODEL_AUDIO_CHUNK_BYTES } from './local-model-offscreen-protocol';
import { MAI_MAX_AUDIO_BYTES, MAI_MESSAGE_TYPE, MAI_PROTOCOL_VERSION, MaiError,
  type MaiRequest, type MaiOperation, type MaiResponse, type MaiEnvelope } from './mai-protocol';
import type { L0TimingRequestCallbacks, L0TimingQueueStatus } from './l0-timing-client';
import { prepareL0TimingTracks, parseL0TimingResponse } from './l0-timing-client';
import type { BrokerRedistributionGroup, BrokerRedistributeTextResponse, CapturedAudioTrack, ExtensionSettings,
  L0DraftResponse, L0TimingResponse, TranscriptJob, TranscriptRow } from './types';

export class MaiBridgeError extends MaiError {
  constructor(readonly operation: MaiOperation, code: string, message: string) {
    super(code, message);
    this.name = 'MaiBridgeError';
  }
}
export type MaiMessageSender = (message: MaiRequest) => Promise<unknown>;
let requestSequence = 0;
function requestId(operation: MaiOperation): string {
  requestSequence += 1;
  return `mai:${operation}:${globalThis.crypto?.randomUUID?.() ?? `${Date.now()}:${requestSequence}`}`;
}
function queue(callbacks: L0TimingRequestCallbacks | undefined, status: L0TimingQueueStatus): void {
  try { callbacks?.onQueueStatus?.(status); } catch { /* Observational callbacks cannot interrupt transcription. */ }
}
export function createMaiClient(sendMessage: MaiMessageSender = (message) => chrome.runtime.sendMessage(message)) {
  async function request<T>(message: MaiRequest): Promise<T> {
    let value: unknown;
    try { value = await sendMessage(message); }
    catch { throw new MaiBridgeError(message.operation, 'background-unavailable', 'The MAI extension background is unavailable. No automatic paid retry was attempted.'); }
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      throw new MaiBridgeError(message.operation, 'invalid-response', 'The background returned an invalid MAI response.');
    }
    const response = value as MaiResponse;
    if (response.type !== message.type || response.version !== message.version || response.requestId !== message.requestId ||
        response.operation !== message.operation || typeof response.ok !== 'boolean') {
      throw new MaiBridgeError(message.operation, 'invalid-response', 'The background returned a mismatched MAI response.');
    }
    if (!response.ok) {
      if (!response.error || typeof response.error.code !== 'string' || typeof response.error.message !== 'string') {
        throw new MaiBridgeError(message.operation, 'invalid-response', 'The background returned an invalid MAI error.');
      }
      throw new MaiBridgeError(message.operation, response.error.code, response.error.message);
    }
    return response.result as T;
  }
  function envelope(operation: MaiOperation): MaiEnvelope {
    return { type: MAI_MESSAGE_TYPE, version: MAI_PROTOCOL_VERSION, requestId: requestId(operation), operation };
  }
  async function upload(blob: Blob): Promise<string> {
    if (!(blob instanceof Blob) || blob.size < 44 || blob.size > MAI_MAX_AUDIO_BYTES) {
      throw new MaiBridgeError('timing', 'invalid-audio', 'MAI audio must be a non-empty WAV within the authorized audio size limit.');
    }
    const transferId = requestId('upload');
    const chunkCount = Math.ceil(blob.size / LOCAL_MODEL_AUDIO_CHUNK_BYTES);
    for (let chunkIndex = 0; chunkIndex < chunkCount; chunkIndex += 1) {
      const bytes = new Uint8Array(await blob.slice(chunkIndex * LOCAL_MODEL_AUDIO_CHUNK_BYTES, (chunkIndex + 1) * LOCAL_MODEL_AUDIO_CHUNK_BYTES).arrayBuffer());
      const result = await request<{ transferId: string; nextChunkIndex: number; complete: boolean }>({
        ...envelope('upload'), operation: 'upload', transferId, chunkIndex, chunkCount, totalBytes: blob.size, dataBase64: encodeAudioChunk(bytes)
      });
      if (!result || result.transferId !== transferId || result.nextChunkIndex !== chunkIndex + 1 || result.complete !== (chunkIndex + 1 === chunkCount)) {
        throw new MaiBridgeError('upload', 'invalid-response', 'The background did not acknowledge the correct audio chunk.');
      }
    }
    return transferId;
  }
  return {
    async generateMaiL0Timing(_settings: ExtensionSettings, job: TranscriptJob, audioTracks: CapturedAudioTrack[], callbacks?: L0TimingRequestCallbacks): Promise<L0TimingResponse> {
      // Imported preparation functions are invoked only after module initialization.
      // Only canonical identity and exact two prepared source lanes cross IPC.
      const taskId = buildCanonicalTaskIdentity(job);
      const timingEnvelope = envelope('timing');
      queue(callbacks, { requestId: timingEnvelope.requestId, status: 'preparing' });
      const sources = [];
      for (const track of prepareL0TimingTracks(job, audioTracks)) {
        sources.push({ lane: track.lane, trackId: track.audio.trackId, transferId: await upload(track.audio.blob) });
      }
      queue(callbacks, { requestId: timingEnvelope.requestId, status: 'running', position: 0, queuedCount: 0 });
      const result = parseL0TimingResponse(await request({ ...timingEnvelope, operation: 'timing', taskId, sources }), taskId);
      queue(callbacks, { requestId: timingEnvelope.requestId, status: 'completed', position: 0, queuedCount: 0 });
      return result;
    },
    async lookupMaiL0Timing(_settings: ExtensionSettings, taskId: string): Promise<L0TimingResponse | null> {
      const result = await request<unknown>({ ...envelope('lookup'), operation: 'lookup', taskId });
      if (result === null) return null;
      return parseL0TimingResponse(result, taskId);
    },
    async generateMaiL0Draft(_settings: ExtensionSettings, job: TranscriptJob): Promise<L0DraftResponse> {
      return request<L0DraftResponse>({ ...envelope('draft'), operation: 'draft', taskId: buildCanonicalTaskIdentity(job) });
    },
    async generateMaiL0SegmentDraft(_settings: ExtensionSettings, taskId: string, row: TranscriptRow): Promise<string> {
      if (row.startSeconds === null || row.endSeconds === null) {
        throw new MaiBridgeError('segment', 'invalid-request', 'MAI native segment lookup requires finite source time bounds.');
      }
      const result = await request<unknown>({ ...envelope('segment'), operation: 'segment', taskId, lane: row.speakerKey,
        startSeconds: row.startSeconds, endSeconds: row.endSeconds });
      if (typeof result !== 'string') throw new MaiBridgeError('segment', 'invalid-response', 'The background returned invalid native segment text.');
      return result;
    },
    async redistributeMaiText(_settings: ExtensionSettings, groups: BrokerRedistributionGroup[]): Promise<BrokerRedistributeTextResponse> {
      return request<BrokerRedistributeTextResponse>({ ...envelope('redistribute'), operation: 'redistribute', groups });
    }
  };
}
const client = createMaiClient();
export const generateMaiL0Timing = client.generateMaiL0Timing;
export const lookupMaiL0Timing = client.lookupMaiL0Timing;
export const generateMaiL0Draft = client.generateMaiL0Draft;
export const generateMaiL0SegmentDraft = client.generateMaiL0SegmentDraft;
export const redistributeMaiText = client.redistributeMaiText;
