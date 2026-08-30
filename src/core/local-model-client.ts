import type { PreparedL0Track } from './l0-client';
import type { L0TimingRequestCallbacks, L0TimingQueueStatus } from './l0-timing-client';
import {
  LOCAL_MODEL_AUDIO_CHUNK_BYTES,
  LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE,
  LOCAL_MODEL_OFFSCREEN_VERSION,
  encodeAudioChunk,
  isLocalModelOffscreenResponse,
  type LocalModelDraftRequest,
  type LocalModelOffscreenRequest,
  type LocalModelOperation,
  type LocalModelSegmentRequest,
  type LocalModelTimingRequest,
  type LocalModelUploadRequest,
  type LocalModelUploadResult,
  type WireCapturedAudioTrack,
  type WirePreparedL0Track
} from './local-model-offscreen-protocol';
import type {
  CapturedAudioTrack,
  ExtensionSettings,
  L0DraftResponse,
  L0TimingResponse,
  TranscriptJob,
  TranscriptRow
} from './types';

export type LocalModelMessageSender = (message: LocalModelOffscreenRequest) => Promise<unknown>;

let requestSequence = 0;

function nextRequestId(operation: LocalModelOperation): string {
  requestSequence += 1;
  const randomId = globalThis.crypto?.randomUUID?.();
  return `local-model:${operation}:${randomId || `${Date.now()}:${requestSequence}`}`;
}

export class LocalModelBridgeError extends Error {
  readonly code: string;
  readonly operation: LocalModelOperation;

  constructor(operation: LocalModelOperation, code: string, message: string) {
    super(`Local model ${operation} failed: ${message}`);
    this.name = 'LocalModelBridgeError';
    this.code = code;
    this.operation = operation;
  }
}

function defaultMessageSender(message: LocalModelOffscreenRequest): Promise<unknown> {
  const sendMessage = globalThis.chrome?.runtime?.sendMessage;
  if (typeof sendMessage !== 'function') {
    return Promise.reject(
      new LocalModelBridgeError(
        message.operation,
        'offscreen-unavailable',
        'Chrome runtime messaging is unavailable.'
      )
    );
  }
  return sendMessage(message);
}

function emitQueueStatus(
  callbacks: L0TimingRequestCallbacks | undefined,
  status: L0TimingQueueStatus
): void {
  try {
    callbacks?.onQueueStatus?.(status);
  } catch {
    // Queue status is observational; callback failures must not abort inference.
  }
}

export function createLocalModelClient(sendMessage: LocalModelMessageSender = defaultMessageSender) {
  async function request<T>(message: LocalModelOffscreenRequest): Promise<T> {
    let response: unknown;
    try {
      response = await sendMessage(message);
    } catch (error) {
      if (error instanceof LocalModelBridgeError) throw error;
      const detail = error instanceof Error ? error.message : String(error);
      throw new LocalModelBridgeError(message.operation, 'offscreen-unavailable', detail);
    }
    if (!isLocalModelOffscreenResponse(response, message)) {
      throw new LocalModelBridgeError(
        message.operation,
        'invalid-response',
        'The offscreen document returned an invalid or mismatched response.'
      );
    }
    if (!response.ok) {
      throw new LocalModelBridgeError(message.operation, response.error.code, response.error.message);
    }
    return response.result as T;
  }

  async function uploadAudioBlob(blob: Blob, parentOperation: LocalModelOperation): Promise<string> {
    if (!blob || typeof blob.type !== 'string' || typeof blob.size !== 'number' || typeof blob.slice !== 'function') {
      throw new LocalModelBridgeError(parentOperation, 'invalid-request', 'Audio payload must be a Blob.');
    }
    const transferId = nextRequestId('upload');
    const chunkCount = Math.max(1, Math.ceil(blob.size / LOCAL_MODEL_AUDIO_CHUNK_BYTES));
    try {
      for (let chunkIndex = 0; chunkIndex < chunkCount; chunkIndex += 1) {
        const start = chunkIndex * LOCAL_MODEL_AUDIO_CHUNK_BYTES;
        const end = Math.min(start + LOCAL_MODEL_AUDIO_CHUNK_BYTES, blob.size);
        const bytes = new Uint8Array(await blob.slice(start, end).arrayBuffer());
        const message: LocalModelUploadRequest = {
          type: LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE,
          version: LOCAL_MODEL_OFFSCREEN_VERSION,
          target: 'background',
          requestId: nextRequestId('upload'),
          operation: 'upload',
          transferId,
          chunkIndex,
          chunkCount,
          totalBytes: blob.size,
          mimeType: blob.type,
          dataBase64: encodeAudioChunk(bytes)
        };
        await request<LocalModelUploadResult>(message);
      }
    } catch (error) {
      if (error instanceof LocalModelBridgeError) {
        throw new LocalModelBridgeError(
          parentOperation,
          error.code,
          `Audio transfer ${transferId} was rejected: ${error.message}`
        );
      }
      const detail = error instanceof Error ? error.message : String(error);
      throw new LocalModelBridgeError(
        parentOperation,
        'invalid-request',
        `Could not serialize audio for Chrome runtime messaging: ${detail}`
      );
    }
    return transferId;
  }

  async function uploadCapturedAudioTracks(
    tracks: CapturedAudioTrack[],
    operation: LocalModelOperation
  ): Promise<WireCapturedAudioTrack[]> {
    const wireTracks: WireCapturedAudioTrack[] = [];
    for (const track of tracks) {
      const audioTransferId = await uploadAudioBlob(track.blob, operation);
      wireTracks.push({
        trackId: track.trackId,
        ...(track.speakerKey === undefined ? {} : { speakerKey: track.speakerKey }),
        ...(track.trackLabel === undefined ? {} : { trackLabel: track.trackLabel }),
        source: track.source,
        audioTransferId,
        mimeType: track.mimeType
      });
    }
    return wireTracks;
  }

  async function uploadPreparedTracks(
    tracks: PreparedL0Track[]
  ): Promise<WirePreparedL0Track[]> {
    const wireTracks: WirePreparedL0Track[] = [];
    for (const track of tracks) {
      const [audio] = await uploadCapturedAudioTracks([track.audio], 'segment');
      wireTracks.push({ lane: track.lane, fieldName: track.fieldName, audio });
    }
    return wireTracks;
  }

  return {
    async generateLocalL0Timing(
      settings: ExtensionSettings,
      job: TranscriptJob,
      audioTracks: CapturedAudioTrack[],
      callbacks?: L0TimingRequestCallbacks
    ): Promise<L0TimingResponse> {
      const requestId = nextRequestId('timing');
      emitQueueStatus(callbacks, { requestId, status: 'preparing' });
      const message: LocalModelTimingRequest = {
        type: LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE,
        version: LOCAL_MODEL_OFFSCREEN_VERSION,
        target: 'background',
        requestId,
        operation: 'timing',
        settings,
        job,
        audioTracks: await uploadCapturedAudioTracks(audioTracks, 'timing')
      };
      emitQueueStatus(callbacks, { requestId, status: 'running', position: 0, queuedCount: 0 });
      const result = await request<L0TimingResponse>(message);
      emitQueueStatus(callbacks, { requestId, status: 'completed', position: 0, queuedCount: 0 });
      return result;
    },

    async generateLocalL0Draft(
      settings: ExtensionSettings,
      job: TranscriptJob,
      audioTracks: CapturedAudioTrack[]
    ): Promise<L0DraftResponse> {
      const message: LocalModelDraftRequest = {
        type: LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE,
        version: LOCAL_MODEL_OFFSCREEN_VERSION,
        target: 'background',
        requestId: nextRequestId('draft'),
        operation: 'draft',
        settings,
        job,
        audioTracks: await uploadCapturedAudioTracks(audioTracks, 'draft')
      };
      return request<L0DraftResponse>(message);
    },

    async generateLocalL0SegmentDraft(
      settings: ExtensionSettings,
      taskId: string,
      row: TranscriptRow,
      tracks: PreparedL0Track[]
    ): Promise<string> {
      const message: LocalModelSegmentRequest = {
        type: LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE,
        version: LOCAL_MODEL_OFFSCREEN_VERSION,
        target: 'background',
        requestId: nextRequestId('segment'),
        operation: 'segment',
        settings,
        taskId,
        row,
        tracks: await uploadPreparedTracks(tracks)
      };
      return request<string>(message);
    }
  };
}

const localModelClient = createLocalModelClient();

export const generateLocalL0Timing = localModelClient.generateLocalL0Timing;
export const generateLocalL0Draft = localModelClient.generateLocalL0Draft;
export const generateLocalL0SegmentDraft = localModelClient.generateLocalL0SegmentDraft;
