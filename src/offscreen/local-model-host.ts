import type { PreparedL0Track } from '../core/l0-client';
import { buildCanonicalTaskIdentity } from '../core/transcript';
import {
  generateLocalL0DraftFromTiming,
  generateLocalL0SegmentDraft,
  generateLocalL0Timing
} from '../core/local-model-runtime';
import { createVolunteer, defaultVolunteerDependencies, loadVolunteerSettings } from './volunteer';
import { isVolunteerMessage } from '../core/volunteer-protocol';
import {
  LOCAL_MODEL_AUDIO_TRANSFER_STALE_MS,
  LOCAL_MODEL_MAX_BUFFERED_AUDIO_BYTES,
  LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE,
  LOCAL_MODEL_OFFSCREEN_VERSION,
  createLocalModelFailure,
  decodeAudioChunk,
  isLocalModelOffscreenRequest,
  type LocalModelDraftSuccessResponse,
  type LocalModelOffscreenRequest,
  type LocalModelOffscreenResponse,
  type LocalModelSegmentSuccessResponse,
  type LocalModelTimingSuccessResponse,
  type LocalModelUploadRequest,
  type LocalModelUploadSuccessResponse,
  type WireCapturedAudioTrack,
  type WirePreparedL0Track
} from '../core/local-model-offscreen-protocol';
import type {
  CapturedAudioTrack,
  ExtensionSettings,
  L0DraftResponse,
  L0TimingResponse,
  TranscriptJob,
  TranscriptRow
} from '../core/types';

type LocalModelRuntime = {
  generateLocalL0Timing: (
    settings: ExtensionSettings,
    job: TranscriptJob,
    audioTracks: CapturedAudioTrack[]
  ) => Promise<L0TimingResponse>;
  generateLocalL0DraftFromTiming: (timing: L0TimingResponse) => Promise<L0DraftResponse>;
  generateLocalL0SegmentDraft: (
    settings: ExtensionSettings,
    taskId: string,
    row: TranscriptRow,
    tracks: PreparedL0Track[]
  ) => Promise<string>;
};

export type LocalModelRuntimeLoader = () => Promise<LocalModelRuntime>;

export interface LocalModelHostOptions {
  now?: () => number;
  maxBufferedBytes?: number;
  staleTransferMs?: number;
}

export interface LocalModelHost {
  handleRequest: (request: LocalModelOffscreenRequest) => Promise<LocalModelOffscreenResponse>;
  runExclusive: <T>(action: () => Promise<T>) => Promise<T>;
}

type PendingAudioTransfer = {
  state: 'pending';
  chunkCount: number;
  totalBytes: number;
  mimeType: string;
  nextChunkIndex: number;
  chunks: Uint8Array<ArrayBuffer>[];
  receivedBytes: number;
  updatedAt: number;
};

type CompleteAudioTransfer = {
  state: 'complete';
  blob: Blob;
  bufferedBytes: number;
  updatedAt: number;
};

type AudioTransfer = PendingAudioTransfer | CompleteAudioTransfer;

class InvalidAudioTransferError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'InvalidAudioTransferError';
  }
}

async function loadLocalModelRuntime(): Promise<LocalModelRuntime> {
  return { generateLocalL0Timing, generateLocalL0DraftFromTiming, generateLocalL0SegmentDraft };
}

export function createLocalModelHost(
  loadRuntime: LocalModelRuntimeLoader = loadLocalModelRuntime,
  options: LocalModelHostOptions = {}
): LocalModelHost {
  const now = options.now ?? Date.now;
  const maxBufferedBytes = options.maxBufferedBytes ?? LOCAL_MODEL_MAX_BUFFERED_AUDIO_BYTES;
  const staleTransferMs = options.staleTransferMs ?? LOCAL_MODEL_AUDIO_TRANSFER_STALE_MS;
  const transfers = new Map<string, AudioTransfer>();
  const timings = new Map<string, L0TimingResponse>();
  let bufferedBytes = 0;
  let inferenceTail: Promise<void> = Promise.resolve();

  function discardTransfer(transferId: string, transfer: AudioTransfer): void {
    bufferedBytes -= transfer.state === 'pending' ? transfer.receivedBytes : transfer.bufferedBytes;
    transfers.delete(transferId);
  }

  function cleanStaleTransfers(timestamp: number): void {
    for (const [transferId, transfer] of transfers) {
      if (timestamp - transfer.updatedAt >= staleTransferMs) {
        discardTransfer(transferId, transfer);
      }
    }
  }

  function handleUpload(request: LocalModelUploadRequest): LocalModelUploadSuccessResponse {
    const timestamp = now();
    cleanStaleTransfers(timestamp);
    const existing = transfers.get(request.transferId);
    if (!existing && request.chunkIndex !== 0) {
      throw new InvalidAudioTransferError(
        `Audio transfer ${request.transferId} is missing chunk 0; received out-of-order chunk ${request.chunkIndex}.`
      );
    }
    if (existing?.state === 'complete') {
      throw new InvalidAudioTransferError(`Audio transfer ${request.transferId} is already complete.`);
    }
    if (existing && request.chunkIndex !== existing.nextChunkIndex) {
      const kind = request.chunkIndex < existing.nextChunkIndex ? 'duplicate' : 'out-of-order';
      throw new InvalidAudioTransferError(
        `Audio transfer ${request.transferId} received ${kind} chunk ${request.chunkIndex}; expected ${existing.nextChunkIndex}.`
      );
    }

    const transfer: PendingAudioTransfer = existing ?? {
      state: 'pending',
      chunkCount: request.chunkCount,
      totalBytes: request.totalBytes,
      mimeType: request.mimeType,
      nextChunkIndex: 0,
      chunks: [],
      receivedBytes: 0,
      updatedAt: timestamp
    };
    if (
      transfer.chunkCount !== request.chunkCount ||
      transfer.totalBytes !== request.totalBytes ||
      transfer.mimeType !== request.mimeType
    ) {
      throw new InvalidAudioTransferError(
        `Audio transfer ${request.transferId} metadata changed before upload completed.`
      );
    }

    const chunk = decodeAudioChunk(request.dataBase64);
    const receivedBytes = transfer.receivedBytes + chunk.byteLength;
    const isFinalChunk = request.chunkIndex === request.chunkCount - 1;
    if (receivedBytes > request.totalBytes || (!isFinalChunk && receivedBytes >= request.totalBytes)) {
      throw new InvalidAudioTransferError(
        `Audio transfer ${request.transferId} chunk bytes exceed the declared total of ${request.totalBytes}.`
      );
    }
    if (isFinalChunk && receivedBytes !== request.totalBytes) {
      throw new InvalidAudioTransferError(
        `Audio transfer ${request.transferId} ended with ${receivedBytes} bytes; expected ${request.totalBytes}.`
      );
    }
    if (bufferedBytes + chunk.byteLength > maxBufferedBytes) {
      throw new InvalidAudioTransferError(
        `Audio transfer buffer limit of ${maxBufferedBytes} bytes would be exceeded.`
      );
    }

    transfer.chunks.push(chunk);
    transfer.receivedBytes = receivedBytes;
    transfer.nextChunkIndex += 1;
    transfer.updatedAt = timestamp;
    bufferedBytes += chunk.byteLength;
    if (isFinalChunk) {
      transfers.set(request.transferId, {
        state: 'complete',
        blob: new Blob(transfer.chunks, { type: transfer.mimeType }),
        bufferedBytes: transfer.receivedBytes,
        updatedAt: timestamp
      });
    } else {
      transfers.set(request.transferId, transfer);
    }

    return {
      type: LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE,
      version: LOCAL_MODEL_OFFSCREEN_VERSION,
      requestId: request.requestId,
      operation: 'upload',
      ok: true,
      result: {
        transferId: request.transferId,
        nextChunkIndex: request.chunkIndex + 1,
        complete: isFinalChunk
      }
    };
  }

  function resolveCapturedAudioTrack(track: WireCapturedAudioTrack): CapturedAudioTrack {
    const transfer = transfers.get(track.audioTransferId);
    if (!transfer || transfer.state !== 'complete') {
      throw new InvalidAudioTransferError(
        `Audio transfer ${track.audioTransferId} is missing or incomplete for track ${track.trackId}.`
      );
    }
    return {
      trackId: track.trackId,
      ...(track.speakerKey === undefined ? {} : { speakerKey: track.speakerKey }),
      ...(track.trackLabel === undefined ? {} : { trackLabel: track.trackLabel }),
      source: track.source,
      blob: transfer.blob,
      mimeType: track.mimeType
    };
  }

  function resolveCapturedAudioTracks(tracks: WireCapturedAudioTrack[]): CapturedAudioTrack[] {
    return tracks.map(resolveCapturedAudioTrack);
  }

  function resolvePreparedTracks(tracks: WirePreparedL0Track[]): PreparedL0Track[] {
    return tracks.map((track) => ({
      lane: track.lane,
      fieldName: track.fieldName,
      audio: resolveCapturedAudioTrack(track.audio)
    }));
  }

  function consumeTransfers(transferIds: Iterable<string>): void {
    for (const transferId of new Set(transferIds)) {
      const transfer = transfers.get(transferId);
      if (transfer) discardTransfer(transferId, transfer);
    }
  }

  async function execute(
    request: Exclude<LocalModelOffscreenRequest, LocalModelUploadRequest>
  ): Promise<LocalModelOffscreenResponse> {
    cleanStaleTransfers(now());
    try {
      if (request.operation === 'timing') {
        let result = timings.get(buildCanonicalTaskIdentity(request.job));
        if (!result) {
          const tracks = resolveCapturedAudioTracks(request.audioTracks);
          const runtime = await loadRuntime();
          result = await runtime.generateLocalL0Timing(request.settings, request.job, tracks);
          timings.set(result.taskId, result);
          if (timings.size > 2) timings.delete(timings.keys().next().value!);
        }
        consumeTransfers(request.audioTracks.map((track) => track.audioTransferId));
        const response: LocalModelTimingSuccessResponse = {
          type: LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE,
          version: LOCAL_MODEL_OFFSCREEN_VERSION,
          requestId: request.requestId,
          operation: 'timing',
          ok: true,
          result
        };
        return response;
      }
      if (request.operation === 'draft') {
        const timing = timings.get(request.taskId);
        if (!timing) throw new Error(`Timing for task ${request.taskId} is not available in the offscreen runtime.`);
        const runtime = await loadRuntime();
        const result = await runtime.generateLocalL0DraftFromTiming(timing);
        const response: LocalModelDraftSuccessResponse = {
          type: LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE,
          version: LOCAL_MODEL_OFFSCREEN_VERSION,
          requestId: request.requestId,
          operation: 'draft',
          ok: true,
          result
        };
        return response;
      }
      const tracks = resolvePreparedTracks(request.tracks);
      const runtime = await loadRuntime();
      const result = await runtime.generateLocalL0SegmentDraft(
        request.settings,
        request.taskId,
        request.row,
        tracks
      );
      consumeTransfers(request.tracks.map((track) => track.audio.audioTransferId));
      const response: LocalModelSegmentSuccessResponse = {
        type: LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE,
        version: LOCAL_MODEL_OFFSCREEN_VERSION,
        requestId: request.requestId,
        operation: 'segment',
        ok: true,
        result
      };
      return response;
    } catch (error) {
      const code = error instanceof InvalidAudioTransferError ? 'invalid-request' : 'inference-failed';
      return createLocalModelFailure(request, code, error);
    }
  }

  function runExclusive<T>(action: () => Promise<T>): Promise<T> {
    const result = inferenceTail.then(action);
    inferenceTail = result.then(
      () => undefined,
      () => undefined
    );
    return result;
  }

  function handleRequest(request: LocalModelOffscreenRequest): Promise<LocalModelOffscreenResponse> {
    if (request.operation === 'upload') {
      try {
        return Promise.resolve(handleUpload(request));
      } catch (error) {
        return Promise.resolve(createLocalModelFailure(request, 'invalid-request', error));
      }
    }
    return runExclusive(() => execute(request));
  }

  return { handleRequest, runExclusive };
}

const runtimeMessages = globalThis.chrome?.runtime?.onMessage;
if (runtimeMessages && typeof runtimeMessages.addListener === 'function') {
  const host = createLocalModelHost();
  const volunteer = createVolunteer({ ...defaultVolunteerDependencies, runExclusive: host.runExclusive });
  // A recovered document must resume polling even if the service worker did not restart.
  void loadVolunteerSettings().then((settings) => {
    if (settings.localModelsEnabled && settings.volunteerInferenceEnabled) volunteer.start();
  }).catch(() => {
    // The background lifecycle reports setup failures to Options.
  });
  runtimeMessages.addListener((message: unknown, _sender, sendResponse) => {
    if (isVolunteerMessage(message, 'offscreen')) {
      sendResponse(message.action === 'start' ? volunteer.start() :
        message.action === 'stop' ? volunteer.stop() : volunteer.getStatus());
      return false;
    }
    if (!isLocalModelOffscreenRequest(message, 'offscreen')) return false;
    void host.handleRequest(message).then(sendResponse);
    return true;
  });
}
