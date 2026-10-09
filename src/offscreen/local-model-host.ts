import { enhanceAudioTracks } from '../core/audio-enhancement-runtime';
import { requireReviewGraderAccess } from '../core/review-grader-access';
import type { PreparedL0Track } from '../core/l0-client';
import { IS_DEV_C_DENOISE, isBrowserLocalMode } from '../core/settings';
import { buildCanonicalTaskIdentity } from '../core/transcript';
import {
  generateLocalL0DraftFromTiming,
  generateLocalL0SegmentDraft,
  generateLocalL0Timing,
  isLocalTimingCurrent,
  LocalTimingUnavailableError
} from '../core/local-model-runtime';
import { createVolunteer, defaultVolunteerDependencies, loadVolunteerSettings } from './volunteer';
import { isVolunteerMessage } from '../core/volunteer-protocol';
import {
  LOCAL_MODEL_AUDIO_CHUNK_BYTES,
  LOCAL_MODEL_AUDIO_TRANSFER_STALE_MS,
  LOCAL_MODEL_MAX_BUFFERED_AUDIO_BYTES,
  LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE,
  LOCAL_MODEL_OFFSCREEN_VERSION,
  createLocalModelFailure,
  decodeAudioChunk,
  encodeAudioChunk,
  isLocalModelOffscreenRequest,
  type LocalModelDraftSuccessResponse,
  type LocalModelDownloadRequest,
  type LocalModelDownloadSuccessResponse,
  type LocalModelEnhanceAudioSuccessResponse,
  type LocalModelEnhancementProgressMessage,
  type LocalModelReleaseRequest,
  type LocalModelReleaseSuccessResponse,
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
  isLocalTimingCurrent: (timing: L0TimingResponse) => Promise<boolean>;
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
  enhanceAudio?: typeof enhanceAudioTracks;
  authorizeEnhancement?: () => Promise<AbortSignal>;
  onProgress?: (message: LocalModelEnhancementProgressMessage) => void | Promise<void>;
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

type OutputAudioTransfer = {
  bytes: Uint8Array<ArrayBuffer>;
  trackId: string;
  chunkCount: number;
  nextChunkIndex: number;
  expires: ReturnType<typeof setTimeout>;
};

class InvalidAudioTransferError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'InvalidAudioTransferError';
  }
}

async function loadLocalModelRuntime(): Promise<LocalModelRuntime> {
  return { generateLocalL0Timing, generateLocalL0DraftFromTiming, generateLocalL0SegmentDraft, isLocalTimingCurrent };
}

export function createLocalModelHost(
  loadRuntime: LocalModelRuntimeLoader = loadLocalModelRuntime,
  options: LocalModelHostOptions = {}
): LocalModelHost {
  const now = options.now ?? Date.now;
  const maxBufferedBytes = options.maxBufferedBytes ?? LOCAL_MODEL_MAX_BUFFERED_AUDIO_BYTES;
  const staleTransferMs = options.staleTransferMs ?? LOCAL_MODEL_AUDIO_TRANSFER_STALE_MS;
  const transfers = new Map<string, AudioTransfer>();
  const outputs = new Map<string, OutputAudioTransfer>();
  const timings = new Map<string, L0TimingResponse>();
  let bufferedBytes = 0;
  let inferenceTail: Promise<void> = Promise.resolve();
  let observedEnhancementGrant: AbortSignal | undefined;

  function discardOutput(transferId: string): void {
    const output = outputs.get(transferId);
    if (!output) return;
    clearTimeout(output.expires);
    bufferedBytes -= output.bytes.byteLength;
    outputs.delete(transferId);
  }

  function handleDownload(request: LocalModelDownloadRequest): LocalModelDownloadSuccessResponse {
    const output = outputs.get(request.transferId);
    if (!output || request.chunkIndex !== output.nextChunkIndex || request.chunkIndex >= output.chunkCount) {
      throw new InvalidAudioTransferError(`Enhanced audio transfer ${request.transferId} is missing, expired, or requested out of order.`);
    }
    const start = request.chunkIndex * LOCAL_MODEL_AUDIO_CHUNK_BYTES;
    const dataBase64 = encodeAudioChunk(output.bytes.subarray(start, Math.min(start + LOCAL_MODEL_AUDIO_CHUNK_BYTES, output.bytes.length)));
    output.nextChunkIndex++;
    const result = { type: 'audio-chunk' as const, trackId: output.trackId, chunkIndex: request.chunkIndex,
      chunkCount: output.chunkCount, totalBytes: output.bytes.byteLength, dataBase64 };
    if (output.nextChunkIndex === output.chunkCount) discardOutput(request.transferId);
    return { type: LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE, version: LOCAL_MODEL_OFFSCREEN_VERSION,
      requestId: request.requestId, operation: 'download', ok: true, result };
  }

  function handleRelease(request: LocalModelReleaseRequest): LocalModelReleaseSuccessResponse {
    consumeTransfers(request.transferIds);
    for (const transferId of request.transferIds) discardOutput(transferId);
    return { type: LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE, version: LOCAL_MODEL_OFFSCREEN_VERSION,
      requestId: request.requestId, operation: 'release', ok: true, result: { released: true } };
  }
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
    request: Exclude<LocalModelOffscreenRequest, LocalModelUploadRequest | LocalModelDownloadRequest | LocalModelReleaseRequest>,
    enhancementAccess?: AbortSignal
  ): Promise<LocalModelOffscreenResponse> {
    cleanStaleTransfers(now());
    try {
      if (request.operation === 'enhanceAudio') {
        if (!enhancementAccess) throw new Error('This operation is unavailable.');
        enhancementAccess.throwIfAborted();
        const tracks = resolveCapturedAudioTracks(request.audioTracks);
        const batch = await (options.enhanceAudio ?? enhanceAudioTracks)(tracks, progress => {
          enhancementAccess.throwIfAborted();
          return options.onProgress?.({
            type: LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE, version: LOCAL_MODEL_OFFSCREEN_VERSION,
            target: 'background', event: 'enhancement-progress', operation: 'enhanceAudio',
            requestId: request.requestId, taskId: request.taskId, progress
          });
        }, { taskId: request.taskId, signal: enhancementAccess });
        enhancementAccess.throwIfAborted();
        consumeTransfers(request.audioTracks.map((track) => track.audioTransferId));
        const totalBytes = batch.tracks.reduce((sum, track) => sum + track.bytes.byteLength, 0);
        if (bufferedBytes + totalBytes > maxBufferedBytes) throw new InvalidAudioTransferError('Enhanced audio exceeds the bounded transfer buffer.');
        const resultTracks = batch.tracks.map((track, index) => {
          const audioTransferId = `enhanced:${request.requestId}:${index}`;
          if (outputs.has(audioTransferId)) throw new InvalidAudioTransferError('An enhanced audio transfer ID is already in use.');
          return { ...track.metadata, audioTransferId };
        });
        for (let index = 0; index < batch.tracks.length; index++) {
          const track = batch.tracks[index], transferId = resultTracks[index].audioTransferId;
          outputs.set(transferId, { bytes: track.bytes, trackId: track.metadata.trackId,
            chunkCount: track.metadata.chunkCount, nextChunkIndex: 0,
            expires: setTimeout(() => discardOutput(transferId), staleTransferMs) });
          bufferedBytes += track.bytes.byteLength;
        }
        const response: LocalModelEnhanceAudioSuccessResponse = {
          type: LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE, version: LOCAL_MODEL_OFFSCREEN_VERSION,
          requestId: request.requestId, operation: 'enhanceAudio', ok: true,
          result: { ok: true, provider: batch.provider, taskId: request.taskId,
            model: batch.model, modelSha256: batch.modelSha256, tracks: resultTracks,
            ...(batch.cacheStatus === undefined ? {} : { cacheStatus: batch.cacheStatus }),
            ...(batch.cacheMessage === undefined ? {} : { cacheMessage: batch.cacheMessage }) }
        };
        return response;
      }
      if (request.operation === 'timing') {
        let result = timings.get(buildCanonicalTaskIdentity(request.job));
        if (result && !(await (await loadRuntime()).isLocalTimingCurrent(result))) {
          timings.delete(result.taskId);
          result = undefined;
        }
        if (!result) {
          const tracks = resolveCapturedAudioTracks(request.audioTracks);
          const runtime = await loadRuntime();
          result = await runtime.generateLocalL0Timing(request.settings, request.job, tracks);
          timings.set(result.taskId, result);
          if (timings.size > 2) timings.delete(timings.keys().next().value!);
        }
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
        const runtime = timing ? await loadRuntime() : null;
        if (!timing || !runtime || !(await runtime.isLocalTimingCurrent(timing))) {
          timings.delete(request.taskId);
          return createLocalModelFailure(request, 'timing-unavailable', new LocalTimingUnavailableError(request.taskId));
        }
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
      const code = error instanceof InvalidAudioTransferError ? 'invalid-request' :
        error instanceof LocalTimingUnavailableError ? 'timing-unavailable' : 'inference-failed';
      return createLocalModelFailure(request, code, error);
    } finally {
      if (request.operation === 'timing') consumeTransfers(request.audioTracks.map((track) => track.audioTransferId));
      if (request.operation === 'enhanceAudio') consumeTransfers(request.audioTracks.map((track) => track.audioTransferId));
      if (request.operation === 'segment') consumeTransfers(request.tracks.map((track) => track.audio.audioTransferId));
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

  async function handleRequest(request: LocalModelOffscreenRequest): Promise<LocalModelOffscreenResponse> {
    if (request.operation === 'download' || request.operation === 'release') {
      try {
        if (request.operation === 'download') {
          const grant = await (options.authorizeEnhancement ?? requireReviewGraderAccess)();
          grant.throwIfAborted();
        }
        return Promise.resolve(request.operation === 'download' ? handleDownload(request) : handleRelease(request));
      } catch (error) {
        return Promise.resolve(createLocalModelFailure(request, 'invalid-request', error));
      }
    }
    if (request.operation === 'upload') {
      try {
        return Promise.resolve(handleUpload(request));
      } catch (error) {
        return Promise.resolve(createLocalModelFailure(request, 'invalid-request', error));
      }
    }
    let enhancementAccess: AbortSignal | undefined;
    if (request.operation === 'enhanceAudio') {
      try {
        enhancementAccess = await (options.authorizeEnhancement ?? requireReviewGraderAccess)();
        enhancementAccess.throwIfAborted();
        if (enhancementAccess !== observedEnhancementGrant) {
          observedEnhancementGrant = enhancementAccess;
          enhancementAccess.addEventListener('abort', () => {
            for (const id of outputs.keys()) discardOutput(id);
            observedEnhancementGrant = undefined;
          }, { once: true });
        }
        await options.onProgress?.({
          type: LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE, version: LOCAL_MODEL_OFFSCREEN_VERSION,
          target: 'background', event: 'enhancement-progress', operation: 'enhanceAudio',
          requestId: request.requestId, taskId: request.taskId,
          progress: { phase: 'queued', trackId: request.audioTracks[0].trackId,
            trackIndex: 0, trackCount: request.audioTracks.length, completedChunks: 0, totalChunks: 0 }
        });
      } catch (error) {
        consumeTransfers(request.audioTracks.map((track) => track.audioTransferId));
        return createLocalModelFailure(request, 'offscreen-unavailable', error);
      }
    }
    return runExclusive(() => execute(request, enhancementAccess));
  }

  return { handleRequest, runExclusive };
}

const runtimeMessages = globalThis.chrome?.runtime?.onMessage;
if (runtimeMessages && typeof runtimeMessages.addListener === 'function') {
  const host = createLocalModelHost(loadLocalModelRuntime, {
    onProgress: async (message) => {
      const acknowledgement: unknown = await chrome.runtime.sendMessage(message);
      if (acknowledgement !== true) {
        const detail = acknowledgement && typeof acknowledgement === 'object' &&
          'message' in acknowledgement && typeof acknowledgement.message === 'string'
          ? acknowledgement.message : 'The owning content tab rejected enhancement progress.';
        throw new Error(detail);
      }
    }
  });
  const volunteer = createVolunteer({ ...defaultVolunteerDependencies, runExclusive: host.runExclusive });
  // A recovered document must resume polling even if the service worker did not restart.
  void loadVolunteerSettings().then((settings) => {
    if (!IS_DEV_C_DENOISE && isBrowserLocalMode(settings) && settings.volunteerInferenceEnabled) volunteer.start();
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
