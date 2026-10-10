import {
  AI_BROKER_EXTENSION_ID_ATTR,
  AI_BROKER_INTERNAL_MESSAGE_TYPE,
  AI_BROKER_INTERNAL_PORT_NAME,
  providerAllowsLocalFallback,
  shouldUseRemoteBroker,
  type AiBrokerInternalRequest,
  type AiBrokerPortMessage,
  type AiBrokerResponse
} from '../core/ai-broker-protocol';
import { captureAudioTracksForDrafting, captureOriginalAudioTracksForEnhancement } from '../core/audio-cues';
import { redistributeTextWithBroker, transcribeSegmentWithBroker } from '../core/backend-client';
import { generateL0SegmentDraft } from '../core/l0-client';
import { enhanceLocalAudio, generateLocalL0SegmentDraft, LocalModelBridgeError } from '../core/local-model-client';
import { generateMaiL0SegmentDraft, redistributeMaiText, MaiBridgeError } from '../core/mai-client';
import { getCurrentL0TimingGeneration, recoverCurrentLocalL0Timing, waitForCurrentL0Timing } from './l0-timing-service';
import { isBrowserLocalMode, loadSettings } from '../core/settings';
import { buildCanonicalTaskIdentity, captureTranscriptJob } from '../core/transcript';
import type { ExtensionSettings, TranscriptRow } from '../core/types';

const AI_BROKER_CONTENT_BUILD = 'port-stream-postmortem-2026-06-23';
const AI_BROKER_CONTENT_BUILD_ATTR = 'data-babel-gold-drafting-ai-broker-build';
const BROKER_BACKEND_PROGRESS_INTERVAL_MS = 5000;

class NoFallbackBrokerError extends Error {
  constructor(cause: unknown) {
    super(cause instanceof Error ? cause.message : String(cause), { cause });
    this.name = 'NoFallbackBrokerError';
  }
}

function brokerError(reason: Extract<AiBrokerResponse, { ok: false }>['reason'], message: string, fallbackAllowed: boolean): Extract<AiBrokerResponse, { ok: false }> {
  return {
    ok: false,
    reason,
    message,
    fallbackAllowed
  };
}

function formatElapsedSeconds(elapsedMs: number): string {
  return Math.max(0, Math.round(elapsedMs / 1000)) + 's';
}

function allowsBrokerFallback(message: AiBrokerInternalRequest, settings: ExtensionSettings | null): boolean {
  if (message.operation === 'enhanceAudio') return false;
  if (settings && settings.mode !== 'advanced') return false;
  if (message.operation === 'transcribeSegmentL0' && settings && isBrowserLocalMode(settings)) {
    return false;
  }
  return settings === null || providerAllowsLocalFallback(settings.aiBrokerProvider);
}

async function withBrokerBackendProgress<T>(
  operation: AiBrokerInternalRequest['operation'],
  emit: ((message: AiBrokerPortMessage) => void) | undefined,
  request: () => Promise<T>,
  waitingTarget = 'Gold Drafting backend'
): Promise<T> {
  const startedAt = Date.now();
  const intervalId = emit
    ? setInterval(() => {
        const elapsedMs = Math.max(0, Date.now() - startedAt);
        emit({
          type: 'event',
          event: 'backend-waiting',
          operation,
          elapsedMs,
          message: `Still waiting for ${waitingTarget} after ${formatElapsedSeconds(elapsedMs)}.`
        });
      }, BROKER_BACKEND_PROGRESS_INTERVAL_MS)
    : null;

  try {
    return await request();
  } finally {
    if (intervalId) {
      clearInterval(intervalId);
    }
  }
}

function isBrokerRequest(message: unknown): message is AiBrokerInternalRequest {
  return (
    Boolean(message && typeof message === 'object') &&
    (message as { type?: unknown }).type === AI_BROKER_INTERNAL_MESSAGE_TYPE &&
    (message as { version?: unknown }).version === 1
  );
}

function remoteRequestBase(settings: ExtensionSettings) {
  return {
    openRouterApiKey: settings.openRouterApiKey,
    model: settings.model || undefined,
    serviceTier: settings.serviceTier,
    reasoningEffort: settings.reasoningEffort
  };
}

function logBrokerRequestFailure(
  message: AiBrokerInternalRequest,
  settings: ExtensionSettings | null,
  error: unknown,
  fallbackAllowed: boolean
): void {
  const details = settings
    ? {
        operation: message.operation,
        backendBaseUrl: settings.backendBaseUrl,
        aiBrokerProvider: settings.aiBrokerProvider,
        fallbackAllowed,
        errorName: error instanceof Error ? error.name : '',
        errorMessage: error instanceof Error ? error.message : String(error),
        error
      }
    : {
        operation: message.operation,
        backendBaseUrl: '',
        aiBrokerProvider: '',
        fallbackAllowed,
        errorName: error instanceof Error ? error.name : '',
        errorMessage: error instanceof Error ? error.message : String(error),
        error
      };
  console.error('[Babel Gold Drafting] Helper AI broker request failed', details);
}


function isValidL0TargetRow(row: TranscriptRow): boolean {
  return Boolean(
    row &&
      typeof row.rowId === 'string' &&
      row.rowId.trim() &&
      typeof row.speakerKey === 'string' &&
      row.speakerKey.trim() &&
      typeof row.startSeconds === 'number' &&
      Number.isFinite(row.startSeconds) &&
      row.startSeconds >= 0 &&
      typeof row.endSeconds === 'number' &&
      Number.isFinite(row.endSeconds) &&
      row.endSeconds > row.startSeconds
  );
}

function captureCurrentCanonicalTaskId(): string {
  return buildCanonicalTaskIdentity(captureTranscriptJob());
}

export type L0SegmentGenerators = {
  remote: typeof generateL0SegmentDraft;
  local: typeof generateLocalL0SegmentDraft;
  mai: typeof generateMaiL0SegmentDraft;
};

const DEFAULT_L0_SEGMENT_GENERATORS: L0SegmentGenerators = {
  remote: generateL0SegmentDraft,
  local: generateLocalL0SegmentDraft,
  mai: generateMaiL0SegmentDraft
};

export async function generateConfiguredL0SegmentText(
  settings: ExtensionSettings,
  taskId: string,
  targetRow: TranscriptRow,
  generators: L0SegmentGenerators = DEFAULT_L0_SEGMENT_GENERATORS
): Promise<string> {
  if (settings.mode === 'simple') return generators.mai(settings, taskId, targetRow);
  if (!isBrowserLocalMode(settings)) return generators.remote(settings, taskId, targetRow);
  try {
    return await generators.local(settings, taskId, targetRow, []);
  } catch (error) {
    if (!(error instanceof LocalModelBridgeError) || error.code !== 'timing-unavailable') throw error;
    const job = captureTranscriptJob();
    if (buildCanonicalTaskIdentity(job) !== taskId) throw new Error('The task changed before its local acoustic cache could be recovered.');
    await recoverCurrentLocalL0Timing(settings, job);
    return generators.local(settings, taskId, targetRow, []);
  }
}

async function handleBrokerRequest(
  message: AiBrokerInternalRequest,
  emit?: (message: AiBrokerPortMessage) => void,
  isConnected: () => boolean = () => true
): Promise<AiBrokerResponse> {
  if (message.operation === 'enhanceAudio') {
    if (!emit || typeof message.taskId !== 'string' || !message.taskId.trim() ||
      Object.keys(message).some((key) => !['type', 'version', 'operation', 'requestId', 'taskId'].includes(key))) {
      return brokerError('invalid-request', 'Audio enhancement requires a streamed request with only the native taskId.', false);
    }
    const isCurrent = () => {
      if (!isConnected()) return false;
      try {
        const job = captureTranscriptJob();
        return job.taskScoped === true && job.jobId === message.taskId;
      } catch { return false; }
    };
    if (!isCurrent()) return brokerError('stale-task', 'The requested native audio task is no longer current.', false);
    try {
      emit({ type: 'event', event: 'capturing-audio', operation: 'enhanceAudio', message: 'Capturing both original native audio tracks.' });
      const tracks = await captureOriginalAudioTracksForEnhancement();
      if (!isCurrent()) return brokerError('stale-task', 'The native audio task changed during capture.', false);
      emit({ type: 'event', event: 'calling-backend', operation: 'enhanceAudio', message: 'Checking the pair cache and GPU: without usable WebGPU, Original audio is sent to a remote swarm worker.' });
      const result = await enhanceLocalAudio(message.taskId, tracks, {
        isCurrent,
        onAudioChunk: (chunk) => emit(chunk),
        onProgress: (progress) => emit({ type: 'event', event: 'enhancement-progress', operation: 'enhanceAudio', progress })
      });
      if (!isCurrent()) return brokerError('stale-task', 'The native audio task changed before enhancement completed.', false);
      return result;
    } catch (error) {
      if (error instanceof LocalModelBridgeError && error.code === 'stale-task') return brokerError('stale-task', error.message, false);
      throw new NoFallbackBrokerError(error);
    }
  }
  const settings = await loadSettings();

  if (settings.mode === 'local') {
    try {
      if (message.operation === 'redistributeText') {
        emit?.({ type: 'event', event: 'calling-backend', operation: message.operation, message: 'Explicit text alignment uses OpenRouter; task audio remains local.' });
        const response = await redistributeMaiText(settings, message.groups);
        if ((await loadSettings()).mode !== 'local') return brokerError('stale-task', 'The model mode changed during text alignment.', false);
        return { ok: true, provider: 'remote-openrouter', results: response.results, model: response.model };
      }
      if (message.operation !== 'transcribeSegment' && message.operation !== 'transcribeSegmentL0') {
        return brokerError('invalid-request', 'Unsupported local browser model operation.', false);
      }
      const job = captureTranscriptJob();
      const taskId = buildCanonicalTaskIdentity(job);
      const requested = message.operation === 'transcribeSegmentL0' ? message.row : message.segment;
      const known = job.rows.find((row) => row.rowId === requested.rowId);
      const row: TranscriptRow = { ...requested, processedRecordingId: known?.processedRecordingId, text: '', index: 0 };
      if (!isValidL0TargetRow(row) ||
          (message.operation === 'transcribeSegmentL0' && message.taskId !== taskId) ||
          (known && known.speakerKey.trim().toLowerCase() !== row.speakerKey.trim().toLowerCase())) {
        return brokerError('stale-task', 'The requested source or transcript task is no longer current.', false);
      }
      emit?.({ type: 'event', event: 'calling-backend', operation: message.operation, message: 'Using local GigaAM + C-denoise on WebGPU; no cloud audio fallback.' });
      await waitForCurrentL0Timing(job, settings);
      const generation = getCurrentL0TimingGeneration();
      const isCurrent = async () => captureCurrentCanonicalTaskId() === taskId &&
        (await loadSettings()).mode === 'local' && getCurrentL0TimingGeneration() === generation;
      if (!(await isCurrent())) return brokerError('stale-task', 'The task or model mode changed while preparing local timing.', false);
      const text = await generateConfiguredL0SegmentText(settings, taskId, row);
      if (!(await isCurrent())) return brokerError('stale-task', 'The task or model mode changed before local transcription completed.', false);
      return message.operation === 'transcribeSegmentL0'
        ? { ok: true, provider: 'local-l0', result: { text } }
        : { ok: true, provider: 'browser-local', text, model: 'GigaAM + C-denoise / WebGPU' };
    } catch (error) {
      throw new NoFallbackBrokerError(error);
    }
  }

  if (settings.mode === 'simple') {
    try {
      if (!settings.openRouterApiKey.trim()) {
        return brokerError('remote-not-configured', 'Simple mode requires an OpenRouter API key. Add your key in the Babel Gold Drafting extension options, then try again.', false);
      }
      if (message.operation === 'redistributeText') {
        emit?.({ type: 'event', event: 'calling-backend', operation: message.operation, message: 'Redistributing text with OpenRouter. This is a separate paid text-model request.' });
        const response = await withBrokerBackendProgress(
          message.operation, emit, () => redistributeMaiText(settings, message.groups), 'OpenRouter text redistribution'
        );
        if ((await loadSettings()).mode !== 'simple') {
          return brokerError('stale-task', 'The drafting mode changed before text redistribution completed.', false);
        }
        return { ok: true, provider: 'remote-openrouter', results: response.results, model: response.model };
      }
      if (message.operation !== 'transcribeSegmentL0' && message.operation !== 'transcribeSegment') {
        return brokerError('invalid-request', 'Unsupported Helper AI broker operation.', false);
      }
      const job = captureTranscriptJob();
      const taskId = buildCanonicalTaskIdentity(job);
      const requestedRow = message.operation === 'transcribeSegmentL0' ? message.row : message.segment;
      const knownRow = job.rows.find((row) => row.rowId === requestedRow.rowId);
      const targetRow: TranscriptRow = {
        ...requestedRow,
        processedRecordingId: knownRow?.processedRecordingId,
        text: '',
        index: 0
      };
      if (!isValidL0TargetRow(targetRow) ||
        (message.operation === 'transcribeSegmentL0' && message.taskId !== taskId) ||
        (knownRow && knownRow.speakerKey.trim().toLowerCase() !== targetRow.speakerKey.trim().toLowerCase()) ||
        (!job.taskScoped && !job.rows.some((row) => row.speakerKey.trim().toLowerCase() === targetRow.speakerKey.trim().toLowerCase()))) {
        return brokerError('stale-task', 'The requested source or transcript task is no longer current.', false);
      }
      targetRow.rowId = targetRow.rowId.trim();
      targetRow.speakerKey = targetRow.speakerKey.trim();
      emit?.({ type: 'event', event: 'calling-backend', operation: message.operation, message: 'Using MAI native transcription and timing; cached source audio results are reused.' });
      await waitForCurrentL0Timing(job, settings);
      const timingGeneration = getCurrentL0TimingGeneration();
      const isCurrent = async () => captureCurrentCanonicalTaskId() === taskId &&
        (await loadSettings()).mode === 'simple' && getCurrentL0TimingGeneration() === timingGeneration;
      if (!(await isCurrent())) return brokerError('stale-task', 'The transcript task or drafting mode changed while transcribing.', false);
      let text: string;
      try {
        text = await generateConfiguredL0SegmentText(settings, taskId, targetRow);
      } catch (error) {
        if (!(error instanceof MaiBridgeError) || error.code !== 'timing-unavailable') throw error;
        // Only an explicit Helper action can recover a missing session cache.
        await waitForCurrentL0Timing(job, settings);
        if (!(await isCurrent())) return brokerError('stale-task', 'The transcript task or drafting mode changed while transcribing.', false);
        text = await generateConfiguredL0SegmentText(settings, taskId, targetRow);
      }
      if (!(await isCurrent())) return brokerError('stale-task', 'The transcript task or drafting mode changed before transcription completed.', false);
      return message.operation === 'transcribeSegmentL0'
        ? { ok: true, provider: 'remote-openrouter', result: { text } }
        : { ok: true, provider: 'remote-openrouter', text, model: 'microsoft/mai-transcribe-2' };
    } catch (error) {
      // Failure policy belongs to the action's originating mode, even if the
      // user switches to Advanced while a cloud request is pending.
      throw new NoFallbackBrokerError(error);
    }
  }

  if (message.operation === 'transcribeSegmentL0') {
    if (
      typeof message.taskId !== 'string' ||
      !message.taskId.trim() ||
      !isValidL0TargetRow(message.row) ||
      captureCurrentCanonicalTaskId() !== message.taskId
    ) {
      return brokerError('stale-task', 'The requested transcript task is no longer current.', false);
    }
    if (emit && isBrowserLocalMode(settings)) {
      emit({ type: 'event', event: 'calling-backend', operation: message.operation, message: 'Using cached full-lane C-denoise WebGPU labels.' });
    }
    if (captureCurrentCanonicalTaskId() !== message.taskId) {
      return brokerError('stale-task', 'The transcript task changed before its cached segment could be read.', false);
    }
    const targetRow: TranscriptRow = {
      ...message.row,
      rowId: message.row.rowId.trim(),
      speakerKey: message.row.speakerKey.trim(),
      text: '',
      index: 0
    };
    await waitForCurrentL0Timing(captureTranscriptJob(), settings);
    if (emit) {
      emit({
        type: 'event',
        event: 'calling-backend',
        operation: message.operation,
        message: isBrowserLocalMode(settings)
          ? 'Running local browser models for the requested segment.'
          : 'Calling the local L0 drafting engine.'
      });
    }
    const text = await withBrokerBackendProgress(
      message.operation,
      emit,
      () => generateConfiguredL0SegmentText(settings, message.taskId, targetRow),
      isBrowserLocalMode(settings) ? 'local WebGPU C-denoise' : 'Gold Drafting backend'
    );
    if (captureCurrentCanonicalTaskId() !== message.taskId) {
      return brokerError('stale-task', 'The transcript task changed before L0 drafting completed.', false);
    }
    return {
      ok: true,
      provider: 'local-l0',
      result: { text }
    };
  }
  const fallbackAllowed = providerAllowsLocalFallback(settings.aiBrokerProvider);

  if (!shouldUseRemoteBroker(settings.aiBrokerProvider)) {
    return brokerError('provider-local-gemini-nano', 'Gold Drafting is configured to use local Gemini Nano for Helper AI.', true);
  }

  if (!settings.openRouterApiKey) {
    return brokerError('remote-not-configured', 'Gold Drafting OpenRouter API key is not configured.', fallbackAllowed);
  }

  if (message.operation === 'transcribeSegment') {
    if (emit) {
      emit({ type: 'event', event: 'capturing-audio', operation: message.operation, message: 'Capturing Babel segment audio.' });
    }
    const audioTracks = await captureAudioTracksForDrafting();
    if (emit) {
      emit({ type: 'event', event: 'calling-backend', operation: message.operation, message: 'Calling Gold Drafting backend.' });
    }
    const response = await withBrokerBackendProgress(
      message.operation,
      emit,
      () => transcribeSegmentWithBroker(
        settings.backendBaseUrl,
        {
          ...remoteRequestBase(settings),
          segment: message.segment
        },
        audioTracks
      )
    );
    return {
      ok: true,
      provider: 'remote-openrouter',
      text: response.text,
      model: response.model
    };
  }

  if (message.operation === 'redistributeText') {
    if (emit) {
      emit({ type: 'event', event: 'calling-backend', operation: message.operation, message: 'Calling Gold Drafting backend.' });
    }
    const response = await withBrokerBackendProgress(
      message.operation,
      emit,
      () => redistributeTextWithBroker(settings.backendBaseUrl, {
        ...remoteRequestBase(settings),
        groups: message.groups
      })
    );
    return {
      ok: true,
      provider: 'remote-openrouter',
      results: response.results,
      model: response.model
    };
  }

  return brokerError('invalid-request', 'Unsupported Helper AI broker operation.', fallbackAllowed);
}

export function publishGoldDraftingExtensionId(root: HTMLElement = document.documentElement): void {
  root.setAttribute(AI_BROKER_CONTENT_BUILD_ATTR, AI_BROKER_CONTENT_BUILD);
  const runtimeId = globalThis.chrome?.runtime?.id;
  if (runtimeId) {
    root.setAttribute(AI_BROKER_EXTENSION_ID_ATTR, runtimeId);
  }
}

async function brokerFailureResponse(
  message: AiBrokerInternalRequest,
  error: unknown
): Promise<Extract<AiBrokerResponse, { ok: false }>> {
  if (message.operation === 'enhanceAudio') {
    logBrokerRequestFailure(message, null, error, false);
    return brokerError('broker-error', error instanceof Error ? error.message : String(error), false);
  }
  let settings: ExtensionSettings | null = null;
  try {
    settings = await loadSettings();
  } catch {
    // Settings may also be unavailable while reporting the original request failure.
  }
  const fallbackAllowed = !(error instanceof NoFallbackBrokerError) && allowsBrokerFallback(message, settings);
  logBrokerRequestFailure(message, settings, error, fallbackAllowed);
  return brokerError('broker-error', error instanceof Error ? error.message : String(error), fallbackAllowed);
}

export function registerAiBrokerContentHandler(): void {
  const runtime = globalThis.chrome?.runtime;
  if (!runtime) {
    return;
  }

  if (runtime.onConnect?.addListener) {
    runtime.onConnect.addListener((port) => {
      if (port.name !== AI_BROKER_INTERNAL_PORT_NAME) {
        return;
      }
      let connected = true;
      port.onDisconnect.addListener(() => { connected = false; });

      port.onMessage.addListener((message: unknown) => {
        if (!isBrokerRequest(message)) {
          port.postMessage({
            type: 'error',
            response: brokerError('invalid-request', 'Invalid Helper AI broker tab port request.', true)
          });
          return;
        }

        const emit = (event: AiBrokerPortMessage) => {
          if (!connected) throw new NoFallbackBrokerError(new Error('The Helper audio stream disconnected.'));
          port.postMessage(event);
        };
        void handleBrokerRequest(message, emit, () => connected)
          .then((response) => {
            if (connected) port.postMessage({ type: 'result', response });
          })
          .catch(async (error) => {
            const response = await brokerFailureResponse(message, error);
            if (connected) port.postMessage({ type: 'error', response });
          });
      });
    });
  }

  if (!runtime.onMessage?.addListener) {
    return;
  }

  runtime.onMessage.addListener((message: unknown, _sender: chrome.runtime.MessageSender, sendResponse: (response: AiBrokerResponse) => void) => {
    if (!isBrokerRequest(message)) {
      return false;
    }

    void handleBrokerRequest(message)
      .then(sendResponse)
      .catch(async (error) => {
        sendResponse(await brokerFailureResponse(message, error));
      });
    return true;
  });
}
