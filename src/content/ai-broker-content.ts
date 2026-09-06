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
import { captureAudioTracksForDrafting } from '../core/audio-cues';
import { redistributeTextWithBroker, transcribeSegmentWithBroker } from '../core/backend-client';
import { generateL0SegmentDraft } from '../core/l0-client';
import { generateLocalL0SegmentDraft } from '../core/local-model-client';
import { prepareL0TimingTracks } from '../core/l0-timing-client';
import { loadSettings } from '../core/settings';
import { buildCanonicalTaskIdentity, captureTranscriptJob } from '../core/transcript';
import type { CapturedAudioTrack, ExtensionSettings, TranscriptRow } from '../core/types';

const AI_BROKER_CONTENT_BUILD = 'port-stream-postmortem-2026-06-23';
const AI_BROKER_CONTENT_BUILD_ATTR = 'data-babel-gold-drafting-ai-broker-build';
const BROKER_BACKEND_PROGRESS_INTERVAL_MS = 5000;

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
  if (message.operation === 'transcribeSegmentL0' && settings?.localModelsEnabled) {
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
};

const DEFAULT_L0_SEGMENT_GENERATORS: L0SegmentGenerators = {
  remote: generateL0SegmentDraft,
  local: generateLocalL0SegmentDraft
};

export async function generateConfiguredL0SegmentText(
  settings: ExtensionSettings,
  taskId: string,
  targetRow: TranscriptRow,
  audioTracks: CapturedAudioTrack[],
  generators: L0SegmentGenerators = DEFAULT_L0_SEGMENT_GENERATORS
): Promise<string> {
  const segmentJob = { jobId: taskId, rows: [targetRow] };
  const tracks = prepareL0TimingTracks(segmentJob, audioTracks);
  return settings.localModelsEnabled
    ? generators.local(settings, taskId, targetRow, tracks)
    : generators.remote(settings, taskId, targetRow, tracks);
}

async function handleBrokerRequest(
  message: AiBrokerInternalRequest,
  emit?: (message: AiBrokerPortMessage) => void
): Promise<AiBrokerResponse> {
  const settings = await loadSettings();

  if (message.operation === 'transcribeSegmentL0') {
    if (
      typeof message.taskId !== 'string' ||
      !message.taskId.trim() ||
      !isValidL0TargetRow(message.row) ||
      captureCurrentCanonicalTaskId() !== message.taskId
    ) {
      return brokerError('stale-task', 'The requested transcript task is no longer current.', false);
    }
    if (emit) {
      emit({ type: 'event', event: 'capturing-audio', operation: message.operation, message: 'Capturing Babel segment audio.' });
    }
    const audioTracks = await captureAudioTracksForDrafting();
    if (captureCurrentCanonicalTaskId() !== message.taskId) {
      return brokerError('stale-task', 'The transcript task changed while audio was being captured.', false);
    }
    const targetRow: TranscriptRow = {
      ...message.row,
      rowId: message.row.rowId.trim(),
      speakerKey: message.row.speakerKey.trim(),
      text: '',
      index: 0
    };
    if (emit) {
      emit({
        type: 'event',
        event: 'calling-backend',
        operation: message.operation,
        message: settings.localModelsEnabled
          ? 'Running local browser models for the requested segment.'
          : 'Calling the local L0 drafting engine.'
      });
    }
    const text = await withBrokerBackendProgress(
      message.operation,
      emit,
      () => generateConfiguredL0SegmentText(settings, message.taskId, targetRow, audioTracks),
      settings.localModelsEnabled ? 'local browser models' : 'Gold Drafting backend'
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
  let settings: ExtensionSettings | null = null;
  try {
    settings = await loadSettings();
  } catch {
    // Settings may also be unavailable while reporting the original request failure.
  }
  const fallbackAllowed = allowsBrokerFallback(message, settings);
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

      port.onMessage.addListener((message: unknown) => {
        if (!isBrokerRequest(message)) {
          port.postMessage({
            type: 'error',
            response: brokerError('invalid-request', 'Invalid Helper AI broker tab port request.', true)
          });
          return;
        }

        const emit = (event: AiBrokerPortMessage) => port.postMessage(event);
        void handleBrokerRequest(message, emit)
          .then((response) => {
            port.postMessage({ type: 'result', response });
          })
          .catch(async (error) => {
            const response = await brokerFailureResponse(message, error);
            port.postMessage({ type: 'error', response });
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
