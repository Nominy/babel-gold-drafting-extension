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
import { loadSettings } from '../core/settings';
import type { ExtensionSettings } from '../core/types';

const AI_BROKER_CONTENT_BUILD = 'port-stream-postmortem-2026-06-23';
const AI_BROKER_CONTENT_BUILD_ATTR = 'data-babel-gold-drafting-ai-broker-build';
const BROKER_BACKEND_PROGRESS_INTERVAL_MS = 5000;

function brokerError(reason: Extract<AiBrokerResponse, { ok: false }>['reason'], message: string, fallbackAllowed: boolean): AiBrokerResponse {
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

async function withBrokerBackendProgress<T>(
  operation: AiBrokerInternalRequest['operation'],
  emit: ((message: AiBrokerPortMessage) => void) | undefined,
  request: () => Promise<T>
): Promise<T> {
  const startedAt = Date.now();
  const intervalId = emit
    ? setInterval(() => {
        const elapsedMs = Math.max(0, Date.now() - startedAt);
        emit({ type: 'event', event: 'backend-waiting', operation, elapsedMs, message: 'Still waiting for Gold Drafting backend after ' + formatElapsedSeconds(elapsedMs) + '.' });
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

async function handleBrokerRequest(
  message: AiBrokerInternalRequest,
  emit?: (message: AiBrokerPortMessage) => void
): Promise<AiBrokerResponse> {
  const settings = await loadSettings();
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
          .catch((error) => {
            void loadSettings()
              .then((settings) => {
                logBrokerRequestFailure(
                  message,
                  settings,
                  error,
                  providerAllowsLocalFallback(settings.aiBrokerProvider)
                );
                port.postMessage({
                  type: 'error',
                  response: brokerError(
                    'broker-error',
                    error instanceof Error ? error.message : String(error),
                    providerAllowsLocalFallback(settings.aiBrokerProvider)
                  )
                });
              })
              .catch(() => {
                logBrokerRequestFailure(message, null, error, true);
                port.postMessage({
                  type: 'error',
                  response: brokerError('broker-error', error instanceof Error ? error.message : String(error), true)
                });
              });
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
      .catch((error) => {
        void loadSettings()
          .then((settings) => {
            logBrokerRequestFailure(
              message,
              settings,
              error,
              providerAllowsLocalFallback(settings.aiBrokerProvider)
            );
            sendResponse(
              brokerError(
                'broker-error',
                error instanceof Error ? error.message : String(error),
                providerAllowsLocalFallback(settings.aiBrokerProvider)
              )
            );
          })
          .catch(() => {
            logBrokerRequestFailure(message, null, error, true);
            sendResponse(
              brokerError('broker-error', error instanceof Error ? error.message : String(error), true)
            );
          });
      });
    return true;
  });
}
