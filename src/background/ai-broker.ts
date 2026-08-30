import './local-model-offscreen';
import {
  AI_BROKER_EXTERNAL_MESSAGE_TYPE,
  AI_BROKER_INTERNAL_MESSAGE_TYPE,
  AI_BROKER_INTERNAL_PORT_NAME,
  AI_BROKER_PORT_NAME,
  providerAllowsLocalFallback,
  shouldUseRemoteBroker,
  type AiBrokerExternalRequest,
  type AiBrokerInternalRequest,
  type AiBrokerPortMessage,
  type AiBrokerResponse
} from '../core/ai-broker-protocol';
import { getLocalModelStatus, type LocalModelStatus } from '../core/local-model-bundle';
import { LOCAL_MODEL_BASE_URL, loadSettings } from '../core/settings';
import type { ExtensionSettings } from '../core/types';

export async function resolveBrokerCapabilities(
  settings: ExtensionSettings,
  getStatus: (baseUrl: string) => Promise<LocalModelStatus> = getLocalModelStatus
) {
  let transcribeSegmentL0 = true;
  if (settings.localModelsEnabled) {
    try {
      transcribeSegmentL0 = (await getStatus(LOCAL_MODEL_BASE_URL)).state === 'ready';
    } catch {
      transcribeSegmentL0 = false;
    }
  }
  const remoteConfigured = Boolean(settings.openRouterApiKey);
  const remoteBrokerAvailable = shouldUseRemoteBroker(settings.aiBrokerProvider) && remoteConfigured;
  return {
    transcribeSegment: remoteBrokerAvailable,
    transcribeSegmentL0,
    redistributeText: remoteBrokerAvailable
  };
}

function isBrokerRequest(message: unknown): message is AiBrokerExternalRequest {
  return (
    Boolean(message && typeof message === 'object') &&
    (message as { type?: unknown }).type === AI_BROKER_EXTERNAL_MESSAGE_TYPE &&
    (message as { version?: unknown }).version === 1
  );
}

function unavailable(
  reason: Extract<AiBrokerResponse, { ok: false }>['reason'],
  message: string,
  fallbackAllowed: boolean
): AiBrokerResponse {
  return {
    ok: false,
    reason,
    message,
    fallbackAllowed
  };
}

function toInternalRequest(request: AiBrokerExternalRequest): AiBrokerInternalRequest {
  return {
    ...request,
    type: AI_BROKER_INTERNAL_MESSAGE_TYPE
  };
}

function postPortMessage(port: chrome.runtime.Port, message: AiBrokerPortMessage): void {
  try {
    port.postMessage(message);
  } catch (_error) {
    // The external Helper port may have closed while the tab request was still running.
  }
}

async function forwardToTab(
  tabId: number,
  request: AiBrokerExternalRequest,
  fallbackAllowed: boolean
): Promise<AiBrokerResponse> {
  try {
    return await chrome.tabs.sendMessage(tabId, toInternalRequest(request));
  } catch (error) {
    return unavailable(
      'tab-broker-unavailable',
      error instanceof Error ? error.message : String(error),
      fallbackAllowed
    );
  }
}

function forwardPortToTab(
  tabId: number,
  request: AiBrokerExternalRequest,
  fallbackAllowed: boolean,
  port: chrome.runtime.Port
): void {
  let settled = false;
  let tabPort: chrome.runtime.Port | null = null;

  try {
    tabPort = chrome.tabs.connect(tabId, { name: AI_BROKER_INTERNAL_PORT_NAME });
  } catch (error) {
    postPortMessage(
      port,
      {
        type: 'error',
        response: unavailable(
          'tab-broker-unavailable',
          error instanceof Error ? error.message : String(error),
          fallbackAllowed
        ) as Extract<AiBrokerResponse, { ok: false }>
      }
    );
    return;
  }

  tabPort.onMessage.addListener((message: AiBrokerPortMessage) => {
    try {
      port.postMessage(message);
    } catch (_error) {
      settled = true;
      try {
        tabPort?.disconnect();
      } catch (_disconnectError) {
        // Chrome already closed the tab port.
      }
      return;
    }
    if (message.type === 'result' || message.type === 'error') {
      settled = true;
      try {
        tabPort?.disconnect();
      } catch (_error) {
        // Chrome already closed the tab port.
      }
    }
  });

  tabPort.onDisconnect.addListener(() => {
    if (settled) {
      return;
    }
    settled = true;
    postPortMessage(
      port,
      {
        type: 'error',
        response: unavailable(
          'tab-broker-unavailable',
          chrome.runtime.lastError?.message || 'Gold Drafting tab AI broker disconnected before returning a result.',
          fallbackAllowed
        ) as Extract<AiBrokerResponse, { ok: false }>
      }
    );
  });

  port.onDisconnect.addListener(() => {
    settled = true;
    try {
      tabPort?.disconnect();
    } catch (_error) {
      // Chrome already closed the tab port.
    }
  });

  try {
    tabPort.postMessage(toInternalRequest(request));
  } catch (error) {
    settled = true;
    postPortMessage(
      port,
      {
        type: 'error',
        response: unavailable(
          'tab-broker-unavailable',
          error instanceof Error ? error.message : String(error),
          fallbackAllowed
        ) as Extract<AiBrokerResponse, { ok: false }>
      }
    );
  }
}

async function handleBrokerRequest(
  request: AiBrokerExternalRequest,
  sender: chrome.runtime.MessageSender
): Promise<AiBrokerResponse> {
  const settings = await loadSettings();
  const fallbackAllowed = request.operation === 'transcribeSegmentL0'
    ? false
    : providerAllowsLocalFallback(settings.aiBrokerProvider);
  const remoteConfigured = Boolean(settings.openRouterApiKey);

  if (request.operation === 'ping') {
    const capabilities = await resolveBrokerCapabilities(settings);
    return {
      ok: true,
      provider: settings.aiBrokerProvider,
      remoteConfigured,
      capabilities
    };
  }

  if (request.operation !== 'transcribeSegmentL0' && !shouldUseRemoteBroker(settings.aiBrokerProvider)) {
    return unavailable('provider-local-gemini-nano', 'Gold Drafting is configured to use local Gemini Nano.', true);
  }

  if (request.operation !== 'transcribeSegmentL0' && !remoteConfigured) {
    return unavailable('remote-not-configured', 'Gold Drafting OpenRouter API key is not configured.', fallbackAllowed);
  }

  const tabId = Number(sender.tab?.id);
  if (!Number.isFinite(tabId)) {
    return unavailable('missing-tab', 'Helper AI broker requests must originate from a Babel tab.', fallbackAllowed);
  }

  return forwardToTab(tabId, request, fallbackAllowed);
}

async function handleBrokerPortRequest(
  request: AiBrokerExternalRequest,
  port: chrome.runtime.Port
): Promise<void> {
  const settings = await loadSettings();
  const fallbackAllowed = request.operation === 'transcribeSegmentL0'
    ? false
    : providerAllowsLocalFallback(settings.aiBrokerProvider);
  const remoteConfigured = Boolean(settings.openRouterApiKey);

  postPortMessage(port, {
    type: 'event',
    event: 'accepted',
    operation: request.operation,
    message: 'Gold Drafting AI broker accepted the request.'
  });

  if (request.operation === 'ping') {
    const capabilities = await resolveBrokerCapabilities(settings);
    postPortMessage(port, {
      type: 'result',
      response: {
        ok: true,
        provider: settings.aiBrokerProvider,
        remoteConfigured,
        capabilities
      }
    });
    return;
  }

  if (request.operation !== 'transcribeSegmentL0' && !shouldUseRemoteBroker(settings.aiBrokerProvider)) {
    postPortMessage(port, {
      type: 'error',
      response: unavailable('provider-local-gemini-nano', 'Gold Drafting is configured to use local Gemini Nano.', true) as Extract<AiBrokerResponse, { ok: false }>
    });
    return;
  }

  if (request.operation !== 'transcribeSegmentL0' && !remoteConfigured) {
    postPortMessage(port, {
      type: 'error',
      response: unavailable('remote-not-configured', 'Gold Drafting OpenRouter API key is not configured.', fallbackAllowed) as Extract<AiBrokerResponse, { ok: false }>
    });
    return;
  }

  const tabId = Number(port.sender?.tab?.id);
  if (!Number.isFinite(tabId)) {
    postPortMessage(port, {
      type: 'error',
      response: unavailable('missing-tab', 'Helper AI broker requests must originate from a Babel tab.', fallbackAllowed) as Extract<AiBrokerResponse, { ok: false }>
    });
    return;
  }

  forwardPortToTab(tabId, request, fallbackAllowed, port);
}

const externalMessageHandler = globalThis.chrome?.runtime?.onMessageExternal;
if (externalMessageHandler && typeof externalMessageHandler.addListener === 'function') {
  externalMessageHandler.addListener((message, sender, sendResponse) => {
    if (!isBrokerRequest(message)) {
      return false;
    }

    void handleBrokerRequest(message, sender)
      .then(sendResponse)
      .catch((error) =>
        sendResponse(
          unavailable('broker-error', error instanceof Error ? error.message : String(error), true)
        )
      );
    return true;
  });
}

const externalConnectHandler = globalThis.chrome?.runtime?.onConnectExternal;
if (externalConnectHandler && typeof externalConnectHandler.addListener === 'function') {
  externalConnectHandler.addListener((port) => {
    if (port.name !== AI_BROKER_PORT_NAME) {
      return;
    }

    port.onMessage.addListener((message: unknown) => {
      if (!isBrokerRequest(message)) {
        postPortMessage(port, {
          type: 'error',
          response: unavailable('invalid-request', 'Invalid Helper AI broker port request.', true) as Extract<AiBrokerResponse, { ok: false }>
        });
        return;
      }

      void handleBrokerPortRequest(message, port).catch((error) => {
        postPortMessage(port, {
          type: 'error',
          response: unavailable('broker-error', error instanceof Error ? error.message : String(error), true) as Extract<AiBrokerResponse, { ok: false }>
        });
      });
    });
  });
}
