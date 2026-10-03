import './local-model-offscreen';
import './l0-timing-token';
import './mai-broker';
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
import { getLocalModelStatus, getCachedBundleDescriptor, type LocalModelStatus } from '../core/local-model-bundle';
import { assertReleasedGraphs } from '../core/inference-release';
import { isBrowserLocalMode, LOCAL_MODEL_BASE_URL, loadSettings } from '../core/settings';
import { isOpenLocalModelOptionsMessage } from '../core/local-model-suggestion-protocol';
import type { ExtensionSettings } from '../core/types';

export async function resolveBrokerCapabilities(
  settings: ExtensionSettings,
  getStatus: (baseUrl: string) => Promise<LocalModelStatus> = getLocalModelStatus
) {
  if (settings.mode === 'simple') {
    const configured = Boolean(settings.openRouterApiKey.trim());
    return {
      transcribeSegment: configured,
      transcribeSegmentL0: configured,
      redistributeText: configured
    };
  }
  let transcribeSegmentL0 = true;
  if (isBrowserLocalMode(settings)) {
    try {
      const status = await getStatus(LOCAL_MODEL_BASE_URL);
      transcribeSegmentL0 = status.state === 'ready' && status.tested === true;
    } catch {
      transcribeSegmentL0 = false;
    }
  }
  const remoteConfigured = Boolean(settings.openRouterApiKey);
  const remoteBrokerAvailable = shouldUseRemoteBroker(settings.aiBrokerProvider) && remoteConfigured;
  return {
    transcribeSegment: settings.mode === 'local' ? transcribeSegmentL0 : remoteBrokerAvailable,
    transcribeSegmentL0,
    redistributeText: settings.mode === 'local' ? remoteConfigured : remoteBrokerAvailable
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
): Extract<AiBrokerResponse, { ok: false }> {
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
        )
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
        )
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
        )
      }
    );
  }
}

type BrokerAdmission =
  | { response: AiBrokerResponse }
  | { tabId: number; fallbackAllowed: boolean };

async function admitBrokerRequest(
  request: AiBrokerExternalRequest,
  sender: chrome.runtime.MessageSender | undefined,
  onAccepted?: () => void
): Promise<BrokerAdmission> {
  const settings = await loadSettings();
  const fallbackAllowed = settings.mode !== 'advanced' || request.operation === 'transcribeSegmentL0'
    ? false
    : providerAllowsLocalFallback(settings.aiBrokerProvider);
  const remoteConfigured = Boolean(settings.openRouterApiKey);
  onAccepted?.();

  if (request.operation === 'ping') {
    const capabilities = await resolveBrokerCapabilities(settings);
    return {
      response: {
        ok: true,
        provider: settings.mode === 'simple' ? 'remote-openrouter' : settings.aiBrokerProvider,
        remoteConfigured,
        capabilities
      }
    };
  }

  if (settings.mode === 'advanced' && request.operation !== 'transcribeSegmentL0' && !shouldUseRemoteBroker(settings.aiBrokerProvider)) {
    return { response: unavailable('provider-local-gemini-nano', 'Gold Drafting is configured to use local Gemini Nano.', true) };
  }

  if ((settings.mode === 'simple' || request.operation !== 'transcribeSegmentL0' && !(settings.mode === 'local' && request.operation === 'transcribeSegment')) && !remoteConfigured) {
    return { response: unavailable('remote-not-configured', 'Gold Drafting OpenRouter API key is not configured.', fallbackAllowed) };
  }
  if (settings.mode === 'local' && (request.operation === 'transcribeSegment' || request.operation === 'transcribeSegmentL0')) {
    const status = await getLocalModelStatus(LOCAL_MODEL_BASE_URL);
    if (status.state !== 'ready' || status.tested !== true) {
      return { response: unavailable('local-models-unavailable', 'Local WebGPU C-denoise is not ready. Download and test the dev model bundle in extension Options.', false) };
    }
  }

  const tabId = Number(sender?.tab?.id);
  if (!Number.isFinite(tabId)) {
    return { response: unavailable('missing-tab', 'Helper AI broker requests must originate from a Babel tab.', fallbackAllowed) };
  }

  return { tabId, fallbackAllowed };
}

async function requestFailure(request: AiBrokerExternalRequest, error: unknown): Promise<Extract<AiBrokerResponse, { ok: false }>> {
  let fallbackAllowed = false;
  try {
    const settings = await loadSettings();
    fallbackAllowed = settings.mode === 'advanced' &&
      request.operation !== 'transcribeSegmentL0' &&
      providerAllowsLocalFallback(settings.aiBrokerProvider);
  } catch {
    // Without a readable mode, never turn a failed paid request into another provider call.
  }
  return unavailable('broker-error', error instanceof Error ? error.message : String(error), fallbackAllowed);
}

async function handleBrokerRequest(
  request: AiBrokerExternalRequest,
  sender: chrome.runtime.MessageSender
): Promise<AiBrokerResponse> {
  const admission = await admitBrokerRequest(request, sender);
  return 'response' in admission
    ? admission.response
    : forwardToTab(admission.tabId, request, admission.fallbackAllowed);
}

async function handleBrokerPortRequest(
  request: AiBrokerExternalRequest,
  port: chrome.runtime.Port
): Promise<void> {
  const admission = await admitBrokerRequest(request, port.sender, () => {
    postPortMessage(port, {
      type: 'event',
      event: 'accepted',
      operation: request.operation,
      message: 'Gold Drafting AI broker accepted the request.'
    });
  });

  if ('response' in admission) {
    const { response } = admission;
    postPortMessage(port, response.ok ? { type: 'result', response } : { type: 'error', response });
    return;
  }

  forwardPortToTab(admission.tabId, request, admission.fallbackAllowed, port);
}

const externalMessageHandler = globalThis.chrome?.runtime?.onMessageExternal;
globalThis.chrome?.runtime?.onInstalled?.addListener(() => {
  void (async () => {
    const settings = await loadSettings();
    if (!isBrowserLocalMode(settings)) return;
    const bundle = await getCachedBundleDescriptor(LOCAL_MODEL_BASE_URL);
    if (bundle?.tested) {
      try { assertReleasedGraphs(bundle.files); return; } catch { /* Model setup is required. */ }
    }
    await chrome.runtime.openOptionsPage();
  })().catch(() => undefined);
});
if (externalMessageHandler && typeof externalMessageHandler.addListener === 'function') {
  externalMessageHandler.addListener((message, sender, sendResponse) => {
    if (!isBrokerRequest(message)) {
      return false;
    }

    void handleBrokerRequest(message, sender)
      .then(sendResponse)
      .catch(async (error) => sendResponse(await requestFailure(message, error)));
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
          response: unavailable('invalid-request', 'Invalid Helper AI broker port request.', true)
        });
        return;
      }

      void handleBrokerPortRequest(message, port).catch(async (error) => {
        postPortMessage(port, { type: 'error', response: await requestFailure(message, error) });
      });
    });
  });
}

const internalMessageHandler = globalThis.chrome?.runtime?.onMessage;
if (internalMessageHandler && typeof internalMessageHandler.addListener === 'function') {
  internalMessageHandler.addListener((message: unknown) => {
    if (!isOpenLocalModelOptionsMessage(message)) return false;
    void globalThis.chrome.tabs.create({
      url: globalThis.chrome.runtime.getURL('options.html#local-model-heading')
    });
    return false;
  });
}
