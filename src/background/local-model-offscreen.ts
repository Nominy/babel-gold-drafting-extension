import { getLocalModelStatus } from '../core/local-model-bundle';
import { LOCAL_MODEL_BASE_URL, SETTINGS_STORAGE_KEY, loadSettings } from '../core/settings';
import { isVolunteerMessage, type VolunteerMessage, type VolunteerStatus } from '../core/volunteer-protocol';
import {
  createLocalModelFailure,
  isLocalModelOffscreenRequest,
  isLocalModelOffscreenResponse,
  toOffscreenRequest,
  type LocalModelFailureResponse,
  type LocalModelOffscreenRequest,
  type LocalModelOffscreenResponse
} from '../core/local-model-offscreen-protocol';

const OFFSCREEN_DOCUMENT_PATH = 'offscreen.html';
const OFFSCREEN_JUSTIFICATION = 'Run opted-in local speech models in an extension-origin document.';

type OffscreenDocumentOptions = {
  url: string;
  reasons: chrome.offscreen.Reason[];
  justification: string;
};

export interface LocalModelOffscreenDependencies {
  hasDocument: () => Promise<boolean>;
  createDocument: (options: OffscreenDocumentOptions) => Promise<void>;
  closeDocument: () => Promise<void>;
  sendMessage: (message: LocalModelOffscreenRequest) => Promise<unknown>;
  workersReason: chrome.offscreen.Reason;
}

export function createLocalModelOffscreenBridge(dependencies: LocalModelOffscreenDependencies) {
  let creationPromise: Promise<void> | null = null;
  let recoveryPromise: Promise<void> | null = null;

  async function ensureDocument(): Promise<void> {
    if (await dependencies.hasDocument()) return;
    if (!creationPromise) {
      creationPromise = dependencies
        .createDocument({
          url: OFFSCREEN_DOCUMENT_PATH,
          reasons: [dependencies.workersReason],
          justification: OFFSCREEN_JUSTIFICATION
        })
        .catch(async (error) => {
          // Another extension context may have won a create race outside this service worker instance.
          if (!(await dependencies.hasDocument())) throw error;
        })
        .finally(() => {
          creationPromise = null;
        });
    }
    await creationPromise;
  }

  async function recoverDocument(): Promise<void> {
    if (!recoveryPromise) {
      recoveryPromise = (async () => {
        try {
          if (await dependencies.hasDocument()) await dependencies.closeDocument();
        } catch {
          // A crashed document can disappear between hasDocument and closeDocument.
        }
        creationPromise = null;
        await ensureDocument();
      })().finally(() => {
        recoveryPromise = null;
      });
    }
    await recoveryPromise;
  }

  async function forwardOnce(request: LocalModelOffscreenRequest): Promise<LocalModelOffscreenResponse> {
    await ensureDocument();
    const forwarded = toOffscreenRequest(request);
    const response = await dependencies.sendMessage(forwarded);
    if (!isLocalModelOffscreenResponse(response, request)) {
      throw new Error('The offscreen document returned an invalid or mismatched response.');
    }
    return response;
  }

  async function forwardRequest(request: LocalModelOffscreenRequest): Promise<LocalModelOffscreenResponse> {
    try {
      return await forwardOnce(request);
    } catch (firstError) {
      try {
        await recoverDocument();
        return await forwardOnce(request);
      } catch (recoveryError) {
        const firstDetail = firstError instanceof Error ? firstError.message : String(firstError);
        const recoveryDetail = recoveryError instanceof Error ? recoveryError.message : String(recoveryError);
        throw new Error(
          `Offscreen document failed and recovery did not succeed. Initial error: ${firstDetail}. Recovery error: ${recoveryDetail}`,
          { cause: recoveryError }
        );
      }
    }
  }

  async function handleRequest(request: LocalModelOffscreenRequest): Promise<LocalModelOffscreenResponse> {
    try {
      return await forwardRequest(request);
    } catch (error) {
      return createLocalModelFailure(request, 'offscreen-unavailable', error);
    }
  }

  return { ensureDocument, forwardRequest, handleRequest };
}
export interface VolunteerLifecycleDependencies {
  loadSettings: typeof loadSettings;
  ready: () => Promise<boolean>;
  hasDocument: () => Promise<boolean>;
  ensureDocument: () => Promise<void>;
  sendMessage: (message: VolunteerMessage) => Promise<VolunteerStatus>;
}

export function createVolunteerLifecycle(dependencies: VolunteerLifecycleDependencies) {
  let state: VolunteerStatus = { state: 'connecting' };
  let tail: Promise<void> = Promise.resolve();
  let enabled = false;

  function reconcile(): Promise<void> {
    const work = tail.then(async () => {
      try {
        const settings = await dependencies.loadSettings();
        enabled = settings.localModelsEnabled;
        if (!enabled || !(await dependencies.ready())) {
          state = { state: 'disabled', detail: enabled ? 'Local model bundle is not ready.' : undefined };
          if (await dependencies.hasDocument()) {
            await dependencies.sendMessage({ type: 'babel-l0-volunteer', target: 'offscreen', action: 'stop' });
          }
          return;
        }
        state = { state: 'connecting' };
        await dependencies.ensureDocument();
        state = await dependencies.sendMessage({ type: 'babel-l0-volunteer', target: 'offscreen', action: 'start' });
      } catch (error) {
        state = { state: 'error', detail: error instanceof Error ? error.message : String(error) };
      }
    });
    tail = work;
    return work;
  }

  async function status(): Promise<VolunteerStatus> {
    if (!enabled || state.state === 'disabled' || state.state === 'error') return state;
    try {
      if (!(await dependencies.hasDocument())) return { state: 'error', detail: 'Local model worker document is unavailable.' };
      return await dependencies.sendMessage({ type: 'babel-l0-volunteer', target: 'offscreen', action: 'status' });
    } catch (error) {
      return { state: 'error', detail: error instanceof Error ? error.message : String(error) };
    }
  }

  return { reconcile, status };
}

function getDefaultDependencies(): LocalModelOffscreenDependencies | null {
  const offscreen = globalThis.chrome?.offscreen;
  const runtime = globalThis.chrome?.runtime;
  if (!offscreen || typeof runtime?.sendMessage !== 'function') return null;
  return {
    hasDocument: () => offscreen.hasDocument(),
    createDocument: (options) => offscreen.createDocument(options),
    closeDocument: () => offscreen.closeDocument(),
    sendMessage: (message) => runtime.sendMessage(message),
    workersReason: chrome.offscreen.Reason.WORKERS
  };
}

const defaultDependencies = getDefaultDependencies();
if (defaultDependencies) {
  const bridge = createLocalModelOffscreenBridge(defaultDependencies);
  chrome.runtime.onMessage.addListener((message: unknown, _sender, sendResponse) => {
    if (!isLocalModelOffscreenRequest(message, 'background')) return false;
    void bridge.handleRequest(message).then(sendResponse).catch((error) => {
      const response: LocalModelFailureResponse = createLocalModelFailure(
        message,
        'offscreen-unavailable',
        error
      );
      sendResponse(response);
    });
    return true;
  });
  const volunteer = createVolunteerLifecycle({
    loadSettings,
    ready: async () => (await getLocalModelStatus(LOCAL_MODEL_BASE_URL)).state === 'ready',
    hasDocument: defaultDependencies.hasDocument,
    ensureDocument: bridge.ensureDocument,
    sendMessage: (message) => chrome.runtime.sendMessage(message)
  });
  chrome.runtime.onMessage.addListener((message: unknown, _sender, sendResponse) => {
    if (!isVolunteerMessage(message, 'background')) return false;
    void (message.action === 'settings' ? loadSettings() : volunteer.status())
      .then(sendResponse).catch((error) => {
        if (message.action === 'settings') sendResponse(null);
        else sendResponse({ state: 'error', detail: error instanceof Error ? error.message : String(error) });
      });
    return true;
  });
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area === 'local' && SETTINGS_STORAGE_KEY in changes) void volunteer.reconcile();
  });
  chrome.runtime.onStartup?.addListener(() => { void volunteer.reconcile(); });
  chrome.runtime.onInstalled?.addListener(() => { void volunteer.reconcile(); });
  void volunteer.reconcile();
}
