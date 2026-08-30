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
}
