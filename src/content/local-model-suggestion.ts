import { loadSettings } from '../core/settings';
import type { ExtensionSettings } from '../core/types';
import { OPEN_LOCAL_MODEL_OPTIONS_MESSAGE_TYPE } from '../core/local-model-suggestion-protocol';

export const LOCAL_MODEL_SUGGESTION_STORAGE_KEY = 'babel_gold_local_model_suggestion_shown_v1';
const SUGGESTION_ID = 'babel-gold-local-model-suggestion';
const STYLE_ID = `${SUGGESTION_ID}-style`;

export type GpuAdapter = {
  info?: { isFallbackAdapter?: boolean };
  limits?: {
    maxBufferSize?: number;
    maxStorageBufferBindingSize?: number;
  };
};

type GpuNavigator = Navigator & {
  gpu?: {
    requestAdapter(options?: { powerPreference?: 'high-performance' | 'low-power' }): Promise<GpuAdapter | null>;
  };
};
export interface LocalModelSuggestionDependencies {
  documentRef: Document;
  loadSettings: () => Promise<ExtensionSettings>;
  wasShown: () => Promise<boolean>;
  markShown: () => Promise<void>;
  requestGpuAdapter: () => Promise<GpuAdapter | null>;
  optionsUrl: string;
  openOptions: () => Promise<void>;
}

export function isSuitableLocalModelGpu(adapter: GpuAdapter | null): boolean {
  if (!adapter || adapter.info?.isFallbackAdapter === true) return false;
  return (
    Number(adapter.limits?.maxBufferSize) >= 256 * 1024 * 1024 &&
    Number(adapter.limits?.maxStorageBufferBindingSize) >= 128 * 1024 * 1024
  );
}

function getStorageArea(): chrome.storage.StorageArea | null {
  return globalThis.chrome?.storage?.local ?? null;
}

function createDefaultDependencies(): LocalModelSuggestionDependencies {
  return {
    documentRef: document,
    loadSettings,
    wasShown: async () => {
      const storage = getStorageArea();
      if (!storage) return true;
      const stored = await storage.get(LOCAL_MODEL_SUGGESTION_STORAGE_KEY);
      return stored[LOCAL_MODEL_SUGGESTION_STORAGE_KEY] === true;
    },
    markShown: async () => {
      const storage = getStorageArea();
      if (!storage) throw new Error('chrome.storage.local is unavailable');
      await storage.set({ [LOCAL_MODEL_SUGGESTION_STORAGE_KEY]: true });
    },
    requestGpuAdapter: () =>
      ((globalThis.navigator as GpuNavigator).gpu?.requestAdapter({ powerPreference: 'high-performance' }) ??
        Promise.resolve(null)),
    optionsUrl:
      globalThis.chrome?.runtime?.getURL('options.html#local-model-heading') ?? '#local-model-heading',
    openOptions: async () => {
      await globalThis.chrome.runtime.sendMessage({
        type: OPEN_LOCAL_MODEL_OPTIONS_MESSAGE_TYPE
      });
    }
  };
}

function renderSuggestion(documentRef: Document, optionsUrl: string, openOptions: () => Promise<void>): void {
  if (documentRef.getElementById(SUGGESTION_ID)) return;
  if (!documentRef.getElementById(STYLE_ID)) {
    const style = documentRef.createElement('style');
    style.id = STYLE_ID;
    style.textContent = `
      #${SUGGESTION_ID} {
        position: fixed;
        right: 18px;
        bottom: 18px;
        z-index: 2147483647;
        width: min(340px, calc(100vw - 36px));
        box-sizing: border-box;
        padding: 18px;
        border: 1px solid rgba(96, 165, 250, 0.45);
        border-radius: 14px;
        background: #0f172a;
        color: #f8fafc;
        box-shadow: 0 18px 48px rgba(15, 23, 42, 0.35);
        font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }
      #${SUGGESTION_ID} strong { display: block; margin-bottom: 6px; font-size: 16px; }
      #${SUGGESTION_ID} p { margin: 0 0 14px; color: #cbd5e1; }
      #${SUGGESTION_ID} .bgd-local-model-actions { display: flex; align-items: center; gap: 12px; }
      #${SUGGESTION_ID} a { padding: 8px 12px; border-radius: 8px; background: #2563eb; color: white; font-weight: 700; text-decoration: none; }
      #${SUGGESTION_ID} button { border: 0; background: transparent; color: #cbd5e1; cursor: pointer; font: inherit; }
    `;
    (documentRef.head ?? documentRef.documentElement).append(style);
  }

  const suggestion = documentRef.createElement('aside');
  suggestion.id = SUGGESTION_ID;
  suggestion.setAttribute('role', 'dialog');
  suggestion.setAttribute('aria-labelledby', `${SUGGESTION_ID}-title`);

  const title = documentRef.createElement('strong');
  title.id = `${SUGGESTION_ID}-title`;
  title.textContent = 'Use your GPU for local AI';
  const copy = documentRef.createElement('p');
  copy.textContent =
    'Your browser supports GPU-accelerated local speech models. Set them up once to run word timing and drafting locally.';
  const actions = documentRef.createElement('div');
  actions.className = 'bgd-local-model-actions';
  const setupLink = documentRef.createElement('a');
  setupLink.href = optionsUrl;
  setupLink.target = '_blank';
  setupLink.rel = 'noopener noreferrer';
  setupLink.textContent = 'Set up local AI';
  setupLink.addEventListener('click', (event) => {
    event.preventDefault();
    void openOptions();
  });
  const dismissButton = documentRef.createElement('button');
  dismissButton.type = 'button';
  dismissButton.textContent = 'Not now';
  dismissButton.addEventListener('click', () => suggestion.remove());
  actions.append(setupLink, dismissButton);
  suggestion.append(title, copy, actions);
  (documentRef.body ?? documentRef.documentElement).append(suggestion);
}

export async function maybeShowLocalModelSuggestion(
  dependencies: LocalModelSuggestionDependencies = createDefaultDependencies()
): Promise<boolean> {
  try {
    if (await dependencies.wasShown()) return false;
    const settings = await dependencies.loadSettings();
    if (settings.localModelsEnabled) return false;
    const adapter = await dependencies.requestGpuAdapter();
    if (!isSuitableLocalModelGpu(adapter)) return false;
    await dependencies.markShown();
    renderSuggestion(dependencies.documentRef, dependencies.optionsUrl, dependencies.openOptions);
    return true;
  } catch {
    return false;
  }
}
