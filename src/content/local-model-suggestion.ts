import { ensureUiStyles, themeRoot, applyComponent } from '@nominy/babel-extension-frontend';
import { isBrowserLocalMode, loadSettings } from '../core/settings';
import type { ExtensionSettings } from '../core/types';
import { OPEN_LOCAL_MODEL_OPTIONS_MESSAGE_TYPE } from '../core/local-model-suggestion-protocol';

export const LOCAL_MODEL_SUGGESTION_STORAGE_KEY = 'babel_gold_local_model_suggestion_shown_v1';
const SUGGESTION_ID = 'babel-gold-local-model-suggestion';
const STYLE_ID = `${SUGGESTION_ID}-style`;
const NATIVE_SUGGESTION_HOST = '[data-transcript-scroll]';
const suggestions = new WeakMap<Document, {
  element: HTMLElement;
  markShown: () => Promise<void>;
  shown: Promise<void> | null;
}>();

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
      ((globalThis.navigator as GpuNavigator).gpu?.requestAdapter() ??
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

function createSuggestion(documentRef: Document, optionsUrl: string, openOptions: () => Promise<void>): HTMLElement {
  ensureUiStyles(documentRef);
  if (!documentRef.getElementById(STYLE_ID)) {
    const style = documentRef.createElement('style');
    style.id = STYLE_ID;
    style.textContent = `
      #${SUGGESTION_ID} { position: static; width: 100%; max-width: 100%; margin: 0; overflow-wrap: anywhere; }
      #${SUGGESTION_ID} .bui-button { max-width: 100%; white-space: normal; }
    `;
    (documentRef.head ?? documentRef.documentElement).append(style);
  }

  const suggestion = documentRef.createElement('aside');
  suggestion.id = SUGGESTION_ID;
  themeRoot(suggestion, 'purple');
  applyComponent(suggestion, 'card');
  suggestion.setAttribute('role', 'region');
  suggestion.setAttribute('aria-labelledby', `${SUGGESTION_ID}-title`);
  suggestion.setAttribute('aria-describedby', `${SUGGESTION_ID}-description`);

  const title = documentRef.createElement('strong');
  title.id = `${SUGGESTION_ID}-title`;
  applyComponent(title, 'title');
  title.textContent = 'Use your GPU for local AI';
  const copy = documentRef.createElement('p');
  copy.id = `${SUGGESTION_ID}-description`;
  applyComponent(copy, 'hint');
  copy.textContent =
    'Your browser supports GPU-accelerated local speech models. Set them up once to run word timing and drafting locally.';
  const actions = documentRef.createElement('div');
  actions.className = 'bgd-local-model-actions bui-row';
  const setupLink = documentRef.createElement('a');
  setupLink.href = optionsUrl;
  applyComponent(setupLink, 'button', { variant: 'primary' });
  setupLink.target = '_blank';
  setupLink.rel = 'noopener noreferrer';
  setupLink.textContent = 'Set up local AI';
  setupLink.addEventListener('click', (event) => {
    event.preventDefault();
    void openOptions();
  });
  const dismissButton = documentRef.createElement('button');
  dismissButton.type = 'button';
  applyComponent(dismissButton, 'button', { variant: 'ghost' });
  dismissButton.textContent = 'Not now';
  dismissButton.addEventListener('click', () => {
    suggestions.delete(documentRef);
    suggestion.remove();
  });
  actions.append(setupLink, dismissButton);
  suggestion.append(title, copy, actions);
  return suggestion;
}

/**
 * Reuse the page's DOM lifecycle to keep the undismissed suggestion in the
 * transcript's scroll flow, never above native Save or feedback controls.
 * A missing/replaced task host must not consume or dismiss the suggestion.
 */
export async function ensureLocalModelSuggestionMounted(documentRef: Document = document): Promise<boolean> {
  const suggestion = suggestions.get(documentRef);
  const host = documentRef.querySelector<HTMLElement>(NATIVE_SUGGESTION_HOST);
  if (!suggestion || !host) return false;
  if (suggestion.element.parentElement !== host) host.prepend(suggestion.element);
  try {
    suggestion.shown ??= suggestion.markShown();
    await suggestion.shown;
    return true;
  } catch {
    return false;
  }
}

export async function maybeShowLocalModelSuggestion(
  dependencies: LocalModelSuggestionDependencies = createDefaultDependencies()
): Promise<boolean> {
  try {
    if (await dependencies.wasShown()) return false;
    const settings = await dependencies.loadSettings();
    if (settings.mode === 'simple' || isBrowserLocalMode(settings)) return false;
    const adapter = await dependencies.requestGpuAdapter();
    if (!isSuitableLocalModelGpu(adapter)) return false;
    suggestions.set(dependencies.documentRef, {
      element: createSuggestion(dependencies.documentRef, dependencies.optionsUrl, dependencies.openOptions),
      markShown: dependencies.markShown,
      shown: null
    });
    return await ensureLocalModelSuggestionMounted(dependencies.documentRef);
  } catch {
    return false;
  }
}
