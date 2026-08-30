import { createSettingsStore } from '@nominy/babel-extension-frontend';
import type { ExtensionSettings } from './types';

export const SETTINGS_STORAGE_KEY = 'babel_gold_drafting_settings';
const DEFAULT_L0_CUSTOM_BASE_URL =
  'https://reviewgen.ovh/a3f73d6cf25fa138be653daaf2d7cd0702c0b2d69c40fb9eaee4e07d4b067dd5';
export const LOCAL_MODEL_BASE_URL = 'https://reviewgen.ovh/browser-model';
export const LOCAL_MODEL_SAMPLE_URL = `${LOCAL_MODEL_BASE_URL}/sample-russian-15s.wav`;


export const DEFAULT_SETTINGS: ExtensionSettings = {
  backendBaseUrl: 'https://reviewgen.ovh',
  projectPreset: 'ru-gold-2sp-v1',
  openRouterApiKey: '',
  model: 'google/gemini-3-flash-preview',
  serviceTier: 'flex',
  reasoningEffort: 'low',
  aiBrokerProvider: 'auto',
  l0ReplacementPreviewEnabled: true,
  l0CustomBaseUrl: DEFAULT_L0_CUSTOM_BASE_URL,
  l0DontRunLlm: false,
  audioInputEnabled: true,
  localModelsEnabled: false,
};

function getStorageArea() {
  const chromeApi = globalThis.chrome;
  return chromeApi?.storage?.local ?? null;
}
export function normalizeL0CustomBaseUrl(input: unknown): string {
  if (typeof input !== 'string' || !input.trim()) {
    return DEFAULT_L0_CUSTOM_BASE_URL;
  }

  const candidate = input.trim();
  try {
    const url = new URL(candidate);
    if (url.protocol !== 'http:' && url.protocol !== 'https:') {
      return DEFAULT_L0_CUSTOM_BASE_URL;
    }
    url.username = '';
    url.password = '';
    url.search = '';
    url.hash = '';
    url.pathname = url.pathname.replace(/\/+$/, '') || '/';
    return url.toString().replace(/\/+$/, '');
  } catch {
    return DEFAULT_L0_CUSTOM_BASE_URL;
  }
}



export function normalizeSettings(input: unknown): ExtensionSettings {
  const raw = input && typeof input === 'object' ? (input as Partial<ExtensionSettings>) : {};
  const backendBaseUrl =
    typeof raw.backendBaseUrl === 'string' && raw.backendBaseUrl.trim()
      ? raw.backendBaseUrl.trim().replace(/\/+$/, '')
      : DEFAULT_SETTINGS.backendBaseUrl;
  const projectPreset = raw.projectPreset === 'ru-gold-2sp-v1' ? raw.projectPreset : DEFAULT_SETTINGS.projectPreset;
  const openRouterApiKey = typeof raw.openRouterApiKey === 'string' ? raw.openRouterApiKey.trim() : DEFAULT_SETTINGS.openRouterApiKey;
  const model = typeof raw.model === 'string' && raw.model.trim() ? raw.model.trim() : DEFAULT_SETTINGS.model;
  const serviceTier =
    raw.serviceTier === 'default' || raw.serviceTier === 'priority' || raw.serviceTier === 'flex'
      ? raw.serviceTier
      : DEFAULT_SETTINGS.serviceTier;
  const reasoningEffort =
    raw.reasoningEffort === 'default' ||
    raw.reasoningEffort === 'none' ||
    raw.reasoningEffort === 'minimal' ||
    raw.reasoningEffort === 'low' ||
    raw.reasoningEffort === 'medium' ||
    raw.reasoningEffort === 'high' ||
    raw.reasoningEffort === 'xhigh'
      ? raw.reasoningEffort
      : DEFAULT_SETTINGS.reasoningEffort;
  const aiBrokerProvider =
    raw.aiBrokerProvider === 'auto' ||
    raw.aiBrokerProvider === 'remote-openrouter' ||
    raw.aiBrokerProvider === 'local-gemini-nano'
      ? raw.aiBrokerProvider
      : DEFAULT_SETTINGS.aiBrokerProvider;
  const l0ReplacementPreviewEnabled =
    typeof raw.l0ReplacementPreviewEnabled === 'boolean'
      ? raw.l0ReplacementPreviewEnabled
      : DEFAULT_SETTINGS.l0ReplacementPreviewEnabled;
  const l0CustomBaseUrl = normalizeL0CustomBaseUrl(raw.l0CustomBaseUrl);
  const l0DontRunLlm = raw.l0DontRunLlm === true;
  const audioInputEnabled =
    typeof raw.audioInputEnabled === 'boolean'
      ? raw.audioInputEnabled
      : DEFAULT_SETTINGS.audioInputEnabled;
  const localModelsEnabled = raw.localModelsEnabled === true;

  return {
    backendBaseUrl,
    projectPreset,
    openRouterApiKey,
    model,
    serviceTier,
    reasoningEffort,
    aiBrokerProvider,
    l0ReplacementPreviewEnabled,
    l0CustomBaseUrl,
    l0DontRunLlm,
    audioInputEnabled,
    localModelsEnabled
  };
}

const settingsStore = createSettingsStore<ExtensionSettings>({
  storageKey: SETTINGS_STORAGE_KEY,
  defaults: DEFAULT_SETTINGS,
  normalize: normalizeSettings,
  getStorageArea
});

export async function loadSettings(): Promise<ExtensionSettings> {
  return settingsStore.loadSettings();
}

export async function saveSettings(settings: ExtensionSettings): Promise<ExtensionSettings> {
  return settingsStore.saveSettings(settings);
}
