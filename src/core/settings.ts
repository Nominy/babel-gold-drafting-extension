import { createSettingsStore } from '@nominy/babel-extension-frontend';
import type { ExtensionSettings } from './types';

export const SETTINGS_STORAGE_KEY = 'babel_gold_drafting_settings';

export const DEFAULT_SETTINGS: ExtensionSettings = {
  backendBaseUrl: 'https://reviewgen.ovh',
  projectPreset: 'ru-gold-2sp-v1',
  openRouterApiKey: '',
  model: 'google/gemini-3-flash-preview',
  serviceTier: 'flex',
  reasoningEffort: 'low',
  aiBrokerProvider: 'auto',
  audioInputEnabled: false
};

function getStorageArea() {
  const chromeApi = globalThis.chrome;
  return chromeApi?.storage?.local ?? null;
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
  const audioInputEnabled = raw.audioInputEnabled === true;

  return {
    backendBaseUrl,
    projectPreset,
    openRouterApiKey,
    model,
    serviceTier,
    reasoningEffort,
    aiBrokerProvider,
    audioInputEnabled
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
