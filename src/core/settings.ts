import { createSettingsStore } from '@nominy/babel-extension-frontend';
import type { ExtensionSettings } from './types';

export const SETTINGS_STORAGE_KEY = 'babel_gold_drafting_settings';

export const DEFAULT_SETTINGS: ExtensionSettings = {
  backendBaseUrl: 'https://reviewgen.ovh',
  projectPreset: 'ru-gold-2sp-v1',
  openRouterApiKey: '',
  model: 'google/gemini-3-flash-preview',
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
  const audioInputEnabled = raw.audioInputEnabled === true;

  return {
    backendBaseUrl,
    projectPreset,
    openRouterApiKey,
    model,
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
