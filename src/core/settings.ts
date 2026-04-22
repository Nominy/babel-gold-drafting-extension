import type { ExtensionSettings } from './types';

export const SETTINGS_STORAGE_KEY = 'babel_gold_drafting_settings';

export const DEFAULT_SETTINGS: ExtensionSettings = {
  backendBaseUrl: 'https://reviewgen.ovh',
  projectPreset: 'ru-gold-2sp-v1'
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

  return {
    backendBaseUrl,
    projectPreset
  };
}

export async function loadSettings(): Promise<ExtensionSettings> {
  const storage = getStorageArea();
  if (!storage) {
    return DEFAULT_SETTINGS;
  }

  return new Promise((resolve) => {
    storage.get(SETTINGS_STORAGE_KEY, (items) => {
      resolve(normalizeSettings(items?.[SETTINGS_STORAGE_KEY]));
    });
  });
}

export async function saveSettings(settings: ExtensionSettings): Promise<ExtensionSettings> {
  const normalized = normalizeSettings(settings);
  const storage = getStorageArea();
  if (!storage) {
    return normalized;
  }

  return new Promise((resolve) => {
    storage.set({ [SETTINGS_STORAGE_KEY]: normalized }, () => resolve(normalized));
  });
}
