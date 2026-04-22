"use strict";
(() => {
  // ../../shared/babel-extension-platform/packages/babel-extension-frontend/src/index.mjs
  function createSettingsStore(config) {
    const getStorageArea2 = config.getStorageArea ?? defaultGetStorageArea;
    return {
      async loadSettings() {
        const storage = getStorageArea2();
        const fallback = config.normalize(config.defaults);
        if (!storage) {
          return fallback;
        }
        return new Promise((resolve) => {
          storage.get(config.storageKey, (items) => {
            const runtime = globalThis.chrome?.runtime;
            if (runtime?.lastError) {
              resolve(fallback);
              return;
            }
            resolve(config.normalize(items?.[config.storageKey]));
          });
        });
      },
      async saveSettings(value) {
        const normalized = config.normalize(value);
        const storage = getStorageArea2();
        if (!storage) {
          return normalized;
        }
        return new Promise((resolve) => {
          storage.set({ [config.storageKey]: normalized }, () => {
            resolve(normalized);
          });
        });
      }
    };
  }
  function defaultGetStorageArea() {
    return globalThis.chrome?.storage?.local ?? null;
  }

  // src/core/settings.ts
  var SETTINGS_STORAGE_KEY = "babel_gold_drafting_settings";
  var DEFAULT_SETTINGS = {
    backendBaseUrl: "https://reviewgen.ovh",
    projectPreset: "ru-gold-2sp-v1"
  };
  function getStorageArea() {
    const chromeApi = globalThis.chrome;
    return chromeApi?.storage?.local ?? null;
  }
  function normalizeSettings(input) {
    const raw = input && typeof input === "object" ? input : {};
    const backendBaseUrl = typeof raw.backendBaseUrl === "string" && raw.backendBaseUrl.trim() ? raw.backendBaseUrl.trim().replace(/\/+$/, "") : DEFAULT_SETTINGS.backendBaseUrl;
    const projectPreset = raw.projectPreset === "ru-gold-2sp-v1" ? raw.projectPreset : DEFAULT_SETTINGS.projectPreset;
    return {
      backendBaseUrl,
      projectPreset
    };
  }
  var settingsStore = createSettingsStore({
    storageKey: SETTINGS_STORAGE_KEY,
    defaults: DEFAULT_SETTINGS,
    normalize: normalizeSettings,
    getStorageArea
  });
  async function loadSettings() {
    return settingsStore.loadSettings();
  }
  async function saveSettings(settings) {
    return settingsStore.saveSettings(settings);
  }

  // src/options/options.ts
  function requireElement(selector) {
    const element = document.querySelector(selector);
    if (!(element instanceof HTMLElement)) {
      throw new Error(`Missing required element: ${selector}`);
    }
    return element;
  }
  async function boot() {
    const backendBaseUrlInput = requireElement("#backendBaseUrl");
    const projectPresetSelect = requireElement("#projectPreset");
    const saveButton = requireElement('[data-role="save"]');
    const status = requireElement('[data-role="status"]');
    const settings = await loadSettings();
    backendBaseUrlInput.value = settings.backendBaseUrl;
    projectPresetSelect.value = settings.projectPreset;
    saveButton.addEventListener("click", () => {
      status.textContent = "Saving...";
      void saveSettings({
        backendBaseUrl: backendBaseUrlInput.value,
        projectPreset: projectPresetSelect.value === "ru-gold-2sp-v1" ? "ru-gold-2sp-v1" : "ru-gold-2sp-v1"
      }).then((saved) => {
        backendBaseUrlInput.value = saved.backendBaseUrl;
        projectPresetSelect.value = saved.projectPreset;
        status.textContent = "Saved. Reload Babel tabs to pick up the new settings.";
      }).catch((error) => {
        status.textContent = error instanceof Error ? error.message : String(error);
      });
    });
  }
  void boot();
})();
//# sourceMappingURL=options.js.map
