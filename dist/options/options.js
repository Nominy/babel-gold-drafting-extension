"use strict";
(() => {
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
  async function loadSettings() {
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
  async function saveSettings(settings) {
    const normalized = normalizeSettings(settings);
    const storage = getStorageArea();
    if (!storage) {
      return normalized;
    }
    return new Promise((resolve) => {
      storage.set({ [SETTINGS_STORAGE_KEY]: normalized }, () => resolve(normalized));
    });
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
