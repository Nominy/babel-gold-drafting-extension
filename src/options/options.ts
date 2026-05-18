import { loadSettings, saveSettings } from '../core/settings';

function requireElement<T extends HTMLElement>(selector: string): T {
  const element = document.querySelector(selector);
  if (!(element instanceof HTMLElement)) {
    throw new Error(`Missing required element: ${selector}`);
  }
  return element as T;
}

async function boot(): Promise<void> {
  const backendBaseUrlInput = requireElement<HTMLInputElement>('#backendBaseUrl');
  const projectPresetSelect = requireElement<HTMLSelectElement>('#projectPreset');
  const openRouterApiKeyInput = requireElement<HTMLInputElement>('#openRouterApiKey');
  const modelInput = requireElement<HTMLInputElement>('#model');
  const serviceTierSelect = requireElement<HTMLSelectElement>('#serviceTier');
  const audioInputEnabledInput = requireElement<HTMLInputElement>('#audioInputEnabled');
  const saveButton = requireElement<HTMLButtonElement>('[data-role="save"]');
  const status = requireElement<HTMLElement>('[data-role="status"]');

  const settings = await loadSettings();
  backendBaseUrlInput.value = settings.backendBaseUrl;
  projectPresetSelect.value = settings.projectPreset;
  openRouterApiKeyInput.value = settings.openRouterApiKey;
  modelInput.value = settings.model;
  serviceTierSelect.value = settings.serviceTier;
  audioInputEnabledInput.checked = settings.audioInputEnabled;

  saveButton.addEventListener('click', () => {
    status.textContent = 'Saving...';
    void saveSettings({
      backendBaseUrl: backendBaseUrlInput.value,
      projectPreset: projectPresetSelect.value === 'ru-gold-2sp-v1' ? 'ru-gold-2sp-v1' : 'ru-gold-2sp-v1',
      openRouterApiKey: openRouterApiKeyInput.value,
      model: modelInput.value,
      serviceTier:
        serviceTierSelect.value === 'default' || serviceTierSelect.value === 'priority' || serviceTierSelect.value === 'flex'
          ? serviceTierSelect.value
          : 'flex',
      audioInputEnabled: audioInputEnabledInput.checked
    })
      .then((saved) => {
        backendBaseUrlInput.value = saved.backendBaseUrl;
        projectPresetSelect.value = saved.projectPreset;
        openRouterApiKeyInput.value = saved.openRouterApiKey;
        modelInput.value = saved.model;
        serviceTierSelect.value = saved.serviceTier;
        audioInputEnabledInput.checked = saved.audioInputEnabled;
        status.textContent = 'Saved. Reload Babel tabs to pick up the new settings.';
      })
      .catch((error) => {
        status.textContent = error instanceof Error ? error.message : String(error);
      });
  });
}

void boot();
