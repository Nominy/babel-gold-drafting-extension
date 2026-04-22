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
  const saveButton = requireElement<HTMLButtonElement>('[data-role="save"]');
  const status = requireElement<HTMLElement>('[data-role="status"]');

  const settings = await loadSettings();
  backendBaseUrlInput.value = settings.backendBaseUrl;
  projectPresetSelect.value = settings.projectPreset;

  saveButton.addEventListener('click', () => {
    status.textContent = 'Saving...';
    void saveSettings({
      backendBaseUrl: backendBaseUrlInput.value,
      projectPreset: projectPresetSelect.value === 'ru-gold-2sp-v1' ? 'ru-gold-2sp-v1' : 'ru-gold-2sp-v1'
    })
      .then((saved) => {
        backendBaseUrlInput.value = saved.backendBaseUrl;
        projectPresetSelect.value = saved.projectPreset;
        status.textContent = 'Saved. Reload Babel tabs to pick up the new settings.';
      })
      .catch((error) => {
        status.textContent = error instanceof Error ? error.message : String(error);
      });
  });
}

void boot();
