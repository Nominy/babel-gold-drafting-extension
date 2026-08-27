import { loadSettings, normalizeL0CustomBaseUrl, saveSettings } from '../core/settings';

function requireElement<T extends HTMLElement>(selector: string): T {
  const element = document.querySelector(selector);
  if (!(element instanceof HTMLElement)) {
    throw new Error(`Missing required element: ${selector}`);
  }
  return element as T;
}
async function requestCustomHostPermission(baseUrl: string): Promise<void> {
  if (!globalThis.chrome?.permissions?.request) {
    return;
  }
  const originPattern = `${new URL(baseUrl).origin}/*`;
  const granted = await chrome.permissions.request({ origins: [originPattern] });
  if (!granted) {
    throw new Error(`Host access is required for the custom L0 endpoint: ${originPattern}`);
  }
}


async function boot(): Promise<void> {
  const backendBaseUrlInput = requireElement<HTMLInputElement>('#backendBaseUrl');
  const projectPresetSelect = requireElement<HTMLSelectElement>('#projectPreset');
  const openRouterApiKeyInput = requireElement<HTMLInputElement>('#openRouterApiKey');
  const modelInput = requireElement<HTMLInputElement>('#model');
  const serviceTierSelect = requireElement<HTMLSelectElement>('#serviceTier');
  const reasoningEffortSelect = requireElement<HTMLSelectElement>('#reasoningEffort');
  const aiBrokerProviderSelect = requireElement<HTMLSelectElement>('#aiBrokerProvider');
  const l0ReplacementPreviewEnabledInput = requireElement<HTMLInputElement>('#l0ReplacementPreviewEnabled');
  const l0ReplacementSettings = requireElement<HTMLElement>('[data-role="l0-replacement-settings"]');
  const l0CustomBaseUrlInput = requireElement<HTMLInputElement>('#l0CustomBaseUrl');
  const l0DontRunLlmInput = requireElement<HTMLInputElement>('#l0DontRunLlm');
  const audioInputEnabledInput = requireElement<HTMLInputElement>('#audioInputEnabled');
  const saveButton = requireElement<HTMLButtonElement>('[data-role="save"]');
  const status = requireElement<HTMLElement>('[data-role="status"]');
  const renderL0ReplacementSettings = (): void => {
    l0ReplacementSettings.hidden = !l0ReplacementPreviewEnabledInput.checked;
  };

  const settings = await loadSettings();
  backendBaseUrlInput.value = settings.backendBaseUrl;
  projectPresetSelect.value = settings.projectPreset;
  openRouterApiKeyInput.value = settings.openRouterApiKey;
  modelInput.value = settings.model;
  serviceTierSelect.value = settings.serviceTier;
  reasoningEffortSelect.value = settings.reasoningEffort;
  aiBrokerProviderSelect.value = settings.aiBrokerProvider;
  l0ReplacementPreviewEnabledInput.checked = settings.l0ReplacementPreviewEnabled;
  l0CustomBaseUrlInput.value = settings.l0CustomBaseUrl;
  l0DontRunLlmInput.checked = settings.l0DontRunLlm;
  audioInputEnabledInput.checked = settings.audioInputEnabled;
  renderL0ReplacementSettings();
  l0ReplacementPreviewEnabledInput.addEventListener('change', renderL0ReplacementSettings);

  saveButton.addEventListener('click', () => {
    status.textContent = 'Saving...';
    const l0CustomBaseUrl = normalizeL0CustomBaseUrl(l0CustomBaseUrlInput.value);

    const permissionRequest = l0ReplacementPreviewEnabledInput.checked
      ? requestCustomHostPermission(l0CustomBaseUrl)
      : Promise.resolve();
    void permissionRequest
      .then(() =>
        saveSettings({
          backendBaseUrl: backendBaseUrlInput.value,
          projectPreset: 'ru-gold-2sp-v1',
          openRouterApiKey: openRouterApiKeyInput.value,
          model: modelInput.value,
          serviceTier:
            serviceTierSelect.value === 'default' || serviceTierSelect.value === 'priority' || serviceTierSelect.value === 'flex'
              ? serviceTierSelect.value
              : 'flex',
          reasoningEffort:
            reasoningEffortSelect.value === 'default' ||
            reasoningEffortSelect.value === 'none' ||
            reasoningEffortSelect.value === 'minimal' ||
            reasoningEffortSelect.value === 'low' ||
            reasoningEffortSelect.value === 'medium' ||
            reasoningEffortSelect.value === 'high' ||
            reasoningEffortSelect.value === 'xhigh'
              ? reasoningEffortSelect.value
              : 'low',
          aiBrokerProvider:
            aiBrokerProviderSelect.value === 'auto' ||
            aiBrokerProviderSelect.value === 'remote-openrouter' ||
            aiBrokerProviderSelect.value === 'local-gemini-nano'
              ? aiBrokerProviderSelect.value
              : 'auto',
          l0ReplacementPreviewEnabled: l0ReplacementPreviewEnabledInput.checked,
          l0CustomBaseUrl,
          l0DontRunLlm: l0DontRunLlmInput.checked,
          audioInputEnabled: audioInputEnabledInput.checked
        })
      )
      .then((saved) => {
        backendBaseUrlInput.value = saved.backendBaseUrl;
        projectPresetSelect.value = saved.projectPreset;
        openRouterApiKeyInput.value = saved.openRouterApiKey;
        modelInput.value = saved.model;
        serviceTierSelect.value = saved.serviceTier;
        reasoningEffortSelect.value = saved.reasoningEffort;
        aiBrokerProviderSelect.value = saved.aiBrokerProvider;
        l0ReplacementPreviewEnabledInput.checked = saved.l0ReplacementPreviewEnabled;
        l0CustomBaseUrlInput.value = saved.l0CustomBaseUrl;
        l0DontRunLlmInput.checked = saved.l0DontRunLlm;
        audioInputEnabledInput.checked = saved.audioInputEnabled;
        renderL0ReplacementSettings();
        status.textContent = 'Saved. Reload Babel tabs to pick up the new settings.';
      })
      .catch((error) => {
        status.textContent = error instanceof Error ? error.message : String(error);
      });
  });
}

void boot();
