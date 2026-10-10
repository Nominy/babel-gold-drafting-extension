import { ensureUiStyles, confirmDialog } from '@nominy/babel-extension-frontend';
import {
  LOCAL_MODEL_BASE_URL,
  LOCAL_MODEL_SAMPLE_URL,
  IS_DEV_C_DENOISE,
  isBrowserLocalMode,
  loadSettings,
  normalizeL0CustomBaseUrl,
  saveSettings
} from '../core/settings';
import {
  getLocalModelStatus,
  getCachedBundleDescriptor,
  markLocalModelWebGpuTested,
  removeLocalModels,
  setupLocalModels,
  type LocalModelStatus
} from '../core/local-model-bundle';
import { transcribeLocalAudio } from '../core/local-model-runtime';
import type { ExtensionSettings } from '../core/types';
import type { VolunteerStatus } from '../core/volunteer-protocol';
const MAX_TEST_AUDIO_SECONDS = 15;
function requireElement<T extends HTMLElement>(selector: string): T {
  const element = document.querySelector(selector);
  if (!(element instanceof HTMLElement)) {
    throw new Error(`Missing required element: ${selector}`);
  }
  return element as T;
}
async function requestHostPermission(baseUrl: string, purpose: string): Promise<void> {
  if (!globalThis.chrome?.permissions?.request) {
    return;
  }
  const originPattern = `${new URL(baseUrl).origin}/*`;
  const granted = await chrome.permissions.request({ origins: [originPattern] });
  if (!granted) {
    throw new Error(`Host access is required to ${purpose}: ${originPattern}`);
  }
}
function formatByteCount(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return '0 B';
  }
  const units = ['B', 'KB', 'MB', 'GB'];
  const unitIndex = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** unitIndex).toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}
export interface OptionsDependencies {
  fetchResource: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;
  transcribeAudio: typeof transcribeLocalAudio;
  volunteerStatus?: () => Promise<VolunteerStatus>;
}
const DEFAULT_OPTIONS_DEPENDENCIES: OptionsDependencies = {
  fetchResource: globalThis.fetch.bind(globalThis),
  transcribeAudio: transcribeLocalAudio
};
export async function boot(overrides: Partial<OptionsDependencies> = {}): Promise<void> {
  ensureUiStyles();
  const dependencies = { ...DEFAULT_OPTIONS_DEPENDENCIES, ...overrides };
  const modeSelect = requireElement<HTMLSelectElement>('#mode');
  const simpleSettings = requireElement<HTMLElement>('[data-role="simple-settings"]');
  const localSettings = requireElement<HTMLElement>('[data-role="local-settings"]');
  const cloudKeySettings = requireElement<HTMLElement>('[data-role="cloud-key-settings"]');
  const localModelSetup = requireElement<HTMLElement>('[data-role="local-model-setup"]');
  const advancedSettings = requireElement<HTMLFieldSetElement>('[data-role="advanced-settings"]');
  const isAdvanced = (): boolean => modeSelect.value === 'advanced';
  const showsLocalModels = (): boolean => modeSelect.value !== 'simple';
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
  const localModelsEnabledInput = requireElement<HTMLInputElement>('#localModelsEnabled');
  const volunteerInferenceEnabledInput = requireElement<HTMLInputElement>('#volunteerInferenceEnabled');
  const volunteerLabel = document.querySelector<HTMLElement>('[data-role="volunteer-label"]');
  const volunteerDescription = document.querySelector<HTMLElement>('[data-role="volunteer-description"]');
  if (volunteerLabel) volunteerLabel.textContent = "Volunteer GPU for other users' ZipEnhancer / L0 audio";
  if (volunteerDescription) volunteerDescription.textContent = 'Off for new installs. This permits downloading other users’ audio and using your GPU/network. ZipEnhancer needs a supported GPU; L0 also needs its verified downloaded bundle. Save Settings to apply.';
  const localModelDownloadButton = requireElement<HTMLButtonElement>('[data-role="local-model-download"]');
  const localModelRemoveButton = requireElement<HTMLButtonElement>('[data-role="local-model-remove"]');
  const localModelTestAudioInput = requireElement<HTMLInputElement>('#localModelTestAudio');
  const localModelTestButton = requireElement<HTMLButtonElement>('[data-role="local-model-test"]');
  const localModelSuppliedTestButton = requireElement<HTMLButtonElement>('[data-role="local-model-supplied-test"]');
  const localModelStatusElement = requireElement<HTMLElement>('[data-role="local-model-status"]');
  const volunteerStatusElement = requireElement<HTMLElement>('[data-role="volunteer-status"]');
  const localModelProgress = requireElement<HTMLProgressElement>('[data-role="local-model-progress"]');
  const saveButton = requireElement<HTMLButtonElement>('[data-role="save"]');
  const status = requireElement<HTMLElement>('[data-role="status"]');
  let localModelStatus: LocalModelStatus = {
    state: 'not-installed',
    completedBytes: 0,
    totalBytes: 0
  };
  let localModelOperationRunning = false;
  let localModelNotice = '';
  let localModelNoticeIsError = false;
  let localModelTestSucceeded = false;
  const renderL0ReplacementSettings = (): void => {
    l0ReplacementSettings.hidden = !l0ReplacementPreviewEnabledInput.checked;
  };
  const renderLocalModelControls = (): void => {
    const hasAudioFile = Boolean(localModelTestAudioInput.files?.[0]);
    const localModelCanEnable = localModelStatus.state === 'ready' && localModelTestSucceeded;
    localModelsEnabledInput.disabled = !isAdvanced() || localModelOperationRunning || !localModelCanEnable;
    volunteerInferenceEnabledInput.disabled = IS_DEV_C_DENOISE || localModelOperationRunning;
    localModelTestAudioInput.disabled = localModelOperationRunning || localModelStatus.state !== 'ready';
    localModelDownloadButton.disabled = localModelOperationRunning;
    localModelRemoveButton.disabled = localModelOperationRunning;
    localModelTestButton.disabled =
      localModelOperationRunning || localModelStatus.state !== 'ready' || !hasAudioFile;
    localModelSuppliedTestButton.disabled = localModelOperationRunning || localModelStatus.state !== 'ready';
    saveButton.disabled = localModelOperationRunning;
    modeSelect.disabled = localModelOperationRunning;
    const showProgress = localModelStatus.state === 'downloading' && localModelStatus.totalBytes > 0;
    localModelProgress.hidden = !showProgress;
    localModelProgress.max = Math.max(localModelStatus.totalBytes, 1);
    localModelProgress.value = Math.min(localModelStatus.completedBytes, localModelProgress.max);
    const statusIsError = localModelNoticeIsError || localModelStatus.state === 'error';
    localModelStatusElement.setAttribute('role', statusIsError ? 'alert' : 'status');
    localModelStatusElement.setAttribute('aria-live', statusIsError ? 'assertive' : 'polite');
    if (localModelNotice) {
      localModelStatusElement.textContent = localModelNotice;
    } else if (localModelStatus.state === 'ready') {
      localModelStatusElement.textContent =
        `C-denoise v3 — ${formatByteCount(localModelStatus.totalBytes)} verified and cached. ` +
        (localModelStatus.tested ? 'WebGPU audio test passed.' : 'Run the WebGPU audio test before use.') +
        ` ASR ${localModelStatus.source?.asrCheckpointSha256 ?? 'unknown'}; ` +
        `C-denoise ${localModelStatus.source?.cDenoiseCheckpointSha256 ?? 'unknown'}.`;
    } else if (localModelStatus.state === 'downloading') {
      const currentPath = localModelStatus.currentPath ? ` (${localModelStatus.currentPath})` : '';
      localModelStatusElement.textContent =
        `Downloading ${formatByteCount(localModelStatus.completedBytes)} of ` +
        `${formatByteCount(localModelStatus.totalBytes)}${currentPath}`;
    } else if (localModelStatus.state === 'error') {
      localModelStatusElement.textContent =
        localModelStatus.error || 'Local model setup failed. Check your connection to the Babel model supplier and try Download again.';
    } else {
      localModelStatusElement.textContent = 'Not downloaded. Download and verify the Babel model bundle before enabling it.';
    }
  };
  const refreshLocalModelStatus = async (): Promise<void> => {
    localModelNotice = '';
    if (!showsLocalModels()) return;
    localModelNoticeIsError = false;
    try {
      localModelStatus = await getLocalModelStatus(LOCAL_MODEL_BASE_URL);
    } catch (error) {
      localModelStatus = {
        state: 'error',
        completedBytes: 0,
        totalBytes: 0,
        error: error instanceof Error ? error.message : String(error)
      };
    }
    renderLocalModelControls();
  };
  const writeSettingsToControls = (settings: ExtensionSettings): void => {
    modeSelect.value = settings.mode;
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
    localModelsEnabledInput.checked = settings.localModelsEnabled;
    volunteerInferenceEnabledInput.checked = settings.volunteerInferenceEnabled;
    renderL0ReplacementSettings();
  };
  let persistedSettings = await loadSettings();
  const refreshVolunteerStatus = async (): Promise<void> => {
    if (!showsLocalModels()) return;
    if (IS_DEV_C_DENOISE) {
      volunteerStatusElement.textContent = 'Own-task WebGPU trial; shared coordinator not enabled. Your stored volunteer preference is preserved, but no shared jobs run in this dev build.';
      volunteerStatusElement.setAttribute('role', 'status');
      return;
    }
    const selectedLocal = modeSelect.value === 'local' || (isAdvanced() && localModelsEnabledInput.checked);
    const savedLocal = isBrowserLocalMode(persistedSettings);
    if (selectedLocal !== savedLocal ||
        volunteerInferenceEnabledInput.checked !== persistedSettings.volunteerInferenceEnabled) {
      const selected = selectedLocal && volunteerInferenceEnabledInput.checked;
      const saved = savedLocal && persistedSettings.volunteerInferenceEnabled;
      volunteerStatusElement.textContent = selected && !saved
        ? 'Volunteer: Save Settings to start volunteering.'
        : !selected && saved
          ? 'Volunteer: Save Settings to stop new volunteer work.'
          : 'Volunteer: Save Settings to apply your participation preference.';
      volunteerStatusElement.setAttribute('role', 'status');
      return;
    }
    let worker: VolunteerStatus;
    if (!savedLocal) {
      worker = { state: 'disabled' };
    } else if (!persistedSettings.volunteerInferenceEnabled) {
      worker = { state: 'disabled', detail: 'Swarm participation is off; local models remain available for your own tasks.' };
    } else {
      try {
        worker = dependencies.volunteerStatus
          ? await dependencies.volunteerStatus()
          : await chrome.runtime.sendMessage({ type: 'babel-l0-volunteer', target: 'background', action: 'status' });
        if (!worker || !['disabled', 'connecting', 'connected', 'busy', 'error'].includes(worker.state)) {
          throw new Error('Worker returned an invalid status.');
        }
      } catch (error) {
        worker = { state: 'error', detail: error instanceof Error ? error.message : String(error) };
      }
    }
    const label: Record<VolunteerStatus['state'], string> = {
      disabled: 'Not volunteering.',
      connecting: 'Connecting to the L0 coordinator…',
      connected: 'Connected and available for volunteer jobs.',
      busy: 'Busy processing a volunteer job.',
      error: 'Disconnected from the L0 coordinator.'
    };
    volunteerStatusElement.textContent = `Volunteer: ${label[worker.state]}${worker.detail ? ` ${worker.detail}` : ''}`;
    volunteerStatusElement.setAttribute('role', worker.state === 'error' ? 'alert' : 'status');
  };
  let localModelsInitialized = false;
  let volunteerStatusRunning = false;
  let stopVolunteerStatus = (): void => {};
  const renderMode = async (): Promise<void> => {
    const advanced = isAdvanced();
    const local = modeSelect.value === 'local';
    simpleSettings.hidden = modeSelect.value !== 'simple';
    localSettings.hidden = !local;
    cloudKeySettings.hidden = false;
    openRouterApiKeyInput.disabled = false;
    requireElement<HTMLElement>('[data-role="gold-llm-settings"]').hidden = modeSelect.value === 'simple';
    advancedSettings.hidden = !advanced;
    advancedSettings.disabled = !advanced;
    localModelSetup.hidden = !showsLocalModels();
    if (!showsLocalModels()) {
      stopVolunteerStatus();
      return;
    }
    if (!localModelsInitialized) {
      localModelsInitialized = true;
      await refreshLocalModelStatus();
      localModelTestSucceeded = localModelStatus.state === 'ready' && localModelStatus.tested === true;
    }
    renderLocalModelControls();
    if (!showsLocalModels()) return;
    await refreshVolunteerStatus();
    if (IS_DEV_C_DENOISE || !showsLocalModels() || volunteerStatusRunning) return;
    const volunteerStatusTimer = setInterval(() => { void refreshVolunteerStatus(); }, 3_000);
    volunteerStatusRunning = true;
    stopVolunteerStatus = (): void => {
      clearInterval(volunteerStatusTimer);
      volunteerStatusRunning = false;
    };
    if (typeof volunteerStatusTimer === 'object' && 'unref' in volunteerStatusTimer) volunteerStatusTimer.unref();
  };
  writeSettingsToControls(persistedSettings);
  await renderMode();
  window.addEventListener('pagehide', () => stopVolunteerStatus(), { once: true });
  modeSelect.addEventListener('change', () => { void renderMode(); });
  l0ReplacementPreviewEnabledInput.addEventListener('change', renderL0ReplacementSettings);
  localModelTestAudioInput.addEventListener('change', renderLocalModelControls);
  localModelsEnabledInput.addEventListener('change', () => { void refreshVolunteerStatus(); });
  volunteerInferenceEnabledInput.addEventListener('change', () => { void refreshVolunteerStatus(); });
  localModelDownloadButton.addEventListener('click', () => {
    if (!showsLocalModels() || localModelOperationRunning) return;
    localModelTestSucceeded = false;
    localModelsEnabledInput.checked = false;
    localModelOperationRunning = true;
    localModelNotice = '';
    localModelNoticeIsError = false;
    localModelStatus = { state: 'downloading', completedBytes: 0, totalBytes: 0 };
    renderLocalModelControls();
    void requestHostPermission(LOCAL_MODEL_BASE_URL, 'download the model bundle')
      .then(() => saveSettings({ ...persistedSettings, localModelsEnabled: false }))
      .then((saved) => {
        persistedSettings = saved;
        void refreshVolunteerStatus();
        return setupLocalModels(LOCAL_MODEL_BASE_URL, (progress) => {
          localModelStatus = { state: 'downloading', ...progress };
          renderLocalModelControls();
        });
      })
      .then((readyStatus) => {
        localModelStatus = readyStatus;
        localModelNotice = 'Download complete and verified. Test a short audio sample before enabling local browser models.';
        localModelNoticeIsError = false;
      })
      .catch((error) => {
        const message = error instanceof Error ? error.message : String(error);
        localModelStatus = { state: 'error', completedBytes: 0, totalBytes: 0, error: message };
        localModelNotice =
          `Download from the Babel model supplier failed: ${message} ` +
          `Verify the model bundle at ${LOCAL_MODEL_BASE_URL} and try again.`;
        localModelNoticeIsError = true;
      })
      .finally(() => {
        localModelOperationRunning = false;
        renderLocalModelControls();
      });
  });
  localModelRemoveButton.addEventListener('click', async () => {
    if (!showsLocalModels() || localModelOperationRunning) return;
    if (!(await confirmDialog({ accent: 'purple', title: 'Remove local models?', message: 'Remove the downloaded local model bundle and disable local browser models?', confirmLabel: 'Remove models' }))) {
      return;
    }
    localModelOperationRunning = true;
    localModelNotice = 'Removing downloaded local models…';
    localModelNoticeIsError = false;
    renderLocalModelControls();
    void saveSettings({ ...persistedSettings, localModelsEnabled: false })
      .then((saved) => {
        persistedSettings = saved;
        localModelsEnabledInput.checked = false;
        void refreshVolunteerStatus();
        return removeLocalModels();
      })
      .then(() => {
        localModelStatus = { state: 'not-installed', completedBytes: 0, totalBytes: 0 };
        localModelTestSucceeded = false;
        localModelTestAudioInput.value = '';
        localModelNotice = 'Downloaded models removed and local browser models disabled.';
        localModelNoticeIsError = false;
      })
      .catch((error) => {
        const message = error instanceof Error ? error.message : String(error);
        localModelNotice = `Could not remove local models: ${message} Try again or reload the options page.`;
        localModelNoticeIsError = true;
      })
      .finally(() => {
        localModelOperationRunning = false;
        renderLocalModelControls();
      });
  });
  const runLocalModelTest = async (file: File): Promise<void> => {
    if (file.size === 0 || (!file.type.startsWith('audio/') && !file.name.toLowerCase().endsWith('.wav'))) {
      throw new Error('Choose a non-empty WAV or another supported audio file.');
    }
    localModelNotice = 'Decoding this sample and running C-denoise on WebGPU…';
    renderLocalModelControls();
    const descriptor = await getCachedBundleDescriptor(LOCAL_MODEL_BASE_URL);
    if (!descriptor) throw new Error('Download and verify the C-denoise v3 bundle before testing.');
    const result = await dependencies.transcribeAudio(file, {
      maxDurationSeconds: MAX_TEST_AUDIO_SECONDS, allowUntestedBundle: true
    });
    const execution = result.execution;
    if (!execution || execution.provider !== 'webgpu' || execution.shaderF16 !== true ||
        execution.denoiseSteps !== 4 || execution.neuralCpuFallback !== false ||
        execution.bundleIdentity !== descriptor.identity ||
        execution.asrCheckpointSha256 !== descriptor.source.asrCheckpointSha256 ||
        execution.cDenoiseCheckpointSha256 !== descriptor.source.cDenoiseCheckpointSha256 ||
        !execution.adapter || execution.adapter.isFallbackAdapter !== false ||
        execution.adapter.shaderF16 !== true ||
        execution.placementAudit?.method !== 'sha256-bound-source-nodes-and-webgpu-dispatch-profile' ||
        execution.placementAudit.graphOptimizations !== 'disabled' ||
        execution.placementAudit.hostMetadataAllowed !== true ||
        execution.placementAudit.graphs.length !== 3 ||
        ['asr/v3_ctc.onnx', 'punctuation/context.fp16.onnx', 'punctuation/denoise.fp16.onnx'].some((path) => {
          const graph = execution.placementAudit.graphs.find((candidate) => candidate.path === path);
          const file = descriptor.files.find((candidate) => candidate.path === path);
          return !graph || !file || graph.sha256 !== file.sha256 ||
            !Number.isSafeInteger(graph.requiredGpuNodes) || graph.requiredGpuNodes <= 0 ||
            graph.verifiedGpuNodes !== graph.requiredGpuNodes ||
            !Number.isSafeInteger(graph.gpuPrograms) || graph.gpuPrograms < graph.requiredGpuNodes ||
            !Number.isSafeInteger(graph.verifiedRuns) ||
            graph.verifiedRuns < (path === 'punctuation/denoise.fp16.onnx' ? 4 : 1);
        })) {
      throw new Error('The test did not prove complete C-denoise WebGPU execution with the current checkpoint bundle. No local activation or cloud fallback occurred.');
    }
    const transcript = result.text.trim() || '[No speech recognized]';
    localModelNotice = `Test succeeded (${result.durationSeconds.toFixed(1)}s): ${transcript}`;
    await markLocalModelWebGpuTested(LOCAL_MODEL_BASE_URL, descriptor.identity);
    localModelStatus = await getLocalModelStatus(LOCAL_MODEL_BASE_URL);
    localModelTestSucceeded = true;
    localModelNoticeIsError = false;
  };
  const startLocalModelTest = (loadFile: () => Promise<File>): void => {
    if (!showsLocalModels() || localModelOperationRunning) return;
    localModelOperationRunning = true;
    localModelNoticeIsError = false;
    renderLocalModelControls();
    void loadFile()
      .then(runLocalModelTest)
      .catch((error) => {
        const message = error instanceof Error ? error.message : String(error);
        localModelNotice = `Local model test failed: ${message}`;
        localModelNoticeIsError = true;
      })
      .finally(() => {
        localModelOperationRunning = false;
        renderLocalModelControls();
      });
  };
  localModelTestButton.addEventListener('click', () => {
    if (!showsLocalModels() || localModelOperationRunning) return;
    const file = localModelTestAudioInput.files?.[0];
    if (!file) {
      localModelNotice = 'Choose a WAV or another supported audio file before testing.';
      localModelNoticeIsError = true;
      renderLocalModelControls();
      return;
    }
    startLocalModelTest(() => Promise.resolve(file));
  });
  localModelSuppliedTestButton.addEventListener('click', () => {
    if (!showsLocalModels() || localModelOperationRunning) return;
    localModelNotice = 'Fetching the supplied public-domain sample…';
    startLocalModelTest(async () => {
      let response: Response;
      try {
        response = await dependencies.fetchResource(LOCAL_MODEL_SAMPLE_URL);
      } catch (error) {
        const detail = error instanceof Error ? error.message : String(error);
        throw new Error(`Could not fetch the supplied sample from the Babel model supplier: ${detail}`);
      }
      if (!response.ok) {
        throw new Error(
          `The Babel model supplier returned HTTP ${response.status} while fetching the supplied sample. Try again later.`
        );
      }
      const blob = await response.blob();
      if (blob.size === 0) {
        throw new Error('The Babel model supplier returned an empty supplied sample. Try again later.');
      }
      return new File([blob], 'sample-russian-15s.wav', { type: blob.type || 'audio/wav' });
    });
  });
  saveButton.addEventListener('click', () => {
    status.textContent = 'Saving...';
    status.setAttribute('role', 'status');
    status.setAttribute('aria-live', 'polite');
    const advanced = isAdvanced();
    const settingsToSave = advanced
      ? {
          mode: 'advanced',
          backendBaseUrl: backendBaseUrlInput.value,
          projectPreset: 'ru-gold-2sp-v1',
          openRouterApiKey: openRouterApiKeyInput.value,
          model: modelInput.value,
          serviceTier: serviceTierSelect.value,
          reasoningEffort: reasoningEffortSelect.value,
          aiBrokerProvider: aiBrokerProviderSelect.value,
          l0ReplacementPreviewEnabled: l0ReplacementPreviewEnabledInput.checked,
          l0CustomBaseUrl: normalizeL0CustomBaseUrl(l0CustomBaseUrlInput.value),
          l0DontRunLlm: l0DontRunLlmInput.checked,
          audioInputEnabled: audioInputEnabledInput.checked,
          localModelsEnabled: localModelsEnabledInput.checked,
          volunteerInferenceEnabled: IS_DEV_C_DENOISE
            ? persistedSettings.volunteerInferenceEnabled
            : volunteerInferenceEnabledInput.checked
        }
      : {
          ...persistedSettings,
          mode: modeSelect.value === 'local' ? 'local' : 'simple',
          ...(modeSelect.value === 'local'
            ? { openRouterApiKey: openRouterApiKeyInput.value, l0DontRunLlm: l0DontRunLlmInput.checked, volunteerInferenceEnabled: IS_DEV_C_DENOISE
                ? persistedSettings.volunteerInferenceEnabled
                : volunteerInferenceEnabledInput.checked }
            : { openRouterApiKey: openRouterApiKeyInput.value })
        };
    const validateLocalModels = async (): Promise<void> => {
      if (!advanced || !settingsToSave.localModelsEnabled) {
        return;
      }
      const currentStatus = await getLocalModelStatus(LOCAL_MODEL_BASE_URL);
      localModelStatus = currentStatus;
      renderLocalModelControls();
      if (currentStatus.state !== 'ready') {
        throw new Error('Download and verify the Babel model bundle before enabling local browser models.');
      }
      if (!localModelTestSucceeded || !currentStatus.tested) {
        throw new Error('Test the ready local model bundle with a short audio sample before enabling it.');
      }
    };
    void validateLocalModels()
      .then(() =>
        advanced && settingsToSave.l0ReplacementPreviewEnabled && !settingsToSave.localModelsEnabled
          ? requestHostPermission(settingsToSave.l0CustomBaseUrl, 'use the custom L0 endpoint')
          : Promise.resolve()
      )
      .then(() => saveSettings(settingsToSave))
      .then((saved) => {
        persistedSettings = saved;
        if (advanced) writeSettingsToControls(saved);
        else if (saved.mode === 'simple') openRouterApiKeyInput.value = saved.openRouterApiKey;
        renderLocalModelControls();
        void renderMode();
        status.textContent = 'Saved. Reload Babel tabs to pick up the new settings.';
        status.setAttribute('role', 'status');
        status.setAttribute('aria-live', 'polite');
      })
      .catch((error) => {
        status.textContent = error instanceof Error ? error.message : String(error);
        status.setAttribute('role', 'alert');
        status.setAttribute('aria-live', 'assertive');
      });
  });
}
if (globalThis.document?.currentScript) {
  void boot();
}
