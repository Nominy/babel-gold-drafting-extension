import {
  LOCAL_MODEL_BASE_URL,
  LOCAL_MODEL_SAMPLE_URL,
  loadSettings,
  normalizeL0CustomBaseUrl,
  saveSettings
} from '../core/settings';
import {
  getLocalModelStatus,
  removeLocalModels,
  setupLocalModels,
  type LocalModelStatus
} from '../core/local-model-bundle';
import { transcribeLocalAudio } from '../core/local-model-runtime';

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

function readAudioDurationSeconds(file: File): Promise<number> {
  const { promise, resolve, reject } = Promise.withResolvers<number>();
  const objectUrl = URL.createObjectURL(file);
  const audio = new Audio();
  const cleanup = (): void => {
    audio.removeAttribute('src');
    audio.load();
    URL.revokeObjectURL(objectUrl);
  };
  audio.preload = 'metadata';
  audio.addEventListener(
    'loadedmetadata',
    () => {
      const durationSeconds = audio.duration;
      cleanup();
      if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) {
        reject(new Error('The selected audio file has no readable duration.'));
        return;
      }
      resolve(durationSeconds);
    },
    { once: true }
  );
  audio.addEventListener(
    'error',
    () => {
      cleanup();
      reject(new Error('The selected file could not be read as audio. Choose a WAV or another supported audio file.'));
    },
    { once: true }
  );
  audio.src = objectUrl;
  return promise;
}
export interface OptionsDependencies {
  fetchResource: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;
  readAudioDuration: (file: File) => Promise<number>;
  transcribeAudio: typeof transcribeLocalAudio;
}

const DEFAULT_OPTIONS_DEPENDENCIES: OptionsDependencies = {
  fetchResource: globalThis.fetch.bind(globalThis),
  readAudioDuration: readAudioDurationSeconds,
  transcribeAudio: transcribeLocalAudio
};


export async function boot(overrides: Partial<OptionsDependencies> = {}): Promise<void> {
  const dependencies = { ...DEFAULT_OPTIONS_DEPENDENCIES, ...overrides };
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
  const localModelDownloadButton = requireElement<HTMLButtonElement>('[data-role="local-model-download"]');
  const localModelRemoveButton = requireElement<HTMLButtonElement>('[data-role="local-model-remove"]');
  const localModelTestAudioInput = requireElement<HTMLInputElement>('#localModelTestAudio');
  const localModelTestButton = requireElement<HTMLButtonElement>('[data-role="local-model-test"]');
  const localModelSuppliedTestButton = requireElement<HTMLButtonElement>('[data-role="local-model-supplied-test"]');
  const localModelStatusElement = requireElement<HTMLElement>('[data-role="local-model-status"]');
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
    localModelsEnabledInput.disabled = localModelOperationRunning || !localModelCanEnable;
    localModelTestAudioInput.disabled = localModelOperationRunning || localModelStatus.state !== 'ready';
    localModelDownloadButton.disabled = localModelOperationRunning;
    localModelRemoveButton.disabled = localModelOperationRunning || localModelStatus.state !== 'ready';
    localModelTestButton.disabled =
      localModelOperationRunning || localModelStatus.state !== 'ready' || !hasAudioFile;
    localModelSuppliedTestButton.disabled = localModelOperationRunning || localModelStatus.state !== 'ready';
    saveButton.disabled = localModelOperationRunning;

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
      localModelStatusElement.textContent = `Ready — ${formatByteCount(localModelStatus.totalBytes)} verified and cached.`;
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

  let persistedSettings = await loadSettings();
  const settings = persistedSettings;
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
  renderL0ReplacementSettings();
  await refreshLocalModelStatus();
  if (settings.localModelsEnabled && localModelStatus.state === 'ready') {
    localModelTestSucceeded = true;
  } else if (settings.localModelsEnabled) {
    localModelsEnabledInput.checked = false;
  }
  renderLocalModelControls();

  l0ReplacementPreviewEnabledInput.addEventListener('change', renderL0ReplacementSettings);
  localModelTestAudioInput.addEventListener('change', renderLocalModelControls);

  localModelDownloadButton.addEventListener('click', () => {
    localModelTestSucceeded = false;
    localModelsEnabledInput.checked = false;
    localModelOperationRunning = true;
    localModelNotice = '';
    localModelNoticeIsError = false;
    localModelStatus = { state: 'downloading', completedBytes: 0, totalBytes: 0 };
    renderLocalModelControls();
    void saveSettings({ ...persistedSettings, localModelsEnabled: false })
      .then((saved) => {
        persistedSettings = saved;
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
          `Check network access to ${LOCAL_MODEL_BASE_URL} and try again.`;
        localModelNoticeIsError = true;
      })
      .finally(() => {
        localModelOperationRunning = false;
        renderLocalModelControls();
      });
  });

  localModelRemoveButton.addEventListener('click', () => {
    if (localModelStatus.state !== 'ready') {
      localModelNotice = 'Only the ready Babel model bundle can be removed.';
      localModelNoticeIsError = true;
      renderLocalModelControls();
      return;
    }
    if (!window.confirm('Remove the downloaded local model bundle and disable local browser models?')) {
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
    localModelNotice = 'Checking the audio length…';
    renderLocalModelControls();
    const durationSeconds = await dependencies.readAudioDuration(file);
    if (durationSeconds > MAX_TEST_AUDIO_SECONDS) {
      throw new Error(
        `The selected audio is ${durationSeconds.toFixed(1)} seconds. Choose a sample no longer than ${MAX_TEST_AUDIO_SECONDS} seconds.`
      );
    }
    localModelNotice = 'Running the downloaded models on this sample…';
    renderLocalModelControls();
    const result = await dependencies.transcribeAudio(file);
    const transcript = result.text.trim() || '[No speech recognized]';
    localModelNotice = `Test succeeded (${result.durationSeconds.toFixed(1)}s): ${transcript}`;
    localModelTestSucceeded = true;
    localModelNoticeIsError = false;
  };

  const startLocalModelTest = (loadFile: () => Promise<File>): void => {
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
    const l0CustomBaseUrl = normalizeL0CustomBaseUrl(l0CustomBaseUrlInput.value);

    const validateLocalModels = async (): Promise<void> => {
      if (!localModelsEnabledInput.checked) {
        return;
      }
      const currentStatus = await getLocalModelStatus(LOCAL_MODEL_BASE_URL);
      localModelStatus = currentStatus;
      renderLocalModelControls();
      if (currentStatus.state !== 'ready') {
        throw new Error('Download and verify the Babel model bundle before enabling local browser models.');
      }
      if (!localModelTestSucceeded) {
        throw new Error('Test the ready local model bundle with a short audio sample before enabling it.');
      }
    };

    void validateLocalModels()
      .then(() =>
        l0ReplacementPreviewEnabledInput.checked && !localModelsEnabledInput.checked
          ? requestHostPermission(l0CustomBaseUrl, 'use the custom L0 endpoint')
          : Promise.resolve()
      )
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
          audioInputEnabled: audioInputEnabledInput.checked,
          localModelsEnabled: localModelsEnabledInput.checked
        })
      )
      .then((saved) => {
        persistedSettings = saved;
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
        localModelsEnabledInput.checked = saved.localModelsEnabled;
        renderL0ReplacementSettings();
        renderLocalModelControls();
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
