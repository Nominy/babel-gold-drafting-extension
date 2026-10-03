import { INFERENCE_RELEASE } from '../src/core/inference-release';
import test from 'node:test';
import assert from 'node:assert/strict';
import { setImmediate as waitForImmediate } from 'node:timers/promises';
import fs from 'node:fs';
import { JSDOM } from 'jsdom';

import { boot, type OptionsDependencies } from '../src/options/options';
import { getCachedBundleDescriptor } from '../src/core/local-model-bundle';

const SETTINGS_KEY = 'babel_gold_drafting_settings';
const POINTER_KEY = 'babel_gold_local_model_bundle_pointer';
const FIXED_BASE_URL = INFERENCE_RELEASE.modelBaseUrl;
const REQUIRED_PATHS = [
  'asr/v3_ctc.onnx',
  'asr/v3_ctc.yaml',
  'punctuation/context.fp16.onnx',
  'punctuation/denoise.fp16.onnx',
  'punctuation/c-denoise.json',
  'punctuation/gpu-placement.json',
  'punctuation/config.json',
  'punctuation/tokenizer.json',
  'punctuation/tokenizer_config.json',
  'punctuation/special_tokens_map.json',
  'punctuation/vocab.txt'
];

const SOURCE = {
  asrCheckpointSha256: '02cea9973d0e839f6a3eeca101b83a93f93a066c2da2e3ebfa176d57e61d84d3',
  cDenoiseCheckpointSha256: '0a01c3535fb66627b13f266bc59ab7b95c2aa85f7413a051d1e17294515ded5a',
  baseModelFiles: { 'model.safetensors': '1'.repeat(64) }
};
type ReadyPointerFixture = {
  releaseId: string;
  version: number;
  cacheName: string;
  baseUrl: string;
  totalBytes: number;
  files: { path: string; bytes: number; sha256: string }[];
  source: typeof SOURCE;
  webgpuTestedAt?: number;
};
function readyPointer(cacheName: string, tested = false): ReadyPointerFixture {
  return { version: 3, releaseId: INFERENCE_RELEASE.id, cacheName, baseUrl: FIXED_BASE_URL, totalBytes: REQUIRED_PATHS.length,
    files: REQUIRED_PATHS.map((path) => ({ path, bytes: 1, sha256: '0'.repeat(64) })),
    source: SOURCE, ...(tested ? { webgpuTestedAt: 1 } : {}) };
}
function installReadyCache(pointer: ReadyPointerFixture, onMatch: (url: string) => void = () => {}) {
  const caseResult = { id: 'case', pass: true, comparisons: { logits: { pass: true, finite: true, maxAbs: 0.01, rmse: 0.001 } } };
  const manifest = { schema: 'babel-browser-model-bundle-v3', releaseId: INFERENCE_RELEASE.id, targetBytes: 1_500_000_000,
    totalBytes: pointer.totalBytes, files: pointer.files, source: pointer.source, pass: true,
    validation: { numericExport: { pass: true, reportSha256: '2'.repeat(64), limits: { maxAbs: 0.2 }, cases: [caseResult], asr: [caseResult] } } };
  let proof: Response | undefined = pointer.webgpuTestedAt
    ? new Response(JSON.stringify(pointer))
    : undefined;
  Object.assign(globalThis, { caches: {
    has: async (name: string) => name === pointer.cacheName,
    open: async () => ({
      match: async (input: string | Request) => {
        const url = String(input);
        if (url.endsWith('/__bundle-pointer.json')) return proof?.clone();
        onMatch(url);
        return url.endsWith('/manifest.json')
          ? new Response(JSON.stringify(manifest))
          : new Response(new Uint8Array([0]));
      },
      put: async (_input: string, response: Response) => { proof = response.clone(); }
    })
  } });
}
async function waitForLocalOperation(dom: JSDOM): Promise<void> {
  const save = dom.window.document.querySelector<HTMLButtonElement>('[data-role="save"]')!;
  if (!save.disabled) return;
  await new Promise<void>((resolve) => {
    const observer = new dom.window.MutationObserver(() => {
      if (!save.disabled) { observer.disconnect(); resolve(); }
    });
    observer.observe(save, { attributes: true, attributeFilter: ['disabled'] });
  });
}


function createDom(): JSDOM {
  const html = fs.readFileSync(new URL('../options.html', import.meta.url), 'utf8');
  const dom = new JSDOM(html, { url: 'chrome-extension://test/options.html' });
  Object.assign(globalThis, {
    window: dom.window,
    self: dom.window,
    document: dom.window.document,
    HTMLElement: dom.window.HTMLElement,
    HTMLInputElement: dom.window.HTMLInputElement,
    HTMLSelectElement: dom.window.HTMLSelectElement,
    HTMLButtonElement: dom.window.HTMLButtonElement,
    HTMLProgressElement: dom.window.HTMLProgressElement
  });
  return dom;
}

function defaultSettings(localModelsEnabled = false): Record<string, unknown> {
  return {
    mode: 'advanced',
    backendBaseUrl: 'https://reviewgen.ovh',
    projectPreset: 'ru-gold-2sp-v1',
    openRouterApiKey: '',
    model: 'google/gemini-3-flash-preview',
    serviceTier: 'flex',
    reasoningEffort: 'low',
    aiBrokerProvider: 'auto',
    l0ReplacementPreviewEnabled: true,
    l0CustomBaseUrl:
      'https://reviewgen.ovh/a3f73d6cf25fa138be653daaf2d7cd0702c0b2d69c40fb9eaee4e07d4b067dd5',
    l0DontRunLlm: false,
    audioInputEnabled: true,
    localModelsEnabled,
    volunteerInferenceEnabled: false,
  };
}

function installChromeStorage(storageData: Record<string, unknown>): {
  getStoredSettings: () => Record<string, unknown> | undefined;
  getPermissionRequests: () => string[][];
} {
  let storedSettings: Record<string, unknown> | undefined;
  const permissionRequests: string[][] = [];
  Object.assign(globalThis, {
    chrome: {
      runtime: {},
      storage: {
        local: {
          get(_key: string | string[], callback?: (items: Record<string, unknown>) => void) {
            callback?.(storageData);
            return Promise.resolve(storageData);
          },
          set(items: Record<string, unknown>, callback?: () => void) {
            Object.assign(storageData, items);
            if (items[SETTINGS_KEY]) {
              storedSettings = items[SETTINGS_KEY] as Record<string, unknown>;
            }
            callback?.();
            return Promise.resolve();
          },
          remove(keys: string | string[]) {
            for (const key of Array.isArray(keys) ? keys : [keys]) {
              delete storageData[key];
            }
            return Promise.resolve();
          }
        }
      },
      permissions: {
        request(options: { origins?: string[] }) {
          permissionRequests.push(options.origins ?? []);
          return Promise.resolve(true);
        },
        remove: () => Promise.resolve(true)
      }
    }
  });
  return {
    getStoredSettings: () => storedSettings,
    getPermissionRequests: () => permissionRequests
  };
}

function unusedDependencies(): OptionsDependencies {
  return {
    fetchResource: async () => {
      throw new Error('sample fetch must not run');
    },
    transcribeAudio: async () => {
      throw new Error('inference must not run');
    }
  };
}

test('legacy options remain Advanced and a failed local download keeps local execution disabled', async (t) => {
  const dom = createDom();
  t.after(() => { dom.window.dispatchEvent(new dom.window.Event('pagehide')); dom.window.close(); });
  const legacySettings = defaultSettings();
  delete legacySettings.mode;
  const storageData: Record<string, unknown> = { [SETTINGS_KEY]: legacySettings };
  const storage = installChromeStorage(storageData);
  let cacheOperations = 0;
  let downloadRequestUrl = '';
  Object.assign(globalThis, {
    caches: new Proxy(
      {},
      {
        get() {
          cacheOperations += 1;
          throw new Error('Cache Storage must not be used while no bundle is installed');
        }
      }
    ),
    fetch: (input: string | URL | Request) => {
      downloadRequestUrl = String(input);
      return Promise.resolve(new Response('supplier unavailable', { status: 503 }));
    }
  });

  await boot(unusedDependencies());
  assert.equal(dom.window.document.querySelector<HTMLSelectElement>('#mode')!.value, 'advanced');

  const enabled = dom.window.document.querySelector<HTMLInputElement>('#localModelsEnabled');
  const download = dom.window.document.querySelector<HTMLButtonElement>('[data-role="local-model-download"]');
  const status = dom.window.document.querySelector<HTMLElement>('[data-role="local-model-status"]');
  const save = dom.window.document.querySelector<HTMLButtonElement>('[data-role="save"]');
  assert.ok(enabled);
  assert.ok(download);
  assert.ok(status);
  assert.ok(save);
  assert.equal(enabled.checked, false);
  assert.equal(enabled.disabled, true);
  assert.equal(download.disabled, false);
  assert.match(status.textContent ?? '', /Not downloaded/);
  assert.equal(cacheOperations, 0);
  assert.deepEqual(storage.getPermissionRequests(), []);

  download.click();
  await waitForImmediate();
  await waitForImmediate();
  assert.equal(downloadRequestUrl, `${FIXED_BASE_URL}/manifest.json`);
  assert.deepEqual(storage.getPermissionRequests(), [['https://reviewgen.ovh/*']]);

  save.click();
  await waitForImmediate();
  const storedSettings = storage.getStoredSettings();
  assert.equal(storedSettings?.localModelsEnabled, false);
  assert.equal(storedSettings?.mode, 'advanced');
  assert.equal('localModelBaseUrl' in (storedSettings ?? {}), false);
});

test('supplied public-domain sample unlocks enable only after successful inference against the fixed ready bundle', async (t) => {
  const dom = createDom();
  t.after(() => { dom.window.dispatchEvent(new dom.window.Event('pagehide')); dom.window.close(); });
  const cacheName = 'babel-gold-local-models:bundle:installed';
  const storageData: Record<string, unknown> = {
    [SETTINGS_KEY]: { ...defaultSettings(false), volunteerInferenceEnabled: true },
    [POINTER_KEY]: readyPointer(cacheName)
  };
  const storage = installChromeStorage(storageData);
  installReadyCache(readyPointer(cacheName));
  Object.assign(globalThis, {
    fetch: () => {
      throw new Error('model bundle network fetch must not run for a ready cached bundle');
    }
  });

  let inferenceSucceeds = false;
  let inferenceCalls = 0;
  let volunteerState: 'connected' | 'busy' | 'error' = 'connected';
  await boot({
    fetchResource: async () => {
      return new Response(new Uint8Array([82, 73, 70, 70]), {
        status: 200,
        headers: { 'content-type': 'audio/wav' }
      });
    },
    transcribeAudio: async () => {
      inferenceCalls += 1;
      if (!inferenceSucceeds) {
        throw new Error('inference rejected the sample');
      }
      return {
        text: 'тест прошёл',
        durationSeconds: 15,
        tokens: [],
        execution: {
          provider: 'webgpu', shaderF16: true, denoiseSteps: 4, neuralCpuFallback: false,
          asrCheckpointSha256: SOURCE.asrCheckpointSha256,
          cDenoiseCheckpointSha256: SOURCE.cDenoiseCheckpointSha256,
          bundleIdentity: (await getCachedBundleDescriptor(FIXED_BASE_URL))!.identity,
          placementAudit: {
            method: 'sha256-bound-source-nodes-and-webgpu-dispatch-profile',
            graphOptimizations: 'disabled', hostMetadataAllowed: true, cpuAttemptsPrevented: false,
            graphs: ['asr/v3_ctc.onnx', 'punctuation/context.fp16.onnx', 'punctuation/denoise.fp16.onnx'].map((path) => ({
              path, sha256: '0'.repeat(64), verifiedRuns: path.includes('denoise') ? 4 : 1,
              requiredGpuNodes: 1, verifiedGpuNodes: 1, gpuPrograms: 1,
              allowedHostMetadataNodes: 1, storageAliasNodes: 1, constantNodes: 1
            }))
          },
          adapter: { isFallbackAdapter: false, shaderF16: true, vendor: 'unit-fixture', architecture: 'unit-fixture',
            device: 'unit-fixture', description: 'UI control fixture, not WebGPU proof',
            maxBufferSize: 1_000_000_000, maxStorageBufferBindingSize: 1_000_000_000 }
        }
      };
    },
    volunteerStatus: async () => ({ state: volunteerState })
  });

  const enabled = dom.window.document.querySelector<HTMLInputElement>('#localModelsEnabled');
  const volunteerEnabled = dom.window.document.querySelector<HTMLInputElement>('#volunteerInferenceEnabled');
  const suppliedTest = dom.window.document.querySelector<HTMLButtonElement>('[data-role="local-model-supplied-test"]');
  const status = dom.window.document.querySelector<HTMLElement>('[data-role="local-model-status"]');
  const save = dom.window.document.querySelector<HTMLButtonElement>('[data-role="save"]');
  const volunteerStatus = dom.window.document.querySelector<HTMLElement>('[data-role="volunteer-status"]');
  assert.ok(enabled);
  assert.ok(volunteerEnabled);
  assert.equal(volunteerEnabled.checked, true);
  assert.ok(suppliedTest);
  assert.ok(status);
  assert.ok(save);
  assert.ok(volunteerStatus);
  assert.equal(enabled.disabled, true);
  assert.equal(suppliedTest.disabled, false);

  suppliedTest.click();
  await waitForLocalOperation(dom);
  assert.equal(inferenceCalls, 1);
  assert.equal(enabled.disabled, true);
  assert.match(status.textContent ?? '', /Local model test failed: inference rejected the sample/);

  inferenceSucceeds = true;
  suppliedTest.click();
  await waitForLocalOperation(dom);
  assert.equal(inferenceCalls, 2);
  assert.equal(enabled.disabled, false);
  enabled.checked = true;
  enabled.dispatchEvent(new dom.window.Event('change'));
  save.click();
  await waitForImmediate();
  const storedSettings = storage.getStoredSettings();
  assert.equal(storedSettings?.localModelsEnabled, true);
  assert.equal('localModelBaseUrl' in (storedSettings ?? {}), false);
  volunteerEnabled.checked = false;
  volunteerEnabled.dispatchEvent(new dom.window.Event('change'));
  save.click();
  await waitForImmediate();
  assert.equal(storage.getStoredSettings()?.localModelsEnabled, true);
  assert.equal(storage.getStoredSettings()?.volunteerInferenceEnabled, false);
  assert.equal(enabled.checked, true);
  volunteerEnabled.checked = true;
  volunteerEnabled.dispatchEvent(new dom.window.Event('change'));
  save.click();
  await waitForImmediate();
  assert.equal(storage.getStoredSettings()?.volunteerInferenceEnabled, true);
  volunteerState = 'busy';
  save.click();
  await waitForImmediate();
  volunteerState = 'error';
  save.click();
  await waitForImmediate();
  enabled.checked = false;
  enabled.dispatchEvent(new dom.window.Event('change'));
  assert.match(volunteerStatus.textContent ?? '', /Save Settings to stop new volunteer work/);
  save.click();
  await waitForImmediate();
  assert.equal(storage.getStoredSettings()?.localModelsEnabled, false);
  assert.match(volunteerStatus.textContent ?? '', /Not volunteering/);
});

test('supplied sample fetch failures are actionable and never run inference', async (t) => {
  const dom = createDom();
  t.after(() => { dom.window.dispatchEvent(new dom.window.Event('pagehide')); dom.window.close(); });
  const cacheName = 'babel-gold-local-models:bundle:installed';
  const storageData: Record<string, unknown> = {
    [SETTINGS_KEY]: defaultSettings(),
    [POINTER_KEY]: readyPointer(cacheName)
  };
  installChromeStorage(storageData);
  installReadyCache(readyPointer(cacheName));
  let inferenceCalls = 0;
  await boot({
    fetchResource: async () => new Response('missing', { status: 404 }),
    transcribeAudio: async () => {
      inferenceCalls += 1;
      return { text: '', durationSeconds: 1, tokens: [] };
    }
  });

  const enabled = dom.window.document.querySelector<HTMLInputElement>('#localModelsEnabled');
  const suppliedTest = dom.window.document.querySelector<HTMLButtonElement>('[data-role="local-model-supplied-test"]');
  const status = dom.window.document.querySelector<HTMLElement>('[data-role="local-model-status"]');
  assert.ok(enabled);
  assert.ok(suppliedTest);
  assert.ok(status);
  suppliedTest.click();
  await waitForLocalOperation(dom);
  assert.equal(inferenceCalls, 0);
  assert.equal(enabled.disabled, true);
  assert.match(status.textContent ?? '', /Babel model supplier returned HTTP 404/);
  assert.equal(status.getAttribute('role'), 'alert');
});

test('settings controls use canonical enum normalization and retain unsaved edits on failure', async (t) => {
  const dom = createDom();
  t.after(() => { dom.window.dispatchEvent(new dom.window.Event('pagehide')); dom.window.close(); });
  const storageData = {
    [SETTINGS_KEY]: {
      ...defaultSettings(),
      serviceTier: 'priority',
      reasoningEffort: 'high',
      aiBrokerProvider: 'remote-openrouter',
      l0ReplacementPreviewEnabled: false
    }
  };
  const storage = installChromeStorage(storageData);
  await boot(unusedDependencies());
  const select = (id: string) => {
    const element = dom.window.document.querySelector<HTMLSelectElement>(`#${id}`);
    assert.ok(element);
    return element;
  };
  const serviceTier = select('serviceTier');
  const reasoningEffort = select('reasoningEffort');
  const provider = select('aiBrokerProvider');
  const save = dom.window.document.querySelector<HTMLButtonElement>('[data-role="save"]');
  const status = dom.window.document.querySelector<HTMLElement>('[data-role="status"]');
  assert.ok(save);
  assert.ok(status);
  assert.equal(serviceTier.value, 'priority');
  assert.equal(reasoningEffort.value, 'high');
  assert.equal(provider.value, 'remote-openrouter');

  // An unrecognized DOM selection must use the same fallback as persisted raw input.
  serviceTier.value = 'invalid';
  reasoningEffort.value = 'invalid';
  provider.value = 'invalid';
  save.click();
  await waitForImmediate();
  assert.equal(storage.getStoredSettings()?.serviceTier, 'flex');
  assert.equal(storage.getStoredSettings()?.reasoningEffort, 'low');
  assert.equal(storage.getStoredSettings()?.aiBrokerProvider, 'auto');
  assert.equal(serviceTier.value, 'flex');
  assert.equal(reasoningEffort.value, 'low');
  assert.equal(provider.value, 'auto');

  serviceTier.value = 'default';
  reasoningEffort.value = 'xhigh';
  provider.value = 'local-gemini-nano';
  save.click();
  await waitForImmediate();
  assert.equal(storage.getStoredSettings()?.serviceTier, 'default');
  assert.equal(storage.getStoredSettings()?.reasoningEffort, 'xhigh');
  assert.equal(storage.getStoredSettings()?.aiBrokerProvider, 'local-gemini-nano');

  Object.assign(globalThis.chrome.storage.local, {
    set() { throw new Error('Storage write failed'); }
  });
  serviceTier.value = 'priority';
  save.click();
  await waitForImmediate();
  assert.equal(status.getAttribute('role'), 'alert');
  assert.match(status.textContent ?? '', /Storage write failed/);
  assert.equal(serviceTier.value, 'priority');
  assert.equal(storage.getStoredSettings()?.serviceTier, 'default');
});

test('explicit Simple options save only the key without local setup or optional permissions', async (t) => {
  const dom = createDom();
  t.after(() => { dom.window.dispatchEvent(new dom.window.Event('pagehide')); dom.window.close(); });
  const previousFetch = globalThis.fetch;
  const previousCaches = globalThis.caches;
  t.after(() => Object.assign(globalThis, { fetch: previousFetch, caches: previousCaches }));
  const storage = installChromeStorage({ [SETTINGS_KEY]: { ...defaultSettings(), mode: 'simple' } });
  let localSideEffects = 0;
  Object.assign(globalThis, {
    fetch: () => { localSideEffects += 1; throw new Error('No Simple setup fetch'); },
    caches: new Proxy({}, { get() { localSideEffects += 1; throw new Error('No Simple cache access'); } })
  });
  await boot({
    ...unusedDependencies(),
    volunteerStatus: async () => { localSideEffects += 1; throw new Error('No Simple volunteer worker'); }
  });
  const document = dom.window.document;
  const mode = document.querySelector<HTMLSelectElement>('#mode')!;
  const advanced = document.querySelector<HTMLFieldSetElement>('[data-role="advanced-settings"]')!;
  const simple = document.querySelector<HTMLElement>('[data-role="simple-settings"]')!;
  const key = document.querySelector<HTMLInputElement>('#openRouterApiKey')!;
  assert.equal(mode.value, 'simple');
  assert.equal(simple.hidden, false);
  assert.equal(advanced.hidden, true);
  assert.equal(advanced.disabled, true);
  key.value = ' simple-key ';
  document.querySelector<HTMLButtonElement>('[data-role="local-model-download"]')!.click();
  document.querySelector<HTMLButtonElement>('[data-role="local-model-supplied-test"]')!.click();
  document.querySelector<HTMLButtonElement>('[data-role="save"]')!.click();
  await waitForImmediate();
  assert.equal(storage.getStoredSettings()?.mode, 'simple');
  assert.equal(storage.getStoredSettings()?.openRouterApiKey, 'simple-key');
  assert.equal(storage.getStoredSettings()?.localModelsEnabled, false);
  assert.deepEqual(storage.getPermissionRequests(), []);
  assert.equal(localSideEffects, 0);
});

test('Simple saves preserve configured Advanced preferences and switching restores controls and permission behavior', async (t) => {
  const dom = createDom();
  t.after(() => { dom.window.dispatchEvent(new dom.window.Event('pagehide')); dom.window.close(); });
  const previousFetch = globalThis.fetch;
  const previousCaches = globalThis.caches;
  t.after(() => Object.assign(globalThis, { fetch: previousFetch, caches: previousCaches }));
  const configured = {
    ...defaultSettings(true),
    mode: 'simple',
    backendBaseUrl: 'https://backend.example',
    l0CustomBaseUrl: 'http://localhost:9010',
    l0DontRunLlm: true,
    volunteerInferenceEnabled: true,
    model: 'configured/model',
    serviceTier: 'priority',
    reasoningEffort: 'xhigh',
    aiBrokerProvider: 'local-gemini-nano'
  };
  const cacheName = 'babel-gold-local-models:bundle:installed';
  const storage = installChromeStorage({
    [SETTINGS_KEY]: configured,
    [POINTER_KEY]: readyPointer(cacheName, true)
  });
  let cacheAccesses = 0;
  let workerRequests = 0;
  installReadyCache(readyPointer(cacheName, true), () => { cacheAccesses += 1; });
  Object.assign(globalThis, {
    fetch: () => { throw new Error('No setup network request expected'); }
  });
  await boot({
    ...unusedDependencies(),
    volunteerStatus: async () => { workerRequests += 1; return { state: 'connected' }; }
  });
  const document = dom.window.document;
  const selectMode = document.querySelector<HTMLSelectElement>('#mode')!;
  const save = document.querySelector<HTMLButtonElement>('[data-role="save"]')!;
  const key = document.querySelector<HTMLInputElement>('#openRouterApiKey')!;
  const localEnabled = document.querySelector<HTMLInputElement>('#localModelsEnabled')!;
  const model = document.querySelector<HTMLInputElement>('#model')!;
  key.value = 'new-key';
  save.click();
  await waitForImmediate();
  assert.deepEqual(storage.getStoredSettings(), { ...configured, openRouterApiKey: 'new-key' });
  assert.equal(cacheAccesses, 0);
  assert.equal(workerRequests, 0);
  assert.deepEqual(storage.getPermissionRequests(), []);

  selectMode.value = 'advanced';
  selectMode.dispatchEvent(new dom.window.Event('change'));
  await waitForImmediate();
  assert.equal(document.querySelector<HTMLFieldSetElement>('[data-role="advanced-settings"]')!.disabled, false);
  assert.equal(model.value, 'configured/model');
  assert.equal(localEnabled.checked, true);
  assert.equal(localEnabled.disabled, false);
  assert.ok(cacheAccesses > 0);

  model.value = 'edited/model';
  selectMode.value = 'simple';
  selectMode.dispatchEvent(new dom.window.Event('change'));
  const previousCacheAccesses = cacheAccesses;
  const previousWorkerRequests = workerRequests;
  save.click();
  await waitForImmediate();
  assert.equal(storage.getStoredSettings()?.model, 'configured/model');
  assert.equal(model.value, 'edited/model');
  assert.equal(cacheAccesses, previousCacheAccesses);
  assert.equal(workerRequests, previousWorkerRequests);

  selectMode.value = 'advanced';
  selectMode.dispatchEvent(new dom.window.Event('change'));
  await waitForImmediate();
  localEnabled.checked = false;
  save.click();
  await waitForImmediate();
  assert.equal(storage.getStoredSettings()?.mode, 'advanced');
  assert.equal(storage.getStoredSettings()?.model, 'edited/model');
  assert.equal(storage.getStoredSettings()?.localModelsEnabled, false);
  assert.deepEqual(storage.getPermissionRequests(), [['http://localhost:9010/*']]);
});

test('new Local mode exposes setup without requesting cloud permissions or activating untested weights', async (t) => {
  const dom = createDom();
  t.after(() => { dom.window.dispatchEvent(new dom.window.Event('pagehide')); dom.window.close(); });
  const storage = installChromeStorage({});
  let networkCalls = 0;
  Object.assign(globalThis, { fetch: async () => { networkCalls += 1; throw new Error('No automatic network'); } });
  await boot(unusedDependencies());
  const document = dom.window.document;
  assert.equal(document.querySelector<HTMLSelectElement>('#mode')!.value, 'local');
  assert.equal(document.querySelector<HTMLElement>('[data-role="local-model-setup"]')!.hidden, false);
  assert.equal(document.querySelector<HTMLElement>('[data-role="cloud-key-settings"]')!.hidden, false);
  assert.equal(document.querySelector<HTMLInputElement>('#openRouterApiKey')!.disabled, false);
  assert.equal(document.querySelector<HTMLInputElement>('#volunteerInferenceEnabled')!.checked, false);
  assert.equal(document.querySelector<HTMLInputElement>('#localModelsEnabled')!.disabled, true);
  document.querySelector<HTMLButtonElement>('[data-role="save"]')!.click();
  await waitForImmediate();
  assert.equal(storage.getStoredSettings()?.mode, 'local');
  assert.equal(storage.getStoredSettings()?.volunteerInferenceEnabled, false);
  assert.equal(storage.getStoredSettings()?.localModelsEnabled, false);
  assert.equal(networkCalls, 0);
  assert.deepEqual(storage.getPermissionRequests(), []);
  assert.equal(document.querySelector<HTMLElement>('[data-role="gold-llm-settings"]')!.hidden, false);
  document.querySelector<HTMLInputElement>('#openRouterApiKey')!.value = 'local-gold-key';
  document.querySelector<HTMLInputElement>('#l0DontRunLlm')!.checked = true;
  document.querySelector<HTMLButtonElement>('[data-role="save"]')!.click();
  await waitForImmediate();
  assert.equal(storage.getStoredSettings()?.openRouterApiKey, 'local-gold-key');
  assert.equal(storage.getStoredSettings()?.l0DontRunLlm, true);
});

test('successful text without complete current-bundle WebGPU evidence cannot activate local models', async (t) => {
  const dom = createDom();
  t.after(() => { dom.window.dispatchEvent(new dom.window.Event('pagehide')); dom.window.close(); });
  const cacheName = 'babel-gold-local-models:bundle:diagnostic-gate';
  const data: Record<string, unknown> = { [SETTINGS_KEY]: defaultSettings(), [POINTER_KEY]: readyPointer(cacheName) };
  installChromeStorage(data);
  installReadyCache(readyPointer(cacheName));
  await boot({
    fetchResource: async () => new Response(new Uint8Array([82, 73, 70, 70]), { headers: { 'content-type': 'audio/wav' } }),
    transcribeAudio: async () => ({ text: 'resolved text is not proof', durationSeconds: 1, tokens: [] })
  });
  dom.window.document.querySelector<HTMLButtonElement>('[data-role="local-model-supplied-test"]')!.click();
  await waitForLocalOperation(dom);
  assert.equal(dom.window.document.querySelector<HTMLInputElement>('#localModelsEnabled')!.disabled, true);
  assert.match(dom.window.document.querySelector<HTMLElement>('[data-role="local-model-status"]')!.textContent ?? '', /did not prove complete C-denoise WebGPU/);
  assert.equal('webgpuTestedAt' in Object(data[POINTER_KEY]), false);
});
