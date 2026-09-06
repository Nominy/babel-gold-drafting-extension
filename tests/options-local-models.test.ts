import test from 'node:test';
import assert from 'node:assert/strict';
import { setImmediate as waitForImmediate } from 'node:timers/promises';
import fs from 'node:fs';
import { JSDOM } from 'jsdom';

import { boot, type OptionsDependencies } from '../src/options/options';

const SETTINGS_KEY = 'babel_gold_drafting_settings';
const POINTER_KEY = 'babel_gold_local_model_bundle_pointer';
const FIXED_BASE_URL = 'https://reviewgen.ovh/browser-model';
const SAMPLE_URL = `${FIXED_BASE_URL}/sample-russian-15s.wav`;
const REQUIRED_PATHS = [
  'asr/v3_ctc.onnx',
  'asr/v3_ctc.yaml',
  'punctuation/model.int8.onnx',
  'punctuation/config.json',
  'punctuation/tokenizer.json',
  'punctuation/tokenizer_config.json',
  'punctuation/special_tokens_map.json',
  'punctuation/vocab.txt'
];

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
    localModelsEnabled
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
    readAudioDuration: async () => {
      throw new Error('audio duration must not be read');
    },
    transcribeAudio: async () => {
      throw new Error('inference must not run');
    }
  };
}

test('fixed-source local models remain opt-in and store no editable model URL', async () => {
  const dom = createDom();
  const storageData: Record<string, unknown> = {};
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

  assert.equal(dom.window.document.querySelector('#localModelBaseUrl'), null);
  assert.ok(dom.window.document.querySelector('[data-role="local-model-supplied-test"]'));
  assert.ok(dom.window.document.querySelector('#localModelTestAudio'));
  assert.ok(dom.window.document.querySelector('[data-role="local-model-test"]'));
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
  assert.match(status.textContent ?? '', /Download from the Babel model supplier failed/);
  assert.match(status.textContent ?? '', /Check network access to https:\/\/reviewgen\.ovh\/browser-model/);
  assert.deepEqual(storage.getPermissionRequests(), []);

  save.click();
  await waitForImmediate();
  const storedSettings = storage.getStoredSettings();
  assert.equal(storedSettings?.localModelsEnabled, false);
  assert.equal('localModelBaseUrl' in (storedSettings ?? {}), false);
});

test('supplied public-domain sample unlocks enable only after successful inference against the fixed ready bundle', async () => {
  const dom = createDom();
  const cacheName = 'babel-gold-local-models:bundle:installed';
  const storageData: Record<string, unknown> = {
    [SETTINGS_KEY]: defaultSettings(false),
    [POINTER_KEY]: {
      version: 1,
      cacheName,
      baseUrl: FIXED_BASE_URL,
      totalBytes: 0,
      files: REQUIRED_PATHS.map((path) => ({ path, bytes: 0, sha256: '0'.repeat(64) }))
    }
  };
  const storage = installChromeStorage(storageData);
  const matchedUrls: string[] = [];
  Object.assign(globalThis, {
    caches: {
      has: (name: string) => Promise.resolve(name === cacheName),
      open: () =>
        Promise.resolve({
          match: (input: string | Request) => {
            matchedUrls.push(String(input));
            return Promise.resolve(new Response());
          }
        })
    },
    fetch: () => {
      throw new Error('model bundle network fetch must not run for a ready cached bundle');
    }
  });

  let inferenceSucceeds = false;
  let inferenceCalls = 0;
  const suppliedFetches: string[] = [];
  const testedFiles: File[] = [];
  await boot({
    fetchResource: async (input) => {
      suppliedFetches.push(String(input));
      return new Response(new Uint8Array([82, 73, 70, 70]), {
        status: 200,
        headers: { 'content-type': 'audio/wav' }
      });
    },
    readAudioDuration: async (file) => {
      testedFiles.push(file);
      return 15;
    },
    transcribeAudio: async () => {
      inferenceCalls += 1;
      if (!inferenceSucceeds) {
        throw new Error('inference rejected the sample');
      }
      return {
        text: 'тест прошёл',
        durationSeconds: 15,
        tokens: []
      };
    }
  });

  const enabled = dom.window.document.querySelector<HTMLInputElement>('#localModelsEnabled');
  const suppliedTest = dom.window.document.querySelector<HTMLButtonElement>('[data-role="local-model-supplied-test"]');
  const status = dom.window.document.querySelector<HTMLElement>('[data-role="local-model-status"]');
  const save = dom.window.document.querySelector<HTMLButtonElement>('[data-role="save"]');
  assert.ok(enabled);
  assert.ok(suppliedTest);
  assert.ok(status);
  assert.ok(save);
  assert.equal(enabled.disabled, true);
  assert.equal(suppliedTest.disabled, false);
  assert.deepEqual(
    matchedUrls,
    REQUIRED_PATHS.map((path) => `${FIXED_BASE_URL}/${path}`)
  );

  suppliedTest.click();
  await waitForImmediate();
  await waitForImmediate();
  assert.equal(inferenceCalls, 1);
  assert.equal(enabled.disabled, true);
  assert.match(status.textContent ?? '', /Local model test failed: inference rejected the sample/);

  inferenceSucceeds = true;
  suppliedTest.click();
  await waitForImmediate();
  await waitForImmediate();
  assert.equal(inferenceCalls, 2);
  assert.equal(enabled.disabled, false);
  assert.match(status.textContent ?? '', /Test succeeded \(15\.0s\): тест прошёл/);
  assert.deepEqual(suppliedFetches, [SAMPLE_URL, SAMPLE_URL]);
  assert.equal(testedFiles.length, 2);
  assert.equal(testedFiles[0]?.name, 'sample-russian-15s.wav');
  assert.equal(testedFiles[0]?.type, 'audio/wav');

  enabled.checked = true;
  save.click();
  await waitForImmediate();
  const storedSettings = storage.getStoredSettings();
  assert.equal(storedSettings?.localModelsEnabled, true);
  assert.equal('localModelBaseUrl' in (storedSettings ?? {}), false);
});

test('supplied sample fetch failures are actionable and never run inference', async () => {
  const dom = createDom();
  const cacheName = 'babel-gold-local-models:bundle:installed';
  const storageData: Record<string, unknown> = {
    [POINTER_KEY]: {
      version: 1,
      cacheName,
      baseUrl: FIXED_BASE_URL,
      totalBytes: 0,
      files: REQUIRED_PATHS.map((path) => ({ path, bytes: 0, sha256: '0'.repeat(64) }))
    }
  };
  installChromeStorage(storageData);
  Object.assign(globalThis, {
    caches: {
      has: () => Promise.resolve(true),
      open: () => Promise.resolve({ match: () => Promise.resolve(new Response()) })
    }
  });
  let inferenceCalls = 0;
  await boot({
    fetchResource: async () => new Response('missing', { status: 404 }),
    readAudioDuration: async () => 1,
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
  await waitForImmediate();
  assert.equal(inferenceCalls, 0);
  assert.equal(enabled.disabled, true);
  assert.match(status.textContent ?? '', /Babel model supplier returned HTTP 404/);
  assert.equal(status.getAttribute('role'), 'alert');
});

test('settings controls use canonical enum normalization and retain unsaved edits on failure', async (t) => {
  const dom = createDom();
  t.after(() => dom.window.close());
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
