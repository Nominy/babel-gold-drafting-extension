import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import {
  isSuitableLocalModelGpu,
  maybeShowLocalModelSuggestion,
  type GpuAdapter,
  type LocalModelSuggestionDependencies
} from '../src/content/local-model-suggestion';
import { DEFAULT_SETTINGS } from '../src/core/settings';

const CAPABLE_GPU: GpuAdapter = {
  info: { isFallbackAdapter: false },
  limits: {
    maxBufferSize: 256 * 1024 * 1024,
    maxStorageBufferBindingSize: 128 * 1024 * 1024
  }
};

function createHarness(overrides: Partial<LocalModelSuggestionDependencies> = {}) {
  const dom = new JSDOM('<!doctype html><html><head></head><body></body></html>');
  let shown = false;
  let markCount = 0;
  let gpuRequestCount = 0;
  let openOptionsCount = 0;
  const dependencies: LocalModelSuggestionDependencies = {
    documentRef: dom.window.document,
    loadSettings: async () => ({ ...DEFAULT_SETTINGS }),
    wasShown: async () => shown,
    markShown: async () => {
      shown = true;
      markCount += 1;
    },
    requestGpuAdapter: async () => {
      gpuRequestCount += 1;
      return CAPABLE_GPU;
    },
    optionsUrl: 'https://extension.test/options.html#local-model-heading',
    openOptions: async () => {
      openOptionsCount += 1;
    },
    ...overrides
  };
  return {
    dom,
    dependencies,
    markCount: () => markCount,
    gpuRequestCount: () => gpuRequestCount,
    openOptionsCount: () => openOptionsCount
  };
}

test('GPU suitability rejects missing, software, and undersized adapters', () => {
  assert.equal(isSuitableLocalModelGpu(null), false);
  assert.equal(isSuitableLocalModelGpu({ ...CAPABLE_GPU, info: { isFallbackAdapter: true } }), false);
  assert.equal(
    isSuitableLocalModelGpu({
      limits: {
        maxBufferSize: 256 * 1024 * 1024 - 1,
        maxStorageBufferBindingSize: 128 * 1024 * 1024
      }
    }),
    false
  );
  assert.equal(isSuitableLocalModelGpu(CAPABLE_GPU), true);
});

test('eligible Babel visit shows one accessible suggestion linked to local model settings', async () => {
  const harness = createHarness();

  assert.equal(await maybeShowLocalModelSuggestion(harness.dependencies), true);
  assert.equal(harness.gpuRequestCount(), 1);
  assert.equal(harness.markCount(), 1);
  const suggestion = harness.dom.window.document.querySelector<HTMLElement>(
    '#babel-gold-local-model-suggestion'
  );
  assert.ok(suggestion);
  assert.equal(suggestion.getAttribute('role'), 'dialog');
  assert.match(suggestion.textContent || '', /Use your GPU for local AI/);
  assert.equal(
    suggestion.querySelector<HTMLAnchorElement>('a')?.href,
    'https://extension.test/options.html#local-model-heading'
  );
  suggestion.querySelector<HTMLAnchorElement>('a')?.click();
  assert.equal(harness.openOptionsCount(), 1);

  suggestion.querySelector<HTMLButtonElement>('button')?.click();
  assert.equal(harness.dom.window.document.querySelector('#babel-gold-local-model-suggestion'), null);
  assert.equal(await maybeShowLocalModelSuggestion(harness.dependencies), false);
  assert.equal(harness.markCount(), 1);
});

test('suggestion is suppressed when already shown, enabled, or GPU-ineligible', async (t) => {
  const cases: Array<{
    name: string;
    overrides: Partial<LocalModelSuggestionDependencies>;
    expectedGpuRequests: number;
  }> = [
    { name: 'already shown', overrides: { wasShown: async () => true }, expectedGpuRequests: 0 },
    {
      name: 'models enabled',
      overrides: { loadSettings: async () => ({ ...DEFAULT_SETTINGS, localModelsEnabled: true }) },
      expectedGpuRequests: 0
    },
    {
      name: 'no suitable GPU',
      overrides: { requestGpuAdapter: async () => null },
      expectedGpuRequests: 0
    }
  ];

  for (const testCase of cases) {
    await t.test(testCase.name, async () => {
      const harness = createHarness(testCase.overrides);
      assert.equal(await maybeShowLocalModelSuggestion(harness.dependencies), false);
      assert.equal(harness.markCount(), 0);
      assert.equal(
        harness.dom.window.document.querySelector('#babel-gold-local-model-suggestion'),
        null
      );
      assert.equal(harness.gpuRequestCount(), testCase.expectedGpuRequests);
    });
  }
});
