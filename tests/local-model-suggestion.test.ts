import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import {
  ensureLocalModelSuggestionMounted,
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
  const dom = new JSDOM(`<!doctype html><html><head></head><body>
    <main>
      <header><button type="button" id="save">Save</button><button type="button" id="feedback">Feedback</button></header>
      <div style="display: flex; min-height: 0">
        <div data-transcript-scroll style="overflow-y: auto; min-height: 0; flex: 1">
          <section data-annotation-id="turn-1"><div data-slate-editor contenteditable="true">Native transcript</div></section>
        </div>
      </div>
    </main>
  </body></html>`);
  let shown = false;
  let markCount = 0;
  let gpuRequestCount = 0;
  let openOptionsCount = 0;
  const dependencies: LocalModelSuggestionDependencies = {
    documentRef: dom.window.document,
    loadSettings: async () => ({ ...DEFAULT_SETTINGS, mode: 'advanced' }),
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
  assert.equal(suggestion.getAttribute('role'), 'region');
  assert.equal(suggestion.getAttribute('aria-modal'), null);
  assert.equal(
    harness.dom.window.document.getElementById(suggestion.getAttribute('aria-labelledby')!)?.textContent,
    'Use your GPU for local AI'
  );
  assert.ok(harness.dom.window.document.getElementById(suggestion.getAttribute('aria-describedby')!));
  assert.equal(suggestion.querySelector('a')?.textContent, 'Set up local AI');
  assert.equal(suggestion.querySelector('button')?.textContent, 'Not now');
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

test('suggestion occupies native scroll flow without overlaying Save or feedback controls', async () => {
  const harness = createHarness();
  const documentRef = harness.dom.window.document;
  const host = documentRef.querySelector<HTMLElement>('[data-transcript-scroll]')!;
  const annotation = host.firstElementChild;
  const save = documentRef.querySelector<HTMLButtonElement>('#save')!;
  const feedback = documentRef.querySelector<HTMLButtonElement>('#feedback')!;
  const nativeHeader = save.parentElement;
  let saveClicks = 0;
  let feedbackClicks = 0;
  save.addEventListener('click', () => { saveClicks += 1; });
  feedback.addEventListener('click', () => { feedbackClicks += 1; });
  save.focus();

  assert.equal(await maybeShowLocalModelSuggestion(harness.dependencies), true);
  const suggestion = documentRef.querySelector<HTMLElement>('#babel-gold-local-model-suggestion')!;
  assert.equal(host.firstElementChild, suggestion);
  assert.equal(suggestion.nextElementSibling, annotation);
  const style = harness.dom.window.getComputedStyle(suggestion);
  assert.equal(style.position, 'static');
  assert.equal(style.display, 'grid');
  assert.equal(style.width, '100%');
  assert.equal(style.maxWidth, '100%');
  assert.equal(harness.dom.window.getComputedStyle(host).overflowY, 'auto');
  assert.equal(save.parentElement, nativeHeader);
  assert.equal(feedback.parentElement, nativeHeader);
  assert.equal(host.contains(save), false);
  assert.equal(host.contains(feedback), false);
  assert.equal(documentRef.activeElement, save);
  assert.equal(harness.dom.window.getComputedStyle(suggestion.querySelector('button')!).whiteSpace, 'normal');
  save.click();
  feedback.click();
  assert.equal(saveClicks, 1);
  assert.equal(feedbackClicks, 1);
});

test('eligible suggestion waits for the native layout before recording it as shown', async () => {
  const harness = createHarness();
  const documentRef = harness.dom.window.document;
  const host = documentRef.querySelector<HTMLElement>('[data-transcript-scroll]')!;
  const parent = host.parentElement!;
  host.remove();

  assert.equal(await maybeShowLocalModelSuggestion(harness.dependencies), false);
  assert.equal(harness.markCount(), 0);
  assert.equal(documentRef.querySelector('#babel-gold-local-model-suggestion'), null);
  assert.equal(await ensureLocalModelSuggestionMounted(documentRef), false);
  parent.append(host);
  assert.equal(await ensureLocalModelSuggestionMounted(documentRef), true);
  assert.equal(host.firstElementChild?.id, 'babel-gold-local-model-suggestion');
  assert.equal(harness.markCount(), 1);
  assert.equal(harness.gpuRequestCount(), 1);
});

test('task replacement remounts the same working controls until explicitly dismissed', async () => {
  const harness = createHarness();
  const documentRef = harness.dom.window.document;
  await maybeShowLocalModelSuggestion(harness.dependencies);
  const suggestion = documentRef.querySelector<HTMLElement>('#babel-gold-local-model-suggestion')!;
  const oldHost = suggestion.parentElement!;
  const parent = oldHost.parentElement!;
  oldHost.remove();
  assert.equal(await ensureLocalModelSuggestionMounted(documentRef), false);
  const newHost = documentRef.createElement('div');
  newHost.setAttribute('data-transcript-scroll', '');
  parent.append(newHost);
  assert.equal(await ensureLocalModelSuggestionMounted(documentRef), true);
  assert.equal(newHost.firstElementChild, suggestion);
  assert.equal(await ensureLocalModelSuggestionMounted(documentRef), true);
  assert.equal(documentRef.querySelectorAll('#babel-gold-local-model-suggestion').length, 1);
  assert.equal(harness.markCount(), 1);
  assert.equal(harness.gpuRequestCount(), 1);
  newHost.replaceChildren();
  assert.equal(await ensureLocalModelSuggestionMounted(documentRef), true);
  assert.equal(newHost.firstElementChild, suggestion);
  suggestion.querySelector<HTMLAnchorElement>('a')!.click();
  assert.equal(harness.openOptionsCount(), 1);
  suggestion.querySelector<HTMLButtonElement>('button')!.click();
  assert.equal(await ensureLocalModelSuggestionMounted(documentRef), false);
  assert.equal(newHost.childElementCount, 0);
  assert.equal(await maybeShowLocalModelSuggestion(harness.dependencies), false);
});

test('inline setup suggestion leaves native readonly state and feedback access unchanged', async () => {
  const harness = createHarness();
  const documentRef = harness.dom.window.document;
  const editor = documentRef.querySelector<HTMLElement>('[data-slate-editor]')!;
  const save = documentRef.querySelector<HTMLButtonElement>('#save')!;
  const feedback = documentRef.querySelector<HTMLButtonElement>('#feedback')!;
  editor.setAttribute('contenteditable', 'false');
  save.disabled = true;
  feedback.focus();
  const annotation = editor.parentElement!;
  const before = annotation.outerHTML;

  assert.equal(await maybeShowLocalModelSuggestion(harness.dependencies), true);
  assert.equal(annotation.outerHTML, before);
  assert.equal(save.disabled, true);
  assert.equal(feedback.disabled, false);
  assert.equal(documentRef.activeElement, feedback);
  const suggestion = documentRef.querySelector<HTMLElement>('#babel-gold-local-model-suggestion')!;
  assert.equal(annotation.contains(suggestion), false);
  assert.equal(suggestion.contains(editor), false);
  suggestion.querySelector<HTMLAnchorElement>('a')!.focus();
  assert.equal(documentRef.activeElement, suggestion.querySelector('a'));
  suggestion.querySelector<HTMLAnchorElement>('a')!.click();
  assert.equal(harness.openOptionsCount(), 1);
});

test('suggestion is suppressed when already shown, enabled, or GPU-ineligible', async (t) => {
  const cases: Array<{
    name: string;
    overrides: Partial<LocalModelSuggestionDependencies>;
    expectedGpuRequests: number;
  }> = [
    { name: 'already shown', overrides: { wasShown: async () => true }, expectedGpuRequests: 0 },
    {
      name: 'simple mode retains but does not suggest advanced local setup',
      overrides: { loadSettings: async () => ({ ...DEFAULT_SETTINGS, mode: 'simple', localModelsEnabled: false }) },
      expectedGpuRequests: 0
    },
    {
      name: 'models enabled',
      overrides: { loadSettings: async () => ({ ...DEFAULT_SETTINGS, mode: 'advanced', localModelsEnabled: true }) },
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
