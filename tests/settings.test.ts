import test from 'node:test';
import assert from 'node:assert/strict';
import {
  DEFAULT_SETTINGS,
  isBrowserLocalMode,
  normalizeL0CustomBaseUrl,
  normalizeSettings,
  loadSettings,
  saveSettings,
  SETTINGS_STORAGE_KEY
} from '../src/core/settings';

test('local browser models stay explicitly opted out unless the stored value is boolean true', () => {
  assert.equal(normalizeSettings({ localModelsEnabled: true }).mode, 'advanced');
  assert.equal(normalizeSettings({ localModelsEnabled: true }).localModelsEnabled, true);
  assert.equal(normalizeSettings({ localModelsEnabled: false }).localModelsEnabled, false);
  assert.equal(normalizeSettings({ localModelsEnabled: 'true' }).localModelsEnabled, false);
  assert.equal(normalizeSettings({ localModelsEnabled: 1 }).localModelsEnabled, false);
});

test('volunteering requires explicit opt-in and preserves stored boolean choices', () => {
  assert.equal(normalizeSettings({ localModelsEnabled: true }).volunteerInferenceEnabled, false);
  assert.equal(normalizeSettings({ volunteerInferenceEnabled: true }).volunteerInferenceEnabled, true);
  assert.equal(normalizeSettings({ volunteerInferenceEnabled: false }).volunteerInferenceEnabled, false);
  assert.equal(normalizeSettings({ volunteerInferenceEnabled: 'true' }).volunteerInferenceEnabled, false);
});


test('local model normalization keeps existing settings fields intact while dropping legacy source URLs', () => {
  const settings = normalizeSettings({
    backendBaseUrl: ' https://backend.example.test/ ',
    projectPreset: 'ru-gold-2sp-v1',
    openRouterApiKey: ' secret ',
    model: ' provider/model ',
    serviceTier: 'priority',
    reasoningEffort: 'medium',
    aiBrokerProvider: 'remote-openrouter',
    l0ReplacementPreviewEnabled: false,
    l0CustomBaseUrl: 'https://l0.example.test',
    l0DontRunLlm: true,
    audioInputEnabled: false,
    localModelsEnabled: true,
    volunteerInferenceEnabled: false,
    localModelBaseUrl: ' https://models.example.test/v1 '
  });

  assert.deepEqual(settings, {
    mode: 'advanced',
    backendBaseUrl: 'https://backend.example.test',
    projectPreset: 'ru-gold-2sp-v1',
    openRouterApiKey: 'secret',
    model: 'provider/model',
    serviceTier: 'priority',
    reasoningEffort: 'medium',
    aiBrokerProvider: 'remote-openrouter',
    l0ReplacementPreviewEnabled: false,
    l0CustomBaseUrl: 'https://l0.example.test',
    l0DontRunLlm: true,
    audioInputEnabled: false,
    localModelsEnabled: true,
    volunteerInferenceEnabled: false
  });
});



test('normalizeSettings persists the L0 replacement controls and trims its URL', () => {
  const settings = normalizeSettings({
    l0ReplacementPreviewEnabled: true,
    l0CustomBaseUrl: ' https://draft.example.test/base///?ignored=1 ',
    l0DontRunLlm: true
  });
  assert.equal(settings.l0ReplacementPreviewEnabled, true);
  assert.equal(settings.l0CustomBaseUrl, 'https://draft.example.test/base');
  assert.equal(settings.l0DontRunLlm, true);
});

test('normalizeL0CustomBaseUrl accepts normalized HTTP bases and falls back to the hosted default', () => {
  assert.equal(normalizeL0CustomBaseUrl('http://localhost:9000///'), 'http://localhost:9000');
  assert.equal(
    normalizeL0CustomBaseUrl('file:///tmp/engine'),
    'https://reviewgen.ovh/a3f73d6cf25fa138be653daaf2d7cd0702c0b2d69c40fb9eaee4e07d4b067dd5'
  );
  assert.equal(normalizeL0CustomBaseUrl('not a URL'), DEFAULT_SETTINGS.l0CustomBaseUrl);
});


test('normalizeSettings keeps only supported OpenRouter service tiers', () => {
  assert.equal(normalizeSettings({ serviceTier: 'flex' }).serviceTier, 'flex');
  assert.equal(normalizeSettings({ serviceTier: 'default' }).serviceTier, 'default');
  assert.equal(normalizeSettings({ serviceTier: 'priority' }).serviceTier, 'priority');
  assert.equal(normalizeSettings({ serviceTier: 'auto' }).serviceTier, 'flex');
});

test('normalizeSettings keeps only supported reasoning efforts', () => {
  assert.equal(normalizeSettings({ reasoningEffort: 'default' }).reasoningEffort, 'default');
  assert.equal(normalizeSettings({ reasoningEffort: 'none' }).reasoningEffort, 'none');
  assert.equal(normalizeSettings({ reasoningEffort: 'minimal' }).reasoningEffort, 'minimal');
  assert.equal(normalizeSettings({ reasoningEffort: 'low' }).reasoningEffort, 'low');
  assert.equal(normalizeSettings({ reasoningEffort: 'medium' }).reasoningEffort, 'medium');
  assert.equal(normalizeSettings({ reasoningEffort: 'high' }).reasoningEffort, 'high');
  assert.equal(normalizeSettings({ reasoningEffort: 'xhigh' }).reasoningEffort, 'xhigh');
  assert.equal(normalizeSettings({ reasoningEffort: 'auto' }).reasoningEffort, 'low');
});

test('normalizeSettings preserves audio opt-in or opt-out and rejects non-booleans to the default', () => {
  assert.equal(normalizeSettings({ audioInputEnabled: true }).audioInputEnabled, true);
  assert.equal(normalizeSettings({ audioInputEnabled: false }).audioInputEnabled, false);
  assert.equal(normalizeSettings({ audioInputEnabled: 'true' }).audioInputEnabled, true);
});

test('normalizeSettings keeps only supported AI broker providers', () => {
  assert.equal(normalizeSettings({ aiBrokerProvider: 'auto' }).aiBrokerProvider, 'auto');
  assert.equal(normalizeSettings({ aiBrokerProvider: 'remote-openrouter' }).aiBrokerProvider, 'remote-openrouter');
  assert.equal(normalizeSettings({ aiBrokerProvider: 'local-gemini-nano' }).aiBrokerProvider, 'local-gemini-nano');
  assert.equal(normalizeSettings({ aiBrokerProvider: 'remote' }).aiBrokerProvider, 'auto');
});

test('raw settings normalize without storage and use Chrome when it becomes available later', async (t) => {
  const previousChrome = globalThis.chrome;
  t.after(() => Object.assign(globalThis, { chrome: previousChrome }));
  Object.assign(globalThis, { chrome: undefined });
  const input: unknown = {
    serviceTier: 'invalid',
    reasoningEffort: 'xhigh',
    aiBrokerProvider: 'remote-openrouter',
    model: '  test-model  '
  };
  const normalized = await saveSettings(input);
  assert.equal(normalized.serviceTier, 'flex');
  assert.equal(normalized.reasoningEffort, 'xhigh');
  assert.equal(normalized.aiBrokerProvider, 'remote-openrouter');
  assert.equal(normalized.model, 'test-model');

  let stored: Record<string, unknown> = {};
  Object.assign(globalThis, {
    chrome: {
      runtime: {},
      storage: { local: {
        get(_key: string, callback: (items: Record<string, unknown>) => void) { callback(stored); },
        set(items: Record<string, unknown>, callback: () => void) {
          stored = items;
          callback();
        }
      } }
    }
  });
  await saveSettings(input);
  assert.deepEqual(stored[SETTINGS_STORAGE_KEY], normalized);
  assert.deepEqual(await loadSettings(), normalized);
});

test('new installs use Local while nonempty legacy settings preserve Advanced workflows', () => {
  for (const input of [undefined, null, {}]) {
    assert.equal(normalizeSettings(input).mode, 'local');
  }
  const legacy = {
    openRouterApiKey: 'existing-key',
    backendBaseUrl: 'https://backend.example',
    model: 'configured/model',
    l0CustomBaseUrl: 'https://self-hosted.example',
    l0ReplacementPreviewEnabled: false,
    l0DontRunLlm: true,
    audioInputEnabled: false,
    localModelsEnabled: true,
    volunteerInferenceEnabled: false,
    aiBrokerProvider: 'local-gemini-nano'
  };
  const advanced = normalizeSettings(legacy);
  assert.equal(advanced.mode, 'advanced');
  for (const [key, value] of Object.entries(legacy)) {
    assert.equal(advanced[key as keyof typeof advanced], value);
  }
  assert.equal(normalizeSettings({ openRouterApiKey: '' }).mode, 'advanced');
});

test('explicit mode takes precedence without changing stored Advanced preferences', () => {
  const advanced = normalizeSettings({
    mode: 'advanced',
    localModelsEnabled: true,
    volunteerInferenceEnabled: true,
    l0DontRunLlm: true,
    model: 'configured/model',
    serviceTier: 'priority',
    reasoningEffort: 'xhigh',
    l0CustomBaseUrl: 'http://localhost:9000'
  });
  const simple = normalizeSettings({ ...advanced, mode: 'simple' });
  assert.deepEqual(simple, { ...advanced, mode: 'simple' });
  assert.deepEqual(normalizeSettings({ ...simple, mode: 'advanced' }), advanced);
  const local = normalizeSettings({ ...advanced, mode: 'local' });
  assert.deepEqual(local, { ...advanced, mode: 'local' });
  assert.equal(isBrowserLocalMode(local), true);
  assert.equal(isBrowserLocalMode({ ...local, localModelsEnabled: false }), true);
  assert.equal(isBrowserLocalMode(advanced), true);
  assert.equal(isBrowserLocalMode({ ...advanced, localModelsEnabled: false }), false);
  assert.equal(isBrowserLocalMode(simple), false);
});

test('loading raw storage distinguishes missing settings from a legacy object and persists mode changes', async (t) => {
  const previousChrome = globalThis.chrome;
  t.after(() => Object.assign(globalThis, { chrome: previousChrome }));
  const stored: Record<string, unknown> = {};
  Object.assign(globalThis, {
    chrome: {
      runtime: {},
      storage: { local: {
        get(_key: string, callback: (items: Record<string, unknown>) => void) { callback(stored); },
        set(items: Record<string, unknown>, callback: () => void) {
          Object.assign(stored, items);
          callback();
        }
      } }
    }
  });
  assert.equal((await loadSettings()).mode, 'local');
  stored[SETTINGS_STORAGE_KEY] = { localModelsEnabled: true, volunteerInferenceEnabled: false };
  const advanced = await loadSettings();
  assert.equal(advanced.mode, 'advanced');
  assert.equal(advanced.localModelsEnabled, true);
  const simple = await saveSettings({ ...advanced, mode: 'simple' });
  assert.deepEqual(await loadSettings(), simple);
  assert.equal(simple.localModelsEnabled, true);
  assert.equal(simple.volunteerInferenceEnabled, false);
  await saveSettings({ ...simple, mode: 'advanced' });
  assert.deepEqual(await loadSettings(), advanced);
});
