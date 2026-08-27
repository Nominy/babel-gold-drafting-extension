import test from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_SETTINGS, normalizeL0CustomBaseUrl, normalizeSettings } from '../src/core/settings';

test('audio input research preview is off by default', () => {
  assert.equal(DEFAULT_SETTINGS.audioInputEnabled, false);
  assert.equal(normalizeSettings({}).audioInputEnabled, false);
});

test('L0 replacement preview and LLM bypass are off by default', () => {
  assert.equal(DEFAULT_SETTINGS.l0ReplacementPreviewEnabled, false);
  assert.equal(DEFAULT_SETTINGS.l0DontRunLlm, false);
  assert.equal(DEFAULT_SETTINGS.l0CustomBaseUrl, 'http://127.0.0.1:8767');
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

test('normalizeL0CustomBaseUrl accepts only normalized HTTP bases', () => {
  assert.equal(normalizeL0CustomBaseUrl('http://localhost:9000///'), 'http://localhost:9000');
  assert.equal(normalizeL0CustomBaseUrl('file:///tmp/engine'), DEFAULT_SETTINGS.l0CustomBaseUrl);
  assert.equal(normalizeL0CustomBaseUrl('not a URL'), DEFAULT_SETTINGS.l0CustomBaseUrl);
});

test('AI broker provider defaults to automatic remote fallback', () => {
  assert.equal(DEFAULT_SETTINGS.aiBrokerProvider, 'auto');
  assert.equal(normalizeSettings({}).aiBrokerProvider, 'auto');
});

test('OpenRouter service tier defaults to flex', () => {
  assert.equal(DEFAULT_SETTINGS.serviceTier, 'flex');
  assert.equal(normalizeSettings({}).serviceTier, 'flex');
});

test('reasoning effort defaults to low to preserve current drafting behavior', () => {
  assert.equal(DEFAULT_SETTINGS.reasoningEffort, 'low');
  assert.equal(normalizeSettings({}).reasoningEffort, 'low');
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

test('normalizeSettings only enables audio input from an explicit true value', () => {
  assert.equal(normalizeSettings({ audioInputEnabled: true }).audioInputEnabled, true);
  assert.equal(normalizeSettings({ audioInputEnabled: false }).audioInputEnabled, false);
  assert.equal(normalizeSettings({ audioInputEnabled: 'true' }).audioInputEnabled, false);
});

test('normalizeSettings keeps only supported AI broker providers', () => {
  assert.equal(normalizeSettings({ aiBrokerProvider: 'auto' }).aiBrokerProvider, 'auto');
  assert.equal(normalizeSettings({ aiBrokerProvider: 'remote-openrouter' }).aiBrokerProvider, 'remote-openrouter');
  assert.equal(normalizeSettings({ aiBrokerProvider: 'local-gemini-nano' }).aiBrokerProvider, 'local-gemini-nano');
  assert.equal(normalizeSettings({ aiBrokerProvider: 'remote' }).aiBrokerProvider, 'auto');
});
