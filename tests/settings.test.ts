import test from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_SETTINGS, normalizeL0CustomBaseUrl, normalizeSettings } from '../src/core/settings';

test('audio-enhanced drafting is on by default and explicit opt-out persists', () => {
  assert.equal(DEFAULT_SETTINGS.audioInputEnabled, true);
  assert.equal(normalizeSettings({}).audioInputEnabled, true);
  assert.equal(normalizeSettings({ audioInputEnabled: false }).audioInputEnabled, false);
});

test('L0 replacement is on with the hosted base while LLM bypass stays off by default', () => {
  assert.equal(DEFAULT_SETTINGS.l0ReplacementPreviewEnabled, true);
  assert.equal(normalizeSettings({}).l0ReplacementPreviewEnabled, true);
  assert.equal(
    normalizeSettings({ l0ReplacementPreviewEnabled: false }).l0ReplacementPreviewEnabled,
    false
  );
  assert.equal(DEFAULT_SETTINGS.l0DontRunLlm, false);
  assert.equal(
    DEFAULT_SETTINGS.l0CustomBaseUrl,
    'https://reviewgen.ovh/a3f73d6cf25fa138be653daaf2d7cd0702c0b2d69c40fb9eaee4e07d4b067dd5'
  );
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
