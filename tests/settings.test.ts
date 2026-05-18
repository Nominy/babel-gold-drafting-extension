import test from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_SETTINGS, normalizeSettings } from '../src/core/settings';

test('audio input research preview is off by default', () => {
  assert.equal(DEFAULT_SETTINGS.audioInputEnabled, false);
  assert.equal(normalizeSettings({}).audioInputEnabled, false);
});

test('OpenRouter service tier defaults to flex', () => {
  assert.equal(DEFAULT_SETTINGS.serviceTier, 'flex');
  assert.equal(normalizeSettings({}).serviceTier, 'flex');
});

test('normalizeSettings keeps only supported OpenRouter service tiers', () => {
  assert.equal(normalizeSettings({ serviceTier: 'flex' }).serviceTier, 'flex');
  assert.equal(normalizeSettings({ serviceTier: 'default' }).serviceTier, 'default');
  assert.equal(normalizeSettings({ serviceTier: 'priority' }).serviceTier, 'priority');
  assert.equal(normalizeSettings({ serviceTier: 'auto' }).serviceTier, 'flex');
});

test('normalizeSettings only enables audio input from an explicit true value', () => {
  assert.equal(normalizeSettings({ audioInputEnabled: true }).audioInputEnabled, true);
  assert.equal(normalizeSettings({ audioInputEnabled: false }).audioInputEnabled, false);
  assert.equal(normalizeSettings({ audioInputEnabled: 'true' }).audioInputEnabled, false);
});
