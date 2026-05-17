import test from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_SETTINGS, normalizeSettings } from '../src/core/settings';

test('audio input research preview is off by default', () => {
  assert.equal(DEFAULT_SETTINGS.audioInputEnabled, false);
  assert.equal(normalizeSettings({}).audioInputEnabled, false);
});

test('normalizeSettings only enables audio input from an explicit true value', () => {
  assert.equal(normalizeSettings({ audioInputEnabled: true }).audioInputEnabled, true);
  assert.equal(normalizeSettings({ audioInputEnabled: false }).audioInputEnabled, false);
  assert.equal(normalizeSettings({ audioInputEnabled: 'true' }).audioInputEnabled, false);
});
