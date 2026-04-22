import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

const REQUIRED_FILES = [
  'src/core/types.ts',
  'src/core/settings.ts',
  'src/core/dom.ts',
  'src/core/transcript.ts',
  'src/core/backend-client.ts',
  'src/content/entry.ts',
  'src/content/overlay.ts',
  'src/options/options.ts',
  'manifest.json',
  'options.html'
];

test('gold drafting extension structure exists', () => {
  for (const relPath of REQUIRED_FILES) {
    assert.equal(fs.existsSync(new URL('../' + relPath, import.meta.url)), true, `${relPath} should exist`);
  }
});
