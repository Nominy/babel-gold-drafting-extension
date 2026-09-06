import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

test('manifest exposes the AI broker service worker to allowed Helper extensions', () => {
  const manifest = JSON.parse(fs.readFileSync(new URL('../manifest.json', import.meta.url), 'utf8'));
  assert.equal(manifest.background.service_worker, 'dist/background/ai-broker.js');
  assert.deepEqual(manifest.externally_connectable, {
    ids: [
      'dldjgploldmldipplklepcpjdjhehald',
      'afpcopjodphibggidgicpjnkgnfhhemi'
    ]
  });
});

test('extension permissions and CSP narrowly allow hosted model assets', () => {
  const manifest = JSON.parse(fs.readFileSync(new URL('../manifest.json', import.meta.url), 'utf8'));
  const extensionCsp = manifest.content_security_policy.extension_pages as string;
  const modelHostPermissions = (manifest.host_permissions as string[]).filter((permission) =>
    permission.includes('reviewgen.ovh')
  );

  assert.deepEqual(modelHostPermissions, ['https://reviewgen.ovh/*']);
  assert.deepEqual(manifest.optional_host_permissions, ['http://*/*', 'https://*/*']);
  assert.doesNotMatch(extensionCsp, /\bblob:/);
  assert.match(extensionCsp, /'wasm-unsafe-eval'/);
});

