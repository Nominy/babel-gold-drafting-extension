import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

const REQUIRED_FILES = [
  'src/core/types.ts',
  'src/core/settings.ts',
  'src/core/dom.ts',
  'src/core/lifecycle.ts',
  'src/core/transcript.ts',
  'src/core/backend-client.ts',
  'src/core/l0-client.ts',
  'src/core/l0-replacement-bridge.ts',
  'src/core/ai-broker-protocol.ts',
  'src/background/ai-broker.ts',
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

test('options page includes passive Ko-fi support link without new host permissions', () => {
  const optionsSource = fs.readFileSync(new URL('../options.html', import.meta.url), 'utf8');
  const manifest = JSON.parse(fs.readFileSync(new URL('../manifest.json', import.meta.url), 'utf8'));
  const hostPermissions = manifest.host_permissions || [];

  assert.match(optionsSource, /https:\/\/ko-fi\.com\/naftsan/);
  assert.match(optionsSource, /if this extension saves you time, consider supporting development on Ko-Fi/);
  assert.equal(hostPermissions.some((permission: string) => /ko-fi\.com/.test(permission)), false);
});

test('Gold Draft work surface includes a small Ko-fi link beside the overlay header', () => {
  const overlaySource = fs.readFileSync(new URL('../src/content/overlay.ts', import.meta.url), 'utf8');

  assert.match(overlaySource, /bgd-header-title/);
  assert.match(overlaySource, /https:\/\/ko-fi\.com\/naftsan/);
  assert.match(overlaySource, /if this extension saves you time, consider supporting development on Ko-Fi/);
  assert.match(overlaySource, /bgd-support-link/);
});

test('L0 replacement is enabled in settings and integrated into Gold generation', () => {
  const optionsSource = fs.readFileSync(new URL('../options.html', import.meta.url), 'utf8');
  const overlaySource = fs.readFileSync(new URL('../src/content/overlay.ts', import.meta.url), 'utf8');
  const manifest = JSON.parse(fs.readFileSync(new URL('../manifest.json', import.meta.url), 'utf8'));

  assert.match(optionsSource, /L0 replacement/);
  assert.match(optionsSource, /id="l0ReplacementPreviewEnabled"/);
  assert.match(optionsSource, /Don't run the LLM/);
  assert.doesNotMatch(optionsSource, /Bearer Token/);
  assert.doesNotMatch(overlaySource, /babel-gold-drafting-l0-button/);
  assert.match(overlaySource, /settings\.l0ReplacementPreviewEnabled/);
  assert.match(overlaySource, /replaceTranscriptWithL0Rows/);
  assert.deepEqual(manifest.optional_host_permissions, ['http://*/*', 'https://*/*']);
});

test('manifest and build expose the AI broker service worker', () => {
  const manifest = JSON.parse(fs.readFileSync(new URL('../manifest.json', import.meta.url), 'utf8'));
  assert.equal(manifest.background.service_worker, 'dist/background/ai-broker.js');
  assert.deepEqual(manifest.externally_connectable, {
    ids: [
      'dldjgploldmldipplklepcpjdjhehald',
      'afpcopjodphibggidgicpjnkgnfhhemi'
    ]
  });

  const esbuildSource = fs.readFileSync(new URL('../esbuild.config.mjs', import.meta.url), 'utf8');
  assert.match(esbuildSource, /entryPoints: \['src\/background\/ai-broker\.ts'\]/);
  assert.match(esbuildSource, /outfile: 'dist\/background\/ai-broker\.js'/);

  const packSource = fs.readFileSync(new URL('../scripts/pack.mjs', import.meta.url), 'utf8');
  assert.match(packSource, /STORE_EXTERNALLY_CONNECTABLE_IDS = \['dldjgploldmldipplklepcpjdjhehald'\]/);
  assert.match(packSource, /externally_connectable: \{\s*ids: STORE_EXTERNALLY_CONNECTABLE_IDS\s*\}/);
});

test('extension permissions and CSP narrowly allow hosted model assets', () => {
  const manifest = JSON.parse(fs.readFileSync(new URL('../manifest.json', import.meta.url), 'utf8'));
  const extensionCsp = manifest.content_security_policy.extension_pages as string;
  const runtimeSource = fs.readFileSync(new URL('../src/core/local-model-runtime.ts', import.meta.url), 'utf8');
  const modelHostPermissions = (manifest.host_permissions as string[]).filter((permission) =>
    permission.includes('reviewgen.ovh')
  );

  assert.deepEqual(modelHostPermissions, ['https://reviewgen.ovh/*']);
  assert.deepEqual(manifest.optional_host_permissions, ['http://*/*', 'https://*/*']);
  assert.doesNotMatch(extensionCsp, /\bblob:/);
  assert.match(extensionCsp, /'wasm-unsafe-eval'/);
  assert.match(runtimeSource, /ort\.env\.wasm\.numThreads = 1/);
});

test('AI broker redistribution review uses the Helper review contract', () => {
  const typesSource = fs.readFileSync(new URL('../src/core/types.ts', import.meta.url), 'utf8');
  const protocolSource = fs.readFileSync(new URL('../src/core/ai-broker-protocol.ts', import.meta.url), 'utf8');
  const contentSource = fs.readFileSync(new URL('../src/content/ai-broker-content.ts', import.meta.url), 'utf8');

  assert.match(typesSource, /fromIndex: number/);
  assert.match(typesSource, /toIndex: number/);
  assert.match(typesSource, /sentenceCount: number/);
  assert.match(typesSource, /groups: BrokerRedistributionGroup\[\]/);
  assert.match(typesSource, /results: BrokerRedistributeTextResult\[\]/);
  assert.match(protocolSource, /groups: BrokerRedistributionGroup\[\]/);
  assert.match(protocolSource, /results: BrokerRedistributeTextResult\[\]/);
  assert.match(contentSource, /groups: message\.groups/);
  assert.match(contentSource, /results: response\.results/);
  assert.doesNotMatch(typesSource, /fromSegmentId/);
  assert.doesNotMatch(typesSource, /toSegmentId/);
  assert.doesNotMatch(contentSource, /group: message\.group/);
  assert.doesNotMatch(contentSource, /review: response\.review/);
  assert.doesNotMatch(typesSource + protocolSource + contentSource, /redistributeTextBatch|redistribute-text-batch/);
});

test('AI broker tab forwarding preserves the configured fallback policy', () => {
  const backgroundSource = fs.readFileSync(new URL('../src/background/ai-broker.ts', import.meta.url), 'utf8');
  assert.match(backgroundSource, /forwardToTab\(tabId, request, fallbackAllowed\)/);
  assert.doesNotMatch(backgroundSource, /tab-broker-unavailable'[\s\S]{0,180}true/);
});

test('AI broker uses external and internal ports for streaming progress', () => {
  const protocolSource = fs.readFileSync(new URL('../src/core/ai-broker-protocol.ts', import.meta.url), 'utf8');
  const backgroundSource = fs.readFileSync(new URL('../src/background/ai-broker.ts', import.meta.url), 'utf8');
  const contentSource = fs.readFileSync(new URL('../src/content/ai-broker-content.ts', import.meta.url), 'utf8');

  assert.match(contentSource, /AI_BROKER_CONTENT_BUILD = 'port-stream-postmortem-/);
  assert.match(contentSource, /data-babel-gold-drafting-ai-broker-build/);
  assert.match(protocolSource, /AI_BROKER_PORT_NAME = 'babel-gold-drafting:ai-broker-port'/);
  assert.match(protocolSource, /AI_BROKER_INTERNAL_PORT_NAME = 'babel-gold-drafting:ai-broker-tab-port'/);
  assert.match(protocolSource, /type: 'event'/);
  assert.match(protocolSource, /event: 'capturing-audio' \| 'calling-backend'/);
  assert.match(protocolSource, /backend-waiting/);
  assert.match(protocolSource, /type: 'result'/);

  assert.match(backgroundSource, /onConnectExternal/);
  assert.match(backgroundSource, /port\.name !== AI_BROKER_PORT_NAME/);
  assert.match(backgroundSource, /chrome\.tabs\.connect\(tabId, \{ name: AI_BROKER_INTERNAL_PORT_NAME \}\)/);
  assert.match(backgroundSource, /tabPort\.onMessage\.addListener/);
  assert.match(backgroundSource, /port\.postMessage\(message\)/);

  assert.match(contentSource, /runtime\.onConnect\.addListener/);
  assert.match(contentSource, /port\.name !== AI_BROKER_INTERNAL_PORT_NAME/);
  assert.match(contentSource, /emit\(\{ type: 'event', event: 'capturing-audio'/);
  assert.match(contentSource, /emit\(\{ type: 'event', event: 'calling-backend'/);
  assert.match(contentSource, /emit\(\{\s*type: 'event',\s*event: 'backend-waiting'/);
  assert.match(contentSource, /setInterval/);
  assert.match(contentSource, /port\.postMessage\(\{ type: 'result', response \}\)/);
});

test('AI broker content handler logs backend exceptions before wrapping them', () => {
  const contentSource = fs.readFileSync(new URL('../src/content/ai-broker-content.ts', import.meta.url), 'utf8');
  assert.match(contentSource, /console\.error\('\[Babel Gold Drafting\] Helper AI broker request failed'/);
  assert.match(contentSource, /operation: message\.operation/);
  assert.match(contentSource, /backendBaseUrl: settings\.backendBaseUrl/);
});
