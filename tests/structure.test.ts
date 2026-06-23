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

test('manifest and build expose the AI broker service worker', () => {
  const manifest = JSON.parse(fs.readFileSync(new URL('../manifest.json', import.meta.url), 'utf8'));
  assert.equal(manifest.background.service_worker, 'dist/background/ai-broker.js');
  assert.deepEqual(manifest.externally_connectable, { ids: ['dldjgploldmldipplklepcpjdjhehald'] });

  const esbuildSource = fs.readFileSync(new URL('../esbuild.config.mjs', import.meta.url), 'utf8');
  assert.match(esbuildSource, /entryPoints: \['src\/background\/ai-broker\.ts'\]/);
  assert.match(esbuildSource, /outfile: 'dist\/background\/ai-broker\.js'/);
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
  assert.match(contentSource, /emit\(\{ type: 'event', event: 'backend-waiting'/);
  assert.match(contentSource, /setInterval/);
  assert.match(contentSource, /port\.postMessage\(\{ type: 'result', response \}\)/);
});

test('AI broker content handler logs backend exceptions before wrapping them', () => {
  const contentSource = fs.readFileSync(new URL('../src/content/ai-broker-content.ts', import.meta.url), 'utf8');
  assert.match(contentSource, /console\.error\('\[Babel Gold Drafting\] Helper AI broker request failed'/);
  assert.match(contentSource, /operation: message\.operation/);
  assert.match(contentSource, /backendBaseUrl: settings\.backendBaseUrl/);
});
