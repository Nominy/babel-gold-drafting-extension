import assert from 'node:assert/strict';
import test from 'node:test';
import { selectZipEnhancementBackend } from '../src/core/audio-enhancement-backend';
import { enhanceAudioTracks } from '../src/core/audio-enhancement-runtime';

function adapter({ fallback = false, fp16 = true, timestamps = true, storage = 32768 } = {}): GPUAdapter {
  return {
    info: { isFallbackAdapter: fallback },
    features: new Set([...(fp16 ? ['shader-f16'] : []), ...(timestamps ? ['timestamp-query'] : [])]),
    limits: { maxComputeWorkgroupStorageSize: storage, maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256, maxStorageBuffersPerShaderStage: 8 },
  } as unknown as GPUAdapter;
}

test('swarm admission covers missing, denied, software, and kernel-incompatible WebGPU adapters', async () => {
  const environments = [
    {}, { gpu: { requestAdapter: async () => null } },
    { gpu: { requestAdapter: async () => { throw new Error('GPU disabled by policy'); } } },
    ...[adapter({ fallback: true }), adapter({ fp16: false }), adapter({ timestamps: false }), adapter({ storage: 16384 })]
      .map(value => ({ gpu: { requestAdapter: async () => value } })),
  ];
  for (const environment of environments) {
    const selected = await selectZipEnhancementBackend(environment);
    assert.equal(selected.backend, 'swarm');
  }
});

test('a usable hardware adapter stays on local WebGPU', async () => {
  assert.equal((await selectZipEnhancementBackend({ gpu: { requestAdapter: async () => adapter() } })).backend, 'webgpu');
});

test('request cancellation rejects before enhancement touches original audio or GPU resources', async t => {
  const controller = new AbortController();
  const reason = new Error('Owning request disconnected');
  controller.abort(reason);
  const blob = new Blob([new Uint8Array(44)], { type: 'audio/wav' });
  t.mock.method(blob, 'arrayBuffer', async () => { assert.fail('Cancelled source must remain unread'); });
  await assert.rejects(enhanceAudioTracks([
    { trackId: 'original', source: 'native', blob, mimeType: 'audio/wav' }
  ], undefined, { signal: controller.signal }), error => error === reason);
});
