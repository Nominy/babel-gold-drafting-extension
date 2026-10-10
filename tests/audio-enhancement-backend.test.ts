import assert from 'node:assert/strict';
import test, { type TestContext } from 'node:test';
import { setTimeout as delay } from 'node:timers/promises';
import { IDBFactory } from 'fake-indexeddb';
import { selectZipEnhancementBackend } from '../src/core/audio-enhancement-backend';
import { enhanceAudioTracks, getAudioEnhancementModel } from '../src/core/audio-enhancement-runtime';
import { encodeEnhancementWav } from '../src/core/audio-enhancement-dsp';
import { enhancementSha256 } from '../src/core/audio-enhancement-cache';
import { getEnhancementClientId, type EnhancementSwarmPayload } from '../src/core/audio-enhancement-swarm-protocol';

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

function installGlobal(t: TestContext, name: string, value: unknown): void {
  const previous = Object.getOwnPropertyDescriptor(globalThis, name);
  Object.defineProperty(globalThis, name, { configurable: true, value });
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, name, previous);
    else Reflect.deleteProperty(globalThis, name);
  });
}

function originalPair(prefix: string) {
  const bytes = encodeEnhancementWav(new Float32Array([0, 0.2, -0.2, 0.1]), 16000, 4);
  return ['A', 'B'].map(lane => ({
    trackId: `${prefix}:${lane}`, speakerKey: lane, trackLabel: `Original ${lane}`, source: 'native',
    mimeType: 'audio/wav', blob: new Blob([bytes], { type: 'audio/wav' })
  }));
}

test('explicit remote admission bypasses probing even a usable hardware GPU', async () => {
  let probes = 0;
  const result = await selectZipEnhancementBackend({ gpu: { requestAdapter: async () => { probes++; return adapter(); } } }, 'swarm');
  assert.equal(result.backend, 'swarm');
  assert.equal(probes, 0);
  if (result.backend === 'swarm') assert.match(result.reason, /Original recordings leave this device/);
});

test('forced remote uploads with usable WebGPU, then reuses the verified pair without probing, host admission, or upload', async t => {
  const tracks = originalPair('cached-remote');
  const model = getAudioEnhancementModel(), baseUrl = 'https://inference.example/private';
  let probes = 0, uploads = 0;
  installGlobal(t, 'navigator', { gpu: { requestAdapter: async () => { probes++; return adapter(); } } });
  installGlobal(t, 'indexedDB', new IDBFactory());
  t.mock.method(globalThis, 'fetch', async (url: string, init: RequestInit) => {
    assert.equal(url, `${baseUrl}/v1/enhance`, 'forced remote cannot fetch local model assets');
    uploads++;
    const form = init.body as FormData;
    const payload = JSON.parse(String(form.get('payload'))) as EnhancementSwarmPayload;
    assert.equal(payload.clientId, getEnhancementClientId());
    assert.deepEqual(payload.model, model);
    const output = new FormData();
    const metadata = [];
    for (let index = 0; index < tracks.length; index++) {
      const source = payload.tracks[index];
      const original = await tracks[index].blob.arrayBuffer();
      assert.deepEqual(await (form.get(source.fieldName) as Blob).arrayBuffer(), original);
      metadata.push({
        trackId: source.trackId, speakerKey: source.speakerKey, trackLabel: source.trackLabel,
        sourceSha256: source.sourceSha256, sampleRate: source.sampleRate, frameCount: source.frameCount,
        mimeType: 'audio/wav', wavSha256: await enhancementSha256(original), totalBytes: original.byteLength, chunkCount: 1
      });
      output.append(source.fieldName, new Blob([original], { type: 'audio/wav' }), `enhanced-${index}.wav`);
    }
    output.append('result', JSON.stringify({
      ok: true, provider: 'swarm', taskId: payload.taskId, model: model.id, modelSha256: model.sha256, tracks: metadata
    }));
    return new Response(output);
  });
  const options = {
    backend: 'swarm' as const, swarmBaseUrl: baseUrl, taskId: 'native-remote',
    runLocal: async <T>(_action: () => Promise<T>): Promise<T> => { assert.fail('Remote and cache work cannot acquire host inference admission'); }
  };
  const notices: string[] = [];
  const first = await enhanceAudioTracks(tracks, progress => {
    if (progress.backendReason) notices.push(progress.backendReason);
  }, options);
  assert.equal(first.provider, 'swarm');
  assert.equal(first.cacheStatus, 'stored');
  assert.ok(notices.some(notice => /off-device/.test(notice)));
  const phases: string[] = [];
  const cached = await enhanceAudioTracks(tracks, progress => {
    phases.push(progress.phase);
    assert.equal(progress.backend, undefined, 'cache reuse does not claim remote execution');
  }, options);
  assert.equal(cached.provider, 'browser-local');
  assert.equal(cached.cacheStatus, 'hit');
  assert.deepEqual(phases, ['cache-lookup', 'cache-lookup', 'cache-hit', 'cache-hit']);
  assert.deepEqual(cached.tracks, first.tracks);
  assert.equal(uploads, 1);
  assert.equal(probes, 0);
});

test('remote failure never falls back to local GPU, even when hardware is usable', async t => {
  let probes = 0, requests = 0;
  installGlobal(t, 'navigator', { gpu: { requestAdapter: async () => { probes++; return adapter(); } } });
  t.mock.method(globalThis, 'fetch', async (url: string) => {
    assert.equal(url, 'https://inference.example/v1/enhance');
    requests++;
    return Response.json({ error: 'No compatible remote GPU available' }, { status: 503 });
  });
  await assert.rejects(enhanceAudioTracks(originalPair('remote-failure'), undefined, {
    backend: 'swarm', swarmBaseUrl: 'https://inference.example', taskId: 'native-failure', cache: false
  }), /No compatible remote GPU available.*Originals remain selected/);
  assert.equal(probes, 0);
  assert.equal(requests, 1);
});

test('a remote wait allows localOnly volunteer work and its failure never waits for or disposes that local GPU setup', async t => {
  const remoteResponse = Promise.withResolvers<Response>(), uploadStarted = Promise.withResolvers<void>();
  const localAssets = Promise.withResolvers<Response>(), localStarted = Promise.withResolvers<void>();
  let probes = 0, localFetches = 0;
  installGlobal(t, 'navigator', { gpu: { requestAdapter: async () => { probes++; return adapter(); } } });
  installGlobal(t, 'chrome', { runtime: { getURL: (path: string) => `chrome-extension://test/${path}` } });
  installGlobal(t, 'indexedDB', new IDBFactory());
  t.mock.method(globalThis, 'fetch', async (url: string) => {
    if (url === 'https://inference.example/v1/enhance') {
      uploadStarted.resolve();
      return remoteResponse.promise;
    }
    if (url.includes('/v1/queue/')) return new Response(null, { status: 404 });
    assert.match(url, /^chrome-extension:\/\/test\/dist\/models\/zipenhancer-webgpu\./);
    localFetches++;
    localStarted.resolve();
    return localAssets.promise;
  });
  const remote = enhanceAudioTracks(originalPair('waiting-remote'), undefined, {
    backend: 'swarm', swarmBaseUrl: 'https://inference.example', taskId: 'native-wait'
  });
  const remoteRejected = assert.rejects(remote, /Remote unavailable/);
  await uploadStarted.promise;
  assert.equal(probes, 0);
  const queuedOwnerController = new AbortController();
  const queuedOwner = enhanceAudioTracks(originalPair('queued-owner'), () => {
    assert.fail('A later owner cannot select or overwrite the active owner pair cache');
  }, { backend: 'auto', signal: queuedOwnerController.signal });
  const queuedOwnerRejected = assert.rejects(queuedOwner, /cancelled queued owner/);
  const local = enhanceAudioTracks(originalPair('volunteer-lease'), undefined, {
    backend: 'swarm', localOnly: true, cache: false, taskId: 'leased-task'
  });
  const localRejected = assert.rejects(local, /packaged ZipEnhancer assets could not be loaded/);
  let entered = false, remoteFinished = false;
  try {
    entered = await Promise.race([localStarted.promise.then(() => true), delay(2000, false)]);
    queuedOwnerController.abort(new Error('cancelled queued owner'));
    remoteResponse.resolve(Response.json({ error: 'Remote unavailable' }, { status: 503 }));
    remoteFinished = await Promise.race([remoteRejected.then(() => true), delay(2000, false)]);
  } finally {
    queuedOwnerController.abort(new Error('cancelled queued owner'));
    remoteResponse.resolve(Response.json({ error: 'Remote unavailable' }, { status: 503 }));
    localAssets.resolve(new Response(null, { status: 503 }));
    await Promise.all([remoteRejected, localRejected, queuedOwnerRejected]);
  }
  assert.equal(entered, true, 'a remote request must not hold enhancementTail against a volunteer lease');
  assert.equal(remoteFinished, true, 'a remote failure must not release or await another request GPU resources');
  assert.equal(probes, 1, 'only the localOnly lease probes the GPU despite saved remote preference');
  assert.equal(localFetches, 2);
});

test('a localOnly worker with no usable GPU rejects instead of re-delegating', async t => {
  installGlobal(t, 'navigator', {});
  const fetch = t.mock.method(globalThis, 'fetch', async () => { assert.fail('A lease cannot delegate again'); });
  await assert.rejects(enhanceAudioTracks(originalPair('ineligible-lease'), undefined, {
    backend: 'swarm', localOnly: true, cache: false, taskId: 'leased-task'
  }), /no usable hardware WebGPU adapter/);
  assert.equal(fetch.mock.callCount(), 0);
});

test('Automatic local inference still acquires host admission before loading any GPU resources', async t => {
  installGlobal(t, 'navigator', { gpu: { requestAdapter: async () => adapter() } });
  const fetch = t.mock.method(globalThis, 'fetch', async () => { assert.fail('Host admission precedes all local model resources'); });
  let admissions = 0;
  await assert.rejects(enhanceAudioTracks(originalPair('auto-admission'), undefined, {
    backend: 'auto', cache: false,
    runLocal: async () => {
      admissions++;
      throw new Error('Host admission cancelled');
    }
  }), /Host admission cancelled/);
  assert.equal(admissions, 1);
  assert.equal(fetch.mock.callCount(), 0);
});
