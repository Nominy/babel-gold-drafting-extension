import test from 'node:test';
import assert from 'node:assert/strict';
import { IDBFactory, IDBObjectStore } from 'fake-indexeddb';
import type { AudioEnhancementChunk, AudioEnhancementProgress } from '@nominy/babel-babel-runtime';
import {
  AUDIO_ENHANCEMENT_CACHE_DATABASE, AUDIO_ENHANCEMENT_CACHE_KEY, AUDIO_ENHANCEMENT_CACHE_STORE,
  createAudioEnhancementPairCache, enhancementSha256, runWithAudioEnhancementPairCache,
  type PreparedEnhancementSource, type ReadyEnhancementPair
} from '../src/core/audio-enhancement-cache';
import { decodeEnhancementWav, encodeEnhancementWav } from '../src/core/audio-enhancement-dsp';
import type { EnhancedAudioTrack } from '../src/core/audio-enhancement-runtime';
import { LOCAL_MODEL_AUDIO_CHUNK_BYTES, LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE, LOCAL_MODEL_OFFSCREEN_VERSION, decodeAudioChunk } from '../src/core/local-model-offscreen-protocol';
import type { CapturedAudioTrack } from '../src/core/types';
import { createLocalModelClient, LocalModelBridgeError } from '../src/core/local-model-client';
import { createLocalModelHost } from '../src/offscreen/local-model-host';

const model = { id: 'verified-test-model', sha256: 'a'.repeat(64) };
function originals(seed = 1): CapturedAudioTrack[] {
  return ['lane-left', 'lane-right'].map((trackId, index) => {
    const samples = Float32Array.from({ length: 83 + index * 7 }, (_, frame) => Math.sin((frame + seed) / (9 + index)) * 0.25);
    return { trackId, speakerKey: `Speaker ${index + 1}`, trackLabel: `Microphone ${index + 1}`, source: `original-${index}.wav`,
      blob: new Blob([encodeEnhancementWav(samples, 16000 + index * 6000, samples.length)]), mimeType: 'audio/wav' };
  });
}
async function generateFixtureEnhancement(sources: PreparedEnhancementSource[]): Promise<EnhancedAudioTrack[]> {
  const enhanced: EnhancedAudioTrack[] = [];
  for (const source of sources) {
    const { samples, sampleRate } = decodeEnhancementWav(source.bytes);
    for (let frame = 0; frame < samples.length; frame++) samples[frame] *= 0.75;
    const bytes = encodeEnhancementWav(samples, sampleRate, samples.length);
    enhanced.push({ bytes, metadata: { trackId: source.trackId, speakerKey: source.track.speakerKey ?? source.trackId,
      trackLabel: source.track.trackLabel ?? source.trackId, mimeType: 'audio/wav', sampleRate, frameCount: samples.length,
      sourceSha256: source.sourceSha256, wavSha256: await enhancementSha256(bytes.buffer), totalBytes: bytes.byteLength,
      chunkCount: Math.ceil(bytes.byteLength / LOCAL_MODEL_AUDIO_CHUNK_BYTES) } });
  }
  return enhanced;
}
async function openRaw(factory: IDBFactory, version = 1): Promise<IDBDatabase> {
  const { promise, resolve, reject } = Promise.withResolvers<IDBDatabase>();
  const request = factory.open(AUDIO_ENHANCEMENT_CACHE_DATABASE, version);
  request.onsuccess = () => resolve(request.result);
  request.onerror = () => reject(request.error);
  return promise;
}
async function rows(factory: IDBFactory): Promise<unknown[]> {
  const db = await openRaw(factory);
  const { promise, resolve, reject } = Promise.withResolvers<unknown[]>();
  const transaction = db.transaction(AUDIO_ENHANCEMENT_CACHE_STORE, 'readonly');
  const request = transaction.objectStore(AUDIO_ENHANCEMENT_CACHE_STORE).getAll();
  transaction.oncomplete = () => { db.close(); resolve(request.result as unknown[]); };
  transaction.onabort = transaction.onerror = () => { db.close(); reject(transaction.error); };
  return promise;
}
async function replaceRaw(factory: IDBFactory, value: unknown): Promise<void> {
  const db = await openRaw(factory);
  const { promise, resolve, reject } = Promise.withResolvers<void>();
  const transaction = db.transaction(AUDIO_ENHANCEMENT_CACHE_STORE, 'readwrite');
  transaction.objectStore(AUDIO_ENHANCEMENT_CACHE_STORE).put(value, AUDIO_ENHANCEMENT_CACHE_KEY);
  transaction.oncomplete = () => { db.close(); resolve(); };
  transaction.onabort = transaction.onerror = () => { db.close(); reject(transaction.error); };
  return promise;
}

function assertPendingOnly(retained: unknown[]): void {
  assert.equal(retained.length, 1);
  const row = retained[0];
  assert.ok(row !== null && typeof row === 'object' && 'state' in row);
  assert.equal(row.state, 'pending');
  assert.equal('lanes' in row, false, 'no partial or previously selected enhanced bytes may remain');
}

test('a fresh controller reuses both verified lanes in requested order with current labels and no generation', async () => {
  const indexedDB = new IDBFactory();
  const cache = createAudioEnhancementPairCache({ indexedDB });
  const initial = await runWithAudioEnhancementPairCache(originals(), model, cache, generateFixtureEnhancement);
  assert.equal(initial.cacheStatus, 'stored');
  cache.close();
  const restarted = createAudioEnhancementPairCache({ indexedDB });
  const reordered = originals().reverse().map((track, index) => ({ ...track, speakerKey: `Current speaker ${index}`, trackLabel: `Current label ${index}` }));
  const progress: AudioEnhancementProgress[] = [];
  const hit = await runWithAudioEnhancementPairCache(reordered, { ...model, id: 'current-artifact-label' }, restarted,
    async () => { assert.fail('A validated hit must not decode PCM, initialize GPU resources, or run model/DSP inference'); },
    (value) => { progress.push(value); });
  assert.equal(hit.cacheStatus, 'hit');
  assert.equal(hit.model, 'current-artifact-label');
  assert.deepEqual(progress.map((value) => value.phase), ['cache-lookup', 'cache-lookup', 'cache-hit', 'cache-hit']);
  assert.ok(progress.every((value) => value.completedChunks === 0 && value.totalChunks === 0));
  for (let index = 0; index < hit.tracks.length; index++) {
    const track = hit.tracks[index], original = initial.tracks.find((candidate) => candidate.metadata.trackId === track.metadata.trackId)!;
    assert.equal(track.metadata.trackId, reordered[index].trackId);
    assert.equal(track.metadata.speakerKey, `Current speaker ${index}`);
    assert.equal(track.metadata.trackLabel, `Current label ${index}`);
    assert.deepEqual(track.bytes, original.bytes);
    assert.equal(await enhancementSha256(track.bytes.buffer), original.metadata.wavSha256);
    assert.equal(track.metadata.sourceSha256, original.metadata.sourceSha256);
  }
  const stored = await rows(indexedDB);
  assert.equal(stored.length, 1);
  const ready = stored[0] as ReadyEnhancementPair;
  assert.deepEqual(Object.keys(ready).sort(), ['key', 'lanes', 'modelSha256', 'nonce', 'schemaVersion', 'state']);
  for (const lane of ready.lanes) assert.deepEqual(Object.keys(lane).sort(), ['frameCount', 'sampleRate', 'sourceSha256', 'trackId', 'wav', 'wavSha256']);
  restarted.close();
});

test('changing a source, its clock, stable lane ID, or model identity replaces the only retained pair', async () => {
  const indexedDB = new IDBFactory(), cache = createAudioEnhancementPairCache({ indexedDB });
  let generated = 0;
  const generate = async (sources: PreparedEnhancementSource[]) => { generated++; return generateFixtureEnhancement(sources); };
  await runWithAudioEnhancementPairCache(originals(), model, cache, generate);
  let prior = (await rows(indexedDB))[0] as ReadyEnhancementPair;
  const changed = originals(2);
  const clockBytes = await changed[1].blob.arrayBuffer();
  new DataView(clockBytes).setUint32(24, 24000, true);
  new DataView(clockBytes).setUint32(28, 48000, true);
  const clockChanged = [changed[0], { ...changed[1], blob: new Blob([clockBytes]) }];
  const laneChanged = [clockChanged[0], { ...clockChanged[1], trackId: 'replacement-lane' }];
  for (const [tracks, artifact] of [[changed, model], [clockChanged, model], [laneChanged, model], [laneChanged, { ...model, sha256: 'b'.repeat(64) }]] as const) {
    const result = await runWithAudioEnhancementPairCache(tracks, artifact, cache, generate);
    const stored = await rows(indexedDB);
    assert.equal(stored.length, 1);
    const current = stored[0] as ReadyEnhancementPair;
    assert.notEqual(current.key, prior.key);
    assert.equal(current.state, 'ready');
    assert.equal(current.modelSha256, artifact.sha256);
    for (const track of result.tracks) {
      const lane = current.lanes.find((value) => value.trackId === track.metadata.trackId)!;
      assert.deepEqual(new Uint8Array(lane.wav), track.bytes);
      const sourceBytes = await tracks.find((value) => value.trackId === lane.trackId)!.blob.arrayBuffer();
      assert.equal(lane.sourceSha256, await enhancementSha256(sourceBytes));
    }
    prior = current;
  }
  await runWithAudioEnhancementPairCache(originals(), model, cache, generate);
  assert.equal(generated, 6, 'returning to an evicted pair must regenerate both lanes');
  cache.close();
});

test('a second-lane model failure leaves no complete pair and propagates the real error', async () => {
  const indexedDB = new IDBFactory(), cache = createAudioEnhancementPairCache({ indexedDB });
  await runWithAudioEnhancementPairCache(originals(), model, cache, generateFixtureEnhancement);
  await assert.rejects(runWithAudioEnhancementPairCache(originals(2), model, cache, async (sources) => {
    await generateFixtureEnhancement([sources[0]]);
    const retained = await rows(indexedDB);
    assertPendingOnly(retained);
    throw new Error('Hardware WebGPU device lost during second-lane enhancement');
  }), /Hardware WebGPU device lost during second-lane/);
  const afterFailure = await rows(indexedDB);
  assertPendingOnly(afterFailure);
  let regenerated = 0;
  const recovered = await runWithAudioEnhancementPairCache(originals(2), model, cache, async (sources) => {
    regenerated += sources.length;
    return generateFixtureEnhancement(sources);
  });
  assert.equal(regenerated, 2);
  assert.equal(recovered.cacheStatus, 'stored');
  cache.close();
});

test('a pre-aborted request does not evict the retained pair or start generation', async () => {
  const indexedDB = new IDBFactory(), cache = createAudioEnhancementPairCache({ indexedDB });
  await runWithAudioEnhancementPairCache(originals(), model, cache, generateFixtureEnhancement);
  const before = await rows(indexedDB), controller = new AbortController(), reason = new Error('Cancelled before selection');
  controller.abort(reason);
  await assert.rejects(runWithAudioEnhancementPairCache(originals(2), model, cache,
    async () => { assert.fail('An aborted request must not generate audio'); }, undefined, controller.signal), error => error === reason);
  assert.deepEqual(await rows(indexedDB), before);
  cache.close();
});

test('abort after complete generation cannot publish a reusable enhanced pair', async () => {
  const indexedDB = new IDBFactory(), cache = createAudioEnhancementPairCache({ indexedDB });
  const controller = new AbortController(), reason = new Error('Cancelled before pair commit');
  await assert.rejects(runWithAudioEnhancementPairCache(originals(), model, cache, async sources => {
    const result = await generateFixtureEnhancement(sources);
    controller.abort(reason);
    return result;
  }, undefined, controller.signal), error => error === reason);
  assertPendingOnly(await rows(indexedDB));
  cache.close();
});

test('abort during the ready-row transaction rolls back both cached output lanes', async t => {
  const indexedDB = new IDBFactory(), cache = createAudioEnhancementPairCache({ indexedDB });
  const controller = new AbortController(), reason = new Error('Cancelled during pair transaction');
  const put = IDBObjectStore.prototype.put;
  IDBObjectStore.prototype.put = function(value: unknown, key?: IDBValidKey) {
    const request = put.call(this, value, key);
    if (value !== null && typeof value === 'object' && 'state' in value && value.state === 'ready') controller.abort(reason);
    return request;
  };
  t.after(() => { IDBObjectStore.prototype.put = put; cache.close(); });
  await assert.rejects(runWithAudioEnhancementPairCache(originals(), model, cache,
    generateFixtureEnhancement, undefined, controller.signal), error => error === reason);
  assertPendingOnly(await rows(indexedDB));
});

test('a newer selection wins when an older complete result finishes late', async () => {
  const indexedDB = new IDBFactory();
  const older = createAudioEnhancementPairCache({ indexedDB });
  const newer = createAudioEnhancementPairCache({ indexedDB });
  const started = Promise.withResolvers<void>(), finish = Promise.withResolvers<void>();
  const first = runWithAudioEnhancementPairCache(originals(), model, older, async (sources) => {
    started.resolve(); await finish.promise; return generateFixtureEnhancement(sources);
  });
  await started.promise;
  const second = await runWithAudioEnhancementPairCache(originals(2), model, newer, generateFixtureEnhancement);
  const retained = (await rows(indexedDB))[0] as ReadyEnhancementPair;
  finish.resolve();
  const stale = await first;
  assert.equal(stale.cacheStatus, 'unavailable');
  assert.match(stale.cacheMessage!, /newer recording selection/i);
  assert.equal(stale.tracks.length, 2, 'a lease rejection must not discard valid generated audio');
  const afterStale = (await rows(indexedDB))[0] as ReadyEnhancementPair;
  assert.equal(afterStale.key, retained.key);
  const reuse = await runWithAudioEnhancementPairCache(originals(2), model, newer, async () => { assert.fail('The newer pair must remain a hit'); });
  assert.deepEqual(reuse.tracks.map((track) => track.bytes), second.tracks.map((track) => track.bytes));
  older.close(); newer.close();
});

for (const corruption of ['wav-bytes', 'wav-header', 'clock', 'duplicate-lane', 'partial-pair', 'unknown-field'] as const) {
  test(`a persisted ${corruption} corruption is discarded before a fresh complete enhancement`, async () => {
    const indexedDB = new IDBFactory(), cache = createAudioEnhancementPairCache({ indexedDB });
    await runWithAudioEnhancementPairCache(originals(), model, cache, generateFixtureEnhancement);
    const corrupt = (await rows(indexedDB))[0] as ReadyEnhancementPair;
    if (corruption === 'wav-bytes') new Uint8Array(corrupt.lanes[0].wav)[44] ^= 1;
    if (corruption === 'wav-header') {
      new DataView(corrupt.lanes[0].wav).setUint32(24, 8000, true);
      corrupt.lanes[0].wavSha256 = await enhancementSha256(corrupt.lanes[0].wav);
    }
    if (corruption === 'clock') corrupt.lanes[0].frameCount++;
    if (corruption === 'duplicate-lane') corrupt.lanes[1] = corrupt.lanes[0];
    const value = corruption === 'partial-pair' ? { ...corrupt, lanes: [corrupt.lanes[0]] } :
      corruption === 'unknown-field' ? { ...corrupt, transcript: 'must not survive in this audio-only cache' } : corrupt;
    await replaceRaw(indexedDB, value);
    let calls = 0;
    const recovered = await runWithAudioEnhancementPairCache(originals(), model, cache, async (sources) => {
      calls++;
      assertPendingOnly(await rows(indexedDB));
      return generateFixtureEnhancement(sources);
    });
    assert.equal(calls, 1);
    assert.equal(recovered.cacheStatus, 'stored');
    const ready = (await rows(indexedDB))[0] as ReadyEnhancementPair;
    assert.equal(ready.lanes.length, 2);
    assert.equal('transcript' in ready, false);
    for (const lane of ready.lanes) assert.equal(await enhancementSha256(lane.wav), lane.wavSha256);
    cache.close();
  });
}

test('storage-open errors leave valid generated audio usable with an explicit cache outcome', async () => {
  const indexedDB = new IDBFactory();
  indexedDB.open = () => { throw new DOMException('Storage disabled by policy', 'SecurityError'); };
  const cache = createAudioEnhancementPairCache({ indexedDB });
  const result = await runWithAudioEnhancementPairCache(originals(), model, cache, generateFixtureEnhancement);
  assert.equal(result.cacheStatus, 'unavailable');
  assert.deepEqual(result.tracks.map(track => decodeEnhancementWav(track.bytes.buffer).samples[0]),
    [0.020782470703125, 0.01873779296875]);
  cache.close();
});

test('a quota exception in the ready write aborts atomically and cannot erase valid generated results', async () => {
  const indexedDB = new IDBFactory(), cache = createAudioEnhancementPairCache({ indexedDB });
  const put = IDBObjectStore.prototype.put;
  try {
    const result = await runWithAudioEnhancementPairCache(originals(), model, cache, async (sources) => {
      const enhanced = await generateFixtureEnhancement(sources);
      IDBObjectStore.prototype.put = function(value: unknown, key?: IDBValidKey) {
        if (value !== null && typeof value === 'object' && 'state' in value && value.state === 'ready') {
          throw new DOMException('Enhanced WAV bytes exceed the local quota', 'QuotaExceededError');
        }
        return put.call(this, value, key);
      };
      return enhanced;
    });
    assert.equal(result.cacheStatus, 'unavailable');
    assert.match(result.cacheMessage!, /local quota/i);
    assert.equal(result.tracks.length, 2);
    const retained = await rows(indexedDB);
    assertPendingOnly(retained);
  } finally { IDBObjectStore.prototype.put = put; cache.close(); }
});

test('an interrupted one-lane result can never commit a ready pair', async () => {
  const indexedDB = new IDBFactory(), cache = createAudioEnhancementPairCache({ indexedDB });
  await assert.rejects(runWithAudioEnhancementPairCache(originals(), model, cache,
    (sources) => generateFixtureEnhancement([sources[0]])), /did not return the complete Original recording selection/i);
  assertPendingOnly(await rows(indexedDB));
  cache.close();
});

test('version changes close cached connections rather than blocking another extension context', async () => {
  const indexedDB = new IDBFactory(), cache = createAudioEnhancementPairCache({ indexedDB });
  await runWithAudioEnhancementPairCache(originals(), model, cache, generateFixtureEnhancement);
  const upgrade = await openRaw(indexedDB, 2);
  upgrade.close();
  const result = await runWithAudioEnhancementPairCache(originals(), model, cache, generateFixtureEnhancement);
  assert.equal(result.cacheStatus, 'unavailable', 'the old schema cannot be silently reused after a future version upgrade');
  assert.match(result.cacheMessage!, /could not be opened/i);
  assert.equal(result.tracks.length, 2);
  cache.close();
});

test('cache-hit progress delivery errors remain real request failures and do not trigger regeneration', async () => {
  const indexedDB = new IDBFactory(), cache = createAudioEnhancementPairCache({ indexedDB });
  await runWithAudioEnhancementPairCache(originals(), model, cache, generateFixtureEnhancement);
  await assert.rejects(runWithAudioEnhancementPairCache(originals(), model, cache,
    async () => { assert.fail('Progress delivery failure is not a cache miss'); },
    (progress) => { if (progress.phase === 'cache-hit') throw new Error('Native progress receiver disconnected'); }), /Native progress receiver disconnected/);
  cache.close();
});

test('storage failures cannot turn a real model failure into an audio success', async () => {
  const indexedDB = new IDBFactory();
  indexedDB.open = () => { throw new DOMException('Storage disabled by policy', 'SecurityError'); };
  const cache = createAudioEnhancementPairCache({ indexedDB });
  await assert.rejects(runWithAudioEnhancementPairCache(originals(), model, cache, async () => {
    throw new Error('ZipEnhancer hardware placement proof failed');
  }), /ZipEnhancer hardware placement proof failed/);
  cache.close();
});

test('one-lane callers remain usable but cannot retain an old pair or a partial ready entry', async () => {
  const indexedDB = new IDBFactory(), cache = createAudioEnhancementPairCache({ indexedDB });
  await runWithAudioEnhancementPairCache(originals(), model, cache, generateFixtureEnhancement);
  const result = await runWithAudioEnhancementPairCache([originals()[0]], model, cache, generateFixtureEnhancement);
  assert.equal(result.tracks.length, 1);
  assert.equal(result.cacheStatus, 'unavailable');
  assert.match(result.cacheMessage!, /both distinct Original recording lanes/i);
  assert.deepEqual(await rows(indexedDB), []);
  cache.close();
});

test('native client streams a persistent hit after host restart using the current task and lane labels', async () => {
  const indexedDB = new IDBFactory();
  let cache = createAudioEnhancementPairCache({ indexedDB });
  let generations = 0;
  const progress: AudioEnhancementProgress[] = [];
  const createHost = () => createLocalModelHost(async () => { throw new Error('Enhancement cannot initialize ASR'); }, {
    authorizeEnhancement: async () => new AbortController().signal,
    enhanceAudio: (tracks, onProgress) => runWithAudioEnhancementPairCache(tracks, model, cache, async (sources) => {
      generations++;
      return generateFixtureEnhancement(sources);
    }, onProgress).then(batch => ({ ...batch, provider: 'browser-local' as const })),
    onProgress: (message) => { progress.push(message.progress); }
  });
  let host = createHost();
  const client = createLocalModelClient((request) => host.handleRequest({ ...request, target: 'offscreen' }));
  const firstChunks: AudioEnhancementChunk[] = [];
  const first = await client.enhanceAudio('prior-native-review', originals(), {
    isCurrent: () => true, onAudioChunk: (chunk) => { firstChunks.push(chunk); }
  });
  assert.equal(first.cacheStatus, 'stored');
  cache.close();
  cache = createAudioEnhancementPairCache({ indexedDB });
  host = createHost();
  progress.length = 0;
  const refreshed = originals().reverse().map((track, index) => ({ ...track, trackLabel: `Refreshed native lane ${index}` }));
  const hitChunks: AudioEnhancementChunk[] = [];
  const hit = await client.enhanceAudio('current-native-review', refreshed, {
    isCurrent: () => true, onAudioChunk: (chunk) => { hitChunks.push(chunk); }
  });
  assert.equal(hit.taskId, 'current-native-review');
  assert.equal(hit.cacheStatus, 'hit');
  assert.equal(generations, 1, 'only the pre-restart request runs the enhancement generator');
  assert.deepEqual(progress.map((value) => value.phase), ['queued', 'cache-lookup', 'cache-lookup', 'cache-hit', 'cache-hit']);
  for (let index = 0; index < hit.tracks.length; index++) {
    const track = hit.tracks[index], before = first.tracks.find((candidate) => candidate.trackId === track.trackId)!;
    const chunks = hitChunks.filter((chunk) => chunk.trackId === track.trackId);
    const bytes = new Uint8Array(track.totalBytes);
    let offset = 0;
    for (const chunk of chunks) { const decoded = decodeAudioChunk(chunk.dataBase64); bytes.set(decoded, offset); offset += decoded.length; }
    assert.equal(offset, bytes.length);
    assert.equal(await enhancementSha256(bytes.buffer), before.wavSha256);
    assert.equal(track.trackLabel, `Refreshed native lane ${index}`);
    assert.equal(track.sourceSha256, before.sourceSha256);
    assert.equal('audioTransferId' in track, false);
  }
  assert.equal((await rows(indexedDB)).length, 1);
  cache.close();
});

test('native client still delivers both playable lanes when persistent cache storage is denied', async () => {
  const indexedDB = new IDBFactory();
  indexedDB.open = () => { throw new DOMException('Storage disabled by policy', 'SecurityError'); };
  const cache = createAudioEnhancementPairCache({ indexedDB });
  const host = createLocalModelHost(async () => { throw new Error('Enhancement cannot initialize ASR'); }, {
    authorizeEnhancement: async () => new AbortController().signal,
    enhanceAudio: (tracks, onProgress) => runWithAudioEnhancementPairCache(tracks, model, cache, generateFixtureEnhancement, onProgress).then(batch => ({ ...batch, provider: 'browser-local' as const }))
  });
  const client = createLocalModelClient((request) => host.handleRequest({ ...request, target: 'offscreen' }));
  const chunks: AudioEnhancementChunk[] = [];
  const result = await client.enhanceAudio('current-native-review', originals(), {
    isCurrent: () => true, onAudioChunk: (chunk) => { chunks.push(chunk); }
  });
  assert.equal(result.cacheStatus, 'unavailable');
  assert.equal(result.tracks.length, 2);
  for (const track of result.tracks) {
    const bytes = decodeAudioChunk(chunks.find((chunk) => chunk.trackId === track.trackId)!.dataBase64);
    assert.equal(await enhancementSha256(bytes.buffer), track.wavSha256);
    assert.equal(decodeEnhancementWav(bytes.buffer).samples.length, track.frameCount);
  }
  cache.close();
});

for (const invalid of ['unknown-status', 'missing-notice', 'unexpected-notice', 'result-secret', 'track-secret', 'envelope-secret'] as const) {
  test(`native client rejects ${invalid} cache metadata before downloading any enhanced bytes`, async (t) => {
    const indexedDB = new IDBFactory(), cache = createAudioEnhancementPairCache({ indexedDB });
    const host = createLocalModelHost(async () => { throw new Error('Enhancement cannot initialize ASR'); }, {
      authorizeEnhancement: async () => new AbortController().signal,
      enhanceAudio: (tracks, onProgress) => runWithAudioEnhancementPairCache(tracks, model, cache, generateFixtureEnhancement, onProgress).then(batch => ({ ...batch, provider: 'browser-local' as const }))
    });
    const outputIds: string[] = [];
    t.after(async () => {
      await host.handleRequest({ type: LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE, version: LOCAL_MODEL_OFFSCREEN_VERSION,
        target: 'offscreen', operation: 'release', requestId: `cleanup:${invalid}`, transferIds: outputIds });
      cache.close();
    });
    const client = createLocalModelClient(async (request) => {
      const response = await host.handleRequest({ ...request, target: 'offscreen' });
      if (!response.ok || response.operation !== 'enhanceAudio') return response;
      const result = response.result;
      outputIds.push(...result.tracks.map((track) => track.audioTransferId));
      if (invalid === 'unknown-status') return { ...response, result: { ...result, cacheStatus: 'saved' } };
      if (invalid === 'missing-notice') return { ...response, result: { ...result, cacheStatus: 'unavailable' } };
      if (invalid === 'unexpected-notice') return { ...response, result: { ...result, cacheMessage: 'not an unavailable cache' } };
      if (invalid === 'result-secret') return { ...response, result: { ...result, transcript: 'must not cross this boundary' } };
      if (invalid === 'track-secret') return { ...response, result: { ...result, tracks: result.tracks.map((track) => ({ ...track, apiKey: 'must not cross this boundary' })) } };
      return { ...response, apiKey: 'must not cross this boundary' };
    });
    await assert.rejects(client.enhanceAudio('current-native-review', originals(), {
      isCurrent: () => true, onAudioChunk: () => { assert.fail('Invalid cache result metadata cannot stream enhanced audio'); }
    }), (error) => error instanceof LocalModelBridgeError && error.code === 'invalid-response');
  });
}


test('leased enhancement bypasses persistence without evicting the volunteer own pair', async () => {
  const indexedDB = new IDBFactory(), cache = createAudioEnhancementPairCache({ indexedDB });
  const own = await runWithAudioEnhancementPairCache(originals(1), model, cache, generateFixtureEnhancement);
  const retained = await rows(indexedDB);
  const leased = await runWithAudioEnhancementPairCache(originals(9), model, null, generateFixtureEnhancement);
  assert.notEqual(leased.tracks[0].metadata.sourceSha256, own.tracks[0].metadata.sourceSha256);
  assert.deepEqual(await rows(indexedDB), retained);
  const hit = await runWithAudioEnhancementPairCache(originals(1), model, cache, async () => {
    assert.fail('Leased audio must not replace the worker owner cache');
  });
  assert.equal(hit.cacheStatus, 'hit');
  cache.close();
});
