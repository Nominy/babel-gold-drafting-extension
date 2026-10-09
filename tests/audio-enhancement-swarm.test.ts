import test from 'node:test';
import assert from 'node:assert/strict';
import { setImmediate } from 'node:timers/promises';
import { enhanceAudioThroughSwarm } from '../src/core/audio-enhancement-swarm';
import { enhancementSha256, type PreparedEnhancementSource } from '../src/core/audio-enhancement-cache';
import { encodeEnhancementWav } from '../src/core/audio-enhancement-dsp';
import {
  ENHANCEMENT_MODEL_ID, ENHANCEMENT_SOURCE_GRAPH_SHA256, parseEnhancementPayload,
  parseEnhancementWorkerProgress, readBoundedSwarmBlob, verifyEnhancementWav,
  type EnhancementSwarmPayload
} from '../src/core/audio-enhancement-swarm-protocol';

const model = { id: ENHANCEMENT_MODEL_ID, sha256: 'a'.repeat(64), sourceGraphSha256: ENHANCEMENT_SOURCE_GRAPH_SHA256 };
const baseUrl = 'http://127.0.0.1:8976/private';

async function fixture() {
  const bytes = encodeEnhancementWav(new Float32Array([0, 0.1, -0.2, 0.1]), 16000, 4).buffer;
  const sha256 = await enhancementSha256(bytes);
  const sources: PreparedEnhancementSource[] = ['lane A', 'lane B'].map(trackId => ({
    trackId, bytes, sourceSha256: sha256, sampleRate: 16000, frameCount: 4,
    track: { trackId, speakerKey: trackId, trackLabel: trackId, source: 'original', mimeType: 'audio/wav', blob: new Blob([bytes]) }
  }));
  const payload = parseEnhancementPayload({ taskId: 'task:private', model,
    tracks: sources.map((source, index) => ({ trackId: source.trackId, speakerKey: source.trackId, trackLabel: source.trackId,
      sourceSha256: sha256, sampleRate: 16000, frameCount: 4, fieldName: `audio:${index + 1}` })) });
  const result = { ok: true, provider: 'swarm', taskId: payload.taskId, model: model.id, modelSha256: model.sha256,
    tracks: sources.map(source => ({ trackId: source.trackId, speakerKey: source.trackId, trackLabel: source.trackId,
      sourceSha256: sha256, sampleRate: 16000, frameCount: 4, mimeType: 'audio/wav', wavSha256: sha256, totalBytes: bytes.byteLength, chunkCount: 1 })) };
  const response = (changed: unknown = result, output = bytes) => {
    const form = new FormData();
    form.append('result', JSON.stringify(changed));
    form.append('audio:1', new Blob([output], { type: 'audio/wav' }), 'a.wav');
    form.append('audio:2', new Blob([bytes], { type: 'audio/wav' }), 'b.wav');
    return new Response(form);
  };
  return { bytes, sha256, sources, payload, result, response };
}

test('swarm client sends exact Original WAVs with owner capability and validates binary model-bound results', async t => {
  const data = await fixture();
  const phases: string[] = [];
  let requests = 0;
  t.mock.method(globalThis, 'fetch', async (url: string, init: RequestInit) => {
    requests++;
    assert.equal(url, `${baseUrl}/v1/enhance`);
    assert.equal(init.redirect, 'error');
    assert.equal(init.method, 'POST');
    const headers = new Headers(init.headers);
    assert.equal(headers.get('X-Babel-Local-Engine'), '1');
    assert.match(headers.get('X-Babel-Request-Id')!, /^[a-f0-9-]{36}$/);
    assert.match(headers.get('Authorization')!, /^Bearer [A-Za-z0-9_-]{43}$/);
    const form = init.body as FormData;
    assert.deepEqual(JSON.parse(String(form.get('payload'))), data.payload);
    assert.deepEqual(Array.from(form.keys()), ['payload', 'audio:1', 'audio:2']);
    assert.deepEqual(await (form.get('audio:1') as Blob).arrayBuffer(), data.bytes);
    return data.response();
  });
  const result = await enhanceAudioThroughSwarm(data.sources, model, { taskId: data.payload.taskId, baseUrl,
    onProgress: progress => { phases.push(progress.phase); assert.equal(progress.backend, 'swarm'); } });
  assert.equal(requests, 1);
  assert.deepEqual(result.map(track => track.metadata), data.result.tracks);
  assert.deepEqual(result[0].bytes.buffer, data.bytes);
});

test('default swarm coordinator is loaded from saved background settings, not public or page operation data', async t => {
  const data = await fixture();
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'chrome');
  Object.defineProperty(globalThis, 'chrome', { configurable: true, value: { runtime: { sendMessage: async (message: unknown) => {
    assert.deepEqual(message, { type: 'babel-l0-volunteer', target: 'background', action: 'settings' });
    return { l0CustomBaseUrl: `${baseUrl}/?not-sent=1#not-sent` };
  } } } });
  t.after(() => { if (previous) Object.defineProperty(globalThis, 'chrome', previous); else Reflect.deleteProperty(globalThis, 'chrome'); });
  t.mock.method(globalThis, 'fetch', async (url: string) => { assert.equal(url, `${baseUrl}/v1/enhance`); return data.response(); });
  await enhanceAudioThroughSwarm(data.sources, model, { taskId: data.payload.taskId });
});

test('no matching worker is actionable and no enhanced output is accepted', async t => {
  const data = await fixture();
  t.mock.method(globalThis, 'fetch', async () => Response.json({ error: 'No matching WebGPU volunteer is registered' }, { status: 503 }));
  await assert.rejects(enhanceAudioThroughSwarm(data.sources, model, { taskId: data.payload.taskId, baseUrl }), /Originals remain selected.*compatible GPU volunteer/);
});

test('client rejects wrong task/model/source identity, clocks, hashes, and malformed enhanced WAVs', async t => {
  const data = await fixture();
  const changed = [
    { ...data.result, taskId: 'another-task' },
    { ...data.result, modelSha256: 'b'.repeat(64) },
    { ...data.result, tracks: [{ ...data.result.tracks[0], sourceSha256: 'c'.repeat(64) }, data.result.tracks[1]] },
    { ...data.result, tracks: [{ ...data.result.tracks[0], frameCount: 5 }, data.result.tracks[1]] },
    { ...data.result, tracks: [{ ...data.result.tracks[0], wavSha256: 'c'.repeat(64) }, data.result.tracks[1]] },
    { ...data.result, tracks: [{ ...data.result.tracks[0], chunkCount: 2 }, data.result.tracks[1]] }
  ];
  for (const result of changed) {
    const mock = t.mock.method(globalThis, 'fetch', async () => data.response(result));
    await assert.rejects(enhanceAudioThroughSwarm(data.sources, model, { taskId: data.payload.taskId, baseUrl }), /Swarm/);
    mock.mock.restore();
  }
  const corrupted = data.bytes.slice(0);
  new DataView(corrupted).setUint32(28, 1, true);
  t.mock.method(globalThis, 'fetch', async () => data.response(data.result, corrupted));
  await assert.rejects(enhanceAudioThroughSwarm(data.sources, model, { taskId: data.payload.taskId, baseUrl }), /byte clock/);
});

test('client refuses duplicate response fields and oversized result streams', async t => {
  const data = await fixture();
  const duplicate = new FormData();
  duplicate.append('result', JSON.stringify(data.result));
  duplicate.append('audio:1', new Blob([data.bytes]));
  duplicate.append('audio:1', new Blob([data.bytes]));
  const mock = t.mock.method(globalThis, 'fetch', async () => new Response(duplicate));
  await assert.rejects(enhanceAudioThroughSwarm(data.sources, model, { taskId: data.payload.taskId, baseUrl }), /missing, duplicate/);
  mock.mock.restore();
  t.mock.method(globalThis, 'fetch', async () => new Response('too large', { headers: { 'Content-Type': 'multipart/form-data; boundary=x', 'Content-Length': '999999999' } }));
  await assert.rejects(enhanceAudioThroughSwarm(data.sources, model, { taskId: data.payload.taskId, baseUrl }), /bounded transfer/);
});

test('owner polling uses the same capability, forwards only real worker progress, and cancels with POST', async t => {
  const data = await fixture(), controller = new AbortController();
  const observed = Promise.withResolvers<void>();
  let postHeaders: Headers | undefined, postSignal: AbortSignal | undefined, polls = 0;
  t.mock.method(globalThis, 'fetch', async (url: string, init: RequestInit) => {
    if (url.endsWith('/v1/enhance')) {
      postHeaders = new Headers(init.headers); postSignal = init.signal as AbortSignal;
      return new Promise<Response>((_resolve, reject) => postSignal!.addEventListener('abort', () => reject(postSignal!.reason), { once: true }));
    }
    polls++;
    assert.equal(url, `${baseUrl}/v1/queue/${postHeaders!.get('X-Babel-Request-Id')}`);
    assert.equal(new Headers(init.headers).get('Authorization'), postHeaders!.get('Authorization'));
    return Response.json({ requestId: postHeaders!.get('X-Babel-Request-Id'), status: 'running', progress: { phase: 'enhancing', trackId: 'lane A', trackIndex: 0, trackCount: 2, completedChunks: 1, totalChunks: 2 } });
  });
  const running = enhanceAudioThroughSwarm(data.sources, model, { taskId: data.payload.taskId, baseUrl, signal: controller.signal,
    onProgress: progress => { if (progress.phase === 'enhancing') { assert.equal(progress.completedChunks, 1); observed.resolve(); } } });
  const rejection = assert.rejects(running, /cancelled by owner/);
  await observed.promise;
  controller.abort(new Error('cancelled by owner'));
  await rejection;
  assert.equal(postSignal!.aborted, true);
  assert.equal(polls, 1);
  await setImmediate();
  assert.equal(polls, 1);
});

test('wrong owner status authorization terminates the pending upload instead of falling back', async t => {
  const data = await fixture();
  let postSignal: AbortSignal | undefined;
  t.mock.method(globalThis, 'fetch', async (url: string, init: RequestInit) => {
    if (url.endsWith('/v1/enhance')) {
      postSignal = init.signal as AbortSignal;
      return new Promise<Response>((_resolve, reject) => postSignal!.addEventListener('abort', () => reject(postSignal!.reason), { once: true }));
    }
    return new Response(null, { status: 403 });
  });
  await assert.rejects(enhanceAudioThroughSwarm(data.sources, model, { taskId: data.payload.taskId, baseUrl }), /HTTP 403/);
  assert.equal(postSignal!.aborted, true);
});

test('navigation during a stalled result download cancels the body instead of retaining the old request', async t => {
  const data = await fixture();
  let cancelled = false, transferChecks = 0;
  t.mock.method(globalThis, 'fetch', async (url: string) => {
    if (url.includes('/v1/queue/')) return new Response(null, { status: 404 });
    return new Response(new ReadableStream<Uint8Array>({ cancel() { cancelled = true; } }),
      { headers: { 'Content-Type': 'multipart/form-data; boundary=unfinished' } });
  });
  await assert.rejects(enhanceAudioThroughSwarm(data.sources, model, { taskId: data.payload.taskId, baseUrl,
    onProgress: progress => {
      if (progress.phase === 'transferring' && ++transferChecks === 2) throw new Error('owner navigated');
    },
  }), /owner navigated/);
  assert.equal(cancelled, true);
});

test('stream limits apply without Content-Length and cancellation releases the reader', async () => {
  let cancelled = false;
  const response = new Response(new ReadableStream<Uint8Array>({
    start(controller) { controller.enqueue(new Uint8Array(9)); },
    cancel() { cancelled = true; }
  }));
  await assert.rejects(readBoundedSwarmBlob(response, 8), /bounded transfer/);
  assert.equal(cancelled, true);
  const controller = new AbortController();
  const hanging = new Response(new ReadableStream<Uint8Array>());
  const pending = readBoundedSwarmBlob(hanging, 8, controller.signal);
  controller.abort(new Error('cancelled read'));
  await assert.rejects(pending, /cancelled read/);
});

test('source validation accepts unchanged stereo float32 but rejects nonfinite PCM and inconsistent RIFF clocks', async () => {
  const data = await fixture();
  const bytes = new ArrayBuffer(44 + 4 * 2 * 4), view = new DataView(bytes);
  new Uint8Array(bytes, 0, 44).set(new Uint8Array(data.bytes, 0, 44));
  view.setUint32(4, bytes.byteLength - 8, true);
  view.setUint16(20, 3, true); view.setUint16(22, 2, true);
  view.setUint32(28, 16000 * 8, true); view.setUint16(32, 8, true); view.setUint16(34, 32, true);
  view.setUint32(40, 32, true);
  const sha256 = await enhancementSha256(bytes);
  await verifyEnhancementWav(bytes, data.payload.tracks[0], sha256);
  await assert.rejects(verifyEnhancementWav(bytes, data.payload.tracks[0], sha256, true), /PCM16 mono/);
  view.setFloat32(44, Number.NaN, true);
  await assert.rejects(verifyEnhancementWav(bytes, data.payload.tracks[0], sha256), /nonfinite/);
});

test('progress and payload validators reject regression, field reuse, transcripts, and malformed clocks', async () => {
  const { payload } = await fixture();
  const progress = { phase: 'enhancing' as const, trackId: 'lane A', trackIndex: 0, trackCount: 2, completedChunks: 2, totalChunks: 3 };
  assert.throws(() => parseEnhancementWorkerProgress({ ...progress, completedChunks: 1 }, payload, progress), /regressed/);
  assert.throws(() => parseEnhancementWorkerProgress({ ...progress, totalChunks: 4 }, payload, progress), /regressed/);
  assert.throws(() => parseEnhancementWorkerProgress({ ...progress, trackId: 'lane B' }, payload), /Invalid/);
  assert.throws(() => parseEnhancementWorkerProgress({ ...progress, phase: 'encoding' }, payload), /Invalid/);
  assert.throws(() => parseEnhancementPayload({ ...payload, transcript: 'never transmitted' }), /payload/);
  assert.throws(() => parseEnhancementPayload({ ...payload, tracks: [payload.tracks[0], payload.tracks[0]] }), /identities/);
  assert.throws(() => parseEnhancementPayload({ ...payload, tracks: [{ ...payload.tracks[0], frameCount: -1 }, payload.tracks[1]] }), /clocks/);
  const badModel: EnhancementSwarmPayload = { ...payload, model: { ...model, sha256: 'not-a-hash' } };
  assert.throws(() => parseEnhancementPayload(badModel), /model descriptor/);
});

test('invalid isolated coordinator override and already cancelled requests never upload to the public coordinator', async t => {
  const data = await fixture();
  const fetch = t.mock.method(globalThis, 'fetch', async () => { assert.fail('No audio may leave this client.'); });
  await assert.rejects(enhanceAudioThroughSwarm(data.sources, model, { taskId: data.payload.taskId, baseUrl: 'file:///tmp/coordinator' }), /HTTP or HTTPS/);
  const controller = new AbortController();
  controller.abort(new Error('already cancelled'));
  await assert.rejects(enhanceAudioThroughSwarm(data.sources, model, { taskId: data.payload.taskId, baseUrl, signal: controller.signal }), /already cancelled/);
  assert.equal(fetch.mock.callCount(), 0);
});
