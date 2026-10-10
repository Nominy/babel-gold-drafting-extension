import test from 'node:test';
import assert from 'node:assert/strict';
import { setImmediate } from 'node:timers/promises';
import { createVolunteer, defaultVolunteerDependencies, parseLease, type VolunteerDependencies } from '../src/offscreen/volunteer';
import { createVolunteerLifecycle } from '../src/background/local-model-offscreen';
import { DEFAULT_SETTINGS, PUBLIC_L0_BASE_URL } from '../src/core/settings';
import type { L0DraftResponse, L0TimingResponse } from '../src/core/types';
import { encodeEnhancementWav } from '../src/core/audio-enhancement-dsp';
import { enhancementSha256 } from '../src/core/audio-enhancement-cache';
import { ENHANCEMENT_MAX_TRACK_BYTES, ENHANCEMENT_MODEL_ID, ENHANCEMENT_SOURCE_GRAPH_SHA256 } from '../src/core/audio-enhancement-swarm-protocol';
import type { EnhancedAudioBatch } from '../src/core/audio-enhancement-runtime';
import { hydratePunctuatedTiming, __localModelRuntimeTesting } from '../src/core/local-model-runtime';

const wav = new Blob([new Uint8Array([82, 73, 70, 70, 0, 0, 0, 0, 87, 65, 86, 69, 0])], { type: 'audio/wav' });
const lease = (operation: 'draft' | 'transcribe', options?: Record<string, unknown>) => ({
  jobId: 'job-1', leaseToken: 'lease-secret', operation,
  payload: operation === 'draft'
    ? { taskId: 'remote-task', timing: timingResult, ...(options ? { options } : {}) }
    : {
      taskId: 'remote-task', tracks: [
        { lane: 'A', fieldName: 'audio:1' }, { lane: 'B', fieldName: 'audio:2' }
      ]
    },
  audio: operation === 'draft' ? [] : [
    { fieldName: 'audio:1', url: '/v1/jobs/job-1/audio/audio%3A1' },
    { fieldName: 'audio:2', url: '/v1/jobs/job-1/audio/audio%3A2' }
  ]
});
const draftResult: L0DraftResponse = {
  rows: [{ id: 'row', lane: 'A', startSeconds: 0, endSeconds: 1, text: 'Hello' }],
  summary: {}, models: {}
};
const timingResult: L0TimingResponse = {
  taskId: 'remote-task', summary: { taskId: 'remote-task' }, models: {},
  tracks: [
    { lane: 'A', tokens: [{ id: 'remote-task:A:0', text: 'Hello', startSeconds: 0, endSeconds: 1 }], segments: [{ id: 'segment-1', startSeconds: 0, endSeconds: 1, startSample: 0, endSample: 16000, sampleRate: 16000 }], pcmSha256: 'a'.repeat(64), sampleRate: 16000 },
    { lane: 'B', tokens: [], segments: [], pcmSha256: 'b'.repeat(64), sampleRate: 16000 }
  ]
};

async function until(check: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (check()) return;
    await setImmediate();
  }
  assert.fail('Volunteer did not reach the expected state.');
}

function fixture(jobs: unknown[], ready = true) {
  const requests: Array<{ url: string; init: RequestInit }> = [];
  const completed: Array<Record<string, unknown>> = [];
  const calls: string[] = [];
  let idle = false;
  const dependencies: VolunteerDependencies = {
    settings: async () => ({ ...DEFAULT_SETTINGS, mode: 'advanced', localModelsEnabled: true, volunteerInferenceEnabled: true }),
    ready: async () => ({ transcribe: ready, draft: ready }),
    runExclusive: async (action) => { calls.push('exclusive'); return action(); },
    draft: async (timing, preserveRows) => {
      assert.deepEqual(timing, timingResult);
      calls.push('draft');
      return preserveRows ? {
        ...draftResult,
        rows: preserveRows.map((row) => ({
          id: row.rowId, lane: row.speakerKey,
          startSeconds: row.startSeconds!, endSeconds: row.endSeconds!,
          text: 'Preserved text'
        }))
      } : draftResult;
    },
    transcribe: async (_settings, _job, audio, _callbacks, taskId) => {
      assert.equal(taskId, 'remote-task');
      assert.deepEqual(audio.map((track) => track.speakerKey), ['A', 'B']);
      calls.push('transcribe');
      return timingResult;
    },
    enhance: async () => { assert.fail('ASR-only fixture must not enhance.'); },
    authorizeEnhancement: async () => new AbortController().signal,
    wait: async (_ms, signal) => {
      idle = true;
      if (signal.aborted) return;
      const { promise, resolve } = Promise.withResolvers<void>();
      signal.addEventListener('abort', () => resolve(), { once: true });
      return promise;
    },
    fetch: async (url, init) => {
      requests.push({ url, init });
      const path = url.slice(PUBLIC_L0_BASE_URL.length);
      if (path === '/v1/workers/register') return Response.json({ workerId: 'volunteer-1', token: 'worker-secret' });
      if (path === '/v1/workers/lease') {
        const next = jobs.shift();
        return next ? Response.json(next) : new Response(null, { status: 204 });
      }
      if (path.includes('/audio/')) {
        assert.equal((init.headers as Record<string, string>).Authorization, 'Bearer lease-secret');
        return new Response(wav, { headers: { 'Content-Type': 'audio/wav' } });
      }
      if (path === '/v1/jobs/job-1/complete') {
        completed.push(JSON.parse(init.body as string) as Record<string, unknown>);
        return Response.json({ ok: true });
      }
      assert.fail(`Unexpected worker request: ${url}`);
    }
  };
  return { dependencies, requests, completed, calls, isIdle: () => idle };
}

test('punctuation lease uses cached timing without downloading audio and completes with credentials', async () => {
  const harness = fixture([lease('draft')]);
  const worker = createVolunteer(harness.dependencies);
  assert.equal(worker.start().state, 'connecting');
  assert.equal(worker.start().state, 'connecting');
  await until(() => harness.isIdle());
  assert.equal(worker.getStatus().state, 'connected');
  assert.deepEqual(harness.calls, ['exclusive', 'draft']);
  assert.deepEqual(harness.completed, [{
    workerId: 'volunteer-1', token: 'worker-secret', leaseToken: 'lease-secret', result: draftResult
  }]);
  assert.deepEqual(harness.requests.map((request) => request.url.slice(PUBLIC_L0_BASE_URL.length)), [
    '/v1/workers/register', '/v1/workers/lease', '/v1/jobs/job-1/complete'
  ]);
  worker.stop();
  assert.equal(worker.getStatus().state, 'disabled');
});

for (const declaredLength of [true, false]) {
  test(`fresh transcription hands a large cached-word draft lease through to completion (${declaredLength ? 'declared' : 'streamed'} size)`, async () => {
    const words = Array.from({ length: 1000 }, (_, index) => ({
      id: `remote-task:A:${index}`, text: 'слово',
      startSeconds: index / 4, endSeconds: (index + 1) / 4
    }));
    const timing: L0TimingResponse = {
      ...timingResult, models: { release: 'c-denoise-v3-2026-10-03-r2' },
      tracks: [{
        ...timingResult.tracks[0], tokens: words,
        punctuationLabels: words.map((_, index) => index === words.length - 1 ? 2 : 0),
        segments: [{ id: 'long-segment', startSeconds: 0, endSeconds: 250,
          startSample: 0, endSample: 4_000_000, sampleRate: 16000 }]
      }, { ...timingResult.tracks[1], punctuationLabels: [] }]
    };
    const draftLease = { ...lease('draft'), payload: { taskId: timing.taskId, timing } };
    const bytes = new TextEncoder().encode(JSON.stringify(draftLease));
    assert.ok(bytes.byteLength > 64 * 1024 && bytes.byteLength < 16 * 1024 * 1024);
    const harness = fixture([lease('transcribe')]);
    const originalFetch = harness.dependencies.fetch, originalWait = harness.dependencies.wait;
    harness.dependencies.transcribe = async () => timing;
    harness.dependencies.draft = async cached => {
      const lane = hydratePunctuatedTiming(cached).get('A')!;
      return { ...draftResult, rows: [{
        id: 'long-segment', lane: 'A', startSeconds: 0, endSeconds: 250,
        text: __localModelRuntimeTesting.renderCachedRange(lane, 0, lane.tokens.length)
      }] };
    };
    let leases = 0;
    harness.dependencies.fetch = async (url, init) => {
      if (url.endsWith('/v1/workers/lease') && ++leases === 2) {
        let offset = 0;
        return new Response(new ReadableStream({
          pull(controller) {
            if (offset === bytes.length) { controller.close(); return; }
            const end = Math.min(offset + 4096, bytes.length);
            controller.enqueue(bytes.slice(offset, end)); offset = end;
          }
        }), { headers: declaredLength ? { 'Content-Length': String(bytes.byteLength) } : {} });
      }
      return originalFetch(url, init);
    };
    harness.dependencies.wait = async (delay, signal) => {
      if (delay === 1000 && harness.completed.length === 1) return;
      return originalWait(delay, signal);
    };
    const worker = createVolunteer(harness.dependencies);
    try {
      worker.start();
      await until(() => harness.isIdle());
      assert.equal(harness.completed.length, 2, worker.getStatus().detail);
      assert.equal(worker.getStatus().state, 'connected');
      const draft = harness.completed[1].result as L0DraftResponse;
      assert.equal(draft.rows[0].text, `Слово ${Array(999).fill('слово').join(' ')}.`);
      assert.equal(draft.rows[0].endSeconds, 250);
    } finally { worker.stop(); }
  });
}

test('preserveRows punctuates cached words while unsupported options report an error to the coordinator', async () => {
  const row = { rowId: 'existing', speakerKey: 'A', startSeconds: 0, endSeconds: 1, text: '', index: 0 };
  const first = fixture([lease('draft', { preserveRows: [row] })]);
  const volunteer = createVolunteer(first.dependencies);
  volunteer.start();
  await until(() => first.isIdle());
  assert.deepEqual(first.calls, ['exclusive', 'draft']);
  assert.deepEqual((first.completed[0].result as L0DraftResponse).rows, [
    { id: 'existing', lane: 'A', startSeconds: 0, endSeconds: 1, text: 'Preserved text' }
  ]);
  volunteer.stop();

  const second = fixture([lease('draft', { preprocessing: 'afftdn' })]);
  const rejected = createVolunteer(second.dependencies);
  rejected.start();
  await until(() => second.isIdle());
  assert.match(second.completed[0].error as string, /Unsupported draft options/);
  assert.deepEqual(second.calls, ['exclusive']);
  rejected.stop();
});

test('transcription lease downloads audio and returns timing with segments', async () => {
  const harness = fixture([lease('transcribe')]);
  const worker = createVolunteer(harness.dependencies);
  worker.start();
  await until(() => harness.isIdle());
  const result = harness.completed[0].result as L0TimingResponse;
  assert.equal(result.taskId, 'remote-task');
  assert.deepEqual(result.tracks[0].segments, timingResult.tracks[0].segments);
  assert.deepEqual(harness.requests.map((request) => request.url.slice(PUBLIC_L0_BASE_URL.length)), [
    '/v1/workers/register', '/v1/workers/lease',
    '/v1/jobs/job-1/audio/audio%3A1', '/v1/jobs/job-1/audio/audio%3A2', '/v1/jobs/job-1/complete'
  ]);
  worker.stop();
});

test('no ready local models means no registration and unleased audio URLs are rejected', async () => {
  const harness = fixture([], false);
  const worker = createVolunteer(harness.dependencies);
  worker.start();
  await until(() => worker.getStatus().state === 'disabled');
  assert.equal(harness.requests.length, 0);
  assert.throws(() => parseLease({
    ...lease('draft'), audio: [{ fieldName: 'audio:1', url: 'https://other.example/audio.wav' }]
  }), /cannot include audio/);
});

test('saved swarm opt-out prevents registration even with ready local models', async () => {
  const harness = fixture([lease('draft')]);
  harness.dependencies.settings = async () => ({
    ...DEFAULT_SETTINGS, mode: 'advanced', localModelsEnabled: true, volunteerInferenceEnabled: false
  });
  const worker = createVolunteer(harness.dependencies);
  worker.start();
  await until(() => worker.getStatus().state === 'disabled');
  assert.match(worker.getStatus().detail ?? '', /Swarm participation is off/);
  assert.deepEqual(harness.requests, []);
});


test('expired worker credentials re-register before another lease and back off between failures', async () => {
  const harness = fixture([]);
  const originalFetch = harness.dependencies.fetch;
  const originalWait = harness.dependencies.wait;
  let leaseCalls = 0;
  let registrations = 0;
  let failedRegistration = false;
  const delays: number[] = [];
  harness.dependencies.fetch = async (url, init) => {
    if (url.endsWith('/v1/workers/register')) {
      registrations += 1;
      if (!failedRegistration) {
        failedRegistration = true;
        throw new Error('offline');
      }
    }
    if (url.endsWith('/v1/workers/lease') && ++leaseCalls === 1) return new Response(null, { status: 401 });
    return originalFetch(url, init);
  };
  harness.dependencies.wait = async (delay, signal) => {
    delays.push(delay);
    if (delay === 2_000) return;
    return originalWait(delay, signal);
  };
  const worker = createVolunteer(harness.dependencies);
  worker.start();
  await until(() => harness.isIdle());
  assert.equal(registrations, 3);
  assert.equal(leaseCalls, 2);
  assert.deepEqual(delays, [2_000, 2_000, 3_000]);
  assert.equal(worker.getStatus().state, 'connected');
  worker.stop();
});

test('a busy volunteer does not request a second lease before finishing the first', async () => {
  const harness = fixture([lease('draft'), lease('draft')]);
  const { promise, resolve } = Promise.withResolvers<void>();
  const originalDraft = harness.dependencies.draft;
  harness.dependencies.draft = async (...args) => {
    await promise;
    return originalDraft(...args);
  };
  const worker = createVolunteer(harness.dependencies);
  worker.start();
  await until(() => worker.getStatus().state === 'busy' && harness.calls.includes('exclusive'));
  assert.equal(harness.requests.filter((request) => request.url.endsWith('/v1/workers/lease')).length, 1);
  assert.equal(harness.completed.length, 0);
  resolve();
  await until(() => harness.completed.length === 1);
  worker.stop();
});

test('service worker defers capability admission to offscreen and stops on disable', async () => {
  let enabled = true;
  let volunteerEnabled = true;
  let exists = false;
  const sent: string[] = [];
  const lifecycle = createVolunteerLifecycle({
    loadSettings: async () => ({
      ...DEFAULT_SETTINGS, mode: 'advanced', localModelsEnabled: enabled, volunteerInferenceEnabled: volunteerEnabled
    }),
    ready: async () => true,
    hasDocument: async () => exists,
    ensureDocument: async () => { exists = true; },
    sendMessage: async (message) => {
      sent.push(message.action);
      return { state: message.action === 'stop' ? 'disabled' : 'connected' };
    }
  });
  await lifecycle.reconcile();
  await lifecycle.reconcile();
  assert.deepEqual(sent, ['start', 'start']);
  assert.deepEqual(await lifecycle.status(), { state: 'connected' });
  volunteerEnabled = false;
  await lifecycle.reconcile();
  assert.deepEqual(sent, ['start', 'start', 'status', 'stop']);
  assert.match((await lifecycle.status()).detail ?? '', /local models remain available for your own tasks/);
  assert.equal(enabled, true);
  volunteerEnabled = true;
  await lifecycle.reconcile();
  assert.equal(sent.at(-1), 'start');
  assert.deepEqual(await lifecycle.status(), { state: 'connected' });
  enabled = false;
  await lifecycle.reconcile();
  assert.deepEqual(sent, ['start', 'start', 'status', 'stop', 'start', 'status', 'stop']);
  assert.equal((await lifecycle.status()).state, 'disabled');
  enabled = true;
  await lifecycle.reconcile();
  assert.equal(sent.at(-1), 'start');
  assert.equal((await lifecycle.status()).state, 'connected');
});

test('Simple mode does not register a volunteer despite retained Advanced local settings', async () => {
  const harness = fixture([lease('transcribe')]);
  harness.dependencies.settings = async () => ({
    ...DEFAULT_SETTINGS, mode: 'simple', localModelsEnabled: true, volunteerInferenceEnabled: true
  });
  const worker = createVolunteer(harness.dependencies);
  worker.start();
  await until(() => worker.getStatus().state === 'disabled');
  assert.deepEqual(harness.requests, []);
  assert.deepEqual(harness.calls, []);
});

test('switching to Simple stops volunteering without checking or deleting the retained bundle', async () => {
  let simple = false;
  const actions: string[] = [];
  const lifecycle = createVolunteerLifecycle({
    loadSettings: async () => ({
      ...DEFAULT_SETTINGS, mode: simple ? 'simple' : 'advanced',
      localModelsEnabled: true, volunteerInferenceEnabled: true
    }),
    ready: async () => true,
    hasDocument: async () => true,
    ensureDocument: async () => undefined,
    sendMessage: async (message) => {
      actions.push(message.action);
      return { state: message.action === 'stop' ? 'disabled' : 'connected' };
    }
  });
  await lifecycle.reconcile();
  simple = true;
  await lifecycle.reconcile();
  assert.deepEqual(actions, ['start', 'stop']);
  assert.equal((await lifecycle.status()).state, 'disabled');
  simple = false;
  await lifecycle.reconcile();
  assert.deepEqual(actions, ['start', 'stop', 'start']);
});

const enhancementModel = { id: ENHANCEMENT_MODEL_ID, sha256: 'e'.repeat(64), sourceGraphSha256: ENHANCEMENT_SOURCE_GRAPH_SHA256 };

async function enhancementFixture() {
  const bytes = encodeEnhancementWav(new Float32Array([0, 0.25, -0.25, 0]), 16000, 4);
  const sha256 = await enhancementSha256(bytes.buffer);
  const tracks = ['Original A', 'Original B'].map((trackId, index) => ({
    trackId, speakerKey: `speaker:${index}`, trackLabel: `Lane ${index + 1}`, fieldName: `audio:${index + 1}`,
    sourceSha256: sha256, sampleRate: 16000, frameCount: 4
  }));
  const enhancementLease = { jobId: 'job-1', leaseToken: 'lease-secret', operation: 'enhance',
    payload: { taskId: 'remote-task', model: enhancementModel, tracks }, audio: lease('transcribe').audio };
  const result: EnhancedAudioBatch = { provider: 'browser-local', model: enhancementModel.id, modelSha256: enhancementModel.sha256,
    tracks: tracks.map(({ fieldName: _fieldName, ...track }) => ({
      bytes, metadata: { ...track, mimeType: 'audio/wav', wavSha256: sha256, totalBytes: bytes.byteLength, chunkCount: 1 }
    })) };
  const harness = fixture([enhancementLease], false);
  const progress: unknown[] = [];
  let uploaded: FormData | undefined;
  harness.dependencies.ready = async () => ({ transcribe: false, draft: false, enhancementModel });
  harness.dependencies.enhance = async (audio, onProgress, options) => {
    harness.calls.push('enhance');
    assert.deepEqual(audio.map(track => track.trackId), tracks.map(track => track.trackId));
    assert.equal(options?.taskId, 'remote-task');
    assert.equal(options?.localOnly, true);
    assert.equal(options?.cache, false);
    assert.equal(options?.signal?.aborted, false);
    for (let trackIndex = 0; trackIndex < 2; trackIndex++) {
      const track = { trackId: tracks[trackIndex].trackId, trackIndex, trackCount: 2 };
      await onProgress?.({ ...track, phase: 'enhancing', completedChunks: 0, totalChunks: 1, backend: 'webgpu' });
      await onProgress?.({ ...track, phase: 'enhancing', completedChunks: 1, totalChunks: 1, backend: 'webgpu' });
      await onProgress?.({ ...track, phase: 'encoding', completedChunks: 1, totalChunks: 1, backend: 'webgpu' });
    }
    return result;
  };
  const originalFetch = harness.dependencies.fetch;
  harness.dependencies.fetch = async (url, init) => {
    if (url.includes('/audio/')) {
      harness.requests.push({ url, init });
      assert.equal(new Headers(init.headers).get('Authorization'), 'Bearer lease-secret');
      return new Response(bytes, { headers: { 'Content-Type': 'audio/wav' } });
    }
    if (url.endsWith('/progress')) {
      harness.requests.push({ url, init });
      progress.push(JSON.parse(String(init.body)));
      return Response.json({ ok: true });
    }
    if (url.endsWith('/complete') && init.body instanceof FormData) {
      harness.requests.push({ url, init });
      assert.equal(new Headers(init.headers).get('Authorization'), 'Bearer lease-secret');
      uploaded = init.body;
      harness.completed.push(JSON.parse(String(init.body.get('payload'))));
      return Response.json({ ok: true });
    }
    return originalFetch(url, init);
  };
  return { ...harness, enhancementLease, result, bytes, progress, uploaded: () => uploaded };
}

test('enhancement-only GPU volunteer registers without ASR, reports real chunks, and completes binary without persistent cache', async () => {
  const harness = await enhancementFixture();
  const worker = createVolunteer(harness.dependencies);
  worker.start();
  await until(() => harness.isIdle());
  assert.equal(worker.getStatus().state, 'connected');
  assert.deepEqual(JSON.parse(String(harness.requests[0].init.body)).operations, ['enhance']);
  assert.deepEqual(JSON.parse(String(harness.requests[0].init.body)).enhancementModel, enhancementModel);
  assert.deepEqual(harness.calls, ['exclusive', 'enhance']);
  assert.equal(harness.progress.length, 6);
  assert.deepEqual(harness.progress[0], { workerId: 'volunteer-1', token: 'worker-secret', leaseToken: 'lease-secret',
    progress: { phase: 'enhancing', trackId: 'Original A', trackIndex: 0, trackCount: 2, completedChunks: 0, totalChunks: 1 } });
  const uploaded = harness.uploaded()!;
  assert.ok(uploaded instanceof FormData);
  assert.deepEqual(Array.from(uploaded.keys()).sort(), ['audio:1', 'audio:2', 'payload']);
  assert.deepEqual(new Uint8Array(await (uploaded.get('audio:1') as Blob).arrayBuffer()), harness.bytes);
  assert.equal((harness.completed[0].result as EnhancedAudioBatch).modelSha256, enhancementModel.sha256);
  worker.stop();
});

test('volunteer uses only the normalized saved coordinator for registration, audio, progress, and completion', async () => {
  const harness = await enhancementFixture();
  const originalFetch = harness.dependencies.fetch;
  const urls: string[] = [];
  harness.dependencies.settings = async () => ({ ...DEFAULT_SETTINGS, volunteerInferenceEnabled: true,
    l0CustomBaseUrl: 'http://127.0.0.1:8976/private/?discard=yes#fragment' });
  harness.dependencies.fetch = async (url, init) => {
    urls.push(url);
    assert.equal(init.redirect, 'error');
    return originalFetch(url.replace('http://127.0.0.1:8976/private', PUBLIC_L0_BASE_URL), init);
  };
  const worker = createVolunteer(harness.dependencies);
  worker.start();
  await until(() => harness.isIdle());
  assert.ok(urls.length > 5);
  assert.ok(urls.every(url => url.startsWith('http://127.0.0.1:8976/private/v1/')));
  worker.stop();
});

test('enhancement lease rejects mismatched model admission before fetching audio or invoking inference', async () => {
  const harness = await enhancementFixture();
  harness.dependencies.ready = async () => ({ transcribe: false, draft: false, enhancementModel: { ...enhancementModel, sha256: 'f'.repeat(64) } });
  const worker = createVolunteer(harness.dependencies);
  worker.start();
  await until(() => harness.isIdle());
  assert.match(String(harness.completed[0].error), /not admitted/);
  assert.deepEqual(harness.calls, []);
  assert.equal(harness.requests.some(request => request.url.includes('/audio/')), false);
  worker.stop();
});

test('corrupt or oversized leased Originals fail before GPU execution and never produce output', async () => {
  for (const oversized of [false, true]) {
    const harness = await enhancementFixture();
    const originalFetch = harness.dependencies.fetch;
    harness.dependencies.fetch = async (url, init) => url.includes('/audio/')
      ? new Response(oversized ? harness.bytes : new Uint8Array(48), { headers: oversized ? { 'Content-Length': String(ENHANCEMENT_MAX_TRACK_BYTES + 1) } : {} })
      : originalFetch(url, init);
    const worker = createVolunteer(harness.dependencies);
    worker.start();
    await until(() => harness.isIdle());
    assert.ok(harness.completed[0].error);
    assert.equal(harness.uploaded(), undefined);
    assert.deepEqual(harness.calls, []);
    worker.stop();
  }
});

test('expired enhancement progress halts execution and reports failure rather than uploading output', async () => {
  const harness = await enhancementFixture();
  const originalFetch = harness.dependencies.fetch;
  harness.dependencies.fetch = async (url, init) => url.endsWith('/progress') ? new Response(null, { status: 409 }) : originalFetch(url, init);
  const worker = createVolunteer(harness.dependencies);
  worker.start();
  await until(() => harness.isIdle());
  assert.match(String(harness.completed[0].error), /progress rejected: HTTP 409/);
  assert.equal(harness.uploaded(), undefined);
  worker.stop();
});

test('stop aborts enhancement execution and prevents late completion', async () => {
  const harness = await enhancementFixture();
  let executionSignal: AbortSignal | undefined;
  const pending = Promise.withResolvers<EnhancedAudioBatch>();
  harness.dependencies.enhance = async (_tracks, _progress, options) => {
    executionSignal = options?.signal;
    return pending.promise;
  };
  const worker = createVolunteer(harness.dependencies);
  worker.start();
  await until(() => executionSignal !== undefined);
  worker.stop();
  assert.equal(executionSignal!.aborted, true);
  pending.resolve(harness.result);
  await setImmediate();
  assert.deepEqual(harness.completed, []);
  assert.equal(worker.getStatus().state, 'disabled');
});

test('enhancement lease parser rejects duplicate fields, foreign URLs, and unpinned model graphs', async () => {
  const { enhancementLease } = await enhancementFixture();
  assert.throws(() => parseLease({ ...enhancementLease, audio: [enhancementLease.audio[0], enhancementLease.audio[0]] }), /audio URLs/);
  assert.throws(() => parseLease({ ...enhancementLease, audio: [{ ...enhancementLease.audio[0], url: 'https://foreign.invalid/audio' }, enhancementLease.audio[1]] }), /audio URLs/);
  assert.throws(() => parseLease({ ...enhancementLease, payload: { ...enhancementLease.payload,
    model: { ...enhancementModel, sourceGraphSha256: '0'.repeat(64) } } }), /model descriptor/);
  assert.throws(() => parseLease({ ...enhancementLease, payload: { ...enhancementLease.payload, transcript: 'private transcript' } }), /payload/);
});

test('without grader access a volunteer cannot probe or advertise enhancement hardware', async t => {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  let requested = 0;
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { gpu: {
    requestAdapter: async () => { requested++; throw new Error('Enhancement hardware must remain untouched'); }
  } } });
  t.after(() => { if (previous) Object.defineProperty(globalThis, 'navigator', previous); else Reflect.deleteProperty(globalThis, 'navigator'); });
  const admission = await defaultVolunteerDependencies.ready();
  assert.equal(admission.enhancementModel, undefined);
  assert.equal(requested, 0);
});

test('worker independently refuses corrupt or remote runtime results before multipart completion', async () => {
  for (const remote of [false, true]) {
    const harness = await enhancementFixture();
    harness.dependencies.enhance = async () => remote
      ? { ...harness.result, provider: 'swarm' }
      : { ...harness.result, tracks: [{ ...harness.result.tracks[0],
        metadata: { ...harness.result.tracks[0].metadata, wavSha256: '0'.repeat(64) } }, harness.result.tracks[1]] };
    const worker = createVolunteer(harness.dependencies);
    worker.start();
    await until(() => harness.isIdle());
    assert.match(String(harness.completed[0].error), remote ? /different enhancement model/ : /SHA-256 mismatch/);
    assert.equal(harness.uploaded(), undefined);
    worker.stop();
  }
});

test('losing grader access after registration prevents leased audio download and inference', async () => {
  const harness = await enhancementFixture();
  harness.dependencies.authorizeEnhancement = async () => { throw new Error('This operation is unavailable.'); };
  const worker = createVolunteer(harness.dependencies);
  worker.start();
  await until(() => harness.isIdle());
  assert.deepEqual(harness.calls, []);
  assert.equal(harness.requests.some(request => request.url.includes('/audio/')), false);
  assert.match(String(harness.completed[0].error), /operation is unavailable/);
  worker.stop();
});
