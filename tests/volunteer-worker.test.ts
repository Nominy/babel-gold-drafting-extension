import test from 'node:test';
import assert from 'node:assert/strict';
import { setImmediate } from 'node:timers/promises';
import { createVolunteer, defaultVolunteerDependencies, parseLease, type VolunteerDependencies } from '../src/offscreen/volunteer';
import { createVolunteerLifecycle } from '../src/background/local-model-offscreen';
import { DEFAULT_SETTINGS, PUBLIC_L0_BASE_URL } from '../src/core/settings';
import type { L0DraftResponse, L0TimingResponse } from '../src/core/types';

const wav = new Blob([new Uint8Array([82, 73, 70, 70, 0, 0, 0, 0, 87, 65, 86, 69, 0])], { type: 'audio/wav' });
const lease = (operation: 'draft' | 'transcribe', options?: Record<string, unknown>) => ({
  jobId: 'job-1', leaseToken: 'lease-secret', operation,
  payload: {
    taskId: 'remote-task', tracks: [
      { lane: 'A', fieldName: 'audio:1' }, { lane: 'B', fieldName: 'audio:2' }
    ], ...(options ? { options } : {})
  },
  audio: [
    { fieldName: 'audio:1', url: '/v1/jobs/job-1/audio/audio%3A1' },
    { fieldName: 'audio:2', url: '/v1/jobs/job-1/audio/audio%3A2' }
  ]
});
const draftResult: L0DraftResponse = {
  rows: [{ id: 'row', lane: 'A', startSeconds: 0, endSeconds: 1, text: 'Hello' }],
  summary: {}, models: {}
};
const timingResult: L0TimingResponse = {
  taskId: 'other-task', summary: {}, models: {},
  tracks: [
    { lane: 'A', tokens: [{ id: 'old', text: 'Hello', startSeconds: 0, endSeconds: 1 }] },
    { lane: 'B', tokens: [] }
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
    settings: async () => ({ ...DEFAULT_SETTINGS, localModelsEnabled: true }),
    ready: async () => ready,
    runExclusive: async (action) => { calls.push('exclusive'); return action(); },
    draft: async (_settings, job, tracks) => {
      assert.equal(job.jobId, 'remote-task');
      assert.deepEqual(tracks.map((track) => track.speakerKey), ['A', 'B']);
      calls.push('draft');
      return draftResult;
    },
    segment: async (_settings, taskId, row) => {
      assert.equal(taskId, 'remote-task');
      calls.push(`segment:${row.rowId}`);
      return 'Preserved text';
    },
    transcribe: async () => { calls.push('transcribe'); return timingResult; },
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

test('worker leases exactly one remote draft, fetches both authorized WAVs and completes with worker and lease credentials', async () => {
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
    '/v1/workers/register', '/v1/workers/lease',
    '/v1/jobs/job-1/audio/audio%3A1', '/v1/jobs/job-1/audio/audio%3A2', '/v1/jobs/job-1/complete'
  ]);
  worker.stop();
  assert.equal(worker.getStatus().state, 'disabled');
});

test('preserveRows uses segment inference while unsupported options report an error to the coordinator', async () => {
  const row = { rowId: 'existing', speakerKey: 'A', startSeconds: 0, endSeconds: 1, text: '', index: 0 };
  const first = fixture([lease('draft', { preserveRows: [row] })]);
  const volunteer = createVolunteer(first.dependencies);
  volunteer.start();
  await until(() => first.isIdle());
  assert.deepEqual(first.calls, ['exclusive', 'segment:existing']);
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

test('timing response normalizes task and token identities to leased request', async () => {
  const harness = fixture([lease('transcribe')]);
  const worker = createVolunteer(harness.dependencies);
  worker.start();
  await until(() => harness.isIdle());
  const result = harness.completed[0].result as L0TimingResponse;
  assert.equal(result.taskId, 'remote-task');
  assert.equal(result.summary.taskId, 'remote-task');
  assert.equal(result.tracks[0].tokens[0].id, 'remote-task:A:0');
  worker.stop();
});

test('no ready local models means no registration and unleased audio URLs are rejected', async () => {
  const harness = fixture([], false);
  const worker = createVolunteer(harness.dependencies);
  worker.start();
  await until(() => worker.getStatus().state === 'disabled');
  assert.equal(harness.requests.length, 0);
  assert.throws(() => parseLease({
    ...lease('draft'), audio: [{ fieldName: 'audio:1', url: 'https://other.example/audio.wav' },
      { fieldName: 'audio:2', url: '/v1/jobs/job-1/audio/audio%3A2' }]
  }), /Invalid volunteer lease/);
});

test('saved swarm opt-out prevents registration even with ready local models', async () => {
  const harness = fixture([lease('draft')]);
  harness.dependencies.settings = async () => ({
    ...DEFAULT_SETTINGS, localModelsEnabled: true, volunteerInferenceEnabled: false
  });
  const worker = createVolunteer(harness.dependencies);
  worker.start();
  await until(() => worker.getStatus().state === 'disabled');
  assert.match(worker.getStatus().detail ?? '', /Swarm participation is off/);
  assert.deepEqual(harness.requests, []);
});

test('offscreen volunteer starts from saved settings without access to chrome.storage', async () => {
  const paths = [
    'asr/v3_ctc.onnx', 'asr/v3_ctc.yaml', 'punctuation/model.fp16.onnx',
    'punctuation/config.json', 'punctuation/tokenizer.json',
    'punctuation/tokenizer_config.json', 'punctuation/special_tokens_map.json',
    'punctuation/vocab.txt'
  ];
  const manifest = {
    schema: 'babel-browser-model-bundle-v2', targetBytes: 1_500_000_000,
    pass: true, totalBytes: 1,
    files: paths.map((path, index) => ({ path, bytes: index === 0 ? 1 : 0, sha256: '0'.repeat(64) }))
  };
  const sent: unknown[] = [];
  Object.assign(globalThis, {
    chrome: {
      runtime: {
        sendMessage: async (message: unknown) => {
          sent.push(message);
          return { ...DEFAULT_SETTINGS, localModelsEnabled: true };
        }
      }
    },
    caches: {
      keys: async () => ['babel-gold-local-models:bundle:installed'],
      open: async () => ({
        match: async (url: string) => url.endsWith('/manifest.json')
          ? Response.json(manifest) : new Response('cached model file')
      })
    }
  });
  try {
    const harness = fixture([lease('draft')]);
    const volunteer = createVolunteer({
      ...harness.dependencies,
      settings: defaultVolunteerDependencies.settings,
      ready: defaultVolunteerDependencies.ready
    });
    volunteer.start();
    await until(() => harness.isIdle());
    assert.equal(volunteer.getStatus().state, 'connected');
    assert.deepEqual(sent, [{ type: 'babel-l0-volunteer', target: 'background', action: 'settings' }]);
    assert.deepEqual(harness.calls, ['exclusive', 'draft']);
    volunteer.stop();
  } finally {
    Reflect.deleteProperty(globalThis, 'chrome');
    Reflect.deleteProperty(globalThis, 'caches');
  }
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

test('service worker starts a ready worker and stops on disable or missing bundle', async () => {
  let enabled = true;
  let volunteerEnabled = true;
  let ready = true;
  let exists = false;
  const sent: string[] = [];
  const lifecycle = createVolunteerLifecycle({
    loadSettings: async () => ({
      ...DEFAULT_SETTINGS, localModelsEnabled: enabled, volunteerInferenceEnabled: volunteerEnabled
    }),
    ready: async () => ready,
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
  assert.deepEqual(await lifecycle.status(), { state: 'disabled', detail: undefined });
  enabled = true;
  ready = false;
  await lifecycle.reconcile();
  assert.equal(sent.at(-1), 'stop');
  assert.equal((await lifecycle.status()).state, 'disabled');
});
