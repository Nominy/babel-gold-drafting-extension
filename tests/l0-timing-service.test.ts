import test from 'node:test';
import assert from 'node:assert/strict';
import {
  isUsableL0TimingJob,
  L0_TIMING_UPDATE_MESSAGE_TYPE,
  L0TimingService,
  type L0TimingServiceDependencies
} from '../src/content/l0-timing-service';
import { getL0TimingAvailability, publishL0TimingAvailability } from '../src/content/l0-timing-availability';
import { DEFAULT_SETTINGS } from '../src/core/settings';
import { buildCanonicalTaskIdentity } from '../src/core/transcript';
import type { CapturedAudioTrack, ExtensionSettings, L0TimingResponse, TranscriptJob } from '../src/core/types';

const job: TranscriptJob = {
  jobId: 'task-42',
  rows: [
    { rowId: 'row-1', speakerKey: 'Speaker 1', processedRecordingId: 'lane-1', startSeconds: 0, endSeconds: 1, text: 'one', index: 0 },
    { rowId: 'row-2', speakerKey: 'Speaker 2', processedRecordingId: 'lane-2', startSeconds: 1, endSeconds: 2, text: 'two', index: 1 }
  ]
};
const taskId = buildCanonicalTaskIdentity(job);
const tracks: CapturedAudioTrack[] = [
  { trackId: 'one', speakerKey: 'Speaker 1', source: 'one.wav', blob: new Blob(['one']), mimeType: 'audio/wav' },
  { trackId: 'two', speakerKey: 'Speaker 2', source: 'two.wav', blob: new Blob(['two']), mimeType: 'audio/wav' }
];
const response: L0TimingResponse = {
  taskId,
  tracks: [
    { lane: 'Speaker 1', tokens: [{ id: 'token-1', text: 'one', startSeconds: 0, endSeconds: 0.5 }], segments: [{ id: 'segment-1', startSeconds: 0, endSeconds: 1, startSample: 0, endSample: 16000, sampleRate: 16000 }], pcmSha256: 'a'.repeat(64), sampleRate: 16000 },
    { lane: 'Speaker 2', tokens: [], segments: [], pcmSha256: 'b'.repeat(64), sampleRate: 16000 }
  ],
  summary: {},
  models: {}
};

function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

async function flushAsyncWork(): Promise<void> {
  await new Promise<void>((resolve) => setImmediate(resolve));
}

function dependencies(overrides: Partial<L0TimingServiceDependencies> = {}): L0TimingServiceDependencies {
  return {
    captureTranscript: () => job,
    currentTaskId: () => taskId,
    currentPathname: () => '/tasks/current',
    captureAudio: async () => tracks,
    getSettings: async () => ({ ...DEFAULT_SETTINGS, mode: 'advanced' }),
    lookupTiming: async () => null,
    localModelStatus: async () => ({ state: 'ready', completedBytes: 1, totalBytes: 1, tested: true }),
    requestLocalDraft: async () => ({ rows: [], summary: {}, models: {} }),
    requestTiming: async () => response,
    publish: () => undefined,
    now: () => 1_000,
    schedule: () => undefined,
    ...overrides
  };
}

test('usable timing jobs accept transcripts with one or more speaker lanes', () => {
  assert.equal(isUsableL0TimingJob(job), true);
  assert.equal(isUsableL0TimingJob({ jobId: 'task-42', rows: [] }), false);
  assert.equal(isUsableL0TimingJob({ jobId: 'task-42', taskScoped: true, rows: [] }), true);
  assert.equal(isUsableL0TimingJob({ ...job, rows: [job.rows[0]] }), true);
});

test('timing waits for both nonempty speaker tracks without starting a model or exhausting retries', async (t) => {
  const scheduled: Array<() => void> = [];
  const delays: number[] = [];
  let captured: CapturedAudioTrack[] = [];
  let requested = 0;
  const errors: unknown[] = [];
  t.mock.method(console, 'error', (...args: unknown[]) => errors.push(args));
  const service = new L0TimingService(dependencies({
    captureAudio: async () => captured,
    requestTiming: async () => { requested += 1; return response; },
    schedule: (callback, delay) => { scheduled.push(callback); delays.push(delay); }
  }));
  service.onLifecycleOpportunity();
  await flushAsyncWork();
  for (let index = 0; index < 5; index += 1) {
    captured = index < 3 ? [tracks[0]] : [tracks[0], { ...tracks[1], blob: new Blob([]) }];
    service.onLifecycleOpportunity();
    assert.equal(scheduled.length, 1, 'DOM churn must not schedule duplicate captures');
    scheduled.shift()!();
    await flushAsyncWork();
  }
  assert.equal(requested, 0);
  assert.deepEqual(errors, []);
  assert.deepEqual(getL0TimingAvailability(), { taskId, status: 'preparing' });
  assert.ok(delays.every((delay) => delay >= 5_000 && delay <= 30_000));
  captured = tracks;
  scheduled.shift()!();
  await flushAsyncWork();
  assert.equal(requested, 1);
  assert.deepEqual(getL0TimingAvailability(), { taskId, status: 'available' });
});

test('cached remote timing publishes tokens without capturing any audio', async () => {
  const published: unknown[] = [];
  const service = new L0TimingService(dependencies({
    captureAudio: async () => { throw new Error('cache hit must not capture audio'); },
    lookupTiming: async (_settings, lookedUpTaskId) => {
      assert.equal(lookedUpTaskId, taskId);
      return response;
    },
    publish: (message) => published.push(message)
  }));
  service.onLifecycleOpportunity();
  await flushAsyncWork();
  assert.deepEqual(published, [{
    type: L0_TIMING_UPDATE_MESSAGE_TYPE,
    version: 1,
    taskId,
    tracks: response.tracks.map(({ lane, tokens }) => ({ lane, tokens }))
  }]);
});

test('an audio readiness timer cannot start timing after navigation', async () => {
  let current = taskId;
  const scheduled: Array<() => void> = [];
  let captures = 0;
  const service = new L0TimingService(dependencies({
    currentTaskId: () => current,
    captureAudio: async () => { captures += 1; return []; },
    schedule: (callback) => scheduled.push(callback)
  }));
  service.onLifecycleOpportunity();
  await flushAsyncWork();
  current = 'another-task';
  scheduled.shift()!();
  await flushAsyncWork();
  assert.equal(captures, 1);
});

test('timing lifecycle deduplicates in-flight and successful tasks and publishes only the exact contract', async () => {
  const pending = deferred<L0TimingResponse>();
  const published: unknown[] = [];
  let requestCount = 0;
  const service = new L0TimingService(dependencies({
    requestTiming: async () => {
      requestCount += 1;
      return pending.promise;
    },
    publish: (message) => published.push(message)
  }));

  service.onLifecycleOpportunity();
  service.onLifecycleOpportunity();
  await flushAsyncWork();
  assert.equal(requestCount, 1);
  pending.resolve(response);
  await flushAsyncWork();
  assert.deepEqual(published, [{
    type: L0_TIMING_UPDATE_MESSAGE_TYPE,
    version: 1,
    taskId,
    tracks: response.tracks.map(({ lane, tokens }) => ({ lane, tokens }))
  }]);
  service.onLifecycleOpportunity();
  await flushAsyncWork();
  assert.equal(requestCount, 1);
});

test('timing lifecycle preserves queued and running availability across DOM changes until completion', async () => {
  const service = new L0TimingService(dependencies({
    requestTiming: async (_settings, _job, _tracks, callbacks) => {
      callbacks.onQueueStatus?.({
        requestId: 'request-1',
        status: 'queued',
        position: 2,
        queuedCount: 2
      });
      service.onLifecycleOpportunity();
      assert.deepEqual(getL0TimingAvailability(), { taskId, status: 'queued', position: 2 });
      callbacks.onQueueStatus?.({
        requestId: 'request-1',
        status: 'running',
        position: 0,
        queuedCount: 1
      });
      service.onLifecycleOpportunity();
      assert.deepEqual(getL0TimingAvailability(), { taskId, status: 'running' });
      return response;
    }
  }));

  service.onLifecycleOpportunity();
  await flushAsyncWork();
  assert.deepEqual(getL0TimingAvailability(), { taskId, status: 'available' });
});

test('timing lifecycle suppresses stale task results', async () => {
  const pending = deferred<L0TimingResponse>();
  const published: unknown[] = [];
  let currentTaskId = taskId;
  const service = new L0TimingService(dependencies({
    currentTaskId: () => currentTaskId,
    requestTiming: async () => pending.promise,
    publish: (message) => published.push(message)
  }));

  service.onLifecycleOpportunity();
  await flushAsyncWork();
  currentTaskId = buildCanonicalTaskIdentity({ ...job, rows: [{ ...job.rows[0], processedRecordingId: 'lane-3' }] });
  pending.resolve(response);
  await flushAsyncWork();
  assert.deepEqual(published, []);
});

test('timing lifecycle reruns on the same route when processed recording identity changes', async () => {
  let currentJob = job;
  let currentTaskId = taskId;
  const published: Array<Parameters<L0TimingServiceDependencies['publish']>[0]> = [];
  let requestCount = 0;
  const service = new L0TimingService(dependencies({
    captureTranscript: () => currentJob,
    currentTaskId: () => currentTaskId,
    requestTiming: async (_settings, requestedJob) => {
      requestCount += 1;
      return { ...response, taskId: buildCanonicalTaskIdentity(requestedJob) };
    },
    publish: (message) => published.push(message)
  }));

  service.onLifecycleOpportunity();
  await flushAsyncWork();
  currentJob = {
    ...job,
    rows: job.rows.map((row) => ({
      ...row,
      processedRecordingId: `${row.processedRecordingId}-next`
    }))
  };
  currentTaskId = buildCanonicalTaskIdentity(currentJob);
  service.onLifecycleOpportunity();
  await flushAsyncWork();

  assert.equal(requestCount, 2);
  assert.deepEqual(published.map((message) => message.taskId), [taskId, currentTaskId]);
});

test('timing lifecycle reruns for a new review action with identical recording lanes', async () => {
  let currentJob = job;
  let currentTaskId = taskId;
  const requestedTaskIds: string[] = [];
  const service = new L0TimingService(dependencies({
    captureTranscript: () => currentJob,
    currentTaskId: () => currentTaskId,
    requestTiming: async (_settings, requestedJob) => {
      const requestedTaskId = buildCanonicalTaskIdentity(requestedJob);
      requestedTaskIds.push(requestedTaskId);
      return { ...response, taskId: requestedTaskId };
    }
  }));

  service.onLifecycleOpportunity();
  await flushAsyncWork();
  currentJob = { ...job, jobId: 'review-action-next' };
  currentTaskId = buildCanonicalTaskIdentity(currentJob);
  service.onLifecycleOpportunity();
  await flushAsyncWork();

  assert.deepEqual(requestedTaskIds, [taskId, currentTaskId]);
});

test('timing lifecycle contains failures and retries only after backoff', async () => {
  const scheduled: Array<() => void> = [];
  let now = 1_000;
  let requestCount = 0;
  const service = new L0TimingService(dependencies({
    now: () => now,
    requestTiming: async () => {
      requestCount += 1;
      if (requestCount === 1) throw new Error('background ASR unavailable');
      return response;
    },
    schedule: (callback) => scheduled.push(callback)
  }));

  assert.doesNotThrow(() => service.onLifecycleOpportunity());
  await flushAsyncWork();
  assert.equal(requestCount, 1);
  assert.equal(scheduled.length, 1);
  service.onLifecycleOpportunity();
  await flushAsyncWork();
  assert.equal(requestCount, 1);
  now = 6_000;
  assert.doesNotThrow(() => scheduled[0]());
  await flushAsyncWork();
  assert.equal(requestCount, 2);
});

test('timing lifecycle exposes unavailable after bounded retries and gates manual regeneration', async () => {
  const scheduled: Array<() => void> = [];
  let requestCount = 0;
  const service = new L0TimingService(dependencies({
    requestTiming: async () => {
      requestCount += 1;
      if (requestCount <= 4) throw new Error('background ASR unavailable');
      return response;
    },
    schedule: (callback) => scheduled.push(callback)
  }));

  service.onLifecycleOpportunity();
  await flushAsyncWork();
  for (let retryIndex = 0; retryIndex < 3; retryIndex += 1) {
    scheduled[retryIndex]();
    await flushAsyncWork();
  }

  assert.equal(requestCount, 4);
  assert.deepEqual(getL0TimingAvailability(), { taskId, status: 'unavailable' });
  service.onLifecycleOpportunity();
  await flushAsyncWork();
  assert.equal(requestCount, 4);
  assert.deepEqual(getL0TimingAvailability(), { taskId, status: 'unavailable' });
  assert.equal(service.retryCurrentTask(), true);
  assert.equal(service.retryCurrentTask(), false);
  assert.deepEqual(getL0TimingAvailability(), { taskId, status: 'preparing' });
  await flushAsyncWork();
  assert.equal(requestCount, 5);
  assert.deepEqual(getL0TimingAvailability(), { taskId, status: 'available' });
  assert.equal(service.retryCurrentTask(), false);
});


for (const mode of ['new-task-event', 'unmounted-route', 'stale-task-identity'] as const) {
  test(`timing wait rejects on ${mode} and allows cleanup before the old request settles`, async () => {
    const pending = deferred<L0TimingResponse>();
    let current = taskId;
    let unmounted = false;
    let pathname = '/tasks/current';
    let busy = true;
    let currentChecks = 0;
    const service = new L0TimingService(dependencies({
      currentPathname: () => pathname,
      captureTranscript: () => {
        if (unmounted) throw new Error('No transcript on this route');
        return job;
      },
      currentTaskId: () => {
        currentChecks += 1;
        if (unmounted) throw new Error('No current task');
        return current;
      },
      requestTiming: async () => pending.promise,
      getSettings: async () => ({ ...DEFAULT_SETTINGS, mode: 'advanced', localModelsEnabled: true }),
    }));
    publishL0TimingAvailability({ taskId, status: 'preparing' });
    const wait = service.waitForTiming(job, { ...DEFAULT_SETTINGS, mode: 'advanced', localModelsEnabled: true })
      .finally(() => { busy = false; });
    const rejected = assert.rejects(wait, /task changed/);
    await flushAsyncWork();
    current = mode === 'stale-task-identity' ? taskId : 'next-task';
    if (mode === 'new-task-event') {
      publishL0TimingAvailability({ taskId: current, status: 'preparing' });
    } else {
      unmounted = mode === 'unmounted-route';
      pathname = '/projects';
      service.onLifecycleOpportunity();
    }
    await rejected;
    assert.equal(busy, false);
    const checksAfterRejection = currentChecks;
    publishL0TimingAvailability({ taskId, status: 'available' });
    assert.equal(currentChecks, checksAfterRejection, 'settled waits must unsubscribe');
    pending.resolve(response);
    await flushAsyncWork();
    assert.equal(busy, false);
  });
}

const simpleSettings: ExtensionSettings = {
  ...DEFAULT_SETTINGS, mode: 'simple', openRouterApiKey: 'test-key',
  localModelsEnabled: true, l0DontRunLlm: false, l0ReplacementPreviewEnabled: false
};

test('Simple page lifecycle only looks up cached timing and never captures, uploads or starts transcription', async () => {
  let lookups = 0;
  let captures = 0;
  let paidRequests = 0;
  let cached: L0TimingResponse | null = null;
  const published: unknown[] = [];
  const service = new L0TimingService(dependencies({
    getSettings: async () => simpleSettings,
    lookupTiming: async () => { lookups += 1; return cached; },
    captureAudio: async () => { captures += 1; return tracks; },
    requestTiming: async () => { paidRequests += 1; return response; },
    publish: (message) => published.push(message)
  }));
  service.onLifecycleOpportunity();
  await flushAsyncWork();
  assert.deepEqual(getL0TimingAvailability(), { taskId, status: 'unavailable' });
  cached = response;
  service.onLifecycleOpportunity();
  await flushAsyncWork();
  assert.equal(lookups, 2);
  assert.equal(captures, 0);
  assert.equal(paidRequests, 0);
  assert.deepEqual(published, [{
    type: L0_TIMING_UPDATE_MESSAGE_TYPE, version: 1, taskId,
    tracks: response.tracks.map(({ lane, tokens }) => ({ lane, tokens }))
  }]);
});

test('Simple explicit actions deduplicate transcription and await real completion rather than the running event', async () => {
  const pending = deferred<L0TimingResponse>();
  let paidRequests = 0;
  let finished = false;
  let cached: L0TimingResponse | null = null;
  const service = new L0TimingService(dependencies({
    getSettings: async () => simpleSettings,
    lookupTiming: async () => cached,
    requestTiming: async (_settings, _job, _tracks, callbacks) => {
      paidRequests += 1;
      callbacks.onQueueStatus?.({ requestId: 'mai-1', status: 'running', position: 0, queuedCount: 0 });
      cached = await pending.promise;
      return cached;
    }
  }));
  const first = service.waitForTiming(job, simpleSettings).then(() => { finished = true; });
  const concurrent = service.waitForTiming(job, simpleSettings);
  await flushAsyncWork();
  assert.equal(paidRequests, 1);
  assert.equal(finished, false);
  pending.resolve(response);
  await Promise.all([first, concurrent]);
  assert.equal(finished, true);
  assert.deepEqual(getL0TimingAvailability(), { taskId, status: 'available' });
  await service.waitForTiming(job, simpleSettings);
  assert.equal(paidRequests, 1, 'repeated explicit actions reuse durable timing');
});

test('Simple failures surface to the explicit caller and never schedule automatic paid retries', async () => {
  let paidRequests = 0;
  let timers = 0;
  const failure = new Error('OpenRouter rejected the key');
  const service = new L0TimingService(dependencies({
    getSettings: async () => simpleSettings,
    requestTiming: async () => { paidRequests += 1; throw failure; },
    schedule: () => { timers += 1; }
  }));
  await assert.rejects(service.waitForTiming(job, simpleSettings), (error) => error === failure);
  for (let index = 0; index < 3; index += 1) {
    service.onLifecycleOpportunity();
    await flushAsyncWork();
  }
  assert.equal(paidRequests, 1);
  assert.equal(timers, 0);
});

test('Simple requires a key before audio capture or a paid request even with legacy local settings enabled', async () => {
  const settings = { ...simpleSettings, openRouterApiKey: '' };
  const service = new L0TimingService(dependencies({
    getSettings: async () => settings,
    captureAudio: async () => { assert.fail('missing key must not capture audio'); },
    requestTiming: async () => { assert.fail('missing key must not start paid transcription'); }
  }));
  await assert.rejects(service.waitForTiming(job, settings), /OpenRouter API key.*extension options/i);
});

test('lost Simple session timing is regenerated only by an explicit action, not a stale completed flag', async () => {
  let cached: L0TimingResponse | null = response;
  let paidRequests = 0;
  const service = new L0TimingService(dependencies({
    getSettings: async () => simpleSettings,
    lookupTiming: async () => cached,
    requestTiming: async () => { paidRequests += 1; cached = response; return response; }
  }));
  await service.waitForTiming(job, simpleSettings);
  cached = null;
  service.onLifecycleOpportunity();
  await flushAsyncWork();
  assert.equal(paidRequests, 0);
  assert.deepEqual(getL0TimingAvailability(), { taskId, status: 'unavailable' });
  await service.waitForTiming(job, simpleSettings);
  assert.equal(paidRequests, 1);
  assert.deepEqual(getL0TimingAvailability(), { taskId, status: 'available' });
});

test('mode changes on the same task invalidate in-flight ownership and do not reuse Advanced timing as Simple timing', async () => {
  let settings: ExtensionSettings = { ...DEFAULT_SETTINGS, mode: 'advanced', localModelsEnabled: true };
  const oldResult = deferred<L0TimingResponse>();
  const published: unknown[] = [];
  let paidSimple = 0;
  const service = new L0TimingService(dependencies({
    getSettings: async () => settings,
    requestTiming: async (requestedSettings) => {
      if (requestedSettings.mode === 'advanced') return oldResult.promise;
      paidSimple += 1;
      return response;
    },
    publish: (message) => published.push(message)
  }));
  const oldWait = service.waitForTiming(job, settings);
  const rejected = assert.rejects(oldWait, /mode changed/);
  await flushAsyncWork();
  settings = simpleSettings;
  service.onLifecycleOpportunity();
  await rejected;
  await flushAsyncWork();
  assert.equal(paidSimple, 0);
  assert.deepEqual(getL0TimingAvailability(), { taskId, status: 'unavailable' });
  oldResult.resolve(response);
  await flushAsyncWork();
  assert.deepEqual(published, []);
  await service.waitForTiming(job, settings);
  assert.equal(paidSimple, 1);
  assert.equal(published.length, 1);
});

test('Simple stale-task completion cannot publish timing or keep an explicit action waiting', async () => {
  let current = taskId;
  const pending = deferred<L0TimingResponse>();
  const published: unknown[] = [];
  const service = new L0TimingService(dependencies({
    getSettings: async () => simpleSettings,
    currentTaskId: () => current,
    requestTiming: async () => pending.promise,
    publish: (message) => published.push(message)
  }));
  const waiting = service.waitForTiming(job, simpleSettings);
  const rejected = assert.rejects(waiting, /task changed/);
  await flushAsyncWork();
  current = 'new-task';
  service.onLifecycleOpportunity();
  await rejected;
  pending.resolve(response);
  await flushAsyncWork();
  assert.deepEqual(published, []);
});

test('switching modes away and back cannot revive an earlier Simple in-flight result', async () => {
  let settings: ExtensionSettings = simpleSettings;
  const pending = deferred<L0TimingResponse>();
  const published: unknown[] = [];
  const service = new L0TimingService(dependencies({
    getSettings: async () => settings,
    requestTiming: async () => pending.promise,
    publish: (message) => published.push(message)
  }));
  const waiting = service.waitForTiming(job, settings);
  const rejected = assert.rejects(waiting, /mode changed/);
  await flushAsyncWork();
  settings = { ...DEFAULT_SETTINGS, mode: 'advanced' };
  service.onSettingsChanged(settings);
  settings = simpleSettings;
  service.onSettingsChanged(settings);
  await rejected;
  pending.resolve(response);
  await flushAsyncWork();
  assert.deepEqual(published, []);
});
