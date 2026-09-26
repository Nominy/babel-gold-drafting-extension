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
import type { CapturedAudioTrack, L0TimingResponse, TranscriptJob } from '../src/core/types';

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
    getSettings: async () => DEFAULT_SETTINGS,
    lookupTiming: async () => null,
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

test('timing lifecycle silently ignores transcript capture failures', () => {
  const service = new L0TimingService(dependencies({
    captureTranscript: () => {
      throw new Error('transcript is not mounted yet');
    }
  }));
  assert.doesNotThrow(() => service.onLifecycleOpportunity());
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
      requestTiming: async () => pending.promise
    }));
    publishL0TimingAvailability({ taskId, status: 'preparing' });
    const wait = service.waitForTiming(job, { ...DEFAULT_SETTINGS, localModelsEnabled: true })
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

test('timing wait cleans up an immediately replayed success and a terminal failure', async () => {
  let currentChecks = 0;
  const service = new L0TimingService(dependencies({
    currentTaskId: () => { currentChecks += 1; return taskId; }
  }));
  publishL0TimingAvailability({ taskId, status: 'available' });
  await service.waitForTiming(job, DEFAULT_SETTINGS);
  const afterSuccess = currentChecks;
  publishL0TimingAvailability({ taskId, status: 'running' });
  assert.equal(currentChecks, afterSuccess);
  publishL0TimingAvailability({ taskId, status: 'unavailable' });
  await assert.rejects(service.waitForTiming(job, DEFAULT_SETTINGS), /unavailable/);
  const afterFailure = currentChecks;
  publishL0TimingAvailability({ taskId, status: 'available' });
  assert.equal(currentChecks, afterFailure);
});
