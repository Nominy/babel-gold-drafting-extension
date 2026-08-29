import test from 'node:test';
import assert from 'node:assert/strict';
import {
  isUsableL0TimingJob,
  L0_TIMING_UPDATE_MESSAGE_TYPE,
  L0TimingService,
  type L0TimingServiceDependencies
} from '../src/content/l0-timing-service';
import { getL0TimingAvailability } from '../src/content/l0-timing-availability';
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
    { lane: 'Speaker 1', tokens: [{ id: 'token-1', text: 'one', startSeconds: 0, endSeconds: 0.5 }] },
    { lane: 'Speaker 2', tokens: [] }
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
    captureAudio: async () => tracks,
    getSettings: async () => DEFAULT_SETTINGS,
    requestTiming: async () => response,
    publish: () => undefined,
    activateStatusTask: () => undefined,
    updateStatus: () => undefined,
    clearStatus: () => undefined,
    now: () => 1_000,
    schedule: () => undefined,
    ...overrides
  };
}

test('usable timing jobs accept transcripts with one or more speaker lanes', () => {
  assert.equal(isUsableL0TimingJob(job), true);
  assert.equal(isUsableL0TimingJob({ jobId: 'task-42', rows: [] }), false);
  assert.equal(isUsableL0TimingJob({ ...job, rows: [job.rows[0]] }), true);
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
    tracks: response.tracks
  }]);
  service.onLifecycleOpportunity();
  await flushAsyncWork();
  assert.equal(requestCount, 1);
});

test('timing lifecycle forwards queue states and clears status before publishing success', async () => {
  const events: string[] = [];
  const service = new L0TimingService(dependencies({
    activateStatusTask: () => events.push('activate'),
    updateStatus: (_taskId, status) => events.push(status.status),
    clearStatus: () => events.push('clear'),
    requestTiming: async (_settings, _job, _tracks, callbacks) => {
      callbacks.onQueueStatus?.({
        requestId: 'request-1',
        status: 'queued',
        position: 2,
        queuedCount: 2
      });
      callbacks.onQueueStatus?.({
        requestId: 'request-1',
        status: 'running',
        position: 0,
        queuedCount: 1
      });
      return response;
    },
    publish: () => events.push('publish')
  }));

  service.onLifecycleOpportunity();
  await flushAsyncWork();
  assert.deepEqual(events, ['activate', 'queued', 'running', 'clear', 'publish']);
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
  const statuses: string[] = [];
  const service = new L0TimingService(dependencies({
    now: () => now,
    requestTiming: async () => {
      requestCount += 1;
      if (requestCount === 1) throw new Error('background ASR unavailable');
      return response;
    },
    schedule: (callback) => scheduled.push(callback),
    updateStatus: (_taskId, status) => statuses.push(status.status),
  }));

  assert.doesNotThrow(() => service.onLifecycleOpportunity());
  await flushAsyncWork();
  assert.equal(requestCount, 1);
  assert.equal(scheduled.length, 1);
  assert.deepEqual(statuses, ['retrying']);
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
