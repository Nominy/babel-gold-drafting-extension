import test from 'node:test';
import assert from 'node:assert/strict';
import { generateL0Timing, getL0QueueEndpoint, getL0TimingEndpoint, parseL0TimingResponse } from '../src/core/l0-timing-client';
import { DEFAULT_SETTINGS } from '../src/core/settings';
import type { CapturedAudioTrack, TranscriptJob } from '../src/core/types';

const job: TranscriptJob = {
  jobId: 'task-42',
  rows: [
    { rowId: 'row-1', speakerKey: 'speaker-1', processedRecordingId: 'speaker-1', startSeconds: 0, endSeconds: 1, text: 'one', index: 0 },
    { rowId: 'row-2', speakerKey: 'speaker-2', processedRecordingId: 'speaker-2', startSeconds: 1, endSeconds: 2, text: 'two', index: 1 }
  ]
};
const singleLaneJob: TranscriptJob = {
  ...job,
  rows: [job.rows[0]]
};
const canonicalTaskId = '{"version":1,"baseTaskId":"task-42","stableLaneIds":["speaker-1"]}';

function wavBlob(): Blob {
  return new Blob([new Uint8Array([82, 73, 70, 70, 4, 0, 0, 0, 87, 65, 86, 69])], { type: 'audio/wav' });
}

const audioTracks: CapturedAudioTrack[] = [
  { trackId: 'first', speakerKey: 'speaker-1', source: 'first.wav', blob: wavBlob(), mimeType: 'audio/wav' },
  { trackId: 'second', speakerKey: 'speaker-2', source: 'second.wav', blob: wavBlob(), mimeType: 'audio/wav' }
];

const timingResponse = {
  taskId: canonicalTaskId,
  tracks: [
    {
      lane: 'speaker-1',
      tokens: [{ id: 'word-1', text: 'One', startSeconds: 0, endSeconds: 0.4 }]
    },
    { lane: 'speaker-2', tokens: [] }
  ],
  summary: { tokenCount: 1 },
  models: { asr: 'whisper' }
};

test('L0 timing routing uses the configured L0 base and transcribe path', () => {
  assert.equal(
    getL0TimingEndpoint({ ...DEFAULT_SETTINGS, l0CustomBaseUrl: 'https://engine.example.test/root///' }),
    'https://engine.example.test/root/v1/transcribe'
  );
});

test('queue routing encodes request identities', () => {
  assert.equal(
    getL0QueueEndpoint(
      { ...DEFAULT_SETTINGS, l0CustomBaseUrl: 'https://engine.test/root/' },
      'request/id with spaces'
    ),
    'https://engine.test/root/v1/queue/request%2Fid%20with%20spaces'
  );
});

test('generateL0Timing derives exactly two WAV lanes when the transcript currently contains only one lane', async () => {
  let requestUrl = '';
  let requestInit: RequestInit | undefined;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
    if (init?.method === 'POST') {
      requestUrl = String(url);
      requestInit = init;
      return new Response(JSON.stringify(timingResponse), { status: 200 });
    }
    return new Response(null, { status: 404 });
  }) as typeof fetch;

  try {
    assert.deepEqual(
      await generateL0Timing(
        { ...DEFAULT_SETTINGS, l0CustomBaseUrl: 'https://engine.test/' },
        singleLaneJob,
        audioTracks
      ),
      timingResponse
    );
    assert.equal(requestUrl, 'https://engine.test/v1/transcribe');
    assert.match(String(new Headers(requestInit?.headers).get('X-Babel-Request-Id')), /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i);
    assert.ok(requestInit?.body instanceof FormData);
    const form = requestInit.body;
    assert.deepEqual(Array.from(form.keys()).sort(), ['audio:1', 'audio:2', 'payload']);
    assert.deepEqual(JSON.parse(String(form.get('payload'))), {
      taskId: canonicalTaskId,
      tracks: [
        { lane: 'speaker-1', fieldName: 'audio:1' },
        { lane: 'speaker-2', fieldName: 'audio:2' }
      ]
    });
    assert.ok(form.get('audio:1') instanceof File);
    assert.ok(form.get('audio:2') instanceof File);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('timing request reports changing queue state and stops polling after POST settles', async () => {
  const statuses: Array<Record<string, unknown>> = [];
  const originalFetch = globalThis.fetch;
  const postGate = Promise.withResolvers<Response>();
  const queuedGate = Promise.withResolvers<void>();
  const runningGate = Promise.withResolvers<void>();
  const pollWaiters: Array<() => void> = [];
  let requestId = '';
  let pollCount = 0;
  globalThis.fetch = (async (_url: string | URL | Request, init?: RequestInit) => {
    if (init?.method === 'POST') {
      requestId = String(new Headers(init.headers).get('X-Babel-Request-Id'));
      return await postGate.promise;
    }
    pollCount += 1;
    const status = pollCount === 1 ? 'queued' : 'running';
    return new Response(
      JSON.stringify({
        requestId,
        status,
        position: status === 'queued' ? 3 : 0,
        queuedCount: status === 'queued' ? 3 : 2
      }),
      { status: 200 }
    );
  }) as typeof fetch;

  try {
    const pending = generateL0Timing(
      { ...DEFAULT_SETTINGS, l0CustomBaseUrl: 'https://engine.test' },
      singleLaneJob,
      audioTracks,
      {
        onQueueStatus: (status) => {
          statuses.push(status);
          if (status.status === 'queued') queuedGate.resolve();
          if (status.status === 'running') runningGate.resolve();
        },
        waitForPoll: (signal) => {
          const gate = Promise.withResolvers<void>();
          const resolve = () => gate.resolve();
          pollWaiters.push(resolve);
          signal.addEventListener('abort', resolve, { once: true });
          return gate.promise;
        }
      }
    );
    await queuedGate.promise;
    assert.equal(pollWaiters.length, 1);
    pollWaiters.shift()?.();
    await runningGate.promise;
    assert.deepEqual(
      statuses.map(({ status, position }) => ({ status, position })),
      [
        { status: 'queued', position: 3 },
        { status: 'running', position: 0 }
      ]
    );
    postGate.resolve(new Response(JSON.stringify(timingResponse), { status: 200 }));
    await pending;
    const settledPollCount = pollCount;
    await Promise.resolve();
    assert.equal(pollCount, settledPollCount);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('timing response parsing accepts silent lanes and rejects invalid or mismatched absolute timestamps', () => {
  assert.deepEqual(parseL0TimingResponse(timingResponse, canonicalTaskId), timingResponse);
  assert.deepEqual(
    parseL0TimingResponse(
      {
        ...timingResponse,
        tracks: timingResponse.tracks.map((track) => ({ ...track, tokens: [] }))
      },
      canonicalTaskId
    ).tracks.map((track) => track.tokens),
    [[], []]
  );
  assert.throws(
    () =>
      parseL0TimingResponse(
        {
          ...timingResponse,
          tracks: [
            {
              lane: 'speaker-1',
              tokens: [{ id: 'bad', text: 'bad', startSeconds: -0.1, endSeconds: 0.2 }]
            },
            timingResponse.tracks[1]
          ]
        },
        'task-42'
      ),
    /invalid timing response/
  );
  assert.throws(() => parseL0TimingResponse(timingResponse, 'new-task'), /invalid timing response/);
});
