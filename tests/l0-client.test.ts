import test from 'node:test';
import assert from 'node:assert/strict';
import { generateL0Draft, getL0DraftEndpoint } from '../src/core/l0-client';
import { DEFAULT_SETTINGS } from '../src/core/settings';
import type { CapturedAudioTrack, TranscriptJob } from '../src/core/types';

const job: TranscriptJob = {
  jobId: 'task-42',
  rows: [
    { rowId: 'row-1', speakerKey: 'speaker-1', startSeconds: 0, endSeconds: 1, text: 'one', index: 0 },
    { rowId: 'row-2', speakerKey: 'speaker-2', startSeconds: 1, endSeconds: 2, text: 'two', index: 1 }
  ]
};

function wavBlob(): Blob {
  return new Blob([new Uint8Array([82, 73, 70, 70, 4, 0, 0, 0, 87, 65, 86, 69])], { type: 'audio/wav' });
}

const tracks: CapturedAudioTrack[] = [
  { trackId: 'first', speakerKey: 'speaker-1', source: 'first.wav', blob: wavBlob(), mimeType: 'audio/wav' },
  { trackId: 'second', speakerKey: 'speaker-2', source: 'second.wav', blob: wavBlob(), mimeType: 'audio/wav' },
  { trackId: 'ignored', speakerKey: 'other', source: 'ignored.wav', blob: wavBlob(), mimeType: 'audio/wav' }
];

const canonicalResponse = {
  rows: [
    { id: 'row-1', lane: 'speaker-1', startSeconds: 0, endSeconds: 1, text: 'One.' },
    { id: 'row-2', lane: 'speaker-2', startSeconds: 1, endSeconds: 2, text: 'two' }
  ],
  summary: { rowCount: 2 },
  models: { asr: 'qwen', formatter: 'punctuation' }
};

test('L0 routing normalizes the configured self-host base and only uses v1/draft', () => {
  assert.equal(getL0DraftEndpoint(DEFAULT_SETTINGS), 'http://127.0.0.1:8767/v1/draft');
  assert.equal(
    getL0DraftEndpoint({ ...DEFAULT_SETTINGS, l0CustomBaseUrl: 'https://engine.example.test/root///' }),
    'https://engine.example.test/root/v1/draft'
  );
});

test('generateL0Draft sends unsegmented canonical payload and exactly two WAV parts', async () => {
  let requestUrl = '';
  let requestInit: RequestInit | undefined;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
    requestUrl = String(url);
    requestInit = init;
    return new Response(JSON.stringify(canonicalResponse), { status: 200 });
  }) as typeof fetch;

  try {
    const response = await generateL0Draft(
      { ...DEFAULT_SETTINGS, l0CustomBaseUrl: 'https://engine.test/' },
      job,
      tracks
    );
    assert.deepEqual(response, canonicalResponse);
    assert.equal(requestUrl, 'https://engine.test/v1/draft');
    assert.ok(requestInit?.body instanceof FormData);
    const form = requestInit.body;
    assert.deepEqual(Array.from(form.keys()).sort(), ['audio:1', 'audio:2', 'payload']);
    assert.deepEqual(JSON.parse(String(form.get('payload'))), {
      taskId: 'task-42',
      tracks: [
        { lane: 'speaker-1', fieldName: 'audio:1' },
        { lane: 'speaker-2', fieldName: 'audio:2' }
      ]
    });
    assert.ok(form.get('audio:1') instanceof File);
    assert.ok(form.get('audio:2') instanceof File);
    const headers = requestInit.headers as Record<string, string>;
    assert.equal(headers.Authorization, undefined);
    assert.equal(headers['X-Babel-Local-Engine'], '1');
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('L0 client surfaces HTTP detail and rejects non-WAV input before sending', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () => new Response(JSON.stringify({ detail: 'engine is busy' }), { status: 429 })) as typeof fetch;
  try {
    await assert.rejects(generateL0Draft(DEFAULT_SETTINGS, job, tracks), /L0 drafting failed: engine is busy/);
    await assert.rejects(
      generateL0Draft(DEFAULT_SETTINGS, job, [
        { ...tracks[0], blob: new Blob(['not wav'], { type: 'audio/wav' }) },
        tracks[1]
      ]),
      /not a WAV file/
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

