import test from 'node:test';
import assert from 'node:assert/strict';
import { generateL0Draft, generateL0SegmentDraft, getL0DraftEndpoint } from '../src/core/l0-client';
import { DEFAULT_SETTINGS } from '../src/core/settings';
import { buildCanonicalTaskIdentity } from '../src/core/transcript';
import type { TranscriptJob } from '../src/core/types';

const job: TranscriptJob = {
  jobId: 'task-42',
  rows: [
    { rowId: 'row-1', speakerKey: 'speaker-1', startSeconds: 0, endSeconds: 1, text: 'one', index: 0 },
    { rowId: 'row-2', speakerKey: 'speaker-2', startSeconds: 1, endSeconds: 2, text: 'two', index: 1 }
  ]
};


const canonicalResponse = {
  rows: [
    { id: 'row-1', lane: 'speaker-1', startSeconds: 0, endSeconds: 1, text: 'One.' },
    { id: 'row-2', lane: 'speaker-2', startSeconds: 1, endSeconds: 2, text: 'two' }
  ],
  summary: { rowCount: 2 },
  models: { asr: 'qwen', formatter: 'punctuation' }
};


test('L0 routing uses the hosted default and normalizes custom self-host bases', () => {
  assert.equal(
    getL0DraftEndpoint(DEFAULT_SETTINGS),
    'https://reviewgen.ovh/a3f73d6cf25fa138be653daaf2d7cd0702c0b2d69c40fb9eaee4e07d4b067dd5/v1/draft'
  );
  assert.equal(
    getL0DraftEndpoint({ ...DEFAULT_SETTINGS, l0CustomBaseUrl: 'https://engine.example.test/root///' }),
    'https://engine.example.test/root/v1/draft'
  );
  assert.equal(
    getL0DraftEndpoint({ ...DEFAULT_SETTINGS, l0CustomBaseUrl: 'not a URL' }),
    'https://reviewgen.ovh/a3f73d6cf25fa138be653daaf2d7cd0702c0b2d69c40fb9eaee4e07d4b067dd5/v1/draft'
  );
});

test('generateL0Draft requests punctuation using canonical task identity without audio', async () => {
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
      job
    );
    assert.deepEqual(response, canonicalResponse);
    assert.equal(requestUrl, 'https://engine.test/v1/draft');
    assert.equal(requestInit?.method, 'POST');
    assert.deepEqual(JSON.parse(String(requestInit?.body)), { taskId: buildCanonicalTaskIdentity(job) });
    assert.equal(new Headers(requestInit?.headers).get('Content-Type'), 'application/json');
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('free L0 segment drafting sends one empty preserved row through the draft endpoint', async () => {
  let requestInit: RequestInit | undefined;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (_url: string | URL | Request, init?: RequestInit) => {
    requestInit = init;
    return new Response(JSON.stringify({
      rows: [
        {
          id: 'empty-row',
          lane: 'speaker-1',
          startSeconds: 4,
          endSeconds: 7,
          text: 'Пунктуированный текст.'
        }
      ],
      summary: { rowCount: 1 },
      models: { asr: 'gigaam', l2: 'punctuation' }
    }), { status: 200 });
  }) as typeof fetch;

  try {
    const text = await generateL0SegmentDraft(
      { ...DEFAULT_SETTINGS, l0CustomBaseUrl: 'https://engine.test' },
      'canonical-task',
      {
        rowId: 'empty-row',
        speakerKey: 'speaker-1',
        startSeconds: 4,
        endSeconds: 7,
        text: '',
        index: 0
      }
    );

    assert.equal(text, 'Пунктуированный текст.');
    assert.deepEqual(JSON.parse(String(requestInit?.body)), {
      taskId: 'canonical-task',
      options: {
        preserveRows: [{
          rowId: 'empty-row',
          speakerKey: 'speaker-1',
          startSeconds: 4,
          endSeconds: 7,
          text: '',
          index: 0
        }]
      }
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('L0 client surfaces HTTP detail for failed punctuation', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () => new Response(JSON.stringify({ detail: 'engine is busy' }), { status: 429 })) as typeof fetch;
  try {
    await assert.rejects(generateL0Draft(DEFAULT_SETTINGS, job), /L0 drafting failed: engine is busy/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

