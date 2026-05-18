import test from 'node:test';
import assert from 'node:assert/strict';
import { generateDraftStream } from '../src/core/backend-client';
import type { CapturedAudioTrack, GenerateDraftRequest } from '../src/core/types';

const request: GenerateDraftRequest = {
  projectPreset: 'ru-gold-2sp-v1',
  jobId: 'job-1',
  rows: [
    {
      rowId: 'r1',
      speakerKey: 'speaker-1',
      startSeconds: 0,
      endSeconds: 1,
      text: 'Привет',
      index: 0
    }
  ],
  openRouterApiKey: 'sk-or-test',
  model: 'google/gemini-3-flash-preview',
  serviceTier: 'flex'
};

test('generateDraftStream attaches audio tracks to the default generate stream route', async () => {
  let seenUrl = '';
  let seenBody: unknown = null;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
    seenUrl = String(url);
    seenBody = init?.body;
    return new Response(
      `event: done\ndata: ${JSON.stringify({
        draftRows: [],
        summary: { totalRows: 0, rewrittenRows: 0, unchangedRows: 0, failedRows: 0, anomalyCounts: {} },
        generationMeta: { model: 'test', rulePackVersion: 'test', generatedAt: '2026-05-17T00:00:00.000Z' }
      })}\n\n`,
      { status: 200, headers: { 'Content-Type': 'text/event-stream' } }
    );
  }) as unknown as typeof fetch;

  try {
    const audioTracks: CapturedAudioTrack[] = [
      {
        trackId: 'audio-1',
        speakerKey: 'speaker-1',
        trackLabel: 'Speaker 1',
        source: 'https://dashboard.babel.audio/audio.webm',
        blob: new Blob([new Uint8Array([1, 2, 3])], { type: 'audio/webm' }),
        mimeType: 'audio/webm'
      }
    ];

    await generateDraftStream('http://127.0.0.1:3001', request, {}, audioTracks);

    assert.equal(seenUrl, 'http://127.0.0.1:3001/api/draft/generate/stream');
    assert.ok(seenBody instanceof FormData);
    const form = seenBody;
    assert.equal(typeof form.get('payload'), 'string');
    assert.ok(form.get('audioTrack:audio-1') instanceof File);
    assert.deepEqual(JSON.parse(String(form.get('audioTrackMeta:audio-1'))), {
      source: 'https://dashboard.babel.audio/audio.webm',
      speakerKey: 'speaker-1',
      trackLabel: 'Speaker 1',
      mimeType: 'audio/webm'
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('generateDraftStream sends plain JSON when no audio tracks are provided', async () => {
  let seenBody: unknown = null;
  let seenContentType = '';
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (_url: string | URL | Request, init?: RequestInit) => {
    seenBody = init?.body;
    seenContentType = String((init?.headers as Record<string, string> | undefined)?.['Content-Type'] || '');
    return new Response(
      `event: done\ndata: ${JSON.stringify({
        draftRows: [],
        summary: { totalRows: 0, rewrittenRows: 0, unchangedRows: 0, failedRows: 0, anomalyCounts: {} },
        generationMeta: { model: 'test', rulePackVersion: 'test', generatedAt: '2026-05-17T00:00:00.000Z' }
      })}\n\n`,
      { status: 200, headers: { 'Content-Type': 'text/event-stream' } }
    );
  }) as unknown as typeof fetch;

  try {
    await generateDraftStream('http://127.0.0.1:3001', request, {});

    assert.equal(typeof seenBody, 'string');
    assert.equal(seenContentType, 'application/json');
    assert.deepEqual(JSON.parse(seenBody as string), request);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('generateDraftStream retains audio tracks when reconciling after the stream connection drops', async () => {
  const calls: Array<{ url: string; body: unknown; contentType: string }> = [];
  const encoder = new TextEncoder();
  const originalFetch = globalThis.fetch;
  const sessionRequest = {
    ...request,
    draftSessionId: 'draft-session-1'
  } as GenerateDraftRequest & { draftSessionId: string };
  const finalResponse = {
    draftRows: [
      {
        rowId: 'r1',
        rewrittenText: 'Привет.',
        status: 'rewritten',
        warnings: []
      }
    ],
    summary: { totalRows: 1, rewrittenRows: 1, unchangedRows: 0, failedRows: 0, anomalyCounts: {} },
    generationMeta: { model: 'test', rulePackVersion: 'test', generatedAt: '2026-05-17T00:00:00.000Z' }
  };

  globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
    const contentType = String((init?.headers as Record<string, string> | undefined)?.['Content-Type'] || '');
    calls.push({ url: String(url), body: init?.body, contentType });

    if (String(url).endsWith('/api/draft/generate/stream')) {
      return new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(encoder.encode('event: started\ndata: {"jobId":"job-1","totalRows":1}\n\n'));
            controller.error(new Error('network lost'));
          }
        }),
        { status: 200, headers: { 'Content-Type': 'text/event-stream' } }
      );
    }

    return new Response(JSON.stringify(finalResponse), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    });
  }) as unknown as typeof fetch;

  try {
    const audioTracks: CapturedAudioTrack[] = [
      {
        trackId: 'audio-1',
        speakerKey: 'speaker-1',
        trackLabel: 'Speaker 1',
        source: 'https://dashboard.babel.audio/audio.webm',
        blob: new Blob([new Uint8Array([1, 2, 3])], { type: 'audio/webm' }),
        mimeType: 'audio/webm'
      }
    ];

    const reconciled = await generateDraftStream('http://127.0.0.1:3001', sessionRequest, {}, audioTracks);

    assert.deepEqual(reconciled, finalResponse);
    assert.equal(calls.length, 2);
    assert.equal(calls[0].url, 'http://127.0.0.1:3001/api/draft/generate/stream');
    assert.ok(calls[0].body instanceof FormData);
    assert.equal(calls[1].url, 'http://127.0.0.1:3001/api/draft/generate');
    assert.ok(calls[1].body instanceof FormData);
    assert.equal(calls[1].contentType, '');
    assert.deepEqual(JSON.parse(String(calls[1].body.get('payload'))), sessionRequest);
    assert.ok(calls[1].body.get('audioTrack:audio-1') instanceof File);
    assert.deepEqual(JSON.parse(String(calls[1].body.get('audioTrackMeta:audio-1'))), {
      source: 'https://dashboard.babel.audio/audio.webm',
      speakerKey: 'speaker-1',
      trackLabel: 'Speaker 1',
      mimeType: 'audio/webm'
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('generateDraftStream retries form reconcile when the first reconnect request also loses network', async () => {
  const calls: Array<{ url: string; body: unknown }> = [];
  const encoder = new TextEncoder();
  const originalFetch = globalThis.fetch;
  const sessionRequest = {
    ...request,
    draftSessionId: 'draft-session-retry'
  } as GenerateDraftRequest & { draftSessionId: string };
  const finalResponse = {
    draftRows: [
      {
        rowId: 'r1',
        rewrittenText: 'Привет.',
        status: 'rewritten',
        warnings: []
      }
    ],
    summary: { totalRows: 1, rewrittenRows: 1, unchangedRows: 0, failedRows: 0, anomalyCounts: {} },
    generationMeta: { model: 'test', rulePackVersion: 'test', generatedAt: '2026-05-17T00:00:00.000Z' }
  };

  globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
    calls.push({ url: String(url), body: init?.body });

    if (String(url).endsWith('/api/draft/generate/stream')) {
      return new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(encoder.encode('event: started\ndata: {"jobId":"job-1","totalRows":1}\n\n'));
            controller.error(new Error('network lost'));
          }
        }),
        { status: 200, headers: { 'Content-Type': 'text/event-stream' } }
      );
    }

    if (String(url).endsWith('/api/draft/generate') && calls.filter((call) => call.url.endsWith('/api/draft/generate')).length === 1) {
      throw new TypeError('Failed to fetch');
    }

    return new Response(JSON.stringify(finalResponse), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    });
  }) as unknown as typeof fetch;

  try {
    const reconciled = await generateDraftStream('http://127.0.0.1:3001', sessionRequest, {});
    const generateCalls = calls.filter((call) => call.url.endsWith('/api/draft/generate'));

    assert.deepEqual(reconciled, finalResponse);
    assert.equal(generateCalls.length, 2);
    assert.ok(generateCalls[1].body instanceof FormData);
    assert.deepEqual(JSON.parse(String(generateCalls[1].body.get('payload'))), sessionRequest);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
