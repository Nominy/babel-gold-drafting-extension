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
  model: 'google/gemini-3-flash-preview'
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
