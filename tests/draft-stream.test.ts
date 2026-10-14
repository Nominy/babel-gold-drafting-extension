import test from 'node:test';
import assert from 'node:assert/strict';
import { generateDraftStream } from '../src/core/backend-client';
import type { GenerateDraftRequest, GenerateDraftResponse } from '../src/core/types';

const request: GenerateDraftRequest = {
  projectPreset: 'ru-gold-2sp-v1', jobId: 'native-task', draftSessionId: 'same-session',
  openRouterApiKey: 'unit-test-no-egress', rows: [{ rowId: 'a', speakerKey: 'lane-a', startSeconds: 0, endSeconds: 1, text: 'hello', index: 0 }]
};
const response: GenerateDraftResponse = {
  draftRows: [{ rowId: 'a', rewrittenText: 'Hello.', status: 'rewritten', warnings: [] }],
  summary: { totalRows: 1, rewrittenRows: 1, unchangedRows: 0, failedRows: 0, anomalyCounts: {} },
  generationMeta: { model: 'unit-fixture', rulePackVersion: 'test', generatedAt: '2026-10-13T00:00:00Z' }
};

for (const separator of ['\n', '\r\n']) {
  test(`SSE ${JSON.stringify(separator)} boundaries survive split chunks and preserve event order`, async (t) => {
    const previous = globalThis.fetch;
    t.after(() => { globalThis.fetch = previous; });
    const source = [
      `event: started${separator}data: ${JSON.stringify({ jobId: request.jobId, totalRows: 1 })}${separator}${separator}`,
      `event: row${separator}data: ${JSON.stringify({ row: response.draftRows[0], completedRows: 1, totalRows: 1, summary: response.summary })}${separator}${separator}`,
      `event: done${separator}data: ${JSON.stringify(response)}${separator}${separator}`
    ].join('');
    globalThis.fetch = async () => new Response(new ReadableStream({ start(controller) {
      const bytes = new TextEncoder().encode(source);
      for (let index = 0; index < bytes.length; index += 7) controller.enqueue(bytes.slice(index, index + 7));
      controller.close();
    } }));
    const events: string[] = [];
    assert.deepEqual(await generateDraftStream('https://unit.invalid', request, {
      onStarted: () => { events.push('started'); }, onRow: () => { events.push('row'); }, onDone: () => { events.push('done'); }
    }), response);
    assert.deepEqual(events, ['started', 'row', 'done']);
  });
}

test('cancel closes a pending stream without reconciliation or a late draft result', async (t) => {
  const previous = globalThis.fetch;
  t.after(() => { globalThis.fetch = previous; });
  const { promise: started, resolve } = Promise.withResolvers<void>();
  let cancelled = false;
  let calls = 0;
  globalThis.fetch = async () => {
    calls += 1;
    return new Response(new ReadableStream({
      start(controller) { controller.enqueue(new TextEncoder().encode('event: started\ndata: {"jobId":"native-task","totalRows":1}\n\n')); },
      cancel() { cancelled = true; }
    }));
  };
  const controller = new AbortController();
  const pending = generateDraftStream('https://unit.invalid', request, {
    onStarted: () => { resolve(); },
    onDone() { assert.fail('Cancelled draft cannot finish'); },
    onReconnect() { assert.fail('Cancellation cannot launch reconciliation'); }
  }, [], controller.signal);
  await started;
  controller.abort(new DOMException('User cancelled', 'AbortError'));
  await assert.rejects(pending, /User cancelled/);
  assert.equal(cancelled, true);
  assert.equal(calls, 1);
});

test('already cancelled requests never transmit credentials or audio', async (t) => {
  const previous = globalThis.fetch;
  t.after(() => { globalThis.fetch = previous; });
  globalThis.fetch = async () => { assert.fail('Cancelled request attempted network access'); };
  const controller = new AbortController();
  controller.abort(new DOMException('Task changed', 'AbortError'));
  await assert.rejects(generateDraftStream('https://unit.invalid', request, {}, [], controller.signal), /Task changed/);
});

test('truncated SSE reconciles the same session and forwards the caller cancellation signal', async (t) => {
  const previous = globalThis.fetch;
  t.after(() => { globalThis.fetch = previous; });
  const controller = new AbortController();
  const calls: string[] = [];
  globalThis.fetch = async (url, init) => {
    calls.push(String(url));
    assert.equal(init?.signal, controller.signal);
    if (String(url).endsWith('/stream')) return new Response('event: started\ndata: {"totalRows":1}\n\n');
    assert.ok(init?.body instanceof FormData);
    assert.equal(JSON.parse(String(init.body.get('payload'))).draftSessionId, request.draftSessionId);
    return Response.json(response);
  };
  assert.deepEqual(await generateDraftStream('https://unit.invalid', request, {}, [], controller.signal), response);
  assert.deepEqual(calls, ['https://unit.invalid/api/draft/generate/stream', 'https://unit.invalid/api/draft/generate']);
});

test('terminal server errors never reconcile partial rows', async (t) => {
  const previous = globalThis.fetch;
  t.after(() => { globalThis.fetch = previous; });
  let calls = 0;
  globalThis.fetch = async () => { calls += 1; return new Response('event: error\ndata: {"error":"native inference refused"}\n\n'); };
  await assert.rejects(generateDraftStream('https://unit.invalid', request, {}), /native inference refused/);
  assert.equal(calls, 1);
});
