import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import {
  L0_REPLACE_READY_REQUEST_TYPE,
  L0_REPLACE_READY_RESPONSE_TYPE,
  L0_REPLACE_REQUEST_TYPE,
  L0_REPLACE_RESPONSE_TYPE,
  replaceTranscriptWithL0Rows,
  requireL0ReplacementConsumer
} from '../src/core/l0-replacement-bridge';
import type { L0DraftRow, TranscriptJob } from '../src/core/types';

const rows: L0DraftRow[] = [
  { id: 'l0-1', lane: 'speaker-1', startSeconds: 0, endSeconds: 1.5, text: 'Первый.' },
  { id: 'l0-2', lane: 'speaker-2', startSeconds: 1.5, endSeconds: 3, text: 'Второй.' }
];
const expected: TranscriptJob = { jobId: 'review-action-42', taskScoped: true, rows: [] };

function installBridgeDom() {
  const dom = new JSDOM('', { url: 'https://dashboard.babel.audio/transcription/job' });
  Object.assign(globalThis, { window: dom.window });
  return dom;
}

test('replacement bridge posts versioned L0 rows and resolves created annotation mappings', async () => {
  const dom = installBridgeDom();
  let request: Record<string, unknown> | null = null;
  const originalPostMessage = dom.window.postMessage.bind(dom.window);
  dom.window.postMessage = ((message: unknown) => {
    request = message as Record<string, unknown>;
    dom.window.dispatchEvent(
      new dom.window.MessageEvent('message', {
        data: {
          type: L0_REPLACE_RESPONSE_TYPE,
          version: 1,
          requestId: request.requestId,
          ok: true,
          created: [
            { id: 'l0-1', annotationId: 'ann-10', lane: 'speaker-1', startSeconds: 0, endSeconds: 1.5 },
            { id: 'l0-2', annotationId: 'ann-11', lane: 'speaker-2', startSeconds: 1.5, endSeconds: 3 }
          ]
        }
      })
    );
  }) as typeof dom.window.postMessage;

  try {
    const created = await replaceTranscriptWithL0Rows(rows, expected);
    assert.ok(request);
    const postedRequest = request as Record<string, unknown>;
    assert.equal(postedRequest.type, L0_REPLACE_REQUEST_TYPE);
    assert.equal(postedRequest.version, 1);
    assert.deepEqual(postedRequest.rows, rows);
    assert.equal(postedRequest.expectedReviewActionId, expected.jobId);
    assert.deepEqual(postedRequest.expectedRows, expected.rows);
    assert.deepEqual(created.map(({ id, annotationId }) => ({ id, annotationId })), [
      { id: 'l0-1', annotationId: 'ann-10' },
      { id: 'l0-2', annotationId: 'ann-11' }
    ]);
  } finally {
    dom.window.postMessage = originalPostMessage;
    dom.window.close();
  }
});

test('replacement bridge surfaces Helper failure without fallback', async () => {
  const dom = installBridgeDom();
  dom.window.postMessage = ((message: unknown) => {
    const request = message as Record<string, unknown>;
    dom.window.dispatchEvent(
      new dom.window.MessageEvent('message', {
        data: {
          type: L0_REPLACE_RESPONSE_TYPE,
          version: 1,
          requestId: request.requestId,
          ok: false,
          reason: 'mutation-failed',
          message: 'Could not create segment'
        }
      })
    );
  }) as typeof dom.window.postMessage;
  try {
    await assert.rejects(replaceTranscriptWithL0Rows(rows, expected), /mutation-failed.*Could not create segment/);
  } finally {
    dom.window.close();
  }
});

test('ready ping resolves confirmed when a Helper that speaks the protocol answers', async () => {
  const dom = installBridgeDom();
  dom.window.postMessage = ((message: unknown) => {
    const request = message as Record<string, unknown>;
    assert.equal(request.type, L0_REPLACE_READY_REQUEST_TYPE);
    dom.window.dispatchEvent(
      new dom.window.MessageEvent('message', {
        data: { type: L0_REPLACE_READY_RESPONSE_TYPE, version: 1, requestId: request.requestId }
      })
    );
  }) as typeof dom.window.postMessage;
  try {
    assert.equal(await requireL0ReplacementConsumer(), true);
  } finally {
    dom.window.close();
  }
});

test('ready ping timeout falls back to the direct request path instead of aborting', async () => {
  const dom = installBridgeDom();
  const timers: Array<() => void> = [];
  dom.window.setTimeout = ((callback: () => void) => {
    timers.push(callback);
    return timers.length;
  }) as typeof dom.window.setTimeout;
  const warnings: string[] = [];
  const originalWarn = console.warn;
  console.warn = (...args: unknown[]) => warnings.push(args.map(String).join(' '));
  // An old Helper never answers the ping; it only serves the replacement request.
  dom.window.postMessage = ((message: unknown) => {
    const request = message as Record<string, unknown>;
    if (request.type !== L0_REPLACE_REQUEST_TYPE) return;
    dom.window.dispatchEvent(
      new dom.window.MessageEvent('message', {
        data: {
          type: L0_REPLACE_RESPONSE_TYPE,
          version: 1,
          requestId: request.requestId,
          ok: true,
          created: rows.map((row, index) => ({ ...row, annotationId: `ann-${index}` }))
        }
      })
    );
  }) as typeof dom.window.postMessage;
  try {
    const ready = requireL0ReplacementConsumer();
    assert.equal(timers.length, 1);
    timers[0]!();
    assert.equal(await ready, false);
    assert.match(warnings.join('\n'), /did not confirm L0 replacement readiness/);
    const created = await replaceTranscriptWithL0Rows(rows, expected);
    assert.deepEqual(created.map((mapping) => mapping.annotationId), ['ann-0', 'ann-1']);
  } finally {
    console.warn = originalWarn;
    dom.window.close();
  }
});

for (const boundary of ['duplicate-id', 'duplicate-annotation', 'wrong-lane', 'changed-time', 'nonfinite-time'] as const) {
  test(`replacement rejects ${boundary} mappings even when the row count matches`, async (t) => {
    const dom = installBridgeDom();
    t.after(() => dom.window.close());
    dom.window.postMessage = ((message: { requestId: string }) => {
      const created = rows.map((row, index) => ({ ...row, annotationId: `annotation-${index}` }));
      if (boundary === 'duplicate-id') created[1].id = created[0].id;
      if (boundary === 'duplicate-annotation') created[1].annotationId = created[0].annotationId;
      if (boundary === 'wrong-lane') created[1].lane = created[0].lane;
      if (boundary === 'changed-time') created[1].startSeconds += 0.5;
      if (boundary === 'nonfinite-time') created[1].endSeconds = Number.NaN;
      dom.window.dispatchEvent(new dom.window.MessageEvent('message', {
        data: { type: L0_REPLACE_RESPONSE_TYPE, version: 1, requestId: message.requestId, ok: true, created }
      }));
    }) as typeof dom.window.postMessage;
    await assert.rejects(replaceTranscriptWithL0Rows(rows, expected), /invalid annotation identities or timings/);
  });
}

test('replacement refuses read-only and unscoped captures without posting a destructive request', (t) => {
  const dom = installBridgeDom();
  t.after(() => dom.window.close());
  dom.window.postMessage = (() => { assert.fail('No replacement request should be posted'); }) as typeof dom.window.postMessage;
  assert.throws(() => replaceTranscriptWithL0Rows(rows, { ...expected, readOnly: true }), /editable native task/);
  assert.throws(() => replaceTranscriptWithL0Rows(rows, { ...expected, taskScoped: false }), /editable native task/);
});

test('cancelled readiness cannot warn late or confirm the next draft request', async (t) => {
  const dom = installBridgeDom();
  t.after(() => dom.window.close());
  const timers: Array<() => void> = [];
  const requests: Array<{ requestId: string }> = [];
  dom.window.setTimeout = ((callback: () => void) => {
    timers.push(callback);
    return timers.length;
  }) as typeof dom.window.setTimeout;
  dom.window.postMessage = ((message: { requestId: string }) => { requests.push(message); }) as typeof dom.window.postMessage;
  const warnings: unknown[] = [];
  t.mock.method(console, 'warn', (...args: unknown[]) => { warnings.push(args); });
  const controller = new AbortController();
  const cancelled = requireL0ReplacementConsumer(controller.signal);
  const rejection = assert.rejects(cancelled, /Old draft cancelled/);
  controller.abort(new Error('Old draft cancelled'));
  await rejection;
  let currentResolved = false;
  const current = requireL0ReplacementConsumer().then((ready) => { currentResolved = true; return ready; });
  timers[0]!();
  dom.window.dispatchEvent(new dom.window.MessageEvent('message', {
    data: { type: L0_REPLACE_READY_RESPONSE_TYPE, version: 1, requestId: requests[0].requestId }
  }));
  await Promise.resolve();
  assert.equal(currentResolved, false);
  assert.deepEqual(warnings, []);
  dom.window.dispatchEvent(new dom.window.MessageEvent('message', {
    data: { type: L0_REPLACE_READY_RESPONSE_TYPE, version: 1, requestId: requests[1].requestId }
  }));
  assert.equal(await current, true);
});

test('already cancelled drafts never send a readiness request', async (t) => {
  const dom = installBridgeDom();
  t.after(() => dom.window.close());
  dom.window.postMessage = (() => { assert.fail('A cancelled draft cannot request Helper readiness'); }) as typeof dom.window.postMessage;
  const controller = new AbortController();
  controller.abort(new Error('Draft already cancelled'));
  await assert.rejects(requireL0ReplacementConsumer(controller.signal), /already cancelled/);
});
