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
import type { L0DraftRow } from '../src/core/types';

const rows: L0DraftRow[] = [
  { id: 'l0-1', lane: 'speaker-1', startSeconds: 0, endSeconds: 1.5, text: 'Первый.' },
  { id: 'l0-2', lane: 'speaker-2', startSeconds: 1.5, endSeconds: 3, text: 'Второй.' }
];

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
    const created = await replaceTranscriptWithL0Rows(rows);
    assert.ok(request);
    const postedRequest = request as Record<string, unknown>;
    assert.equal(postedRequest.type, L0_REPLACE_REQUEST_TYPE);
    assert.equal(postedRequest.version, 1);
    assert.deepEqual(postedRequest.rows, rows);
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
    await assert.rejects(replaceTranscriptWithL0Rows(rows), /mutation-failed.*Could not create segment/);
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
    const created = await replaceTranscriptWithL0Rows(rows);
    assert.deepEqual(created.map((mapping) => mapping.annotationId), ['ann-0', 'ann-1']);
  } finally {
    console.warn = originalWarn;
    dom.window.close();
  }
});
