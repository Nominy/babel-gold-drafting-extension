import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { registerLifecycle } from '../src/core/lifecycle';
import { applyDraftRows, buildDiffPreviewItems, captureTranscriptJob, restoreCapturedRows } from '../src/core/transcript';

function installDom(html: string) {
  const dom = new JSDOM(html, { url: 'https://dashboard.babel.audio/transcription/RU-transcription?jobId=job-42' });
  Object.assign(globalThis, {
    window: dom.window,
    document: dom.window.document,
    HTMLElement: dom.window.HTMLElement,
    HTMLTextAreaElement: dom.window.HTMLTextAreaElement,
    InputEvent: dom.window.InputEvent,
    Event: dom.window.Event,
    Location: dom.window.Location,
    MutationObserver: dom.window.MutationObserver
  });
  return dom;
}

function attachFiber(textarea: HTMLTextAreaElement, annotationId: string, processedRecordingId: string) {
  const fiberKey = '__reactFiber$test';
  (textarea as unknown as Record<string, unknown>)[fiberKey] = {
    memoizedProps: {
      annotation: {
        id: annotationId,
        processedRecordingId
      }
    },
    return: null
  };
}

test('captureTranscriptJob extracts locked row payload from Babel-like DOM', () => {
  const dom = installDom(`
    <table>
      <tbody>
        <tr>
          <td>1</td>
          <td>Speaker A</td>
          <td>00:00:01.0</td>
          <td>00:00:03.0</td>
          <td><textarea placeholder="What was said">privet</textarea></td>
        </tr>
        <tr>
          <td>2</td>
          <td>Speaker B</td>
          <td>00:00:03.0</td>
          <td>00:00:05.0</td>
          <td><textarea placeholder="What was said">da</textarea></td>
        </tr>
      </tbody>
    </table>
  `);

  const textareas = dom.window.document.querySelectorAll<HTMLTextAreaElement>('textarea');
  attachFiber(textareas[0], 'ann-1', 'spk-1');
  attachFiber(textareas[1], 'ann-2', 'spk-2');

  const job = captureTranscriptJob(dom.window.document, dom.window.location);

  assert.equal(job.jobId, 'job-42');
  assert.deepEqual(job.rows, [
    {
      rowId: 'ann-1',
      speakerKey: 'spk-1',
      startSeconds: 1,
      endSeconds: 3,
      text: 'privet',
      index: 0
    },
    {
      rowId: 'ann-2',
      speakerKey: 'spk-2',
      startSeconds: 3,
      endSeconds: 5,
      text: 'da',
      index: 1
    }
  ]);
});

test('applyDraftRows updates matching textareas without changing row structure', () => {
  const dom = installDom(`
    <table>
      <tbody>
        <tr>
          <td>1</td>
          <td>Speaker A</td>
          <td>00:00:01.0</td>
          <td>00:00:03.0</td>
          <td><textarea placeholder="What was said">privet</textarea></td>
        </tr>
      </tbody>
    </table>
  `);

  const textarea = dom.window.document.querySelector<HTMLTextAreaElement>('textarea');
  assert.ok(textarea);
  attachFiber(textarea, 'ann-1', 'spk-1');

  const result = applyDraftRows([
    {
      rowId: 'ann-1',
      rewrittenText: 'Privet.',
      status: 'rewritten',
      warnings: []
    }
  ]);

  assert.equal(result.appliedCount, 1);
  assert.deepEqual(result.missingRowIds, []);
  assert.equal(textarea.value, 'Privet.');
});

test('restoreCapturedRows replays original snapshot and diff preview shows changes', () => {
  const dom = installDom(`
    <table>
      <tbody>
        <tr>
          <td>1</td>
          <td>Speaker A</td>
          <td>00:00:01.0</td>
          <td>00:00:03.0</td>
          <td><textarea placeholder="What was said">privet</textarea></td>
        </tr>
      </tbody>
    </table>
  `);

  const textarea = dom.window.document.querySelector<HTMLTextAreaElement>('textarea');
  assert.ok(textarea);
  attachFiber(textarea, 'ann-1', 'spk-1');

  const captured = captureTranscriptJob(dom.window.document, dom.window.location);
  textarea.value = 'Privet.';

  const restoreResult = restoreCapturedRows(captured);
  assert.equal(restoreResult.appliedCount, 1);
  assert.equal(textarea.value, 'privet');

  const diff = buildDiffPreviewItems(captured.rows, [
    {
      rowId: 'ann-1',
      rewrittenText: 'Privet.',
      status: 'rewritten',
      warnings: ['terminal_punctuation_added']
    }
  ]);

  assert.equal(diff.length, 1);
  assert.equal(diff[0].before, 'privet');
  assert.equal(diff[0].after, 'Privet.');
});

test('drafting lifecycle re-ensures mount target on DOM churn without teardown', async () => {
  const dom = installDom('<main></main>');
  let ensureCount = 0;

  registerLifecycle({
    ensureMagicButton: () => {
      ensureCount += 1;
    }
  });

  assert.equal(ensureCount, 1);

  dom.window.document.querySelector('main')?.appendChild(dom.window.document.createElement('section'));
  await new Promise((resolve) => dom.window.setTimeout(resolve, 0));

  assert.equal(ensureCount, 2);
});
