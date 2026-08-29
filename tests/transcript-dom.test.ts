import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { registerLifecycle } from '../src/core/lifecycle';
import {
  applyDraftRows,
  buildCanonicalTaskIdentity,
  buildDiffPreviewItems,
  buildJobId,
  captureTranscriptJob,
  restoreCapturedRows
} from '../src/core/transcript';

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

function attachFiber(
  textarea: HTMLTextAreaElement,
  annotationId: string,
  processedRecordingId: string,
  reviewActionId = ''
) {
  const fiberKey = '__reactFiber$test';
  (textarea as unknown as Record<string, unknown>)[fiberKey] = {
    memoizedProps: {
      annotation: {
        id: annotationId,
        processedRecordingId
      }
    },
    return: reviewActionId
      ? {
          memoizedProps: { reviewActionId },
          return: null
        }
      : null
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
      processedRecordingId: 'spk-1',
      startSeconds: 1,
      endSeconds: 3,
      text: 'privet',
      index: 0
    },
    {
      rowId: 'ann-2',
      speakerKey: 'spk-2',
      startSeconds: 3,
      processedRecordingId: 'spk-2',
      endSeconds: 5,
      text: 'da',
      index: 1
    }
  ]);
});

test('captureTranscriptJob prefers the review action over a shared route identity', () => {
  const dom = installDom(`
    <table><tbody><tr>
      <td>1</td><td>Speaker A</td><td>00:00:01.0</td><td>00:00:03.0</td>
      <td><textarea placeholder="What was said">privet</textarea></td>
    </tr></tbody></table>
  `);
  const textarea = dom.window.document.querySelector<HTMLTextAreaElement>('textarea');
  assert.ok(textarea);
  attachFiber(textarea, 'ann-1', 'recording-a', 'review-action-new');

  const job = captureTranscriptJob(dom.window.document, dom.window.location);

  assert.equal(job.jobId, 'review-action-new');
  assert.equal(
    buildCanonicalTaskIdentity(job),
    '{"version":1,"baseTaskId":"review-action-new","stableLaneIds":[]}'
  );
  assert.equal(
    buildCanonicalTaskIdentity({
      ...job,
      rows: job.rows.map((row) => ({ ...row, processedRecordingId: 'hidden-lane-change' }))
    }),
    buildCanonicalTaskIdentity(job),
    'lane visibility must not change a review-scoped task identity'
  );
});

test('captureTranscriptJob uses the main-world published review action in the isolated world', () => {
  const dom = installDom(`
    <table><tbody><tr>
      <td>1</td><td>Speaker A</td><td>00:00:01.0</td><td>00:00:03.0</td>
      <td><textarea placeholder="What was said">privet</textarea></td>
    </tr></tbody></table>
  `);
  dom.window.document.documentElement.setAttribute(
    'data-babel-review-action-id',
    'review-action-bridged'
  );

  const job = captureTranscriptJob(dom.window.document, dom.window.location);

  assert.equal(job.jobId, 'review-action-bridged');
});

test('captureTranscriptJob detects start and end columns instead of assuming fixed indexes', () => {
  const dom = installDom(`
    <table>
      <tbody>
        <tr>
          <td>Speaker 1</td>
          <td>02:30.22</td>
          <td>02:36.78</td>
          <td><textarea placeholder="What was said">А Spotify и YouTube Music. Ну, из того, что я заметил, эти алгоритмы, они стараются.</textarea></td>
        </tr>
      </tbody>
    </table>
  `);

  const job = captureTranscriptJob(dom.window.document, dom.window.location);

  assert.deepEqual(job.rows, [
    {
      rowId: 'row:Speaker 1:02:30.22:02:36.78:0',
      speakerKey: 'Speaker 1',
      startSeconds: 150.22,
      endSeconds: 156.78,
      text: 'А Spotify и YouTube Music. Ну, из того, что я заметил, эти алгоритмы, они стараются.',
      index: 0
    }
  ]);
});

test('canonical task identity distinguishes stable lanes on the same route and orders them deterministically', () => {
  const baseTaskId = buildJobId({ pathname: '/transcription/RU-tx-gold-non-bg', search: '' });
  const row = {
    rowId: 'ann-1',
    speakerKey: 'Speaker 1',
    startSeconds: 0,
    endSeconds: 1,
    text: '',
    index: 0
  };
  const first = buildCanonicalTaskIdentity({
    jobId: baseTaskId,
    rows: [
      { ...row, rowId: 'ann-2', processedRecordingId: ' recording-b ' },
      { ...row, processedRecordingId: 'recording-a' },
      { ...row, rowId: 'ann-3', processedRecordingId: 'recording-a' }
    ]
  });
  const second = buildCanonicalTaskIdentity({
    jobId: baseTaskId,
    rows: [{ ...row, processedRecordingId: 'recording-c' }]
  });

  assert.equal(first, '{"version":1,"baseTaskId":"/transcription/RU-tx-gold-non-bg","stableLaneIds":["recording-a","recording-b"]}');
  assert.equal(second, '{"version":1,"baseTaskId":"/transcription/RU-tx-gold-non-bg","stableLaneIds":["recording-c"]}');
  assert.notEqual(first, second);
});

test('canonical task identity supports one lane and falls back to speaker keys only without processed recording IDs', () => {
  assert.equal(
    buildCanonicalTaskIdentity({
      jobId: ' job-42 ',
      rows: [{
        rowId: 'ann-1',
        speakerKey: ' Speaker 1 ',
        startSeconds: 0,
        endSeconds: 1,
        text: '',
        index: 0
      }]
    }),
    '{"version":1,"baseTaskId":"job-42","stableLaneIds":["Speaker 1"]}'
  );
  assert.equal(
    buildJobId({ pathname: '/fallback', search: '?jobId=%20%20&transcriptionChunkId=%20chunk-9%20&id=ignored' }),
    'chunk-9'
  );
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
  let timingCount = 0;

  registerLifecycle({
    ensureMagicButton: () => {
      ensureCount += 1;
    }
  }, () => {
    timingCount += 1;
  });

  assert.equal(ensureCount, 1);
  assert.equal(timingCount, 1);

  dom.window.document.querySelector('main')?.appendChild(dom.window.document.createElement('section'));
  await new Promise((resolve) => dom.window.setTimeout(resolve, 0));

  assert.equal(ensureCount, 2);
  assert.equal(timingCount, 2);
});
