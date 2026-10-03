import test from 'node:test';
import assert from 'node:assert/strict';
import { buildActivityRows, segmentSamplesByActivity } from '../src/core/local-audio-segmentation';
import { buildSegmentWindows } from '../src/core/c-denoise-runtime';
import { shouldRunGoldLlmAfterL0 } from '../src/content/overlay';
import { DEFAULT_SETTINGS } from '../src/core/settings';

const rate = 16_000;
function waveform(seconds: number, active: Array<[number, number]>): Float32Array {
  const samples = new Float32Array(seconds * rate);
  for (const [start, end] of active) samples.fill(0.2, start * rate, end * rate);
  return samples;
}

test('audio rows retain a long thought across word-count, duration and short-pause cuts', () => {
  const words = Array.from({ length: 80 }, (_, index) => ({ text: `w${index}`, startSeconds: index * 0.3, endSeconds: index * 0.3 + 0.15 }));
  const audio = waveform(25, [[0, 10], [10.9, 24]]);
  assert.equal(segmentSamplesByActivity(audio).length, 1);
  const rows = buildActivityRows(words, audio);
  assert.equal(rows.length, 1);
  assert.deepEqual([rows[0].wordStart, rows[0].wordEnd], [0, 80]);
});

test('a real audio pause separates rows and retains even quiet words exactly once', () => {
  const words = [
    { text: 'один', startSeconds: 0.5, endSeconds: 1 },
    { text: 'тихо', startSeconds: 2, endSeconds: 2.2 },
    { text: 'два', startSeconds: 4, endSeconds: 4.5 }
  ];
  const rows = buildActivityRows(words, waveform(6, [[0.4, 1.4], [3.8, 5]]));
  assert.deepEqual(rows.map(row => [row.wordStart, row.wordEnd]), [[0, 2], [2, 3]]);
  assert.ok(rows[0].endSample <= rows[1].startSample);
  assert.deepEqual(buildActivityRows(words, new Float32Array(6 * rate)).map(row => [row.wordStart, row.wordEnd]), [[0, 3]]);
});

test('punctuation owns a complete fitting segment in a single joint pass', () => {
  const windows = buildSegmentWindows([{ wordStart: 0, wordEnd: 80 }], Array(80).fill(3));
  assert.deepEqual(windows, [{ left: 0, coreLeft: 0, coreRight: 80, right: 80 }]);
});

test('long segments have overlapping punctuation context with complete unique word ownership', () => {
  const costs = Array(200).fill(3);
  const windows = buildSegmentWindows([{ wordStart: 0, wordEnd: 200 }], costs);
  assert.equal(windows[0].coreLeft, 0);
  assert.equal(windows.at(-1)!.coreRight, 200);
  windows.forEach((window, index) => {
    assert.ok((window.right - window.left) * 3 + 2 <= 256);
    if (index) {
      assert.equal(window.coreLeft, windows[index - 1].coreRight);
      assert.ok(window.left < window.coreLeft);
      assert.ok(window.left < windows[index - 1].right);
    }
  });
  assert.throws(() => buildSegmentWindows([{ wordStart: 1, wordEnd: 2 }], [3, 3]), /exactly once/);
});

test('Local honors the LLM preference and configured key; Simple stays native', () => {
  assert.equal(shouldRunGoldLlmAfterL0(DEFAULT_SETTINGS), false);
  assert.equal(shouldRunGoldLlmAfterL0({ ...DEFAULT_SETTINGS, openRouterApiKey: 'key' }), true);
  assert.equal(shouldRunGoldLlmAfterL0({ ...DEFAULT_SETTINGS, openRouterApiKey: 'key', l0DontRunLlm: true }), false);
  assert.equal(shouldRunGoldLlmAfterL0({ ...DEFAULT_SETTINGS, mode: 'simple', openRouterApiKey: 'key' }), false);
});
