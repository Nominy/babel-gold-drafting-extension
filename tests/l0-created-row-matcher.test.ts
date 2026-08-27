import test from 'node:test';
import assert from 'node:assert/strict';
import { matchL0CreatedRows } from '../src/core/l0-created-row-matcher';
import type { L0DraftRow, TranscriptRow } from '../src/core/types';

function engineRow(
  id: string,
  lane: string,
  startSeconds: number,
  endSeconds: number,
  text: string
): L0DraftRow {
  return { id, lane, startSeconds, endSeconds, text };
}

function capturedRow(
  rowId: string,
  speakerKey: string,
  startSeconds: number,
  endSeconds: number,
  text: string,
  index: number
): TranscriptRow {
  return { rowId, speakerKey, startSeconds, endSeconds, text, index };
}

test('matches populated rows by normalized lane, timestamps, and final text', () => {
  const expected = [engineRow('l0-1', ' Speaker   One ', 1, 2.5, 'Готовый текст.')];
  const captured = [capturedRow('visible-row-1', 'speaker one', 1, 2.5, 'Готовый текст.', 0)];
  const matched = matchL0CreatedRows(expected, captured);
  assert.equal(matched.length, 1);
  assert.equal(matched[0].engineRow, expected[0]);
  assert.equal(matched[0].capturedRow, captured[0]);
});

test('matches reordered visible rows and returns them in engine response order', () => {
  const expected = [
    engineRow('l0-1', 'speaker-1', 0, 1.5, 'Первый.'),
    engineRow('l0-2', 'speaker-2', 1.5, 3, 'Второй.')
  ];
  const captured = [
    capturedRow('visible-row-2', 'speaker-2', 1.5, 3, 'Второй.', 0),
    capturedRow('visible-row-1', 'speaker-1', 0, 1.5, 'Первый.', 1)
  ];
  assert.deepEqual(
    matchL0CreatedRows(expected, captured).map(({ capturedRow: row }) => row.rowId),
    ['visible-row-1', 'visible-row-2']
  );
});

test('accepts timestamps rounded to displayed hundredths of a second', () => {
  const expected = [engineRow('l0-rounded', 'speaker-1', 12.345, 18.675, 'Округлено.')];
  const captured = [capturedRow('visible-rounded', 'speaker-1', 12.35, 18.68, 'Округлено.', 0)];
  assert.equal(matchL0CreatedRows(expected, captured)[0].capturedRow.rowId, 'visible-rounded');
});

test('rejects duplicate rows when more than one observable bijection is possible', () => {
  const expected = [
    engineRow('l0-a', 'speaker-1', 4, 5, 'Повтор.'),
    engineRow('l0-b', 'speaker-1', 4, 5, 'Повтор.')
  ];
  const captured = [
    capturedRow('visible-a', 'speaker-1', 4, 5, 'Повтор.', 0),
    capturedRow('visible-b', 'speaker-1', 4, 5, 'Повтор.', 1)
  ];
  assert.throws(() => matchL0CreatedRows(expected, captured), /Ambiguous duplicate Helper-created L0 rows/);
});

test('reports an engine row missing from the visible transcript', () => {
  const expected = [
    engineRow('l0-1', 'speaker-1', 0, 1, 'Первый.'),
    engineRow('l0-2', 'speaker-2', 1, 2, 'Второй.')
  ];
  const captured = [capturedRow('visible-row-1', 'speaker-1', 0, 1, 'Первый.', 0)];
  assert.throws(
    () => matchL0CreatedRows(expected, captured),
    /1 replaced L0 row\(s\) are missing.*Missing: l0-2/s
  );
});

test('reports final text that Helper did not populate exactly', () => {
  const expected = [engineRow('l0-text', 'speaker-1', 2, 3, 'Финальный текст.')];
  const captured = [capturedRow('visible-text', 'speaker-1', 2, 3, '', 0)];
  assert.throws(() => matchL0CreatedRows(expected, captured), /Final text mismatch: l0-text/);
});

test('reports visible rows that were not returned by the engine', () => {
  const expected = [engineRow('l0-1', 'speaker-1', 0, 1, 'Первый.')];
  const captured = [
    capturedRow('visible-row-1', 'speaker-1', 0, 1, 'Первый.', 0),
    capturedRow('stale-row', 'speaker-2', 8, 9, 'Старый.', 1)
  ];
  assert.throws(
    () => matchL0CreatedRows(expected, captured),
    /1 unexpected visible transcript row\(s\).*Unexpected: stale-row/s
  );
});
