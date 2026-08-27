import type { L0DraftRow, TranscriptRow } from './types';

/**
 * Babel displays segment timestamps rounded to hundredths of a second. The
 * largest round-to-nearest error is 0.005s; the epsilon absorbs binary
 * floating-point representation at that boundary.
 */
export const L0_VISIBLE_TIMESTAMP_TOLERANCE_SECONDS = 0.005 + 1e-9;

export interface MatchedL0CreatedRow {
  engineRow: L0DraftRow;
  capturedRow: TranscriptRow;
}

function normalizeLane(lane: string): string {
  return lane.trim().replace(/\s+/g, ' ').toLocaleLowerCase();
}

function timestampsMatch(expected: number, captured: number | null): boolean {
  return (
    captured !== null &&
    Number.isFinite(captured) &&
    Math.abs(expected - captured) <= L0_VISIBLE_TIMESTAMP_TOLERANCE_SECONDS
  );
}

function rowShapeMatches(expected: L0DraftRow, captured: TranscriptRow): boolean {
  return (
    normalizeLane(expected.lane) === normalizeLane(captured.speakerKey) &&
    timestampsMatch(expected.startSeconds, captured.startSeconds) &&
    timestampsMatch(expected.endSeconds, captured.endSeconds)
  );
}

function describeEngineRow(row: L0DraftRow): string {
  return `${row.id} (${row.lane} ${row.startSeconds.toFixed(3)}-${row.endSeconds.toFixed(3)})`;
}

function describeCapturedRow(row: TranscriptRow): string {
  const start = row.startSeconds === null ? '?' : row.startSeconds.toFixed(3);
  const end = row.endSeconds === null ? '?' : row.endSeconds.toFixed(3);
  return `${row.rowId} (${row.speakerKey} ${start}-${end})`;
}

function findPerfectMatching(candidates: number[][], capturedCount: number): {
  expectedToCaptured: number[];
  capturedToExpected: number[];
} | null {
  const capturedToExpected = Array<number>(capturedCount).fill(-1);

  const assign = (expectedIndex: number, visited: boolean[]): boolean => {
    for (const capturedIndex of candidates[expectedIndex]) {
      if (visited[capturedIndex]) {
        continue;
      }
      visited[capturedIndex] = true;
      const currentExpected = capturedToExpected[capturedIndex];
      if (currentExpected === -1 || assign(currentExpected, visited)) {
        capturedToExpected[capturedIndex] = expectedIndex;
        return true;
      }
    }
    return false;
  };

  for (let expectedIndex = 0; expectedIndex < candidates.length; expectedIndex += 1) {
    if (!assign(expectedIndex, Array<boolean>(capturedCount).fill(false))) {
      return null;
    }
  }

  const expectedToCaptured = Array<number>(candidates.length).fill(-1);
  for (let capturedIndex = 0; capturedIndex < capturedToExpected.length; capturedIndex += 1) {
    const expectedIndex = capturedToExpected[capturedIndex];
    if (expectedIndex >= 0) {
      expectedToCaptured[expectedIndex] = capturedIndex;
    }
  }
  return { expectedToCaptured, capturedToExpected };
}

function hasAlternativeMatchingCycle(
  candidates: number[][],
  expectedToCaptured: number[],
  capturedToExpected: number[]
): boolean {
  const edges = candidates.map((capturedCandidates, expectedIndex) =>
    capturedCandidates
      .filter((capturedIndex) => capturedIndex !== expectedToCaptured[expectedIndex])
      .map((capturedIndex) => capturedToExpected[capturedIndex])
      .filter((otherExpected) => otherExpected >= 0)
  );
  const states = Array<number>(candidates.length).fill(0);

  const visitsCycle = (expectedIndex: number): boolean => {
    if (states[expectedIndex] === 1) {
      return true;
    }
    if (states[expectedIndex] === 2) {
      return false;
    }
    states[expectedIndex] = 1;
    for (const nextExpected of edges[expectedIndex]) {
      if (visitsCycle(nextExpected)) {
        return true;
      }
    }
    states[expectedIndex] = 2;
    return false;
  };

  return states.some((_, expectedIndex) => visitsCycle(expectedIndex));
}

/**
 * Associates each engine row with exactly one row recaptured from the visible
 * transcript. Annotation IDs are deliberately not inputs: content scripts
 * cannot observe page-world React annotation expandos.
 */
export function matchL0CreatedRows(
  engineRows: L0DraftRow[],
  capturedRows: TranscriptRow[]
): MatchedL0CreatedRow[] {
  if (!engineRows.length && !capturedRows.length) {
    return [];
  }

  const structuralCandidates = engineRows.map((engineRow) =>
    capturedRows.flatMap((capturedRow, capturedIndex) =>
      rowShapeMatches(engineRow, capturedRow) ? [capturedIndex] : []
    )
  );
  const textCandidates = structuralCandidates.map((candidates, engineIndex) =>
    candidates.filter((capturedIndex) => capturedRows[capturedIndex].text === engineRows[engineIndex].text)
  );

  const problems: string[] = [];
  if (capturedRows.length < engineRows.length) {
    problems.push(`${engineRows.length - capturedRows.length} replaced L0 row(s) are missing from the visible transcript.`);
  } else if (capturedRows.length > engineRows.length) {
    problems.push(`${capturedRows.length - engineRows.length} unexpected visible transcript row(s) remain after replacement.`);
  }

  const missingRows = engineRows.filter((_, index) => structuralCandidates[index].length === 0);
  if (missingRows.length) {
    problems.push(`Missing: ${missingRows.map(describeEngineRow).join(', ')}.`);
  }

  const textMismatchRows = engineRows.filter(
    (_, index) => structuralCandidates[index].length > 0 && textCandidates[index].length === 0
  );
  if (textMismatchRows.length) {
    problems.push(`Final text mismatch: ${textMismatchRows.map(describeEngineRow).join(', ')}.`);
  }

  const unexpectedRows = capturedRows.filter((_, capturedIndex) =>
    structuralCandidates.every((candidates) => !candidates.includes(capturedIndex))
  );
  if (unexpectedRows.length) {
    problems.push(`Unexpected: ${unexpectedRows.map(describeCapturedRow).join(', ')}.`);
  }

  if (problems.length) {
    throw new Error(`Could not verify Helper-created L0 rows without rewriting the transcript. ${problems.join(' ')}`);
  }

  const matching = findPerfectMatching(textCandidates, capturedRows.length);
  if (!matching || matching.expectedToCaptured.some((capturedIndex) => capturedIndex < 0)) {
    throw new Error(
      'Could not bijectively match Helper-created L0 rows; visible rows are missing, duplicated, or conflict within the timestamp tolerance.'
    );
  }
  if (hasAlternativeMatchingCycle(textCandidates, matching.expectedToCaptured, matching.capturedToExpected)) {
    throw new Error(
      'Ambiguous duplicate Helper-created L0 rows: more than one lane/timestamp/text bijection matches the visible transcript.'
    );
  }

  return engineRows.map((engineRow, expectedIndex) => ({
    engineRow,
    capturedRow: capturedRows[matching.expectedToCaptured[expectedIndex]]
  }));
}
