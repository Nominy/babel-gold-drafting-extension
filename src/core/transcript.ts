import {
  getReactFiber,
  getTranscriptRowElements,
  normalizeText,
  parseTimeValue,
  ROW_TEXTAREA_SELECTOR,
  setControlledTextareaValue
} from './dom';
import { PAGE_TASK_ID_ATTRIBUTE } from './audio-intercept-protocol';
import type { ApplyDraftResult, DiffPreviewItem, DraftRowResult, TranscriptJob, TranscriptRow } from './types';

type RowIdentity = {
  rowId: string | null;
  speakerKey: string;
  processedRecordingId: string;
  startText: string;
  endText: string;
  startSeconds: number | null;
  endSeconds: number | null;
};

type ReactFiberNode = {
  memoizedProps?: unknown;
  return?: ReactFiberNode | null;
};

function readFiniteNumber(value: unknown): number | null {
  const numeric = typeof value === 'number' ? value : typeof value === 'string' ? Number(value) : Number.NaN;
  return Number.isFinite(numeric) ? numeric : null;
}

function readVisibleTimeCells(row: HTMLTableRowElement): {
  speakerKey: string;
  startText: string;
  endText: string;
  startSeconds: number | null;
  endSeconds: number | null;
} {
  const cells = Array.from(row.children).filter((child): child is HTMLElement => child instanceof HTMLElement);
  const textarea = row.querySelector<HTMLTextAreaElement>(ROW_TEXTAREA_SELECTOR);
  const textareaCell = textarea?.closest('td, th');
  const textareaCellIndex = textareaCell ? cells.indexOf(textareaCell as HTMLElement) : -1;
  const cellsBeforeText = textareaCellIndex >= 0 ? cells.slice(0, textareaCellIndex) : cells;
  const timeCells = cellsBeforeText
    .map((cell, index) => {
      const text = normalizeText(cell);
      return {
        cell,
        index,
        text,
        seconds: parseTimeValue(text)
      };
    })
    .filter((item): item is { cell: HTMLElement; index: number; text: string; seconds: number } => item.seconds !== null);
  const start = timeCells.at(-2);
  const end = timeCells.at(-1);
  const speakerCell = start
    ? cellsBeforeText
        .slice(0, start.index)
        .reverse()
        .find((cell) => normalizeText(cell) && parseTimeValue(normalizeText(cell)) === null)
    : (row.children[1] as HTMLElement | undefined);

  return {
    speakerKey: normalizeText(speakerCell),
    startText: start?.text || '',
    endText: end?.text || '',
    startSeconds: start?.seconds ?? null,
    endSeconds: end?.seconds ?? null
  };
}

function readRowIdentity(row: HTMLTableRowElement): RowIdentity {
  const visibleCells = readVisibleTimeCells(row);

  const identity: RowIdentity = {
    rowId: null,
    speakerKey: visibleCells.speakerKey,
    processedRecordingId: '',
    startText: visibleCells.startText,
    endText: visibleCells.endText,
    startSeconds: visibleCells.startSeconds,
    endSeconds: visibleCells.endSeconds
  };

  const fiber = (getReactFiber(row) || getReactFiber(row.querySelector(ROW_TEXTAREA_SELECTOR))) as
    | { memoizedProps?: unknown; return?: unknown }
    | null;

  let current: { memoizedProps?: unknown; return?: unknown } | null = fiber;
  let depth = 0;
  while (current && depth < 12) {
    const props = current.memoizedProps;
    const annotation =
      props &&
      typeof props === 'object' &&
      'annotation' in props &&
      props.annotation &&
      typeof props.annotation === 'object'
        ? (props.annotation as Record<string, unknown>)
        : null;

    if (annotation && typeof annotation.id === 'string' && annotation.id.trim()) {
      identity.rowId = annotation.id.trim();
      identity.processedRecordingId =
        annotation.processedRecordingId != null ? String(annotation.processedRecordingId).trim() : '';
      const trackLabel = typeof annotation.trackLabel === 'string' ? annotation.trackLabel.trim() : '';
      if (identity.processedRecordingId || trackLabel) {
        identity.speakerKey = identity.processedRecordingId || trackLabel;
      }
      identity.startSeconds = readFiniteNumber(annotation.startTimeInSeconds) ?? identity.startSeconds;
      identity.endSeconds = readFiniteNumber(annotation.endTimeInSeconds) ?? identity.endSeconds;
      break;
    }

    current = current.return as { memoizedProps?: unknown; return?: unknown } | null;
    depth += 1;
  }

  return identity;
}

function readReviewActionId(row: HTMLTableRowElement): string {
  const textarea = row.querySelector<HTMLTextAreaElement>(ROW_TEXTAREA_SELECTOR);
  const seeds = [getReactFiber(row), getReactFiber(textarea)].filter(Boolean) as ReactFiberNode[];
  for (const seed of seeds) {
    let current: ReactFiberNode | null = seed;
    let depth = 0;
    while (current && depth <= 30) {
      const props = current.memoizedProps;
      if (props && typeof props === 'object' && 'reviewActionId' in props) {
        const reviewActionId = (props as Record<string, unknown>).reviewActionId;
        if (typeof reviewActionId === 'string' && reviewActionId.trim()) {
          return reviewActionId.trim();
        }
      }
      current = current.return ?? null;
      depth += 1;
    }
  }
  return '';
}

function readPublishedReviewActionId(root: ParentNode): string {
  const documentRef =
    'documentElement' in root
      ? (root as Document)
      : ((root as Node).ownerDocument ?? null);
  return documentRef?.documentElement?.getAttribute(PAGE_TASK_ID_ATTRIBUTE)?.trim() || '';
}

function makeFallbackRowId(identity: RowIdentity, rowIndex: number): string {
  return `row:${identity.speakerKey}:${identity.startText}:${identity.endText}:${rowIndex}`;
}

export function buildJobId(locationLike: Pick<Location, 'pathname' | 'search'> = window.location): string {
  const search = locationLike.search || '';
  const pathname = locationLike.pathname || '';
  const query = new URLSearchParams(search);
  const explicitId = ['jobId', 'transcriptionChunkId', 'annotationId', 'id']
    .map((key) => query.get(key)?.trim() || '')
    .find(Boolean);

  return explicitId || `${pathname}${search}`;
}

export function buildCanonicalTaskIdentity(job: Pick<TranscriptJob, 'jobId' | 'rows'>): string {
  const processedRecordingIds = job.rows
    .map((row) => row.processedRecordingId?.trim() || '')
    .filter(Boolean);
  const laneValues = processedRecordingIds.length > 0
    ? processedRecordingIds
    : job.rows.map((row) => row.speakerKey.trim()).filter(Boolean);
  const stableLaneIds = Array.from(new Set(laneValues)).sort();
  return JSON.stringify({
    version: 1,
    baseTaskId: job.jobId.trim(),
    stableLaneIds
  });
}

export function captureTranscriptJob(
  root: ParentNode = document,
  locationLike: Pick<Location, 'pathname' | 'search'> = window.location
): TranscriptJob {
  const rowElements = getTranscriptRowElements(root);
  const rows = rowElements.map<TranscriptRow>((row, index) => {
    const identity = readRowIdentity(row);
    const textarea = row.querySelector<HTMLTextAreaElement>(ROW_TEXTAREA_SELECTOR);

    return {
      rowId: identity.rowId || makeFallbackRowId(identity, index),
      speakerKey: identity.speakerKey,
      ...(identity.processedRecordingId ? { processedRecordingId: identity.processedRecordingId } : {}),
      startSeconds: identity.startSeconds,
      endSeconds: identity.endSeconds,
      text: textarea?.value || '',
      index
    };
  });

  const reviewActionId =
    readPublishedReviewActionId(root)
    || rowElements.map(readReviewActionId).find(Boolean)
    || '';
  return {
    jobId: reviewActionId || buildJobId(locationLike),
    rows
  };
}

function buildLocatorMap(root: ParentNode = document): Map<string, HTMLTextAreaElement> {
  const map = new Map<string, HTMLTextAreaElement>();
  for (const [index, row] of getTranscriptRowElements(root).entries()) {
    const identity = readRowIdentity(row);
    const textarea = row.querySelector<HTMLTextAreaElement>(ROW_TEXTAREA_SELECTOR);
    if (!textarea) {
      continue;
    }

    const key = identity.rowId || makeFallbackRowId(identity, index);
    map.set(key, textarea);
  }
  return map;
}

export function applyDraftRows(draftRows: DraftRowResult[], root: ParentNode = document): ApplyDraftResult {
  const locatorMap = buildLocatorMap(root);
  let appliedCount = 0;
  const missingRowIds: string[] = [];

  for (const row of draftRows) {
    const textarea = locatorMap.get(row.rowId);
    if (!textarea) {
      missingRowIds.push(row.rowId);
      continue;
    }

    setControlledTextareaValue(textarea, row.rewrittenText);
    appliedCount += 1;
  }

  return {
    appliedCount,
    missingRowIds
  };
}

export function restoreCapturedRows(job: TranscriptJob, root: ParentNode = document): ApplyDraftResult {
  return applyDraftRows(
    job.rows.map((row) => ({
      rowId: row.rowId,
      rewrittenText: row.text,
      status: 'unchanged',
      warnings: []
    })),
    root
  );
}

export function buildDiffPreviewItems(originalRows: TranscriptRow[], draftRows: DraftRowResult[]): DiffPreviewItem[] {
  const originalById = new Map(originalRows.map((row) => [row.rowId, row]));

  return draftRows
    .map<DiffPreviewItem | null>((draftRow) => {
      const original = originalById.get(draftRow.rowId);
      if (!original) {
        return null;
      }

      const before = original.text;
      const after = draftRow.rewrittenText;
      if (before === after && draftRow.status !== 'failed') {
        return null;
      }

      return {
        rowId: draftRow.rowId,
        index: original.index,
        before,
        after,
        status: draftRow.status,
        warnings: [...draftRow.warnings]
      };
    })
    .filter((item): item is DiffPreviewItem => item !== null)
    .sort((left, right) => left.index - right.index);
}
