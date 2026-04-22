import {
  getReactFiber,
  getTranscriptRowElements,
  normalizeText,
  parseTimeValue,
  ROW_TEXTAREA_SELECTOR,
  setControlledTextareaValue
} from './dom';
import type { ApplyDraftResult, DiffPreviewItem, DraftRowResult, TranscriptJob, TranscriptRow } from './types';

type RowIdentity = {
  rowId: string | null;
  speakerKey: string;
  startText: string;
  endText: string;
};

function readRowIdentity(row: HTMLTableRowElement): RowIdentity {
  const startCell = row.children[2] as HTMLElement | undefined;
  const endCell = row.children[3] as HTMLElement | undefined;
  const speakerCell = row.children[1] as HTMLElement | undefined;

  const identity: RowIdentity = {
    rowId: null,
    speakerKey: normalizeText(speakerCell),
    startText: normalizeText(startCell),
    endText: normalizeText(endCell)
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
      const processedRecordingId =
        annotation.processedRecordingId != null ? String(annotation.processedRecordingId).trim() : '';
      const trackLabel = typeof annotation.trackLabel === 'string' ? annotation.trackLabel.trim() : '';
      if (processedRecordingId || trackLabel) {
        identity.speakerKey = processedRecordingId || trackLabel;
      }
      break;
    }

    current = current.return as { memoizedProps?: unknown; return?: unknown } | null;
    depth += 1;
  }

  return identity;
}

function makeFallbackRowId(identity: RowIdentity, rowIndex: number): string {
  return `row:${identity.speakerKey}:${identity.startText}:${identity.endText}:${rowIndex}`;
}

export function buildJobId(locationLike: Pick<Location, 'pathname' | 'search'> = window.location): string {
  const search = locationLike.search || '';
  const pathname = locationLike.pathname || '';
  const query = new URLSearchParams(search);
  const explicitId =
    query.get('jobId') ||
    query.get('transcriptionChunkId') ||
    query.get('annotationId') ||
    query.get('id');

  return explicitId ? explicitId.trim() : `${pathname}${search}`;
}

export function captureTranscriptJob(
  root: ParentNode = document,
  locationLike: Pick<Location, 'pathname' | 'search'> = window.location
): TranscriptJob {
  const rows = getTranscriptRowElements(root).map<TranscriptRow>((row, index) => {
    const identity = readRowIdentity(row);
    const textarea = row.querySelector<HTMLTextAreaElement>(ROW_TEXTAREA_SELECTOR);
    const startCell = row.children[2] as HTMLElement | undefined;
    const endCell = row.children[3] as HTMLElement | undefined;

    return {
      rowId: identity.rowId || makeFallbackRowId(identity, index),
      speakerKey: identity.speakerKey,
      startSeconds: startCell ? parseTimeValue(normalizeText(startCell)) : null,
      endSeconds: endCell ? parseTimeValue(normalizeText(endCell)) : null,
      text: textarea?.value || '',
      index
    };
  });

  return {
    jobId: buildJobId(locationLike),
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
