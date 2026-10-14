import { readAudioEnhancementState } from '@nominy/babel-babel-runtime';
import { PAGE_TASK_ID_ATTRIBUTE } from './audio-intercept-protocol';
import { dispatchTranscriptEvent, parseTranscriptEvent, TRANSCRIPT_REQUEST_EVENT, TRANSCRIPT_RESPONSE_EVENT, type TranscriptRequest, type TranscriptResponse } from './transcript-protocol';
import type { ApplyDraftResult, DiffPreviewItem, DraftRowResult, TranscriptJob, TranscriptRow } from './types';


export function buildJobId(locationLike: Pick<Location, 'pathname' | 'search'> = window.location): string {
  const search = locationLike.search || '';
  const pathname = locationLike.pathname || '';
  const query = new URLSearchParams(search);
  const explicitId = ['jobId', 'transcriptionChunkId', 'annotationId', 'id']
    .map((key) => query.get(key)?.trim() || '')
    .find(Boolean);

  return explicitId || `${pathname}${search}`;
}

export function buildCanonicalTaskIdentity(
  job: Pick<TranscriptJob, 'jobId' | 'rows' | 'taskScoped' | 'audioVariantKey' | 'audioDependent'>
): string {
  const processedRecordingIds = job.rows
    .map((row) => row.processedRecordingId?.trim() || '')
    .filter(Boolean);
  const laneValues = processedRecordingIds.length > 0
    ? processedRecordingIds
    : job.rows.map((row) => row.speakerKey.trim()).filter(Boolean);
  const stableLaneIds = job.taskScoped ? [] : Array.from(new Set(laneValues)).sort();
  return JSON.stringify({
    version: 1,
    baseTaskId: job.jobId.trim(),
    stableLaneIds,
    ...(job.audioDependent !== false && job.audioVariantKey ? { audioVariantKey: job.audioVariantKey } : {})
  });
}

export function captureTranscriptJob(
  root: ParentNode = document,
  locationLike: Pick<Location, 'pathname' | 'search'> = window.location
): TranscriptJob {
  const documentRef = 'documentElement' in root ? root as Document : (root as Node).ownerDocument;
  if (!documentRef) throw new Error('Native transcript document is unavailable.');
  const requestId = crypto.randomUUID();
  const pending: { response?: TranscriptResponse } = {};
  const receive = (event: Event): void => {
    const value = parseTranscriptEvent(event) as TranscriptResponse | null;
    if (value?.version === 1 && value.requestId === requestId) pending.response = value;
  };
  documentRef.addEventListener(TRANSCRIPT_RESPONSE_EVENT, receive);
  try {
    dispatchTranscriptEvent(documentRef, TRANSCRIPT_REQUEST_EVENT, { version: 1, requestId, operation: 'capture' });
  } finally {
    documentRef.removeEventListener(TRANSCRIPT_RESPONSE_EVENT, receive);
  }
  const response = pending.response;
  if (!response) {
    // Audio identity remains available before transcript mount. This is not an
    // editable snapshot: apply always requires a live MAIN-world capture.
    const taskId = documentRef.documentElement?.getAttribute(PAGE_TASK_ID_ATTRIBUTE)?.trim();
    const audio = readAudioEnhancementState(documentRef);
    return {
      jobId: taskId || buildJobId(locationLike), rows: [], readOnly: true,
      ...(taskId ? { taskScoped: true } : {}),
      ...(taskId && audio?.taskId === taskId && audio.strength > 0 ? { audioVariantKey: audio.variantKey } : {})
    };
  }
  if (!response.ok) throw new Error(response.error);
  if (!('job' in response)) throw new Error('Invalid native transcript capture response.');
  return response.job;
}

export function applyDraftRows(
  draftRows: DraftRowResult[],
  expected: TranscriptJob,
  root: ParentNode = document
): Promise<ApplyDraftResult> {
  const documentRef = 'documentElement' in root ? root as Document : (root as Node).ownerDocument;
  if (!documentRef) return Promise.reject(new Error('Native transcript document is unavailable.'));
  const request: TranscriptRequest = {
    version: 1, requestId: crypto.randomUUID(), operation: 'apply', expected, rows: draftRows
  };
  const { promise, resolve, reject } = Promise.withResolvers<ApplyDraftResult>();
  const cleanup = (): void => {
    clearTimeout(timeout);
    documentRef.removeEventListener(TRANSCRIPT_RESPONSE_EVENT, receive);
  };
  const receive = (event: Event): void => {
    const response = parseTranscriptEvent(event) as TranscriptResponse | null;
    if (response?.version !== 1 || response.requestId !== request.requestId) return;
    cleanup();
    if (!response.ok) reject(new Error(response.error));
    else if ('result' in response) resolve(response.result);
    else reject(new Error('Invalid native transcript apply response.'));
  };
  const timeout = setTimeout(() => {
    cleanup();
    reject(new Error('Native transcript apply timed out. Check the current transcript before retrying.'));
  }, 30000);
  documentRef.addEventListener(TRANSCRIPT_RESPONSE_EVENT, receive);
  dispatchTranscriptEvent(documentRef, TRANSCRIPT_REQUEST_EVENT, request);
  return promise;
}

export function restoreCapturedRows(
  job: TranscriptJob,
  expected: TranscriptJob,
  root: ParentNode = document
): Promise<ApplyDraftResult> {
  expected = { ...expected, audioDependent: job.audioDependent };
  const currentById = new Map(expected.rows.map((row) => [row.rowId, row]));
  if (buildCanonicalTaskIdentity(job) !== buildCanonicalTaskIdentity(expected) ||
    job.rows.length !== expected.rows.length || job.rows.some((row) => {
      const current = currentById.get(row.rowId);
      return !current || current.processedRecordingId !== row.processedRecordingId ||
        current.speakerKey !== row.speakerKey || current.startSeconds !== row.startSeconds || current.endSeconds !== row.endSeconds;
    })) {
    return Promise.reject(new Error('The task or transcript structure changed. The original snapshot cannot be restored.'));
  }
  return applyDraftRows(job.rows.map((row) => ({
    rowId: row.rowId, rewrittenText: row.text, status: 'unchanged', warnings: []
  })), expected, root);
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
