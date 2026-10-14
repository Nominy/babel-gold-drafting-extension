import { readAudioEnhancementState, readBabelTranscriptBindings, waitFor, type BabelNativeTranscriptBindings } from '@nominy/babel-babel-runtime';
import { buildCanonicalTaskIdentity } from '../core/transcript';
import { dispatchTranscriptEvent, parseTranscriptEvent, TRANSCRIPT_REQUEST_EVENT, TRANSCRIPT_RESPONSE_EVENT, type TranscriptRequest, type TranscriptResponse } from '../core/transcript-protocol';
import type { TranscriptJob, TranscriptRow } from '../core/types';

type Bindings = BabelNativeTranscriptBindings;
type ReadBindings = (documentRef: Document) => Bindings | null;

function capture(bindings: Bindings, documentRef: Document, includePendingText = true): TranscriptJob {
  const audio = readAudioEnhancementState(documentRef);
  const annotations = bindings.annotations.filter((row) => row.type === 'transcription')
    .slice().sort((a, b) => (a.startTimeInSeconds ?? 0) - (b.startTimeInSeconds ?? 0));
  return {
    jobId: bindings.reviewActionId, taskScoped: true, readOnly: bindings.readOnly,
    ...(audio?.taskId === bindings.reviewActionId && audio.strength > 0 ? { audioVariantKey: audio.variantKey } : {}),
    rows: annotations.map((row, index) => ({
      rowId: row.id,
      speakerKey: row.processedRecordingId ?? '',
      ...(row.processedRecordingId ? { processedRecordingId: row.processedRecordingId } : {}),
      startSeconds: row.startTimeInSeconds ?? null, endSeconds: row.endTimeInSeconds ?? null,
      text: (includePendingText ? bindings.pendingTextById[row.id] : undefined) ??
        (typeof row.content === 'string' ? row.content : ''), index
    }))
  };
}

function sameRow(left: TranscriptRow, right: TranscriptRow): boolean {
  return left.rowId === right.rowId && left.processedRecordingId === right.processedRecordingId &&
    left.speakerKey === right.speakerKey && left.startSeconds === right.startSeconds &&
    left.endSeconds === right.endSeconds && left.text === right.text;
}

// Install in MAIN at document_start, independently of audio capture opt-in.
export function registerTranscriptBridge(documentRef: Document = document, readBindings: ReadBindings = readBabelTranscriptBindings): () => void {
  let applying = false;
  const readCurrent = (): Bindings => {
    const bindings = readBindings(documentRef);
    if (!bindings?.reviewActionId) throw new Error('The native transcript is not ready. Wait for the task to load.');
    return bindings;
  };
  const requireExpected = (expected: TranscriptJob): Bindings => {
    const bindings = readCurrent();
    if (bindings.readOnly) throw new Error('This transcript is read-only.');
    const current = capture(bindings, documentRef);
    current.audioDependent = expected.audioDependent;
    if (!expected.taskScoped || buildCanonicalTaskIdentity(current) !== buildCanonicalTaskIdentity(expected)) {
      throw new Error('The task or audio source changed. Generate a new draft.');
    }
    const byId = new Map(current.rows.map((row) => [row.rowId, row]));
    if (byId.size !== current.rows.length || expected.rows.length !== current.rows.length ||
      expected.rows.some((row) => !byId.has(row.rowId) || !sameRow(row, byId.get(row.rowId)!))) {
      throw new Error('The transcript changed since capture. Generate a new draft before applying.');
    }
    return bindings;
  };
  const handle = async (request: Extract<TranscriptRequest, { operation: 'apply' }>): Promise<TranscriptResponse> => {
    if (applying) throw new Error('Another transcript update is still running.');
    let bindings = requireExpected(request.expected);
    const original = new Map(request.expected.rows.map((row) => [row.rowId, row]));
    const seen = new Set<string>();
    for (const row of request.rows) {
      if (!row || typeof row.rowId !== 'string' || typeof row.rewrittenText !== 'string' ||
        seen.has(row.rowId) || !original.has(row.rowId)) throw new Error('Draft contains duplicate or unknown rows. Nothing was applied.');
      seen.add(row.rowId);
    }
    const changed = request.rows.filter((row) => row.rewrittenText !== original.get(row.rowId)!.text);
    if (changed.length && request.expected.rows.some((row) => Object.hasOwn(bindings.pendingTextById, row.rowId))) {
      // A live Slate edit belongs in the native model/history before the draft
      // command. Otherwise Undo would discard that user's uncommitted typing.
      const committed = await waitFor(() => {
        const current = requireExpected(request.expected);
        const rows = capture(current, documentRef, false).rows;
        return rows.length === request.expected.rows.length &&
          rows.every((row, index) => sameRow(row, request.expected.rows[index]!)) ? current : null;
      }, 2000, 16);
      if (!committed) throw new Error('Finish editing the current turn before applying the draft.');
      if (applying) throw new Error('Another transcript update is still running.');
      bindings = requireExpected(request.expected);
    }
    const changeText = bindings.panel.onChangeAnnotationContent;
    if (typeof changeText !== 'function') throw new Error('Native transcript editing is unavailable.');
    const history = bindings.history;
    if (!history) throw new Error('Native transcript history is unavailable. Nothing was applied.');
    const previousCommands = history.undoStack;
    const appliedCommands: Array<{ do(): void; undo(): void }> = [];
    applying = true;
    try {
      // Native callbacks enqueue functional state/history updates. Keep the
      // whole write in one event turn so input/navigation cannot interleave.
      for (const row of changed) {
        const expectedDepth = previousCommands.length + appliedCommands.length;
        try {
          changeText(row.rowId, row.rewrittenText);
        } finally {
          // Capture even a command whose callback throws after committing it.
          const current = readCurrent();
          const stack = current.history?.undoStack;
          if (current.reviewActionId !== request.expected.jobId || current.readOnly || !stack ||
            (stack.length !== expectedDepth && stack.length !== expectedDepth + 1) ||
            previousCommands.some((command, index) => stack[index] !== command) ||
            appliedCommands.some((command, index) => stack[previousCommands.length + index] !== command)) {
            throw new Error('The native edit did not produce its expected history receipt.');
          }
          if (stack.length === expectedDepth + 1) appliedCommands.push(stack[stack.length - 1]!);
        }
        if (previousCommands.length + appliedCommands.length !== expectedDepth + 1) {
          throw new Error('The native edit did not create an undoable command.');
        }
      }
      const replacements = new Map(request.rows.map((row) => [row.rowId, row.rewrittenText]));
      const after: TranscriptJob = {
        ...request.expected,
        rows: request.expected.rows.map((row) => ({ ...row, text: replacements.get(row.rowId) ?? row.text }))
      };
      const committed = await waitFor(() => {
        const current = readCurrent();
        if (current.reviewActionId !== request.expected.jobId || current.readOnly) {
          throw new Error('The task changed during apply.');
        }
        const actual = capture(current, documentRef);
        return actual.rows.length === after.rows.length && actual.rows.every((row, index) => sameRow(row, after.rows[index]!))
          ? current : null;
      }, 2000, 16);
      if (!committed) throw new Error('The native editor did not commit the complete draft.');
      requireExpected(after);
      return { version: 1, requestId: request.requestId, ok: true, result: { appliedCount: request.rows.length, missingRowIds: [] } };
    } catch (error) {
      // Roll back through native history, restoring metadata as well as text.
      // Never undo commands from another task or a concurrent independent edit.
      const current = await waitFor(() => {
        const now = readCurrent();
        if (now.reviewActionId !== request.expected.jobId || now.readOnly) {
          throw new Error('The task changed before native rollback.');
        }
        return now.history && now.history.undoStack.length >= previousCommands.length + appliedCommands.length ? now : null;
      }, 2000, 16);
      if (!current) throw new Error('Native history did not commit before rollback. Check the transcript before retrying.');
      const stack = current.history?.undoStack;
      if (current.reviewActionId !== request.expected.jobId || current.readOnly || !stack ||
        stack.length !== previousCommands.length + appliedCommands.length ||
        previousCommands.some((command, index) => stack[index] !== command) ||
        appliedCommands.some((command, index) => stack[previousCommands.length + index] !== command)) {
        throw new Error(`Draft update failed; native history changed before rollback. ${error instanceof Error ? error.message : error}`);
      }
      for (let index = 0; index < appliedCommands.length; index += 1) current.history!.undo();
      const restored = await waitFor(() => {
        const now = capture(readCurrent(), documentRef);
        return now.jobId === request.expected.jobId && now.rows.length === request.expected.rows.length &&
          now.rows.every((row, index) => sameRow(row, request.expected.rows[index]!)) ? true : null;
      }, 2000, 16);
      if (!restored) throw new Error('Draft update failed and native rollback did not complete. Check the transcript before retrying.');
      throw error;
    } finally {
      applying = false;
    }
  };
  const receive = (event: Event): void => {
    const request = parseTranscriptEvent(event) as TranscriptRequest | null;
    if (!request || request.version !== 1 || typeof request.requestId !== 'string') return;
    const reply = (response: TranscriptResponse): void => dispatchTranscriptEvent(documentRef, TRANSCRIPT_RESPONSE_EVENT, response);
    const fail = (error: unknown): void => reply({ version: 1, requestId: request.requestId, ok: false, error: error instanceof Error ? error.message : String(error) });
    if (request.operation === 'capture') {
      try {
        const bindings = readBindings(documentRef);
        if (bindings) reply({ version: 1, requestId: request.requestId, ok: true, job: capture(bindings, documentRef) });
      }
      catch (error) { fail(error); }
    } else if (request.operation === 'apply' && Array.isArray(request.rows) && Array.isArray(request.expected?.rows)) {
      void handle(request).then(reply, fail);
    }
  };
  documentRef.addEventListener(TRANSCRIPT_REQUEST_EVENT, receive);
  return () => documentRef.removeEventListener(TRANSCRIPT_REQUEST_EVENT, receive);
}
