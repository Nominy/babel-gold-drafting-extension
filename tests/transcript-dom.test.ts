import test, { type TestContext } from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { AUDIO_ENHANCEMENT_PROTOCOL_VERSION, AUDIO_ENHANCEMENT_STATE_ATTRIBUTE, type BabelNativeAnnotation, type BabelNativeTranscriptBindings } from '@nominy/babel-babel-runtime';
import { applyDraftRows, buildCanonicalTaskIdentity, buildDiffPreviewItems, buildJobId, captureTranscriptJob, restoreCapturedRows } from '../src/core/transcript';
import { registerTranscriptBridge } from '../src/content/transcript-bridge';
import { matchL0CreatedRows } from '../src/core/l0-created-row-matcher';
import type { DraftRowResult } from '../src/core/types';

type Command = { do(): void; undo(): void };
function installNative(t: TestContext) {
  const dom = new JSDOM('<main><section data-annotation-id="a"><div role="textbox" aria-label="Edit turn text">Rendered text is not authoritative</div></section></main>', {
    url: 'https://dashboard.babel.audio/transcription/RU-transcription?jobId=route-id'
  });
  const before = { window: globalThis.window, document: globalThis.document };
  Object.assign(globalThis, { window: dom.window, document: dom.window.document });
  let annotations: BabelNativeAnnotation[] = [
    { id: 'a', type: 'transcription', content: 'privet', processedRecordingId: 'lane-a', startTimeInSeconds: 1.25, endTimeInSeconds: 3.5, metadata: { untouched: true } },
    { id: 'b', type: 'transcription', content: 'da', processedRecordingId: 'lane-b', startTimeInSeconds: 3.5, endTimeInSeconds: 5.125, metadata: { lowConfidenceResolved: false } }
  ];
  const undoStack: Command[] = [];
  const redoStack: Command[] = [];
  const writes: string[] = [];
  let failId = '';
  const pendingTextById: Record<string, string> = Object.create(null);
  const history = {
    undoStack, redoStack, canUndo: false, canRedo: false,
    executeAndPush(command: Command) { command.do(); undoStack.push(command); redoStack.length = 0; },
    undo() { const command = undoStack.pop(); if (command) { command.undo(); redoStack.push(command); } },
    redo() { const command = redoStack.pop(); if (command) { command.do(); undoStack.push(command); } }
  };
  const onChangeAnnotationContent = (id: string, text: string) => {
    writes.push(id);
    if (id === failId) throw new Error('native content callback failed');
    const previous = annotations.find((row) => row.id === id)!;
    const next = { ...previous, content: text, metadata: { lowConfidenceResolved: true }, updatedAt: 'native-update' };
    history.executeAndPush({
      do() { annotations = annotations.map((row) => row.id === id ? next : row); delete pendingTextById[id]; },
      undo() { annotations = annotations.map((row) => row.id === id ? previous : row); delete pendingTextById[id]; }
    });
  };
  const bindings: BabelNativeTranscriptBindings = {
    reviewActionId: 'review-action-42', readOnly: false,
    get annotations() { return annotations; },
    pendingTextById,
    tracks: [{ id: 'lane-a', label: 'Same speaker label' }, { id: 'lane-b', label: 'Same speaker label' }],
    panel: {
      onChangeAnnotationContent,
      onSave() {}, onChangeAnnotationTrack() {}, onChangeAnnotationTime() {}, onDeleteAnnotation() {}, onMergeAnnotations() {}, onCreateAnnotation() {}
    },
    transcript: {
      get turns() { return annotations.slice(0, 1); },
      editingId: null,
      localEdits: false,
      handlers: {
        onChangeText: onChangeAnnotationContent,
        onActivate() { throw new Error('Unexpected native activation during transcript mutation'); },
        onDeactivate() { throw new Error('Unexpected native deactivation during transcript mutation'); },
        onSelect() { throw new Error('Unexpected native selection during transcript mutation'); }
      }
    },
    controller: null, history
  };
  const dispose = registerTranscriptBridge(dom.window.document, () => ({ ...bindings, history: { ...history, undoStack: [...undoStack] } }));
  t.after(() => { dispose(); dom.window.close(); Object.assign(globalThis, before); });
  return {
    dom, bindings, history, writes, pendingTextById,
    get annotations() { return annotations; },
    set annotations(value: BabelNativeAnnotation[]) { annotations = value; },
    fail(id: string) { failId = id; }
  };
}
const rewritten: DraftRowResult[] = [
  { rowId: 'a', rewrittenText: 'Привет.', status: 'rewritten', warnings: [] },
  { rowId: 'b', rewrittenText: 'Да.', status: 'rewritten', warnings: [] }
];

test('MAIN capture reads exact native identities and all lanes, not DOM labels or filtered turns', (t) => {
  const fixture = installNative(t);
  const job = captureTranscriptJob();
  assert.equal(job.jobId, 'review-action-42');
  assert.equal(job.taskScoped, true);
  assert.deepEqual(job.rows, [
    { rowId: 'a', speakerKey: 'lane-a', processedRecordingId: 'lane-a', startSeconds: 1.25, endSeconds: 3.5, text: 'privet', index: 0 },
    { rowId: 'b', speakerKey: 'lane-b', processedRecordingId: 'lane-b', startSeconds: 3.5, endSeconds: 5.125, text: 'da', index: 1 }
  ]);
  assert.equal(fixture.dom.window.document.querySelectorAll('textarea').length, 0);
  assert.equal(buildCanonicalTaskIdentity(job), '{"version":1,"baseTaskId":"review-action-42","stableLaneIds":[]}');
  const mapped = matchL0CreatedRows([{ id: 'engine-a', lane: 'lane-a', startSeconds: 1.25, endSeconds: 3.5, text: 'privet' }], [job.rows[0]!]);
  assert.equal(mapped[0].capturedRow.rowId, 'a');
});

test('empty and read-only native transcripts keep real task identity without editable DOM', (t) => {
  const fixture = installNative(t);
  fixture.bindings.readOnly = true;
  fixture.dom.window.document.querySelector('[role="textbox"]')!.remove();
  assert.equal(captureTranscriptJob().rows.length, 2);
  assert.equal(captureTranscriptJob().readOnly, true);
  fixture.annotations = [];
  assert.deepEqual(captureTranscriptJob(), { jobId: 'review-action-42', taskScoped: true, readOnly: true, rows: [] });
});

test('apply changes native state only and retains per-row native undo, redo and metadata', async (t) => {
  const fixture = installNative(t);
  const captured = captureTranscriptJob();
  const originals = structuredClone(fixture.annotations);
  assert.deepEqual(await applyDraftRows(rewritten, captured), { appliedCount: 2, missingRowIds: [] });
  assert.deepEqual(fixture.annotations.map((row) => row.content), ['Привет.', 'Да.']);
  assert.equal(fixture.dom.window.document.querySelector('[role="textbox"]')!.textContent, 'Rendered text is not authoritative');
  fixture.history.undo();
  assert.deepEqual(fixture.annotations[1], originals[1]);
  fixture.history.undo();
  assert.deepEqual(fixture.annotations, originals);
  fixture.history.redo();
  assert.equal(fixture.annotations[0].content, 'Привет.');
});

for (const boundary of ['task', 'text', 'speaker', 'start', 'end', 'deleted', 'added', 'readonly'] as const) {
  test(`apply rejects ${boundary} changes atomically before any native callback`, async (t) => {
    const fixture = installNative(t);
    const expected = captureTranscriptJob();
    if (boundary === 'task') fixture.bindings.reviewActionId = 'new-task';
    if (boundary === 'readonly') fixture.bindings.readOnly = true;
    if (boundary === 'text') fixture.annotations[1].content = 'a manual edit';
    if (boundary === 'speaker') fixture.annotations[1].processedRecordingId = 'lane-a';
    if (boundary === 'start') fixture.annotations[1].startTimeInSeconds = 4;
    if (boundary === 'end') fixture.annotations[1].endTimeInSeconds = 6;
    if (boundary === 'deleted') fixture.annotations = fixture.annotations.slice(0, 1);
    if (boundary === 'added') fixture.annotations = [...fixture.annotations, { ...fixture.annotations[1], id: 'new' }];
    const before = structuredClone(fixture.annotations);
    await assert.rejects(applyDraftRows(rewritten, expected), /changed|read-only/);
    assert.deepEqual(fixture.annotations, before);
    assert.deepEqual(fixture.writes, []);
  });
}

for (const rows of [[...rewritten, rewritten[0]], [...rewritten, { ...rewritten[0], rowId: 'missing' }]]) {
  test('duplicate or unknown draft rows cannot cause a partial apply', async (t) => {
    const fixture = installNative(t);
    await assert.rejects(applyDraftRows(rows, captureTranscriptJob()), /duplicate or unknown/);
    assert.deepEqual(fixture.writes, []);
  });
}

test('native callback failure rolls back text and metadata through native undo', async (t) => {
  const fixture = installNative(t);
  const captured = captureTranscriptJob();
  const originals = structuredClone(fixture.annotations);
  fixture.fail('b');
  await assert.rejects(applyDraftRows(rewritten, captured), /native content callback failed/);
  assert.deepEqual(fixture.annotations, originals);
  assert.equal(fixture.history.undoStack.length, 0);
  fixture.fail('');
  await applyDraftRows(rewritten, captured);
  assert.deepEqual(fixture.annotations.map((row) => row.content), ['Привет.', 'Да.']);
});

test('Restore Original intentionally reverts current text, but refuses changed timing and speaker structure', async (t) => {
  const fixture = installNative(t);
  const captured = captureTranscriptJob();
  fixture.annotations[0].content = 'explicitly discarded manual edit';
  await restoreCapturedRows(captured, captureTranscriptJob());
  assert.equal(fixture.annotations[0].content, 'privet');
  fixture.annotations[0].startTimeInSeconds = 0;
  const before = fixture.writes.length;
  await assert.rejects(restoreCapturedRows(captured, captureTranscriptJob()), /structure changed/);
  assert.equal(fixture.writes.length, before);
  const preview = buildDiffPreviewItems(captured.rows, rewritten);
  assert.deepEqual(preview.map(({ before, after }) => [before, after]), [['privet', 'Привет.'], ['da', 'Да.']]);
});

test('unchanged rows do not pollute undo or alter native metadata', async (t) => {
  const fixture = installNative(t);
  const job = captureTranscriptJob();
  await applyDraftRows(job.rows.map((row) => ({ rowId: row.rowId, rewrittenText: row.text, status: 'unchanged', warnings: [] })), job);
  assert.deepEqual(fixture.writes, []);
  assert.equal(fixture.history.undoStack.length, 0);
});

test('committed audio variants partition task cache identity and stale-source drafts cannot apply', async (t) => {
  const fixture = installNative(t);
  const root = fixture.dom.window.document.documentElement;
  const original = captureTranscriptJob();
  const originalIdentity = buildCanonicalTaskIdentity(original);
  const state = { version: AUDIO_ENHANCEMENT_PROTOCOL_VERSION, taskId: original.jobId, status: 'ready', strength: 100, desiredStrength: 100, revision: 1, variantKey: `zipenhancer:${'a'.repeat(64)}:${'b'.repeat(64)}` };
  const publish = () => root.setAttribute(AUDIO_ENHANCEMENT_STATE_ATTRIBUTE, JSON.stringify(state));
  publish();
  const enhanced = buildCanonicalTaskIdentity(captureTranscriptJob());
  assert.notEqual(enhanced, originalIdentity);
  await assert.rejects(applyDraftRows(rewritten, original), /audio source changed/);
  state.revision += 1;
  state.status = 'switching';
  state.desiredStrength = 35;
  publish();
  assert.equal(buildCanonicalTaskIdentity(captureTranscriptJob()), enhanced);
  state.status = 'ready'; state.strength = 35; state.variantKey = 'e'.repeat(64); publish();
  assert.notEqual(buildCanonicalTaskIdentity(captureTranscriptJob()), enhanced);
  state.strength = 0; state.variantKey = ''; publish();
  assert.equal(buildCanonicalTaskIdentity(captureTranscriptJob()), originalIdentity);
  state.taskId = 'different-task'; state.strength = 100; state.variantKey = 'f'.repeat(64); publish();
  assert.equal(buildCanonicalTaskIdentity(captureTranscriptJob()), originalIdentity);
});

test('route fallback and unscoped recording identities remain deterministic for pre-mount audio capture', () => {
  assert.equal(buildJobId({ pathname: '/fallback', search: '?jobId=%20%20&transcriptionChunkId=%20chunk-9%20&id=ignored' }), 'chunk-9');
  const rows = ['b', 'a', 'a'].map((lane, index) => ({ rowId: `${index}`, speakerKey: lane, processedRecordingId: lane, startSeconds: 0, endSeconds: 1, text: '', index }));
  assert.equal(buildCanonicalTaskIdentity({ jobId: ' route ', rows }), '{"version":1,"baseTaskId":"route","stableLaneIds":["a","b"]}');
});

test('apply waits for the committed native model rather than acknowledging queued callbacks', async (t) => {
  const fixture = installNative(t);
  const queued: Array<() => void> = [];
  fixture.bindings.panel.onChangeAnnotationContent = (id, text) => {
    const previous = fixture.annotations.find((row) => row.id === id)!;
    const next = { ...previous, content: text };
    // Zustand records the native command synchronously, while React commits
    // its queued model update later.
    fixture.history.executeAndPush({
      do() { queued.push(() => { fixture.annotations = fixture.annotations.map((row) => row.id === id ? next : row); }); },
      undo() { queued.push(() => { fixture.annotations = fixture.annotations.map((row) => row.id === id ? previous : row); }); }
    });
  };
  const expected = captureTranscriptJob();
  let completed = false;
  const pending = applyDraftRows(rewritten, expected).then((result) => { completed = true; return result; });
  await Promise.resolve();
  assert.equal(completed, false);
  assert.equal(fixture.annotations[0].content, 'privet');
  queued.forEach((commit) => commit());
  await pending;
  assert.equal(completed, true);
  assert.deepEqual(fixture.annotations.map((row) => row.content), ['Привет.', 'Да.']);
});

test('raw native tags, punctuation and whitespace survive capture and apply without DOM serialization', async (t) => {
  const fixture = installNative(t);
  fixture.annotations[0].content = '  [смех] <whisper>Да…</whisper>  ';
  const expected = captureTranscriptJob();
  assert.equal(expected.rows[0].text, '  [смех] <whisper>Да…</whisper>  ');
  const after = ' [смех] <whisper>Нет?!</whisper> ';
  await applyDraftRows([{ ...rewritten[0], rewrittenText: after }], expected);
  assert.equal(fixture.annotations[0].content, after);
  assert.equal(fixture.annotations[1].content, 'da');
});

test('live Slate raw text is captured without waiting for debounce or reading rendered tag labels', (t) => {
  const fixture = installNative(t);
  fixture.pendingTextById.a = '  Новый [смех] текст.  ';
  assert.equal(captureTranscriptJob().rows[0].text, '  Новый [смех] текст.  ');
  assert.equal(fixture.annotations[0].content, 'privet');
  assert.deepEqual(fixture.writes, []);
});

test('pending manual typing invalidates an older draft before the native model debounce', async (t) => {
  const fixture = installNative(t);
  const expected = captureTranscriptJob();
  fixture.pendingTextById.a = 'New live edit not committed yet.';
  await assert.rejects(applyDraftRows(rewritten, expected), /transcript changed/);
  assert.equal(fixture.annotations[0].content, 'privet');
  assert.deepEqual(fixture.writes, []);
});

test('apply lets captured pending typing commit first so native Undo restores that typing', async (t) => {
  const fixture = installNative(t);
  const typed = 'Новый [смех] текст.';
  fixture.pendingTextById.a = typed;
  const expected = captureTranscriptJob();
  const pending = applyDraftRows([rewritten[0]], expected);
  assert.deepEqual(fixture.writes, []);
  fixture.bindings.panel.onChangeAnnotationContent('a', typed);
  await pending;
  assert.equal(fixture.annotations[0].content, rewritten[0].rewrittenText);
  fixture.history.undo();
  assert.equal(fixture.annotations[0].content, typed);
  fixture.history.undo();
  assert.equal(fixture.annotations[0].content, 'privet');
});

test('navigation during the pending-edit commit wait prevents every draft mutation', async (t) => {
  const fixture = installNative(t);
  fixture.pendingTextById.a = 'Uncommitted edit.';
  const pending = applyDraftRows(rewritten, captureTranscriptJob());
  fixture.bindings.reviewActionId = 'different-native-task';
  await assert.rejects(pending, /task or audio source changed/);
  assert.deepEqual(fixture.writes, []);
});

test('native blur normalization requires a fresh capture instead of silently rewriting the pending snapshot', async (t) => {
  const fixture = installNative(t);
  fixture.pendingTextById.a = '  repeated   spaces  ';
  const pending = applyDraftRows(rewritten, captureTranscriptJob());
  fixture.bindings.panel.onChangeAnnotationContent('a', 'repeated spaces');
  await assert.rejects(pending, /transcript changed/);
  assert.deepEqual(fixture.writes, ['a']);
  assert.equal(fixture.annotations[0].content, 'repeated spaces');
});

test('rollback never undoes an unrelated same-depth command after native history ABA', async (t) => {
  const fixture = installNative(t);
  const originalChange = fixture.bindings.panel.onChangeAnnotationContent;
  fixture.bindings.panel.onChangeAnnotationContent = (id, text) => {
    if (id === 'a') return originalChange(id, text);
    fixture.history.undo();
    originalChange('a', 'New independent user edit.');
    throw new Error('Second native write failed after an independent command.');
  };
  await assert.rejects(applyDraftRows(rewritten, captureTranscriptJob()), /native history changed before rollback/);
  assert.equal(fixture.annotations[0].content, 'New independent user edit.');
  assert.equal(fixture.annotations[1].content, 'da');
  assert.equal(fixture.history.undoStack.length, 1);
  fixture.history.undo();
  assert.equal(fixture.annotations[0].content, 'privet');
});

test('a native callback throwing after its history push rolls back its complete command', async (t) => {
  const fixture = installNative(t);
  const original = structuredClone(fixture.annotations);
  const originalChange = fixture.bindings.panel.onChangeAnnotationContent;
  fixture.bindings.panel.onChangeAnnotationContent = (id, text) => {
    originalChange(id, text);
    if (id === 'b') throw new Error('Native callback failed after history push.');
  };
  await assert.rejects(applyDraftRows(rewritten, captureTranscriptJob()), /failed after history push/);
  assert.deepEqual(fixture.annotations, original);
  assert.equal(fixture.history.undoStack.length, 0);
});

test('same-depth history ABA during the native commit wait cannot roll back a newer user command', async (t) => {
  const fixture = installNative(t);
  const pending = applyDraftRows(rewritten, captureTranscriptJob());
  // The write callbacks ran synchronously; handle is awaiting its committed
  // snapshot acknowledgement. Replace its last command at the same stack depth.
  assert.equal(fixture.history.undoStack.length, 2);
  fixture.history.undo();
  fixture.bindings.panel.onChangeAnnotationContent('b', 'Independent edit during commit acknowledgement.');
  assert.equal(fixture.history.undoStack.length, 2);
  await assert.rejects(pending, /native history changed before rollback/);
  assert.equal(fixture.annotations[0].content, rewritten[0].rewrittenText);
  assert.equal(fixture.annotations[1].content, 'Independent edit during commit acknowledgement.');
  assert.equal(fixture.history.undoStack.length, 2);
  fixture.history.undo();
  assert.equal(fixture.annotations[1].content, 'da');
});

test('text-only drafts apply and restore across unused audio source changes without weakening row guards', async (t) => {
  const fixture = installNative(t);
  const captured = { ...captureTranscriptJob(), audioDependent: false };
  const identity = buildCanonicalTaskIdentity(captured);
  fixture.dom.window.document.documentElement.setAttribute(AUDIO_ENHANCEMENT_STATE_ATTRIBUTE, JSON.stringify({
    version: AUDIO_ENHANCEMENT_PROTOCOL_VERSION, taskId: captured.jobId, status: 'ready',
    strength: 90, desiredStrength: 90, revision: 1, variantKey: 'd'.repeat(64)
  }));
  assert.equal(buildCanonicalTaskIdentity({ ...captureTranscriptJob(), audioDependent: false }), identity);
  assert.notEqual(buildCanonicalTaskIdentity(captureTranscriptJob()), identity);
  await applyDraftRows(rewritten, captured);
  assert.deepEqual(fixture.annotations.map((row) => row.content), ['Привет.', 'Да.']);
  await restoreCapturedRows(captured, captureTranscriptJob());
  assert.deepEqual(fixture.annotations.map((row) => row.content), ['privet', 'da']);
  fixture.pendingTextById.a = 'A newer user edit.';
  const writes = fixture.writes.length;
  await assert.rejects(applyDraftRows(rewritten, captured), /transcript changed/);
  assert.equal(fixture.writes.length, writes);
});
