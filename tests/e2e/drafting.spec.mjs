import { test, expect } from '@nominy/babel-extension-e2e/test'
import { TEXT as ROW, fillText, editText, readTexts as values, saved, focusTranscript } from '@nominy/babel-extension-e2e/native-helpers';
import { expectNativeTexts } from './native-transcript.mjs';

const WAND = '#babel-gold-drafting-magic-button';
const OVERLAY = '#babel-gold-drafting-overlay';
const STREAM = '/api/draft/generate/stream';
const FINAL = '/api/draft/generate';
const draftCalls = (state) => state.calls.filter((call) => call.path === STREAM || call.path === FINAL);
function speechForInference(babel) {
  if (babel.ai === 'placeholder') return {};
  if (!babel.hasSpeechFixtures) {
    throw new Error('Real audio drafting requires --speech-fixtures=DIR with authorized two-speaker WAVs and aligned annotations.');
  }
  return { fixture: 'speech' };
}

async function completedDraft(babel) {
  const call = draftCalls(await babel.state()).findLast((candidate) =>
    candidate.outcome === 'success' && Array.isArray(candidate.response?.draftRows));
  expect(call, 'A completed provider draft must exist before native apply').toBeDefined();
  const results = new Map(call.response.draftRows.map((row) => [row.rowId, row]));
  return call.body.rows.map((row) => {
    const result = results.get(row.rowId);
    expect(result, `Provider result for captured row ${row.rowId}`).toBeDefined();
    return { before: row.text, after: result.rewrittenText, status: result.status };
  });
}

async function configure(page, babel, settings = {}, overrides = {}) {
  await babel.setExtensionSettings('gold', {
    mode: 'advanced',
    backendBaseUrl: babel.apiURL, l0CustomBaseUrl: babel.apiURL,
    openRouterApiKey: 'e2e-non-secret-admission-key', audioInputEnabled: false,
    l0ReplacementPreviewEnabled: false, localModelsEnabled: false, ...settings,
  });
  const state = await babel.reset('baseline', overrides);
  await expect(page.locator(ROW)).toHaveCount(state.action.annotations.length);
  await expect(page.locator(WAND)).toBeVisible();
}
async function ready(page) {
  await expect(page.locator(`${OVERLAY} .bgd-status`)).toContainText('Draft ready.', { timeout: 60_000 });
  await expect(page.getByRole('button', { name: 'Apply Draft', exact: true })).toBeEnabled();
}

for (const extensions of [['gold'], ['helper', 'gold', 'review']]) {
  test.describe(`Gold drafting with ${extensions.join('+')}`, () => {
    test.use({ extensions });

    test('streams a locked preview, restores the capture, and applies to native state', async ({ page, babel }) => {
      await configure(page, babel);
      const original = await values(page);
      await babel.control({ routes: { [STREAM]: { eventDelayMs: 500 } } });
      await page.locator(WAND).click();
      await expect(page.locator(`${OVERLAY} .bgd-status`)).toContainText(/rows complete/);
      await expect(page.getByRole('button', { name: 'Apply Draft', exact: true })).toBeDisabled();
      expect(await values(page)).toEqual(original);
      await ready(page);
      const preview = await page.locator(`${OVERLAY} .bgd-card .bgd-diff-pane:last-child .bgd-diff-content`).allTextContents();
      const result = await completedDraft(babel);
      expect(preview).toEqual(result.filter((row) => row.before !== row.after || row.status === 'failed').map((row) => row.after || '(empty)'));
      if (babel.ai === 'placeholder') expect(result.some((row) => row.before !== row.after)).toBe(true);
      expect(await values(page)).toEqual(original);
      // Restore acts on the captured snapshot while its preview is still open.
      // Do not assume that reopening the wand is a separate restore command.
      // The preview backdrop covers pointer access; keyboard focus still opens
      // the genuine native Slate editor without dismissing the captured draft.
      await page.locator(ROW).first().focus();
      await expect(page.locator(ROW).first()).toHaveAttribute('contenteditable', 'true');
      await fillText(page.locator(ROW).first(), 'Редактирование после захвата.');
      await page.locator(ROW).first().press('Enter');
      await expect(page.locator(ROW).first()).toHaveText('Редактирование после захвата.');
      await page.getByRole('button', { name: 'Restore Original', exact: true }).click();
      await expect.poll(() => values(page)).toEqual(original);
      await page.getByRole('button', { name: 'Apply Draft', exact: true }).click();
      await expect(page.locator(OVERLAY)).toBeHidden();
      await expectNativeTexts(page, result.map((row) => row.after));
      const applied = await values(page);
      // A normal native edit forces React to reconcile; a DOM-only setter cannot pass.
      await editText(page, 0, `${applied[0]} Проверка состояния.`);
    });

    test('retries reconciliation after a truncated stream without duplicating the draft session', async ({ page, babel }) => {
      await configure(page, babel);
      await babel.control({ routes: {
        [STREAM]: { eventDelayMs: 200, truncateAfterEvents: 2 },
        [FINAL]: { delayMs: 800, disconnect: true, times: 1 },
      } });
      await page.locator(WAND).click();
      await expect(page.locator(`${OVERLAY} .bgd-status`)).toContainText('Reconciling');
      await ready(page);
      const calls = draftCalls(await babel.state());
      expect(calls.filter((call) => call.path === STREAM)).toHaveLength(1);
      expect(calls.filter((call) => call.path === FINAL)).toHaveLength(2);
      expect(calls.filter((call) => call.path === FINAL).map((call) => call.outcome)).toEqual(['network-error', 'success']);
      expect(calls[0].body.draftSessionId).toBeTruthy();
      expect(new Set(calls.map((call) => call.body.draftSessionId))).toEqual(new Set([calls[0].body.draftSessionId]));
      const result = await completedDraft(babel);
      await page.getByRole('button', { name: 'Apply Draft', exact: true }).click();
      await expectNativeTexts(page, result.map((row) => row.after));
    });

    test('shows a backend error, leaves native rows intact, and allows a fresh retry', async ({ page, babel }) => {
      await configure(page, babel);
      const original = await values(page);
      await babel.control({ routes: { [STREAM]: { error: { status: 422, message: 'E2E drafting refused' }, times: 1 } } });
      await page.locator(WAND).click();
      await expect(page.locator(`${OVERLAY} .bgd-status[data-error="true"]`)).toContainText('E2E drafting refused');
      await expect(page.getByRole('button', { name: 'Apply Draft', exact: true })).toBeDisabled();
      expect(await values(page)).toEqual(original);
      await page.getByRole('button', { name: 'Close', exact: true }).click();
      await page.locator(WAND).click();
      await ready(page);
      const result = await completedDraft(babel);
      await page.getByRole('button', { name: 'Apply Draft', exact: true }).click();
      await expectNativeTexts(page, result.map((row) => row.after));
    });

    test('a terminal SSE error does not reconcile or expose a partially streamed draft for apply', async ({ page, babel }) => {
      await configure(page, babel);
      const original = await values(page);
      await babel.control({ routes: { [STREAM]: { eventDelayMs: 300, streamError: 'E2E terminal inference failure', times: 1 } } });
      await page.locator(WAND).click();
      await expect(page.locator(`${OVERLAY} .bgd-status[data-error="true"]`)).toContainText('E2E terminal inference failure');
      await expect(page.getByRole('button', { name: 'Apply Draft', exact: true })).toBeDisabled();
      expect(await values(page)).toEqual(original);
      expect(draftCalls(await babel.state()).filter((call) => call.path === FINAL)).toEqual([]);
      await page.getByRole('button', { name: 'Close', exact: true }).click();
      await page.locator(WAND).click();
      await ready(page);
      const result = await completedDraft(babel);
      await page.getByRole('button', { name: 'Apply Draft', exact: true }).click();
      await expectNativeTexts(page, result.map((row) => row.after));
    });

    test('mounts once after repeated task navigation and reload', async ({ page, babel }) => {
      await configure(page, babel);
      const route = page.url();
      for (const suffix of ['&e2eNavigation=1', '&e2eNavigation=2', '']) {
        await page.goto(`${route}${suffix}`);
        await expect(page.locator(WAND)).toHaveCount(1);
        await expect(page.locator(OVERLAY)).toHaveCount(1);
      }
      await page.reload();
      await expect(page.locator(WAND)).toHaveCount(1);
      await page.locator(WAND).click();
      await ready(page);
      expect(draftCalls(await babel.state()).filter((call) => call.path === STREAM)).toHaveLength(1);
    });
    test('a manual edit after preview is not overwritten by Apply Draft', async ({ page, babel }) => {
      await configure(page, babel);
      await page.locator(WAND).click();
      await ready(page);
      await page.locator(ROW).first().focus();
      await fillText(page.locator(ROW).first(), 'Новый ручной текст.');
      await page.locator(ROW).first().press('Enter');
      await page.getByRole('button', { name: 'Apply Draft', exact: true }).click();
      await expect(page.locator(`${OVERLAY} .bgd-status[data-error="true"]`)).toContainText('transcript changed');
      await expect(page.locator(OVERLAY)).toBeVisible();
      await expect.poll(() => page.evaluate(() => window.__BABEL_E2E__.snapshot().annotations[0].content)).toBe('Новый ручной текст.');
    });

    test('closing a pending stream cancels it and a new draft cannot receive stale callbacks', async ({ page, babel }) => {
      await configure(page, babel);
      const original = await values(page);
      await babel.control({ routes: { [STREAM]: { eventDelayMs: 1500, times: 1 } } });
      await page.locator(WAND).click();
      await expect(page.locator(`${OVERLAY} .bgd-status`)).toContainText(/rows complete/);
      await page.getByRole('button', { name: 'Close', exact: true }).click();
      await expect(page.locator(OVERLAY)).toBeHidden();
      await expect(page.locator(WAND)).toBeEnabled();
      expect(await values(page)).toEqual(original);
      await page.locator(WAND).click();
      await ready(page);
      await page.getByRole('button', { name: 'Apply Draft', exact: true }).click();
      await expect(page.locator(OVERLAY)).toBeHidden();
      const calls = draftCalls(await babel.state());
      expect(calls.filter((call) => call.path === FINAL)).toHaveLength(0);
      expect(calls.filter((call) => call.path === STREAM)).toHaveLength(2);
    });

    test('native undo restores every drafted row and native Save persists the applied model', async ({ page, babel }) => {
      await configure(page, babel);
      const original = await page.evaluate(() => window.__BABEL_E2E__.snapshot().annotations);
      await page.locator(WAND).click();
      await ready(page);
      const result = await completedDraft(babel);
      await page.getByRole('button', { name: 'Apply Draft', exact: true }).click();
      await expect(page.locator(OVERLAY)).toBeHidden();
      await expectNativeTexts(page, result.map((row) => row.after));
      await saved(page, babel);
      await expect.poll(async () => (await babel.state()).action.annotations.map((row) => row.content)).toEqual(result.map((row) => row.after));
      await focusTranscript(page);
      for (const row of result.filter((row) => row.before !== row.after)) {
        await page.keyboard.press('Control+z');
      }
      await expect.poll(() => page.evaluate(() => window.__BABEL_E2E__.snapshot().annotations)).toEqual(original);
    });

    test('read-only native tasks reject drafting before any provider request', async ({ page, babel }) => {
      await configure(page, babel);
      await babel.reset('readonly');
      await expect(page.locator(WAND)).toBeVisible();
      const original = await page.evaluate(() => window.__BABEL_E2E__.snapshot().annotations);
      await page.locator(WAND).click();
      await expect(page.locator(`${OVERLAY} .bgd-status[data-error="true"]`)).toContainText('read-only');
      expect(draftCalls(await babel.state())).toEqual([]);
      expect(await page.evaluate(() => window.__BABEL_E2E__.snapshot().annotations)).toEqual(original);
    });
  });
}

test.describe('Gold audio decisions', () => {
  test.use({ extensions: ['gold'] });

  for (const transport of ['fetch', 'xhr', 'blob']) {
  test(`captures both speaker lanes through native ${transport} audio delivery and submits binary audio`, async ({ page, babel }) => {
    const audioResponses = [];
    page.on('response', (response) => {
      if (response.headers()['content-type']?.startsWith('audio/')) {
        audioResponses.push({ url: response.url(), type: response.request().resourceType() });
      }
    });
    await configure(page, babel, { audioInputEnabled: true }, { audio: { ...speechForInference(babel), transport } });
    await page.locator(WAND).click();
    await ready(page);
    await expect(page.locator(`${OVERLAY} .bgd-audio-guard`)).toBeHidden();
    const call = draftCalls(await babel.state()).find((item) => item.path === STREAM);
    const capturedLanes = new Set(call.body.rows.map((row) => row.speakerKey));
    expect(capturedLanes.size).toBe(2);
    expect(new Set(call.files.map((track) =>
      capturedLanes.has(track.speakerKey) ? track.speakerKey : track.trackLabel))).toEqual(capturedLanes);
    expect(call.files).toHaveLength(2);
    for (const file of call.files) expect(file.size).toBeGreaterThan(44);
    if (transport === 'xhr') expect(audioResponses.some((response) => response.type === 'xhr')).toBe(true);
    if (transport === 'fetch') expect(audioResponses.some((response) => response.type === 'fetch')).toBe(true);
    if (transport === 'blob') {
      const nativeAudio = await page.evaluate(() => window.__BABEL_E2E__.snapshot().audio);
      expect(nativeAudio.ready).toBe(true);
      expect(new Set(nativeAudio.tracks.map((track) => track.id))).toEqual(
        new Set(call.files.map((file) => file.fieldName.slice('audioTrack:'.length))));
      for (const file of call.files) expect(file.source).toMatch(/^blob:/);
    }
  });
  }

  test('missing lane audio blocks generation until the user chooses text-only', async ({ page, babel }) => {
    await configure(page, babel, { audioInputEnabled: true }, { audio: { laneCount: 0 } });
    await page.locator(WAND).click();
    await expect(page.locator(`${OVERLAY} .bgd-audio-guard`)).toContainText('No speaker-lane audio was captured');
    expect(draftCalls(await babel.state())).toHaveLength(0);
    await expect(page.getByRole('button', { name: 'Use Captured Audio', exact: true })).toHaveCount(0);
    await page.getByRole('button', { name: 'Retry Audio', exact: true }).click();
    await expect(page.locator(`${OVERLAY} .bgd-audio-guard`)).toBeVisible();
    expect(draftCalls(await babel.state())).toHaveLength(0);
    await page.getByRole('button', { name: 'Continue Text Only', exact: true }).click();
    await ready(page);
    const call = draftCalls(await babel.state()).find((item) => item.path === STREAM);
    expect(call.files).toEqual([]);
  });

  test('partial audio requires an explicit decision and sends only the captured lane', async ({ page, babel }) => {
    await configure(page, babel, { audioInputEnabled: true }, { audio: { ...speechForInference(babel), laneCount: 1 } });
    await page.locator(WAND).click();
    await expect(page.locator(`${OVERLAY} .bgd-audio-guard`)).toContainText('Only 1 of 2 speaker lane(s)');
    expect(draftCalls(await babel.state())).toHaveLength(0);
    await page.getByRole('button', { name: 'Use Captured Audio', exact: true }).click();
    await ready(page);
    const call = draftCalls(await babel.state()).find((item) => item.path === STREAM);
    expect([call.files[0].speakerKey, call.files[0].trackLabel]).toContain(call.body.rows[0].speakerKey);
    expect(call.files).toHaveLength(1);
  });

  test('missing BYOK reports admission failure without sending a generation request', async ({ page, babel }) => {
    await configure(page, babel, { openRouterApiKey: '' });
    const original = await values(page);
    await page.locator(WAND).click();
    await expect(page.locator(`${OVERLAY} .bgd-status[data-error="true"]`)).toContainText('OpenRouter API key is required');
    expect(draftCalls(await babel.state())).toHaveLength(0);
    expect(await values(page)).toEqual(original);
  });
});
