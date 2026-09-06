import { test, expect } from '../../../../shared/babel-extension-platform/packages/babel-extension-e2e/src/test.mjs';

const ROW = 'textarea[placeholder^="What was said"]';
const WAND = '#babel-gold-drafting-magic-button';
const STATUS = '#babel-gold-drafting-overlay .bgd-status';
const PANEL = '.bgd-timing-hover-panel';
const texts = (page) => page.locator(ROW).evaluateAll((elements) => elements.map((element) => element.value));
const nativeAnnotations = (page) => page.evaluate(() => window.__BABEL_E2E__.snapshot().annotations);
const byAnnotationTime = (left, right) => left.startTimeInSeconds - right.startTimeInSeconds
  || left.endTimeInSeconds - right.endTimeInSeconds || left.id.localeCompare(right.id);
const annotationData = (rows) => rows.slice().sort(byAnnotationTime).map(({
  id, content, processedRecordingId, startTimeInSeconds, endTimeInSeconds
}) => ({ id, content, processedRecordingId, startTimeInSeconds, endTimeInSeconds }));
function speechAudio(babel) {
  if (babel.ai === 'placeholder') return {};
  expect(babel.hasSpeechFixtures, 'Real L0 success requires --speech-fixtures=DIR; synthetic tones are not speech.').toBe(true);
  return { audio: { fixture: 'speech' } };
}

async function configure(page, babel, settings = {}, overrides = {}) {
  await babel.setExtensionSettings('gold', {
    backendBaseUrl: babel.apiURL, l0CustomBaseUrl: babel.apiURL,
    openRouterApiKey: 'e2e-non-secret-admission-key', localModelsEnabled: false,
    l0ReplacementPreviewEnabled: true, l0DontRunLlm: true, ...settings,
  });
  const state = await babel.reset('baseline', overrides);
  await expect(page.locator(ROW)).toHaveCount(state.action.annotations.length);
  await expect(page.locator(WAND)).toBeVisible();
  return state;
}


// Replacement genuinely requires Helper. Keep Gold-only admission in its own
// installed-extension configuration, rather than masking it with a consumer.
test.describe('Gold without a replacement consumer', () => {
  test.use({ extensions: ['gold'] });
  test('default L0 replacement reports missing Helper promptly and preserves every native row @local-engine', async ({ page, babel }) => {
    await configure(page, babel);
    const original = await texts(page);
    await page.locator(WAND).click();
    await expect(page.locator(`${STATUS}[data-error="true"]`)).toContainText(/Helper/i, { timeout: 15_000 });
    await expect(page.locator(WAND)).toBeEnabled();
    await expect(page.getByRole('button', { name: 'Apply Draft', exact: true })).toBeDisabled();
    expect(await texts(page)).toEqual(original);
    expect((await babel.state()).calls.filter((call) => call.path.startsWith('/api/draft/'))).toEqual([]);
    expect((await babel.state()).calls.filter((call) => call.path === '/v1/draft')).toEqual([]);
  });
});

test.describe('Gold and Helper L0 integration', () => {
  test.use({ extensions: ['helper', 'gold'] });

  test('L0 replaces native rows and timings without running an optional LLM @local-engine', async ({ page, babel }) => {
    if (babel.ai !== 'placeholder') test.setTimeout(180_000);
    await page.addInitScript(() => {
      window.__goldL0Replacements = [];
      window.addEventListener('message', (event) => {
        if (event.source === window && event.data?.type === 'babel-gold-drafting:l0-replace-response'
          && event.data.version === 1 && event.data.ok === true) {
          window.__goldL0Replacements.push(event.data);
        }
      });
    });
    await configure(page, babel, {}, speechAudio(babel));
    const original = await texts(page);
    const originalIds = (await nativeAnnotations(page)).map((row) => row.id);
    await page.locator(WAND).click();
    await expect(page.locator(STATUS)).toContainText('L0 replacement complete', { timeout: babel.ai === 'placeholder' ? 60_000 : 150_000 });
    await expect(page.locator(STATUS)).toContainText('LLM drafting was skipped');
    const state = await babel.state();
    const replacement = state.calls.find((call) => call.path === '/v1/draft');
    expect(replacement).toBeDefined();
    expect(replacement.files).toHaveLength(2);
    expect(state.calls.filter((call) => call.path.startsWith('/api/draft/'))).toEqual([]);
    if (babel.ai === 'placeholder') await expect.poll(() => texts(page)).not.toEqual(original);
    await page.getByRole('button', { name: 'Close', exact: true }).click();
    const annotations = await nativeAnnotations(page);
    expect(annotations.every((row) => !originalIds.includes(row.id))).toBe(true);
    const replaced = await page.evaluate(() => window.__goldL0Replacements.at(-1));
    expect(replaced).toBeDefined();
    expect(annotations).toHaveLength(replacement.response.rows.length);
    for (const generated of replacement.response.rows) {
      const mapping = replaced.created.find((row) => row.id === generated.id);
      expect(mapping).toBeDefined();
      const created = annotations.find((row) => row.id === mapping.annotationId);
      expect(created).toBeDefined();
      expect(created.content).toBe(generated.text);
      expect(created.startTimeInSeconds).toBeCloseTo(generated.startSeconds, 2);
      expect(created.endTimeInSeconds).toBeCloseTo(generated.endSeconds, 2);
    }
    await page.getByRole('button', { name: 'Save progress', exact: true }).click();
    // Replacement changes native unsaved state; persistence belongs to Save.
    await expect.poll(async () => annotationData((await babel.state()).action.annotations))
      .toEqual(annotationData(annotations));
    expect(annotations.every((row) => Number.isFinite(row.startTimeInSeconds) && row.startTimeInSeconds < row.endTimeInSeconds)).toBe(true);
    expect(new Set(annotations.map((row) => row.id)).size).toBe(annotations.length);
    expect(new Set(annotations.map((row) => row.processedRecordingId))).toEqual(new Set(['speaker-1', 'speaker-2']));
    await page.reload();
    await expect.poll(() => page.evaluate(() => window.__BABEL_E2E__?.snapshot().ready === true)).toBe(true);
    await expect.poll(async () => annotationData(await nativeAnnotations(page))).toEqual(annotationData(annotations));
    await expect.poll(async () => (await texts(page)).sort()).toEqual(annotations.map((row) => row.content).sort());
  });

  test('L0 replacement can recapture its created rows before Gold stream apply', async ({ page, babel }) => {
    if (babel.ai !== 'placeholder') test.setTimeout(180_000);
    await configure(page, babel, { l0DontRunLlm: false, audioInputEnabled: false },
      speechAudio(babel));
    await page.locator(WAND).click();
    await expect(page.locator(STATUS)).toContainText('Draft ready.', { timeout: babel.ai === 'placeholder' ? 60_000 : 150_000 });
    const state = await babel.state();
    const request = state.calls.find((call) => call.path === '/api/draft/generate/stream');
    expect(request).toBeDefined();
    const annotations = await nativeAnnotations(page);
    const nativeIds = annotations.map((row) => row.id).sort();
    // Content-world capture can use its supported DOM locator IDs instead of
    // inaccessible page-world React IDs; verify the newly created native data.
    const byTime = (left, right) => left.startSeconds - right.startSeconds || left.endSeconds - right.endSeconds || left.text.localeCompare(right.text);
    expect(request.body.rows.map(({ text, startSeconds, endSeconds }) => ({ text, startSeconds, endSeconds })).sort(byTime))
      .toEqual(annotations.map((row) => ({
        text: row.content, startSeconds: row.startTimeInSeconds, endSeconds: row.endTimeInSeconds,
      })).sort(byTime));
    await page.getByRole('button', { name: 'Apply Draft', exact: true }).click();
    const draft = state.draftSessions[request.body.draftSessionId].result;
    const draftById = new Map(draft.draftRows.map((row) => [row.rowId, row.rewrittenText]));
    const expectedTexts = request.body.rows.map((row) => draftById.get(row.rowId) ?? row.text);
    await expect.poll(() => texts(page)).toEqual(expectedTexts);
    if (babel.ai === 'placeholder') expect(expectedTexts).not.toEqual(request.body.rows.map((row) => row.text));
    await page.locator(ROW).first().focus();
    expect(await page.locator(ROW).count()).toBe(nativeIds.length);
    await expect.poll(async () => (await nativeAnnotations(page)).map((row) => row.content).sort())
      .toEqual([...expectedTexts].sort());
    const applied = await nativeAnnotations(page);
    expect(applied.map((row) => row.id).sort()).toEqual(nativeIds);
    await page.getByRole('button', { name: 'Save progress', exact: true }).click();
    await expect.poll(async () => annotationData((await babel.state()).action.annotations))
      .toEqual(annotationData(applied));
    await page.reload();
    await expect.poll(() => page.evaluate(() => window.__BABEL_E2E__?.snapshot().ready === true)).toBe(true);
    await expect.poll(async () => annotationData(await nativeAnnotations(page))).toEqual(annotationData(applied));
    await expect.poll(async () => (await texts(page)).sort()).toEqual(applied.map((row) => row.content).sort());
  });

  test('queue progression publishes current per-lane timestamps only after real timing completion @local-engine', async ({ page, babel }) => {
    if (babel.ai !== 'placeholder') test.setTimeout(180_000);
    await page.addInitScript(() => {
      window.__goldTimingObservations = [];
      window.addEventListener('message', (event) => {
        if (event.source === window && event.data?.type === 'babel-gold-drafting:l0-timing-update') {
          window.__goldTimingObservations.push(event.data);
        }
      });
    });
    const state = await configure(page, babel, {}, {
      ...speechAudio(babel),
      controls: { routes: { '/v1/transcribe': { queuedMs: 1800, runningMs: 1800 } } }
    });
    await page.locator(WAND).hover();
    if (babel.ai === 'placeholder') {
      await expect(page.locator(PANEL)).toHaveAttribute('data-status', 'queued', { timeout: 15_000 });
      expect(await page.evaluate(() => window.__goldTimingObservations)).toEqual([]);
      await expect(page.locator(PANEL)).toHaveAttribute('data-status', 'running', { timeout: 15_000 });
    }
    await expect(page.locator(PANEL)).toHaveAttribute('data-status', 'available', { timeout: babel.ai === 'placeholder' ? 30_000 : 150_000 });
    const updates = await page.evaluate(() => window.__goldTimingObservations);
    expect(updates).toHaveLength(1);
    expect(updates[0].taskId).toBe(await page.locator(PANEL).getAttribute('data-task-id'));
    expect(updates[0].tracks).toHaveLength(2);
    const completed = await babel.state();
    const timing = completed.calls.find((call) => call.path === '/v1/transcribe' && call.outcome === 'success');
    expect(timing).toBeDefined();
    expect(updates[0].tracks).toEqual(timing.response.tracks);
    for (const call of completed.calls.filter((call) => call.path.startsWith('/v1/queue/') && call.outcome === 'success')) {
      const queued = call.response;
      expect(queued.requestId).toBe(decodeURIComponent(call.path.split('/').at(-1)));
      expect(['queued', 'running', 'completed']).toContain(queued.status);
      expect(Number.isInteger(queued.queuedCount) && queued.queuedCount >= 0).toBe(true);
      if (queued.status === 'queued') expect(Number.isInteger(queued.position) && queued.position > 0).toBe(true);
      else expect(queued.position).toBe(0);
    }
    for (const track of updates[0].tracks) {
      if (babel.ai === 'placeholder') expect(track.tokens.length).toBeGreaterThan(1);
      for (let index = 0; index < track.tokens.length; index += 1) {
        const token = track.tokens[index];
        expect(token.startSeconds).toBeGreaterThanOrEqual(0);
        expect(token.endSeconds).toBeGreaterThan(token.startSeconds);
        expect(token.endSeconds).toBeLessThanOrEqual(state.audio.duration);
        if (index > 0) expect(token.startSeconds).toBeGreaterThanOrEqual(track.tokens[index - 1].startSeconds);
      }
    }
  });

  test('failed automatic timing remains editable and manual regeneration recovers after retries exhaust @local-engine', async ({ page, babel }) => {
    test.setTimeout(babel.ai === 'placeholder' ? 90_000 : 210_000);
    await configure(page, babel, {}, {
      ...speechAudio(babel),
      controls: { routes: { '/v1/transcribe': { error: { status: 503, message: 'E2E timing unavailable' } } } }
    });
    await page.locator(WAND).hover();
    await expect(page.locator(PANEL)).toHaveAttribute('data-status', 'retrying', { timeout: 15_000 });
    await page.locator(ROW).first().fill('Редактор работает при ошибке таймингов.');
    await page.locator(ROW).nth(1).focus();
    await expect(page.locator(ROW).first()).toHaveValue('Редактор работает при ошибке таймингов.');
    await expect(page.locator(PANEL)).toHaveAttribute('data-status', 'unavailable', { timeout: 50_000 });
    await babel.control({ routes: {} });
    await page.locator(WAND).hover();
    await page.getByRole('button', { name: 'Regenerate timestamp data', exact: true }).click();
    await expect(page.locator(PANEL)).toHaveAttribute('data-status', 'available', { timeout: babel.ai === 'placeholder' ? 30_000 : 150_000 });
    await expect(page.locator(ROW).first()).toHaveValue('Редактор работает при ошибке таймингов.');
    await page.locator(WAND).hover();
    await expect(page.getByRole('button', { name: 'Regenerate timestamp data', exact: true })).toBeHidden();
  });

  test('pending L0 work cannot populate a newly navigated task @local-engine', async ({ page, babel, request }) => {
    if (babel.ai !== 'placeholder') test.setTimeout(180_000);
    const hold = 'stale-l0-draft';
    const audio = speechAudio(babel);
    await configure(page, babel, {}, {
      ...audio,
      controls: { routes: { '/v1/draft': { holdResponse: hold, times: 1 } } }
    });
    await page.locator(WAND).click();
    await expect.poll(async () => (await babel.state()).events.some((event) => event.type === 'response-held' && event.hold === hold), {
      timeout: babel.ai === 'placeholder' ? 15_000 : 150_000
    }).toBe(true);
    const nextId = '55555555-5555-4555-8555-555555555555';
    const state = await babel.state();
    const heldResponse = state.calls.find((call) => call.path === '/v1/draft' && call.outcome === 'held')?.response;
    expect(heldResponse).toBeDefined();
    const nextRows = state.action.annotations.map((row, index) => ({
      ...row, id: `next-row-${index}`, reviewActionId: nextId, content: `Новая задача ${index + 1}.`
    }));
    // Native submission navigates without replacing the document. Drop Claim
    // is admin-only and is not part of this ordinary worker's task surface.
    await page.getByRole('button', { name: 'Submit Review', exact: true }).focus();
    await page.keyboard.press('Enter');
    const submission = page.getByRole('dialog', { name: 'Confirm Submission' });
    await expect(submission).toBeVisible();
    await submission.getByRole('button', { name: 'Submit', exact: true }).focus();
    await page.keyboard.press('Enter');
    await expect(page).toHaveURL(/\/projects$/);
    await expect(page.locator('#babel-gold-drafting-overlay')).toBeHidden();
    const reset = await request.post(`${babel.apiURL}/__e2e__/reset`, {
      data: { scenario: 'baseline', overrides: { ...audio, action: { actionId: nextId, reviewActionId: nextId, annotations: nextRows } } },
    });
    expect(reset.ok()).toBeTruthy();
    await page.getByRole('link', { name: /RU-tx-gold/ }).click();
    await expect(page.locator('html')).toHaveAttribute('data-babel-review-action-id', nextId);
    await expect.poll(() => page.evaluate(() => window.__BABEL_E2E__.snapshot().reviewActionId)).toBe(nextId);
    await expect(page.locator('#babel-gold-drafting-overlay')).toBeHidden();
    const [staleResponse] = await Promise.all([
      page.waitForResponse((response) => new URL(response.url()).pathname === '/v1/draft'),
      babel.control({ release: { [hold]: true } })
    ]);
    await staleResponse.finished();
    expect(staleResponse.status()).toBe(200);
    expect(await staleResponse.json()).toEqual(heldResponse);
    await expect(page.locator(STATUS)).toHaveAttribute('data-error', 'true');
    await expect(page.getByRole('button', { name: 'Apply Draft', exact: true, includeHidden: true })).toBeDisabled();
    await expect.poll(async () => annotationData(await nativeAnnotations(page))).toEqual(annotationData(nextRows));
    await expect.poll(async () => (await texts(page)).sort()).toEqual(nextRows.map((row) => row.content).sort());
    await expect(page.locator(WAND)).toHaveCount(1);
    const nextState = await babel.state();
    expect(nextState.action.annotations.map((row) => row.content)).toEqual(nextRows.map((row) => row.content));
    await page.locator(ROW).first().fill('Новая задача остаётся редактируемой.');
    await page.locator(ROW).nth(1).focus();
    await expect(page.locator(ROW).first()).toHaveValue('Новая задача остаётся редактируемой.');
  });
});
