import { test, expect } from '../../../../shared/babel-extension-platform/packages/babel-extension-e2e/src/test.mjs';

const MODEL_FILE = '/browser-model/asr/v3_ctc.onnx';
const CACHE_PREFIX = 'babel-gold-local-models:bundle:';
const STATUS = '[data-role="local-model-status"]';
const download = (page) => page.locator('[data-role="local-model-download"]');
const remove = (page) => page.locator('[data-role="local-model-remove"]');
const sample = (page) => page.locator('[data-role="local-model-supplied-test"]');

async function openOptions(babel) {
  await babel.setExtensionSettings('gold', {
    backendBaseUrl: babel.apiURL, l0CustomBaseUrl: babel.apiURL,
    l0ReplacementPreviewEnabled: false, localModelsEnabled: false,
  });
  const options = await babel.options('gold');
  await expect(download(options)).toBeEnabled();
  return options;
}
async function install(options) {
  await download(options).click();
  await expect(options.locator(STATUS)).toContainText('Download complete and verified.', { timeout: 120_000 });
  await expect(remove(options)).toBeEnabled();
}
async function cachesIn(options) {
  return options.evaluate(async (prefix) => (await caches.keys()).filter((key) => key.startsWith(prefix)), CACHE_PREFIX);
}
async function offscreens(options) {
  return options.evaluate(() => chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] }));
}

test.describe('Gold actual options, cache, and offscreen controls', () => {
  test.use({ extensions: ['gold'] });

  test('saves normalized settings through the UI and retains them after options reload', async ({ babel }) => {
    const options = await openOptions(babel);
    await options.locator('#backendBaseUrl').fill(`${babel.apiURL}/`);
    await options.locator('#openRouterApiKey').fill('  e2e-non-secret-admission-key  ');
    await options.locator('#model').fill('e2e-model-choice');
    await options.locator('#serviceTier').selectOption('priority');
    await options.locator('#reasoningEffort').selectOption('high');
    await options.locator('#aiBrokerProvider').selectOption('auto');
    await options.locator('#audioInputEnabled').uncheck();
    await options.locator('#l0ReplacementPreviewEnabled').check();
    await expect(options.locator('[data-role="l0-replacement-settings"]')).toBeVisible();
    await options.locator('#l0CustomBaseUrl').fill(`${babel.apiURL}/?discard=query#discard`);
    await options.locator('#l0DontRunLlm').check();
    await options.locator('[data-role="save"]').click();
    await expect(options.locator('[data-role="status"]')).toContainText('Saved.');
    const granted = await options.evaluate((origin) => chrome.permissions.contains({ origins: [`${origin}/*`] }), new URL(babel.apiURL).origin);
    expect(granted).toBe(true);
    await options.reload();
    await expect(options.locator('#backendBaseUrl')).toHaveValue(babel.apiURL);
    await expect(options.locator('#l0CustomBaseUrl')).toHaveValue(babel.apiURL);
    await expect(options.locator('#openRouterApiKey')).toHaveValue('e2e-non-secret-admission-key');
    await expect(options.locator('#model')).toHaveValue('e2e-model-choice');
    await expect(options.locator('#serviceTier')).toHaveValue('priority');
    await expect(options.locator('#reasoningEffort')).toHaveValue('high');
    await expect(options.locator('#audioInputEnabled')).not.toBeChecked();
    await expect(options.locator('#l0DontRunLlm')).toBeChecked();
    await options.close();
  });

  test.describe('native permission denial', () => {
    test.use({ nativePermissionDialogs: true });
  test('denied optional host permission does not persist an unusable endpoint', async ({ babel }) => {
    const options = await openOptions(babel);
    const deniedOrigin = 'https://e2e-denied.invalid';
    const deniedPattern = `${deniedOrigin}/*`;
    // This HTTPS origin is optional, unlike the runner's required loopback host.
    // Fail before Save if staging or profile setup has accidentally pregranted it.
    expect(await options.evaluate((origin) => chrome.permissions.contains({ origins: [origin] }), deniedPattern)).toBe(false);
    await options.locator('#l0ReplacementPreviewEnabled').check();
    await options.locator('#l0CustomBaseUrl').fill(deniedOrigin);
    // Save must receive a real native denial; an ungranted origin alone is not
    // enough. Keep chrome.permissions and the user-gesture Save handler real.
    await options.locator('[data-role="save"]').click();
    await babel.cancelExtensionPermission('gold');
    await expect(options.locator('[data-role="status"][role="alert"]')).toContainText('Host access is required');
    expect(await options.evaluate((origin) => chrome.permissions.contains({ origins: [origin] }), deniedPattern)).toBe(false);
    await options.reload();
    await expect(options.locator('#l0CustomBaseUrl')).toHaveValue(babel.apiURL);
    await options.close();
  });
  });

  test('download progress disables unsafe controls and an integrity failure removes staged data', { tag: '@browser-models' }, async ({ page, babel }) => {
    const options = await openOptions(babel);
    await expect(options.locator('#localModelsEnabled')).toBeDisabled();
    await expect(sample(options)).toBeDisabled();
    await babel.control({ routes: { [MODEL_FILE]: { delayMs: 1000, corrupt: true, times: 1 } } });
    await download(options).click();
    await expect(options.locator('[data-role="local-model-progress"]')).toBeVisible();
    await expect(options.locator('[data-role="save"]')).toBeDisabled();
    await expect(download(options)).toBeDisabled();
    await expect(options.locator(STATUS)).toContainText(/SHA-256|size .*expected/, { timeout: 30_000 });
    await expect(options.locator(STATUS)).toHaveAttribute('role', 'alert');
    await expect(options.locator('#localModelsEnabled')).toBeDisabled();
    await expect(sample(options)).toBeDisabled();
    expect(await cachesIn(options)).toEqual([]);
    await babel.control({ routes: {} });
    await install(options);
    expect(await cachesIn(options)).toHaveLength(1);
    await options.close();
  });

  test('verified bundle survives reload, a failed update preserves it, and Remove deletes it', { tag: '@browser-models' }, async ({ page, babel }) => {
    const options = await openOptions(babel);
    // Placeholder mode serves explicitly marked synthetic bytes. This verifies
    // the real manifest/hash/cache install contract, not successful ONNX execution.
    await install(options);
    const initialCache = await cachesIn(options);
    expect(initialCache).toHaveLength(1);
    await options.reload();
    await expect(options.locator(STATUS)).toContainText('verified and cached');
    await expect(sample(options)).toBeEnabled();
    await expect(options.locator('#localModelsEnabled')).toBeDisabled();
    await babel.control({ routes: { [MODEL_FILE]: { error: { status: 503, message: 'E2E model supplier unavailable' }, times: 1 } } });
    await download(options).click();
    await expect(options.locator(STATUS)).toContainText('HTTP 503');
    expect(await cachesIn(options)).toEqual(initialCache);
    await options.reload();
    await expect(options.locator(STATUS)).toContainText('verified and cached');
    options.once('dialog', (dialog) => dialog.dismiss());
    await remove(options).click();
    await expect(remove(options)).toBeEnabled();
    expect(await cachesIn(options)).toEqual(initialCache);
    options.once('dialog', (dialog) => dialog.accept());
    await remove(options).click();
    await expect(options.locator(STATUS)).toContainText('removed');
    expect(await cachesIn(options)).toEqual([]);
    await expect(sample(options)).toBeDisabled();
    await expect(options.locator('#localModelsEnabled')).not.toBeChecked();
    await options.reload();
    await expect(options.locator(STATUS)).toContainText('Not downloaded');
    await options.close();
  });

  test('sample retrieval errors are actionable, allow retry, and never enable untested weights', { tag: '@browser-models' }, async ({ page, babel }) => {
    const options = await openOptions(babel);
    await install(options);
    await babel.control({ routes: { '/browser-model/sample-russian-15s.wav': { error: { status: 502, message: 'E2E sample offline' }, times: 1 } } });
    await sample(options).click();
    await expect(options.locator(STATUS)).toContainText('HTTP 502');
    await expect(options.locator('#localModelsEnabled')).toBeDisabled();
    await expect(sample(options)).toBeEnabled();
    await options.close();
  });

  test('rejects an overlong real WAV before inference and recovers its controls', { tag: '@browser-models' }, async ({ page, babel }) => {
    const options = await openOptions(babel);
    await install(options);
    const sampleRate = 8000;
    const frames = sampleRate * 16;
    const wav = Buffer.alloc(44 + frames * 2);
    wav.write('RIFF'); wav.writeUInt32LE(wav.length - 8, 4); wav.write('WAVEfmt ', 8);
    wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22);
    wav.writeUInt32LE(sampleRate, 24); wav.writeUInt32LE(sampleRate * 2, 28);
    wav.writeUInt16LE(2, 32); wav.writeUInt16LE(16, 34); wav.write('data', 36); wav.writeUInt32LE(frames * 2, 40);
    await options.locator('#localModelTestAudio').setInputFiles({ name: 'sixteen-seconds.wav', mimeType: 'audio/wav', buffer: wav });
    await options.locator('[data-role="local-model-test"]').click();
    await expect(options.locator(STATUS)).toContainText('no longer than 15 seconds');
    await expect(options.locator('#localModelsEnabled')).toBeDisabled();
    await expect(options.locator('[data-role="local-model-test"]')).toBeEnabled();
    await options.close();
  });

  test.describe('offscreen recovery with the required replacement consumer', () => {
    test.use({ extensions: ['helper', 'gold'] });
  test('persisted enabled but absent weights fail in a real offscreen document and recover after closure', { tag: '@browser-models' }, async ({ page, babel }) => {
    const options = await openOptions(babel);
    await babel.setExtensionSettings('gold', { localModelsEnabled: true, l0ReplacementPreviewEnabled: true, l0DontRunLlm: true });
    await babel.reset();
    await expect(page.locator('#babel-gold-drafting-magic-button')).toBeVisible();
    await page.locator('#babel-gold-drafting-magic-button').click();
    await expect(page.locator('#babel-gold-drafting-overlay .bgd-status[data-error="true"]')).toContainText(/model|cache|download/i, { timeout: 60_000 });
    await expect.poll(() => offscreens(options)).toHaveLength(1);
    const old = (await offscreens(options))[0].documentId;
    expect((await babel.state()).calls.filter((call) => call.path === '/v1/draft' || call.path.startsWith('/api/draft/'))).toEqual([]);
    await options.evaluate(() => chrome.offscreen.closeDocument());
    await expect.poll(() => offscreens(options)).toEqual([]);
    await page.getByRole('button', { name: 'Close', exact: true }).click();
    await page.locator('#babel-gold-drafting-magic-button').click();
    await expect(page.locator('#babel-gold-drafting-overlay .bgd-status[data-error="true"]')).toContainText(/model|cache|download/i, { timeout: 60_000 });
    await expect.poll(async () => (await offscreens(options)).map((item) => item.documentId)).not.toContain(old);
    await expect.poll(() => offscreens(options)).toHaveLength(1);
    await expect(page.getByRole('button', { name: 'Apply Draft', exact: true })).toBeDisabled();
    await options.close();
  });
  });
});

test.describe('Gold enabled browser models with real Helper replacement', () => {
  test.use({ extensions: ['helper', 'gold'] });

  test('tests an installed sample, enables models, and replaces native rows through offscreen inference', { tag: '@browser-models' }, async ({ page, babel }) => {
    test.setTimeout(180_000);
    if (babel.browserModels === 'real') {
      expect(babel.hasSpeechFixtures,
        'Real browser-model replacement requires explicit --speech-fixtures=DIR with speech WAVs and annotations; synthetic tones are not a transcription-success fixture.').toBe(true);
    }
    const options = await openOptions(babel);
    await install(options);
    await sample(options).click();
    await expect(options.locator(STATUS)).toContainText('Test succeeded', { timeout: 120_000 });
    if (babel.browserModels === 'placeholder') {
      await expect(options.locator(STATUS)).toContainText(/E2E|placeholder/i);
    }
    await expect(options.locator('#localModelsEnabled')).toBeEnabled();
    await options.locator('#localModelsEnabled').check();
    await options.locator('#l0ReplacementPreviewEnabled').check();
    await options.locator('#l0DontRunLlm').check();
    await options.locator('[data-role="save"]').click();
    await expect(options.locator('[data-role="status"]')).toContainText('Saved.');
    await options.reload();
    await expect(options.locator('#localModelsEnabled')).toBeChecked();
    const state = await babel.reset('baseline',
      babel.browserModels === 'real' ? { audio: { fixture: 'speech' } } : {});
    const rows = page.locator('textarea[placeholder^="What was said"]');
    await expect(rows).toHaveCount(state.action.annotations.length);
    await page.evaluate(() => {
      window.__goldModelReplacementObservations = [];
      window.addEventListener('message', (event) => {
        if (event.source === window && event.data?.version === 1
          && ['babel-gold-drafting:l0-replace-request', 'babel-gold-drafting:l0-replace-response'].includes(event.data?.type)) {
          window.__goldModelReplacementObservations.push(structuredClone(event.data));
        }
      });
    });
    await page.locator('#babel-gold-drafting-magic-button').click();
    await expect(page.locator('#babel-gold-drafting-overlay .bgd-status')).toContainText('L0 replacement complete', { timeout: 120_000 });
    const observations = await page.evaluate(() => window.__goldModelReplacementObservations);
    const requests = observations.filter((event) => event.type === 'babel-gold-drafting:l0-replace-request');
    expect(requests).toHaveLength(1);
    const request = requests[0];
    const response = observations.find((event) => event.type === 'babel-gold-drafting:l0-replace-response'
      && event.requestId === request.requestId);
    expect(response?.ok).toBe(true);
    expect(response.created).toHaveLength(request.rows.length);
    const originalIds = new Set(state.action.annotations.map(({ id }) => id));
    expect(response.created.every(({ annotationId }) => !originalIds.has(annotationId))).toBe(true);
    const expectedRows = response.created.map((created) => ({
      id: created.annotationId,
      content: request.rows.find((row) => row.id === created.id).text,
    })).sort((left, right) => left.id.localeCompare(right.id));
    // Consume the actual ONNX result, including a legitimate unchanged transcript.
    // A completion label alone does not prove Helper applied the returned rows.
    await expect.poll(() => page.evaluate(() => window.__BABEL_E2E__.snapshot().annotations
      .map(({ id, content }) => ({ id, content }))
      .sort((left, right) => left.id.localeCompare(right.id)))).toEqual(expectedRows);
    await expect.poll(() => offscreens(options)).toHaveLength(1);
    expect((await babel.state()).calls.filter((call) => call.path === '/v1/draft' || call.path.startsWith('/api/draft/'))).toEqual([]);
    await page.getByRole('button', { name: 'Close', exact: true }).click();
    await rows.first().focus();
    const nativeText = await rows.first().inputValue();
    await rows.first().fill(`${nativeText} Проверка.`);
    await rows.last().focus();
    await expect(rows.first()).toHaveValue(`${nativeText} Проверка.`);
    await options.close();
  });
});

test.describe('Gold installed-model inference failure boundary', () => {
  test.use({ extensions: ['gold'] });

  test('a hash-valid failing model reports inference failure, remains disabled, and can be reinstalled', { tag: '@browser-models' }, async ({ page, babel }) => {
    test.setTimeout(240_000);
    await babel.control({ models: { inferenceError: true } });
    const options = await openOptions(babel);
    await install(options);
    await sample(options).click();
    await expect(options.locator(STATUS)).toContainText('Local model test failed:', { timeout: 120_000 });
    await expect(options.locator('#localModelsEnabled')).toBeDisabled();
    await expect(sample(options)).toBeEnabled();
    // The verified artifact is still installed; inference failure must not
    // silently select remote inference or pretend the user enabled it.
    expect(await cachesIn(options)).toHaveLength(1);
    expect((await babel.state()).calls.filter((call) => call.path.startsWith('/api/draft/') || call.path === '/v1/draft')).toEqual([]);
    options.once('dialog', (dialog) => dialog.accept());
    await remove(options).click();
    await expect(options.locator(STATUS)).toContainText('removed');
    await babel.control({ models: {} });
    await install(options);
    await sample(options).click();
    await expect(options.locator(STATUS)).toContainText('Test succeeded', { timeout: 120_000 });
    await expect(options.locator('#localModelsEnabled')).toBeEnabled();
    await options.close();
  });
});
