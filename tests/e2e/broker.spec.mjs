import { test, expect } from '../../../../shared/babel-extension-platform/packages/babel-extension-e2e/src/test.mjs';

const TYPE = 'babel-gold-drafting:ai-broker';
const PORT = 'babel-gold-drafting:ai-broker-port';
const ENDPOINT = '/api/broker/transcribe-segment';
const ROW = 'textarea[placeholder^="What was said"]';
const segment = { rowId: 'row-1', speakerKey: 'speaker-1', startSeconds: 0.5, endSeconds: 2.5 };
const request = (operation, fields = {}) => ({ type: TYPE, version: 1, operation, ...fields });

// Execute only the public Chrome transport in Helper's existing isolated world.
// No handler injection, service replacement, forged sender.tab, or fabricated result.
async function helperWorld(context, page, helperId) {
  const session = await context.newCDPSession(page);
  const ids = [];
  session.on('Runtime.executionContextCreated', ({ context: executionContext }) => ids.push(executionContext.id));
  await session.send('Runtime.enable');
  let world;
  for (const id of ids) {
    const result = await session.send('Runtime.evaluate', {
      contextId: id, expression: 'globalThis.chrome?.runtime?.id', returnByValue: true,
    });
    if (result.result.value === helperId) { world = id; break; }
  }
  expect(world, 'The actual installed Helper content world must be present').toBeDefined();
  return {
    async run(expression) {
      const result = await session.send('Runtime.evaluate', {
        contextId: world, expression, awaitPromise: true, returnByValue: true,
      });
      if (result.exceptionDetails) throw new Error(result.exceptionDetails.text);
      return result.result.value;
    },
    close: () => session.detach(),
  };
}
async function send(world, extensionId, payload) {
  return world.run(`chrome.runtime.sendMessage(${JSON.stringify(extensionId)}, ${JSON.stringify(payload)})`);
}
async function stream(world, extensionId, payload) {
  return world.run(`new Promise((resolve, reject) => {
    const port = chrome.runtime.connect(${JSON.stringify(extensionId)}, {name: ${JSON.stringify(PORT)}});
    const events = [];
    port.onMessage.addListener(message => {
      events.push(message);
      if (message.type === 'result' || message.type === 'error') {
        resolve(events); port.disconnect();
      }
    });
    port.onDisconnect.addListener(() => {
      if (!events.some(event => event.type === 'result' || event.type === 'error'))
        reject(new Error(chrome.runtime.lastError?.message || 'Broker disconnected before a terminal result'));
    });
    port.postMessage(${JSON.stringify(payload)});
  })`);
}
async function configure(page, babel, inference = false) {
  const useSpeech = inference && babel.ai !== 'placeholder';
  if (useSpeech && !babel.hasSpeechFixtures) {
    throw new Error('Real broker transcription requires --speech-fixtures=DIR with authorized two-speaker WAVs and aligned annotations.');
  }
  await babel.setExtensionSettings('gold', {
    backendBaseUrl: babel.apiURL, l0CustomBaseUrl: babel.apiURL,
    openRouterApiKey: 'e2e-non-secret-admission-key', aiBrokerProvider: 'remote-openrouter',
    l0ReplacementPreviewEnabled: false, localModelsEnabled: false,
  });
  const state = await babel.reset('baseline',
    useSpeech ? { audio: { fixture: 'speech' } } : {});
  await expect(page.locator(ROW)).toHaveCount(state.action.annotations.length);
  await expect(page.locator('html')).toHaveAttribute('data-babel-gold-drafting-extension-id', babel.extensionIds.gold);
  await expect.poll(() => page.evaluate(() => window.__BABEL_E2E__.snapshot().audio.ready)).toBe(true);
  expect(await page.evaluate(() => window.__BABEL_E2E__.snapshot().audio.tracks.length)).toBe(2);
  const row = state.action.annotations[0];
  return {
    rowId: row.id, speakerKey: row.processedRecordingId,
    startSeconds: row.startTimeInSeconds, endSeconds: row.endTimeInSeconds,
  };
}
async function chooseProvider(babel, provider, key) {
  const options = await babel.options('gold');
  await options.locator('#aiBrokerProvider').selectOption(provider);
  await options.locator('#openRouterApiKey').fill(key);
  await options.locator('[data-role="save"]').click();
  await expect(options.locator('[data-role="status"]')).toContainText('Saved.');
  await options.close();
}

test.describe('Gold public broker transports with real installed Helper', () => {
  test.use({ extensions: ['helper', 'gold'] });

  test('provider selection in options changes admission and fallback without backend egress', async ({ page, context, babel }) => {
    await configure(page, babel);
    const world = await helperWorld(context, page, babel.extensionIds.helper);
    try {
      await chooseProvider(babel, 'remote-openrouter', '');
      const remote = await send(world, babel.extensionIds.gold, request('transcribeSegment', { segment }));
      expect(remote).toMatchObject({ ok: false, reason: 'remote-not-configured', fallbackAllowed: false });
      await chooseProvider(babel, 'auto', '');
      const automatic = await stream(world, babel.extensionIds.gold, request('transcribeSegment', { segment }));
      expect(automatic.at(-1).response).toMatchObject({ ok: false, reason: 'remote-not-configured', fallbackAllowed: true });
      await chooseProvider(babel, 'local-gemini-nano', 'e2e-non-secret-admission-key');
      const local = await send(world, babel.extensionIds.gold, request('transcribeSegment', { segment }));
      expect(local).toMatchObject({ ok: false, reason: 'provider-local-gemini-nano', fallbackAllowed: true });
      const capabilities = await send(world, babel.extensionIds.gold, request('ping'));
      expect(capabilities.capabilities).toMatchObject({ transcribeSegment: false, redistributeText: false, transcribeSegmentL0: true });
      expect((await babel.state()).calls.filter((call) => call.path.startsWith('/api/broker/'))).toEqual([]);
    } finally { await world.close(); }
  });

  test('message and port requests traverse real worker, tab capture, and backend with ordered progress', async ({ page, context, babel }) => {
    const segment = await configure(page, babel, true);
    const world = await helperWorld(context, page, babel.extensionIds.helper);
    try {
      // Both real public transports capture the same loaded task concurrently.
      // Neither consumer may clear the other's speaker-lane audio.
      const [messageResult, events] = await Promise.all([
        send(world, babel.extensionIds.gold, request('transcribeSegment', { segment })),
        stream(world, babel.extensionIds.gold, request('transcribeSegment', { segment })),
      ]);
      expect(messageResult.ok).toBe(true);
      expect(messageResult.provider).toBe('remote-openrouter');
      const progress = events.filter((event) => event.type === 'event').map((event) => event.event);
      expect(progress.indexOf('accepted')).toBeGreaterThanOrEqual(0);
      expect(progress.indexOf('capturing-audio')).toBeGreaterThan(progress.indexOf('accepted'));
      expect(progress.indexOf('calling-backend')).toBeGreaterThan(progress.indexOf('capturing-audio'));
      expect(events.filter((event) => event.type === 'result')).toHaveLength(1);
      expect(events.at(-1).response).toMatchObject({ ok: true, provider: 'remote-openrouter' });
      const calls = (await babel.state()).calls.filter((call) => call.path === ENDPOINT);
      expect(calls).toHaveLength(2);
      for (const call of calls) {
        expect(call.files).toHaveLength(2);
        expect(new Set(call.files.map((file) => file.speakerKey)).size).toBe(2);
        for (const file of call.files) expect(file.size).toBeGreaterThan(44);
      }
      expect(calls.every((call) => call.body.segment.rowId === segment.rowId)).toBe(true);
      // No live inference wording is pinned: identity/routing and progress are the contract.
      expect(events.at(-1).response.text.trim()).not.toBe('');
    } finally { await world.close(); }
  });

  test('non-Babel options tabs and tabless workers cannot bypass admission or permit L0 fallback', async ({ page, context, babel }) => {
    await configure(page, babel);
    const helperOptions = await babel.options('helper');
    const remote = await helperOptions.evaluate(({ id, payload }) => chrome.runtime.sendMessage(id, payload), {
      id: babel.extensionIds.gold, payload: request('transcribeSegment', { segment }),
    });
    expect(remote).toMatchObject({ ok: false, reason: 'tab-broker-unavailable', fallbackAllowed: false });
    const l0 = await helperOptions.evaluate(({ id, payload }) => chrome.runtime.sendMessage(id, payload), {
      id: babel.extensionIds.gold,
      payload: request('transcribeSegmentL0', { taskId: 'not-current', row: { ...segment, text: '', index: 0 } }),
    });
    expect(l0).toMatchObject({ ok: false, reason: 'tab-broker-unavailable', fallbackAllowed: false });
    // An options page opened in a tab has a sender tab in real Chromium.
    // The installed Helper worker supplies a genuine tabless sender instead.
    const helperWorker = context.serviceWorkers().find((worker) =>
      worker.url().startsWith(`chrome-extension://${babel.extensionIds.helper}/`));
    expect(helperWorker, 'The actual installed Helper background worker must be available').toBeDefined();
    for (const payload of [
      request('transcribeSegment', { segment }),
      request('transcribeSegmentL0', { taskId: 'not-current', row: { ...segment, text: '', index: 0 } }),
    ]) {
      const result = await helperWorker.evaluate(({ id, payload }) => chrome.runtime.sendMessage(id, payload), {
        id: babel.extensionIds.gold, payload,
      });
      expect(result).toMatchObject({ ok: false, reason: 'missing-tab', fallbackAllowed: false });
    }
    expect((await babel.state()).calls.filter((call) => call.path === ENDPOINT || call.path === '/v1/draft')).toEqual([]);
    await helperOptions.close();
  });

  test('stale L0 task is refused before audio/backend work', async ({ page, context, babel }) => {
    await configure(page, babel);
    const world = await helperWorld(context, page, babel.extensionIds.helper);
    try {
      const before = (await babel.state()).calls.filter((call) => call.path === '/v1/draft').length;
      const result = await stream(world, babel.extensionIds.gold, request('transcribeSegmentL0', {
        taskId: 'a-different-review-action', row: { ...segment, text: '', index: 0 },
      }));
      expect(result.at(-1).response).toMatchObject({ ok: false, reason: 'stale-task', fallbackAllowed: false });
      expect(result.some((event) => event.event === 'calling-backend')).toBe(false);
      expect((await babel.state()).calls.filter((call) => call.path === '/v1/draft')).toHaveLength(before);
    } finally { await world.close(); }
  });

  test('disconnect during a pending port request does not poison the next real request', async ({ page, context, babel }) => {
    const segment = await configure(page, babel, true);
    await babel.control({ routes: { [ENDPOINT]: { delayMs: 1500, times: 1 } } });
    const world = await helperWorld(context, page, babel.extensionIds.helper);
    try {
      const disconnected = await world.run(`new Promise(resolve => {
        const port = chrome.runtime.connect(${JSON.stringify(babel.extensionIds.gold)}, {name:${JSON.stringify(PORT)}});
        port.onMessage.addListener(message => {
          if (message.event === 'calling-backend') { port.disconnect(); resolve(message.event); }
        });
        port.postMessage(${JSON.stringify(request('transcribeSegment', { segment }))});
      })`);
      expect(disconnected).toBe('calling-backend');
      const result = await stream(world, babel.extensionIds.gold, request('transcribeSegment', { segment }));
      expect(result.at(-1).response).toMatchObject({ ok: true, provider: 'remote-openrouter' });
      expect(result.filter((event) => event.type === 'result')).toHaveLength(1);
      await expect.poll(async () => (await babel.state()).calls.filter((call) => call.path === ENDPOINT).length).toBe(2);
    } finally { await world.close(); }
  });

  test('backend failure respects strict provider policy and a subsequent request recovers', async ({ page, context, babel }) => {
    const segment = await configure(page, babel, true);
    await babel.control({ routes: { [ENDPOINT]: { error: { status: 429, message: 'E2E broker capacity' }, times: 1 } } });
    const world = await helperWorld(context, page, babel.extensionIds.helper);
    try {
      const failed = await stream(world, babel.extensionIds.gold, request('transcribeSegment', { segment }));
      expect(failed.at(-1).response).toMatchObject({ ok: false, reason: 'broker-error', fallbackAllowed: false });
      expect(failed.at(-1).response.message).toContain('E2E broker capacity');
      const recovered = await send(world, babel.extensionIds.gold, request('transcribeSegment', { segment }));
      expect(recovered).toMatchObject({ ok: true, provider: 'remote-openrouter' });
    } finally { await world.close(); }
  });
});

test.describe('Gold runtime allowlist with all three products', () => {
  test.use({ extensions: ['helper', 'gold', 'review'] });

  test('Chrome refuses an unrelated installed extension before Gold broker dispatch', async ({ page, babel }) => {
    await configure(page, babel);
    const reviewOptions = await babel.options('review');
    const result = await reviewOptions.evaluate(async ({ id, payload }) => {
      try {
        return { response: await chrome.runtime.sendMessage(id, payload) };
      } catch (error) {
        return { error: error.message };
      }
    }, { id: babel.extensionIds.gold, payload: request('ping') });
    expect(result.response).toBeUndefined();
    expect(result.error).toMatch(/connect|receiving|access|allowed/i);
    expect((await babel.state()).calls.filter((call) => call.path.startsWith('/api/broker/'))).toEqual([]);
    await reviewOptions.close();
  });
});
