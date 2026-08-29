import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { AUDIO_ENABLE_CAPTURE_MESSAGE_TYPE } from '../src/core/audio-intercept-protocol';

test('main-world audio interceptor stays dormant until the extension enables capture', async () => {
  const dom = new JSDOM('<main></main>', { url: 'https://dashboard.babel.audio/transcription/RU-transcription' });
  const originalFetch = (async () => new Response('ok')) as typeof fetch;
  dom.window.fetch = originalFetch;

  Object.assign(globalThis, {
    window: dom.window,
    document: dom.window.document,
    XMLHttpRequest: dom.window.XMLHttpRequest
  });

  const moduleUrl = new URL('../src/content/audio-request-interceptor.ts', import.meta.url);
  moduleUrl.search = `?case=${Date.now()}`;
  await import(moduleUrl.href);

  assert.equal(dom.window.fetch, originalFetch);

  dom.window.dispatchEvent(
    new dom.window.MessageEvent('message', {
      source: dom.window as unknown as MessageEventSource,
      data: { type: AUDIO_ENABLE_CAPTURE_MESSAGE_TYPE }
    })
  );

  assert.notEqual(dom.window.fetch, originalFetch);
});
