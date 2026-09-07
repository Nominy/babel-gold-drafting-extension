import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import {
  AUDIO_ENABLE_CAPTURE_MESSAGE_TYPE,
  AUDIO_FLUSH_REQUEST_MESSAGE_TYPE,
  AUDIO_RESPONSE_MESSAGE_TYPE,
  AUDIO_SOURCE_MESSAGE_TYPE,
  PAGE_TASK_ID_ATTRIBUTE,
  PAGE_TASK_ID_REQUEST_MESSAGE_TYPE,
  PAGE_TASK_ID_RESPONSE_MESSAGE_TYPE
} from '../src/core/audio-intercept-protocol';

function installEditor(dom: JSDOM, url: string) {
  const recordings = [{ processedRecordingId: 'speaker-1', speaker: 1, processedRecordingUrl: url }];
  const editor = { memoizedProps: { reviewActionId: 'task-a', transcriptionChunkProcessedRecordings: recordings } };
  const root: any = { child: editor };
  root.stateNode = { current: root };
  Object.assign(dom.window.document.querySelector('main')!, { __reactFiber$test: root });
  return recordings;
}

test('main-world audio interceptor stays dormant until the extension enables capture', async () => {
  const dom = new JSDOM('<main></main>', { url: 'https://dashboard.babel.audio/transcription/RU-transcription' });
  const originalFetch = (async () => new Response('ok')) as typeof fetch;
  dom.window.fetch = originalFetch;

  Object.assign(globalThis, {
    window: dom.window,
    document: dom.window.document,
    XMLHttpRequest: dom.window.XMLHttpRequest
  });

  // Module-load listeners must bind to this test's window, not the import-time global.
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

test('native waveform blob URLs retain their speaker lane mapping', async () => {
  const dom = new JSDOM('<main><div id="wave"></div></main>', {
    url: 'https://dashboard.babel.audio/transcription/RU-transcription'
  });
  const host = dom.window.document.getElementById('wave')!;
  host.attachShadow({ mode: 'open' }).innerHTML = '<div part="wrapper"></div>';
  const blobUrl = 'blob:https://dashboard.babel.audio/9f3dc4ef-a467-44b5-9551-c5311a49e317';
  installEditor(dom, blobUrl);
  Object.assign(globalThis, {
    window: dom.window,
    document: dom.window.document,
    HTMLElement: dom.window.HTMLElement,
    ShadowRoot: dom.window.ShadowRoot,
    XMLHttpRequest: dom.window.XMLHttpRequest
  });
  // Reload the side-effect entrypoint after installing this test's page world.
  const moduleUrl = new URL('../src/content/audio-request-interceptor.ts', import.meta.url);
  moduleUrl.search = '?case=native-blob-lane';
  await import(moduleUrl.href);
  const sources: Array<{ url: string; speakerKey?: string; trackLabel?: string }> = [];
  const flushed = new Promise<void>((resolve) => {
    dom.window.addEventListener('message', (event) => {
      if (event.data?.type === AUDIO_SOURCE_MESSAGE_TYPE) sources.push(event.data);
      if (event.data?.type === 'test:flush-complete') resolve();
    });
  });
  for (const type of [AUDIO_ENABLE_CAPTURE_MESSAGE_TYPE, AUDIO_FLUSH_REQUEST_MESSAGE_TYPE]) {
    dom.window.dispatchEvent(new dom.window.MessageEvent('message', {
      source: dom.window as unknown as MessageEventSource,
      data: { type }
    }));
  }
  dom.window.postMessage({ type: 'test:flush-complete' }, '*');
  await flushed;
  assert.deepEqual(sources.map(({ url, speakerKey, trackLabel }) => ({ url, speakerKey, trackLabel })), [
    { url: blobUrl, speakerKey: 'speaker-1', trackLabel: 'Speaker 1' }
  ]);
  dom.window.close();
});

test('native task identity survives deleting every annotation and follows the committed root', async () => {
  const dom = new JSDOM('<main><table><tbody><tr><td><textarea placeholder="What was said"></textarea></td></tr></tbody></table></main>', {
    url: 'https://dashboard.babel.audio/transcription/RU-transcription'
  });
  const body = dom.window.document.querySelector('tbody')!;
  const textarea = dom.window.document.querySelector('textarea')!;
  type NativeFiber = {
    memoizedProps?: { reviewActionId: string; transcriptionChunkProcessedRecordings: unknown[] };
    return?: NativeFiber;
    child?: NativeFiber;
    alternate?: NativeFiber;
    stateNode?: { current: NativeFiber };
  };
  const rootA: NativeFiber = {};
  const rootB: NativeFiber = {};
  const rootState = { current: rootA };
  rootA.stateNode = rootState;
  rootB.stateNode = rootState;
  const actionA: NativeFiber = { memoizedProps: { reviewActionId: 'task-a', transcriptionChunkProcessedRecordings: [] }, return: rootA };
  const actionB: NativeFiber = { memoizedProps: { reviewActionId: 'task-b', transcriptionChunkProcessedRecordings: [] }, return: rootB };
  actionA.alternate = actionB;
  actionB.alternate = actionA;
  rootA.child = actionA;
  rootB.child = actionB;
  const bodyA: NativeFiber = { return: actionA };
  const bodyB: NativeFiber = { return: actionB };
  bodyA.alternate = bodyB;
  bodyB.alternate = bodyA;
  actionA.child = bodyA;
  actionB.child = bodyB;
  const textareaFiber: NativeFiber = { return: bodyA };
  bodyA.child = textareaFiber;
  Object.assign(dom.window.document.querySelector('main')!, { __reactFiber$test: bodyA });
  Object.assign(textarea, { __reactFiber$test: textareaFiber });
  Object.assign(globalThis, {
    window: dom.window,
    document: dom.window.document,
    HTMLElement: dom.window.HTMLElement,
    XMLHttpRequest: dom.window.XMLHttpRequest
  });
  // The entrypoint must install its listener in this test's native page world.
  const moduleUrl = new URL('../src/content/audio-request-interceptor.ts', import.meta.url);
  moduleUrl.search = '?case=empty-native-action';
  await import(moduleUrl.href);
  dom.window.dispatchEvent(new dom.window.MessageEvent('message', {
    source: dom.window as unknown as MessageEventSource,
    data: { type: AUDIO_ENABLE_CAPTURE_MESSAGE_TYPE }
  }));
  let requestSequence = 0;
  const readTaskId = () => new Promise<string>((resolve) => {
    const requestId = `identity-${++requestSequence}`;
    const onMessage = (event: MessageEvent) => {
      if (event.data?.type !== PAGE_TASK_ID_RESPONSE_MESSAGE_TYPE || event.data.requestId !== requestId) return;
      dom.window.removeEventListener('message', onMessage);
      resolve(event.data.reviewActionId);
    };
    dom.window.addEventListener('message', onMessage);
    dom.window.dispatchEvent(new dom.window.MessageEvent('message', {
      source: dom.window as unknown as MessageEventSource,
      data: { type: PAGE_TASK_ID_REQUEST_MESSAGE_TYPE, requestId }
    }));
  });

  assert.equal(await readTaskId(), 'task-a');
  body.replaceChildren();
  delete bodyA.child;
  assert.equal(await readTaskId(), 'task-a');
  assert.equal(dom.window.document.documentElement.getAttribute(PAGE_TASK_ID_ATTRIBUTE), 'task-a');

  // React reuses the DOM anchor; its expando still points at task A's branch.
  rootState.current = rootB;
  assert.equal(await readTaskId(), 'task-b');
  assert.equal(dom.window.document.documentElement.getAttribute(PAGE_TASK_ID_ATTRIBUTE), 'task-b');

  // Missing committed editor state must not resurrect the stale alternate.
  delete rootB.child;
  assert.equal(await readTaskId(), '');
  assert.equal(dom.window.document.documentElement.hasAttribute(PAGE_TASK_ID_ATTRIBUTE), false);
  dom.window.close();
});

test('intercepted audio keeps its capture-time lane when the registry URL rotates', async () => {
  const dom = new JSDOM('<main><div id="wave"></div></main>', {
    url: 'https://dashboard.babel.audio/transcription/RU-transcription'
  });
  const host = dom.window.document.getElementById('wave')!;
  host.attachShadow({ mode: 'open' }).innerHTML = '<div part="wrapper"></div>';
  const signedUrl = 'https://audio.example.com/job-42/speaker-1.wav?X-Amz-Signature=first';
  const recordings = installEditor(dom, signedUrl);
  dom.window.fetch = (async () =>
    new Response(new Uint8Array([1, 2, 3, 4]), { headers: { 'content-type': 'audio/wav' } })) as typeof fetch;
  Object.assign(globalThis, {
    window: dom.window,
    document: dom.window.document,
    HTMLElement: dom.window.HTMLElement,
    ShadowRoot: dom.window.ShadowRoot,
    XMLHttpRequest: dom.window.XMLHttpRequest
  });
  const moduleUrl = new URL('../src/content/audio-request-interceptor.ts', import.meta.url);
  moduleUrl.search = '?case=capture-time-lane';
  await import(moduleUrl.href);
  const responses: Array<{ url: string; speakerKey?: string; trackLabel?: string }> = [];
  const captured = new Promise<void>((resolve) => {
    dom.window.addEventListener('message', (event) => {
      if (event.data?.type !== AUDIO_RESPONSE_MESSAGE_TYPE) return;
      responses.push(event.data);
      resolve();
    });
  });
  const post = (data: Record<string, unknown>) => dom.window.dispatchEvent(new dom.window.MessageEvent('message', {
    source: dom.window as unknown as MessageEventSource,
    data
  }));
  const flush = async () => {
    const done = new Promise<void>((resolve) => {
      const onMessage = (event: MessageEvent) => {
        if (event.data?.type !== 'test:flush-complete') return;
        dom.window.removeEventListener('message', onMessage);
        resolve();
      };
      dom.window.addEventListener('message', onMessage);
    });
    post({ type: AUDIO_FLUSH_REQUEST_MESSAGE_TYPE });
    dom.window.postMessage({ type: 'test:flush-complete' }, '*');
    await done;
  };

  post({ type: AUDIO_ENABLE_CAPTURE_MESSAGE_TYPE });
  await dom.window.fetch(signedUrl);
  // The interceptor reads the cloned body off the fetch promise chain.
  await captured;
  assert.deepEqual(responses.map(({ url, speakerKey, trackLabel }) => ({ url, speakerKey, trackLabel })), [
    { url: signedUrl, speakerKey: 'speaker-1', trackLabel: 'Speaker 1' }
  ]);
  responses.length = 0;
  await flush();
  assert.deepEqual(responses.map(({ url, speakerKey, trackLabel }) => ({ url, speakerKey, trackLabel })), [
    { url: signedUrl, speakerKey: 'speaker-1', trackLabel: 'Speaker 1' }
  ]);

  // The presigned URL rotated: the flush-time lookup misses, the lane stays.
  recordings[0].processedRecordingUrl = signedUrl.replace('first', 'second');
  responses.length = 0;
  await flush();
  assert.deepEqual(responses.map(({ url, speakerKey, trackLabel }) => ({ url, speakerKey, trackLabel })), [
    { url: signedUrl, speakerKey: 'speaker-1', trackLabel: 'Speaker 1' }
  ]);

  // A flush-time hit is authoritative and replaces the capture-time lane.
  recordings[0].processedRecordingUrl = signedUrl;
  recordings[0].speaker = 2;
  responses.length = 0;
  await flush();
  assert.deepEqual(responses.map(({ url, speakerKey, trackLabel }) => ({ url, speakerKey, trackLabel })), [
    { url: signedUrl, speakerKey: 'speaker-1', trackLabel: 'Speaker 2' }
  ]);
  dom.window.close();
});
