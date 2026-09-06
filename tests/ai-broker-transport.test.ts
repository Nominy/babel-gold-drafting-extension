import test from 'node:test';
import assert from 'node:assert/strict';
import { setImmediate as waitForImmediate } from 'node:timers/promises';
import { DEFAULT_SETTINGS, SETTINGS_STORAGE_KEY } from '../src/core/settings';
import {
  AI_BROKER_EXTERNAL_MESSAGE_TYPE,
  AI_BROKER_INTERNAL_MESSAGE_TYPE,
  AI_BROKER_PORT_NAME,
  AI_BROKER_INTERNAL_PORT_NAME
} from '../src/core/ai-broker-protocol';
import type { AiBrokerPortMessage, AiBrokerResponse } from '../src/core/ai-broker-protocol';
import { registerAiBrokerContentHandler } from '../src/content/ai-broker-content';

interface TestEvent<Args extends unknown[]> {
  addListener(listener: (...args: Args) => unknown): void;
  emit(...args: Args): unknown[];
}

interface TestPort {
  name: string;
  sender: chrome.runtime.MessageSender;
  messages: AiBrokerPortMessage[];
  onMessage: TestEvent<[unknown]>;
  onDisconnect: TestEvent<[]>;
  postMessage(message: AiBrokerPortMessage): void;
  disconnect(): void;
  readonly disconnected: boolean;
}

function event<Args extends unknown[]>(): TestEvent<Args> {
  const listeners: Array<(...args: Args) => unknown> = [];
  return {
    addListener(listener: (...args: Args) => unknown) { listeners.push(listener); },
    emit(...args: Args) { return listeners.map((listener) => listener(...args)); }
  };
}

function port(name: string, sender: chrome.runtime.MessageSender = {}): TestPort {
  const messages: AiBrokerPortMessage[] = [];
  const onMessage = event<[unknown]>();
  const onDisconnect = event<[]>();
  let disconnected = false;
  return {
    name, sender, messages, onMessage, onDisconnect,
    postMessage(message: AiBrokerPortMessage) {
      if (disconnected) throw new Error('Port closed');
      messages.push(message);
    },
    disconnect() {
      if (disconnected) return;
      disconnected = true;
      onDisconnect.emit();
    },
    get disconnected() { return disconnected; }
  };
}

test('broker transports preserve admission, failure policy and disconnect framing', async (t) => {
  const previousChrome = globalThis.chrome;
  const previousFetch = globalThis.fetch;
  const previousConsoleError = console.error;
  t.after(() => {
    Object.assign(globalThis, { chrome: previousChrome, fetch: previousFetch });
    console.error = previousConsoleError;
  });
  let settings = { ...DEFAULT_SETTINGS, openRouterApiKey: 'test-key' };
  let storageFailure = false;
  const onMessageExternal = event<[unknown, chrome.runtime.MessageSender, (response: AiBrokerResponse) => void]>();
  const onConnectExternal = event<[TestPort]>();
  const onMessage = event<[unknown, chrome.runtime.MessageSender, (response: AiBrokerResponse) => void]>();
  const onConnect = event<[TestPort]>();
  let tabPort = port(AI_BROKER_INTERNAL_PORT_NAME);
  Object.assign(globalThis, {
    chrome: {
      runtime: { onMessageExternal, onConnectExternal, onMessage, onConnect },
      storage: { local: { get(_key: string, callback: (items: object) => void) {
        if (storageFailure) throw new Error('Storage unavailable');
        callback({ [SETTINGS_STORAGE_KEY]: settings });
      } } },
      tabs: {
        sendMessage: async () => { throw new Error('Tab unavailable'); },
        connect: () => tabPort
      }
    }
  });
  // The service worker registers listeners at import time, after Chrome becomes available.
  await import('../src/background/ai-broker');
  registerAiBrokerContentHandler();
  const request = { type: AI_BROKER_EXTERNAL_MESSAGE_TYPE, version: 1, operation: 'redistributeText', groups: [], requestId: 'request-1' };
  const sender = { tab: { id: 12 } } as chrome.runtime.MessageSender;
  const sendMessage = (message: unknown, source = sender, internal = false) => new Promise<AiBrokerResponse>((resolve) => {
    const accepted = (internal ? onMessage : onMessageExternal).emit(message, source, resolve);
    assert.ok(accepted.includes(true));
  });
  const sendPort = async (message: unknown, source = sender, internal = false) => {
    const connection = port(internal ? AI_BROKER_INTERNAL_PORT_NAME : AI_BROKER_PORT_NAME, source);
    (internal ? onConnect : onConnectExternal).emit(connection);
    connection.onMessage.emit(message);
    await waitForImmediate();
    return connection;
  };
  const terminal = (connection: TestPort) => {
    const message = connection.messages.at(-1);
    assert.ok(message && message.type !== 'event');
    return message;
  };

  await t.test('ping and rejected admissions agree across transports, with accepted before port results', async () => {
    for (const scenario of [
      { provider: 'auto', key: '', operation: 'ping', reason: undefined, fallback: undefined },
      { provider: 'local-gemini-nano', key: '', operation: 'redistributeText', reason: 'provider-local-gemini-nano', fallback: true },
      { provider: 'auto', key: '', operation: 'redistributeText', reason: 'remote-not-configured', fallback: true },
      { provider: 'remote-openrouter', key: '', operation: 'redistributeText', reason: 'remote-not-configured', fallback: false },
      { provider: 'remote-openrouter', key: 'key', operation: 'redistributeText', reason: 'missing-tab', fallback: false },
      { provider: 'local-gemini-nano', key: '', operation: 'transcribeSegmentL0', reason: 'missing-tab', fallback: false }
    ] as const) {
      settings = { ...DEFAULT_SETTINGS, aiBrokerProvider: scenario.provider, openRouterApiKey: scenario.key };
      const message = { ...request, operation: scenario.operation };
      const response = await sendMessage(message, {});
      const connection = await sendPort(message, {});
      const accepted = connection.messages[0];
      assert.ok(accepted?.type === 'event');
      assert.equal(accepted.event, 'accepted');
      assert.deepEqual(terminal(connection).response, response);
      assert.equal(terminal(connection).type, response.ok ? 'result' : 'error');
      if (!response.ok) {
        assert.equal(response.reason, scenario.reason);
        assert.equal(response.fallbackAllowed, scenario.fallback);
      } else {
        assert.deepEqual(response, {
          ok: true, provider: 'auto', remoteConfigured: false,
          capabilities: { transcribeSegment: false, transcribeSegmentL0: true, redistributeText: false }
        });
      }
    }
  });

  await t.test('settings failure emits no accepted event and retains the transport error', async () => {
    storageFailure = true;
    const response = await sendMessage(request);
    const connection = await sendPort(request);
    assert.deepEqual(connection.messages, [{ type: 'error', response }]);
    assert.deepEqual(response, { ok: false, reason: 'broker-error', message: 'Storage unavailable', fallbackAllowed: true });
    storageFailure = false;
  });

  await t.test('tab failure keeps remote-only policy and disconnected forwarding emits one terminal error', async () => {
    settings = { ...DEFAULT_SETTINGS, aiBrokerProvider: 'remote-openrouter', openRouterApiKey: 'key' };
    const response = await sendMessage(request);
    assert.equal(response.ok, false);
    if (response.ok) throw new Error('Expected tab failure');
    assert.equal(response.reason, 'tab-broker-unavailable');
    assert.equal(response.fallbackAllowed, false);
    tabPort = port(AI_BROKER_INTERNAL_PORT_NAME);
    const connection = await sendPort(request);
    assert.deepEqual(tabPort.messages, [{ ...request, type: AI_BROKER_INTERNAL_MESSAGE_TYPE }]);
    tabPort.disconnect();
    const result = terminal(connection);
    assert.equal(result.type, 'error');
    assert.equal(result.response.ok, false);
    if (result.response.ok) throw new Error('Expected disconnect failure');
    assert.equal(result.response.reason, 'tab-broker-unavailable');
    assert.equal(result.response.fallbackAllowed, false);
    tabPort.onDisconnect.emit();
    assert.equal(connection.messages.filter((message) => message.type === 'error').length, 1);
  });

  await t.test('success and caller disconnect close the tab port without a synthetic error', async () => {
    tabPort = port(AI_BROKER_INTERNAL_PORT_NAME);
    const connection = await sendPort(request);
    const result = { type: 'result', response: { ok: true, provider: 'remote-openrouter', results: [], model: 'test' } } as const;
    tabPort.onMessage.emit(result);
    assert.equal(tabPort.disconnected, true);
    assert.deepEqual(connection.messages.slice(1), [result]);
    tabPort = port(AI_BROKER_INTERNAL_PORT_NAME);
    const closed = await sendPort(request);
    closed.disconnect();
    assert.equal(tabPort.disconnected, true);
    assert.equal(closed.messages.length, 1);
  });

  await t.test('content reloads policy after backend failure and logs once for either transport', async () => {
    const logged: unknown[][] = [];
    console.error = (...args) => { logged.push(args); };
    for (const internalPort of [false, true]) {
      for (const reloadFails of [false, true]) {
        settings = { ...DEFAULT_SETTINGS, aiBrokerProvider: 'auto', openRouterApiKey: 'key' };
        storageFailure = false;
        globalThis.fetch = async () => {
          settings.aiBrokerProvider = 'remote-openrouter';
          storageFailure = reloadFails;
          return new Response(JSON.stringify({ error: 'Backend rejected request' }), { status: 502 });
        };
        const message = { ...request, type: AI_BROKER_INTERNAL_MESSAGE_TYPE };
        const before = logged.length;
        const response = internalPort
          ? terminal(await sendPort(message, sender, true)).response
          : await sendMessage(message, sender, true);
        assert.deepEqual(response, {
          ok: false, reason: 'broker-error', message: 'Backend rejected request', fallbackAllowed: reloadFails
        });
        assert.equal(logged.length, before + 1);
      }
    }
    storageFailure = false;
  });

  await t.test('content rechecks provider policy independently and frames ordinary rejections as results', async () => {
    settings = { ...DEFAULT_SETTINGS, aiBrokerProvider: 'local-gemini-nano', openRouterApiKey: 'key' };
    globalThis.fetch = async () => { throw new Error('Backend must not be called'); };
    const message = { ...request, type: AI_BROKER_INTERNAL_MESSAGE_TYPE };
    const response = await sendMessage(message, sender, true);
    const result = terminal(await sendPort(message, sender, true));
    assert.equal(result.type, 'result');
    assert.deepEqual(result.response, response);
    assert.equal(response.ok, false);
    if (response.ok) throw new Error('Expected local provider rejection');
    assert.equal(response.reason, 'provider-local-gemini-nano');
    assert.equal(response.fallbackAllowed, true);
  });
});
