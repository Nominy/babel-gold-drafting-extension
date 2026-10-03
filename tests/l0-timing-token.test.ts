import test from 'node:test';
import assert from 'node:assert/strict';
import { createL0TimingTokenHandler, type L0TimingTokenHandler } from '../src/background/l0-timing-token';
import {
  L0_TIMING_TOKEN_MESSAGE_TYPE,
  L0_TIMING_TOKEN_VERSION,
  isL0TimingTokenResponse,
  type L0TimingTokenRequest,
  type L0TimingTokenResponse
} from '../src/core/l0-timing-token-protocol';

const request: L0TimingTokenRequest = {
  type: L0_TIMING_TOKEN_MESSAGE_TYPE,
  version: L0_TIMING_TOKEN_VERSION,
  taskId: '{"version":1,"baseTaskId":"task-1","stableLaneIds":[]}'
};
const sender = { id: 'gold-extension' };

function send(handler: L0TimingTokenHandler, message = request) {
  return new Promise<L0TimingTokenResponse>((resolve, reject) => {
    if (!handler(JSON.parse(JSON.stringify(message)), sender, resolve)) reject(new Error('Request rejected'));
  });
}

test('concurrent tabs share one durable credential across delayed storage reads and writes and worker restart', async () => {
  const stored: Record<string, string> = {};
  const readStarted = Promise.withResolvers<void>();
  const readGate = Promise.withResolvers<void>();
  const writeStarted = Promise.withResolvers<void>();
  const writeGate = Promise.withResolvers<void>();
  let reads = 0;
  let writes = 0;
  const storage = {
    get: async (key: string) => {
      reads += 1;
      const snapshot = { [key]: stored[key] };
      readStarted.resolve();
      await readGate.promise;
      return snapshot;
    },
    set: async (items: Record<string, string>) => {
      writes += 1;
      writeStarted.resolve();
      await writeGate.promise;
      Object.assign(stored, items);
    }
  };
  const handler = createL0TimingTokenHandler(storage, sender.id);
  const first = send(handler);
  const second = send(handler);
  await readStarted.promise;
  assert.equal(reads, 1);
  readGate.resolve();
  await writeStarted.promise;
  const third = send(handler);
  let delivered = false;
  void third.then(() => { delivered = true; });
  await Promise.resolve();
  assert.equal(delivered, false, 'a credential is not usable until persisted');
  writeGate.resolve();
  const responses = await Promise.all([first, second, third]);
  assert.ok(responses.every((response) => isL0TimingTokenResponse(response, request) && response.ok));
  assert.deepEqual(responses, [responses[0], responses[0], responses[0]]);
  assert.equal(writes, 1);
  const restarted = createL0TimingTokenHandler(storage, sender.id);
  assert.deepEqual(await send(restarted), responses[0]);
  assert.equal(writes, 1);
  const another = await send(handler, { ...request, taskId: 'another-task' });
  assert.equal(another.ok, true);
  if (another.ok && responses[0].ok) assert.notEqual(another.token, responses[0].token);
});

test('failed persistence never hands out a credential and a later request can allocate successfully', async () => {
  const stored: Record<string, string> = {};
  let fail = true;
  const handler = createL0TimingTokenHandler({
    get: async () => stored,
    set: async (items) => {
      if (fail) throw new Error('storage unavailable');
      Object.assign(stored, items);
    }
  }, sender.id);
  const failed = await send(handler);
  assert.deepEqual(failed, { ...request, ok: false, error: 'storage unavailable' });
  fail = false;
  const response = await send(handler);
  assert.equal(response.ok, true);
  if (response.ok) assert.equal(Object.values(stored)[0], response.token);
});

test('credential messages reject other extensions and malformed requests without touching storage', () => {
  const handler = createL0TimingTokenHandler({
    get: async () => { assert.fail('invalid messages must not access storage'); },
    set: async () => { assert.fail('invalid messages must not write storage'); }
  }, sender.id);
  const respond = () => assert.fail('invalid messages must not receive credentials');
  for (const message of [null, [], { ...request, type: 'other' }, { ...request, version: 2 },
    { ...request, taskId: '' }, { ...request, taskId: '  ' }, { ...request, taskId: 42 }]) {
    assert.equal(handler(message, sender, respond), false);
  }
  assert.equal(handler(request, { id: 'other-extension' }, respond), false);
  assert.equal(handler(request, {}, respond), false);
  assert.equal(isL0TimingTokenResponse({ ...request, ok: true, token: 'a'.repeat(43), taskId: 'wrong' }, request), false);
  assert.equal(isL0TimingTokenResponse({ ...request, ok: true, token: 'not-a-capability' }, request), false);
});
