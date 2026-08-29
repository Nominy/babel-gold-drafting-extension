import test from 'node:test';
import assert from 'node:assert/strict';
import {
  getL0TimingAvailability,
  publishL0TimingAvailability,
  requestL0TimingRegeneration,
  setL0TimingRetryHandler,
  subscribeL0TimingAvailability
} from '../src/content/l0-timing-availability';

test('timing availability publishes current task state and replays it to subscribers', () => {
  const received: string[] = [];
  publishL0TimingAvailability({ taskId: 'task-a', status: 'preparing' });
  const dispose = subscribeL0TimingAvailability((availability) => {
    received.push(`${availability.taskId}:${availability.status}`);
  });
  publishL0TimingAvailability({ taskId: 'task-a', status: 'available' });
  publishL0TimingAvailability({ taskId: 'task-a', status: 'available' });
  dispose();
  publishL0TimingAvailability({ taskId: 'task-b', status: 'unavailable' });

  assert.deepEqual(received, ['task-a:preparing', 'task-a:available']);
  assert.deepEqual(getL0TimingAvailability(), {
    taskId: 'task-b',
    status: 'unavailable'
  });
});

test('timing availability suppresses duplicate DOM-facing updates', () => {
  const positions: number[] = [];
  publishL0TimingAvailability({ taskId: 'task-queue', status: 'queued', position: 3 });
  const dispose = subscribeL0TimingAvailability((availability) => {
    if (availability.status === 'queued') positions.push(availability.position);
  });
  publishL0TimingAvailability({ taskId: 'task-queue', status: 'queued', position: 3 });
  publishL0TimingAvailability({ taskId: 'task-queue', status: 'queued', position: 2 });
  dispose();

  assert.deepEqual(positions, [3, 2]);
});

test('manual regeneration delegates only to the active retry handler', () => {
  let attempts = 0;
  setL0TimingRetryHandler(() => {
    attempts += 1;
    return attempts === 1;
  });
  assert.equal(requestL0TimingRegeneration(), true);
  assert.equal(requestL0TimingRegeneration(), false);
  assert.equal(attempts, 2);
  setL0TimingRetryHandler(null);
  assert.equal(requestL0TimingRegeneration(), false);
});
