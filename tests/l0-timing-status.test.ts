import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { L0TimingStatusPill } from '../src/content/l0-timing-status';

function createHarness() {
  const dom = new JSDOM('<!doctype html><html><head></head><body></body></html>');
  const scheduled = new Map<number, () => void>();
  let nextId = 1;
  const pill = new L0TimingStatusPill(
    dom.window.document,
    1_500,
    (callback) => {
      const id = nextId;
      nextId += 1;
      scheduled.set(id, callback);
      return id;
    },
    (id) => {
      scheduled.delete(id);
    }
  );
  return {
    dom,
    pill,
    scheduled,
    runNext() {
      const entry = scheduled.entries().next().value as [number, () => void] | undefined;
      assert.ok(entry, 'expected a scheduled status render');
      scheduled.delete(entry[0]);
      entry[1]();
    }
  };
}

test('fast timing completion clears delayed status before it flashes', () => {
  const harness = createHarness();
  harness.pill.update('task-a', { requestId: 'request-a', status: 'preparing' });
  assert.equal(harness.dom.window.document.querySelector('[role="status"]'), null);
  assert.equal(harness.scheduled.size, 1);

  harness.pill.clear('task-a');
  assert.equal(harness.scheduled.size, 0);
  assert.equal(harness.dom.window.document.querySelector('[role="status"]'), null);
});

test('status pill distinguishes queue position, running generation, and retry', () => {
  const harness = createHarness();
  harness.pill.update('task-a', {
    requestId: 'request-a',
    status: 'queued',
    position: 4,
    queuedCount: 4
  });
  harness.runNext();

  const status = harness.dom.window.document.querySelector<HTMLElement>('[role="status"]');
  assert.ok(status);
  assert.equal(status.textContent, 'In transcription queue · #4');
  assert.equal(status.getAttribute('aria-live'), 'polite');
  assert.equal(status.getAttribute('aria-atomic'), 'true');
  assert.ok(status.querySelector('.babel-gold-l0-timing-activity'));
  assert.match(harness.dom.window.document.head.textContent || '', /pointer-events: none/);
  assert.match(harness.dom.window.document.head.textContent || '', /bottom: 18px/);
  assert.doesNotMatch(harness.dom.window.document.head.textContent || '', /top: 50%/);

  harness.pill.update('task-a', {
    requestId: 'request-a',
    status: 'running',
    position: 0,
    queuedCount: 2
  });
  assert.equal(status.textContent, 'Generating word timing…');

  harness.pill.update('task-a', { status: 'retrying' });
  assert.equal(status.textContent, 'Retrying word timing…');
});

test('activating another task removes stale queue status', () => {
  const harness = createHarness();
  harness.pill.update('task-a', { requestId: 'request-a', status: 'preparing' });
  harness.runNext();
  assert.ok(harness.dom.window.document.querySelector('[role="status"]'));

  harness.pill.activateTask('task-b');
  assert.equal(harness.dom.window.document.querySelector('[role="status"]'), null);
  harness.pill.clear('task-a');
  assert.equal(harness.scheduled.size, 0);
});
