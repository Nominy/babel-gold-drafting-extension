import test from 'node:test';
import assert from 'node:assert/strict';
import { enhanceAudioTracks } from '../src/core/audio-enhancement-runtime';
import { reviewGraderAccess } from '../src/core/review-grader-access';

test('without Review Grader, direct enhancement cannot read source audio or initialize model/network resources', async t => {
  t.after(() => reviewGraderAccess().dispose());
  const blob = new Blob([new Uint8Array(44)], { type: 'audio/wav' });
  let sourceReads = 0, networkRequests = 0;
  t.mock.method(blob, 'arrayBuffer', async () => { sourceReads++; throw new Error('Source must remain unread'); });
  t.mock.method(globalThis, 'fetch', async () => { networkRequests++; throw new Error('No model or audio requests are permitted'); });
  await assert.rejects(enhanceAudioTracks([{ trackId: 'private-lane', source: 'original', blob, mimeType: 'audio/wav' }]), /operation is unavailable/);
  assert.equal(sourceReads, 0);
  assert.equal(networkRequests, 0);
});
