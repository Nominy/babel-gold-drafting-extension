import test from 'node:test';
import assert from 'node:assert/strict';
import { INFERENCE_RELEASE, assertReleasedGraphs } from '../src/core/inference-release';
import { hydratePunctuatedTiming, __localModelRuntimeTesting } from '../src/core/local-model-runtime';
import type { L0TimingResponse } from '../src/core/types';

test('required release rejects a stale graph before inference', () => {
  const files = Object.entries(INFERENCE_RELEASE.graphs).map(([path, sha256]) => ({ path, sha256 }));
  assert.doesNotThrow(() => assertReleasedGraphs(files));
  assert.throws(() => assertReleasedGraphs(files.slice(1)), /model update is required/);
  assert.throws(() => assertReleasedGraphs(files.map(file => ({ ...file, sha256: '0'.repeat(64) }))), /model update is required/);
});

test('another worker restores completed acoustic punctuation without re-inferring words', () => {
  const timing: L0TimingResponse = { taskId: 'portable-labels', summary: {}, models: { release: INFERENCE_RELEASE.id }, tracks: [{
    lane: 'speaker-1', pcmSha256: 'a'.repeat(64), sampleRate: 16000, punctuationLabels: [1, 3],
    tokens: [{ id: '1', text: 'ну', startSeconds: 0, endSeconds: 0.3 }, { id: '2', text: 'да', startSeconds: 0.4, endSeconds: 0.8 }],
    segments: [{ id: 'segment', startSeconds: 0, endSeconds: 1, startSample: 0, endSample: 16000, sampleRate: 16000 }]
  }] };
  const snapshot = structuredClone(timing);
  const lane = hydratePunctuatedTiming(timing).get('speaker-1')!;
  assert.equal(__localModelRuntimeTesting.renderCachedRange(lane, 0, 2), 'Ну, да?');
  assert.deepEqual(timing, snapshot);
  assert.throws(() => hydratePunctuatedTiming({ ...timing, models: { release: 'old' } }), /unavailable/);
  assert.throws(() => hydratePunctuatedTiming({ ...timing, tracks: [{ ...timing.tracks[0], punctuationLabels: [2] }] }), /unavailable/);
  assert.throws(() => hydratePunctuatedTiming({ ...timing, tracks: [{ ...timing.tracks[0], punctuationLabels: [1, 7] }] }), /unavailable/);
});
