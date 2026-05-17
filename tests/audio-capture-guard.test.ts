import test from 'node:test';
import assert from 'node:assert/strict';
import { assessAudioCaptureForDrafting } from '../src/core/audio-capture-guard';
import type { CapturedAudioTrack, TranscriptJob } from '../src/core/types';

const twoSpeakerJob: TranscriptJob = {
  jobId: 'job-1',
  rows: [
    {
      rowId: 'r1',
      speakerKey: 'speaker-1',
      startSeconds: 0,
      endSeconds: 1,
      text: 'one',
      index: 0
    },
    {
      rowId: 'r2',
      speakerKey: 'speaker-2',
      startSeconds: 1,
      endSeconds: 2,
      text: 'two',
      index: 1
    }
  ]
};

function track(overrides: Partial<CapturedAudioTrack>): CapturedAudioTrack {
  return {
    trackId: overrides.trackId || 'audio-1',
    source: overrides.source || 'blob:https://dashboard.babel.audio/audio',
    blob: new Blob([new Uint8Array([1])], { type: 'audio/webm' }),
    mimeType: 'audio/webm',
    ...overrides
  };
}

test('assessAudioCaptureForDrafting flags generic audio tracks as missing speaker-lane audio', () => {
  const issue = assessAudioCaptureForDrafting(twoSpeakerJob, [track({ trackId: 'audio-1' })]);

  assert.deepEqual(issue, {
    kind: 'missing',
    expectedSpeakerLanes: 2,
    capturedSpeakerLanes: 0,
    capturedTracks: 1
  });
});

test('assessAudioCaptureForDrafting flags partial capture when one speaker lane is missing', () => {
  const issue = assessAudioCaptureForDrafting(twoSpeakerJob, [
    track({ trackId: 'speaker-1', speakerKey: 'speaker-1', trackLabel: 'Speaker 1' })
  ]);

  assert.deepEqual(issue, {
    kind: 'partial',
    expectedSpeakerLanes: 2,
    capturedSpeakerLanes: 1,
    capturedTracks: 1
  });
});

test('assessAudioCaptureForDrafting accepts one lane per transcript speaker', () => {
  const issue = assessAudioCaptureForDrafting(twoSpeakerJob, [
    track({ trackId: 'speaker-1', speakerKey: 'speaker-1', trackLabel: 'Speaker 1' }),
    track({ trackId: 'speaker-2', speakerKey: 'speaker-2', trackLabel: 'Speaker 2' })
  ]);

  assert.equal(issue, null);
});
