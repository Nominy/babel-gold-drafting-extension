import test from 'node:test';
import assert from 'node:assert/strict';
import { isLikelyAudioSource } from '../src/core/audio-url';

test('isLikelyAudioSource does not treat the babel.audio hostname as audio by itself', () => {
  assert.equal(
    isLikelyAudioSource(
      'https://clerk.babel.audio/v1/environment?__clerk_api_version=2025-11-10&_clerk_js_version=5.125.10',
      'application/json'
    ),
    false
  );
  assert.equal(
    isLikelyAudioSource('https://dashboard.babel.audio/api/trpc/transcriptions.getReviewActionDataById?batch=1', ''),
    false
  );
});

test('isLikelyAudioSource accepts real audio responses and audio file paths', () => {
  assert.equal(isLikelyAudioSource('https://example.com/opaque', 'audio/wav'), true);
  assert.equal(
    isLikelyAudioSource(
      'https://davidai-audio-recordings.s3.us-east-2.amazonaws.com/transcription-chunks/prod/job/chunk/speaker-1.wav?X-Amz-Signature=test',
      ''
    ),
    true
  );
});
