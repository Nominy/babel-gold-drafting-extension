import type { AcousticWord } from './c-denoise-acoustic';
import type { WordRange } from './c-denoise-runtime';
const SAMPLE_RATE = 16_000;
export interface ActivitySegment { startSample: number; endSample: number }
const ACTIVITY_FRAME_SAMPLES = SAMPLE_RATE / 100;
const ACTIVITY_BRIDGE_FRAMES = 16;
const MINIMUM_ACTIVITY_FRAMES = 12;
const COARSE_SILENCE_FRAMES = 100;
const NOISE_FLOOR_PERCENTILE = 0.2;
const DBFS_FLOOR = 1 / 32_768;
const ACTIVITY_THRESHOLD_MARGIN_DB = 8;
const ACTIVITY_THRESHOLD_MIN_DBFS = -60;
const ACTIVITY_THRESHOLD_MAX_DBFS = -36;

function frameDbfs(samples: Float32Array, startSample: number, endSample: number): number {
  let squaredSum = 0;
  for (let index = startSample; index < endSample; index += 1) {
    const sample = samples[index];
    squaredSum += sample * sample;
  }
  const rms = Math.sqrt(squaredSum / (endSample - startSample));
  return 20 * Math.log10(Math.max(rms, DBFS_FLOOR));
}

function smoothActivity(activity: Uint8Array): void {
  let runStart = 0;
  while (runStart < activity.length) {
    const active = activity[runStart];
    let runEnd = runStart + 1;
    while (runEnd < activity.length && activity[runEnd] === active) runEnd += 1;
    if (
      active === 0 &&
      runStart > 0 &&
      runEnd < activity.length &&
      runEnd - runStart <= ACTIVITY_BRIDGE_FRAMES
    ) {
      activity.fill(1, runStart, runEnd);
    }
    runStart = runEnd;
  }

  runStart = 0;
  while (runStart < activity.length) {
    const active = activity[runStart];
    let runEnd = runStart + 1;
    while (runEnd < activity.length && activity[runEnd] === active) runEnd += 1;
    if (active === 1 && runEnd - runStart < MINIMUM_ACTIVITY_FRAMES) {
      activity.fill(0, runStart, runEnd);
    }
    runStart = runEnd;
  }
}

function appendTrimmedActivitySegment(
  segments: ActivitySegment[],
  activity: Uint8Array,
  sampleCount: number,
  leftSample: number,
  rightSample: number
): void {
  const firstFrame = Math.floor(leftSample / ACTIVITY_FRAME_SAMPLES);
  const frameEnd = Math.min(
    activity.length,
    Math.ceil(rightSample / ACTIVITY_FRAME_SAMPLES)
  );
  let firstActive = -1;
  let lastActive = -1;
  for (let frame = firstFrame; frame < frameEnd; frame += 1) {
    if (activity[frame] === 0) continue;
    if (firstActive < 0) firstActive = frame;
    lastActive = frame;
  }
  if (firstActive < 0) return;
  const startSample = Math.max(leftSample, firstActive * ACTIVITY_FRAME_SAMPLES);
  const endSample = Math.min(
    rightSample,
    sampleCount,
    (lastActive + 1) * ACTIVITY_FRAME_SAMPLES
  );
  if (endSample > startSample) segments.push({ startSample, endSample });
}

export function segmentSamplesByActivity(samples: Float32Array): ActivitySegment[] {
  if (!samples.length) return [];
  const frameCount = Math.ceil(samples.length / ACTIVITY_FRAME_SAMPLES);
  const frameDbfsValues = new Float64Array(frameCount);
  for (let frame = 0; frame < frameCount; frame += 1) {
    const startSample = frame * ACTIVITY_FRAME_SAMPLES;
    frameDbfsValues[frame] = frameDbfs(
      samples,
      startSample,
      Math.min(samples.length, startSample + ACTIVITY_FRAME_SAMPLES)
    );
  }
  const orderedDbfs = frameDbfsValues.slice();
  orderedDbfs.sort();
  const percentilePosition = (orderedDbfs.length - 1) * NOISE_FLOOR_PERCENTILE;
  const percentileLower = Math.floor(percentilePosition);
  const percentileUpper = Math.ceil(percentilePosition);
  const percentileFraction = percentilePosition - percentileLower;
  const noiseFloor =
    orderedDbfs[percentileLower] * (1 - percentileFraction) +
    orderedDbfs[percentileUpper] * percentileFraction;
  const threshold = Math.min(
    ACTIVITY_THRESHOLD_MAX_DBFS,
    Math.max(ACTIVITY_THRESHOLD_MIN_DBFS, noiseFloor + ACTIVITY_THRESHOLD_MARGIN_DB)
  );

  const activity = new Uint8Array(frameCount);
  for (let frame = 0; frame < frameCount; frame += 1) {
    if (frameDbfsValues[frame] >= threshold) activity[frame] = 1;
  }
  smoothActivity(activity);

  let firstActive = 0;
  while (firstActive < activity.length && activity[firstActive] === 0) firstActive += 1;
  if (firstActive === activity.length) return [];
  let lastActiveEnd = activity.length;
  while (lastActiveEnd > firstActive && activity[lastActiveEnd - 1] === 0) lastActiveEnd -= 1;

  const segments: ActivitySegment[] = [];
  let leftSample = firstActive * ACTIVITY_FRAME_SAMPLES;
  let frame = firstActive;
  while (frame < lastActiveEnd) {
    if (activity[frame] !== 0) {
      frame += 1;
      continue;
    }
    const silenceStart = frame;
    while (frame < lastActiveEnd && activity[frame] === 0) frame += 1;
    if (
      frame - silenceStart >= COARSE_SILENCE_FRAMES &&
      silenceStart > firstActive &&
      frame < lastActiveEnd
    ) {
      const rightSample =
        Math.floor((silenceStart + frame) / 2) * ACTIVITY_FRAME_SAMPLES;
      appendTrimmedActivitySegment(
        segments,
        activity,
        samples.length,
        leftSample,
        rightSample
      );
      leftSample = rightSample;
    }
  }
  appendTrimmedActivitySegment(
    segments,
    activity,
    samples.length,
    leftSample,
    Math.min(samples.length, lastActiveEnd * ACTIVITY_FRAME_SAMPLES)
  );
  return segments;
}


/** Audio activity owns fixed display boundaries; ASR cannot stretch rows into silence. */
export function buildActivityRows(words: readonly AcousticWord[], samples: Float32Array): Array<WordRange & ActivitySegment> {
  if (!words.length) return [];
  const activity = segmentSamplesByActivity(samples);
  const rows: Array<WordRange & ActivitySegment> = [];
  let owned = 0;
  for (const segment of activity) {
    const wordStart = owned;
    while (owned < words.length && (words[owned].startSeconds + words[owned].endSeconds) / 2 < segment.endSample / SAMPLE_RATE) {
      const word = words[owned];
      if (word.startSeconds < segment.startSample / SAMPLE_RATE - 1e-7 || word.endSeconds > segment.endSample / SAMPLE_RATE + 1e-7) {
        throw new Error('ASR words must be recognized inside their audio activity segment; silence cannot extend a row.');
      }
      owned++;
    }
    if (owned > wordStart) rows.push({ ...segment, wordStart, wordEnd: owned });
  }
  if (owned !== words.length) throw new Error('ASR words fall outside audio activity segments.');
  return rows;
}
