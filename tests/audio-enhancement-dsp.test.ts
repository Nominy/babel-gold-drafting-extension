import test from 'node:test';
import assert from 'node:assert/strict';
import {
  decodeEnhancementWav,
  encodeEnhancementWav,
  enhanceZipSamples,
  readEnhancementWavHeader,
  ZIP_SAMPLE_RATE,
  ZIP_STRIDE_SAMPLES,
  ZIP_CHUNK_SAMPLES,
  ZIP_HOP_SIZE,
  zipStft,
  createZipStftSequence
} from '../src/core/audio-enhancement-dsp';

test('native decoded float32 fallback preserves frame clock and quiet speech precision', () => {
  const frames = [[0.00000123, -0.00000234], [0.25, 0.5], [-0.75, 0.25]];
  const bytes = new ArrayBuffer(44 + frames.length * 8), view = new DataView(bytes);
  view.setUint32(0, 0x52494646, false); view.setUint32(4, bytes.byteLength - 8, true);
  view.setUint32(8, 0x57415645, false); view.setUint32(12, 0x666d7420, false);
  view.setUint32(16, 16, true); view.setUint16(20, 3, true); view.setUint16(22, 2, true);
  view.setUint32(24, 48000, true); view.setUint32(28, 48000 * 8, true);
  view.setUint16(32, 8, true); view.setUint16(34, 32, true);
  view.setUint32(36, 0x64617461, false); view.setUint32(40, frames.length * 8, true);
  for (let frame = 0; frame < frames.length; frame++) {
    for (let channel = 0; channel < 2; channel++) view.setFloat32(44 + frame * 8 + channel * 4, frames[frame][channel], true);
  }
  const decoded = decodeEnhancementWav(bytes);
  assert.equal(decoded.sampleRate, 48000);
  assert.deepEqual(Array.from(decoded.samples), frames.map(row => Math.fround((Math.fround(row[0]) + Math.fround(row[1])) / 2)));
  const output = encodeEnhancementWav(decoded.samples, decoded.sampleRate, frames.length);
  assert.equal(new DataView(output.buffer).getUint32(40, true), frames.length * 2);
  view.setFloat32(44, Number.NaN, true);
  const header = readEnhancementWavHeader(bytes);
  assert.equal(header.sampleRate, 48000);
  assert.equal(header.frameCount, frames.length);
  assert.equal(header.channels, 2);
  assert.equal(header.format, 3);
  assert.equal('samples' in header, false, 'cache clock lookup must not decode or allocate original PCM samples');
  assert.throws(() => decodeEnhancementWav(bytes), /nonfinite/);
});

test('adjacent STFT reuse is bit-exact through reflected boundaries, quiet samples and transferred outputs', () => {
  const lane = Float32Array.from({ length: ZIP_CHUNK_SAMPLES + 2 * ZIP_STRIDE_SAMPLES }, (_, index) =>
    Math.fround((index % 997 < 31 ? 0.00000013 : 0.21) * Math.sin(index / 13) + 0.00002 * Math.cos(index / 79)));
  const sequence = createZipStftSequence();
  for (let start = 0; start + ZIP_CHUNK_SAMPLES <= lane.length; start += ZIP_STRIDE_SAMPLES) {
    const chunk = lane.subarray(start, start + ZIP_CHUNK_SAMPLES);
    const expected = zipStft(chunk), actual = sequence(chunk);
    assert.equal(actual.frames, ZIP_CHUNK_SAMPLES / ZIP_HOP_SIZE + 1);
    assert.deepEqual(new Uint32Array(actual.magnitude.buffer), new Uint32Array(expected.magnitude.buffer));
    assert.deepEqual(new Uint32Array(actual.phase.buffer), new Uint32Array(expected.phase.buffer));
    structuredClone(actual.magnitude, { transfer: [actual.magnitude.buffer] });
    structuredClone(actual.phase, { transfer: [actual.phase.buffer] });
  }
  const otherLane = Float32Array.from({ length: ZIP_CHUNK_SAMPLES }, (_, index) => 0.3 * Math.cos(index / 43));
  const restarted = createZipStftSequence()(otherLane), reference = zipStft(otherLane);
  assert.deepEqual(new Uint32Array(restarted.magnitude.buffer), new Uint32Array(reference.magnitude.buffer));
  assert.deepEqual(new Uint32Array(restarted.phase.buffer), new Uint32Array(reference.phase.buffer));
});

test('chunk completion counts every exact window including the final partial window', async () => {
  for (const [frameCount, expectedChunks] of [[400, 1], [ZIP_STRIDE_SAMPLES + ZIP_SAMPLE_RATE + 137, 2]] as const) {
    const source = Float32Array.from({ length: frameCount }, (_, index) => 0.25 * Math.sin(index / 19));
    const started = Promise.withResolvers<void>(), inferenceGate = Promise.withResolvers<void>();
    const completed: Array<[number, number]> = [];
    const known: number[] = [];
    let inferenceCalls = 0;
    const enhancement = enhanceZipSamples(source, ZIP_SAMPLE_RATE, async (magnitude, phase) => {
      inferenceCalls++;
      if (inferenceCalls === 1) {
        started.resolve();
        await inferenceGate.promise;
      }
      for (let index = 0; index < magnitude.length; index++) magnitude[index] *= 0.9;
      return { magnitude, phase };
    }, (count, total) => { completed.push([count, total]); }, (total) => { known.push(total); });
    await started.promise;
    assert.deepEqual(known, [expectedChunks]);
    assert.deepEqual(completed, [], 'a submitted but unfinished GPU window is not completed');
    inferenceGate.resolve();
    const output = await enhancement;
    assert.equal(inferenceCalls, expectedChunks);
    assert.deepEqual(completed, Array.from({ length: expectedChunks }, (_, index) => [index + 1, expectedChunks]));
    assert.equal(output.length, source.length);
    assert.ok(output.every(Number.isFinite));
    assert.notDeepEqual(output, source);
  }
});

for (const failure of ['inference', 'inverse-STFT'] as const) {
  test(`a failed ${failure} window never advances completed chunks`, async () => {
    const source = Float32Array.from({ length: ZIP_STRIDE_SAMPLES + ZIP_SAMPLE_RATE + 137 }, (_, index) => 0.25 * Math.sin(index / 19));
    const completed: Array<[number, number]> = [];
    let inferenceCalls = 0;
    await assert.rejects(enhanceZipSamples(source, ZIP_SAMPLE_RATE, async (magnitude, phase) => {
      inferenceCalls++;
      if (inferenceCalls === 2) {
        if (failure === 'inference') throw new Error('WebGPU device lost during the second window');
        phase[0] = Number.NaN;
      }
      for (let index = 0; index < magnitude.length; index++) magnitude[index] *= 0.9;
      return { magnitude, phase };
    }, (count, total) => { completed.push([count, total]); }),
    failure === 'inference' ? /WebGPU device lost/ : /invalid spectral values/);
    assert.equal(inferenceCalls, 2);
    assert.deepEqual(completed, [[1, 2]]);
  });
}
