import test from 'node:test';
import assert from 'node:assert/strict';

import { __localModelRuntimeTesting as runtime } from '../src/core/local-model-runtime';

const FRONTEND_FIXTURE_INDICES = [
  0, 1, 2, 3, 10, 31, 32, 33, 47, 63, 64, 95, 96, 127, 128, 159, 160, 191
];

// Calculated with torch 2.9.1 / torchaudio 2.9.1 MelSpectrogram using the exact
// GigaAM parameters (periodic Hann, n_fft=320, hop=160, center=false, HTK mel).
const PYTHON_LOG_MEL_FIXTURE = [
  -10.84756088256836,
  -10.063607215881348,
  -10.064454078674316,
  -9.603693962097168,
  -8.90012264251709,
  4.438756465911865,
  4.438756942749023,
  5.588564872741699,
  -3.794724941253662,
  3.02093505859375,
  3.1943273544311523,
  -9.223891258239746,
  -3.670184850692749,
  -8.875046730041504,
  -8.873992919921875,
  -13.67418098449707,
  -13.68295955657959,
  -5.215675354003906
];
test('GigaAM frontend has exact dimensions and matches the Python numeric fixture', () => {
  const samples = new Float32Array(640);
  for (let index = 0; index < samples.length; index += 1) {
    samples[index] =
      0.3 * Math.sin((2 * Math.PI * 440 * index) / 16_000) +
      0.1 * Math.cos((2 * Math.PI * 1_000 * index) / 16_000) +
      ((index % 17) - 8) * 0.001;
  }

  const features = runtime.extractLogMelFeatures(samples);
  assert.equal(features.frames, 3);
  assert.equal(features.melBins, 64);
  assert.equal(features.data.length, 64 * 3);
  FRONTEND_FIXTURE_INDICES.forEach((index, fixtureIndex) => {
    assert.ok(
      Math.abs(features.data[index] - PYTHON_LOG_MEL_FIXTURE[fixtureIndex]) < 2e-4,
      `feature ${index}: browser=${features.data[index]}, python=${PYTHON_LOG_MEL_FIXTURE[fixtureIndex]}`
    );
  });
});

test('CTC decoding collapses adjacent repeats before blank removal and preserves timestamp frames', () => {
  const vocabulary = Array.from(' абвгдежзийклмнопрстуфхцчшщъыьэюя');
  const classCount = 34;
  const frameClasses = [
    vocabulary.indexOf('а'),
    vocabulary.indexOf('а'),
    33,
    vocabulary.indexOf('а'),
    vocabulary.indexOf(' '),
    vocabulary.indexOf(' '),
    33,
    vocabulary.indexOf('б'),
    vocabulary.indexOf('б')
  ];
  const logits = new Float32Array(frameClasses.length * classCount).fill(-100);
  frameClasses.forEach((classIndex, frame) => {
    logits[frame * classCount + classIndex] = 10;
  });

  const decoded = runtime.decodeCtc(logits, frameClasses.length, frameClasses.length, 9);
  assert.equal(decoded.rawText, 'аа б');
  assert.deepEqual(decoded.words, [
    { text: 'аа', startSeconds: 0, endSeconds: 4 },
    { text: 'б', startSeconds: 7, endSeconds: 8 }
  ]);
});

test('silent CTC output is a successful empty transcript', async () => {
  const timeSteps = 4;
  const classCount = 34;
  const logits = new Float32Array(timeSteps * classCount).fill(-100);
  for (let frame = 0; frame < timeSteps; frame += 1) {
    logits[frame * classCount + 33] = 10;
  }

  assert.deepEqual(runtime.decodeCtc(logits, timeSteps, timeSteps, 1), {
    rawText: '',
    words: []
  });
  const result = await runtime.transcribeWithRecognizer(
    new Float32Array(16_000),
    async (samples) => ({ durationSeconds: samples.length / 16_000, tokens: [] })
  );
  assert.deepEqual(result, {
    text: '',
    durationSeconds: 1,
    tokens: []
  });
});

test('full-track recognition chunks more than 200 seconds, offsets timestamps, and deduplicates overlap', async () => {
  const sampleRate = 16_000;
  const samples = new Float32Array(205 * sampleRate);
  const calls: Array<{ startSample: number; length: number }> = [];
  const result = await runtime.recognizeSamplesInChunks(
    samples,
    async (chunk, startSample) => {
      calls.push({ startSample, length: chunk.length });
      const startSeconds = startSample / sampleRate;
      const tokens = [{
        text: `chunk-${startSeconds}`,
        startSeconds: 0.2,
        endSeconds: 0.4
      }];
      if (startSeconds === 0) {
        tokens.push({ text: 'overlap', startSeconds: 29.1, endSeconds: 29.3 });
      } else if (startSeconds === 29) {
        tokens.push({ text: 'overlap', startSeconds: 0.1, endSeconds: 0.3 });
      }
      return { durationSeconds: chunk.length / sampleRate, tokens };
    }
  );

  assert.deepEqual(calls.map((call) => call.startSample / sampleRate), [0, 29, 58, 87, 116, 145, 174, 203]);
  assert.ok(calls.every((call, index) => index === 0 || call.startSample <=
    calls[index - 1].startSample + calls[index - 1].length));
  assert.equal(calls.at(-1)!.startSample + calls.at(-1)!.length, samples.length);
  assert.equal(result.durationSeconds, 205);
  assert.equal(result.tokens.filter((token) => token.text === 'overlap').length, 1);
  assert.deepEqual(result.tokens.at(-1), {
    text: 'chunk-203',
    startSeconds: 203.2,
    endSeconds: 203.4
  });
});

test('segment transcription crops the target interval before recognition and excludes adjacent words', async () => {
  const sampleRate = 16_000;
  const samples = Float32Array.from(
    { length: 10 * sampleRate },
    (_, sampleIndex) => sampleIndex
  );
  const transcript = await runtime.transcribeSampleInterval(
    samples,
    2.25,
    3.75,
    async (cropped, startSample) => {
      assert.equal(startSample, 0);
      assert.equal(cropped.length, 1.5 * sampleRate);
      assert.equal(cropped[0], 2.25 * sampleRate);
      assert.equal(cropped.at(-1), 3.75 * sampleRate - 1);
      const tokens = [];
      if (cropped[0] < 2.25 * sampleRate) {
        tokens.push({ text: 'before', startSeconds: 0, endSeconds: 0.1 });
      }
      tokens.push({ text: 'inside', startSeconds: 0.5, endSeconds: 1 });
      if (cropped.at(-1)! >= 3.75 * sampleRate) {
        tokens.push({ text: 'after', startSeconds: 1.4, endSeconds: 1.5 });
      }
      return { durationSeconds: cropped.length / sampleRate, tokens };
    },
    async (words) => words.map(() => 'O' as const)
  );

  assert.equal(transcript.text, 'Inside');
  assert.deepEqual(transcript.tokens, [{
    text: 'inside',
    startSeconds: 0.5,
    endSeconds: 1
  }]);
});

test('punctuation alignment uses each source word first subtoken and rejects incomplete coverage', () => {
  const predictions = Uint8Array.from([0, 1, 0, 4, 6, 0, 3, 0]);
  assert.deepEqual(runtime.firstSubtokenLabels(predictions, Uint16Array.from([1, 3, 6]), 3), [
    'COMMA',
    'HYPHEN_JOIN',
    'QUESTION'
  ]);
  assert.throws(
    () => runtime.firstSubtokenLabels(predictions, Uint16Array.from([1, 1, 6]), 3),
    /lost or reordered source word 1/
  );
  assert.throws(
    () => runtime.firstSubtokenLabels(predictions, Uint16Array.from([1, 3]), 3),
    /represented 2 first subtokens for 3 words/
  );
});

test('production punctuation rendering preserves lexical words and boundary semantics', () => {
  const rendered = runtime.renderBoundaryLabels(
    ['привет', 'мир', 'по', 'русски', 'да'],
    ['COMMA', 'PERIOD', 'HYPHEN_JOIN', 'DASH_SINGLE', 'QUESTION']
  );
  assert.deepEqual(rendered, {
    text: 'Привет, мир. По-русски- да?',
    sentenceStart: true
  });
  assert.throws(
    () => runtime.renderBoundaryLabels(['слово'], []),
    /label count 0 does not match source word count 1/
  );
});

test('duration policy defaults smoke transcription to 15 seconds and null keeps full audio', () => {
  assert.equal(runtime.resolveMaxDurationSeconds(undefined), 15);
  assert.equal(runtime.resolveMaxDurationSeconds(null), null);
  assert.equal(runtime.resolveMaxDurationSeconds(2.25), 2.25);
  assert.equal(runtime.clippedFrameCount(960_000, 48_000, 15), 720_000);
  assert.equal(runtime.clippedFrameCount(960_000, 48_000, null), 960_000);
  assert.equal(runtime.clippedFrameCount(10_000, 44_100, 0.1), 4_410);
  assert.throws(() => runtime.resolveMaxDurationSeconds(0), /must be positive/);
  assert.throws(() => runtime.resolveMaxDurationSeconds(Number.NaN), /must be positive/);
});

test('float16 feature packing uses IEEE-754 round-to-nearest-even values', () => {
  assert.equal(runtime.float32ToFloat16(0), 0x0000);
  assert.equal(runtime.float32ToFloat16(1), 0x3c00);
  assert.equal(runtime.float32ToFloat16(-2), 0xc000);
  assert.equal(runtime.float32ToFloat16(65_504), 0x7bff);
  assert.equal(runtime.float32ToFloat16(2 ** -14), 0x0400);
  assert.equal(runtime.float32ToFloat16(2 ** -24), 0x0001);
  assert.equal(runtime.float16ToFloat32(0x3c00), 1);
  assert.equal(runtime.float16ToFloat32(0xc000), -2);
});

test('float16 tensor output decodes Uint16 bits but preserves native decoded values', () => {
  assert.equal(runtime.readFloat16Value(Uint16Array.of(0x3e00)), 1.5);

  const NativeFloat16Array = Reflect.get(globalThis, 'Float16Array') as
    | { from(values: ArrayLike<number>): ArrayLike<number> }
    | undefined;
  const decodedValues = NativeFloat16Array
    ? NativeFloat16Array.from([1.5])
    : new Float32Array([1.5]);
  assert.equal(runtime.readFloat16Value(decodedValues), 1.5);
});

test('draft row grouping follows production gap, span, and 32-word boundaries', () => {
  const gapWords = [
    { text: 'а', startSeconds: 0, endSeconds: 0.4 },
    { text: 'б', startSeconds: 0.5, endSeconds: 1 },
    { text: 'в', startSeconds: 1.8, endSeconds: 2.1 }
  ];
  assert.deepEqual(runtime.groupWordRows(gapWords), [
    { start: 0, end: 2 },
    { start: 2, end: 3 }
  ]);

  const spanWords = [
    { text: 'а', startSeconds: 0, endSeconds: 5.5 },
    { text: 'б', startSeconds: 6, endSeconds: 11.5 },
    { text: 'в', startSeconds: 12, endSeconds: 12.01 }
  ];
  assert.deepEqual(runtime.groupWordRows(spanWords), [
    { start: 0, end: 2 },
    { start: 2, end: 3 }
  ]);

  const cappedWords = Array.from({ length: 33 }, (_, index) => ({
    text: `w${index}`,
    startSeconds: index * 0.1,
    endSeconds: index * 0.1 + 0.05
  }));
  assert.deepEqual(runtime.groupWordRows(cappedWords), [
    { start: 0, end: 32 },
    { start: 32, end: 33 }
  ]);
});
