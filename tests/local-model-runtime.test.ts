import test from 'node:test';
import assert from 'node:assert/strict';

import { __localModelRuntimeTesting as runtime } from '../src/core/local-model-runtime';

const FRONTEND_FIXTURE_INDICES = [
  0, 1, 2, 3, 10, 31, 32, 33, 47, 63, 64, 95, 96, 127, 128, 159, 160, 191
];

// Calculated with the accepted GigaAM checkpoint's restored BF16 window and
// mel-bank buffers; freshly constructed torchaudio buffers produce different logits.
const PYTHON_LOG_MEL_FIXTURE = [
  -10.334663391113281,
  -9.317130088806152,
  -9.445328712463379,
  -9.088786125183105,
  -8.891809463500977,
  4.438273906707764,
  4.438173294067383,
  5.587215423583984,
  -3.8078739643096924,
  3.0221946239471436,
  3.1961851119995117,
  -9.170083045959473,
  -3.6744611263275146,
  -8.618334770202637,
  -8.568355560302734,
  -9.405864715576172,
  -10.097670555114746,
  -5.2834320068359375
];
function activitySamples(...runs: Array<[durationMs: number, amplitude: number]>): Float32Array {
  const sampleRate = 16_000;
  const samples = new Float32Array(
    runs.reduce((total, [durationMs]) => total + Math.round(durationMs * sampleRate / 1_000), 0)
  );
  let offset = 0;
  for (const [durationMs, amplitude] of runs) {
    const end = offset + Math.round(durationMs * sampleRate / 1_000);
    samples.fill(amplitude, offset, end);
    offset = end;
  }
  return samples;
}

test('GigaAM frontend matches restored checkpoint buffers on the Python numeric fixture', () => {
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

test('draft recognition isolates speech separated by silence and retains every segment word', async () => {
  const samples = activitySamples(
    [1_000, 0], [300, 0.1], [2_000, 0], [300, 0.1], [1_000, 0]
  );
  const segments = runtime.segmentSamplesByActivity(samples);
  const calls: number[] = [];
  const result = await runtime.recognizeActivitySegments(samples, segments, async (chunk, startSample) => {
    calls.push(startSample);
    return {
      durationSeconds: chunk.length / 16_000,
      tokens: [{ text: `utterance-${calls.length}`, startSeconds: 0, endSeconds: 0.1 }]
    };
  });

  assert.deepEqual(calls, [16_000, 52_800]);
  assert.deepEqual(result.tokens.map(({ text }) => text), ['utterance-1', 'utterance-2']);
  assert.deepEqual(result.tokens.map(({ startSeconds }) => startSeconds), [1, 3.3]);
});

test('long speech splits near lowest energy before the 24-second model limit', async () => {
  const samples = new Float32Array(31 * 16_000).fill(0.1);
  samples.fill(0, 21 * 16_000, 21 * 16_000 + 1_920);
  const calls: Array<[number, number]> = [];
  const result = await runtime.recognizeActivitySegments(
    samples,
    [{ startSample: 0, endSample: samples.length }],
    async (chunk, startSample) => {
      calls.push([startSample, chunk.length]);
      return {
        durationSeconds: chunk.length / 16_000,
        tokens: [{ text: `part-${calls.length}`, startSeconds: 0.1, endSeconds: 0.2 }]
      };
    }
  );

  assert.equal(calls.length, 2);
  assert.equal(calls[0][0], 0);
  assert.ok(calls[0][1] >= 20 * 16_000 && calls[0][1] <= 22 * 16_000);
  assert.equal(calls[1][0], calls[0][1]);
  assert.equal(calls[1][0] + calls[1][1], samples.length);
  assert.deepEqual(result.tokens.map(({ text }) => text), ['part-1', 'part-2']);
  assert.equal(result.tokens[1].startSeconds, calls[1][0] / 16_000 + 0.1);
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

test('S2 activity segmentation keeps 0.9 seconds of internal silence in one row', () => {
  const samples = activitySamples(
    [200, 0],
    [300, 0.1],
    [900, 0],
    [300, 0.1],
    [200, 0]
  );

  assert.deepEqual(runtime.segmentSamplesByActivity(samples), [
    { startSample: 3_200, endSample: 27_200 }
  ]);
});

test('S2 activity segmentation splits on exactly 1.0 seconds and trims to active frames', () => {
  const samples = activitySamples(
    [200, 0],
    [300, 0.1],
    [1_000, 0],
    [300, 0.1],
    [200, 0]
  );

  assert.deepEqual(runtime.segmentSamplesByActivity(samples), [
    { startSample: 3_200, endSample: 8_000 },
    { startSample: 24_000, endSample: 28_800 }
  ]);
});

test('S2 row assignment uses word midpoints and half-open activity boundaries', () => {
  const segments = [
    { startSample: 3_200, endSample: 8_000 },
    { startSample: 24_000, endSample: 28_800 }
  ];
  const words = [
    { text: 'start', startSeconds: 0.19, endSeconds: 0.21 },
    { text: 'first-end', startSeconds: 0.49, endSeconds: 0.51 },
    { text: 'second-start', startSeconds: 1.49, endSeconds: 1.51 },
    { text: 'second-end', startSeconds: 1.79, endSeconds: 1.81 }
  ];

  assert.deepEqual(runtime.groupWordsByActivitySegments(words, segments), [
    { ...segments[0], wordStart: 0, wordEnd: 1 },
    { ...segments[1], wordStart: 2, wordEnd: 3 }
  ]);
});

test('S2 draft ordering sorts overlapping words by midpoint before grouping', () => {
  const startSorted = [
    { text: 'later-midpoint', startSeconds: 0, endSeconds: 1.2 },
    { text: 'earlier-midpoint', startSeconds: 0.3, endSeconds: 0.5 }
  ];
  const midpointSorted = [...startSorted].sort(runtime.compareWordsByMidpoint);

  assert.deepEqual(midpointSorted.map((word) => word.text), [
    'earlier-midpoint',
    'later-midpoint'
  ]);
  assert.deepEqual(
    runtime.groupWordsByActivitySegments(midpointSorted, [
      { startSample: 0, endSample: 8_000 },
      { startSample: 8_000, endSample: 16_000 }
    ]),
    [
      { startSample: 0, endSample: 8_000, wordStart: 0, wordEnd: 1 },
      { startSample: 8_000, endSample: 16_000, wordStart: 1, wordEnd: 2 }
    ]
  );
});

test('S2 smoothing bridges 160ms gaps before applying the 120ms minimum activity', () => {
  const bridged = activitySamples(
    [200, 0],
    [70, 0.1],
    [160, 0],
    [70, 0.1],
    [200, 0]
  );
  const tooShort = activitySamples([200, 0], [110, 0.1], [200, 0]);
  const minimum = activitySamples([200, 0], [120, 0.1], [200, 0]);

  assert.deepEqual(runtime.segmentSamplesByActivity(bridged), [
    { startSample: 3_200, endSample: 8_000 }
  ]);
  assert.deepEqual(runtime.segmentSamplesByActivity(tooShort), []);
  assert.deepEqual(runtime.segmentSamplesByActivity(minimum), [
    { startSample: 3_200, endSample: 5_120 }
  ]);
});

test('S2 grouping has no word-count or transcript-span row cap', () => {
  const words = Array.from({ length: 33 }, (_, index) => ({
    text: `w${index}`,
    startSeconds: index * 0.4,
    endSeconds: index * 0.4 + 0.1
  }));

  assert.deepEqual(
    runtime.groupWordsByActivitySegments(words, [{ startSample: 0, endSample: 224_000 }]),
    [{ startSample: 0, endSample: 224_000, wordStart: 0, wordEnd: 33 }]
  );
});

test('draft rows retain the service UUID for identical prepared PCM and boundaries', async () => {
  const rowId = await runtime.draftRowId(
    'parity-RU-tx-gold-bg-noise',
    'track-2',
    'fec29e8d1bb38a0018d20a1633b2840c1fccdd4c415b87ab5ae5bfc914cd6fec',
    4_000,
    505_600
  );
  assert.equal(rowId, '2263f035-ecb5-5d9e-9df2-280bcb91adbc');
  assert.notEqual(
    await runtime.draftRowId(
      'parity-RU-tx-gold-bg-noise',
      'track-2',
      'fec29e8d1bb38a0018d20a1633b2840c1fccdd4c415b87ab5ae5bfc914cd6fec',
      4_000,
      505_760
    ),
    rowId
  );
});
