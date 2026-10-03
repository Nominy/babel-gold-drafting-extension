import test from 'node:test';
import assert from 'node:assert/strict';

import { __localModelRuntimeTesting as runtime } from '../src/core/local-model-runtime';
import { __cDenoiseRuntimeTesting as denoise } from '../src/core/c-denoise-runtime';
import { __cDenoiseAcousticTesting as acoustic } from '../src/core/c-denoise-acoustic';

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
});

test('native full-lane recognition covers quiet speech and offsets every chunk without VAD filtering or duplicate removal', async () => {
  const samples = new Float32Array(31 * 16_000).fill(0.001);
  samples.fill(0, 21 * 16_000, 21 * 16_000 + 1_920);
  const calls: Array<[number, number]> = [];
  const result = await runtime.recognizeSamplesInChunks(samples, async (chunk, start) => {
    calls.push([start, chunk.length]);
    return { durationSeconds: chunk.length / 16_000, tokens: [
      { text: 'тихо', startSeconds: 0.1, endSeconds: 0.2 },
      { text: 'тихо', startSeconds: 0.3, endSeconds: 0.4 }
    ] };
  });
  assert.equal(calls[0][0], 0);
  assert.equal(calls[1][0], calls[0][1]);
  assert.equal(calls.at(-1)![0] + calls.at(-1)![1], samples.length);
  assert.ok(calls.every(([, length]) => length <= 24 * 16_000));
  assert.deepEqual(result.tokens.map((word) => word.text), ['тихо', 'тихо', 'тихо', 'тихо']);
  assert.equal(result.tokens[2].startSeconds, calls[1][0] / 16_000 + 0.1);
});

test('native recognition refuses invalid word clocks instead of silently dropping source words', async () => {
  await assert.rejects(runtime.recognizeSamplesInChunks(new Float32Array(16_000), async () => ({
    durationSeconds: 1, tokens: [{ text: 'не терять', startSeconds: 0.2, endSeconds: 1.1 }]
  })), /invalid word.timestamp/);
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

test('short-audio limits reject the complete source instead of silently transcribing a clipped prefix', async () => {
  const rate = 8_000, frames = rate + 1;
  const bytes = new ArrayBuffer(44 + frames * 2), view = new DataView(bytes);
  for (const [offset, text] of [[0, 'RIFF'], [8, 'WAVE'], [12, 'fmt '], [36, 'data']] as const) {
    for (let index = 0; index < text.length; index += 1) view.setUint8(offset + index, text.charCodeAt(index));
  }
  view.setUint32(4, bytes.byteLength - 8, true); view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); view.setUint16(22, 1, true); view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  view.setUint32(40, frames * 2, true); view.setInt16(44 + (frames - 1) * 2, 127, true);
  const blob = new Blob([bytes], { type: 'audio/wav' });
  await assert.rejects(runtime.prepareDraftAudio(blob, 1), /Choose a sample no longer than 1 seconds/);
  const full = await runtime.prepareDraftAudio(blob);
  assert.equal(full.source.length, frames);
  assert.equal(full.source[frames - 1], 127 / 32768);
  const exact = await runtime.prepareDraftAudio(blob, frames / rate);
  assert.equal(exact.source.length, frames);
  assert.equal(exact.source[frames - 1], full.source[frames - 1]);
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

test('runtime rows own all words and cut at gap, span and count boundaries', () => {
  const words = Array.from({ length: 33 }, (_, index) => ({ text: `w${index}`, startSeconds: index * 0.2, endSeconds: index * 0.2 + 0.1 }));
  assert.deepEqual(denoise.buildRows(words), [{ wordStart: 0, wordEnd: 32 }, { wordStart: 32, wordEnd: 33 }]);
  assert.deepEqual(denoise.buildRows([
    { text: 'а', startSeconds: 0, endSeconds: 1 },
    { text: 'б', startSeconds: 1.8, endSeconds: 2 },
    { text: 'в', startSeconds: 2.1, endSeconds: 13.8 },
    { text: 'г', startSeconds: 13.8, endSeconds: 13.8001 }
  ]), [{ wordStart: 0, wordEnd: 1 }, { wordStart: 1, wordEnd: 3 }, { wordStart: 3, wordEnd: 4 }]);
  assert.deepEqual(denoise.buildRows([
    { text: 'а', startSeconds: 0, endSeconds: 1 },
    { text: 'б', startSeconds: -1, endSeconds: -1 },
    { text: 'в', startSeconds: 1.9, endSeconds: 2 }
  ], Uint8Array.of(1, 0, 1)), [{ wordStart: 0, wordEnd: 3 }]);
});

test('context budget reduces farther context first, preserves intact words, and owns every center exactly once', () => {
  const windows = denoise.buildWindows([{ wordStart: 0, wordEnd: 70 }], new Array(70).fill(8));
  assert.deepEqual(windows, [
    { left: 0, coreLeft: 0, coreRight: 31, right: 31 },
    { left: 31, coreLeft: 31, coreRight: 62, right: 62 },
    { left: 39, coreLeft: 62, coreRight: 70, right: 70 }
  ]);
  assert.throws(() => denoise.buildWindows([{ wordStart: 0, wordEnd: 1 }], [255]), /fit intact/);
  const context = denoise.buildWindows([{ wordStart: 0, wordEnd: 32 }, { wordStart: 32, wordEnd: 64 }, { wordStart: 64, wordEnd: 96 }], new Array(96).fill(3));
  assert.deepEqual(context[1], { left: 6, coreLeft: 32, coreRight: 64, right: 90 });
});

test('denoise keeps frozen candidates, stable equal-confidence ranks, masked padding and four exact levels', async () => {
  const observed: Array<{ noisy: bigint[]; level: number }> = [];
  const result = await denoise.refineLabels(Uint8Array.of(1, 1, 1, 1, 0), async (noisy, level, step) => {
    observed.push({ noisy: Array.from(noisy), level });
    const output = new Float32Array(5 * 7);
    for (let word = 0; word < 5; word += 1) output[word * 7 + (step + 1)] = 1;
    return output;
  });
  assert.deepEqual(observed, [
    { noisy: [7n, 7n, 7n, 7n, 7n], level: 1 },
    { noisy: [7n, 7n, 7n, 1n, 7n], level: 0.75 },
    { noisy: [7n, 7n, 2n, 1n, 7n], level: 0.5 },
    { noisy: [7n, 3n, 2n, 1n, 7n], level: 0.25 }
  ]);
  assert.deepEqual(Array.from(result), [4, 3, 2, 1, 0]);
  await assert.rejects(denoise.refineLabels(Uint8Array.of(1), async () => new Float32Array(7).fill(NaN)), /nonfinite/);
});

test('context feeds preserve token ownership using GPU-compatible int32 masks and word indices', () => {
  const tokenizer = { cls_token_id: 101, sep_token_id: 102 } as unknown as Parameters<typeof denoise.encodeWindow>[0];
  const feeds = denoise.encodeWindow(tokenizer, [[5, 6], [7]], { left: 0, coreLeft: 0, coreRight: 2, right: 2 });
  try {
    assert.equal(feeds.input_ids.type, 'int64');
    assert.deepEqual(Array.from(feeds.input_ids.data as BigInt64Array), [101n, 5n, 6n, 7n, 102n]);
    assert.equal(feeds.attention_mask.type, 'int32');
    assert.deepEqual(Array.from(feeds.attention_mask.data as Int32Array), [1, 1, 1, 1, 1]);
    assert.equal(feeds.first_subtoken.type, 'int32');
    assert.deepEqual(Array.from(feeds.first_subtoken.data as Int32Array), [1, 3]);
  } finally { for (const tensor of Object.values(feeds)) tensor.dispose(); }
});

test('cached interval rendering selects half-open original word ownership and full-stream labels without mutation', () => {
  const tokens = [
    { text: 'до', startSeconds: 0, endSeconds: 1 },
    { text: 'внутри', startSeconds: 1, endSeconds: 2 },
    { text: 'после', startSeconds: 2, endSeconds: 3 }
  ];
  const snapshot = structuredClone(tokens), lane = { tokens, labelIds: Uint8Array.of(1, 3, 2), pcmSha256: 'source', ranges: [] };
  assert.equal(runtime.renderCachedInterval(lane, 1.5, 2.5), 'внутри?');
  assert.equal(runtime.renderCachedInterval(lane, 2.5, 3.5), 'После.');
  assert.deepEqual(tokens, snapshot);
  assert.throws(() => runtime.renderCachedInterval(lane, 0.8, 1.2), /no immutable ASR words/);
});

test('local acoustic sampling uses real frames, earlier nearest ties, half-open coverage and zero masked gaps', () => {
  const times = Float64Array.of(0.02, 0.06, 0.1), features = new Uint16Array(3 * 768), prosody = new Uint16Array(12);
  features.fill(runtime.float32ToFloat16(1), 0, 768);
  features.fill(runtime.float32ToFloat16(2), 768, 1536);
  features.fill(runtime.float32ToFloat16(3), 1536);
  const offsets = Array.from({ length: 24 }, (_, index) => (index - 1) * 0.04);
  const sampled = acoustic.sampleLocalAudio(Float64Array.of(0.02), [{ times, features, prosody, startSeconds: 0, endSeconds: 0.12 }], 0.04, offsets);
  assert.deepEqual(Array.from(sampled.mask), [0, 1, 1, 1, ...new Array(20).fill(0)]);
  assert.equal(runtime.float16ToFloat32(sampled.data[772]), 1);
  assert.equal(runtime.float16ToFloat32(sampled.data[2 * 772]), 2);
  assert.equal(sampled.data[4 * 772], 0);
  const tied = acoustic.sampleLocalAudio(Float64Array.of(0.02), [{ times: Float64Array.of(0, 0.04), features: features.subarray(0, 1536), prosody: prosody.subarray(0, 8), startSeconds: 0, endSeconds: 0.08 }], 0.04, offsets);
  assert.equal(runtime.float16ToFloat32(tied.data[772]), 1);
});

test('prosody measures original waveform pitch and energy and resets energy deltas across encoder gaps', () => {
  const waveform = Float32Array.from({ length: 8000 }, (_, index) => 0.5 * Math.sin(2 * Math.PI * 200 * index / 8000));
  const values = acoustic.extractProsody(waveform, 8000, Float64Array.of(0.2, 0.24, 0.4), 0.04);
  assert.ok(Math.abs(runtime.float16ToFloat32(values[0]) - (Math.log(0.5 / Math.sqrt(2)) + 4) / 2) < 0.002);
  assert.ok(Math.abs(runtime.float16ToFloat32(values[1]) - Math.log(200 / 180) / 0.6) < 0.002);
  assert.equal(values[11], 0);
  const silence = acoustic.extractProsody(new Float32Array(8000), 8000, Float64Array.of(0, 0.04), 0.04);
  assert.equal(runtime.float16ToFloat32(silence[0]), -4);
  assert.equal(runtime.float16ToFloat32(silence[1]), 0);
  assert.equal(runtime.float16ToFloat32(silence[2]), -1);
});
