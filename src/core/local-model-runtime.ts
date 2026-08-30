import { BertTokenizer } from '@huggingface/transformers';
import * as ort from 'onnxruntime-web/webgpu';

import { prepareL0Tracks, type PreparedL0Track } from './l0-client';
import { getCachedLocalModelFile } from './local-model-bundle';
import { LOCAL_MODEL_BASE_URL } from './settings';
import {
  prepareL0TimingTracks,
  type L0TimingQueueStatus,
  type L0TimingRequestCallbacks
} from './l0-timing-client';
import { buildCanonicalTaskIdentity } from './transcript';
import type {
  CapturedAudioTrack,
  ExtensionSettings,
  L0DraftResponse,
  L0TimingResponse,
  TranscriptJob
} from './types';

const SAMPLE_RATE = 16_000;
const N_FFT = 320;
const HOP_LENGTH = 160;
const MEL_BINS = 64;
const SPECTRUM_BINS = N_FFT / 2 + 1;
const ASR_CLASS_COUNT = 34;
const ASR_BLANK_ID = 33;
const ASR_VOCABULARY = Array.from(' абвгдежзийклмнопрстуфхцчшщъыьэюя');
const PUNCTUATION_LABELS = [
  'O',
  'COMMA',
  'PERIOD',
  'QUESTION',
  'HYPHEN_JOIN',
  'DASH_SINGLE',
  'DASH_DOUBLE'
] as const;
const PUNCTUATION_SUFFIXES = ['', ',', '.', '?', '-', '-', '--'] as const;
const PUNCTUATION_CHUNK_WORDS = 60;
const PUNCTUATION_MAX_TOKENS = 512;
const ASR_CHUNK_SECONDS = 30;
const ASR_CHUNK_OVERLAP_SECONDS = 1;
const ASR_CHUNK_SAMPLES = ASR_CHUNK_SECONDS * SAMPLE_RATE;
const ASR_CHUNK_OVERLAP_SAMPLES = ASR_CHUNK_OVERLAP_SECONDS * SAMPLE_RATE;
const ASR_CHUNK_STRIDE_SAMPLES = ASR_CHUNK_SAMPLES - ASR_CHUNK_OVERLAP_SAMPLES;

const ASR_MODEL_PATH = 'asr/v3_ctc.onnx';
const PUNCTUATION_MODEL_PATH = 'punctuation/model.int8.onnx';
const PUNCTUATION_CONFIG_PATH = 'punctuation/config.json';
const TOKENIZER_PATH = 'punctuation/tokenizer.json';
const TOKENIZER_CONFIG_PATH = 'punctuation/tokenizer_config.json';

export interface LocalTranscriptResult {
  text: string;
  durationSeconds: number;
  tokens: Array<{
    text: string;
    startSeconds: number;
    endSeconds: number;
  }>;
}

type LocalWord = LocalTranscriptResult['tokens'][number];
type PunctuationLabel = (typeof PUNCTUATION_LABELS)[number];
type Session = ort.InferenceSession;
type Tokenizer = InstanceType<typeof BertTokenizer>;
type DecodedCtc = { rawText: string; words: LocalWord[] };
type TranscriptWithLabels = LocalTranscriptResult & { labels: PunctuationLabel[] };
type SampleRecognition = { durationSeconds: number; tokens: LocalWord[] };
type SampleRecognizer = (samples: Float32Array, startSample: number) => Promise<SampleRecognition>;
type WordPunctuator = (words: readonly string[]) => Promise<PunctuationLabel[]>;

type Radix2Plan = {
  size: number;
  reversed: Uint32Array;
  cos: Float64Array;
  sin: Float64Array;
};

type BluesteinPlan = {
  size: number;
  convolutionSize: number;
  radix: Radix2Plan;
  chirpCos: Float64Array;
  chirpSin: Float64Array;
  kernelReal: Float64Array;
  kernelImag: Float64Array;
};

type MelPlan = {
  hann: Float64Array;
  weights: Float64Array;
  firstBin: Uint16Array;
  lastBin: Uint16Array;
  fft: BluesteinPlan;
};

type PunctuationResources = {
  session: Session;
  tokenizer: Tokenizer;
  labels: readonly PunctuationLabel[];
};

let asrSessionPromise: Promise<Session> | null = null;
let punctuationResourcesPromise: Promise<PunctuationResources> | null = null;
let ortConfigured = false;
const radix2Plans = new Map<number, Radix2Plan>();
const bluesteinPlans = new Map<number, BluesteinPlan>();
let melPlan: MelPlan | null = null;
const floatBits = new Uint32Array(1);
const floatScratch = new Float32Array(floatBits.buffer);

function actionableError(stage: string, error: unknown): Error {
  const detail = error instanceof Error ? error.message : String(error);
  return new Error(`Local model ${stage} failed: ${detail}`, { cause: error });
}

function assertPositiveInteger(value: number, label: string): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${label} must be a positive integer; received ${value}.`);
  }
}

function getRadix2Plan(size: number): Radix2Plan {
  const cached = radix2Plans.get(size);
  if (cached) return cached;
  assertPositiveInteger(size, 'FFT size');
  if ((size & (size - 1)) !== 0) {
    throw new Error(`Radix-2 FFT size ${size} is not a power of two.`);
  }
  const reversed = new Uint32Array(size);
  let reversedIndex = 0;
  for (let index = 1; index < size; index += 1) {
    let bit = size >>> 1;
    while (reversedIndex & bit) {
      reversedIndex ^= bit;
      bit >>>= 1;
    }
    reversedIndex ^= bit;
    reversed[index] = reversedIndex;
  }
  const cos = new Float64Array(size >>> 1);
  const sin = new Float64Array(size >>> 1);
  for (let index = 0; index < cos.length; index += 1) {
    const angle = (2 * Math.PI * index) / size;
    cos[index] = Math.cos(angle);
    sin[index] = Math.sin(angle);
  }
  const plan = { size, reversed, cos, sin };
  radix2Plans.set(size, plan);
  return plan;
}

function fftRadix2(real: Float64Array, imaginary: Float64Array, plan: Radix2Plan, inverse: boolean): void {
  const { size, reversed, cos, sin } = plan;
  if (real.length !== size || imaginary.length !== size) {
    throw new Error(`FFT buffers must both have length ${size}.`);
  }
  for (let index = 0; index < size; index += 1) {
    const target = reversed[index];
    if (target <= index) continue;
    const realValue = real[index];
    real[index] = real[target];
    real[target] = realValue;
    const imaginaryValue = imaginary[index];
    imaginary[index] = imaginary[target];
    imaginary[target] = imaginaryValue;
  }
  for (let width = 2; width <= size; width *= 2) {
    const halfWidth = width >>> 1;
    const rootStride = size / width;
    for (let offset = 0; offset < size; offset += width) {
      for (let element = 0; element < halfWidth; element += 1) {
        const rootIndex = element * rootStride;
        const rootReal = cos[rootIndex];
        const rootImaginary = inverse ? sin[rootIndex] : -sin[rootIndex];
        const right = offset + element + halfWidth;
        const rightReal = real[right] * rootReal - imaginary[right] * rootImaginary;
        const rightImaginary = real[right] * rootImaginary + imaginary[right] * rootReal;
        const left = offset + element;
        const leftReal = real[left];
        const leftImaginary = imaginary[left];
        real[left] = leftReal + rightReal;
        imaginary[left] = leftImaginary + rightImaginary;
        real[right] = leftReal - rightReal;
        imaginary[right] = leftImaginary - rightImaginary;
      }
    }
  }
  if (inverse) {
    for (let index = 0; index < size; index += 1) {
      real[index] /= size;
      imaginary[index] /= size;
    }
  }
}

function getBluesteinPlan(size: number): BluesteinPlan {
  const cached = bluesteinPlans.get(size);
  if (cached) return cached;
  assertPositiveInteger(size, 'Bluestein FFT size');
  let convolutionSize = 1;
  while (convolutionSize < size * 2 - 1) convolutionSize *= 2;
  const radix = getRadix2Plan(convolutionSize);
  const chirpCos = new Float64Array(size);
  const chirpSin = new Float64Array(size);
  const kernelReal = new Float64Array(convolutionSize);
  const kernelImag = new Float64Array(convolutionSize);
  for (let index = 0; index < size; index += 1) {
    const angle = (Math.PI * index * index) / size;
    const real = Math.cos(angle);
    const imaginary = Math.sin(angle);
    chirpCos[index] = real;
    chirpSin[index] = imaginary;
    kernelReal[index] = real;
    kernelImag[index] = imaginary;
    if (index !== 0) {
      kernelReal[convolutionSize - index] = real;
      kernelImag[convolutionSize - index] = imaginary;
    }
  }
  fftRadix2(kernelReal, kernelImag, radix, false);
  const plan = {
    size,
    convolutionSize,
    radix,
    chirpCos,
    chirpSin,
    kernelReal,
    kernelImag
  };
  bluesteinPlans.set(size, plan);
  return plan;
}

function fftBluestein(
  input: Float32Array,
  inputOffset: number,
  window: Float64Array,
  plan: BluesteinPlan,
  workReal: Float64Array,
  workImaginary: Float64Array,
  outputReal: Float64Array,
  outputImaginary: Float64Array
): void {
  const { size, convolutionSize, chirpCos, chirpSin, kernelReal, kernelImag, radix } = plan;
  workReal.fill(0);
  workImaginary.fill(0);
  for (let index = 0; index < size; index += 1) {
    const sample = input[inputOffset + index] * window[index];
    workReal[index] = sample * chirpCos[index];
    workImaginary[index] = -sample * chirpSin[index];
  }
  fftRadix2(workReal, workImaginary, radix, false);
  for (let index = 0; index < convolutionSize; index += 1) {
    const real = workReal[index];
    const imaginary = workImaginary[index];
    workReal[index] = real * kernelReal[index] - imaginary * kernelImag[index];
    workImaginary[index] = real * kernelImag[index] + imaginary * kernelReal[index];
  }
  fftRadix2(workReal, workImaginary, radix, true);
  for (let index = 0; index < size; index += 1) {
    const real = workReal[index];
    const imaginary = workImaginary[index];
    const chirpReal = chirpCos[index];
    const chirpImaginary = chirpSin[index];
    outputReal[index] = real * chirpReal + imaginary * chirpImaginary;
    outputImaginary[index] = imaginary * chirpReal - real * chirpImaginary;
  }
}

function hzToHtkMel(hz: number): number {
  return 2595 * Math.log10(1 + hz / 700);
}

function htkMelToHz(mel: number): number {
  return 700 * (10 ** (mel / 2595) - 1);
}

function getMelPlan(): MelPlan {
  if (melPlan) return melPlan;
  const hann = new Float64Array(N_FFT);
  for (let index = 0; index < N_FFT; index += 1) {
    hann[index] = 0.5 - 0.5 * Math.cos((2 * Math.PI * index) / N_FFT);
  }
  const melMin = hzToHtkMel(0);
  const melMax = hzToHtkMel(SAMPLE_RATE / 2);
  const points = new Float64Array(MEL_BINS + 2);
  for (let index = 0; index < points.length; index += 1) {
    points[index] = htkMelToHz(melMin + ((melMax - melMin) * index) / (MEL_BINS + 1));
  }
  const weights = new Float64Array(MEL_BINS * SPECTRUM_BINS);
  const firstBin = new Uint16Array(MEL_BINS);
  const lastBin = new Uint16Array(MEL_BINS);
  for (let mel = 0; mel < MEL_BINS; mel += 1) {
    const lower = points[mel];
    const center = points[mel + 1];
    const upper = points[mel + 2];
    let first = SPECTRUM_BINS;
    let last = 0;
    for (let bin = 0; bin < SPECTRUM_BINS; bin += 1) {
      const frequency = (bin * SAMPLE_RATE) / N_FFT;
      const down = (frequency - lower) / (center - lower);
      const up = (upper - frequency) / (upper - center);
      const weight = Math.max(0, Math.min(down, up));
      weights[mel * SPECTRUM_BINS + bin] = weight;
      if (weight > 0) {
        first = Math.min(first, bin);
        last = bin + 1;
      }
    }
    firstBin[mel] = first;
    lastBin[mel] = last;
  }
  melPlan = { hann, weights, firstBin, lastBin, fft: getBluesteinPlan(N_FFT) };
  return melPlan;
}

function featureFrameCount(sampleCount: number): number {
  return sampleCount < N_FFT ? 0 : Math.floor((sampleCount - N_FFT) / HOP_LENGTH) + 1;
}

function extractLogMel(
  samples: Float32Array,
  write: (index: number, value: number) => void
): { frames: number; melBins: number } {
  const frames = featureFrameCount(samples.length);
  if (frames === 0) {
    throw new Error(
      `Decoded audio has ${samples.length} samples; GigaAM needs at least ${N_FFT} samples (20 ms).`
    );
  }
  const plan = getMelPlan();
  const workReal = new Float64Array(plan.fft.convolutionSize);
  const workImaginary = new Float64Array(plan.fft.convolutionSize);
  const outputReal = new Float64Array(N_FFT);
  const outputImaginary = new Float64Array(N_FFT);
  const power = new Float64Array(SPECTRUM_BINS);
  for (let frame = 0; frame < frames; frame += 1) {
    fftBluestein(
      samples,
      frame * HOP_LENGTH,
      plan.hann,
      plan.fft,
      workReal,
      workImaginary,
      outputReal,
      outputImaginary
    );
    for (let bin = 0; bin < SPECTRUM_BINS; bin += 1) {
      power[bin] = outputReal[bin] ** 2 + outputImaginary[bin] ** 2;
    }
    for (let mel = 0; mel < MEL_BINS; mel += 1) {
      let melPower = 0;
      const weightOffset = mel * SPECTRUM_BINS;
      for (let bin = plan.firstBin[mel]; bin < plan.lastBin[mel]; bin += 1) {
        melPower += power[bin] * plan.weights[weightOffset + bin];
      }
      const logMel = Math.log(Math.max(1e-9, Math.min(1e9, melPower)));
      write(mel * frames + frame, logMel);
    }
  }
  return { frames, melBins: MEL_BINS };
}

function float32ToFloat16(value: number): number {
  floatScratch[0] = value;
  const source = floatBits[0];
  const sign = (source >>> 16) & 0x8000;
  let exponent = ((source >>> 23) & 0xff) - 127 + 15;
  let mantissa = source & 0x7fffff;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa = (mantissa | 0x800000) >>> (1 - exponent);
    return sign | ((mantissa + 0x0fff + ((mantissa >>> 13) & 1)) >>> 13);
  }
  if (exponent >= 31) return sign | (mantissa ? 0x7e00 : 0x7c00);
  mantissa += 0x0fff + ((mantissa >>> 13) & 1);
  if (mantissa & 0x800000) {
    mantissa = 0;
    exponent += 1;
    if (exponent >= 31) return sign | 0x7c00;
  }
  return sign | (exponent << 10) | (mantissa >>> 13);
}

function float16ToFloat32(value: number): number {
  const sign = value & 0x8000 ? -1 : 1;
  const exponent = (value >>> 10) & 0x1f;
  const mantissa = value & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (mantissa / 1024);
  if (exponent === 31) return mantissa ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * 2 ** (exponent - 15) * (1 + mantissa / 1024);
}

function extractLogMelFloat16(samples: Float32Array): { data: Uint16Array; frames: number } {
  const frames = featureFrameCount(samples.length);
  if (frames === 0) {
    throw new Error(
      `Decoded audio has ${samples.length} samples; GigaAM needs at least ${N_FFT} samples (20 ms).`
    );
  }
  const data = new Uint16Array(MEL_BINS * frames);
  extractLogMel(samples, (index, value) => {
    data[index] = float32ToFloat16(value);
  });
  return { data, frames };
}

function resolveMaxDurationSeconds(value: number | null | undefined): number | null {
  if (value === undefined) return 15;
  if (value === null) return null;
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`maxDurationSeconds must be positive, finite, or null; received ${value}.`);
  }
  return value;
}

function clippedFrameCount(frameCount: number, sampleRate: number, maxDurationSeconds: number | null): number {
  if (maxDurationSeconds === null) return frameCount;
  return Math.min(frameCount, Math.floor(sampleRate * maxDurationSeconds));
}

async function decodeAndResampleAudio(blob: Blob, maxDurationSeconds: number | null): Promise<Float32Array> {
  if (!(blob instanceof Blob) || blob.size === 0) {
    throw new Error('Audio input is empty or is not a Blob.');
  }
  const audioContextConstructor = (
    globalThis as typeof globalThis & { webkitAudioContext?: typeof AudioContext }
  ).AudioContext || (globalThis as typeof globalThis & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!audioContextConstructor) {
    throw new Error('WebAudio AudioContext is unavailable in this browser context.');
  }
  const context = new audioContextConstructor();
  let decoded: AudioBuffer;
  try {
    decoded = await context.decodeAudioData(await blob.arrayBuffer());
  } finally {
    await context.close().catch(() => undefined);
  }
  if (decoded.numberOfChannels <= 0 || decoded.length <= 0 || decoded.sampleRate <= 0) {
    throw new Error('WebAudio decoded an empty or invalid audio buffer.');
  }
  const sourceFrames = clippedFrameCount(decoded.length, decoded.sampleRate, maxDurationSeconds);
  if (sourceFrames <= 0) throw new Error('Duration clipping removed all decoded audio samples.');
  if (decoded.sampleRate === SAMPLE_RATE) {
    const mono = new Float32Array(sourceFrames);
    for (let channel = 0; channel < decoded.numberOfChannels; channel += 1) {
      const source = decoded.getChannelData(channel);
      for (let frame = 0; frame < sourceFrames; frame += 1) mono[frame] += source[frame];
    }
    const scale = 1 / decoded.numberOfChannels;
    for (let frame = 0; frame < sourceFrames; frame += 1) mono[frame] *= scale;
    return mono;
  }
  if (typeof OfflineAudioContext === 'undefined') {
    throw new Error('WebAudio OfflineAudioContext is unavailable for 16 kHz resampling.');
  }
  const outputFrames = Math.max(1, Math.round((sourceFrames * SAMPLE_RATE) / decoded.sampleRate));
  const offline = new OfflineAudioContext(1, outputFrames, SAMPLE_RATE);
  const monoBuffer = offline.createBuffer(1, sourceFrames, decoded.sampleRate);
  const mono = monoBuffer.getChannelData(0);
  for (let channel = 0; channel < decoded.numberOfChannels; channel += 1) {
    const source = decoded.getChannelData(channel);
    for (let frame = 0; frame < sourceFrames; frame += 1) mono[frame] += source[frame];
  }
  const scale = 1 / decoded.numberOfChannels;
  for (let frame = 0; frame < sourceFrames; frame += 1) mono[frame] *= scale;
  const sourceNode = offline.createBufferSource();
  sourceNode.buffer = monoBuffer;
  sourceNode.connect(offline.destination);
  sourceNode.start();
  const rendered = await offline.startRendering();
  return rendered.getChannelData(0).slice();
}

function configureOrtRuntime(): void {
  if (ortConfigured) return;
  const runtime = (globalThis as typeof globalThis & {
    chrome?: { runtime?: { getURL?: (path: string) => string } };
  }).chrome?.runtime;
  if (typeof runtime?.getURL !== 'function') {
    throw new Error('chrome.runtime.getURL is unavailable; cannot locate bundled ONNX Runtime WASM files.');
  }
  // Keep the WASM fallback single-threaded because Chrome MV3 forbids the
  // blob: worker used by ORT; WebGPU remains the primary execution provider.
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = runtime.getURL('dist/vendor/ort/');
  ortConfigured = true;
}

async function cachedArrayBuffer(path: string): Promise<ArrayBuffer> {
  const response = await getCachedLocalModelFile(path, LOCAL_MODEL_BASE_URL);
  if (!response) {
    throw new Error(`Required cached model file "${path}" is missing. Install local models in Options.`);
  }
  if (!response.ok) {
    throw new Error(`Cached model file "${path}" returned HTTP ${response.status}. Reinstall local models.`);
  }
  const bytes = await response.arrayBuffer();
  if (bytes.byteLength === 0) throw new Error(`Cached model file "${path}" is empty. Reinstall local models.`);
  return bytes;
}

async function cachedJson(path: string): Promise<Record<string, unknown>> {
  const bytes = await cachedArrayBuffer(path);
  try {
    const parsed = JSON.parse(new TextDecoder().decode(bytes));
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) throw new Error('expected a JSON object');
    return parsed as Record<string, unknown>;
  } catch (error) {
    throw new Error(`Cached model file "${path}" is not valid JSON: ${String(error)}`);
  }
}

function requireSessionNames(
  session: Session,
  requiredInputs: readonly string[],
  requiredOutputs: readonly string[],
  modelName: string
): void {
  for (const name of requiredInputs) {
    if (!session.inputNames.includes(name)) {
      throw new Error(`${modelName} graph is missing required input "${name}" (has ${session.inputNames.join(', ')}).`);
    }
  }
  for (const name of requiredOutputs) {
    if (!session.outputNames.includes(name)) {
      throw new Error(`${modelName} graph is missing required output "${name}" (has ${session.outputNames.join(', ')}).`);
    }
  }
}

async function createInferenceSession(path: string): Promise<Session> {
  configureOrtRuntime();
  const bytes = await cachedArrayBuffer(path);
  return ort.InferenceSession.create(bytes, {
    executionProviders: ['webgpu', 'wasm'],
    graphOptimizationLevel: 'all'
  });
}

async function getAsrSession(): Promise<Session> {
  if (!asrSessionPromise) {
    asrSessionPromise = (async () => {
      const session = await createInferenceSession(ASR_MODEL_PATH);
      requireSessionNames(
        session,
        ['features', 'feature_lengths'],
        ['log_probs', 'encoded_lengths'],
        'GigaAM CTC'
      );
      return session;
    })().catch((error) => {
      asrSessionPromise = null;
      throw actionableError('ASR initialization', error);
    });
  }
  return asrSessionPromise;
}

function validatePunctuationLabels(config: Record<string, unknown>): readonly PunctuationLabel[] {
  const id2label = config.id2label;
  if (!id2label || typeof id2label !== 'object' || Array.isArray(id2label)) {
    throw new Error('Punctuation config.json has no id2label object.');
  }
  const labels = PUNCTUATION_LABELS.map((expected, index) => {
    const actual = (id2label as Record<string, unknown>)[String(index)];
    if (actual !== expected) {
      throw new Error(`Punctuation label ${index} must be "${expected}", received "${String(actual)}".`);
    }
    return expected;
  });
  return labels;
}

async function getPunctuationResources(): Promise<PunctuationResources> {
  if (!punctuationResourcesPromise) {
    punctuationResourcesPromise = (async () => {
      const [session, modelConfig, tokenizerJson, tokenizerConfig] = await Promise.all([
        createInferenceSession(PUNCTUATION_MODEL_PATH),
        cachedJson(PUNCTUATION_CONFIG_PATH),
        cachedJson(TOKENIZER_PATH),
        cachedJson(TOKENIZER_CONFIG_PATH)
      ]);
      requireSessionNames(session, ['input_ids', 'attention_mask'], ['logits'], 'punctuation');
      for (const input of session.inputNames) {
        if (!['input_ids', 'attention_mask', 'token_type_ids'].includes(input)) {
          throw new Error(`Punctuation graph has unsupported required input "${input}".`);
        }
      }
      const tokenizer = new BertTokenizer(tokenizerJson, tokenizerConfig);
      return { session, tokenizer, labels: validatePunctuationLabels(modelConfig) };
    })().catch((error) => {
      punctuationResourcesPromise = null;
      throw actionableError('punctuation initialization', error);
    });
  }
  return punctuationResourcesPromise;
}

function tensorEncodedLength(tensor: ort.Tensor): number {
  if (tensor.dims.length !== 1 || tensor.dims[0] !== 1 || tensor.data.length !== 1) {
    throw new Error(`encoded_lengths must have shape [1], received [${tensor.dims.join(', ')}].`);
  }
  const value = Number(tensor.data[0]);
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`encoded_lengths[0] must be a positive safe integer, received ${String(tensor.data[0])}.`);
  }
  return value;
}

function tensorNumericValue(data: ort.Tensor['data'], index: number, type: string): number {
  const raw = data[index];
  if (typeof raw === 'bigint') return Number(raw);
  if (typeof raw === 'string') return Number.NaN;
  return type === 'float16' && data instanceof Uint16Array
    ? float16ToFloat32(Number(raw))
    : Number(raw);
}

function decodeCtcTensor(
  logProbabilities: ort.Tensor,
  encodedLength: number,
  durationSeconds: number
): DecodedCtc {
  const dims = logProbabilities.dims;
  if (dims.length !== 3 || dims[0] !== 1 || dims[2] !== ASR_CLASS_COUNT) {
    throw new Error(`log_probs must have shape [1, T, ${ASR_CLASS_COUNT}], received [${dims.join(', ')}].`);
  }
  const timeSteps = dims[1];
  if (!Number.isInteger(timeSteps) || encodedLength > timeSteps) {
    throw new Error(`encoded_lengths[0]=${encodedLength} exceeds log_probs time dimension ${timeSteps}.`);
  }
  if (logProbabilities.data.length !== timeSteps * ASR_CLASS_COUNT) {
    throw new Error(
      `log_probs data has ${logProbabilities.data.length} values; expected ${timeSteps * ASR_CLASS_COUNT}.`
    );
  }
  const tokenIds: number[] = [];
  const tokenFrames: number[] = [];
  let previous = -1;
  for (let frame = 0; frame < encodedLength; frame += 1) {
    const offset = frame * ASR_CLASS_COUNT;
    let bestClass = 0;
    let bestValue = tensorNumericValue(logProbabilities.data, offset, logProbabilities.type);
    for (let classIndex = 1; classIndex < ASR_CLASS_COUNT; classIndex += 1) {
      const value = tensorNumericValue(logProbabilities.data, offset + classIndex, logProbabilities.type);
      if (value > bestValue) {
        bestClass = classIndex;
        bestValue = value;
      }
    }
    if (bestClass !== previous && bestClass !== ASR_BLANK_ID) {
      tokenIds.push(bestClass);
      tokenFrames.push(frame);
    }
    previous = bestClass;
  }
  const frameShift = durationSeconds / encodedLength;
  const words: LocalWord[] = [];
  let characters = '';
  let firstFrame = -1;
  let lastFrame = -1;
  const commit = () => {
    const text = characters.trim();
    if (text && firstFrame >= 0 && lastFrame >= firstFrame) {
      words.push({
        text,
        startSeconds: firstFrame * frameShift,
        endSeconds: (lastFrame + 1) * frameShift
      });
    }
    characters = '';
    firstFrame = -1;
    lastFrame = -1;
  };
  let rawText = '';
  for (let index = 0; index < tokenIds.length; index += 1) {
    const token = ASR_VOCABULARY[tokenIds[index]];
    if (token === undefined) throw new Error(`CTC emitted unknown vocabulary id ${tokenIds[index]}.`);
    rawText += token;
    if (token === ' ') {
      commit();
      continue;
    }
    if (firstFrame < 0) firstFrame = tokenFrames[index];
    lastFrame = tokenFrames[index];
    characters += token;
  }
  commit();
  return { rawText, words };
}

function casefoldWord(word: string): string {
  return word.normalize('NFKC').toLocaleLowerCase('ru-RU').replace(/ß/g, 'ss');
}

function tokenizerSpecialId(tokenizer: Tokenizer, property: string, fallbackToken: string): number {
  const value = (tokenizer as unknown as Record<string, unknown>)[property];
  if (Number.isInteger(value) && Number(value) >= 0) return Number(value);
  const fallback = tokenizer.convert_tokens_to_ids(fallbackToken);
  if (!Number.isInteger(fallback) || fallback < 0) {
    throw new Error(`Tokenizer has no usable ${property}.`);
  }
  return fallback;
}

function encodePunctuationWords(tokenizer: Tokenizer, words: readonly string[]): {
  inputIds: BigInt64Array;
  attentionMask: BigInt64Array;
  tokenTypeIds: BigInt64Array;
  firstSubtokenPositions: Uint16Array;
} {
  const encodedWords: number[][] = [];
  let tokenCount = 2;
  for (let index = 0; index < words.length; index += 1) {
    const word = words[index];
    if (typeof word !== 'string' || !word || word.trim() !== word || /\s/u.test(word)) {
      throw new Error(`Punctuation source word ${index} is not one non-empty lexical token.`);
    }
    const ids = tokenizer.encode(casefoldWord(word), { add_special_tokens: false });
    if (!ids.length || ids.some((id) => !Number.isInteger(id) || id < 0)) {
      throw new Error(`Punctuation tokenizer cannot represent source word ${index} ("${word}").`);
    }
    encodedWords.push(ids);
    tokenCount += ids.length;
  }
  if (tokenCount > PUNCTUATION_MAX_TOKENS) {
    throw new Error(`Punctuation token sequence needs ${tokenCount} tokens, above model limit ${PUNCTUATION_MAX_TOKENS}.`);
  }
  const inputIds = new BigInt64Array(tokenCount);
  const attentionMask = new BigInt64Array(tokenCount);
  const tokenTypeIds = new BigInt64Array(tokenCount);
  const firstSubtokenPositions = new Uint16Array(words.length);
  inputIds[0] = BigInt(tokenizerSpecialId(tokenizer, 'cls_token_id', '[CLS]'));
  attentionMask[0] = 1n;
  let offset = 1;
  for (let wordIndex = 0; wordIndex < encodedWords.length; wordIndex += 1) {
    firstSubtokenPositions[wordIndex] = offset;
    for (const id of encodedWords[wordIndex]) {
      inputIds[offset] = BigInt(id);
      attentionMask[offset] = 1n;
      offset += 1;
    }
  }
  inputIds[offset] = BigInt(tokenizerSpecialId(tokenizer, 'sep_token_id', '[SEP]'));
  attentionMask[offset] = 1n;
  return { inputIds, attentionMask, tokenTypeIds, firstSubtokenPositions };
}

function firstSubtokenLabels(
  predictionIds: ArrayLike<number>,
  firstSubtokenPositions: ArrayLike<number>,
  wordCount: number,
  labels: readonly PunctuationLabel[] = PUNCTUATION_LABELS
): PunctuationLabel[] {
  if (firstSubtokenPositions.length !== wordCount) {
    throw new Error(
      `Punctuation tokenizer represented ${firstSubtokenPositions.length} first subtokens for ${wordCount} words.`
    );
  }
  const result: PunctuationLabel[] = [];
  let previous = -1;
  for (let wordIndex = 0; wordIndex < wordCount; wordIndex += 1) {
    const position = Number(firstSubtokenPositions[wordIndex]);
    if (!Number.isInteger(position) || position <= previous || position >= predictionIds.length) {
      throw new Error(`Punctuation tokenizer lost or reordered source word ${wordIndex}.`);
    }
    const labelId = Number(predictionIds[position]);
    const label = labels[labelId];
    if (!label) throw new Error(`Punctuation model emitted unsupported label id ${labelId} at word ${wordIndex}.`);
    result.push(label);
    previous = position;
  }
  return result;
}

async function predictPunctuationChunk(words: readonly string[]): Promise<PunctuationLabel[]> {
  if (!words.length) return [];
  const resources = await getPunctuationResources();
  let encoded: ReturnType<typeof encodePunctuationWords>;
  try {
    encoded = encodePunctuationWords(resources.tokenizer, words);
  } catch (error) {
    if (words.length > 1 && error instanceof Error && error.message.includes('above model limit')) {
      const middle = Math.floor(words.length / 2);
      return [
        ...(await predictPunctuationChunk(words.slice(0, middle))),
        ...(await predictPunctuationChunk(words.slice(middle)))
      ];
    }
    throw error;
  }
  const sequenceLength = encoded.inputIds.length;
  const feeds: Record<string, ort.Tensor> = {
    input_ids: new ort.Tensor('int64', encoded.inputIds, [1, sequenceLength]),
    attention_mask: new ort.Tensor('int64', encoded.attentionMask, [1, sequenceLength])
  };
  if (resources.session.inputNames.includes('token_type_ids')) {
    feeds.token_type_ids = new ort.Tensor('int64', encoded.tokenTypeIds, [1, sequenceLength]);
  }
  const output = await resources.session.run(feeds);
  const logits = output.logits;
  if (!logits) throw new Error('Punctuation graph did not return logits.');
  if (
    logits.dims.length !== 3 ||
    logits.dims[0] !== 1 ||
    logits.dims[1] !== sequenceLength ||
    logits.dims[2] !== PUNCTUATION_LABELS.length
  ) {
    throw new Error(
      `Punctuation logits must have shape [1, ${sequenceLength}, ${PUNCTUATION_LABELS.length}], received [${logits.dims.join(', ')}].`
    );
  }
  if (logits.data.length !== sequenceLength * PUNCTUATION_LABELS.length) {
    throw new Error('Punctuation logits data length does not cover every tokenizer position and label.');
  }
  const predictions = new Uint8Array(sequenceLength);
  for (let token = 0; token < sequenceLength; token += 1) {
    const offset = token * PUNCTUATION_LABELS.length;
    let bestLabel = 0;
    let bestValue = tensorNumericValue(logits.data, offset, logits.type);
    for (let label = 1; label < PUNCTUATION_LABELS.length; label += 1) {
      const value = tensorNumericValue(logits.data, offset + label, logits.type);
      if (value > bestValue) {
        bestLabel = label;
        bestValue = value;
      }
    }
    predictions[token] = bestLabel;
  }
  return firstSubtokenLabels(
    predictions,
    encoded.firstSubtokenPositions,
    words.length,
    resources.labels
  );
}

async function predictPunctuation(words: readonly string[]): Promise<PunctuationLabel[]> {
  const labels: PunctuationLabel[] = [];
  for (let offset = 0; offset < words.length; offset += PUNCTUATION_CHUNK_WORDS) {
    labels.push(...(await predictPunctuationChunk(words.slice(offset, offset + PUNCTUATION_CHUNK_WORDS))));
  }
  if (labels.length !== words.length) {
    throw new Error(`Punctuation model covered ${labels.length} of ${words.length} source words.`);
  }
  return labels;
}

function capitalizeLexicalToken(word: string): string {
  const characters = Array.from(word);
  for (let index = 0; index < characters.length; index += 1) {
    const upper = characters[index].toLocaleUpperCase('ru-RU');
    if (upper !== characters[index]) {
      characters[index] = upper;
      return characters.join('');
    }
    if (characters[index].toLocaleLowerCase('ru-RU') !== characters[index]) return word;
  }
  return word;
}

function renderBoundaryLabels(
  words: readonly string[],
  labels: readonly PunctuationLabel[],
  sentenceStart = true
): { text: string; sentenceStart: boolean } {
  if (words.length !== labels.length) {
    throw new Error(`Punctuation label count ${labels.length} does not match source word count ${words.length}.`);
  }
  let text = '';
  let previousLabel: PunctuationLabel | null = null;
  let nextSentenceStart = sentenceStart;
  for (let index = 0; index < words.length; index += 1) {
    const word = words[index];
    const label = labels[index];
    const labelIndex = PUNCTUATION_LABELS.indexOf(label);
    if (labelIndex < 0) throw new Error(`Unsupported punctuation label "${String(label)}".`);
    const displayed = nextSentenceStart ? capitalizeLexicalToken(word) : word;
    if (text) text += previousLabel === 'HYPHEN_JOIN' ? '' : ' ';
    text += displayed + PUNCTUATION_SUFFIXES[labelIndex];
    previousLabel = label;
    nextSentenceStart = label === 'PERIOD' || label === 'QUESTION';
  }
  return { text, sentenceStart: nextSentenceStart };
}

async function recognizeSampleChunk(samples: Float32Array): Promise<SampleRecognition> {
  const durationSeconds = samples.length / SAMPLE_RATE;
  const features = extractLogMelFloat16(samples);
  const session = await getAsrSession();
  const output = await session.run({
    features: new ort.Tensor('float16', features.data, [1, MEL_BINS, features.frames]),
    feature_lengths: new ort.Tensor('int64', BigInt64Array.of(BigInt(features.frames)), [1])
  });
  const logProbabilities = output.log_probs;
  const encodedLengths = output.encoded_lengths;
  if (!logProbabilities || !encodedLengths) {
    throw new Error('GigaAM graph did not return both log_probs and encoded_lengths.');
  }
  const decoded = decodeCtcTensor(
    logProbabilities,
    tensorEncodedLength(encodedLengths),
    durationSeconds
  );
  return { durationSeconds, tokens: decoded.words };
}

function isOverlapDuplicate(existing: LocalWord, incoming: LocalWord, chunkStartSeconds: number): boolean {
  if (casefoldWord(existing.text) !== casefoldWord(incoming.text)) return false;
  if (
    existing.endSeconds < chunkStartSeconds ||
    incoming.startSeconds > chunkStartSeconds + ASR_CHUNK_OVERLAP_SECONDS
  ) {
    return false;
  }
  const overlap = Math.min(existing.endSeconds, incoming.endSeconds) -
    Math.max(existing.startSeconds, incoming.startSeconds);
  if (overlap > 0) return true;
  const existingCenter = (existing.startSeconds + existing.endSeconds) / 2;
  const incomingCenter = (incoming.startSeconds + incoming.endSeconds) / 2;
  return Math.abs(existingCenter - incomingCenter) <= 0.12;
}

async function recognizeSamplesInChunks(
  samples: Float32Array,
  recognizer: SampleRecognizer = recognizeSampleChunk
): Promise<SampleRecognition> {
  const tokens: LocalWord[] = [];
  for (let startSample = 0; startSample < samples.length; startSample += ASR_CHUNK_STRIDE_SAMPLES) {
    const endSample = Math.min(samples.length, startSample + ASR_CHUNK_SAMPLES);
    const recognized = await recognizer(samples.subarray(startSample, endSample), startSample);
    const startSeconds = startSample / SAMPLE_RATE;
    for (const token of recognized.tokens) {
      const offsetToken = {
        text: token.text,
        startSeconds: token.startSeconds + startSeconds,
        endSeconds: token.endSeconds + startSeconds
      };
      if (
        startSample > 0 &&
        tokens.some((existing) => isOverlapDuplicate(existing, offsetToken, startSeconds))
      ) {
        continue;
      }
      tokens.push(offsetToken);
    }
    if (endSample === samples.length) break;
  }
  tokens.sort(
    (left, right) =>
      left.startSeconds - right.startSeconds ||
      left.endSeconds - right.endSeconds ||
      left.text.localeCompare(right.text)
  );
  return { durationSeconds: samples.length / SAMPLE_RATE, tokens };
}


async function transcribeSamples(
  samples: Float32Array,
  recognizer?: SampleRecognizer,
  punctuator: WordPunctuator = predictPunctuation
): Promise<TranscriptWithLabels> {
  const recognized = recognizer
    ? await recognizer(samples, 0)
    : await recognizeSamplesInChunks(samples);
  const words = recognized.tokens.map((word) => word.text);
  const labels = await punctuator(words);
  const rendered = renderBoundaryLabels(words, labels);
  if (recognized.tokens.length && !rendered.text) {
    throw new Error('Punctuation rendering produced empty text for non-empty ASR words.');
  }
  return { text: rendered.text, ...recognized, labels };
}

export async function transcribeLocalAudio(
  blob: Blob,
  options: { maxDurationSeconds?: number | null } = {}
): Promise<LocalTranscriptResult> {
  try {
    const maxDurationSeconds = resolveMaxDurationSeconds(options.maxDurationSeconds);
    const samples = await decodeAndResampleAudio(blob, maxDurationSeconds);
    const result = await transcribeSamples(samples);
    return { text: result.text, durationSeconds: result.durationSeconds, tokens: result.tokens };
  } catch (error) {
    throw actionableError('transcription', error);
  }
}


function emitTimingStatus(
  callbacks: L0TimingRequestCallbacks | undefined,
  status: L0TimingQueueStatus
): void {
  try {
    callbacks?.onQueueStatus?.(status);
  } catch {
    // Queue presentation is observational and must never affect transcription.
  }
}

function modelsSummary(): Record<string, unknown> {
  return {
    asr: {
      name: 'gigaam-v3-ctc-domain',
      runtime: 'onnxruntime-web',
      executionProviders: ['webgpu', 'wasm'],
      inputDtype: 'float16'
    },
    l2: {
      name: 'punctuation-production-spacing-int8',
      runtime: 'onnxruntime-web',
      executionProviders: ['webgpu', 'wasm'],
      labels: [...PUNCTUATION_LABELS]
    }
  };
}

export async function generateLocalL0Timing(
  _settings: ExtensionSettings,
  job: TranscriptJob,
  audioTracks: CapturedAudioTrack[],
  callbacks?: L0TimingRequestCallbacks
): Promise<L0TimingResponse> {
  const startedAt = performance.now();
  const prepared = prepareL0TimingTracks(job, audioTracks);
  const taskId = buildCanonicalTaskIdentity(job);
  const requestId = `browser-local:${taskId}`;
  emitTimingStatus(callbacks, { requestId, status: 'preparing' });
  emitTimingStatus(callbacks, { requestId, status: 'running', position: 0, queuedCount: 0 });
  const tracks: L0TimingResponse['tracks'] = [];
  for (const track of prepared) {
    let result: { durationSeconds: number; tokens: LocalWord[] };
    try {
      const samples = await decodeAndResampleAudio(track.audio.blob, null);
      result = await recognizeSamplesInChunks(samples);
    } catch (error) {
      throw actionableError(`timing lane "${track.lane}"`, error);
    }
    tracks.push({
      lane: track.lane,
      tokens: result.tokens.map((token, tokenIndex) => ({
        id: `${taskId}:${track.lane}:${tokenIndex}`,
        ...token
      }))
    });
  }
  const tokenCount = tracks.reduce((total, track) => total + track.tokens.length, 0);
  emitTimingStatus(callbacks, { requestId, status: 'completed', position: 0, queuedCount: 0 });
  return {
    taskId,
    tracks,
    summary: {
      taskId,
      trackCount: tracks.length,
      tokenCount,
      latencyMs: Math.round(performance.now() - startedAt),
      provider: 'browser-local'
    },
    models: modelsSummary()
  };
}

function cropSampleInterval(
  samples: Float32Array,
  startSeconds: number,
  endSeconds: number
): Float32Array {
  if (
    !Number.isFinite(startSeconds) ||
    !Number.isFinite(endSeconds) ||
    startSeconds < 0 ||
    endSeconds <= startSeconds
  ) {
    throw new Error(`Segment interval must be finite and positive; received ${startSeconds}-${endSeconds}.`);
  }
  const startSample = Math.round(startSeconds * SAMPLE_RATE);
  const endSample = Math.round(endSeconds * SAMPLE_RATE);
  if (
    !Number.isSafeInteger(startSample) ||
    !Number.isSafeInteger(endSample) ||
    startSample >= samples.length ||
    endSample > samples.length ||
    endSample <= startSample
  ) {
    throw new Error(
      `Segment interval ${startSeconds}-${endSeconds} is outside decoded track duration ` +
      `${samples.length / SAMPLE_RATE}.`
    );
  }
  return samples.slice(startSample, endSample);
}

async function transcribeSampleInterval(
  samples: Float32Array,
  startSeconds: number,
  endSeconds: number,
  recognizer?: SampleRecognizer,
  punctuator: WordPunctuator = predictPunctuation
): Promise<TranscriptWithLabels> {
  const cropped = cropSampleInterval(samples, startSeconds, endSeconds);
  return transcribeSamples(cropped, recognizer, punctuator);
}

export async function generateLocalL0SegmentDraft(
  _settings: ExtensionSettings,
  _taskId: string,
  row: TranscriptJob['rows'][number],
  tracks: PreparedL0Track[]
): Promise<string> {
  const targetLane = row.speakerKey.trim().toLocaleLowerCase();
  const targetTrack = tracks.find(
    (track) => track.lane.trim().toLocaleLowerCase() === targetLane
  );
  if (!targetTrack) {
    throw new Error(`Local L0 segment drafting has no audio track for speaker lane "${row.speakerKey}".`);
  }
  if (row.startSeconds === null || row.endSeconds === null) {
    throw new Error('Local L0 segment drafting requires finite row start and end timestamps.');
  }
  try {
    const samples = await decodeAndResampleAudio(targetTrack.audio.blob, null);
    const transcript = await transcribeSampleInterval(samples, row.startSeconds, row.endSeconds);
    return transcript.text;
  } catch (error) {
    throw actionableError(`segment lane "${targetTrack.lane}"`, error);
  }
}

function groupWordRows(words: readonly LocalWord[]): Array<{ start: number; end: number }> {
  const groups: Array<{ start: number; end: number }> = [];
  if (!words.length) return groups;
  let start = 0;
  for (let index = 1; index < words.length; index += 1) {
    const shouldSplit =
      words[index].startSeconds - words[index - 1].endSeconds >= 0.8 ||
      words[index].endSeconds - words[start].startSeconds > 12 ||
      index - start >= 32;
    if (shouldSplit) {
      groups.push({ start, end: index });
      start = index;
    }
  }
  groups.push({ start, end: words.length });
  return groups;
}

export async function generateLocalL0Draft(
  _settings: ExtensionSettings,
  job: TranscriptJob,
  audioTracks: CapturedAudioTrack[]
): Promise<L0DraftResponse> {
  const startedAt = performance.now();
  const transcriptLanes = new Set(
    job.rows.map((row) => row.speakerKey.trim().toLocaleLowerCase()).filter(Boolean)
  );
  const prepared =
    transcriptLanes.size === 1
      ? prepareL0TimingTracks(job, audioTracks)
      : prepareL0Tracks(job, audioTracks);
  const rows: Array<{ id: string; lane: string; startSeconds: number; endSeconds: number; text: string }> = [];
  let wordCount = 0;
  for (let laneIndex = 0; laneIndex < prepared.length; laneIndex += 1) {
    const track = prepared[laneIndex];
    let transcript: TranscriptWithLabels;
    try {
      const samples = await decodeAndResampleAudio(track.audio.blob, null);
      transcript = await transcribeSamples(samples);
    } catch (error) {
      throw actionableError(`draft lane "${track.lane}"`, error);
    }
    wordCount += transcript.tokens.length;
    const groups = groupWordRows(transcript.tokens);
    let sentenceStart = true;
    for (let groupIndex = 0; groupIndex < groups.length; groupIndex += 1) {
      const group = groups[groupIndex];
      const lexicalWords = transcript.tokens.slice(group.start, group.end).map((word) => word.text);
      const rendered = renderBoundaryLabels(
        lexicalWords,
        transcript.labels.slice(group.start, group.end),
        sentenceStart
      );
      sentenceStart = rendered.sentenceStart;
      const first = transcript.tokens[group.start];
      const last = transcript.tokens[group.end - 1];
      rows.push({
        id: `${job.jobId}:${track.lane}:${String(groupIndex).padStart(6, '0')}`,
        lane: track.lane,
        startSeconds: Math.max(0, Number((first.startSeconds - 0.01).toFixed(6))),
        endSeconds: Number(Math.min(transcript.durationSeconds, last.endSeconds + 0.01).toFixed(6)),
        text: rendered.text
      });
    }
  }
  const laneOrder = new Map(prepared.map((track, index) => [track.lane, index]));
  rows.sort(
    (left, right) =>
      left.startSeconds - right.startSeconds ||
      (laneOrder.get(left.lane) ?? 0) - (laneOrder.get(right.lane) ?? 0) ||
      left.endSeconds - right.endSeconds
  );
  if (!rows.length || rows.some((row) => !row.text || row.endSeconds <= row.startSeconds)) {
    throw new Error('Local draft did not produce non-empty, positive-duration rows.');
  }
  return {
    rows,
    summary: {
      taskId: job.jobId,
      trackCount: prepared.length,
      rowCount: rows.length,
      wordCount,
      latencyMs: Math.round(performance.now() - startedAt),
      provider: 'browser-local'
    },
    models: modelsSummary()
  } as L0DraftResponse;
}

/** Pure deterministic hooks used by the focused browser-DSP contract tests. */
export const __localModelRuntimeTesting = {
  extractLogMelFeatures(samples: Float32Array): { data: Float32Array; frames: number; melBins: number } {
    const frames = featureFrameCount(samples.length);
    if (!frames) {
      extractLogMel(samples, () => undefined);
      throw new Error('unreachable');
    }
    const data = new Float32Array(MEL_BINS * frames);
    extractLogMel(samples, (index, value) => {
      data[index] = value;
    });
    return { data, frames, melBins: MEL_BINS };
  },
  decodeCtc(
    values: Float32Array,
    timeSteps: number,
    encodedLength: number,
    durationSeconds: number
  ): DecodedCtc {
    return decodeCtcTensor(
      new ort.Tensor('float32', values, [1, timeSteps, ASR_CLASS_COUNT]),
      encodedLength,
      durationSeconds
    );
  },
  readFloat16Value(data: ArrayLike<number>, index = 0): number {
    return tensorNumericValue(data as ort.Tensor['data'], index, 'float16');
  },
  async transcribeWithRecognizer(
    samples: Float32Array,
    recognizer: SampleRecognizer
  ): Promise<LocalTranscriptResult> {
    const transcript = await transcribeSamples(
      samples,
      recognizer,
      async (words) => words.map(() => 'O' as const)
    );
    return {
      text: transcript.text,
      durationSeconds: transcript.durationSeconds,
      tokens: transcript.tokens
    };
  },
  recognizeSamplesInChunks,
  transcribeSampleInterval,
  cropSampleInterval,
  encodePunctuationWords,
  firstSubtokenLabels,
  renderBoundaryLabels,
  resolveMaxDurationSeconds,
  clippedFrameCount,
  groupWordRows,
  float32ToFloat16,
  float16ToFloat32
};
