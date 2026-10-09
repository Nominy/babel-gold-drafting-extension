import { BertTokenizer } from '@huggingface/transformers';
import { INFERENCE_RELEASE, assertReleasedGraphs } from './inference-release';
import * as ort from 'onnxruntime-web/webgpu';
import { runExclusiveGpuInference } from './local-gpu-run-queue';

import { CHECKPOINT_FRONTEND_BF16 } from './gigaam-frontend-buffers';
import { denoiseForActivity } from './ffmpeg-audio-denoise';
import { buildActivityRows, segmentSamplesByActivity, type ActivitySegment } from './local-audio-segmentation';
import { prepareRawPcm16, highpassSource, resampleToPcm16 } from './ffmpeg-audio-raw';
import {
  boundaryCenters, extractProsody, float16ToFloat32, float32ToFloat16, resamplePoly,
  sampleLocalAudio, type AcousticFrames
} from './c-denoise-acoustic';
import {
  ASR_SOURCE_SHA, C_DENOISE_SOURCE_SHA, C_DENOISE_LABELS, predictCDenoise,
  validateCDenoiseConfig, type CDenoiseConfig, type CDenoiseResources, type CDenoiseLabel
} from './c-denoise-runtime';
import type { PreparedL0Track } from './l0-client';
import { getCachedBundleDescriptor, getCachedLocalModelFile } from './local-model-bundle';
import {
  bindPlacementGraph, GpuRunAudit, validatePlacementPolicy,
  type GraphPlacementDiagnostic, type PlacementGraph, type PlacementPolicy
} from './local-gpu-placement';
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
const PUNCTUATION_LABELS = C_DENOISE_LABELS;
const PUNCTUATION_SUFFIXES = ['', ',', '.', '?', '-', '-', '--'] as const;
const ASR_SILENCE_TARGET_SAMPLES = 22 * SAMPLE_RATE;
const ASR_SILENCE_MAX_SAMPLES = 24 * SAMPLE_RATE;
const ASR_SILENCE_SEARCH_SAMPLES = 2 * SAMPLE_RATE;
const ASR_ENERGY_WINDOW_SAMPLES = Math.round(0.12 * SAMPLE_RATE);

const ASR_MODEL_PATH = 'asr/v3_ctc.onnx';
const PUNCTUATION_CONTEXT_PATH = 'punctuation/context.fp16.onnx';
const PUNCTUATION_DENOISE_PATH = 'punctuation/denoise.fp16.onnx';
const C_DENOISE_CONFIG_PATH = 'punctuation/c-denoise.json';
const PUNCTUATION_CONFIG_PATH = 'punctuation/config.json';
const TOKENIZER_PATH = 'punctuation/tokenizer.json';
const TOKENIZER_CONFIG_PATH = 'punctuation/tokenizer_config.json';
const GPU_PLACEMENT_PATH = 'punctuation/gpu-placement.json';

export interface LocalTranscriptResult {
  text: string;
  durationSeconds: number;
  readonly execution?: LocalExecutionDiagnostic;
  tokens: Array<{
    text: string;
    startSeconds: number;
    endSeconds: number;
  }>;
}

type LocalWord = LocalTranscriptResult['tokens'][number];
type PunctuationLabel = CDenoiseLabel;
type Session = ort.InferenceSession;
type DecodedCtc = { rawText: string; words: LocalWord[] };
type SampleRecognition = { durationSeconds: number; tokens: LocalWord[] };
type SampleRecognizer = (samples: Float32Array, startSample: number) => Promise<SampleRecognition>;


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

export interface LocalAdapterDiagnostic {
  readonly vendor: string;
  readonly architecture: string;
  readonly device: string;
  readonly description: string;
  readonly isFallbackAdapter: false;
  readonly shaderF16: true;
  readonly maxBufferSize: number;
  readonly maxStorageBufferBindingSize: number;
}
export interface LocalExecutionDiagnostic {
  readonly provider: 'webgpu';
  readonly shaderF16: true;
  readonly denoiseSteps: 4;
  readonly neuralCpuFallback: false;
  readonly asrCheckpointSha256: string;
  readonly cDenoiseCheckpointSha256: string;
  readonly bundleIdentity: string;
  readonly adapter: LocalAdapterDiagnostic;
  readonly placementAudit: {
    readonly method: 'sha256-bound-source-nodes-and-webgpu-dispatch-profile';
    readonly graphOptimizations: 'disabled';
    readonly hostMetadataAllowed: true;
    readonly cpuAttemptsPrevented: false;
    readonly graphs: readonly GraphPlacementDiagnostic[];
  };
}
interface PreparedLocalAudio {
  raw: Float32Array;
  activity: Float32Array;
  featureSamples: Float32Array;
  source: Float32Array;
  sourceRate: number;
  pcmSha256: string;
}
interface GpuLimits { maxBufferSize: number; maxStorageBufferBindingSize: number }
interface HardwareGpuDevice {
  limits: GpuLimits;
  features: { has: (feature: string) => boolean };
  queue: { onSubmittedWorkDone: () => Promise<void> };
  lost: Promise<{ message: string }>;
}
type HardwareGpuAdapter = GPUAdapter & { readonly isFallbackAdapter?: boolean };
type CachedLane = { pcmSha256: string; tokens: LocalWord[]; labelIds: Uint8Array; ranges: ReturnType<typeof buildActivityRows> };
type CachedTask = { identity: string; timing: L0TimingResponse; lanes: Map<string, CachedLane> };
const taskCache = new Map<string, CachedTask>();
let currentBundleIdentity: string | null = null;
let gpuPromise: Promise<void> | null = null;
let adapterDiagnostic: LocalAdapterDiagnostic | null = null;
let cDenoiseConfigPromise: Promise<CDenoiseConfig> | null = null;
let placementPolicyPromise: Promise<PlacementPolicy> | null = null;
let gpuDevice: HardwareGpuDevice | null = null;
const placementDiagnostics = new Map<string, GraphPlacementDiagnostic>();

let asrSessionPromise: Promise<Session> | null = null;
let punctuationResourcesPromise: Promise<CDenoiseResources> | null = null;
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

function getMelPlan(): MelPlan {
  if (melPlan) return melPlan;
  const coefficientCount = N_FFT + MEL_BINS * SPECTRUM_BINS;
  const packed = atob(CHECKPOINT_FRONTEND_BF16);
  if (packed.length !== coefficientCount * 2) {
    throw new Error('GigaAM checkpoint frontend buffers have the wrong length.');
  }
  const coefficients = new Float64Array(coefficientCount);
  for (let index = 0; index < coefficientCount; index += 1) {
    floatBits[0] = (packed.charCodeAt(index * 2) | (packed.charCodeAt(index * 2 + 1) << 8)) << 16;
    coefficients[index] = floatScratch[0];
  }
  const hann = coefficients.subarray(0, N_FFT);
  const weights = coefficients.subarray(N_FFT);
  const firstBin = new Uint16Array(MEL_BINS);
  const lastBin = new Uint16Array(MEL_BINS);
  for (let mel = 0; mel < MEL_BINS; mel += 1) {
    let first = SPECTRUM_BINS;
    let last = 0;
    for (let bin = 0; bin < SPECTRUM_BINS; bin += 1) {
      if (weights[mel * SPECTRUM_BINS + bin] > 0) {
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


async function decodeSourceAudio(blob: Blob, maxDurationSeconds: number | null): Promise<{ samples: Float32Array; sampleRate: number }> {
  if (!(blob instanceof Blob) || !blob.size) throw new Error('Audio input is empty or is not a Blob.');
  const bytes = await blob.arrayBuffer();
  const view = new DataView(bytes);
  if (bytes.byteLength >= 12 && view.getUint32(0, false) === 0x52494646 && view.getUint32(8, false) === 0x57415645) {
    const decoded = decodePcm16Wav(bytes);
    const durationSeconds = decoded.samples.length / decoded.sampleRate;
    if (maxDurationSeconds !== null && durationSeconds > maxDurationSeconds) {
      throw new Error(`The selected audio is ${durationSeconds.toFixed(1)} seconds. Choose a sample no longer than ${maxDurationSeconds} seconds.`);
    }
    return decoded;
  }
  const Constructor = globalThis.AudioContext || (globalThis as typeof globalThis & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!Constructor) throw new Error('WebAudio decoding is unavailable.');
  const context = new Constructor();
  let decoded: AudioBuffer;
  try { decoded = await context.decodeAudioData(bytes); } finally { await context.close(); }
  const durationSeconds = decoded.length / decoded.sampleRate;
  if (maxDurationSeconds !== null && durationSeconds > maxDurationSeconds) {
    throw new Error(`The selected audio is ${durationSeconds.toFixed(1)} seconds. Choose a sample no longer than ${maxDurationSeconds} seconds.`);
  }
  const length = decoded.length;
  if (!length || !decoded.numberOfChannels) throw new Error('Decoded audio is empty.');
  const mono = new Float32Array(length);
  for (let channel = 0; channel < decoded.numberOfChannels; channel += 1) {
    const data = decoded.getChannelData(channel);
    for (let sample = 0; sample < length; sample += 1) mono[sample] += data[sample] / decoded.numberOfChannels;
  }
  return { samples: mono, sampleRate: decoded.sampleRate };
}

function decodePcm16Wav(bytes: ArrayBuffer): { samples: Float32Array; sampleRate: number } {
  const view = new DataView(bytes);
  if (
    bytes.byteLength < 44 ||
    view.getUint32(0, false) !== 0x52494646 ||
    view.getUint32(8, false) !== 0x57415645
  ) {
    throw new Error('Local draft requires a RIFF WAV audio track.');
  }
  let channels = 0;
  let sampleRate = 0;
  let bitsPerSample = 0;
  let format = 0;
  let dataOffset = -1;
  let dataLength = 0;
  for (let offset = 12; offset + 8 <= bytes.byteLength;) {
    const chunkSize = view.getUint32(offset + 4, true);
    const next = offset + 8 + chunkSize + (chunkSize & 1);
    if (next > bytes.byteLength + 1) throw new Error('WAV contains a truncated chunk.');
    const chunk = view.getUint32(offset, false);
    if (chunk === 0x666d7420) {
      if (chunkSize < 16) throw new Error('WAV has an invalid format chunk.');
      format = view.getUint16(offset + 8, true);
      channels = view.getUint16(offset + 10, true);
      sampleRate = view.getUint32(offset + 12, true);
      bitsPerSample = view.getUint16(offset + 22, true);
    } else if (chunk === 0x64617461) {
      dataOffset = offset + 8;
      dataLength = chunkSize;
    }
    offset = next;
  }
  if (format !== 1 || channels !== 1 || bitsPerSample !== 16 || sampleRate <= 0) {
    throw new Error('Local draft requires mono 16-bit PCM WAV audio.');
  }
  if (dataOffset < 0 || dataLength < 2 || dataLength % 2) {
    throw new Error('WAV has no complete PCM16 samples.');
  }
  const samples = new Float32Array(dataLength / 2);
  for (let index = 0; index < samples.length; index += 1) {
    samples[index] = view.getInt16(dataOffset + index * 2, true) / 32_768;
  }
  return { samples, sampleRate };
}

function normalizePcm16(samples: Int16Array): Float32Array {
  const normalized = new Float32Array(samples.length);
  for (let index = 0; index < samples.length; index += 1) {
    normalized[index] = samples[index] / 32_768;
  }
  return normalized;
}

async function prepareDraftAudio(blob: Blob, maxDurationSeconds: number | null = null): Promise<PreparedLocalAudio> {
  const { samples, sampleRate } = await decodeSourceAudio(blob, maxDurationSeconds);
  if (!samples.length || samples.some((value) => !Number.isFinite(value))) throw new Error('Source PCM is empty or nonfinite.');
  const rawPcm = prepareRawPcm16(samples, sampleRate, Math.round(samples.length * SAMPLE_RATE / sampleRate));
  const pcmSha256 = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', rawPcm.buffer as ArrayBuffer)))
    .map((byte) => byte.toString(16).padStart(2, '0')).join('');
  const activity = normalizePcm16(resampleToPcm16(denoiseForActivity(highpassSource(samples, sampleRate), sampleRate), sampleRate, rawPcm.length));
  return { raw: normalizePcm16(rawPcm), activity, featureSamples: resamplePoly(samples, sampleRate, SAMPLE_RATE), source: samples, sourceRate: sampleRate, pcmSha256 };
}

async function draftRowId(
  taskId: string,
  lane: string,
  pcmSha256: string,
  startSample: number,
  endSample: number
): Promise<string> {
  const namespace = Uint8Array.of(
    0x54, 0x05, 0x7e, 0x89, 0xdf, 0xb6, 0x5f, 0x31,
    0x92, 0x5d, 0x61, 0x19, 0xe4, 0x8b, 0xda, 0xc4
  );
  const name = new TextEncoder().encode(
    `${taskId}|${lane}|${pcmSha256}|${startSample}|${endSample}`
  );
  const input = new Uint8Array(namespace.length + name.length);
  input.set(namespace);
  input.set(name, namespace.length);
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-1', input));
  digest[6] = (digest[6] & 0x0f) | 0x50;
  digest[8] = (digest[8] & 0x3f) | 0x80;
  const hex = Array.from(digest.subarray(0, 16), (byte) => byte.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

function configureOrtRuntime(): void {
  if (ortConfigured) return;
  const runtime = (globalThis as typeof globalThis & {
    chrome?: { runtime?: { getURL?: (path: string) => string } };
  }).chrome?.runtime;
  if (typeof runtime?.getURL !== 'function') {
    throw new Error('chrome.runtime.getURL is unavailable; cannot locate bundled ONNX Runtime WASM files.');
  }
  // WASM may run proven host metadata. Neural results are accepted only after
  // every required source compute node has actual GPU dispatch evidence.
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = runtime.getURL('dist/vendor/ort/');
  // Initialization must not replace another model's in-flight observer.
  ort.env.webgpu.profiling ??= { mode: 'default' };
  ort.env.webgpu.profiling.mode = 'default';
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

async function getCDenoiseConfig(): Promise<CDenoiseConfig> {
  if (!cDenoiseConfigPromise) cDenoiseConfigPromise = cachedJson(C_DENOISE_CONFIG_PATH).then(validateCDenoiseConfig).catch((error) => { cDenoiseConfigPromise = null; throw error; });
  return cDenoiseConfigPromise;
}
async function getPlacementPolicy(): Promise<PlacementPolicy> {
  if (!placementPolicyPromise) placementPolicyPromise = cachedJson(GPU_PLACEMENT_PATH)
    .then((value) => validatePlacementPolicy(value, { asrCheckpointSha256: ASR_SOURCE_SHA, cDenoiseCheckpointSha256: C_DENOISE_SOURCE_SHA }))
    .catch((error) => { placementPolicyPromise = null; throw error; });
  return placementPolicyPromise;
}

function auditSession(session: Session, graph: PlacementGraph): Session {
  const run = session.run;
  session.run = ((...args: Parameters<Session['run']>) => {
    const audited = runExclusiveGpuInference(async () => {
      if (!gpuDevice) throw new Error('The hardware WebGPU audit device is unavailable.');
      const device = gpuDevice;
      // Drain prior GPU work before installing this run's profiling observer.
      // Dispatch readback is awaited separately before accepting its outputs.
      await device.queue.onSubmittedWorkDone();
      const audit = new GpuRunAudit(graph);
      const profiling = ort.env.webgpu.profiling!, previousObserver = profiling.ondata;
      profiling.ondata = data => audit.observe(data);
      let outputs: ort.InferenceSession.ReturnType | undefined;
      try {
        outputs = await run.apply(session, args);
        await device.queue.onSubmittedWorkDone();
        const previous = placementDiagnostics.get(graph.path);
        placementDiagnostics.set(graph.path, await audit.finishAfterDispatches((previous?.verifiedRuns ?? 0) + 1));
        return outputs;
      } catch (error) {
        placementDiagnostics.delete(graph.path);
        if (outputs) for (const tensor of Object.values(outputs)) tensor.dispose();
        throw error;
      } finally {
        try { await device.queue.onSubmittedWorkDone(); }
        finally { profiling.ondata = previousObserver; }
      }
    });
    return audited;
  }) as Session['run'];
  return session;
}

async function requireHardwareGpu(): Promise<void> {
  if (!gpuPromise) gpuPromise = (async () => {
    const config = await getCDenoiseConfig();
    const gpu = navigator.gpu;
    if (!gpu) throw new Error('WebGPU is unavailable. Enable hardware acceleration and use a supported Chrome GPU; Local has no CPU/cloud fallback.');
    const existingAdapter = ort.env.webgpu.adapter as HardwareGpuAdapter | undefined;
    const adapter = (existingAdapter ?? await gpu.requestAdapter({ powerPreference: 'high-performance', forceFallbackAdapter: false })) as HardwareGpuAdapter | null;
    if (!adapter || (adapter.info.isFallbackAdapter ?? adapter.isFallbackAdapter) !== false || !adapter.features.has('shader-f16')) throw new Error('Local C-denoise needs a confirmed hardware WebGPU adapter with shader-f16. No CPU/cloud fallback is allowed.');
    if (!adapter.features.has('timestamp-query')) throw new Error('Strict neural WebGPU placement requires hardware timestamp-query profiling. This GPU cannot prove dispatch placement; no CPU neural results are accepted.');
    const required = config.required_gpu_buffer_bytes;
    if (adapter.limits.maxBufferSize < required || adapter.limits.maxStorageBufferBindingSize < required) {
      throw new Error(`WebGPU adapter cannot bind the largest C-denoise weight (${required} bytes): maxBufferSize=${adapter.limits.maxBufferSize}, maxStorageBufferBindingSize=${adapter.limits.maxStorageBufferBindingSize}. Use a GPU/browser with larger limits.`);
    }
    // The pinned JSEP backend owns its device. Supplying this confirmed adapter
    // keeps hardware selection exact without allocating an unused second device.
    if (!existingAdapter) ort.env.webgpu.adapter = adapter;
    const info = adapter.info;
    adapterDiagnostic = Object.freeze({ vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description,
      isFallbackAdapter: false, shaderF16: true, maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize });
  })().catch((error) => { gpuPromise = null; throw error; });
  await gpuPromise;
  const required = (await getCDenoiseConfig()).required_gpu_buffer_bytes;
  if (!adapterDiagnostic || adapterDiagnostic.maxBufferSize < required || adapterDiagnostic.maxStorageBufferBindingSize < required) {
    throw new Error(`The active WebGPU device cannot bind this model's largest weight (${required} bytes). Reopen the extension on a compatible hardware GPU.`);
  }
}

async function createInferenceSession(path: string): Promise<Session> {
  configureOrtRuntime();
  await requireHardwareGpu();
  const config = await getCDenoiseConfig(), policy = await getPlacementPolicy();
  const graph = Object.values(config.graphs).find((candidate) => candidate.path === path);
  const placement = Object.values(policy.graphs).find((candidate) => candidate.path === path);
  if (!graph || !placement) throw new Error(`No verified neural GPU placement policy exists for ${path}.`);
  const bytes = await cachedArrayBuffer(path);
  await bindPlacementGraph(placement, path, bytes);
  const externalData = await Promise.all((graph.external_data ?? []).map(async (file) => ({ path: file.location, data: new Uint8Array(await cachedArrayBuffer(file.path)) })));
  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: [{ name: 'webgpu', preferredLayout: 'NCHW' }],
    // Preserve SHA-bound source node names for per-run dispatch coverage.
    graphOptimizationLevel: 'disabled',
    externalData
  });
  // ORT types this public device handle as Promise<unknown>; JSEP returns its GPUDevice.
  const device = await ort.env.webgpu.device as HardwareGpuDevice | undefined;
  if (!device?.features.has('shader-f16') ||
      !(device.features.has('timestamp-query') || device.features.has('chromium-experimental-timestamp-query-inside-passes')) ||
      device.limits.maxBufferSize < config.required_gpu_buffer_bytes ||
      device.limits.maxStorageBufferBindingSize < config.required_gpu_buffer_bytes) {
    await session.release();
    throw new Error('The actual ORT WebGPU device cannot execute and audit this model; no CPU neural result is accepted.');
  }
  if (gpuDevice !== device) {
    gpuDevice = device;
    adapterDiagnostic = Object.freeze({ ...adapterDiagnostic!, maxBufferSize: device.limits.maxBufferSize,
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize });
    void device.lost.then((lost) => { adapterDiagnostic = null; gpuDevice = null; gpuPromise = null; placementDiagnostics.clear(); taskCache.clear(); currentBundleIdentity = null; console.error(`C-denoise WebGPU device lost: ${lost.message}`); });
  }
  return auditSession(session, placement);
}

async function getAsrSession(): Promise<Session> {
  if (!asrSessionPromise) {
    asrSessionPromise = (async () => {
      const session = await createInferenceSession(ASR_MODEL_PATH);
      requireSessionNames(
        session,
        ['features', 'feature_lengths'],
        ['log_probs', 'encoded_lengths', 'encoder_features'],
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

async function getPunctuationResources(): Promise<CDenoiseResources> {
  if (!punctuationResourcesPromise) punctuationResourcesPromise = (async () => {
    const [context, denoise, config, tokenizerJson, tokenizerConfig, modelConfig] = await Promise.all([
      createInferenceSession(PUNCTUATION_CONTEXT_PATH), createInferenceSession(PUNCTUATION_DENOISE_PATH),
      getCDenoiseConfig(), cachedJson(TOKENIZER_PATH), cachedJson(TOKENIZER_CONFIG_PATH), cachedJson(PUNCTUATION_CONFIG_PATH)
    ]);
    const id2label = modelConfig.id2label as Record<string, unknown> | undefined;
    if (!id2label || PUNCTUATION_LABELS.some((label, index) => id2label[String(index)] !== label)) throw new Error('C-denoise tokenizer/base classifier label order differs.');
    requireSessionNames(context, ['input_ids', 'attention_mask', 'token_type_ids', 'first_subtoken', 'word_mask', 'context_flags'], ['text_features', 'base_logits'], 'C-denoise context');
    requireSessionNames(denoise, ['text_features', 'base_logits', 'local_audio', 'local_audio_mask', 'audio_valid', 'word_mask', 'noisy_labels', 'noise_level'], ['logits'], 'C-denoise core');
    return { context, denoise, config, tokenizer: new BertTokenizer(tokenizerJson, tokenizerConfig) };
  })().catch((error) => { punctuationResourcesPromise = null; throw actionableError('C-denoise initialization', error); });
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

async function runAsr(samples: Float32Array, outputs: readonly string[] = ['log_probs', 'encoded_lengths', 'encoder_features']): Promise<Record<string, ort.Tensor>> {
  const features = extractLogMelFloat16(samples), session = await getAsrSession();
  const featureTensor = new ort.Tensor('float16', features.data, [1, MEL_BINS, features.frames]);
  const lengthTensor = new ort.Tensor('int64', BigInt64Array.of(BigInt(features.frames)), [1]);
  try { return await session.run({ features: featureTensor, feature_lengths: lengthTensor }, outputs); }
  finally { featureTensor.dispose(); lengthTensor.dispose(); }
}

async function recognizeSampleChunk(samples: Float32Array): Promise<SampleRecognition> {
  const durationSeconds = samples.length / SAMPLE_RATE, output = await runAsr(samples, ['log_probs', 'encoded_lengths']);
  try {
    if (!output.log_probs || !output.encoded_lengths) throw new Error('GigaAM graph did not return its accepted CTC outputs.');
    return { durationSeconds, tokens: decodeCtcTensor(output.log_probs, tensorEncodedLength(output.encoded_lengths), durationSeconds).words };
  } finally { for (const tensor of Object.values(output)) tensor.dispose(); }
}


function silenceChunkBoundaries(samples: Float32Array): number[] {
  const boundaries = [0];
  let cursor = 0;
  while (samples.length - cursor > ASR_SILENCE_MAX_SAMPLES) {
    const target = cursor + ASR_SILENCE_TARGET_SAMPLES;
    const left = Math.max(cursor + 12 * SAMPLE_RATE, target - ASR_SILENCE_SEARCH_SAMPLES);
    const right = Math.min(samples.length - ASR_ENERGY_WINDOW_SAMPLES, target + ASR_SILENCE_SEARCH_SAMPLES);
    let cut = Math.min(samples.length, cursor + ASR_SILENCE_MAX_SAMPLES);
    if (right > left) {
      let minimumEnergy = Number.POSITIVE_INFINITY;
      for (let index = left; index <= right; index += ASR_ENERGY_WINDOW_SAMPLES / 2) {
        let energy = 0;
        for (let sample = index; sample < index + ASR_ENERGY_WINDOW_SAMPLES; sample += 1) {
          energy += samples[sample] ** 2;
        }
        if (energy < minimumEnergy) {
          minimumEnergy = energy;
          cut = index;
        }
      }
    }
    boundaries.push(cut);
    cursor = cut;
  }
  boundaries.push(samples.length);
  return boundaries;
}

async function recognizeSamplesInChunks(samples: Float32Array, recognizer: SampleRecognizer = recognizeSampleChunk): Promise<SampleRecognition> {
  const tokens: LocalWord[] = [], boundaries = silenceChunkBoundaries(samples);
  for (let index = 0; index + 1 < boundaries.length; index += 1) {
    const start = boundaries[index], end = boundaries[index + 1];
    if (end - start < N_FFT) {
      // A final sub-20ms tail cannot make an encoder frame, but is not VAD-filtered.
      continue;
    }
    const result = await recognizer(samples.subarray(start, end), start);
    for (const word of result.tokens) {
      const token = { text: word.text, startSeconds: word.startSeconds + start / SAMPLE_RATE, endSeconds: word.endSeconds + start / SAMPLE_RATE };
      if (!token.text || !Number.isFinite(token.startSeconds) || !Number.isFinite(token.endSeconds) || token.endSeconds <= token.startSeconds || word.startSeconds < 0 || word.endSeconds > (end - start) / SAMPLE_RATE + 1e-7) {
        throw new Error('GigaAM emitted an invalid word/timestamp. Local does not silently discard words.');
      }
      tokens.push(token);
    }
  }
  return { durationSeconds: samples.length / SAMPLE_RATE, tokens };
}

function firstWordAtOrAfterMidpoint(words: readonly LocalWord[], seconds: number): number {
  let left = 0;
  let right = words.length;
  while (left < right) {
    const middle = left + Math.floor((right - left) / 2);
    const word = words[middle];
    if ((word.startSeconds + word.endSeconds) / 2 < seconds) left = middle + 1;
    else right = middle;
  }
  return left;
}

async function recognizeActivitySegments(samples: Float32Array, segments: readonly ActivitySegment[], recognizer: SampleRecognizer = recognizeSampleChunk): Promise<SampleRecognition & { ranges: ReturnType<typeof buildActivityRows> }> {
  const tokens: LocalWord[] = [], ranges: ReturnType<typeof buildActivityRows> = [];
  for (const segment of segments) {
    const wordStart = tokens.length;
    const recognized = await recognizeSamplesInChunks(samples.subarray(segment.startSample, segment.endSample),
      (chunk, localStart) => recognizer(chunk, segment.startSample + localStart));
    for (const word of recognized.tokens) {
      tokens.push({ ...word, startSeconds: word.startSeconds + segment.startSample / SAMPLE_RATE,
        endSeconds: word.endSeconds + segment.startSample / SAMPLE_RATE });
    }
    if (tokens.length > wordStart) ranges.push({ ...segment, wordStart, wordEnd: tokens.length });
  }
  return { durationSeconds: samples.length / SAMPLE_RATE, tokens, ranges };
}

export class LocalTimingUnavailableError extends Error {
  constructor(taskId: string) {
    super(`Full-stream C-denoise timing/labels for task ${taskId} are unavailable or belong to an older model. Capture the complete speaker lanes again.`);
    this.name = 'LocalTimingUnavailableError';
  }
}

async function ensureCurrentBundle(allowUntestedBundle = false): Promise<string> {
  const descriptor = await getCachedBundleDescriptor(LOCAL_MODEL_BASE_URL);
  if (descriptor) assertReleasedGraphs(descriptor.files);
  if (!descriptor) throw new Error('The verified C-denoise model bundle is missing. Install Local models in Options.');
  if (!allowUntestedBundle && !descriptor.tested) throw new Error('The C-denoise bundle has not passed an actual WebGPU test. Run Test in Options before using Local.');
  if (descriptor.identity !== currentBundleIdentity) {
    const previousAsr = asrSessionPromise, previousPunctuation = punctuationResourcesPromise;
    asrSessionPromise = null; punctuationResourcesPromise = null; cDenoiseConfigPromise = null;
    placementPolicyPromise = null; placementDiagnostics.clear();
    taskCache.clear();
    if (previousAsr) await previousAsr.then((session) => session.release()).catch(() => undefined);
    if (previousPunctuation) await previousPunctuation.then(async (resources) => { await resources.context.release(); await resources.denoise.release(); }).catch(() => undefined);
    currentBundleIdentity = descriptor.identity;
  }
  return descriptor.identity;
}

function executionDiagnostic(): LocalExecutionDiagnostic {
  if (!adapterDiagnostic || !currentBundleIdentity) throw new Error('The hardware WebGPU execution descriptor is unavailable.');
  const graphs = [ASR_MODEL_PATH, PUNCTUATION_CONTEXT_PATH, PUNCTUATION_DENOISE_PATH].map((path) => {
    const diagnostic = placementDiagnostics.get(path);
    if (!diagnostic || diagnostic.verifiedGpuNodes !== diagnostic.requiredGpuNodes ||
      diagnostic.verifiedRuns < (path === PUNCTUATION_DENOISE_PATH ? 4 : 1)) throw new Error(`Strict neural WebGPU placement has not been proved for ${path}.`);
    return diagnostic;
  });
  return Object.freeze({ provider: 'webgpu', shaderF16: true, denoiseSteps: 4, neuralCpuFallback: false,
    asrCheckpointSha256: ASR_SOURCE_SHA, cDenoiseCheckpointSha256: C_DENOISE_SOURCE_SHA,
    bundleIdentity: currentBundleIdentity, adapter: adapterDiagnostic,
    placementAudit: Object.freeze({ method: 'sha256-bound-source-nodes-and-webgpu-dispatch-profile',
      graphOptimizations: 'disabled', hostMetadataAllowed: true, cpuAttemptsPrevented: false, graphs: Object.freeze(graphs) }) });
}

function modelsSummary(): Record<string, unknown> {
  return Object.freeze({
    release: INFERENCE_RELEASE.id,
    asr: Object.freeze({ name: 'gigaam-v3-ctc-domain', graph: ASR_MODEL_PATH, sourceSha256: ASR_SOURCE_SHA, runtime: 'onnxruntime-web', executionProviders: Object.freeze(['webgpu']), inputDtype: 'float16' }),
    l2: Object.freeze({ name: 'C-denoise', graphs: Object.freeze([PUNCTUATION_CONTEXT_PATH, PUNCTUATION_DENOISE_PATH]), sourceSha256: C_DENOISE_SOURCE_SHA, runtime: 'onnxruntime-web', executionProviders: Object.freeze(['webgpu']), steps: 4, labels: Object.freeze([...PUNCTUATION_LABELS]) }),
    execution: executionDiagnostic()
  });
}

/** Separate original-waveform encoder pass: 20s owned cores and 2s context, never recognition chunk reuse. */
async function extractAcousticFrames(audio: PreparedLocalAudio): Promise<AcousticFrames> {
  const config = await getCDenoiseConfig(), samples = audio.featureSamples;
  const stride = config.frame_stride_samples, center = config.frame_center_offset_samples;
  const coreSamples = 20 * SAMPLE_RATE, contextSamples = 2 * SAMPLE_RATE;
  const chunks: Uint16Array[] = [], positions: number[] = [];
  for (let coreStart = 0; coreStart < samples.length; coreStart += coreSamples) {
    const coreEnd = Math.min(samples.length, coreStart + coreSamples);
    const inputStart = Math.max(0, coreStart - contextSamples), inputEnd = Math.min(samples.length, coreEnd + contextSamples);
    if (inputEnd - inputStart < N_FFT) continue;
    const output = await runAsr(samples.subarray(inputStart, inputEnd), ['encoded_lengths', 'encoder_features']);
    try {
      if (!output.encoded_lengths) throw new Error('Acoustic graph returned no encoded_lengths.');
      const count = tensorEncodedLength(output.encoded_lengths), tensor = output.encoder_features;
      if (!tensor || tensor.type !== 'float16' || tensor.dims.length !== 3 || tensor.dims[0] !== 1 || tensor.dims[1] !== 768 || tensor.dims[2] < count || tensor.data.length !== 768 * tensor.dims[2]) {
        throw new Error('GigaAM encoder_features must be actual float16 [1,768,T] features.');
      }
      const origin = inputStart + center;
      const left = Math.max(0, Math.min(count, Math.ceil((coreStart - origin) / stride)));
      const right = Math.max(left, Math.min(count, Math.ceil((coreEnd - origin) / stride)));
      const owned = new Uint16Array((right - left) * 768);
      for (let frame = left; frame < right; frame += 1) {
        const time = (origin + frame * stride) / SAMPLE_RATE;
        if (positions.length && time <= positions[positions.length - 1]) throw new Error('GigaAM acoustic frame ownership overlaps at a chunk seam.');
        positions.push(time);
        for (let channel = 0; channel < 768; channel += 1) {
          const index = channel * tensor.dims[2] + frame;
          const packed = tensor.data instanceof Uint16Array ? tensor.data[index] : float32ToFloat16(tensorNumericValue(tensor.data, index, tensor.type));
          if (!Number.isFinite(float16ToFloat32(packed))) throw new Error('GigaAM produced nonfinite acoustic features.');
          owned[(frame - left) * 768 + channel] = packed;
        }
      }
      chunks.push(owned);
    } finally { for (const tensor of Object.values(output)) tensor.dispose(); }
  }
  if (!positions.length) throw new Error('GigaAM produced no owned acoustic frames. Local cannot substitute text-only punctuation.');
  const times = Float64Array.from(positions), features = new Uint16Array(times.length * 768);
  let offset = 0;
  for (const chunk of chunks) { features.set(chunk, offset); offset += chunk.length; }
  return { times, features, prosody: extractProsody(audio.source, audio.sourceRate, times, stride / SAMPLE_RATE),
    startSeconds: 0, endSeconds: samples.length / SAMPLE_RATE };
}

async function recognizeAndLabelAudio(audio: PreparedLocalAudio): Promise<SampleRecognition & { labelIds: Uint8Array; ranges: ReturnType<typeof buildActivityRows> }> {
  const recognized = await recognizeActivitySegments(audio.raw, segmentSamplesByActivity(audio.activity));
  const ranges = recognized.ranges;
  if (!recognized.tokens.length) return { ...recognized, labelIds: new Uint8Array(), ranges };
  const frames = await extractAcousticFrames(audio), resources = await getPunctuationResources();
  const localAudio = sampleLocalAudio(boundaryCenters(recognized.tokens), [frames], resources.config.frame_stride_samples / SAMPLE_RATE, resources.config.local_offsets_seconds);
  const labelIds = await predictCDenoise(recognized.tokens, localAudio, resources, ranges);
  return { ...recognized, labelIds, ranges };
}

export async function transcribeLocalAudio(blob: Blob, options: { maxDurationSeconds?: number | null; allowUntestedBundle?: boolean } = {}): Promise<LocalTranscriptResult> {
  try {
    await ensureCurrentBundle(options.allowUntestedBundle === true);
    const audio = await prepareDraftAudio(blob, resolveMaxDurationSeconds(options.maxDurationSeconds));
    const result = await recognizeAndLabelAudio(audio);
    if (!result.tokens.length) throw new Error('No speech words were recognized; C-denoise was not exercised. Test with a recording containing speech.');
    return { text: renderBoundaryLabels(result.tokens.map((word) => word.text), Array.from(result.labelIds, (id) => PUNCTUATION_LABELS[id])).text,
      durationSeconds: result.durationSeconds, tokens: result.tokens, execution: executionDiagnostic() };
  } catch (error) { throw actionableError('transcription', error); }
}

function emitTimingStatus(callbacks: L0TimingRequestCallbacks | undefined, status: L0TimingQueueStatus): void {
  try { callbacks?.onQueueStatus?.(status); } catch { /* Observational queue UI must not affect inference. */ }
}

export async function generateLocalL0Timing(_settings: ExtensionSettings, job: TranscriptJob, audioTracks: CapturedAudioTrack[],
  callbacks?: L0TimingRequestCallbacks, taskIdOverride?: string): Promise<L0TimingResponse> {
  const startedAt = performance.now(), identity = await ensureCurrentBundle();
  const prepared = prepareL0TimingTracks(job, audioTracks), taskId = taskIdOverride ?? buildCanonicalTaskIdentity(job), requestId = `browser-local:${taskId}`;
  emitTimingStatus(callbacks, { requestId, status: 'preparing' });
  emitTimingStatus(callbacks, { requestId, status: 'running', position: 0, queuedCount: 0 });
  const tracks: L0TimingResponse['tracks'] = [], lanes = new Map<string, CachedLane>();
  for (const track of prepared) {
    try {
      const audio = await prepareDraftAudio(track.audio.blob), result = await recognizeAndLabelAudio(audio);
      const ranges = result.ranges;
      const segments = await Promise.all(ranges.map(async (range) => {
        const startSample = range.startSample;
        const endSample = range.endSample;
        return { id: await draftRowId(taskId, track.lane, audio.pcmSha256, startSample, endSample),
          startSeconds: startSample / SAMPLE_RATE, endSeconds: endSample / SAMPLE_RATE, startSample, endSample, sampleRate: SAMPLE_RATE };
      }));
      tracks.push({ punctuationLabels: Array.from(result.labelIds), lane: track.lane, pcmSha256: audio.pcmSha256, sampleRate: SAMPLE_RATE,
        tokens: result.tokens.map((token, index) => ({ id: `${taskId}:${track.lane}:${index}`, ...token })), segments });
      lanes.set(track.lane, { pcmSha256: audio.pcmSha256, tokens: result.tokens, labelIds: result.labelIds, ranges });
    } catch (error) { throw actionableError(`timing lane \"${track.lane}\"`, error); }
  }
  const timing: L0TimingResponse = { taskId, tracks,
    summary: { taskId, trackCount: tracks.length, tokenCount: tracks.reduce((count, track) => count + track.tokens.length, 0), latencyMs: Math.round(performance.now() - startedAt), provider: 'browser-local' },
    models: modelsSummary() };
  taskCache.delete(taskId); taskCache.set(taskId, { identity, timing, lanes });
  if (taskCache.size > 2) taskCache.delete(taskCache.keys().next().value!);
  emitTimingStatus(callbacks, { requestId, status: 'completed', position: 0, queuedCount: 0 });
  return timing;
}

function timingMatchesCached(timing: L0TimingResponse, cached: CachedTask): boolean {
  return timing.tracks.length === cached.lanes.size && timing.tracks.every((track) => {
    const lane = cached.lanes.get(track.lane);
    return lane && lane.labelIds.length === lane.tokens.length && track.pcmSha256 === lane.pcmSha256 && track.tokens.length === lane.tokens.length
      && track.tokens.every((token, index) => token.text === lane.tokens[index].text && token.startSeconds === lane.tokens[index].startSeconds && token.endSeconds === lane.tokens[index].endSeconds);
  });
}

export async function isLocalTimingCurrent(timing: L0TimingResponse): Promise<boolean> {
  const identity = await ensureCurrentBundle(), cached = taskCache.get(timing.taskId);
  return Boolean(cached && cached.identity === identity && timingMatchesCached(timing, cached));
}

async function requireCachedTask(taskId: string, timing?: L0TimingResponse): Promise<CachedTask> {
  const identity = await ensureCurrentBundle();
  if (!taskCache.has(taskId) && timing) {
    taskCache.set(taskId, { identity, timing, lanes: hydratePunctuatedTiming(timing) });
    if (taskCache.size > 2) taskCache.delete(taskCache.keys().next().value!);
  }
  const cached = taskCache.get(taskId);
  if (!cached || cached.identity !== identity || timing && !timingMatchesCached(timing, cached)) throw new LocalTimingUnavailableError(taskId);
  return cached;
}

export function hydratePunctuatedTiming(timing: L0TimingResponse): Map<string, CachedLane> {
  if (timing.models.release !== INFERENCE_RELEASE.id) throw new LocalTimingUnavailableError(timing.taskId);
  const lanes = new Map<string, CachedLane>();
  for (const track of timing.tracks) {
    const ids = track.punctuationLabels;
    if (!ids || ids.length !== track.tokens.length || ids.some(id => !Number.isInteger(id) || id < 0 || id >= PUNCTUATION_LABELS.length) || lanes.has(track.lane)) throw new LocalTimingUnavailableError(timing.taskId);
    let owned = 0;
    const ranges = track.segments.map(segment => {
      const start = firstWordAtOrAfterMidpoint(track.tokens, segment.startSeconds);
      const end = firstWordAtOrAfterMidpoint(track.tokens, segment.endSeconds);
      if (start !== owned || end <= start) throw new LocalTimingUnavailableError(timing.taskId);
      owned = end;
      return { wordStart: start, wordEnd: end, startSample: segment.startSample, endSample: segment.endSample };
    });
    if (owned !== track.tokens.length) throw new LocalTimingUnavailableError(timing.taskId);
    lanes.set(track.lane, { tokens: track.tokens, pcmSha256: track.pcmSha256, labelIds: Uint8Array.from(ids), ranges });
  }
  return lanes;
}

function renderCachedRange(lane: CachedLane, start: number, end: number): string {
  if (start < 0 || end <= start || end > lane.tokens.length) throw new Error('Cached row has no immutable ASR words.');
  const previous = start ? PUNCTUATION_LABELS[lane.labelIds[start - 1]] : null;
  return renderBoundaryLabels(lane.tokens.slice(start, end).map((word) => word.text),
    Array.from(lane.labelIds.subarray(start, end), (id) => PUNCTUATION_LABELS[id]), !start || previous === 'PERIOD' || previous === 'QUESTION').text;
}

function renderCachedInterval(lane: CachedLane, startSeconds: number | null, endSeconds: number | null): string {
  if (startSeconds === null || endSeconds === null || !Number.isFinite(startSeconds) || !Number.isFinite(endSeconds) || startSeconds < 0 || endSeconds <= startSeconds) throw new Error('Local segment drafting requires finite positive row timestamps.');
  const start = firstWordAtOrAfterMidpoint(lane.tokens, startSeconds), end = firstWordAtOrAfterMidpoint(lane.tokens, endSeconds);
  return renderCachedRange(lane, start, end);
}

export async function generateLocalL0SegmentDraft(_settings: ExtensionSettings, taskId: string, row: TranscriptJob['rows'][number], _tracks: PreparedL0Track[]): Promise<string> {
  const cached = await requireCachedTask(taskId);
  const lane = Array.from(cached.lanes.entries()).find(([key]) => key.trim().toLocaleLowerCase() === row.speakerKey.trim().toLocaleLowerCase())?.[1];
  if (!lane) throw new LocalTimingUnavailableError(taskId);
  // Supplied segment audio is intentionally not cropped/re-inferred: labels own the complete lane context.
  return renderCachedInterval(lane, row.startSeconds, row.endSeconds);
}

export async function generateLocalL0DraftFromTiming(timing: L0TimingResponse, preserveRows?: TranscriptJob['rows']): Promise<L0DraftResponse> {
  const startedAt = performance.now(), cached = await requireCachedTask(timing.taskId, timing), rows: L0DraftResponse['rows'] = [];
  let wordCount = 0;
  if (preserveRows) {
    for (const row of preserveRows) {
      const lane = cached.lanes.get(row.speakerKey);
      if (!lane) throw new LocalTimingUnavailableError(timing.taskId);
      rows.push({ id: row.rowId, lane: row.speakerKey, startSeconds: row.startSeconds!, endSeconds: row.endSeconds!, text: renderCachedInterval(lane, row.startSeconds, row.endSeconds) });
    }
  } else {
    for (const track of cached.timing.tracks) {
      const lane = cached.lanes.get(track.lane)!, ranges = lane.ranges;
      if (ranges.length !== track.segments.length) throw new LocalTimingUnavailableError(timing.taskId);
      for (let index = 0; index < ranges.length; index += 1) {
        const range = ranges[index], segment = track.segments[index];
        wordCount += range.wordEnd - range.wordStart;
        rows.push({ id: segment.id, lane: track.lane, startSeconds: segment.startSeconds, endSeconds: segment.endSeconds, text: renderCachedRange(lane, range.wordStart, range.wordEnd) });
      }
    }
    const laneOrder = new Map(cached.timing.tracks.map((track, index) => [track.lane, index]));
    rows.sort((left, right) => left.startSeconds - right.startSeconds || (laneOrder.get(left.lane) ?? 0) - (laneOrder.get(right.lane) ?? 0) || left.endSeconds - right.endSeconds);
  }
  if (!rows.length || rows.some((row) => !row.text || row.endSeconds <= row.startSeconds)) throw new Error('Local draft did not produce non-empty, positive-duration rows.');
  // Rendering completed labels is host work. Preserve the transcription's neural
  // execution evidence even when this renderer is a different browser worker.
  return { rows, summary: { taskId: timing.taskId, trackCount: timing.tracks.length, rowCount: rows.length, wordCount, latencyMs: Math.round(performance.now() - startedAt), provider: 'browser-local' }, models: cached.timing.models };
}

/** DSP contract hooks and audited hardware-session access for browser golden trials. */
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
  recognizeSamplesInChunks,
  recognizeActivitySegments,
  silenceChunkBoundaries,
  renderBoundaryLabels,
  resolveMaxDurationSeconds,
  prepareDraftAudio,
  draftRowId,
  renderCachedRange,
  renderCachedInterval,
  timingMatchesCached,
  extractAcousticFrames,
  ensureCurrentBundle,
  getAsrSession,
  getPunctuationResources,
  getCDenoiseConfig,
  runAsr,
  executionDiagnostic,
  float32ToFloat16,
  float16ToFloat32
};
