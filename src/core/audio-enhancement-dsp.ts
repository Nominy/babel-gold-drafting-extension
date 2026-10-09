import { resamplePoly } from './c-denoise-acoustic';

export const ZIP_SAMPLE_RATE = 16_000;
export const ZIP_FFT_SIZE = 400;
export const ZIP_HOP_SIZE = 100;
export const ZIP_BINS = 201;
export const ZIP_CHUNK_SAMPLES = 4 * ZIP_SAMPLE_RATE;
export const ZIP_STRIDE_SAMPLES = 3 * ZIP_SAMPLE_RATE;
const CONVOLUTION_SIZE = 1024;
const f32 = Math.fround;

// The existing GigaAM frontend uses Bluestein for non-power-of-two FFTs.
// This fixed-size plan uses the same radix-2 convolution for the reference's 400-point transform.
const reversed = new Uint16Array(CONVOLUTION_SIZE);
const rootsReal = new Float64Array(CONVOLUTION_SIZE / 2);
const rootsImaginary = new Float64Array(CONVOLUTION_SIZE / 2);
const chirpReal = new Float64Array(ZIP_FFT_SIZE);
const chirpImaginary = new Float64Array(ZIP_FFT_SIZE);
const kernelReal = new Float64Array(CONVOLUTION_SIZE);
const kernelImaginary = new Float64Array(CONVOLUTION_SIZE);
const hann = new Float32Array(ZIP_FFT_SIZE);
for (let index = 0; index < CONVOLUTION_SIZE; index++) {
  let source = index, target = 0;
  for (let bit = 0; bit < 10; bit++) { target = (target << 1) | (source & 1); source >>>= 1; }
  reversed[index] = target;
}
for (let index = 0; index < rootsReal.length; index++) {
  rootsReal[index] = Math.cos(2 * Math.PI * index / CONVOLUTION_SIZE);
  rootsImaginary[index] = Math.sin(2 * Math.PI * index / CONVOLUTION_SIZE);
}
for (let index = 0; index < ZIP_FFT_SIZE; index++) {
  const angle = Math.PI * index * index / ZIP_FFT_SIZE;
  chirpReal[index] = kernelReal[index] = Math.cos(angle);
  chirpImaginary[index] = kernelImaginary[index] = Math.sin(angle);
  if (index) {
    kernelReal[CONVOLUTION_SIZE - index] = chirpReal[index];
    kernelImaginary[CONVOLUTION_SIZE - index] = chirpImaginary[index];
  }
  // torch.hann_window defaults to periodic=True, not the symmetric window.
  hann[index] = f32(0.5 - 0.5 * Math.cos(2 * Math.PI * index / ZIP_FFT_SIZE));
}

function radix2(real: Float64Array, imaginary: Float64Array, inverse: boolean): void {
  for (let index = 0; index < CONVOLUTION_SIZE; index++) {
    const target = reversed[index];
    if (target <= index) continue;
    const r = real[index], i = imaginary[index];
    real[index] = real[target]; real[target] = r;
    imaginary[index] = imaginary[target]; imaginary[target] = i;
  }
  for (let width = 2; width <= CONVOLUTION_SIZE; width *= 2) {
    const half = width / 2, stride = CONVOLUTION_SIZE / width;
    for (let start = 0; start < CONVOLUTION_SIZE; start += width) {
      for (let index = 0; index < half; index++) {
        const left = start + index, right = left + half, root = index * stride;
        const ri = inverse ? rootsImaginary[root] : -rootsImaginary[root];
        const r = real[right] * rootsReal[root] - imaginary[right] * ri;
        const i = real[right] * ri + imaginary[right] * rootsReal[root];
        real[right] = real[left] - r; imaginary[right] = imaginary[left] - i;
        real[left] += r; imaginary[left] += i;
      }
    }
  }
  if (inverse) for (let index = 0; index < CONVOLUTION_SIZE; index++) {
    real[index] /= CONVOLUTION_SIZE; imaginary[index] /= CONVOLUTION_SIZE;
  }
}
radix2(kernelReal, kernelImaginary, false);

function transform(real: Float64Array, imaginary: Float64Array, workReal: Float64Array, workImaginary: Float64Array, inverse: boolean): void {
  workReal.fill(0); workImaginary.fill(0);
  for (let index = 0; index < ZIP_FFT_SIZE; index++) {
    const r = real[index], i = inverse ? -imaginary[index] : imaginary[index];
    workReal[index] = r * chirpReal[index] + i * chirpImaginary[index];
    workImaginary[index] = i * chirpReal[index] - r * chirpImaginary[index];
  }
  radix2(workReal, workImaginary, false);
  for (let index = 0; index < CONVOLUTION_SIZE; index++) {
    const r = workReal[index], i = workImaginary[index];
    workReal[index] = r * kernelReal[index] - i * kernelImaginary[index];
    workImaginary[index] = r * kernelImaginary[index] + i * kernelReal[index];
  }
  radix2(workReal, workImaginary, true);
  for (let index = 0; index < ZIP_FFT_SIZE; index++) {
    const r = workReal[index], i = workImaginary[index];
    real[index] = (r * chirpReal[index] + i * chirpImaginary[index]) / (inverse ? ZIP_FFT_SIZE : 1);
    imaginary[index] = (i * chirpReal[index] - r * chirpImaginary[index]) / (inverse ? -ZIP_FFT_SIZE : 1);
  }
}

export interface ZipSpectrum {
  magnitude: Float32Array;
  phase: Float32Array;
  frames: number;
}

const ZIP_FRAMES = ZIP_CHUNK_SAMPLES / ZIP_HOP_SIZE + 1;
const REUSE_FIRST_FRAME = Math.ceil(ZIP_FFT_SIZE / 2 / ZIP_HOP_SIZE);
const REUSE_LAST_FRAME = (ZIP_CHUNK_SAMPLES - ZIP_STRIDE_SAMPLES) / ZIP_HOP_SIZE - REUSE_FIRST_FRAME;
const REUSE_FRAME_COUNT = REUSE_LAST_FRAME - REUSE_FIRST_FRAME + 1;
interface StftOverlap {
  magnitude: Float32Array;
  phase: Float32Array;
  valid: boolean;
}

function computeZipStft(samples: Float32Array, overlap?: StftOverlap): ZipSpectrum {
  if (samples.length !== ZIP_CHUNK_SAMPLES) throw new Error('ZipEnhancer requires exactly four seconds per model chunk.');
  const frames = Math.floor(samples.length / ZIP_HOP_SIZE) + 1;
  const magnitude = new Float32Array(ZIP_BINS * frames), phase = new Float32Array(magnitude.length);
  const real = new Float64Array(ZIP_FFT_SIZE), imaginary = new Float64Array(ZIP_FFT_SIZE);
  const workReal = new Float64Array(CONVOLUTION_SIZE), workImaginary = new Float64Array(CONVOLUTION_SIZE);
  for (let frame = 0; frame < frames; frame++) {
    if (overlap?.valid && frame >= REUSE_FIRST_FRAME && frame <= REUSE_LAST_FRAME) {
      for (let bin = 0; bin < ZIP_BINS; bin++) {
        const target = bin * frames + frame, cached = bin * REUSE_FRAME_COUNT + frame - REUSE_FIRST_FRAME;
        magnitude[target] = overlap.magnitude[cached];
        phase[target] = overlap.phase[cached];
      }
      continue;
    }
    imaginary.fill(0);
    for (let index = 0; index < ZIP_FFT_SIZE; index++) {
      let source = frame * ZIP_HOP_SIZE + index - ZIP_FFT_SIZE / 2;
      if (source < 0) source = -source;
      if (source >= samples.length) source = 2 * samples.length - 2 - source;
      real[index] = f32(samples[source] * hann[index]);
    }
    transform(real, imaginary, workReal, workImaginary, false);
    for (let bin = 0; bin < ZIP_BINS; bin++) {
      const r = f32(real[bin]), i = f32(imaginary[bin]), target = bin * frames + frame;
      magnitude[target] = f32(f32(Math.sqrt(f32(f32(f32(r * r) + f32(i * i)) + 1e-9))) ** 0.3);
      phase[target] = Math.atan2(i, f32(r + 1e-5));
    }
  }
  return { magnitude, phase, frames };
}

/** Independent four-second window, including reflected boundaries. */
export function zipStft(samples: Float32Array): ZipSpectrum {
  return computeZipStft(samples);
}

/** One contiguous, globally normalized lane; output buffers may be transferred or mutated. */
export function createZipStftSequence(): (samples: Float32Array) => ZipSpectrum {
  const overlap: StftOverlap = {
    magnitude: new Float32Array(ZIP_BINS * REUSE_FRAME_COUNT),
    phase: new Float32Array(ZIP_BINS * REUSE_FRAME_COUNT),
    valid: false
  };
  return samples => {
    const spectral = computeZipStft(samples, overlap);
    const tailStart = ZIP_STRIDE_SAMPLES / ZIP_HOP_SIZE + REUSE_FIRST_FRAME;
    for (let bin = 0; bin < ZIP_BINS; bin++) {
      const start = bin * spectral.frames + tailStart;
      overlap.magnitude.set(spectral.magnitude.subarray(start, start + REUSE_FRAME_COUNT), bin * REUSE_FRAME_COUNT);
      overlap.phase.set(spectral.phase.subarray(start, start + REUSE_FRAME_COUNT), bin * REUSE_FRAME_COUNT);
    }
    overlap.valid = true;
    return spectral;
  };
}

// Fixed 641-frame overlap-add envelope, constructed in the reference addition order.
const istftEnvelope = new Float64Array((ZIP_FRAMES - 1) * ZIP_HOP_SIZE + ZIP_FFT_SIZE);
for (let frame = 0; frame < ZIP_FRAMES; frame++) {
  for (let index = 0; index < ZIP_FFT_SIZE; index++) {
    istftEnvelope[frame * ZIP_HOP_SIZE + index] += f32(hann[index] * hann[index]);
  }
}

export function zipIstft(magnitude: Float32Array, phase: Float32Array, frames: number): Float32Array {
  if (frames !== ZIP_CHUNK_SAMPLES / ZIP_HOP_SIZE + 1 || magnitude.length !== ZIP_BINS * frames || phase.length !== magnitude.length) {
    throw new Error('ZipEnhancer returned an invalid magnitude/phase shape.');
  }
  const length = (frames - 1) * ZIP_HOP_SIZE + ZIP_FFT_SIZE;
  const sum = new Float64Array(length);
  const real = new Float64Array(ZIP_FFT_SIZE), imaginary = new Float64Array(ZIP_FFT_SIZE);
  const workReal = new Float64Array(CONVOLUTION_SIZE), workImaginary = new Float64Array(CONVOLUTION_SIZE);
  for (let frame = 0; frame < frames; frame++) {
    for (let bin = 0; bin < ZIP_BINS; bin++) {
      const source = bin * frames + frame, amplitude = f32(magnitude[source] ** (1 / 0.3));
      if (!Number.isFinite(amplitude) || !Number.isFinite(phase[source]) || amplitude < 0) throw new Error('ZipEnhancer produced invalid spectral values.');
      real[bin] = f32(amplitude * f32(Math.cos(phase[source])));
      imaginary[bin] = bin === 0 || bin === ZIP_BINS - 1 ? 0 : f32(amplitude * f32(Math.sin(phase[source])));
      if (bin > 0 && bin < ZIP_BINS - 1) { real[ZIP_FFT_SIZE - bin] = real[bin]; imaginary[ZIP_FFT_SIZE - bin] = -imaginary[bin]; }
    }
    transform(real, imaginary, workReal, workImaginary, true);
    for (let index = 0; index < ZIP_FFT_SIZE; index++) {
      const target = frame * ZIP_HOP_SIZE + index;
      sum[target] += f32(f32(real[index]) * hann[index]);
    }
  }
  const output = new Float32Array(ZIP_CHUNK_SAMPLES);
  for (let index = 0; index < output.length; index++) {
    const source = index + ZIP_FFT_SIZE / 2;
    if (istftEnvelope[source] <= 1e-11) throw new Error('ZipEnhancer inverse STFT violates the overlap-add envelope.');
    output[index] = sum[source] / istftEnvelope[source];
  }
  return output;
}

export interface EnhancementWavHeader {
  sampleRate: number;
  frameCount: number;
  channels: number;
  format: number;
  bits: number;
  bytesPerSample: number;
  blockAlign: number;
  dataOffset: number;
}

export function readEnhancementWavHeader(bytes: ArrayBuffer): EnhancementWavHeader {
  const view = new DataView(bytes);
  if (bytes.byteLength < 44 || view.getUint32(0, false) !== 0x52494646 || view.getUint32(8, false) !== 0x57415645) throw new Error('ZipEnhancer requires a RIFF PCM16 or float32 WAV source.');
  let rate = 0, channels = 0, format = 0, bits = 0, dataOffset = -1, dataLength = 0;
  for (let offset = 12; offset + 8 <= bytes.byteLength;) {
    const size = view.getUint32(offset + 4, true), next = offset + 8 + size + (size & 1);
    if (offset + 8 + size > bytes.byteLength) throw new Error('ZipEnhancer source WAV is truncated.');
    const id = view.getUint32(offset, false);
    if (id === 0x666d7420) {
      if (size < 16) throw new Error('ZipEnhancer source WAV has an invalid format chunk.');
      format = view.getUint16(offset + 8, true); channels = view.getUint16(offset + 10, true);
      rate = view.getUint32(offset + 12, true); bits = view.getUint16(offset + 22, true);
    } else if (id === 0x64617461) { dataOffset = offset + 8; dataLength = size; }
    offset = next;
  }
  const bytesPerSample = format === 1 && bits === 16 ? 2 : format === 3 && bits === 32 ? 4 : 0;
  const blockAlign = channels * bytesPerSample;
  if (!bytesPerSample || !channels || !rate || dataOffset < 0 || dataLength < blockAlign || dataLength % blockAlign) {
    throw new Error('ZipEnhancer requires nonempty PCM16 or float32 WAV audio.');
  }
  return { sampleRate: rate, frameCount: dataLength / blockAlign, channels, format, bits, bytesPerSample, blockAlign, dataOffset };
}

export function decodeEnhancementWav(bytes: ArrayBuffer): { samples: Float32Array; sampleRate: number } {
  const { sampleRate, frameCount, channels, format, bytesPerSample, blockAlign, dataOffset } = readEnhancementWavHeader(bytes);
  const view = new DataView(bytes), samples = new Float32Array(frameCount);
  for (let index = 0; index < samples.length; index++) {
    let mixed = 0;
    for (let channel = 0; channel < channels; channel++) {
      const offset = dataOffset + index * blockAlign + channel * bytesPerSample;
      const value = format === 1 ? view.getInt16(offset, true) / 32768 : view.getFloat32(offset, true);
      if (!Number.isFinite(value)) throw new Error('ZipEnhancer source PCM is nonfinite.');
      mixed += value;
    }
    samples[index] = mixed / channels;
  }
  return { samples, sampleRate };
}

export function encodeEnhancementWav(samples: Float32Array, sampleRate: number, frameCount: number): Uint8Array<ArrayBuffer> {
  if (!Number.isSafeInteger(frameCount) || frameCount < 1 || samples.length < frameCount || !Number.isInteger(sampleRate) || sampleRate < 1) throw new Error('ZipEnhancer could not preserve the source clock.');
  const bytes = new Uint8Array(44 + frameCount * 2), view = new DataView(bytes.buffer);
  view.setUint32(0, 0x52494646, false); view.setUint32(4, bytes.length - 8, true);
  view.setUint32(8, 0x57415645, false); view.setUint32(12, 0x666d7420, false); view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  view.setUint32(36, 0x64617461, false); view.setUint32(40, frameCount * 2, true);
  for (let index = 0; index < frameCount; index++) {
    const sample = samples[index];
    if (!Number.isFinite(sample)) throw new Error('ZipEnhancer produced nonfinite output PCM.');
    // Quantization only: no peak/RMS normalization, gate, or quiet-speech suppression.
    view.setInt16(44 + index * 2, Math.max(-32768, Math.min(32767, Math.round(sample * 32768))), true);
  }
  return bytes;
}

export async function enhanceZipSamples(
  source: Float32Array,
  sampleRate: number,
  infer: (magnitude: Float32Array, phase: Float32Array, frames: number) => Promise<{ magnitude: Float32Array; phase: Float32Array }>,
  onChunkCompleted?: (completedChunks: number, totalChunks: number) => void | Promise<void>,
  onChunksReady?: (totalChunks: number) => void | Promise<void>
): Promise<Float32Array> {
  const samples = resamplePoly(source, sampleRate, ZIP_SAMPLE_RATE);
  let power = 0;
  for (const value of samples) { if (!Number.isFinite(value)) throw new Error('ZipEnhancer source PCM is nonfinite.'); power += value * value; }
  if (!(power > 0)) throw new Error('ZipEnhancer cannot enhance an all-zero source lane: the reference RMS normalization is undefined.');
  const scale = Math.sqrt(samples.length / power);
  const paddedLength = ZIP_CHUNK_SAMPLES + Math.max(0, Math.ceil((samples.length + ZIP_SAMPLE_RATE - ZIP_CHUNK_SAMPLES) / ZIP_STRIDE_SAMPLES)) * ZIP_STRIDE_SAMPLES;
  const totalChunks = 1 + (paddedLength - ZIP_CHUNK_SAMPLES) / ZIP_STRIDE_SAMPLES;
  await onChunksReady?.(totalChunks);
  const output = new Float32Array(samples.length), chunk = new Float32Array(ZIP_CHUNK_SAMPLES);
  const edge = (ZIP_CHUNK_SAMPLES - ZIP_STRIDE_SAMPLES) / 2;
  const stft = createZipStftSequence();
  for (let start = 0; start + ZIP_CHUNK_SAMPLES <= paddedLength; start += ZIP_STRIDE_SAMPLES) {
    chunk.fill(0);
    for (let index = 0; index < Math.min(chunk.length, samples.length - start); index++) chunk[index] = samples[start + index] * scale;
    const spectral = stft(chunk), enhanced = await infer(spectral.magnitude, spectral.phase, spectral.frames);
    const waveform = zipIstft(enhanced.magnitude, enhanced.phase, spectral.frames);
    const left = start === 0 ? 0 : edge, right = Math.min(ZIP_CHUNK_SAMPLES - edge, samples.length - start);
    for (let index = left; index < right; index++) output[start + index] = waveform[index] / scale;
    await onChunkCompleted?.(start / ZIP_STRIDE_SAMPLES + 1, totalChunks);
  }
  return resamplePoly(output, ZIP_SAMPLE_RATE, sampleRate);
}
