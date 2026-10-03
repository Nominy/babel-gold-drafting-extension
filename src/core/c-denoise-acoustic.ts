export interface AcousticWord { text: string; startSeconds: number; endSeconds: number }
export interface AcousticFrames {
  times: Float64Array;
  /** Row-major [F,768], IEEE float16 bits. */
  features: Uint16Array;
  prosody: Uint16Array;
  startSeconds: number;
  endSeconds: number;
}
export const LOCAL_FRAMES = 24;
export const AUDIO_DIM = 772;
const bits = new Uint32Array(1);
const floats = new Float32Array(bits.buffer);

export function float32ToFloat16(value: number): number {
  floats[0] = value;
  const source = bits[0], sign = (source >>> 16) & 0x8000;
  let exponent = ((source >>> 23) & 255) - 112;
  let mantissa = source & 0x7fffff;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa = (mantissa | 0x800000) >>> (1 - exponent);
    return sign | ((mantissa + 0xfff + ((mantissa >>> 13) & 1)) >>> 13);
  }
  if (exponent >= 31) return sign | (mantissa ? 0x7e00 : 0x7c00);
  mantissa += 0xfff + ((mantissa >>> 13) & 1);
  if (mantissa & 0x800000) { mantissa = 0; exponent += 1; }
  return sign | (exponent >= 31 ? 0x7c00 : (exponent << 10) | (mantissa >>> 13));
}
export function float16ToFloat32(value: number): number {
  const sign = value & 0x8000 ? -1 : 1, exponent = (value >>> 10) & 31, mantissa = value & 1023;
  if (exponent === 0) return sign * 2 ** -14 * (mantissa / 1024);
  if (exponent === 31) return mantissa ? NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + mantissa / 1024);
}
export function roundEven(value: number): number {
  const floor = Math.floor(value), fraction = value - floor;
  return fraction < 0.5 ? floor : fraction > 0.5 ? floor + 1 : floor + (floor % 2 !== 0 ? 1 : 0);
}
function gcd(a: number, b: number): number { while (b) { const next = a % b; a = b; b = next; } return a; }
function besselI0(value: number): number {
  const square = value * value / 4;
  let sum = 1, term = 1;
  for (let index = 1; index < 100; index += 1) {
    term *= square / (index * index); sum += term;
    if (term < sum * 1e-16) break;
  }
  return sum;
}
const filters = new Map<string, Float32Array>();
/** scipy.signal.resample_poly default Kaiser(beta=5), constant padding, zero-phase crop. */
export function resamplePoly(source: Float32Array, sourceRate: number, targetRate: number): Float32Array {
  if (!Number.isInteger(sourceRate) || sourceRate < 1 || !Number.isInteger(targetRate) || targetRate < 1) {
    throw new Error('PCM resampling requires positive integer sample rates.');
  }
  if (sourceRate === targetRate) return source;
  const divisor = gcd(sourceRate, targetRate), up = targetRate / divisor, down = sourceRate / divisor;
  const key = `${up}/${down}`, maximum = Math.max(up, down), half = 10 * maximum;
  let taps = filters.get(key);
  if (!taps) {
    const size = half * 2 + 1, coefficients = new Float64Array(size), denominator = besselI0(5);
    let sum = 0;
    for (let index = 0; index < size; index += 1) {
      const offset = index - half, argument = offset / maximum;
      const sinc = offset === 0 ? 1 : Math.sin(Math.PI * argument) / (Math.PI * argument);
      const window = besselI0(5 * Math.sqrt(Math.max(0, 1 - (offset / half) ** 2))) / denominator;
      coefficients[index] = sinc * window / maximum; sum += coefficients[index];
    }
    taps = new Float32Array(size);
    for (let index = 0; index < size; index += 1) taps[index] = Math.fround(Math.fround(coefficients[index] / sum) * up);
    filters.set(key, taps);
  }
  const prePad = down - half % down, remove = (half + prePad) / down;
  const output = new Float32Array(Math.ceil(source.length * up / down));
  for (let index = 0; index < output.length; index += 1) {
    const position = (index + remove) * down - prePad;
    const first = Math.max(0, Math.ceil((position - (taps.length - 1)) / up));
    const last = Math.min(source.length - 1, Math.floor(position / up));
    // upfirdn's float32 accumulator and increasing source order.
    let sum = 0;
    for (let sample = first; sample <= last; sample += 1) sum = Math.fround(sum + Math.fround(source[sample] * taps[position - sample * up]));
    output[index] = sum;
  }
  return output;
}

const FFT_SIZE = 1024;
const reversed = new Uint16Array(FFT_SIZE);
const cos = new Float64Array(FFT_SIZE / 2), sin = new Float64Array(FFT_SIZE / 2);
for (let index = 0; index < FFT_SIZE; index += 1) {
  let value = index, target = 0;
  for (let bit = 0; bit < 10; bit += 1) { target = (target << 1) | (value & 1); value >>>= 1; }
  reversed[index] = target;
}
for (let index = 0; index < cos.length; index += 1) {
  cos[index] = Math.cos(2 * Math.PI * index / FFT_SIZE); sin[index] = Math.sin(2 * Math.PI * index / FFT_SIZE);
}
function fft(real: Float64Array, imaginary: Float64Array, inverse: boolean): void {
  for (let index = 0; index < FFT_SIZE; index += 1) {
    const target = reversed[index];
    if (target <= index) continue;
    const realValue = real[index], imaginaryValue = imaginary[index];
    real[index] = real[target]; real[target] = realValue;
    imaginary[index] = imaginary[target]; imaginary[target] = imaginaryValue;
  }
  for (let width = 2; width <= FFT_SIZE; width *= 2) {
    const half = width / 2, stride = FFT_SIZE / width;
    for (let start = 0; start < FFT_SIZE; start += width) {
      for (let index = 0; index < half; index += 1) {
        const left = start + index, right = left + half, root = index * stride;
        const rootImaginary = inverse ? sin[root] : -sin[root];
        const r = real[right] * cos[root] - imaginary[right] * rootImaginary;
        const i = real[right] * rootImaginary + imaginary[right] * cos[root];
        real[right] = real[left] - r; imaginary[right] = imaginary[left] - i;
        real[left] += r; imaginary[left] += i;
      }
    }
  }
  if (inverse) for (let index = 0; index < FFT_SIZE; index += 1) { real[index] /= FFT_SIZE; imaginary[index] /= FFT_SIZE; }
}
const clip = (value: number, low: number, high: number): number => Math.max(low, Math.min(high, value));
/** Waveform-only features at actual encoder times, including scipy's source crop/resampling phase. */
export function extractProsody(source: Float32Array, nativeRate: number, times: Float64Array, strideSeconds: number): Uint16Array {
  const output = new Uint16Array(times.length * 4);
  if (!times.length) return output;
  const left = Math.max(0, Math.floor((times[0] - 0.025) * nativeRate));
  const right = Math.min(source.length, Math.ceil((times[times.length - 1] + 0.025) * nativeRate));
  const waveform = resamplePoly(source.subarray(left, right), nativeRate, 8000);
  const window = new Float32Array(320), real = new Float64Array(FFT_SIZE), imaginary = new Float64Array(FFT_SIZE);
  let previousEnergy = 0;
  for (let frame = 0; frame < times.length; frame += 1) {
    const center = roundEven((times[frame] - left / nativeRate) * 8000);
    let squareSum = 0, sum = 0;
    for (let index = 0; index < 320; index += 1) {
      const sample = center + index - 160;
      const value = sample >= 0 && sample < waveform.length ? waveform[sample] : 0;
      window[index] = value; squareSum += Math.fround(value * value); sum += value;
    }
    const rms = Math.fround(Math.sqrt(Math.fround(squareSum / 320)));
    const energy = Math.fround(Math.log(Math.max(rms, 1e-6))), mean = Math.fround(sum / 320);
    real.fill(0); imaginary.fill(0);
    for (let index = 0; index < 320; index += 1) real[index] = Math.fround(window[index] - mean);
    fft(real, imaginary, false);
    for (let index = 0; index < FFT_SIZE; index += 1) { real[index] = real[index] ** 2 + imaginary[index] ** 2; imaginary[index] = 0; }
    fft(real, imaginary, true);
    let lag = 20;
    for (let candidate = 21; candidate <= 123; candidate += 1) if (real[candidate] > real[lag]) lag = candidate;
    const strength = rms < 1e-4 ? 0 : clip(real[lag] / Math.max(real[0], 1e-12), 0, 1);
    const offset = frame * 4;
    output[offset] = float32ToFloat16(clip(Math.fround((energy + 4) / 2), -4, 4));
    output[offset + 1] = float32ToFloat16(strength >= 0.35 ? clip(Math.log(8000 / lag / 180) / 0.6, -3, 3) : 0);
    output[offset + 2] = float32ToFloat16(strength * 2 - 1);
    output[offset + 3] = float32ToFloat16(frame && times[frame] - times[frame - 1] <= strideSeconds * 1.55 ? clip(Math.fround(energy - previousEnergy), -3, 3) : 0);
    previousEnergy = energy;
  }
  return output;
}
export function boundaryCenters(words: readonly AcousticWord[]): Float64Array {
  return Float64Array.from(words, (word, index) => index + 1 < words.length && words[index + 1].startSeconds >= word.endSeconds
    ? (word.endSeconds + words[index + 1].startSeconds) / 2 : word.endSeconds);
}
export function sampleLocalAudio(centers: Float64Array, candidates: readonly AcousticFrames[], strideSeconds: number,
  offsets: readonly number[] = Array.from({ length: LOCAL_FRAMES }, (_, index) => -0.6 + index * 1.2 / 23)):
  { data: Uint16Array; mask: Uint8Array; valid: Uint8Array } {
  if (offsets.length !== LOCAL_FRAMES || offsets.some((value, index) => !Number.isFinite(value) || index > 0 && value <= offsets[index - 1])) {
    throw new Error('C-denoise requires 24 finite increasing boundary offsets.');
  }
  const data = new Uint16Array(centers.length * LOCAL_FRAMES * AUDIO_DIM), mask = new Uint8Array(centers.length * LOCAL_FRAMES), valid = new Uint8Array(centers.length);
  const best = new Float64Array(mask.length).fill(Infinity);
  for (const candidate of candidates) {
    const times = candidate.times;
    if (!times.length) continue;
    for (let word = 0; word < centers.length; word += 1) {
      for (let slot = 0; slot < LOCAL_FRAMES; slot += 1) {
        const target = centers[word] + offsets[slot];
        if (target < candidate.startSeconds || target >= candidate.endSeconds) continue;
        let left = 0, right = times.length;
        while (left < right) { const middle = (left + right) >>> 1; if (times[middle] < target) left = middle + 1; else right = middle; }
        const upper = Math.min(left, times.length - 1), lower = Math.max(upper - 1, 0);
        const frame = Math.abs(times[lower] - target) <= Math.abs(times[upper] - target) ? lower : upper;
        const distance = Math.abs(times[frame] - target), position = word * LOCAL_FRAMES + slot;
        if (distance > strideSeconds * 0.55 + 1e-5 || distance >= best[position] - 1e-7) continue;
        const destination = position * AUDIO_DIM;
        data.set(candidate.features.subarray(frame * 768, (frame + 1) * 768), destination);
        data.set(candidate.prosody.subarray(frame * 4, (frame + 1) * 4), destination + 768);
        mask[position] = 1; valid[word] = 1; best[position] = distance;
      }
    }
  }
  return { data, mask, valid };
}
export const __cDenoiseAcousticTesting = { resamplePoly, extractProsody, sampleLocalAudio, boundaryCenters, roundEven };
