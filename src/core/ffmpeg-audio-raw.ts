/*
 * JavaScript port of FFmpeg n9.0.1 libavfilter/af_biquads.c and
 * libswresample/{resample.c,resample_template.c,aarch64/resample.S,audioconvert.c}.
 * FFmpeg copyright (c) Paul B Mahol, Rob Sykes, Michael Niedermayer,
 * Xiaogang Zhang and contributors; LGPL-2.1-or-later.
 * https://github.com/FFmpeg/FFmpeg/tree/n9.0.1
 * Kaiser I0 here uses its defining series rather than FFmpeg's Boost-derived
 * polynomial (which has a separate Boost Software License).
 */

const OUTPUT_RATE = 16_000;
const f32 = Math.fround;

/** highpass=f=45: two poles, Q=0.707, direct form I, FLTP. */
export function highpassSource(samples: Float32Array, sampleRate: number): Float32Array {
  return filterHighpass(samples, sampleRate, false);
}

function filterHighpass(samples: Float32Array, sampleRate: number, integer: false): Float32Array;
function filterHighpass(samples: Float32Array, sampleRate: number, integer: true): Int16Array;
function filterHighpass(samples: Float32Array, sampleRate: number, integer: boolean): Float32Array | Int16Array {
  if (!Number.isInteger(sampleRate) || sampleRate <= 90) {
    throw new RangeError('highpass source rate must be an integer above 90 Hz');
  }

  const w0 = 2 * Math.PI * 45 / sampleRate;
  const alpha = Math.sin(w0) / (2 * 0.707);
  const a0 = 1 + alpha;
  const a1 = f32(2 * Math.cos(w0) / a0);
  const a2 = f32(-(1 - alpha) / a0);
  const b0 = f32(((1 + Math.cos(w0)) / 2) / a0);
  const b1 = f32(-(1 + Math.cos(w0)) / a0);
  const b2 = b0;
  const output = integer ? new Int16Array(samples.length) : new Float32Array(samples.length);
  let i1 = 0, i2 = 0, o1 = 0, o2 = 0;

  // Clang/ARM64 evaluates b1's product first (FMUL), then contracts b2,
  // b0, a2 and a1 in that order (FMADD). Reversing b1/b2 changes feedback
  // after thousands of samples, even though their sum is algebraically equal.
  let i = 0;
  for (; i + 1 < samples.length; i += 2) {
    const x0 = integer ? Math.trunc(samples[i] * 32768) : samples[i];
    o2 = f32(f32(f32(f32(f32(i1 * b1) + i2 * b2) + x0 * b0) + o2 * a2) + o1 * a1);
    i2 = x0;
    output[i] = integer ? Math.max(-32768, Math.min(32767, Math.trunc(o2))) : o2;
    const x1 = integer ? Math.trunc(samples[i + 1] * 32768) : samples[i + 1];
    o1 = f32(f32(f32(f32(f32(i2 * b1) + i1 * b2) + x1 * b0) + o1 * a2) + o2 * a1);
    i1 = x1;
    output[i + 1] = integer ? Math.max(-32768, Math.min(32767, Math.trunc(o1))) : o1;
  }
  if (i < samples.length) {
    const x = integer ? Math.trunc(samples[i] * 32768) : samples[i];
    const value = f32(f32(f32(f32(f32(i1 * b1) + x * b0) + i2 * b2) + o1 * a1) + o2 * a2);
    output[i] = integer ? Math.max(-32768, Math.min(32767, Math.trunc(value))) : value;
  }
  return output;
}

function gcd(a: number, b: number): number {
  while (b) [a, b] = [b, a % b];
  return a;
}

// Modified Bessel I0, needed only on [0, 9]. All operations are double as in
// FFmpeg's filter-bank construction; at this beta the convergent series gives
// coefficients within a few double ULPs of av_bessel_i0.
function besselI0(x: number): number {
  const quarterSquare = x * x / 4;
  let sum = 1, term = 1;
  for (let k = 1; k <= 32; k++) {
    term *= quarterSquare / (k * k);
    const next = sum + term;
    if (next === sum) break;
    sum = next;
  }
  return sum;
}

function makeKaiserBank(inputRate: number): { bank: Float32Array; phases: number; taps: number; stride: number } {
  // swr defaults: filter_size=32, phase_shift=10, exact_rational=1,
  // cutoff=0.97, filter_type=kaiser, kaiser_beta=9.
  const factor = Math.min(0.97 * OUTPUT_RATE / inputRate, 1);
  const taps = Math.ceil(32 / factor + 1) & ~1; // FFALIGN(ceil(32/factor), 2)
  const phases = Math.min(1024, OUTPUT_RATE / gcd(OUTPUT_RATE, inputRate));
  const stride = (taps + 7) & ~7; // FFALIGN(filter_length, 8)
  const bank = new Float32Array((phases + 1) * stride);
  const tab = new Float64Array(taps);
  const center = (taps - 1) >> 1;
  let norm = 0;

  for (let phase = 0; phase < (phases & 1 ? phases : phases / 2 + 1); phase++) {
    for (let tap = 0; tap < taps; tap++) {
      const x = Math.PI * ((tap - center) - phase / phases) * factor;
      const w = 2 * x / (factor * taps * Math.PI);
      const y = (x === 0 ? 1 : Math.sin(x) / x) * besselI0(9 * Math.sqrt(Math.max(1 - w * w, 0)));
      tab[tap] = y;
      if (phase === 0) norm += y;
    }
    const offset = phase * stride;
    for (let tap = 0; tap < taps; tap++) bank[offset + tap] = tab[tap] / norm;
    // build_filter mirrors only half the phases, including the even midpoint.
    if (!(phases & 1)) {
      const opposite = (phases - phase) * stride;
      for (let tap = 0; tap < taps; tap++) bank[opposite + taps - 1 - tap] = bank[offset + tap];
    }
  }
  // The extra phase is phase 0 shifted by one tap, used by linear_interp.
  bank[phases * stride] = bank[stride - 1];
  bank.set(bank.subarray(0, stride - 1), phases * stride + 1);
  return { bank, phases, taps, stride };
}

/**
 * swr FLTP -> S16 at 16 kHz. `outputFrames` is the apad/atrim whole_len;
 * samples beyond the naturally flushed resampler output are zero padded.
 */
export function resampleToPcm16(samples: Float32Array, sampleRate: number, outputFrames: number): Int16Array {
  return resampleFloat(samples, sampleRate, outputFrames, false);
}

function resampleFloat(samples: Float32Array, sampleRate: number, outputFrames: number, rawPcmPackets: boolean): Int16Array {
  if (!Number.isInteger(sampleRate) || sampleRate <= 0 || !Number.isSafeInteger(outputFrames) || outputFrames < 0) {
    throw new RangeError('invalid source sample rate or output frame count');
  }
  const output = new Int16Array(outputFrames);
  if (!samples.length || !outputFrames) return output;
  const naturallyProduced = Math.min(outputFrames, Math.round(samples.length * OUTPUT_RATE / sampleRate));
  if (sampleRate === OUTPUT_RATE) {
    for (let n = 0; n < naturallyProduced; n++) output[n] = toPcm16(samples[n]);
    return output;
  }
  const { bank, phases, taps, stride } = makeKaiserBank(sampleRate);
  const center = (taps - 1) >> 1;
  const rateGcd = gcd(OUTPUT_RATE, sampleRate * phases);
  const sourceIncr = OUTPUT_RATE / rateGcd;
  const destinationIncr = sampleRate * phases / rateGcd;
  const phaseStep = Math.floor(destinationIncr / sourceIncr);
  const fractionStep = destinationIncr % sourceIncr;
  let phase = 0, fraction = 0;
  let source = -center;
  const last = samples.length - 1;
  const simdTaps = taps & ~7;
  // FFmpeg's mono PCM16 WAV decoder feeds 4096-source-sample packets. In
  // the raw graph swr_convert emits all samples whose last FIR tap exists;
  // the final 0..15 samples of EACH emitted frame use audioconvert.c's
  // scalar lrintf instead of ARM64's 16-wide NEON narrowing.
  const sourceTail = taps - 1 - center;
  let sourceConsumed = Math.min(samples.length, 4096);
  let packetStart = 0;
  let packetEnd = Math.min(naturallyProduced, Math.max(0, Math.ceil((sourceConsumed - sourceTail) * OUTPUT_RATE / sampleRate)));
  let scalarFrom = packetEnd - ((packetEnd - packetStart) & 15);
  for (let n = 0; n < naturallyProduced; n++) {
    const filter = phase * stride;
    let value = 0;
    if (fractionStep !== 0) {
      // resample_linear (scalar float implementation): adjacent phase
      // convolutions followed by the fractional-phase interpolation.
      let next = 0;
      for (let j = 0; j < taps; j++) {
        const x = samples[reflect(source + j, last)];
        value = f32(value + x * bank[filter + j]);
        next = f32(next + x * bank[filter + stride + j]);
      }
      value = f32(value + f32(next - value) * (1 / sourceIncr) * fraction);
    } else if (taps >= 8) {
      // arm64 resample.S accumulates four fused float lanes over a multiple
      // of EIGHT taps, reduces them with two pairwise FADDP instructions,
      // and finishes the remaining taps with scalar FMADD in C. The 44.1k
      // filter has 92 taps: the last four are not included in the SIMD sum.
      let p0 = 0, p1 = 0, p2 = 0, p3 = 0;
      if (source >= 0 && source + taps <= samples.length) {
        for (let j = 0; j < simdTaps; j += 4) {
          p0 = f32(p0 + samples[source + j] * bank[filter + j]);
          p1 = f32(p1 + samples[source + j + 1] * bank[filter + j + 1]);
          p2 = f32(p2 + samples[source + j + 2] * bank[filter + j + 2]);
          p3 = f32(p3 + samples[source + j + 3] * bank[filter + j + 3]);
        }
      } else {
        for (let j = 0; j < simdTaps; j += 4) {
          p0 = f32(p0 + samples[reflect(source + j, last)] * bank[filter + j]);
          p1 = f32(p1 + samples[reflect(source + j + 1, last)] * bank[filter + j + 1]);
          p2 = f32(p2 + samples[reflect(source + j + 2, last)] * bank[filter + j + 2]);
          p3 = f32(p3 + samples[reflect(source + j + 3, last)] * bank[filter + j + 3]);
        }
      }
      value = f32(f32(p0 + p1) + f32(p2 + p3));
      for (let j = simdTaps; j < taps; j++) {
        value = f32(value + samples[reflect(source + j, last)] * bank[filter + j]);
      }
    } else {
      for (let j = 0; j < taps; j++) {
        value = f32(value + samples[reflect(source + j, last)] * bank[filter + j]);
      }
    }
    if (rawPcmPackets) {
      while (n >= packetEnd && packetEnd < naturallyProduced) {
        packetStart = packetEnd;
        if (sourceConsumed < samples.length) {
          sourceConsumed = Math.min(samples.length, sourceConsumed + 4096);
          packetEnd = Math.min(naturallyProduced, Math.max(packetStart,
            Math.ceil((sourceConsumed - sourceTail) * OUTPUT_RATE / sampleRate)));
        } else {
          packetEnd = naturallyProduced; // reflected tail emitted by swr flush
        }
        scalarFrom = packetEnd - ((packetEnd - packetStart) & 15);
      }
    }
    output[n] = rawPcmPackets && n >= scalarFrom ? toPcm16Scalar(value) : toPcm16(value);
    phase += phaseStep;
    fraction += fractionStep;
    if (fraction >= sourceIncr) {
      fraction -= sourceIncr;
      phase++;
    }
    source += Math.floor(phase / phases);
    phase %= phases;
  }
  return output;
}

/**
 * Raw graph: s16 -> highpass(s16p) -> swr(FLTP internally) -> s16.
 * The denoised graph instead negotiates FLTP at the highpass input.
 */
export function prepareRawPcm16(samples: Float32Array, sampleRate: number, outputFrames: number): Int16Array {
  if (!Number.isInteger(sampleRate) || sampleRate <= 90 || !Number.isSafeInteger(outputFrames) || outputFrames < 0) {
    throw new RangeError('invalid source sample rate or output frame count');
  }
  if (!samples.length || !outputFrames) return new Int16Array(outputFrames);
  const filtered = filterHighpass(samples, sampleRate, true);
  const normalized = new Float32Array(filtered.length);
  for (let i = 0; i < filtered.length; i++) normalized[i] = filtered[i] / 32768;
  return resampleFloat(normalized, sampleRate, outputFrames, true);
}

function reflect(index: number, last: number): number {
  // swri_resampler.invert_initial_buffer mirrors around sample zero; flush
  // mirrors around the last available sample. The resampler never asks for
  // farther than its half-filter delay at either end.
  if (index < 0) index = -index;
  else if (index > last) index = 2 * last - index + 1;
  return Math.max(0, Math.min(last, index));
}

function toPcm16(value: number): number {
  // aarch64/audio_convert_neon.S: fcvtzs(x * 2^31), sqrshrn #16.
  // SWR_DITHER_NONE adds no noise. FCVTZS truncates Q31 toward zero BEFORE
  // SQRSHRN's rounded 16-bit narrowing; this matters for values within one
  // Q31 quantum of an exact half-integer PCM sample.
  const q31 = Math.max(-2147483648, Math.min(2147483647, Math.trunc(value * 2147483648)));
  const rounded = Math.floor((q31 + 32768) / 65536);
  return Math.max(-32768, Math.min(32767, rounded));
}

function toPcm16Scalar(value: number): number {
  // audioconvert.c's fallback for an unaligned packet tail: lrintf uses
  // nearest-even instead of NEON SQRSHRN's upward half-tie.
  const scaled = f32(value * 32768);
  const lower = Math.floor(scaled);
  const fraction = scaled - lower;
  const rounded = fraction > 0.5 || (fraction === 0.5 && (lower & 1) !== 0) ? lower + 1 : lower;
  return Math.max(-32768, Math.min(32767, rounded));
}
