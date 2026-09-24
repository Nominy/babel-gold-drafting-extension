/*
 * Mono FLTP adaptation of FFmpeg n9.0.1 libavfilter/af_afftdn.c (2018 FFmpeg Project),
 * https://github.com/FFmpeg/FFmpeg/blob/n9.0.1/libavfilter/af_afftdn.c
 * Copyright (c) 2018 The FFmpeg Project. Licensed under LGPL-2.1-or-later;
 * see https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html .
 * This port retains afftdn's nr=10:nf=-50:tn=1 signal path, not a generic gate.
 */

const SCALE = 1 << 23;
const C = Math.LN10 * 0.1;

// AV_TX_FLOAT_RDFT uses single-precision spectral samples; the noise model,
// window, accumulation and band computations in afftdn are double precision.
// Keep FFT working arithmetic double, rounding its observable FLTP boundaries.
function fft(re: Float64Array, im: Float64Array, inverse: boolean): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let v = re[i]; re[i] = re[j]; re[j] = v;
      v = im[i]; im[i] = im[j]; im[j] = v;
    }
  }
  for (let span = 2; span <= n; span <<= 1) {
    const angle = (inverse ? 2 : -2) * Math.PI / span;
    const wr = Math.cos(angle), wi = Math.sin(angle);
    for (let base = 0; base < n; base += span) {
      let r = 1, ii = 0;
      for (let j = 0; j < span / 2; j++) {
        const a = base + j, b = a + span / 2;
        const tr = r * re[b] - ii * im[b];
        const ti = r * im[b] + ii * re[b];
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
        const next = r * wr - ii * wi;
        ii = r * wi + ii * wr;
        r = next;
      }
    }
  }
}

function limitGain(a: number, b: number): number {
  if (a > 1) return (b * a - 1) / (b + a - 2);
  if (a < 1) return (b * a - 2 * a + 1) / (b - a);
  return 1;
}

/**
 * Process already-highpassed, mono normalized FLTP samples with FFmpeg n9.0.1
 * afftdn=nr=10:nf=-50:tn=1. The output retains FFmpeg's two-hop latency and
 * the exact input length (afftdn does not drain its overlap-add buffer at EOF).
 */
export function denoiseForActivity(samples: Float32Array, sampleRate: number): Float32Array {
  if (!Number.isInteger(sampleRate) || sampleRate < 80) throw new RangeError('Invalid sample rate');
  if (!samples.length) return new Float32Array(0);

  const hop = Math.trunc(sampleRate / 80);
  const windowLength = 3 * hop;
  let fftLength = 1;
  while (fftLength <= windowLength) fftLength *= 2;
  const bins = fftLength / 2 + 1;
  const rate = Math.fround(sampleRate); // afftdn's sample_rate field is float

  const bin2band = new Int32Array(bins);
  for (let i = 0; i < bins; i++) {
    const f = (0.5 * i * rate) / fftLength;
    const d = f / 7500;
    bin2band[i] = Math.round(1.25 * (13 * Math.atan(7.6e-4 * f) + 3.5 * Math.atan(d * d)));
  }
  const bandCount = bin2band[bins - 1] + 1;
  const alpha = new Float64Array(bandCount);
  const beta = new Float64Array(bandCount);
  const sar = hop / rate;
  for (let i = 0, j = 0; i < bins; i++) {
    if (bin2band[i] > j) {
      const d6 = (i - 1) * rate / fftLength;
      const d7 = Math.min(0.008 + 2.2 / d6, 0.03);
      alpha[j] = Math.exp(-sar / d7);
      beta[j] = 1 - alpha[j];
      j = bin2band[i];
    }
  }

  const window = new Float64Array(windowLength);
  const wscale = Math.sqrt(8 / (9 * fftLength));
  let sum = 0;
  for (let i = 0; i < windowLength; i++) {
    const sine = Math.sin(i * Math.PI / windowLength);
    const v = wscale * sine * sine;
    window[i] = v;
    sum += v * v;
  }
  const floor = 2 ** 48 * Math.exp(-23.025558369790467) * 0.5 * sum;

  // Default white-noise profile has zero band_noise. set_band_parameters
  // linearly interpolates that profile to zero for every FFT bin.
  const maxVar = floor * Math.exp(50 * C);
  const absVar = Math.max(maxVar, 1);
  const maxGain = Math.exp(10 * 0.5 * C);
  const minAbsVar = absVar / (maxGain * maxGain);

  const spread = new Float64Array(bandCount * bandCount);
  const bandWeights = new Float64Array(bandCount);
  const priorBand = new Float64Array(bandCount);
  const bandExcit = new Float64Array(bandCount);
  const bandAmt = new Float64Array(bandCount);
  const p1 = 0.1 ** (2.5 / 1.25);
  const p2 = 0.1 ** (1 / 1.25);
  for (let k = 0; k < bins; k++) bandWeights[bin2band[k]] += 1;
  for (let m = 0; m < bandCount; m++) {
    let norm = 0;
    for (let n = 0; n < bandCount; n++) {
      const value = n < m ? p2 ** (m - n) : n > m ? p1 ** (n - m) : 1;
      spread[m * bandCount + n] = value;
      norm += value * bandWeights[n];
    }
    let scale: number;
    if (m < Math.round(12 * 1.25)) scale = 0.1 ** (1.45 + 0.1 * m / 1.25);
    else scale = 0.1 ** (2.5 - 0.2 * (m / 1.25 - 14));
    scale = Math.max(0.1 ** 2.5, Math.min(0.1, scale)) / norm;
    for (let n = 0; n < bandCount; n++) spread[m * bandCount + n] *= scale;
  }
  let noiseFloor = -50;
  let trackedAbsVar = absVar;
  let trackedMinVar = minAbsVar;

  const winframe = new Float32Array(windowLength);
  const overlap = new Float64Array(windowLength);
  const re = new Float64Array(fftLength);
  const im = new Float64Array(fftLength);
  const prior = new Float64Array(bins);
  const gain = new Float64Array(bins);
  const cleanPower = new Float64Array(bins);
  const noisy = new Float64Array(bins);
  const output = new Float32Array(samples.length);

  for (let start = 0, frame = 0; start < samples.length; start += hop, frame++) {
    const count = Math.min(hop, samples.length - start);
    winframe.copyWithin(0, hop);
    winframe.set(samples.subarray(start, start + count), windowLength - hop);
    winframe.fill(0, windowLength - hop + count);
    for (let i = 0; i < windowLength; i++) {
      re[i] = Math.fround(window[i] * winframe[i] * SCALE);
      im[i] = 0;
    }
    re.fill(0, windowLength);
    im.fill(0, windowLength);
    fft(re, im, false);
    for (let i = 0; i < bins; i++) {
      re[i] = Math.fround(re[i]);
      im[i] = Math.fround(im[i]);
      noisy[i] = Math.hypot(re[i], im[i]);
      const power = noisy[i] * noisy[i];
      const ratio = frame === 0 ? 1 : 0.5;
      const newMagVar = ratio * prior[i] + (1 - ratio) * Math.max(power / trackedAbsVar - 1, 0);
      const newGain = newMagVar / (1 + newMagVar);
      prior[i] = power / trackedAbsVar * newGain * newGain;
      cleanPower[i] = power * newGain * newGain;
      gain[i] = newGain;
    }

    // tn=1: only spectrally flat blocks revise the absolute noise floor.
    let logSum = 0, magSum = 0, aboveFloor = 0;
    for (let i = 0; i < bins; i++) {
      if (noisy[i] > floor) {
        logSum += Math.log(noisy[i]);
        magSum += noisy[i];
        aboveFloor++;
      }
    }
    const mean = magSum / Math.max(aboveFloor, 1);
    if (Math.exp(logSum / Math.max(aboveFloor, 1)) / mean > 0.8) {
      // Recalculation of abs_var and min_abs_var happens below for this frame.
      let offset = 0;
      for (let i = 0; i < bins; i++) offset = Math.max(offset, Math.abs(noisy[i] - mean));
      const rawFloor = 10 * Math.log10(mean) - 100 + offset / mean;
      // av_clipd uses FFMAX/FFMIN, so NaN (the empty-spectrum -Inf + Inf
      // case) clips to -90 rather than poisoning the filter's noise model.
      const newFloor = Math.min(-20, rawFloor > -90 ? rawFloor : -90);
      noiseFloor = 0.1 * newFloor + noiseFloor * 0.9;
      trackedAbsVar = Math.max(floor * Math.exp((100 + noiseFloor) * C), 1);
      trackedMinVar = trackedAbsVar / (maxGain * maxGain);
    }

    bandExcit.fill(0);
    bandAmt.fill(0);
    for (let i = 0; i < bins; i++) bandExcit[bin2band[i]] += cleanPower[i];
    for (let i = 0; i < bandCount; i++) {
      bandExcit[i] = Math.max(bandExcit[i], alpha[i] * bandExcit[i] + beta[i] * priorBand[i]);
      priorBand[i] = bandExcit[i];
    }
    for (let j = 0; j < bandCount; j++) {
      for (let k = 0; k < bandCount; k++) bandAmt[j] += spread[j * bandCount + k] * bandExcit[k];
    }
    for (let i = 0; i < bins; i++) {
      const amount = bandAmt[bin2band[i]];
      if (amount > trackedAbsVar) gain[i] = 1;
      else if (amount > trackedMinVar) gain[i] = limitGain(gain[i], Math.sqrt(trackedAbsVar / amount));
      else gain[i] = limitGain(gain[i], maxGain);
      const g = Math.fround(gain[i]);
      re[i] = Math.fround(Math.fround(re[i]) * g);
      im[i] = Math.fround(Math.fround(im[i]) * g);
    }
    for (let i = 1; i < fftLength / 2; i++) {
      re[fftLength - i] = re[i];
      im[fftLength - i] = -im[i];
    }
    fft(re, im, true); // FFmpeg's inverse transform is unnormalized.
    for (let i = 0; i < windowLength; i++) {
      overlap[i] += window[i] * Math.fround(re[i]) / SCALE;
    }
    for (let i = 0; i < count; i++) output[start + i] = overlap[i];
    overlap.copyWithin(0, hop);
    overlap.fill(0, windowLength - hop);
  }
  return output;
}
