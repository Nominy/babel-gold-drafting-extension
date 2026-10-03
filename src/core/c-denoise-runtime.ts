import { BertTokenizer } from '@huggingface/transformers';
import * as ort from 'onnxruntime-web/webgpu';
import { AUDIO_DIM, LOCAL_FRAMES, float16ToFloat32, float32ToFloat16, type AcousticWord } from './c-denoise-acoustic';

export const C_DENOISE_LABELS = ['O', 'COMMA', 'PERIOD', 'QUESTION', 'HYPHEN_JOIN', 'DASH_SINGLE', 'DASH_DOUBLE'] as const;
export type CDenoiseLabel = typeof C_DENOISE_LABELS[number];
export const ASR_SOURCE_SHA = '02cea9973d0e839f6a3eeca101b83a93f93a066c2da2e3ebfa176d57e61d84d3';
export const C_DENOISE_SOURCE_SHA = '0a01c3535fb66627b13f266bc59ab7b95c2aa85f7413a051d1e17294515ded5a';
export interface CDenoiseConfig {
  schema: string;
  labels: readonly string[];
  sample_rate: number;
  frame_stride_samples: number;
  frame_center_offset_samples: number;
  required_gpu_buffer_bytes: number;
  local_offsets_seconds: number[];
  sources: { asrCheckpointSha256: string; cDenoiseCheckpointSha256: string; baseModelFiles: Record<string, string> };
  graphs: Record<'asr' | 'context' | 'denoise', { path: string; inputs?: Array<{ name: string; dtype: string }>; external_data?: Array<{ path: string; location: string }> }>;
}
export interface ContextWindow { left: number; coreLeft: number; coreRight: number; right: number }
export interface WordRange { wordStart: number; wordEnd: number }
export interface CDenoiseResources {
  tokenizer: InstanceType<typeof BertTokenizer>;
  context: ort.InferenceSession;
  denoise: ort.InferenceSession;
  config: CDenoiseConfig;
}
export function validateCDenoiseConfig(value: Record<string, unknown>): CDenoiseConfig {
  const expected: Record<string, number | string> = {
    schema: 'babel-c-denoise-webgpu-v1', core_words: 32, context_words: 32, max_tokens: 256, steps: 4,
    audio_dim: 772, text_dim: 1024, local_frames: 24, sample_rate: 16000, audio_core_seconds: 20, audio_context_seconds: 2,
    frame_stride_samples: 640, frame_center_offset_samples: 160,
    boundary_rule: 'gap-midpoint-if-nonoverlap-otherwise-word-end'
  };
  for (const [key, actual] of Object.entries(expected)) if (value[key] !== actual) throw new Error(`C-denoise config ${key} must be ${actual}. Reinstall the verified C-denoise bundle.`);
  const config = value as unknown as CDenoiseConfig;
  if (!Array.isArray(config.labels) || config.labels.join('|') !== C_DENOISE_LABELS.join('|')) throw new Error('C-denoise label order differs from the seven learned labels.');
  if (!Number.isInteger(config.frame_stride_samples) || config.frame_stride_samples <= 0 || !Number.isFinite(config.frame_center_offset_samples)
    || 320000 % config.frame_stride_samples || 32000 % config.frame_stride_samples) throw new Error('C-denoise encoder frame geometry is invalid.');
  if (!Number.isSafeInteger(config.required_gpu_buffer_bytes) || config.required_gpu_buffer_bytes <= 0) throw new Error('C-denoise config has no verified largest GPU buffer size.');
  if (!config.sources || config.sources.asrCheckpointSha256 !== ASR_SOURCE_SHA || config.sources.cDenoiseCheckpointSha256 !== C_DENOISE_SOURCE_SHA) throw new Error('C-denoise source checkpoints do not match the accepted model.');
  if (!Array.isArray(config.local_offsets_seconds) || config.local_offsets_seconds.length !== LOCAL_FRAMES
    || config.local_offsets_seconds.some((offset, index) => Math.abs(offset - (-0.6 + index * 1.2 / 23)) > 1e-12)) throw new Error('C-denoise boundary offsets differ from the learned acoustic contract.');
  const prosody = value.prosody as Record<string, unknown> | undefined;
  const expectedProsody: Record<string, number | string> = { sample_rate: 8000, window_samples: 320, fft_size: 1024, min_lag: 20, max_lag: 123, rounding: 'ties-to-even', voiced_strength: 0.35, low_rms_threshold: 1e-4 };
  for (const [key, expectedValue] of Object.entries(expectedProsody)) if (prosody?.[key] !== expectedValue) throw new Error(`C-denoise prosody ${key} differs from the source-waveform contract.`);
  if (JSON.stringify(value.context_flags) !== JSON.stringify(['left_context_exists', 'right_context_exists', 'left_context_missing', 'right_context_missing'])) throw new Error('C-denoise context flag order is invalid.');
  const rows = value.row_policy as Record<string, unknown> | undefined;
  if (rows?.gap_seconds !== 0.8 || rows?.max_span_seconds !== 12 || rows?.max_words !== 32) throw new Error('C-denoise runtime row policy differs from training.');
  const paths = { asr: 'asr/v3_ctc.onnx', context: 'punctuation/context.fp16.onnx', denoise: 'punctuation/denoise.fp16.onnx' };
  for (const key of ['asr', 'context', 'denoise'] as const) if (config.graphs?.[key]?.path !== paths[key]) throw new Error(`C-denoise ${key} graph path is invalid.`);
  for (const name of ['attention_mask', 'first_subtoken']) {
    if (config.graphs.context.inputs?.find((input) => input.name === name)?.dtype !== 'int32') {
      throw new Error(`C-denoise ${name} requires GPU-compatible int32 inputs. Download the updated WebGPU bundle.`);
    }
  }
  if (config.graphs.denoise.inputs?.find((input) => input.name === 'noisy_labels')?.dtype !== 'int32') {
    throw new Error('C-denoise noisy_labels requires GPU-compatible int32 inputs. Download the updated WebGPU bundle.');
  }
  return config;
}
/** Legacy ContextDataset grouping policy for checkpoint parity; no word filtering, rewriting, or timing interpolation. */
export function buildRows(words: readonly AcousticWord[], timeValid?: ArrayLike<boolean | number>): WordRange[] {
  if (!words.length) return [];
  if (timeValid && timeValid.length !== words.length) throw new Error('Word timing validity does not cover the immutable source words.');
  const valid = (index: number): boolean => timeValid ? Boolean(timeValid[index]) :
    Number.isFinite(words[index].startSeconds) && Number.isFinite(words[index].endSeconds) && words[index].startSeconds >= 0 && words[index].endSeconds > words[index].startSeconds;
  const rows: WordRange[] = [];
  let start = 0;
  for (let index = 1; index < words.length; index += 1) {
    if (index - start >= 32 ||
      valid(index) && valid(index - 1) && words[index].startSeconds - words[index - 1].endSeconds >= 0.8 ||
      valid(index) && valid(start) && words[index].endSeconds - words[start].startSeconds > 12) {
      rows.push({ wordStart: start, wordEnd: index }); start = index;
    }
  }
  rows.push({ wordStart: start, wordEnd: words.length });
  return rows;
}
export function buildWindows(rows: readonly WordRange[], costs: readonly number[]): ContextWindow[] {
  const prefix = [0];
  for (const cost of costs) {
    if (!Number.isInteger(cost) || cost < 1 || cost > 254) throw new Error('A source word cannot fit intact in the C-denoise 256-token budget.');
    prefix.push(prefix[prefix.length - 1] + cost);
  }
  const windows: ContextWindow[] = [];
  let owned = 0;
  for (const row of rows) {
    let start = row.wordStart;
    while (start < row.wordEnd) {
      if (start !== owned) throw new Error('C-denoise window construction lost or duplicated central ownership.');
      let end = Math.min(row.wordEnd, start + 32);
      while (prefix[end] - prefix[start] > 254) end -= 1;
      if (end <= start) throw new Error('C-denoise source word cannot fit intact.');
      let left = Math.max(0, start - 32), right = Math.min(costs.length, end + 32);
      while (prefix[right] - prefix[left] > 254) {
        if (start - left > right - end) left += 1;
        else if (right > end) right -= 1;
        else left += 1;
      }
      windows.push({ left, coreLeft: start, coreRight: end, right });
      owned = start = end;
    }
  }
  if (owned !== costs.length) throw new Error('C-denoise window construction omitted source boundaries.');
  return windows;
}
/** Punctuate each audio segment jointly; only the tokenizer budget can split its core. */
export function buildSegmentWindows(rows: readonly WordRange[], costs: readonly number[]): ContextWindow[] {
  const prefix = [0];
  for (const cost of costs) {
    if (!Number.isInteger(cost) || cost < 1 || cost > 254) throw new Error('A source word cannot fit intact in the C-denoise 256-token budget.');
    prefix.push(prefix[prefix.length - 1] + cost);
  }
  const windows: ContextWindow[] = [];
  let owned = 0;
  for (const row of rows) {
    if (row.wordStart !== owned || row.wordEnd <= row.wordStart || row.wordEnd > costs.length) throw new Error('Audio segments must own every source word exactly once.');
    let start = row.wordStart;
    const fits = prefix[row.wordEnd] - prefix[start] <= 254;
    while (start < row.wordEnd) {
      let end = start;
      // Long segments reserve half the token budget for overlapping context.
      const budget = fits ? 254 : Math.max(126, costs[start]);
      while (end < row.wordEnd && prefix[end + 1] - prefix[start] <= budget) end++;
      let left = Math.max(0, start - 32), right = Math.min(costs.length, end + 32);
      while (prefix[right] - prefix[left] > 254) {
        if (start - left > right - end) left++;
        else if (right > end) right--;
        else left++;
      }
      windows.push({ left, coreLeft: start, coreRight: end, right });
      owned = start = end;
    }
  }
  if (owned !== costs.length) throw new Error('Audio segments omitted source words.');
  return windows;
}
function specialId(tokenizer: CDenoiseResources['tokenizer'], property: string, fallback: string): number {
  const value = (tokenizer as unknown as Record<string, unknown>)[property];
  const id = Number.isInteger(value) ? Number(value) : tokenizer.convert_tokens_to_ids(fallback);
  if (!Number.isInteger(id) || id < 0) throw new Error(`Tokenizer has no usable ${property}.`);
  return id;
}
export function tokenizeWords(tokenizer: CDenoiseResources['tokenizer'], words: readonly AcousticWord[]): number[][] {
  return words.map((word, index) => {
    if (!word.text || word.text.trim() !== word.text || /\s/u.test(word.text)) throw new Error(`C-denoise source word ${index} is not a lexical token.`);
    const ids = tokenizer.encode(word.text.toLocaleLowerCase('ru-RU').replace(/ß/g, 'ss').replace(/ς/g, 'σ'), { add_special_tokens: false });
    if (!ids.length || ids.some((id) => !Number.isInteger(id) || id < 0)) throw new Error(`C-denoise tokenizer lost source word ${index}.`);
    return ids;
  });
}
export function encodeWindow(tokenizer: CDenoiseResources['tokenizer'], tokens: readonly number[][], window: ContextWindow,
  missing: readonly [boolean, boolean] = [false, false]): Record<string, ort.Tensor> {
  const selected = tokens.slice(window.left, window.right), count = selected.length;
  const sequence = selected.reduce((sum, ids) => sum + ids.length, 2);
  if (sequence > 256) throw new Error('C-denoise tokenizer changed word costs; refusing truncation.');
  const inputIds = new BigInt64Array(sequence), first = new Int32Array(count);
  inputIds[0] = BigInt(specialId(tokenizer, 'cls_token_id', '[CLS]'));
  let position = 1;
  for (let word = 0; word < count; word += 1) { first[word] = position; for (const id of selected[word]) inputIds[position++] = BigInt(id); }
  inputIds[position] = BigInt(specialId(tokenizer, 'sep_token_id', '[SEP]'));
  return {
    input_ids: new ort.Tensor('int64', inputIds, [1, sequence]),
    attention_mask: new ort.Tensor('int32', new Int32Array(sequence).fill(1), [1, sequence]),
    token_type_ids: new ort.Tensor('int64', new BigInt64Array(sequence), [1, sequence]),
    first_subtoken: new ort.Tensor('int32', first, [1, count]),
    word_mask: new ort.Tensor('bool', new Uint8Array(count).fill(1), [1, count]),
    context_flags: new ort.Tensor('float32', Float32Array.of(Number(window.left > 0), Number(window.right < tokens.length), Number(window.left === 0 && missing[0]), Number(window.right === tokens.length && missing[1])), [1, 4])
  };
}
/** Four-step aligned_punctuation_model.predict, including stable low-confidence masking and frozen candidates. */
export async function refineLabels(wordMask: Uint8Array, run: (noisy: BigInt64Array, level: number, step: number) => Promise<Float32Array>): Promise<Uint8Array> {
  const count = wordMask.reduce((sum, valid) => sum + Number(Boolean(valid)), 0);
  if (!count) throw new Error('C-denoise requires at least one valid word.');
  const noisy = new BigInt64Array(wordMask.length).fill(7n), candidates = new Float32Array(wordMask.length * 7);
  const confidence = new Float32Array(wordMask.length), labels = new Uint8Array(wordMask.length), order = Array.from({ length: wordMask.length }, (_, index) => index);
  for (let step = 0; step < 4; step += 1) {
    let maskedCount = 0;
    for (let word = 0; word < wordMask.length; word += 1) if (wordMask[word] && noisy[word] === 7n) maskedCount += 1;
    const output = await run(noisy, Math.fround(maskedCount / count), step);
    if (output.length !== candidates.length) throw new Error('C-denoise logits do not cover every source boundary.');
    for (let word = 0; word < wordMask.length; word += 1) {
      const offset = word * 7;
      if (wordMask[word] && noisy[word] === 7n) {
        candidates.set(output.subarray(offset, offset + 7), offset);
        let best = 0;
        for (let label = 0; label < 7; label += 1) {
          if (!Number.isFinite(candidates[offset + label])) throw new Error(`C-denoise emitted nonfinite logits for word ${word}.`);
          if (candidates[offset + label] > candidates[offset + best]) best = label;
        }
        labels[word] = best;
        let denominator = 0;
        for (let label = 0; label < 7; label += 1) denominator += Math.exp(candidates[offset + label] - candidates[offset + best]);
        confidence[word] = 1 / denominator;
      }
    }
    if (step === 3) return labels;
    const remaining = Math.floor((count * (3 - step) + 3) / 4);
    order.sort((left, right) => (wordMask[left] ? confidence[left] : Infinity) - (wordMask[right] ? confidence[right] : Infinity) || left - right);
    for (let rank = 0; rank < order.length; rank += 1) { const word = order[rank]; noisy[word] = rank < remaining || !wordMask[word] ? 7n : BigInt(labels[word]); }
  }
  throw new Error('C-denoise iteration did not complete.');
}
function halfTensor(tensor: ort.Tensor | undefined, count: number, width: number, name: string): ort.Tensor {
  if (!tensor || tensor.type !== 'float16' || tensor.dims.join(',') !== `1,${count},${width}` || tensor.data.length !== count * width) throw new Error(`${name} must be float16 [1,${count},${width}].`);
  return tensor;
}
export async function predictCDenoise(words: readonly AcousticWord[], audio: { data: Uint16Array; mask: Uint8Array; valid: Uint8Array }, resources: CDenoiseResources, segments?: readonly WordRange[]): Promise<Uint8Array> {
  if (!words.length) return new Uint8Array();
  const tokens = tokenizeWords(resources.tokenizer, words), costs = tokens.map((ids) => ids.length);
  const windows = segments ? buildSegmentWindows(segments, costs) : buildWindows(buildRows(words), costs);
  const labels = new Uint8Array(words.length);
  for (const window of windows) {
    const count = window.right - window.left, contextFeeds = encodeWindow(resources.tokenizer, tokens, window);
    let contextOutput: Record<string, ort.Tensor> = {};
    const coreFeeds: Record<string, ort.Tensor> = {};
    try {
      // This is the sole BERT pass for this window; four iterations only run the small learned core.
      contextOutput = await resources.context.run(contextFeeds);
      coreFeeds.text_features = halfTensor(contextOutput.text_features, count, 1024, 'text_features');
      coreFeeds.base_logits = halfTensor(contextOutput.base_logits, count, 7, 'base_logits');
      coreFeeds.word_mask = contextFeeds.word_mask;
      coreFeeds.local_audio = new ort.Tensor('float16', audio.data.subarray(window.left * LOCAL_FRAMES * AUDIO_DIM, window.right * LOCAL_FRAMES * AUDIO_DIM), [1, count, LOCAL_FRAMES, AUDIO_DIM]);
      coreFeeds.local_audio_mask = new ort.Tensor('bool', audio.mask.subarray(window.left * LOCAL_FRAMES, window.right * LOCAL_FRAMES), [1, count, LOCAL_FRAMES]);
      coreFeeds.audio_valid = new ort.Tensor('bool', audio.valid.subarray(window.left, window.right), [1, count]);
      const predicted = await refineLabels(new Uint8Array(count).fill(1), async (noisy, level) => {
        const noisyTensor = new ort.Tensor('int32', Int32Array.from(noisy, Number), [1, count]), levelTensor = new ort.Tensor('float16', Uint16Array.of(float32ToFloat16(level)), [1]);
        let output: Record<string, ort.Tensor> = {};
        try {
          output = await resources.denoise.run({ ...coreFeeds, noisy_labels: noisyTensor, noise_level: levelTensor });
          const logits = halfTensor(output.logits, count, 7, 'C-denoise logits');
          return Float32Array.from(logits.data as ArrayLike<number>, (value) => logits.data instanceof Uint16Array ? float16ToFloat32(value) : Number(value));
        } finally { noisyTensor.dispose(); levelTensor.dispose(); for (const tensor of Object.values(output)) tensor.dispose(); }
      });
      labels.set(predicted.subarray(window.coreLeft - window.left, window.coreRight - window.left), window.coreLeft);
    } finally {
      for (const tensor of Object.values(contextOutput)) tensor.dispose();
      for (const tensor of Object.values(contextFeeds)) tensor.dispose();
      for (const key of ['local_audio', 'local_audio_mask', 'audio_valid']) coreFeeds[key]?.dispose();
    }
  }
  return labels;
}
export const __cDenoiseRuntimeTesting = { buildRows, buildWindows, buildSegmentWindows, encodeWindow, refineLabels, tokenizeWords, validateCDenoiseConfig };
