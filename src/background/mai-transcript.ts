import { MaiError } from '../core/mai-protocol';
import type { L0DraftRow, L0TimingTrack } from '../core/types';
import type { MaiPcmAudio } from './mai-audio';

export interface MaiNativeWord { word: string; start: number; end: number }
export interface MaiNativeResponse {
  text: string;
  words?: MaiNativeWord[];
  segments?: Array<{ text: string; start: number; end: number }>;
}
export interface MaiCachedWord {
  text: string;
  nativeText: string;
  startSeconds: number;
  endSeconds: number;
}
export interface MaiLaneResult {
  track: L0TimingTrack;
  rows: L0DraftRow[];
  words: MaiCachedWord[];
  nativeText: string;
  durationSeconds: number;
  timingRepairs: number;
  chunkBoundarySeparators: number;
}
export const MAI_MAX_ROW_SECONDS = 12;
export const MAI_MAX_ROW_WORDS = 40;
const LEXICAL = /[\p{L}\p{M}\p{N}]+(?:['’][\p{L}\p{M}\p{N}]+)*/gu;

export function parseMaiNativeResponse(value: unknown): MaiNativeResponse {
  if (!value || typeof value !== 'object' || Array.isArray(value) || !('text' in value) || typeof value.text !== 'string') {
    throw new MaiError('invalid-provider-response', 'MAI returned no native transcript text.');
  }
  const candidate = value as MaiNativeResponse;
  if (candidate.text.length > 2_000_000 || (candidate.words !== undefined && (!Array.isArray(candidate.words) ||
      candidate.words.length > 200_000 || !candidate.words.every((word) => word && typeof word === 'object' &&
      typeof word.word === 'string' && word.word.length <= 4096 && typeof word.start === 'number' &&
      Number.isFinite(word.start) && typeof word.end === 'number' && Number.isFinite(word.end))))) {
    throw new MaiError('invalid-provider-response', 'MAI returned invalid word timestamp metadata.');
  }
  return { text: candidate.text, ...(candidate.words === undefined ? {} : { words: candidate.words.map(({ word, start, end }) => ({ word, start, end })) }) };
}

export function adaptMaiChunk(
  response: MaiNativeResponse,
  audio: Pick<MaiPcmAudio, 'sampleRate'>,
  startSample: number,
  endSample: number
): { words: MaiCachedWord[]; timingRepairs: number; nativeText: string } {
  const nativeLexemes = Array.from(response.text.matchAll(LEXICAL));
  if (!nativeLexemes.length) {
    if (response.text.trim() || response.words?.some((word) => Array.from(word.word.matchAll(LEXICAL)).length)) {
      throw new MaiError('word-timestamps-unavailable', 'MAI returned text that cannot be aligned to lexical word timestamps. Native output is retained; no automatic paid retry.');
    }
    return { words: [], timingRepairs: 0, nativeText: response.text };
  }
  if (!response.words?.length) {
    throw new MaiError('word-timestamps-unavailable', 'MAI returned speech without word timestamps. Native output is retained; no automatic paid retry.');
  }
  const lexicalWords = response.words.flatMap((word) => {
    const pieces = Array.from(word.word.matchAll(LEXICAL));
    return pieces.map((piece, index) => ({
      text: piece[0],
      start: word.start + (word.end - word.start) * index / pieces.length,
      end: word.start + (word.end - word.start) * (index + 1) / pieces.length
    }));
  });
  if (lexicalWords.length !== nativeLexemes.length || lexicalWords.some((word, index) =>
      word.text.toLocaleLowerCase() !== nativeLexemes[index][0].toLocaleLowerCase())) {
    throw new MaiError('word-timestamps-unavailable', 'MAI native text and word timestamps do not align. Native output is retained rather than inventing or dropping words.');
  }
  const frames = endSample - startSample;
  if (lexicalWords.length > frames) throw new MaiError('invalid-provider-response', 'MAI returned more words than available PCM sample positions.');
  let timingRepairs = 0;
  const starts: number[] = [];
  const ends: number[] = [];
  for (let index = 0; index < lexicalWords.length; index += 1) {
    const word = lexicalWords[index];
    const rawStart = Math.round(word.start * audio.sampleRate);
    const rawEnd = Math.round(word.end * audio.sampleRate);
    // Reserve one actual sample for each remaining word. Repair reversals in
    // recognition order, never by sorting/dropping the native recognized text.
    const boundedStart = Math.max(index === 0 ? 0 : starts[index - 1] + 1,
      Math.min(frames - (lexicalWords.length - index), Math.max(0, rawStart)));
    starts.push(boundedStart);
    // MAI occasionally assigns a single acknowledgement a many-second span.
    // Bound that metadata deterministically; retain the recognized word once.
    ends.push(Math.min(frames, boundedStart + MAI_MAX_ROW_SECONDS * audio.sampleRate - 1, Math.max(boundedStart + 1, rawEnd)));
    if (boundedStart !== rawStart || ends[index] !== rawEnd) timingRepairs += 1;
  }
  const boundaries = [0];
  for (let index = 1; index < nativeLexemes.length; index += 1) {
    const previousEnd = nativeLexemes[index - 1].index! + nativeLexemes[index - 1][0].length;
    const nextStart = nativeLexemes[index].index!;
    const between = response.text.slice(previousEnd, nextStart);
    const whitespace = Array.from(between.matchAll(/\s+/gu));
    const last = whitespace[whitespace.length - 1];
    // Keep trailing punctuation on the old word, and opening quotes/hyphens
    // after whitespace on the next. Every character is owned exactly once.
    boundaries.push(last ? previousEnd + last.index! + last[0].length : nextStart);
  }
  boundaries.push(response.text.length);
  const words = nativeLexemes.map((lexeme, index): MaiCachedWord => {
    const end = Math.max(starts[index] + 1, Math.min(ends[index], starts[index + 1] ?? frames));
    if (end !== ends[index]) timingRepairs += 1;
    return {
      text: lexeme[0],
      nativeText: response.text.slice(boundaries[index], boundaries[index + 1]),
      startSeconds: (startSample + starts[index]) / audio.sampleRate,
      endSeconds: (startSample + end) / audio.sampleRate
    };
  });
  return { words, timingRepairs, nativeText: response.text };
}
export async function buildMaiLaneResult(
  taskId: string,
  lane: string,
  audio: MaiPcmAudio,
  chunks: Array<{ response: MaiNativeResponse; startSample: number; endSample: number }>
): Promise<MaiLaneResult> {
  const words: MaiCachedWord[] = [];
  let nativeText = '';
  let timingRepairs = 0;
  let chunkBoundarySeparators = 0;
  for (const chunk of chunks) {
    const adapted = adaptMaiChunk(chunk.response, audio, chunk.startSample, chunk.endSample);
    const separator = nativeText.length > 0 && adapted.nativeText.length > 0 &&
      !/\s$/u.test(nativeText) && !/^\s/u.test(adapted.nativeText) ? '\n' : '';
    if (separator && adapted.words.length) {
      adapted.words[0].nativeText = separator + adapted.words[0].nativeText;
      chunkBoundarySeparators += 1;
    }
    words.push(...adapted.words);
    nativeText += separator + adapted.nativeText;
    timingRepairs += adapted.timingRepairs;
  }
  const rows: L0DraftRow[] = [];
  for (let first = 0; first < words.length;) {
    let last = first + 1;
    while (last < words.length && last - first < MAI_MAX_ROW_WORDS &&
        words[last].endSeconds - words[first].startSeconds <= MAI_MAX_ROW_SECONDS &&
        words[last].startSeconds - words[last - 1].endSeconds <= 1.5) {
      last += 1;
    }
    // Prefer a native sentence boundary within the last half of a bounded row.
    if (last < words.length) {
      for (let candidate = last - 1; candidate > first + Math.floor((last - first) / 2); candidate -= 1) {
        if (/[.!?…][\p{Pe}\p{Pf}"']*\s*$/u.test(words[candidate - 1].nativeText)) { last = candidate; break; }
      }
    }
    const startSeconds = words[first].startSeconds;
    const endSeconds = words[last - 1].endSeconds;
    const startSample = Math.round(startSeconds * audio.sampleRate);
    const endSample = Math.round(endSeconds * audio.sampleRate);
    const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(JSON.stringify([taskId, lane, audio.pcmSha256, startSample, endSample])));
    const id = `mai:${Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('')}`;
    rows.push({ id, lane, startSeconds, endSeconds, text: words.slice(first, last).map((word) => word.nativeText).join('') });
    first = last;
  }
  return {
    track: {
      lane, pcmSha256: audio.pcmSha256, sampleRate: audio.sampleRate,
      tokens: words.map(({ text, startSeconds, endSeconds }, index) => ({ id: `${taskId}:${lane}:${index}`, text, startSeconds, endSeconds })),
      segments: rows.map((row) => ({ id: row.id, startSeconds: row.startSeconds, endSeconds: row.endSeconds,
        startSample: Math.round(row.startSeconds * audio.sampleRate), endSample: Math.round(row.endSeconds * audio.sampleRate), sampleRate: audio.sampleRate }))
    },
    rows, words, nativeText, durationSeconds: audio.durationSeconds, timingRepairs, chunkBoundarySeparators
  };
}
export function clipMaiNativeText(result: MaiLaneResult, startSeconds: number, endSeconds: number): string {
  if (!Number.isFinite(startSeconds) || !Number.isFinite(endSeconds) || startSeconds < 0 || endSeconds <= startSeconds ||
      startSeconds >= result.durationSeconds) throw new MaiError('invalid-request', 'Segment interval is outside the source audio.');
  const end = Math.min(endSeconds, result.durationSeconds);
  return result.words.filter((word) => {
    const midpoint = (word.startSeconds + word.endSeconds) / 2;
    return midpoint >= startSeconds && midpoint < end;
  }).map((word) => word.nativeText).join('');
}
