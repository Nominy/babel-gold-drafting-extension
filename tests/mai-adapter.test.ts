import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createMaiBroker } from '../src/background/mai-broker';
import { parseMaiPcmWav, splitMaiAudio, MAI_PROVIDER_AUDIO_BYTES } from '../src/background/mai-audio';
import { adaptMaiChunk, buildMaiLaneResult, clipMaiNativeText, MAI_MAX_ROW_SECONDS, MAI_MAX_ROW_WORDS } from '../src/background/mai-transcript';
import { parseMaiRedistributionReview } from '../src/background/mai-redistribution';
import type { MaiStore } from '../src/background/mai-store';
import { createMaiClient, MaiBridgeError } from '../src/core/mai-client';
import { MAI_MESSAGE_TYPE, MAI_PROTOCOL_VERSION, isMaiRequest, type MaiRequest } from '../src/core/mai-protocol';
import { DEFAULT_SETTINGS } from '../src/core/settings';
import { buildCanonicalTaskIdentity } from '../src/core/transcript';
import type { CapturedAudioTrack, TranscriptJob } from '../src/core/types';

const settings = { ...DEFAULT_SETTINGS, mode: 'simple' as const, openRouterApiKey: 'sk-or-test-secret' };
const sender = { id: 'gold-extension', tab: { id: 7 }, frameId: 0, documentId: 'document-1' } as chrome.runtime.MessageSender;
const job: TranscriptJob = { jobId: 'task-native', rows: [
  { rowId: 'old-left', speakerKey: 'Speaker 1', startSeconds: 0, endSeconds: 2, text: 'private existing transcript', index: 0 },
  { rowId: 'old-right', speakerKey: 'Speaker 2', startSeconds: 0, endSeconds: 2, text: 'private existing transcript', index: 1 }
] };
function wav(seconds = 4, sampleRate = 16000, seed = 1): Uint8Array<ArrayBuffer> {
  const frames = Math.round(seconds * sampleRate);
  const bytes = new Uint8Array(44 + frames * 2);
  const view = new DataView(bytes.buffer);
  for (const [offset, text] of [[0, 'RIFF'], [8, 'WAVE'], [12, 'fmt '], [36, 'data']] as const) {
    for (let index = 0; index < text.length; index += 1) bytes[offset + index] = text.charCodeAt(index);
  }
  view.setUint32(4, bytes.length - 8, true); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true); view.setUint32(40, frames * 2, true);
  for (let frame = 0; frame < frames; frame += 1) view.setInt16(44 + frame * 2, ((frame * 31 + seed) % 20000) - 10000, true);
  return bytes;
}
function tracks(left = wav(), right = wav(4, 16000, 2)): CapturedAudioTrack[] {
  return [left, right].map((bytes, index) => ({ trackId: `source-${index}`, speakerKey: `Speaker ${index + 1}`,
    source: 'private-source.wav', mimeType: 'audio/wav', blob: new Blob([bytes], { type: 'audio/wav' }) }));
}
function memoryStore(): MaiStore {
  const values = new Map<string, unknown>();
  return { async get<T>(key: string) { return structuredClone(values.get(key)) as T | undefined; },
    async set<T>(key: string, value: T) { values.set(key, structuredClone(value)); } };
}
function harness(provider: (body: FormData | string, count: number) => Promise<Response> | Response, store = memoryStore()) {
  let calls = 0;
  const messages: MaiRequest[] = [];
  const broker = createMaiBroker({ extensionId: 'gold-extension', loadSettings: async () => settings, store,
    fetch: (async (_url: unknown, init: RequestInit) => { calls += 1; return provider(init.body as FormData | string, calls); }) as typeof fetch });
  const client = createMaiClient(async (message) => { messages.push(message); return broker.handle(message, sender); });
  return { broker, client, messages, calls: () => calls, store };
}
const native = { text: '«Ну-ну», я... я сказал: "Да!"', words: [
  { word: 'Ну-ну', start: 0.2, end: 0.6, speaker: 88 },
  { word: 'я...', start: 0.7, end: 0.8, speaker: 1 },
  { word: 'я', start: 1, end: 1.1, speaker: 2 },
  { word: 'сказал:', start: 1.2, end: 1.6, speaker: 9 },
  { word: 'Да!', start: 1.8, end: 2, speaker: 3 }
] };

 test('native punctuation, capitalization, partial/repeated words survive timing, draft and lane-isolated clipping', async () => {
  const h = harness((_body, call) => Response.json(call === 1 ? native : { text: 'А-а, нет.', words: [
    { word: 'А-а,', start: 0.2, end: 0.6, speaker: 88 }, { word: 'нет.', start: 0.8, end: 1, speaker: 88 }
  ] }));
  const taskId = buildCanonicalTaskIdentity(job);
  assert.equal(await h.client.lookupMaiL0Timing(settings, taskId), null);
  assert.equal(h.calls(), 0);
  await assert.rejects(h.client.generateMaiL0Draft(settings, job), (error) => error instanceof MaiBridgeError && error.code === 'timing-unavailable');
  const timing = await h.client.generateMaiL0Timing(settings, job, tracks());
  assert.deepEqual(timing.tracks.map((track) => track.lane), ['Speaker 1', 'Speaker 2']);
  assert.deepEqual(timing.tracks[0].tokens.map((word) => word.text), ['Ну', 'ну', 'я', 'я', 'сказал', 'Да']);
  const draft = await h.client.generateMaiL0Draft(settings, job);
  assert.equal(draft.rows.filter((row) => row.lane === 'Speaker 1').map((row) => row.text).join(''), native.text);
  assert.equal(await h.client.generateMaiL0SegmentDraft(settings, taskId, { ...job.rows[0], startSeconds: 0.65, endSeconds: 1.15 }), 'я... я ');
  assert.equal(await h.client.generateMaiL0SegmentDraft(settings, taskId, { ...job.rows[1], startSeconds: 0.65, endSeconds: 1.15 }), 'нет.');
  assert.equal(h.calls(), 2);
  const payloads = JSON.stringify(h.messages);
  assert.equal(payloads.includes(settings.openRouterApiKey), false);
  assert.equal(payloads.includes('private existing transcript'), false);
  assert.equal(payloads.includes('private-source.wav'), false);
});

test('six backwards starts are repaired in recognition order with positive real sample bounds', async () => {
  const audio = await parseMaiPcmWav(wav(2));
  const text = 'Раз два три четыре пять шесть семь.';
  const result = adaptMaiChunk({ text, words: text.replace('.', '').split(' ').map((word, index) => ({ word,
    start: 1.8 - index * 0.2, end: index === 6 ? 50 : 1.85 - index * 0.2 })) }, audio, 0, audio.frameCount);
  assert.equal(result.words.map((word) => word.nativeText).join(''), text);
  assert.deepEqual(result.words.map((word) => word.text), text.replace('.', '').split(' '));
  for (let index = 0; index < result.words.length; index += 1) {
    const word = result.words[index];
    assert.ok(word.startSeconds >= 0 && word.endSeconds > word.startSeconds && word.endSeconds <= 2);
    if (index > 0) assert.ok(word.startSeconds > result.words[index - 1].startSeconds);
  }
  assert.ok(result.timingRepairs >= 6);
});

test('several-minute provider segments become bounded rows without losing quotes, hyphens or whitespace', async () => {
  const audio = await parseMaiPcmWav(wav(180));
  const words = Array.from({ length: 160 }, (_, index) => ({ word: `слово${index}`, start: index, end: index + 0.5 }));
  const text = `«${words.map((word) => word.word).join(' - ')}».`;
  const result = await buildMaiLaneResult('task', 'Known source', audio, [{ response: { text, words,
    segments: [{ text, start: 0, end: 180 }] }, startSample: 0, endSample: audio.frameCount }]);
  assert.equal(result.rows.map((row) => row.text).join(''), text);
  for (const row of result.rows) {
    assert.ok(row.endSeconds - row.startSeconds <= MAI_MAX_ROW_SECONDS);
    assert.ok(row.text.match(/слово\d+/g)!.length <= MAI_MAX_ROW_WORDS);
    assert.equal(row.lane, 'Known source');
  }
  const a = clipMaiNativeText(result, 0, 12);
  const b = clipMaiNativeText(result, 12, 500);
  assert.equal(a + b, text);
  assert.throws(() => clipMaiNativeText(result, 180, 181), /outside/);
});

test('implausibly long native word metadata is bounded without rewriting or duplicating its text', async () => {
  const audio = await parseMaiPcmWav(wav(100));
  const result = await buildMaiLaneResult('task', 'Speaker 2', audio, [{ response: {
    text: 'Мгм.\n', words: [{ word: 'Мгм.', start: 59.76, end: 82.36 }]
  }, startSample: 0, endSample: audio.frameCount }]);
  assert.equal(result.rows[0].text, 'Мгм.\n');
  assert.equal(result.track.tokens[0].text, 'Мгм');
  assert.equal(result.rows[0].startSeconds, 59.76);
  assert.ok(result.rows[0].endSeconds - result.rows[0].startSeconds <= MAI_MAX_ROW_SECONDS);
  assert.ok(result.timingRepairs > 0);
});

test('long WAV chunks own every real PCM frame once, remain below provider cap, and retain full source SHA', async () => {
  const bytes = wav(1700, 8000);
  assert.ok(bytes.length > 25 * 1024 * 1024);
  const audio = await parseMaiPcmWav(bytes);
  assert.equal(audio.pcmSha256, createHash('sha256').update(bytes.subarray(44)).digest('hex'));
  const chunks = splitMaiAudio(audio);
  let nextSample = 0;
  const reconstructed = createHash('sha256');
  for (const chunk of chunks) {
    assert.equal(chunk.startSample, nextSample);
    assert.ok(chunk.wav.size < MAI_PROVIDER_AUDIO_BYTES);
    const actual = await parseMaiPcmWav(new Uint8Array(await chunk.wav.arrayBuffer()));
    assert.equal(actual.frameCount, chunk.endSample - chunk.startSample);
    reconstructed.update(actual.pcm);
    nextSample = chunk.endSample;
  }
  assert.equal(nextSample, audio.frameCount);
  assert.equal(reconstructed.digest('hex'), audio.pcmSha256);
});

test('audio beyond the provider inline cap transcribes with truthful offsets and cache-only native draft', async () => {
  const longBytes = wav(1700, 8000);
  const audio = await parseMaiPcmWav(longBytes);
  const ownership = splitMaiAudio(audio);
  const h = harness(async (body, call) => {
    assert.ok(body instanceof FormData);
    const realChunk = await parseMaiPcmWav(new Uint8Array(await (body.get('file') as Blob).arrayBuffer()));
    assert.ok(realChunk.bytes.length < 25 * 1024 * 1024);
    return Response.json({ text: `слово${call}`, words: [{ word: `слово${call}`, start: 0.1, end: 0.3 }] });
  });
  const timing = await h.client.generateMaiL0Timing(settings, job, tracks(longBytes));
  assert.deepEqual(timing.tracks[0].tokens.map((word) => word.text), ownership.map((_chunk, index) => `слово${index + 1}`));
  for (let index = 0; index < ownership.length; index += 1) {
    assert.equal(timing.tracks[0].tokens[index].startSeconds, (ownership[index].startSample + 800) / 8000);
  }
  const draft = await h.client.generateMaiL0Draft(settings, job);
  assert.equal(draft.rows.filter((row) => row.lane === 'Speaker 1').map((row) => row.text).join(''),
    ownership.map((_chunk, index) => `слово${index + 1}`).join('\n'));
  assert.equal(h.calls(), ownership.length + 1);
});

test('chunk boundaries declare a separator owned by the next native word without changing provider fragments', async () => {
  const audio = await parseMaiPcmWav(wav(8));
  const chunks = [
    { response: { text: '«слово»', words: [{ word: 'слово', start: 0.2, end: 0.4 }] }, startSample: 0, endSample: 32000 },
    { response: { text: 'дальше-тут', words: [{ word: 'дальше-тут', start: 0.2, end: 0.4 }] }, startSample: 32000, endSample: 64000 },
    { response: { text: ' конец.', words: [{ word: 'конец.', start: 0.2, end: 0.4 }] }, startSample: 64000, endSample: 128000 }
  ];
  const result = await buildMaiLaneResult('task', 'Speaker 1', audio, chunks);
  assert.equal(result.nativeText, '«слово»\nдальше-тут конец.');
  assert.equal(result.rows.map((row) => row.text).join(''), result.nativeText);
  assert.equal(clipMaiNativeText(result, 0, 2), '«слово»');
  assert.equal(clipMaiNativeText(result, 2, 4), '\nдальше-тут');
  assert.equal(result.chunkBoundarySeparators, 1);
});

test('parallel task requests share paid calls and durable cache survives broker recreation', async () => {
  const h = harness(() => Response.json(native));
  const [one, two] = await Promise.all([h.client.generateMaiL0Timing(settings, job, tracks()), h.client.generateMaiL0Timing(settings, job, tracks())]);
  assert.deepEqual(one, two);
  assert.equal(h.calls(), 2);
  const reloaded = harness(() => { throw new Error('must not pay for cache'); }, h.store);
  assert.deepEqual(await reloaded.client.lookupMaiL0Timing(settings, buildCanonicalTaskIdentity(job)), one);
  assert.deepEqual(await reloaded.client.generateMaiL0Draft(settings, job), await h.client.generateMaiL0Draft(settings, job));
  await reloaded.client.generateMaiL0Timing(settings, job, tracks());
  assert.equal(reloaded.calls(), 0);
});

test('explicit retry after partial provider failure never repeats a completed source', async () => {
  const h = harness((_body, call) => call === 2 ? Response.json({ error: { message: settings.openRouterApiKey } }, { status: 429 }) : Response.json(native));
  await assert.rejects(h.client.generateMaiL0Timing(settings, job, tracks()), (error) => error instanceof MaiBridgeError && error.code === 'rate-limited' && !error.message.includes(settings.openRouterApiKey));
  assert.equal(h.calls(), 2);
  const restarted = harness(() => Response.json(native), h.store);
  const result = await restarted.client.generateMaiL0Timing(settings, job, tracks());
  assert.equal(result.tracks.length, 2);
  assert.equal(restarted.calls(), 1);
});

test('interrupted paid result warns once then permits explicit retry while reusing the completed source', async () => {
  const durable = memoryStore();
  let completedWrites = 0;
  const interruptedStore: MaiStore = {
    get: durable.get,
    async set<T>(key: string, value: T) {
      if (value && typeof value === 'object' && 'status' in value && value.status === 'completed') {
        completedWrites += 1;
        if (completedWrites === 2) throw new Error('Worker stopped before retaining the paid second-source result.');
      }
      await durable.set(key, value);
    }
  };
  const interrupted = harness(() => Response.json(native), interruptedStore);
  await assert.rejects(interrupted.client.generateMaiL0Timing(settings, job, tracks()));
  assert.equal(interrupted.calls(), 2);
  const restarted = harness(() => Response.json(native), durable);
  await assert.rejects(restarted.client.generateMaiL0Timing(settings, job, tracks()), (error) =>
    error instanceof MaiBridgeError && error.code === 'transcription-interrupted' &&
    error.message.includes('may repeat that charge') && error.message.includes('Retry again'));
  assert.equal(restarted.calls(), 0);
  const recovered = await restarted.client.generateMaiL0Timing(settings, job, tracks());
  assert.deepEqual(recovered.tracks.map((track) => track.lane), ['Speaker 1', 'Speaker 2']);
  assert.equal(restarted.calls(), 1);
});

test('silent lanes stay empty; missing word timestamps are visible and paid success is not repeated', async () => {
  const silent = harness((_body, call) => Response.json(call === 1 ? { text: '', words: [] } : native));
  const result = await silent.client.generateMaiL0Timing(settings, job, tracks());
  assert.deepEqual(result.tracks[0].tokens, []);
  assert.deepEqual(result.tracks[0].segments, []);
  assert.equal((await silent.client.generateMaiL0Draft(settings, job)).rows.some((row) => row.lane === 'Speaker 1'), false);
  const missing = harness(() => Response.json({ text: 'Распознано.', segments: [{ text: 'Распознано.', start: 0, end: 4 }] }));
  for (let attempt = 0; attempt < 2; attempt += 1) {
    await assert.rejects(missing.client.generateMaiL0Timing(settings, job, tracks()), (error) => error instanceof MaiBridgeError && error.code === 'word-timestamps-unavailable');
  }
  assert.equal(missing.calls(), 1);
});

test('request admission rejects invalid bytes, secret-bearing payloads, outside senders and stolen transfers before payment', async () => {
  const h = harness(() => Response.json(native));
  const lookup = { type: MAI_MESSAGE_TYPE, version: MAI_PROTOCOL_VERSION, requestId: 'r', operation: 'lookup', taskId: 'task' } as const;
  assert.equal(isMaiRequest({ ...lookup, settings }), false);
  const outside = await h.broker.handle(lookup, { id: 'other-extension' });
  assert.equal(outside.ok, false);
  await assert.rejects(h.client.generateMaiL0Timing(settings, job, tracks(new Uint8Array(100))), /PCM16/);
  const capture = harness(() => Response.json(native));
  const client = createMaiClient(async (message) => {
    if (message.operation === 'timing') return capture.broker.handle(message, { ...sender, tab: { id: 8 } } as chrome.runtime.MessageSender);
    return capture.broker.handle(message, sender);
  });
  await assert.rejects(client.generateMaiL0Timing(settings, job, tracks()), /owned by this sender/);
  assert.equal(h.calls(), 0);
  assert.equal(capture.calls(), 0);
});

test('401 and 402 surface actionable sanitized failures; malformed saved keys never reach provider', async () => {
  for (const status of [401, 402]) {
    const h = harness(() => Response.json({ error: { message: `Bearer ${settings.openRouterApiKey}` } }, { status }));
    await assert.rejects(h.client.generateMaiL0Timing(settings, job, tracks()), (error) => error instanceof MaiBridgeError &&
      error.code === (status === 401 ? 'invalid-key' : 'insufficient-credit') && !error.message.includes(settings.openRouterApiKey));
    assert.equal(h.calls(), 1);
  }
  let paid = false;
  const broker = createMaiBroker({ extensionId: 'gold-extension', loadSettings: async () => ({ ...settings, openRouterApiKey: 'bad\nkey' }),
    store: memoryStore(), fetch: (async () => { paid = true; return Response.json(native); }) as typeof fetch });
  const client = createMaiClient((message) => broker.handle(message, sender));
  await assert.rejects(client.generateMaiL0Timing(settings, job, tracks()), (error) => error instanceof MaiBridgeError && error.code === 'invalid-key');
  assert.equal(paid, false);
});

test('redistribution reviews only adjacent whole-sentence moves and never returns rewritten allocations', () => {
  assert.deepEqual(parseMaiRedistributionReview('```json\n{"acceptDraft":false,"moves":[{"fromIndex":1,"toIndex":2,"sentenceCount":1}],"notes":"native"}\n```', 2),
    { acceptDraft: false, moves: [{ fromIndex: 1, toIndex: 2, sentenceCount: 1 }], notes: 'native' });
  assert.throws(() => parseMaiRedistributionReview('{"moves":[{"fromIndex":1,"toIndex":3,"sentenceCount":1}]}', 3), /adjacent/);
  assert.throws(() => parseMaiRedistributionReview('{"moves":[{"fromIndex":2,"toIndex":3,"sentenceCount":1}]}', 2), /adjacent/);
});
