import { loadSettings } from '../core/settings';
import { decodeAudioChunk, LOCAL_MODEL_AUDIO_TRANSFER_STALE_MS } from '../core/local-model-offscreen-protocol';
import { MAI_MAX_AUDIO_BYTES, MAI_MESSAGE_TYPE, MAI_PROTOCOL_VERSION, MaiError, isMaiRequest,
  type MaiRequest, type MaiResponse, type MaiTimingRequest, type MaiUploadRequest, type MaiEnvelope } from '../core/mai-protocol';
import type { ExtensionSettings, L0DraftResponse, L0TimingResponse } from '../core/types';
import { parseMaiPcmWav, splitMaiAudio, type MaiPcmAudio } from './mai-audio';
import { buildMaiLaneResult, clipMaiNativeText, parseMaiNativeResponse, type MaiLaneResult } from './mai-transcript';
import { reviewMaiRedistributions, type MaiAuthorizedFetch } from './mai-redistribution';
import { createMaiResultStore, type MaiStore } from './mai-store';

export const MAI_TRANSCRIPTION_MODEL = 'microsoft/mai-transcribe-2';
interface MaiTransfer {
  owner: string;
  totalBytes: number;
  chunkCount: number;
  nextChunkIndex: number;
  bytes: Uint8Array<ArrayBuffer>;
  updatedAt: number;
}
interface MaiTaskResult {
  timing: L0TimingResponse;
  draft: L0DraftResponse;
  lanes: MaiLaneResult[];
}
type MaiChunkRecord = { status: 'completed'; payload: unknown } | { status: 'pending' } | { status: 'failed' };
export interface MaiBrokerDependencies {
  extensionId: string;
  loadSettings: () => Promise<ExtensionSettings>;
  fetch: typeof fetch;
  store: MaiStore;
  now?: () => number;
}

export function createMaiBroker(deps: MaiBrokerDependencies) {
  const transfers = new Map<string, MaiTransfer>();
  const laneFlights = new Map<string, Promise<MaiLaneResult>>();
  const taskFlights = new Map<string, Promise<MaiTaskResult>>();
  const now = deps.now ?? Date.now;
  function ownerFor(sender: chrome.runtime.MessageSender): string {
    if (!deps.extensionId || sender.id !== deps.extensionId) throw new MaiError('invalid-sender', 'MAI requests must come from this extension.');
    return JSON.stringify([sender.id, sender.tab?.id ?? null, sender.frameId ?? null, sender.documentId ?? sender.url ?? null]);
  }
  function upload(request: MaiUploadRequest, owner: string) {
    for (const [id, transfer] of transfers) {
      if (now() - transfer.updatedAt > LOCAL_MODEL_AUDIO_TRANSFER_STALE_MS) transfers.delete(id);
    }
    let transfer = transfers.get(request.transferId);
    if (!transfer) {
      if (request.chunkIndex !== 0) throw new MaiError('invalid-request', 'Audio transfer must start with chunk zero.');
      const buffered = Array.from(transfers.values()).reduce((sum, value) => sum + value.totalBytes, 0);
      if (buffered + request.totalBytes > MAI_MAX_AUDIO_BYTES * 2) throw new MaiError('invalid-request', 'MAI audio transfer capacity is exceeded.');
      transfer = { owner, totalBytes: request.totalBytes, chunkCount: request.chunkCount, nextChunkIndex: 0,
        bytes: new Uint8Array(request.totalBytes), updatedAt: now() };
      transfers.set(request.transferId, transfer);
    }
    if (transfer.owner !== owner || transfer.totalBytes !== request.totalBytes || transfer.chunkCount !== request.chunkCount ||
        transfer.nextChunkIndex !== request.chunkIndex) throw new MaiError('invalid-request', 'Audio transfer ownership or chunk order does not match.');
    const bytes = decodeAudioChunk(request.dataBase64);
    const offset = request.chunkIndex * 512 * 1024;
    transfer.bytes.set(bytes, offset);
    transfer.nextChunkIndex += 1;
    transfer.updatedAt = now();
    return { transferId: request.transferId, nextChunkIndex: transfer.nextChunkIndex, complete: transfer.nextChunkIndex === transfer.chunkCount };
  }
  async function authorizedRequest(allowLocalText = false): Promise<MaiAuthorizedFetch> {
    const settings = await deps.loadSettings();
    if (settings.mode !== 'simple' && !(allowLocalText && settings.mode === 'local')) {
      throw new MaiError('wrong-mode', 'MAI audio transcription requires Simple mode; Local permits only explicitly requested cloud text alignment.');
    }
    const key = settings.openRouterApiKey.trim();
    if (!key) throw new MaiError('missing-key', 'Set an OpenRouter API key before explicitly requesting cloud transcription or text alignment.');
    if (key.length > 1024 || /[\s\u0000-\u001f\u007f]/u.test(key)) throw new MaiError('invalid-key', 'The saved OpenRouter API key is invalid.');
    return async (path, body, contentType) => {
      let response: Response;
      try {
        response = await deps.fetch(`https://openrouter.ai/api/v1/${path}`, {
          method: 'POST', headers: { Authorization: `Bearer ${key}`, Accept: 'application/json', ...(contentType ? { 'Content-Type': contentType } : {}) },
          body, signal: AbortSignal.timeout(path === 'audio/transcriptions' ? 240_000 : 120_000), redirect: 'error'
        });
      } catch {
        throw new MaiError('provider-failed', 'OpenRouter request failed or timed out. No automatic paid retry was attempted.');
      }
      if (!response.ok) {
        const details: Record<number, [string, string]> = {
          401: ['invalid-key', 'OpenRouter rejected the saved API key (HTTP 401).'],
          402: ['insufficient-credit', 'OpenRouter credits are insufficient (HTTP 402).'],
          429: ['rate-limited', 'OpenRouter rate limit reached (HTTP 429). Retry only when you choose to.']
        };
        const [code, message] = details[response.status] ?? ['provider-failed', `OpenRouter provider request failed (HTTP ${response.status}). No automatic paid retry was attempted.`];
        // Provider error bodies can echo credentials, uploaded data, or prompt text.
        // Never forward those bodies to a content script or the page.
        throw new MaiError(code, message);
      }
      try {
        const payload: unknown = await response.json();
        if (payload && typeof payload === 'object' && 'error' in payload) throw new Error('provider error envelope');
        return payload;
      } catch { throw new MaiError('invalid-provider-response', 'OpenRouter returned an invalid response. No automatic paid retry was attempted.'); }
    };
  }
  async function transcribeLane(taskId: string, lane: string, trackId: string, audio: MaiPcmAudio, request: MaiAuthorizedFetch): Promise<MaiLaneResult> {
    const laneKey = JSON.stringify(['mai-lane-v1', taskId, lane, trackId, audio.pcmSha256, audio.sampleRate, audio.channels]);
    const existingFlight = laneFlights.get(laneKey);
    if (existingFlight) return existingFlight;
    const flight = (async () => {
      const cached = await deps.store.get<MaiLaneResult>(laneKey);
      if (cached) return cached;
      const chunks = [];
      for (const chunk of splitMaiAudio(audio)) {
        const chunkKey = JSON.stringify([laneKey, chunk.startSample, chunk.endSample]);
        let record = await deps.store.get<MaiChunkRecord>(chunkKey);
        if (record?.status === 'pending') {
          // This call is an explicit action, but first warn about the uncertain
          // prior charge. A subsequent explicit Retry can proceed without any
          // opaque clear-cache operation; completed sources/chunks stay intact.
          await deps.store.set<MaiChunkRecord>(chunkKey, { status: 'failed' });
          throw new MaiError('transcription-interrupted', 'A previous paid transcription was interrupted before its result was retained. Its charge is uncertain. Choose Retry again to transcribe the missing audio; this may repeat that charge. Completed audio will be reused.');
        }
        if (record?.status !== 'completed') {
          await deps.store.set<MaiChunkRecord>(chunkKey, { status: 'pending' });
          try {
            const form = new FormData();
            form.set('file', chunk.wav, 'source.wav');
            form.set('model', MAI_TRANSCRIPTION_MODEL);
            form.set('language', 'ru');
            form.set('response_format', 'verbose_json');
            form.append('timestamp_granularities[]', 'segment');
            form.append('timestamp_granularities[]', 'word');
            form.set('provider', JSON.stringify({ options: { azure: {
              enhancedMode: { enabled: true, model: 'MAI-Transcribe-2', modelOptions: { transcribeStyle: 'verbatim', timestamps: 'word' } },
              diarization: { enabled: false }
            } } }));
            const payload = await request('audio/transcriptions', form);
            record = { status: 'completed', payload };
          } catch (error) {
            await deps.store.set<MaiChunkRecord>(chunkKey, { status: 'failed' });
            throw error;
          }
          // Persist paid success before adapting timings, so unusable timestamps
          // and later source failures do not trigger a second paid transcription.
          await deps.store.set(chunkKey, record);
        }
        chunks.push({ response: parseMaiNativeResponse(record.payload), startSample: chunk.startSample, endSample: chunk.endSample });
      }
      const result = await buildMaiLaneResult(taskId, lane, audio, chunks);
      await deps.store.set(laneKey, result);
      return result;
    })();
    laneFlights.set(laneKey, flight);
    try { return await flight; } finally { laneFlights.delete(laneKey); }
  }
  async function timing(request: MaiTimingRequest, owner: string): Promise<L0TimingResponse> {
    const sources = [];
    for (const source of request.sources) {
      const transfer = transfers.get(source.transferId);
      if (!transfer || transfer.owner !== owner || transfer.nextChunkIndex !== transfer.chunkCount) {
        throw new MaiError('invalid-request', 'A complete audio transfer owned by this sender is required.');
      }
      sources.push({ ...source, audio: await parseMaiPcmWav(transfer.bytes) });
    }
    const signature = JSON.stringify([request.taskId, ...sources.map((source) => [source.lane, source.trackId, source.audio.pcmSha256])]);
    const existingFlight = taskFlights.get(signature);
    if (existingFlight) {
      try { return (await existingFlight).timing; }
      finally { for (const source of sources) transfers.delete(source.transferId); }
    }
    const flight = (async (): Promise<MaiTaskResult> => {
      const paidRequest = await authorizedRequest();
      const lanes: MaiLaneResult[] = [];
      // Finish/persist each isolated lane independently before starting the next.
      // An explicit retry after lane 2 failure reuses successful lane 1/chunks.
      for (const source of sources) lanes.push(await transcribeLane(request.taskId, source.lane, source.trackId, source.audio, paidRequest));
      const summary = { provider: 'openrouter-mai', taskId: request.taskId, trackCount: lanes.length,
        tokenCount: lanes.reduce((sum, lane) => sum + lane.words.length, 0),
        timingRepairs: lanes.reduce((sum, lane) => sum + lane.timingRepairs, 0),
        chunkBoundarySeparators: lanes.reduce((sum, lane) => sum + (lane.chunkBoundarySeparators ?? 0), 0),
        nativePunctuation: true, verbatim: true, rewriting: false };
      const models = { l0: MAI_TRANSCRIPTION_MODEL, punctuation: 'native', rewriting: null };
      const result: MaiTaskResult = {
        timing: { taskId: request.taskId, tracks: lanes.map((lane) => lane.track), summary, models },
        draft: { rows: lanes.flatMap((lane) => lane.rows), summary, models }, lanes
      };
      await deps.store.set(`mai-task-v1:${request.taskId}`, result);
      return result;
    })();
    taskFlights.set(signature, flight);
    try { return (await flight).timing; }
    finally {
      taskFlights.delete(signature);
      for (const source of sources) transfers.delete(source.transferId);
    }
  }
  async function handle(message: unknown, sender: chrome.runtime.MessageSender): Promise<MaiResponse> {
    const candidate = message as Partial<MaiRequest> | null;
    const envelope: MaiEnvelope = { type: MAI_MESSAGE_TYPE, version: MAI_PROTOCOL_VERSION,
      requestId: typeof candidate?.requestId === 'string' ? candidate.requestId : '', operation: candidate?.operation ?? 'lookup' };
    try {
      const owner = ownerFor(sender);
      if (!isMaiRequest(message)) throw new MaiError('invalid-request', 'Invalid MAI request. Settings, credentials, and transcript rows are not accepted.');
      let result: unknown;
      if (message.operation === 'upload') result = upload(message, owner);
      else if (message.operation === 'timing') result = await timing(message, owner);
      else if (message.operation === 'redistribute') result = await reviewMaiRedistributions(message.groups, await authorizedRequest(true));
      else {
        const cached = await deps.store.get<MaiTaskResult>(`mai-task-v1:${message.taskId}`);
        if (message.operation === 'lookup') result = cached?.timing ?? null;
        else {
          if (!cached) throw new MaiError('timing-unavailable', 'Native MAI transcription is not cached for this task. Use the explicit transcription action first.');
          if (message.operation === 'draft') result = cached.draft;
          else if (message.operation === 'segment') {
            const lane = cached.lanes.find((candidate) => candidate.track.lane === message.lane);
            if (!lane) throw new MaiError('timing-unavailable', 'No cached native transcription exists for the requested source lane.');
            result = clipMaiNativeText(lane, message.startSeconds, message.endSeconds);
          }
        }
      }
      return { ...envelope, ok: true, result };
    } catch (error) {
      return { ...envelope, ok: false, error: error instanceof MaiError ? { code: error.code, message: error.message } :
        { code: 'cache-unavailable', message: 'MAI native result storage is unavailable. No fallback or automatic paid retry was attempted.' } };
    }
  }
  return { handle };
}
const runtime = globalThis.chrome?.runtime;
if (runtime?.onMessage && typeof runtime.onMessage.addListener === 'function') {
  const broker = createMaiBroker({ extensionId: runtime.id, loadSettings, fetch: globalThis.fetch.bind(globalThis), store: createMaiResultStore() });
  runtime.onMessage.addListener((message: unknown, sender, sendResponse) => {
    if (!message || typeof message !== 'object' || !('type' in message) || message.type !== MAI_MESSAGE_TYPE) return false;
    void broker.handle(message, sender).then(sendResponse);
    return true;
  });
}
