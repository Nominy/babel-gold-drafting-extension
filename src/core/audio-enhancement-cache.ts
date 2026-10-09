import type { AudioEnhancementProgress } from '@nominy/babel-babel-runtime';
import { readEnhancementWavHeader } from './audio-enhancement-dsp';
import { LOCAL_MODEL_AUDIO_CHUNK_BYTES } from './local-model-offscreen-protocol';
import type { EnhancedAudioBatch, EnhancedAudioTrack } from './audio-enhancement-runtime';
import type { CapturedAudioTrack } from './types';

export const AUDIO_ENHANCEMENT_CACHE_DATABASE = 'babel-gold-drafting-audio-enhancement';
export const AUDIO_ENHANCEMENT_CACHE_STORE = 'current';
export const AUDIO_ENHANCEMENT_CACHE_KEY = 'pair';
const SHA256 = /^[a-f0-9]{64}$/;

export interface EnhancementCacheLane {
  trackId: string;
  sourceSha256: string;
  sampleRate: number;
  frameCount: number;
}
export interface EnhancementCacheIdentity {
  key: string;
  modelSha256: string;
  lanes: [EnhancementCacheLane, EnhancementCacheLane];
}
export interface EnhancementCacheLease { key: string; modelSha256: string; nonce: string }
export interface CachedEnhancementLane extends EnhancementCacheLane {
  wavSha256: string;
  wav: ArrayBuffer;
}
interface PendingPair {
  schemaVersion: 1;
  state: 'pending';
  key: string;
  modelSha256: string;
  nonce: string;
}
export interface ReadyEnhancementPair extends Omit<PendingPair, 'state'> {
  state: 'ready';
  lanes: [CachedEnhancementLane, CachedEnhancementLane];
}
export interface EnhancementCacheSelection {
  lease: EnhancementCacheLease;
  ready?: ReadyEnhancementPair;
}
export interface AudioEnhancementPairCache {
  select(identity: EnhancementCacheIdentity): Promise<EnhancementCacheSelection>;
  invalidate(lease: EnhancementCacheLease): Promise<boolean>;
  commit(lease: EnhancementCacheLease, identity: EnhancementCacheIdentity, lanes: CachedEnhancementLane[], signal?: AbortSignal): Promise<boolean>;
  clear(): Promise<void>;
  close(): void;
}
export interface EnhancementPairCacheOptions {
  indexedDB?: IDBFactory;
}

function positiveInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0;
}
function isCachedLane(value: unknown): value is CachedEnhancementLane {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return false;
  const lane = value as Partial<CachedEnhancementLane>;
  return Object.keys(lane).every((key) => ['trackId', 'sourceSha256', 'sampleRate', 'frameCount', 'wavSha256', 'wav'].includes(key)) &&
    typeof lane.trackId === 'string' && lane.trackId.length > 0 &&
    typeof lane.sourceSha256 === 'string' && SHA256.test(lane.sourceSha256) &&
    positiveInteger(lane.sampleRate) && positiveInteger(lane.frameCount) &&
    typeof lane.wavSha256 === 'string' && SHA256.test(lane.wavSha256) && lane.wav instanceof ArrayBuffer &&
    lane.wav.byteLength === 44 + lane.frameCount * 2;
}
function isReadyPair(value: unknown): value is ReadyEnhancementPair {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return false;
  const pair = value as Partial<ReadyEnhancementPair>;
  return Object.keys(pair).every((key) => ['schemaVersion', 'state', 'key', 'modelSha256', 'nonce', 'lanes'].includes(key)) &&
    pair.schemaVersion === 1 && pair.state === 'ready' && typeof pair.key === 'string' && SHA256.test(pair.key) &&
    typeof pair.modelSha256 === 'string' && SHA256.test(pair.modelSha256) &&
    typeof pair.nonce === 'string' && pair.nonce.length > 0 && Array.isArray(pair.lanes) &&
    pair.lanes.length === 2 && pair.lanes.every(isCachedLane) && pair.lanes[0].trackId !== pair.lanes[1].trackId;
}
function matchesLease(value: unknown, lease: EnhancementCacheLease): value is PendingPair | ReadyEnhancementPair {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return false;
  const pair = value as Partial<PendingPair | ReadyEnhancementPair>;
  return pair.schemaVersion === 1 && (pair.state === 'pending' || pair.state === 'ready') && pair.key === lease.key &&
    pair.modelSha256 === lease.modelSha256 && pair.nonce === lease.nonce;
}
function pending(lease: EnhancementCacheLease): PendingPair {
  return { schemaVersion: 1, state: 'pending', key: lease.key, modelSha256: lease.modelSha256, nonce: lease.nonce };
}
function matchesIdentity(lanes: CachedEnhancementLane[], identity: EnhancementCacheIdentity): boolean {
  return lanes.length === 2 && new Set(lanes.map((lane) => lane.trackId)).size === 2 && identity.lanes.every((source) => {
    const lane = lanes.find((candidate) => candidate.trackId === source.trackId);
    return lane?.sourceSha256 === source.sourceSha256 && lane.sampleRate === source.sampleRate && lane.frameCount === source.frameCount;
  });
}

export async function enhancementSha256(bytes: ArrayBuffer): Promise<string> {
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), (byte) => byte.toString(16).padStart(2, '0')).join('');
}
async function validLaneBytes(lanes: CachedEnhancementLane[], identity: EnhancementCacheIdentity): Promise<boolean> {
  if (!lanes.every(isCachedLane) || !matchesIdentity(lanes, identity)) return false;
  try {
    for (const lane of lanes) {
      const header = readEnhancementWavHeader(lane.wav);
      const view = new DataView(lane.wav);
      if (header.format !== 1 || header.bits !== 16 || header.channels !== 1 || header.dataOffset !== 44 ||
        header.sampleRate !== lane.sampleRate || header.frameCount !== lane.frameCount ||
        view.getUint32(4, true) !== lane.wav.byteLength - 8 || view.getUint32(28, true) !== lane.sampleRate * 2 || view.getUint16(32, true) !== 2 ||
        await enhancementSha256(lane.wav) !== lane.wavSha256) return false;
    }
    return true;
  } catch { return false; }
}

/** One fixed row; a new pending selection owns the only commit lease. */
export function createAudioEnhancementPairCache(options: EnhancementPairCacheOptions = {}): AudioEnhancementPairCache {
  let database: Promise<IDBDatabase> | undefined;
  function open(): Promise<IDBDatabase> {
    if (database) return database;
    const { promise, resolve, reject } = Promise.withResolvers<IDBDatabase>();
    database = promise;
    let abandoned = false;
    try {
      const request = (options.indexedDB ?? globalThis.indexedDB).open(AUDIO_ENHANCEMENT_CACHE_DATABASE, 1);
      request.onupgradeneeded = () => {
        if (!request.result.objectStoreNames.contains(AUDIO_ENHANCEMENT_CACHE_STORE)) {
          request.result.createObjectStore(AUDIO_ENHANCEMENT_CACHE_STORE);
        }
      };
      request.onerror = () => { reject(new Error('Local enhanced-audio storage could not be opened.', { cause: request.error })); };
      request.onblocked = () => {
        abandoned = true;
        reject(new Error('Local enhanced-audio storage is blocked by another extension context.'));
      };
      request.onsuccess = () => {
        const db = request.result;
        if (abandoned || database !== promise) { db.close(); reject(new Error('Local enhanced-audio storage was closed before opening completed.')); return; }
        db.onversionchange = () => { db.close(); if (database === promise) database = undefined; };
        db.onclose = () => { if (database === promise) database = undefined; };
        resolve(db);
      };
    } catch (error) { reject(new Error('Local enhanced-audio storage is unavailable.', { cause: error })); }
    void promise.catch(() => { if (database === promise) database = undefined; });
    return promise;
  }
  async function mutate<T>(action: (store: IDBObjectStore, result: (value: T) => void, fail: (error: unknown) => void) => void, signal?: AbortSignal): Promise<T> {
    signal?.throwIfAborted();
    const db = await open();
    signal?.throwIfAborted();
    const { promise, resolve, reject } = Promise.withResolvers<T>();
    let value: T;
    let failure: unknown;
    try {
      const transaction = db.transaction(AUDIO_ENHANCEMENT_CACHE_STORE, 'readwrite', { durability: 'strict' });
      let settled = false;
      const cleanup = () => { signal?.removeEventListener('abort', abort); };
      const abort = () => {
        if (settled) return;
        failure = signal?.reason;
        try { transaction.abort(); }
        catch (error) {
          // A completed commit is irreversible even if its completion event is queued.
          if (!(error instanceof DOMException) || error.name !== 'InvalidStateError') {
            cleanup();
            reject(error);
          }
        }
      };
      signal?.addEventListener('abort', abort, { once: true });
      transaction.oncomplete = () => { settled = true; cleanup(); resolve(value); };
      transaction.onerror = transaction.onabort = () => {
        settled = true;
        cleanup();
        const cause = failure ?? transaction.error;
        reject(new Error(`Local enhanced-audio storage transaction failed${cause instanceof Error ? `: ${cause.message}` : '.'}`, { cause }));
      };
      const store = transaction.objectStore(AUDIO_ENHANCEMENT_CACHE_STORE);
      const result = (next: T) => { value = next; };
      const fail = (error: unknown) => { failure = error; transaction.abort(); };
      try { action(store, result, fail); } catch (error) { fail(error); }
    } catch (error) { reject(new Error('Local enhanced-audio storage transaction could not start.', { cause: error })); }
    return promise;
  }
  function readCurrent<T>(store: IDBObjectStore, action: (value: unknown) => T, result: (value: T) => void, fail: (error: unknown) => void): void {
    const request = store.get(AUDIO_ENHANCEMENT_CACHE_KEY);
    request.onsuccess = () => {
      try { result(action(request.result as unknown)); } catch (error) { fail(error); }
    };
  }
  return {
    async select(identity) {
      const lease = { key: identity.key, modelSha256: identity.modelSha256, nonce: crypto.randomUUID() };
      return mutate<EnhancementCacheSelection>((store, result, fail) => readCurrent(store, (current) => {
        if (isReadyPair(current) && current.key === identity.key && current.modelSha256 === identity.modelSha256 && matchesIdentity(current.lanes, identity)) {
          return { lease: { key: current.key, modelSha256: current.modelSha256, nonce: current.nonce }, ready: current };
        }
        store.clear();
        store.put(pending(lease), AUDIO_ENHANCEMENT_CACHE_KEY);
        return { lease };
      }, result, fail));
    },
    async invalidate(lease) {
      return mutate<boolean>((store, result, fail) => readCurrent(store, (current) => {
        if (!matchesLease(current, lease)) return false;
        store.clear();
        store.put(pending(lease), AUDIO_ENHANCEMENT_CACHE_KEY);
        return true;
      }, result, fail));
    },
    async commit(lease, identity, lanes, signal) {
      signal?.throwIfAborted();
      if (lease.key !== identity.key || lease.modelSha256 !== identity.modelSha256 || !await validLaneBytes(lanes, identity)) {
        throw new Error('Local enhanced-audio cache requires both complete, verified recording lanes.');
      }
      signal?.throwIfAborted();
      return mutate<boolean>((store, result, fail) => readCurrent(store, (current) => {
        if (!matchesLease(current, lease) || current.state !== 'pending') return false;
        const ready: ReadyEnhancementPair = { ...pending(lease), state: 'ready', lanes: [lanes[0], lanes[1]] };
        store.put(ready, AUDIO_ENHANCEMENT_CACHE_KEY);
        return true;
      }, result, fail), signal);
    },
    async clear() { await mutate<void>((store, result) => { store.clear(); result(undefined); }); },
    close() {
      const previous = database;
      database = undefined;
      void previous?.then((db) => db.close()).catch(() => undefined);
    }
  };
}

export interface PreparedEnhancementSource extends EnhancementCacheLane {
  track: CapturedAudioTrack;
  bytes: ArrayBuffer;
}
export interface EnhancementCacheModel { id: string; sha256: string }

/** Source hashing/header checks precede all model resources and PCM sample allocation. */
export async function runWithAudioEnhancementPairCache(
  tracks: CapturedAudioTrack[],
  model: EnhancementCacheModel,
  cache: AudioEnhancementPairCache | null,
  generate: (sources: PreparedEnhancementSource[]) => Promise<EnhancedAudioTrack[]>,
  onProgress?: (progress: AudioEnhancementProgress) => void | Promise<void>,
  signal?: AbortSignal
): Promise<Omit<EnhancedAudioBatch, 'provider'>> {
  signal?.throwIfAborted();
  const sources: PreparedEnhancementSource[] = [];
  for (let trackIndex = 0; trackIndex < tracks.length; trackIndex++) {
    const track = tracks[trackIndex];
    await onProgress?.({ phase: cache ? 'cache-lookup' : 'queued', trackId: track.trackId, trackIndex, trackCount: tracks.length, completedChunks: 0, totalChunks: 0 });
    signal?.throwIfAborted();
    const bytes = await track.blob.arrayBuffer(), sourceSha256 = await enhancementSha256(bytes);
    signal?.throwIfAborted();
    const { sampleRate, frameCount } = readEnhancementWavHeader(bytes);
    sources.push({ track, bytes, trackId: track.trackId, sourceSha256, sampleRate, frameCount });
  }
  let identity: EnhancementCacheIdentity | undefined;
  let lease: EnhancementCacheLease | undefined;
  let hit: EnhancedAudioTrack[] | undefined;
  let cacheMessage: string | undefined;
  if (cache && sources.length === 2 && sources[0].trackId !== sources[1].trackId) {
    const sorted = sources.map(({ trackId, sourceSha256, sampleRate, frameCount }) => ({ trackId, sourceSha256, sampleRate, frameCount }))
      .sort((left, right) => left.trackId < right.trackId ? -1 : left.trackId > right.trackId ? 1 : 0);
    const key = await enhancementSha256(new TextEncoder().encode(JSON.stringify([model.sha256, sorted.map(({ trackId, sourceSha256 }) => [trackId, sourceSha256])])).buffer);
    identity = { key, modelSha256: model.sha256, lanes: [sorted[0], sorted[1]] };
    try {
      const selected = await cache.select(identity);
      signal?.throwIfAborted();
      lease = selected.lease;
      if (selected.ready) {
        if (await validLaneBytes(selected.ready.lanes, identity)) {
          hit = sources.map((source) => {
            const lane = selected.ready!.lanes.find((candidate) => candidate.trackId === source.trackId)!;
            return { bytes: new Uint8Array(lane.wav), metadata: {
              trackId: source.trackId, speakerKey: source.track.speakerKey ?? source.trackId,
              trackLabel: source.track.trackLabel ?? source.track.speakerKey ?? source.trackId, mimeType: 'audio/wav' as const,
              sampleRate: lane.sampleRate, frameCount: lane.frameCount, sourceSha256: lane.sourceSha256, wavSha256: lane.wavSha256,
              totalBytes: lane.wav.byteLength, chunkCount: Math.ceil(lane.wav.byteLength / LOCAL_MODEL_AUDIO_CHUNK_BYTES)
            } };
          });
        } else if (!await cache.invalidate(lease)) {
          lease = undefined;
          cacheMessage = 'A newer recording selection replaced this local cache entry; these enhanced recordings were not retained.';
        }
      }
    } catch (error) {
      signal?.throwIfAborted();
      lease = undefined;
      cacheMessage = cacheFailureMessage(error);
    }
  } else if (cache) {
    cacheMessage = 'Local enhanced-audio reuse requires both distinct Original recording lanes.';
    try { await cache.clear(); } catch (error) { cacheMessage = cacheFailureMessage(error); }
  }
  if (hit) {
    for (let trackIndex = 0; trackIndex < tracks.length; trackIndex++) {
      await onProgress?.({ phase: 'cache-hit', trackId: tracks[trackIndex].trackId, trackIndex, trackCount: tracks.length, completedChunks: 0, totalChunks: 0 });
    }
    signal?.throwIfAborted();
    return { model: model.id, modelSha256: model.sha256, tracks: hit, cacheStatus: 'hit' };
  }
  signal?.throwIfAborted();
  const enhanced = await generate(sources);
  signal?.throwIfAborted();
  if (enhanced.length !== sources.length || enhanced.some((track, index) => track.metadata.trackId !== sources[index].trackId)) {
    throw new Error('ZipEnhancer did not return the complete Original recording selection.');
  }
  if (cache && identity && lease) {
    try {
      const lanes = enhanced.map(({ bytes, metadata }) => ({ trackId: metadata.trackId, sourceSha256: metadata.sourceSha256,
        sampleRate: metadata.sampleRate, frameCount: metadata.frameCount, wavSha256: metadata.wavSha256, wav: bytes.buffer }));
      if (await cache.commit(lease, identity, lanes, signal)) {
        return { model: model.id, modelSha256: model.sha256, tracks: enhanced, cacheStatus: 'stored' };
      }
      cacheMessage = 'A newer recording selection replaced this local cache entry; these enhanced recordings were not retained.';
    } catch (error) { signal?.throwIfAborted(); cacheMessage = cacheFailureMessage(error); }
  }
  return { model: model.id, modelSha256: model.sha256, tracks: enhanced, ...(cache ? { cacheStatus: 'unavailable' as const, cacheMessage } : {}) };
}

function cacheFailureMessage(error: unknown): string {
  return `Enhanced audio is available, but local reuse is unavailable: ${error instanceof Error ? error.message : String(error)}`;
}
