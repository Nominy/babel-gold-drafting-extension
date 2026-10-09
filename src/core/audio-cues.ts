import {
  AUDIO_ENHANCEMENT_ACTIVE_REQUEST, AUDIO_ENHANCEMENT_ACTIVE_RESPONSE,
  AUDIO_ENHANCEMENT_PROTOCOL_VERSION, readAudioEnhancementState, type ActiveAudioRequest, type AudioEnhancementState
} from '@nominy/babel-babel-runtime';
import type { CapturedAudioTrack } from './types';
import {
  AUDIO_FLUSH_REQUEST_MESSAGE_TYPE,
  AUDIO_RESPONSE_MESSAGE_TYPE,
  AUDIO_SOURCE_MESSAGE_TYPE,
  type AudioSourceMessage,
  type AudioResponseMessage
} from './audio-intercept-protocol';
import { isBlobUrl } from './audio-url';
import { buildCanonicalTaskIdentity, captureTranscriptJob } from './transcript';

interface InterceptedAudioTrack {
  url: string;
  trackId?: string;
  speakerKey?: string;
  trackLabel?: string;
  mimeType: string;
  bytes: ArrayBuffer;
  capturedAt: number;
}

interface DiscoveredAudioSource {
  url: string;
  trackId?: string;
  speakerKey?: string;
  trackLabel?: string;
  mimeType?: string;
  discoveredAt: number;
}

const MAX_CAPTURE_BYTES = 220 * 1024 * 1024;
const MAX_CAPTURED_RESPONSES = 8;
const MAX_DISCOVERED_SOURCES = 64;

interface AudioCaptureSession {
  taskId: string;
  pathname: string;
  queryTaskId: string;
  interceptedAudioByUrl: Map<string, InterceptedAudioTrack>;
  discoveredAudioSourceByUrl: Map<string, DiscoveredAudioSource>;
  activeCaptures: number;
  interceptedBytes: number;
}

let installedWindow: Window | null = null;
let audioCaptureSession: AudioCaptureSession | null = null;

// Task query keys can change before the published review action catches up.
// Ignore panel/hash navigation, but retain explicit task identifiers separately.
function readQueryTaskId(): string {
  const query = new URLSearchParams(window.location.search);
  return JSON.stringify(['jobId', 'transcriptionChunkId', 'annotationId', 'id']
    .map((key) => query.get(key)?.trim() || ''));
}

function isCurrentAudioCaptureSession(session: AudioCaptureSession): boolean {
  return (
    session.pathname === window.location.pathname &&
    session.queryTaskId === readQueryTaskId() &&
    session.taskId === buildCanonicalTaskIdentity(captureTranscriptJob())
  );
}

function getAudioCaptureSession(): AudioCaptureSession {
  if (!audioCaptureSession || !isCurrentAudioCaptureSession(audioCaptureSession)) {
    audioCaptureSession = {
      taskId: buildCanonicalTaskIdentity(captureTranscriptJob()),
      pathname: window.location.pathname,
      queryTaskId: readQueryTaskId(),
      interceptedAudioByUrl: new Map(),
      discoveredAudioSourceByUrl: new Map(),
      activeCaptures: 0,
      interceptedBytes: 0
    };
  }
  return audioCaptureSession;
}

function clearAudioCaptureSession(session: AudioCaptureSession): void {
  session.interceptedAudioByUrl.clear();
  session.discoveredAudioSourceByUrl.clear();
  session.interceptedBytes = 0;
}

function assertAudioCaptureTask(session: AudioCaptureSession): void {
  if (!isCurrentAudioCaptureSession(session)) {
    throw new Error('Audio capture task changed before capture completed.');
  }
}

function sourceForAudioElement(audio: HTMLMediaElement): string {
  const direct = audio.currentSrc || audio.getAttribute('src') || '';
  if (direct) {
    return direct;
  }

  const source = audio.querySelector<HTMLSourceElement>('source[src]');
  return source?.src || source?.getAttribute('src') || '';
}

function toAbsoluteUrl(source: string): string {
  return new URL(source, window.location.href).toString();
}


function isObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object');
}

function isAudioResponseMessage(value: unknown): value is AudioResponseMessage {
  return (
    isObject(value) &&
    value.type === AUDIO_RESPONSE_MESSAGE_TYPE &&
    typeof value.url === 'string' &&
    typeof value.mimeType === 'string' &&
    typeof value.capturedAt === 'number' &&
    value.bytes instanceof ArrayBuffer
  );
}

function isAudioSourceMessage(value: unknown): value is AudioSourceMessage {
  return (
    isObject(value) &&
    value.type === AUDIO_SOURCE_MESSAGE_TYPE &&
    typeof value.url === 'string' &&
    typeof value.discoveredAt === 'number'
  );
}

function readNonEmptyString(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value.trim() : undefined;
}

function getTrackIdentity(record: {
  trackId?: string;
  speakerKey?: string;
  trackLabel?: string;
}): string | null {
  return record.speakerKey || record.trackLabel || record.trackId || null;
}

function hasLaneMapping(record: {
  trackId?: string;
  speakerKey?: string;
  trackLabel?: string;
}): boolean {
  return Boolean(getTrackIdentity(record));
}

function markSeen(seen: Set<string>, record: { url: string; trackId?: string; speakerKey?: string; trackLabel?: string }): void {
  seen.add(record.url);
  const identity = getTrackIdentity(record);
  if (identity) {
    seen.add(`track:${identity}`);
  }
}

function hasSeenTrack(seen: Set<string>, record: { trackId?: string; speakerKey?: string; trackLabel?: string }): boolean {
  const identity = getTrackIdentity(record);
  return Boolean(identity && seen.has(`track:${identity}`));
}

function compareAudioRecords(left: { url: string }, right: { url: string }): number {
  const leftBlob = isBlobUrl(left.url);
  const rightBlob = isBlobUrl(right.url);
  if (leftBlob !== rightBlob) {
    return leftBlob ? 1 : -1;
  }
  return 0;
}

function handleAudioResponseMessage(event: MessageEvent): void {
  if (event.source && event.source !== window) {
    return;
  }
  if (isAudioSourceMessage(event.data)) {
    const { discoveredAudioSourceByUrl } = getAudioCaptureSession();
    const url = toAbsoluteUrl(event.data.url);
    const current = discoveredAudioSourceByUrl.get(url);
    if (!current || current.discoveredAt <= event.data.discoveredAt) {
      discoveredAudioSourceByUrl.set(url, {
        url,
        trackId: readNonEmptyString(event.data.trackId),
        speakerKey: readNonEmptyString(event.data.speakerKey),
        trackLabel: readNonEmptyString(event.data.trackLabel),
        mimeType: readNonEmptyString(event.data.mimeType),
        discoveredAt: event.data.discoveredAt
      });
    }
    while (discoveredAudioSourceByUrl.size > MAX_DISCOVERED_SOURCES) {
      discoveredAudioSourceByUrl.delete(discoveredAudioSourceByUrl.keys().next().value!);
    }
    return;
  }

  if (!isAudioResponseMessage(event.data) || !event.data.bytes.byteLength || event.data.bytes.byteLength > MAX_CAPTURE_BYTES) {
    return;
  }

  const session = getAudioCaptureSession();
  const { interceptedAudioByUrl } = session;
  const url = toAbsoluteUrl(event.data.url);
  const current = interceptedAudioByUrl.get(url);
  if (current && current.capturedAt > event.data.capturedAt) {
    return;
  }
  if (current) {
    session.interceptedBytes -= current.bytes.byteLength;
    interceptedAudioByUrl.delete(url);
  }
  session.interceptedBytes += event.data.bytes.byteLength;

  interceptedAudioByUrl.set(url, {
    url,
    trackId: readNonEmptyString(event.data.trackId),
    speakerKey: readNonEmptyString(event.data.speakerKey),
    trackLabel: readNonEmptyString(event.data.trackLabel),
    mimeType: event.data.mimeType || 'application/octet-stream',
    bytes: event.data.bytes,
    capturedAt: event.data.capturedAt
  });
  while (interceptedAudioByUrl.size > MAX_CAPTURED_RESPONSES || session.interceptedBytes > MAX_CAPTURE_BYTES) {
    const oldest = interceptedAudioByUrl.values().next().value!;
    session.interceptedBytes -= oldest.bytes.byteLength;
    interceptedAudioByUrl.delete(oldest.url);
  }
}

export function installAudioRequestCapture(): void {
  ensureAudioRequestCaptureInstalled();
  window.postMessage({ type: AUDIO_FLUSH_REQUEST_MESSAGE_TYPE }, '*');
}

function ensureAudioRequestCaptureInstalled(): void {
  if (installedWindow === window) {
    return;
  }

  audioCaptureSession = null;
  installedWindow = window;
  window.addEventListener('message', handleAudioResponseMessage);
}

async function requestAudioFlush(): Promise<void> {
  ensureAudioRequestCaptureInstalled();
  window.postMessage({ type: AUDIO_FLUSH_REQUEST_MESSAGE_TYPE }, '*');
  await new Promise((resolve) => window.setTimeout(resolve, 180));
}

function appendInterceptedTracks(
  session: AudioCaptureSession,
  tracks: CapturedAudioTrack[],
  seen: Set<string>,
  currentLaneSourceUrls: Set<string>
): void {
  const intercepted = Array.from(session.interceptedAudioByUrl.values()).sort((a, b) => {
    const sourceOrder = compareAudioRecords(a, b);
    return sourceOrder || a.capturedAt - b.capturedAt;
  });
  for (const record of intercepted) {
    if (
      !hasLaneMapping(record) ||
      seen.has(record.url) ||
      hasSeenTrack(seen, record) ||
      (currentLaneSourceUrls.size > 0 && !currentLaneSourceUrls.has(record.url))
    ) {
      continue;
    }
    markSeen(seen, record);
    tracks.push({
      trackId: record.trackId || `audio-${tracks.length + 1}`,
      speakerKey: record.speakerKey,
      trackLabel: record.trackLabel,
      source: record.url,
      blob: new Blob([record.bytes], { type: record.mimeType }),
      mimeType: record.mimeType
    });
  }
}

function getCurrentLaneSourceUrls(session: AudioCaptureSession): Set<string> {
  const urls = new Set<string>();
  for (const record of session.discoveredAudioSourceByUrl.values()) {
    if (hasLaneMapping(record)) {
      urls.add(record.url);
    }
  }
  return urls;
}

/**
 * Fetches one lane's audio. An unavailable lane (HTTP error, revoked blob URL,
 * rejected body read, empty body) resolves to `null` so the remaining lanes
 * still reach the capture guard, but the reason is logged: a silently missing
 * lane on live is indistinguishable from a lane that never existed.
 */
async function fetchAvailableAudio(source: string): Promise<Blob | null> {
  let reason: string;
  try {
    const response = await fetch(source, {
      credentials: new URL(source, window.location.href).origin === window.location.origin ? 'include' : 'omit'
    });
    if (!response.ok) {
      reason = `HTTP ${response.status}`;
    } else {
      const blob = await response.blob();
      if (blob.size) return blob;
      reason = 'empty body';
    }
  } catch (error) {
    reason = error instanceof Error ? error.message : String(error);
  }
  console.warn(`[babel-gold-drafting] Audio lane unavailable (${reason}): ${source}`);
  return null;
}

async function appendDiscoveredSourceTracks(
  session: AudioCaptureSession,
  tracks: CapturedAudioTrack[],
  seen: Set<string>
): Promise<void> {
  const discovered = Array.from(session.discoveredAudioSourceByUrl.values()).sort((a, b) => {
    const sourceOrder = compareAudioRecords(a, b);
    return sourceOrder || a.discoveredAt - b.discoveredAt;
  });
  for (const record of discovered) {
    if (!hasLaneMapping(record) || seen.has(record.url) || hasSeenTrack(seen, record)) {
      continue;
    }
    seen.add(record.url);

    const blob = await fetchAvailableAudio(record.url);
    if (!blob) continue;
    markSeen(seen, record);
    tracks.push({
      trackId: record.trackId || `audio-${tracks.length + 1}`,
      speakerKey: record.speakerKey,
      trackLabel: record.trackLabel,
      source: record.url,
      blob,
      mimeType: blob.type || record.mimeType || 'application/octet-stream'
    });
  }
}

function isPendingAudioSelection(state: AudioEnhancementState): boolean {
  return state.status === 'switching' || (state.desiredStrength !== state.strength && state.status !== 'error');
}

export function isAudioSelectionReady(root: ParentNode = document): boolean {
  const documentRef = 'documentElement' in root ? root as Document : (root as Node).ownerDocument;
  const state = readAudioEnhancementState(documentRef);
  return !state || state.taskId !== captureTranscriptJob(root).jobId || !isPendingAudioSelection(state);
}

function captureHelperSelectedAudio(root: ParentNode, selection: 'active' | 'original'): Promise<CapturedAudioTrack[]> | null {
  const documentRef = 'documentElement' in root ? root as Document : (root as Node).ownerDocument;
  const state = readAudioEnhancementState(documentRef);
  const job = captureTranscriptJob(root);
  if (!state || state.taskId !== job.jobId) return null;
  if (selection === 'active' && isPendingAudioSelection(state)) {
    return Promise.reject(new Error('The selected audio is preparing or switching. Wait for its paired source commit.'));
  }
  const requestId = crypto.randomUUID();
  const { promise, resolve, reject } = Promise.withResolvers<CapturedAudioTrack[]>();
  const finish = (result: CapturedAudioTrack[] | Error) => {
    window.clearTimeout(timeout);
    window.removeEventListener('message', receive);
    if (result instanceof Error) reject(result); else resolve(result);
  };
  const receive = (event: MessageEvent) => {
    const data = event.data;
    if (event.source !== window || !isObject(data) || data.type !== AUDIO_ENHANCEMENT_ACTIVE_RESPONSE || data.requestId !== requestId) return;
    const current = readAudioEnhancementState(documentRef);
    if (data.version !== AUDIO_ENHANCEMENT_PROTOCOL_VERSION || data.taskId !== job.jobId ||
        !current || current.taskId !== job.jobId || captureTranscriptJob(root).jobId !== job.jobId) {
      finish(new Error('The audio task changed during selected-source capture.'));
      return;
    }
    if (selection === 'active' && (isPendingAudioSelection(current) || current.strength !== data.strength ||
        current.variantKey !== data.variantKey || current.strength !== state.strength || current.variantKey !== state.variantKey)) {
      finish(new Error('The selected audio changed during capture.'));
      return;
    }
    if (selection === 'original' && (data.strength !== 0 || data.variantKey !== '')) {
      finish(new Error('Original audio capture must not return enhanced or mixed audio.'));
      return;
    }
    if (data.available !== true) {
      finish(new Error('The selected native audio buffers are not ready. No original-source substitution was used.'));
      return;
    }
    if (!Array.isArray(data.tracks) || data.tracks.length !== 2) {
      finish(new Error('Selected audio must contain both native recording lanes.'));
      return;
    }
    const ids = new Set<string>();
    const tracks: CapturedAudioTrack[] = [];
    for (const track of data.tracks) {
      if (!isObject(track) || typeof track.trackId !== 'string' || !track.trackId ||
          typeof track.speakerKey !== 'string' || !track.speakerKey || typeof track.trackLabel !== 'string' ||
          track.mimeType !== 'audio/wav' || !(track.bytes instanceof ArrayBuffer) ||
          track.bytes.byteLength === 0 || track.bytes.byteLength > MAX_CAPTURE_BYTES || ids.has(track.speakerKey)) {
        finish(new Error('Selected native audio contains an invalid or duplicated lane.'));
        return;
      }
      ids.add(track.speakerKey);
      tracks.push({ trackId: track.trackId, speakerKey: track.speakerKey, trackLabel: track.trackLabel,
        source: data.strength === 0 ? `helper-selected:original:${track.trackId}`
          : `helper-selected:${data.strength}:${data.variantKey}:${track.trackId}`, mimeType: 'audio/wav',
        blob: new Blob([track.bytes], { type: 'audio/wav' }) });
    }
    finish(tracks);
  };
  const timeout = window.setTimeout(() => finish(new Error('Helper selected-audio capture timed out.')), 5000);
  window.addEventListener('message', receive);
  try {
    window.postMessage({ type: AUDIO_ENHANCEMENT_ACTIVE_REQUEST, version: AUDIO_ENHANCEMENT_PROTOCOL_VERSION,
      requestId, taskId: job.jobId, selection } satisfies ActiveAudioRequest, '*');
  } catch (error) {
    finish(error instanceof Error ? error : new Error(String(error)));
  }
  return promise;
}

export async function captureAudioTracksForDrafting(root: ParentNode = document): Promise<CapturedAudioTrack[]> {
  ensureAudioRequestCaptureInstalled();
  const selected = captureHelperSelectedAudio(root, 'active');
  return selected ?? captureNetworkAudio(root);
}

export async function captureOriginalAudioTracksForEnhancement(root: ParentNode = document): Promise<CapturedAudioTrack[]> {
  ensureAudioRequestCaptureInstalled();
  const original = captureHelperSelectedAudio(root, 'original');
  return original ?? captureNetworkAudio(root);
}

async function captureNetworkAudio(root: ParentNode): Promise<CapturedAudioTrack[]> {
  ensureAudioRequestCaptureInstalled();
  const session = getAudioCaptureSession();
  session.activeCaptures += 1;
  try {
    await requestAudioFlush();
    assertAudioCaptureTask(session);
    const seen = new Set<string>();
    const seenDomSources = new Set<string>();
    const sources = Array.from(root.querySelectorAll('audio'))
      .map((audio) => sourceForAudioElement(audio))
      .filter(Boolean)
      .map(toAbsoluteUrl)
      .filter((source) => {
        if (seenDomSources.has(source)) {
          return false;
        }
        seenDomSources.add(source);
        return true;
      });

    const tracks: CapturedAudioTrack[] = [];
    appendInterceptedTracks(session, tracks, seen, getCurrentLaneSourceUrls(session));
    await appendDiscoveredSourceTracks(session, tracks, seen);
    assertAudioCaptureTask(session);
    if (tracks.some(hasLaneMapping)) {
      return tracks;
    }

    for (const source of sources) {
      if (seen.has(source)) {
        continue;
      }
      seen.add(source);
      const blob = await fetchAvailableAudio(source);
      if (!blob) continue;
      tracks.push({
        trackId: `audio-${tracks.length + 1}`,
        source,
        blob,
        mimeType: blob.type || 'application/octet-stream'
      });
    }

    assertAudioCaptureTask(session);
    return tracks;
  } finally {
    session.activeCaptures -= 1;
    // Other readers keep this task's cache alive, never a later task's cache.
    if (session.activeCaptures === 0) clearAudioCaptureSession(session);
  }
}
