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
  pageUrl: string;
  interceptedAudioByUrl: Map<string, InterceptedAudioTrack>;
  discoveredAudioSourceByUrl: Map<string, DiscoveredAudioSource>;
  activeCaptures: number;
  interceptedBytes: number;
}

let installedWindow: Window | null = null;
let audioCaptureSession: AudioCaptureSession | null = null;

function getAudioCaptureSession(): AudioCaptureSession {
  const taskId = buildCanonicalTaskIdentity(captureTranscriptJob());
  const pageUrl = window.location.href;
  if (!audioCaptureSession || audioCaptureSession.taskId !== taskId || audioCaptureSession.pageUrl !== pageUrl) {
    audioCaptureSession = {
      taskId,
      pageUrl,
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
  if (session.pageUrl !== window.location.href || session.taskId !== buildCanonicalTaskIdentity(captureTranscriptJob())) {
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

async function fetchAvailableAudio(source: string): Promise<Blob | null> {
  try {
    const response = await fetch(source, {
      credentials: new URL(source, window.location.href).origin === window.location.origin ? 'include' : 'omit'
    });
    if (!response.ok) return null;
    const blob = await response.blob();
    return blob.size ? blob : null;
  } catch {
    // Revoked blob URLs and rejected body reads are unavailable lanes too.
    // Returning the remaining tracks lets the capture guard require a decision.
    return null;
  }
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

export async function captureAudioTracksForDrafting(root: ParentNode = document): Promise<CapturedAudioTrack[]> {
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
