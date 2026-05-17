import type { CapturedAudioTrack } from './types';
import {
  AUDIO_FLUSH_REQUEST_MESSAGE_TYPE,
  AUDIO_RESPONSE_MESSAGE_TYPE,
  AUDIO_SOURCE_MESSAGE_TYPE,
  type AudioSourceMessage,
  type AudioResponseMessage
} from './audio-intercept-protocol';
import { isBlobUrl } from './audio-url';

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

const interceptedAudioByUrl = new Map<string, InterceptedAudioTrack>();
const discoveredAudioSourceByUrl = new Map<string, DiscoveredAudioSource>();
let installedWindow: Window | null = null;

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
  return record.trackId || record.speakerKey || record.trackLabel || null;
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
    return;
  }

  if (!isAudioResponseMessage(event.data) || !event.data.bytes.byteLength) {
    return;
  }

  const url = toAbsoluteUrl(event.data.url);
  const current = interceptedAudioByUrl.get(url);
  if (current && current.capturedAt > event.data.capturedAt) {
    return;
  }

  interceptedAudioByUrl.set(url, {
    url,
    trackId: readNonEmptyString(event.data.trackId),
    speakerKey: readNonEmptyString(event.data.speakerKey),
    trackLabel: readNonEmptyString(event.data.trackLabel),
    mimeType: event.data.mimeType || 'application/octet-stream',
    bytes: event.data.bytes.slice(0),
    capturedAt: event.data.capturedAt
  });
}

export function installAudioRequestCapture(): void {
  if (installedWindow === window) {
    window.postMessage({ type: AUDIO_FLUSH_REQUEST_MESSAGE_TYPE }, '*');
    return;
  }

  interceptedAudioByUrl.clear();
  discoveredAudioSourceByUrl.clear();
  installedWindow = window;
  window.addEventListener('message', handleAudioResponseMessage);
  window.postMessage({ type: AUDIO_FLUSH_REQUEST_MESSAGE_TYPE }, '*');
}

async function requestAudioFlush(): Promise<void> {
  installAudioRequestCapture();
  window.postMessage({ type: AUDIO_FLUSH_REQUEST_MESSAGE_TYPE }, '*');
  await new Promise((resolve) => window.setTimeout(resolve, 180));
}

function appendInterceptedTracks(tracks: CapturedAudioTrack[], seen: Set<string>): void {
  const intercepted = Array.from(interceptedAudioByUrl.values()).sort((a, b) => {
    const sourceOrder = compareAudioRecords(a, b);
    return sourceOrder || a.capturedAt - b.capturedAt;
  });
  for (const record of intercepted) {
    if (!hasLaneMapping(record) || seen.has(record.url) || hasSeenTrack(seen, record)) {
      continue;
    }
    markSeen(seen, record);
    tracks.push({
      trackId: record.trackId || `audio-${tracks.length + 1}`,
      speakerKey: record.speakerKey,
      trackLabel: record.trackLabel,
      source: record.url,
      blob: new Blob([record.bytes.slice(0)], { type: record.mimeType }),
      mimeType: record.mimeType
    });
  }
}

async function appendDiscoveredSourceTracks(tracks: CapturedAudioTrack[], seen: Set<string>): Promise<void> {
  const discovered = Array.from(discoveredAudioSourceByUrl.values()).sort((a, b) => {
    const sourceOrder = compareAudioRecords(a, b);
    return sourceOrder || a.discoveredAt - b.discoveredAt;
  });
  for (const record of discovered) {
    if (!hasLaneMapping(record) || seen.has(record.url) || hasSeenTrack(seen, record)) {
      continue;
    }
    markSeen(seen, record);

    const response = await fetch(record.url, { credentials: 'include' });
    if (!response.ok) {
      throw new Error(`Audio fetch failed: ${response.status}`);
    }
    const blob = await response.blob();
    if (!blob.size) {
      continue;
    }
    tracks.push({
      trackId: record.trackId || `audio-${tracks.length + 1}`,
      speakerKey: record.speakerKey,
      trackLabel: record.trackLabel,
      source: record.url,
      blob,
      mimeType: record.mimeType || blob.type || 'application/octet-stream'
    });
  }
}

export async function captureAudioTracksForDrafting(root: ParentNode = document): Promise<CapturedAudioTrack[]> {
  await requestAudioFlush();
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
  appendInterceptedTracks(tracks, seen);
  await appendDiscoveredSourceTracks(tracks, seen);
  if (tracks.some(hasLaneMapping)) {
    return tracks;
  }

  for (const source of sources) {
    if (seen.has(source)) {
      continue;
    }
    seen.add(source);
    const response = await fetch(source, { credentials: 'include' });
    if (!response.ok) {
      throw new Error(`Audio fetch failed: ${response.status}`);
    }
    const blob = await response.blob();
    if (!blob.size) {
      continue;
    }
    tracks.push({
      trackId: `audio-${tracks.length + 1}`,
      source,
      blob,
      mimeType: blob.type || 'application/octet-stream'
    });
  }

  return tracks;
}
