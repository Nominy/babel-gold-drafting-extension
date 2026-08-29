import {
  AUDIO_ENABLE_CAPTURE_MESSAGE_TYPE,
  AUDIO_FLUSH_REQUEST_MESSAGE_TYPE,
  AUDIO_RESPONSE_MESSAGE_TYPE,
  AUDIO_SOURCE_MESSAGE_TYPE,
  PAGE_TASK_ID_ATTRIBUTE,
  PAGE_TASK_ID_REQUEST_MESSAGE_TYPE,
  PAGE_TASK_ID_RESPONSE_MESSAGE_TYPE,
  type AudioEnableCaptureMessage,
  type AudioSourceMessage,
  type AudioInterceptSource,
  type AudioResponseMessage,
  type PageTaskIdResponseMessage
} from '../core/audio-intercept-protocol';
import { isLikelyAudioSource } from '../core/audio-url';

declare global {
  interface Window {
    __babelGoldDraftingAudioInterceptorInstalled?: boolean;
  }
}

const MAX_CAPTURE_BYTES = 220 * 1024 * 1024;
const MAX_STORED_RESPONSES = 8;

type StoredAudioResponse = Omit<AudioResponseMessage, 'type'>;
type TrackMapping = Pick<AudioResponseMessage, 'trackId' | 'speakerKey' | 'trackLabel' | 'mappingSource'>;

const storedResponses: StoredAudioResponse[] = [];

function safe<T>(read: () => T, fallback: T): T {
  try {
    return read();
  } catch {
    return fallback;
  }
}

function isAudioResponse(url: string, mimeType: string): boolean {
  return isLikelyAudioSource(url, mimeType);
}

function isAudioEnableCaptureMessage(value: unknown): value is AudioEnableCaptureMessage {
  return Boolean(value && typeof value === 'object' && (value as { type?: unknown }).type === AUDIO_ENABLE_CAPTURE_MESSAGE_TYPE);
}

function toAbsoluteUrl(url: string): string {
  try {
    return new URL(url, window.location.href).toString();
  } catch {
    return url;
  }
}

function getRequestUrl(input: RequestInfo | URL): string {
  if (input instanceof Request) {
    return input.url;
  }
  return toAbsoluteUrl(String(input));
}

function getReactInternalValue(element: Element | null, prefix: string): unknown {
  if (!(element instanceof HTMLElement)) {
    return null;
  }

  for (const name of safe(() => Object.getOwnPropertyNames(element), [])) {
    if (typeof name === 'string' && name.startsWith(prefix)) {
      return safe(() => (element as unknown as Record<string, unknown>)[name], null);
    }
  }

  return null;
}

function getReactFiber(element: Element | null): Record<string, unknown> | null {
  const fiber = getReactInternalValue(element, '__reactFiber$');
  return fiber && typeof fiber === 'object' ? (fiber as Record<string, unknown>) : null;
}

function readCurrentReviewActionId(): string {
  const textarea = document.querySelector('textarea[placeholder^="What was said"]');
  let fiber = getReactFiber(textarea);
  let depth = 0;
  while (fiber && depth <= 30) {
    const props = fiber.memoizedProps;
    if (props && typeof props === 'object' && 'reviewActionId' in props) {
      const reviewActionId = readString((props as Record<string, unknown>).reviewActionId);
      if (reviewActionId) return reviewActionId;
    }
    fiber = fiber.return && typeof fiber.return === 'object'
      ? (fiber.return as Record<string, unknown>)
      : null;
    depth += 1;
  }
  return '';
}

function getTrackDetailsForHost(host: HTMLElement): Record<string, unknown> | null {
  let owner = getReactFiber(host) || getReactFiber(host.parentElement);
  let depth = 0;

  while (owner && depth < 20) {
    const props = owner.memoizedProps;
    const track =
      props && typeof props === 'object' && 'track' in props && props.track && typeof props.track === 'object'
        ? (props.track as Record<string, unknown>)
        : null;
    if (track) {
      return track;
    }

    owner = owner.return && typeof owner.return === 'object' ? (owner.return as Record<string, unknown>) : null;
    depth += 1;
  }

  return null;
}

function getWaveformRegistryFromValue(value: unknown): Record<string, unknown> | null {
  const registry =
    value && typeof value === 'object' && !Array.isArray(value) && 'current' in value
      ? (value as { current?: unknown }).current
      : value;

  if (!registry || typeof registry !== 'object' || Array.isArray(registry)) {
    return null;
  }

  const candidate = registry as Record<string, unknown>;
  const hasWaveEntry = Object.keys(candidate).some((key) => {
    const entry = candidate[key];
    return Boolean(entry && typeof entry === 'object' && 'wavesurfer' in entry);
  });

  return hasWaveEntry ? candidate : null;
}

function getWaveformRegistryFromHost(host: HTMLElement): Record<string, unknown> | null {
  let owner = getReactFiber(host) || getReactFiber(host.parentElement);
  let depth = 0;

  while (owner && depth < 16) {
    let hook = owner.memoizedState && typeof owner.memoizedState === 'object'
      ? (owner.memoizedState as Record<string, unknown>)
      : null;
    let hookIndex = 0;

    while (hook && hookIndex < 24) {
      const registry = getWaveformRegistryFromValue(hook.memoizedState);
      if (registry) {
        return registry;
      }

      hook = hook.next && typeof hook.next === 'object' ? (hook.next as Record<string, unknown>) : null;
      hookIndex += 1;
    }

    owner = owner.return && typeof owner.return === 'object' ? (owner.return as Record<string, unknown>) : null;
    depth += 1;
  }

  return null;
}

function getWaveformHosts(): HTMLDivElement[] {
  return Array.from(document.querySelectorAll('div')).filter((node): node is HTMLDivElement => {
    if (!(node instanceof HTMLElement) || !(node.shadowRoot instanceof ShadowRoot)) {
      return false;
    }
    return Boolean(node.shadowRoot.querySelector('[part="scroll"], [part="wrapper"]'));
  });
}

function readString(value: unknown): string {
  return typeof value === 'string' && value.trim() ? value.trim() : '';
}

function collectAudioUrls(value: unknown, urls: Set<string> = new Set(), depth = 0): Set<string> {
  if (!value || (typeof value !== 'object' && typeof value !== 'function') || depth > 2) {
    return urls;
  }

  const record = value as Record<string, unknown>;
  for (const key of [
    'url',
    'src',
    'currentSrc',
    'href',
    'audioUrl',
    'fileUrl',
    'downloadUrl',
    'recordingUrl',
    'signedUrl',
    'source',
    'waveformUrl'
  ]) {
    const text = readString(record[key]);
    if (text && isAudioResponse(text, '')) {
      urls.add(toAbsoluteUrl(text));
    }
  }

  for (const key of ['media', 'options', 'backend', 'track', 'recording', 'processedRecording', 'audio', 'source']) {
    collectAudioUrls(record[key], urls, depth + 1);
  }

  if (typeof record.getMediaElement === 'function') {
    collectAudioUrls(safe(() => (record.getMediaElement as () => unknown)(), null), urls, depth + 1);
  }

  return urls;
}

function collectWaveformAudioMappings(): Map<string, TrackMapping> {
  const mappings = new Map<string, TrackMapping>();

  for (const host of getWaveformHosts()) {
    const track = getTrackDetailsForHost(host);
    const hostTrackId = readString(track?.id);
    const trackLabel = readString(track?.label) || readString(track?.name);
    const registry = getWaveformRegistryFromHost(host);
    if (!registry) {
      continue;
    }

    for (const key of Object.keys(registry)) {
      if (hostTrackId && key !== hostTrackId) {
        continue;
      }

      const entry = registry[key];
      if (!entry || typeof entry !== 'object') {
        continue;
      }

      const entryRecord = entry as Record<string, unknown>;
      const wave = entryRecord.wavesurfer;
      const trackId = hostTrackId || readString(key);
      const urls = collectAudioUrls(entry);
      collectAudioUrls(wave, urls);

      for (const url of urls) {
        mappings.set(url, {
          trackId: trackId || undefined,
          speakerKey: trackId || trackLabel || undefined,
          trackLabel: trackLabel || undefined,
          mappingSource: 'wavesurfer-react-registry'
        });
      }
    }
  }

  return mappings;
}

function postAudioSources(): void {
  const mappings = collectWaveformAudioMappings();
  for (const [url, mapping] of mappings.entries()) {
    window.postMessage(
      {
        type: AUDIO_SOURCE_MESSAGE_TYPE,
        url,
        ...mapping,
        mimeType: 'application/octet-stream',
        discoveredAt: Date.now()
      } satisfies AudioSourceMessage,
      '*'
    );
  }
}

function getTrackMapping(url: string): TrackMapping {
  const mappings = collectWaveformAudioMappings();
  return mappings.get(toAbsoluteUrl(url)) || {};
}

function enrichAudioRecord(record: StoredAudioResponse): StoredAudioResponse {
  if (record.speakerKey || record.trackId || record.trackLabel) {
    return record;
  }

  return {
    ...record,
    ...getTrackMapping(record.url)
  };
}

function rememberAndPost(record: StoredAudioResponse): void {
  const enriched = enrichAudioRecord(record);
  storedResponses.push(enriched);
  while (storedResponses.length > MAX_STORED_RESPONSES) {
    storedResponses.shift();
  }

  window.postMessage(
    {
      type: AUDIO_RESPONSE_MESSAGE_TYPE,
      ...enriched,
      bytes: enriched.bytes.slice(0)
    } satisfies AudioResponseMessage,
    '*'
  );
}

async function captureArrayBuffer(args: {
  url: string;
  mimeType: string;
  source: AudioInterceptSource;
  bytes: ArrayBuffer;
}): Promise<void> {
  if (!args.bytes.byteLength || args.bytes.byteLength > MAX_CAPTURE_BYTES) {
    return;
  }

  rememberAndPost({
    url: args.url,
    ...getTrackMapping(args.url),
    mimeType: args.mimeType || 'application/octet-stream',
    source: args.source,
    capturedAt: Date.now(),
    bytes: args.bytes.slice(0)
  });
}

function captureFetchResponse(response: Response, requestUrl: string): void {
  const url = response.url || requestUrl;
  const mimeType = response.headers.get('content-type') || '';
  if (!isAudioResponse(url, mimeType)) {
    return;
  }

  const contentLength = Number(response.headers.get('content-length') || 0);
  if (Number.isFinite(contentLength) && contentLength > MAX_CAPTURE_BYTES) {
    return;
  }

  void response
    .clone()
    .arrayBuffer()
    .then((bytes) => captureArrayBuffer({ url, mimeType, source: 'fetch', bytes }))
    .catch(() => undefined);
}

function installFetchInterceptor(): void {
  const originalFetch = window.fetch;
  if (typeof originalFetch !== 'function') {
    return;
  }

  window.fetch = function interceptedFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
    const requestUrl = getRequestUrl(input);
    return originalFetch.call(this, input, init).then((response) => {
      captureFetchResponse(response, requestUrl);
      return response;
    });
  };
}

function responseToArrayBuffer(response: XMLHttpRequest['response']): ArrayBuffer | null {
  if (response instanceof ArrayBuffer) {
    return response;
  }
  if (response instanceof Blob) {
    return null;
  }
  return null;
}

function installXhrInterceptor(): void {
  const originalOpen = XMLHttpRequest.prototype.open;
  const originalSend = XMLHttpRequest.prototype.send;
  const requestUrls = new WeakMap<XMLHttpRequest, string>();

  XMLHttpRequest.prototype.open = function interceptedOpen(
    method: string,
    url: string | URL,
    async?: boolean,
    username?: string | null,
    password?: string | null
  ): void {
    requestUrls.set(this, toAbsoluteUrl(String(url)));
    return originalOpen.call(this, method, url, async ?? true, username ?? undefined, password ?? undefined);
  };

  XMLHttpRequest.prototype.send = function interceptedSend(body?: Document | XMLHttpRequestBodyInit | null): void {
    this.addEventListener(
      'load',
      () => {
        const url = this.responseURL || requestUrls.get(this) || '';
        const mimeType = this.getResponseHeader('content-type') || '';
        if (!isAudioResponse(url, mimeType)) {
          return;
        }

        if (this.response instanceof Blob) {
          void this.response
            .arrayBuffer()
            .then((bytes) => captureArrayBuffer({ url, mimeType, source: 'xhr', bytes }))
            .catch(() => undefined);
          return;
        }

        const bytes = responseToArrayBuffer(this.response);
        if (bytes) {
          void captureArrayBuffer({ url, mimeType, source: 'xhr', bytes });
        }
      },
      { once: true }
    );
    return originalSend.call(this, body);
  };
}

function installFlushHandler(): void {
  window.addEventListener('message', (event) => {
    if (event.source !== window) {
      return;
    }
    if (event.data?.type === PAGE_TASK_ID_REQUEST_MESSAGE_TYPE && typeof event.data.requestId === 'string') {
      const reviewActionId = readCurrentReviewActionId();
      const root = document.documentElement;
      if (root) {
        if (reviewActionId) root.setAttribute(PAGE_TASK_ID_ATTRIBUTE, reviewActionId);
        else root.removeAttribute(PAGE_TASK_ID_ATTRIBUTE);
      }
      window.postMessage({
        type: PAGE_TASK_ID_RESPONSE_MESSAGE_TYPE,
        requestId: event.data.requestId,
        reviewActionId
      } satisfies PageTaskIdResponseMessage, '*');
      return;
    }
    if (!event.data || typeof event.data !== 'object' || event.data.type !== AUDIO_FLUSH_REQUEST_MESSAGE_TYPE) {
      return;
    }
    postAudioSources();
    for (let index = 0; index < storedResponses.length; index += 1) {
      const record = enrichAudioRecord(storedResponses[index]!);
      storedResponses[index] = record;
      window.postMessage(
        {
          type: AUDIO_RESPONSE_MESSAGE_TYPE,
          ...record,
          bytes: record.bytes.slice(0)
        } satisfies AudioResponseMessage,
        '*'
      );
    }
  });
}

function installAudioInterceptor(): void {
  if (window.__babelGoldDraftingAudioInterceptorInstalled) {
    postAudioSources();
    return;
  }

  window.__babelGoldDraftingAudioInterceptorInstalled = true;
  installFetchInterceptor();
  installXhrInterceptor();
  installFlushHandler();
}

window.addEventListener('message', (event) => {
  if (event.source !== window) {
    return;
  }
  if (!isAudioEnableCaptureMessage(event.data)) {
    return;
  }
  installAudioInterceptor();
});
