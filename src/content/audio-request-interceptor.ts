import { readBabelEditorState } from '@nominy/babel-babel-runtime';
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

function readCurrentReviewActionId(): string {
  return readBabelEditorState()?.reviewActionId || '';
}

function publishReviewActionId(reviewActionId: string): void {
  const root = document.documentElement;
  if (reviewActionId) root.setAttribute(PAGE_TASK_ID_ATTRIBUTE, reviewActionId);
  else root.removeAttribute(PAGE_TASK_ID_ATTRIBUTE);
}

function collectEditorAudioMappings(): Map<string, TrackMapping> {
  const mappings = new Map<string, TrackMapping>();
  for (const track of readBabelEditorState()?.tracks || []) {
    if (!track.audioUrl) continue;
    mappings.set(toAbsoluteUrl(track.audioUrl), {
      trackId: track.id, speakerKey: track.id, trackLabel: track.label,
      mappingSource: 'react-editor-recordings'
    });
  }
  return mappings;
}

function postAudioSources(mappings = collectEditorAudioMappings()): void {
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

// Called once when the bytes are captured and again on every flush, because a
// lane can be registered after its audio was fetched. Presigned URLs rotate on
// live, so a URL miss says nothing about the lane: keep the mapping that
// was recorded at capture time.
function enrichAudioRecord(record: StoredAudioResponse, mappings = collectEditorAudioMappings()): StoredAudioResponse {
  const mapping = mappings.get(toAbsoluteUrl(record.url));
  return mapping ? { ...record, ...mapping } : record;
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
    mimeType: args.mimeType || 'application/octet-stream',
    source: args.source,
    capturedAt: Date.now(),
    bytes: args.bytes.slice(0)
  });
}

function captureFetchResponse(response: Response, requestUrl: string): void {
  const url = response.url || requestUrl;
  const mimeType = response.headers.get('content-type') || '';
  if (!response.ok || !isAudioResponse(url, mimeType)) {
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
        if (this.status < 200 || this.status >= 300 || !isAudioResponse(url, mimeType)) {
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
      publishReviewActionId(reviewActionId);
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
    const mappings = collectEditorAudioMappings();
    postAudioSources(mappings);
    for (let index = 0; index < storedResponses.length; index += 1) {
      const record = enrichAudioRecord(storedResponses[index]!, mappings);
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
