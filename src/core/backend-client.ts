import { normalizeBaseUrl } from '@nominy/babel-extension-frontend';
import type {
  GenerateDraftErrorEvent,
  GenerateDraftRequest,
  GenerateDraftResponse,
  GenerateDraftRowEvent,
  GenerateDraftStartedEvent,
  CapturedAudioTrack,
  BrokerTranscribeSegmentRequest,
  BrokerTranscribeSegmentResponse,
  BrokerRedistributeTextRequest,
  BrokerRedistributeTextResponse
} from './types';

class DraftStreamHttpError extends Error {}

class DraftStreamServerError extends Error {}

const RECONCILE_RETRY_DELAYS_MS = [100, 300, 700, 1500, 3000, 5000, 10000, 20000];
const BROKER_TRANSCRIBE_TIMEOUT_MS = 300000;
const BROKER_REDISTRIBUTE_TIMEOUT_MS = 120000;

type BrokerRequestOptions = {
  timeoutMs?: number;
};

type GenerateDraftStreamHandlers = {
  onStarted?: (event: GenerateDraftStartedEvent) => void;
  onRow?: (event: GenerateDraftRowEvent) => void;
  onDone?: (response: GenerateDraftResponse) => void;
  onReconnect?: (error: Error) => void;
};

function getEndpointUrl(backendBaseUrl: string, path: string): string {
  return `${normalizeBaseUrl(backendBaseUrl)}${path}`;
}

async function parseJsonResponse(response: Response): Promise<unknown> {
  const text = await response.text();

  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function getErrorMessage(status: number, payload: unknown): string {
  if (payload && typeof payload === 'object' && 'error' in payload && typeof payload.error === 'string') {
    return payload.error;
  }
  if (typeof payload === 'string') {
    return `HTTP ${status}: ${payload.slice(0, 240)}`;
  }
  return `HTTP ${status}`;
}

function createRequestFormData(
  payload: GenerateDraftRequest | BrokerTranscribeSegmentRequest | BrokerRedistributeTextRequest,
  audioTracks: CapturedAudioTrack[] = []
): FormData {
  const body = new FormData();
  body.set('payload', JSON.stringify(payload));
  for (const track of audioTracks) {
    const extension = track.mimeType.includes('wav') ? 'wav' : track.mimeType.includes('mpeg') ? 'mp3' : 'bin';
    body.append(`audioTrack:${track.trackId}`, track.blob, `${track.trackId}.${extension}`);
    body.set(
      `audioTrackMeta:${track.trackId}`,
      JSON.stringify({
        source: track.source,
        speakerKey: track.speakerKey || '',
        trackLabel: track.trackLabel || '',
        mimeType: track.mimeType
      })
    );
  }
  return body;
}

async function parseObjectResponse<T>(response: Response): Promise<T> {
  const payload = await parseJsonResponse(response);
  if (!response.ok) {
    throw new Error(getErrorMessage(response.status, payload));
  }
  if (!payload || typeof payload !== 'object') {
    throw new Error('Backend returned non-JSON payload.');
  }
  return payload as T;
}

function getBrokerRequestTimeoutMs(options: BrokerRequestOptions | undefined, fallbackMs: number): number {
  const timeoutMs = Number(options?.timeoutMs);
  return Number.isFinite(timeoutMs) && timeoutMs > 0 ? timeoutMs : fallbackMs;
}

async function fetchBrokerJsonResponse<T>(
  backendBaseUrl: string,
  path: string,
  payload: BrokerTranscribeSegmentRequest | BrokerRedistributeTextRequest,
  audioTracks: CapturedAudioTrack[],
  timeoutMs: number
): Promise<T> {
  const endpointUrl = getEndpointUrl(backendBaseUrl, path);
  const controller = new AbortController();
  const timeoutId = setTimeout(() => {
    controller.abort(new Error(`Gold Drafting broker request to ${endpointUrl} timed out after ${timeoutMs}ms.`));
  }, timeoutMs);

  try {
    const response = await fetch(endpointUrl, {
      method: 'POST',
      headers: {
        Accept: 'application/json'
      },
      body: createRequestFormData(payload, audioTracks),
      signal: controller.signal
    });

    return await parseObjectResponse<T>(response);
  } catch (error) {
    if (controller.signal.aborted) {
      const reason = (controller.signal as AbortSignal & { reason?: unknown }).reason;
      throw reason instanceof Error
        ? reason
        : new Error(`Gold Drafting broker request to ${endpointUrl} timed out after ${timeoutMs}ms.`);
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}

export async function transcribeSegmentWithBroker(
  backendBaseUrl: string,
  payload: BrokerTranscribeSegmentRequest,
  audioTracks: CapturedAudioTrack[] = [],
  options: BrokerRequestOptions = {}
): Promise<BrokerTranscribeSegmentResponse> {
  return fetchBrokerJsonResponse<BrokerTranscribeSegmentResponse>(
    backendBaseUrl,
    '/api/broker/transcribe-segment',
    payload,
    audioTracks,
    getBrokerRequestTimeoutMs(options, BROKER_TRANSCRIBE_TIMEOUT_MS)
  );
}

export async function redistributeTextWithBroker(
  backendBaseUrl: string,
  payload: BrokerRedistributeTextRequest,
  audioTracks: CapturedAudioTrack[] = [],
  options: BrokerRequestOptions = {}
): Promise<BrokerRedistributeTextResponse> {
  return fetchBrokerJsonResponse<BrokerRedistributeTextResponse>(
    backendBaseUrl,
    '/api/broker/redistribute-text',
    payload,
    audioTracks,
    getBrokerRequestTimeoutMs(options, BROKER_REDISTRIBUTE_TIMEOUT_MS)
  );
}

async function reconcileDraftWithFormPayload(
  backendBaseUrl: string,
  payload: GenerateDraftRequest,
  audioTracks: CapturedAudioTrack[] = [],
  signal?: AbortSignal
): Promise<GenerateDraftResponse> {
  const body = createRequestFormData(payload, audioTracks);

  const response = await fetch(getEndpointUrl(backendBaseUrl, '/api/draft/generate'), {
    method: 'POST',
    headers: {
      Accept: 'application/json'
    },
    body,
    signal
  });

  return parseObjectResponse<GenerateDraftResponse>(response);
}

function normalizeError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

function delay(ms: number, signal?: AbortSignal): Promise<void> {
  signal?.throwIfAborted();
  const { promise, resolve, reject } = Promise.withResolvers<void>();
  const abort = (): void => {
    clearTimeout(timeout);
    reject(signal?.reason ?? new DOMException('Draft cancelled', 'AbortError'));
  };
  const timeout = setTimeout(() => {
    signal?.removeEventListener('abort', abort);
    resolve();
  }, ms);
  signal?.addEventListener('abort', abort, { once: true });
  return promise;
}

function isReconnectableGenerateError(error: Error): boolean {
  const message = error.message.toLowerCase();
  return (
    message.includes('could not reach backend') ||
    message.includes('failed to fetch') ||
    message.includes('networkerror') ||
    message.includes('load failed')
  );
}

async function reconcileDraftSession(
  backendBaseUrl: string,
  payload: GenerateDraftRequest,
  handlers: GenerateDraftStreamHandlers,
  streamError: Error,
  audioTracks: CapturedAudioTrack[] = [],
  signal?: AbortSignal
): Promise<GenerateDraftResponse> {
  let lastError = streamError;
  signal?.throwIfAborted();
  handlers.onReconnect?.(streamError);

  for (let attempt = 0; attempt <= RECONCILE_RETRY_DELAYS_MS.length; attempt += 1) {
    if (attempt > 0) {
      await delay(RECONCILE_RETRY_DELAYS_MS[attempt - 1], signal);
      handlers.onReconnect?.(lastError);
    }

    try {
      signal?.throwIfAborted();
      const response = await reconcileDraftWithFormPayload(backendBaseUrl, payload, audioTracks, signal);
      signal?.throwIfAborted();
      return response;
    } catch (error) {
      signal?.throwIfAborted();
      const normalizedError = normalizeError(error);
      if (!isReconnectableGenerateError(normalizedError)) {
        throw normalizedError;
      }
      lastError = normalizedError;
    }
  }

  throw lastError;
}

async function generateDraftStreamCore(
  backendBaseUrl: string,
  payload: GenerateDraftRequest,
  handlers: GenerateDraftStreamHandlers,
  audioTracks: CapturedAudioTrack[] = [],
  signal?: AbortSignal
): Promise<GenerateDraftResponse> {
  signal?.throwIfAborted();
  const hasAudioTracks = audioTracks.length > 0;
  const body = hasAudioTracks ? createRequestFormData(payload, audioTracks) : JSON.stringify(payload);
  const headers: Record<string, string> = {
    Accept: 'text/event-stream'
  };

  if (!hasAudioTracks) {
    headers['Content-Type'] = 'application/json';
  }

  const response = await fetch(getEndpointUrl(backendBaseUrl, '/api/draft/generate/stream'), {
    method: 'POST',
    headers,
    body,
    signal
  });

  if (!response.ok) {
    throw new DraftStreamHttpError(getErrorMessage(response.status, await parseJsonResponse(response)));
  }

  if (!response.body) {
    throw new Error('Draft backend did not return a stream body.');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let finalResponse: GenerateDraftResponse | null = null;

  const processEventBlock = (block: string): void => {
    signal?.throwIfAborted();
    const lines = block.split(/\r?\n/);
    let eventName = 'message';
    const dataLines: string[] = [];

    for (const line of lines) {
      if (!line) {
        continue;
      }
      if (line.startsWith('event:')) {
        eventName = line.slice('event:'.length).trim();
        continue;
      }
      if (line.startsWith('data:')) {
        dataLines.push(line.slice('data:'.length).trimStart());
      }
    }

    if (!dataLines.length) {
      return;
    }

    const payloadText = dataLines.join('\n');
    const parsed = JSON.parse(payloadText) as unknown;

    if (eventName === 'started') {
      handlers.onStarted?.(parsed as GenerateDraftStartedEvent);
      return;
    }

    if (eventName === 'row') {
      handlers.onRow?.(parsed as GenerateDraftRowEvent);
      return;
    }

    if (eventName === 'done') {
      finalResponse = parsed as GenerateDraftResponse;
      handlers.onDone?.(finalResponse);
      return;
    }

    if (eventName === 'error') {
      const errorPayload = parsed as GenerateDraftErrorEvent;
      throw new DraftStreamServerError(errorPayload.error || 'Draft stream failed.');
    }
  };

  const abort = (): void => { void reader.cancel().catch(() => undefined); };
  signal?.addEventListener('abort', abort, { once: true });
  try {
    while (true) {
      signal?.throwIfAborted();
      const { value, done } = await reader.read();
      signal?.throwIfAborted();
      buffer += decoder.decode(value, { stream: !done });
      let separator = /\r?\n\r?\n/.exec(buffer);
      while (separator) {
        const block = buffer.slice(0, separator.index).trim();
        buffer = buffer.slice(separator.index + separator[0].length);
        if (block) processEventBlock(block);
        separator = /\r?\n\r?\n/.exec(buffer);
      }
      if (done) break;
    }
    if (buffer.trim()) processEventBlock(buffer.trim());
  } finally {
    signal?.removeEventListener('abort', abort);
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }

  if (!finalResponse) {
    throw new Error('Draft stream finished without a final response.');
  }

  return finalResponse;
}

export async function generateDraftStream(
  backendBaseUrl: string,
  payload: GenerateDraftRequest,
  handlers: GenerateDraftStreamHandlers,
  audioTracks: CapturedAudioTrack[] = [],
  signal?: AbortSignal
): Promise<GenerateDraftResponse> {
  try {
    return await generateDraftStreamCore(backendBaseUrl, payload, handlers, audioTracks, signal);
  } catch (error) {
    signal?.throwIfAborted();
    if (error instanceof DraftStreamHttpError || error instanceof DraftStreamServerError || !payload.draftSessionId) {
      throw error;
    }

    const normalizedError = normalizeError(error);
    return reconcileDraftSession(backendBaseUrl, payload, handlers, normalizedError, audioTracks, signal);
  }
}
