import { createJsonClient, normalizeBaseUrl } from '@nominy/babel-extension-frontend';
import type {
  GenerateDraftErrorEvent,
  GenerateDraftRequest,
  GenerateDraftResponse,
  GenerateDraftRowEvent,
  GenerateDraftStartedEvent,
  CapturedAudioTrack
} from './types';

class DraftStreamHttpError extends Error {}

class DraftStreamServerError extends Error {}

const RECONCILE_RETRY_DELAYS_MS = [100, 300, 700, 1500, 3000, 5000, 10000, 20000];

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

async function parseGenerateDraftResponse(response: Response): Promise<GenerateDraftResponse> {
  const payload = await parseJsonResponse(response);
  if (!response.ok) {
    throw new Error(getErrorMessage(response.status, payload));
  }
  if (!payload || typeof payload !== 'object') {
    throw new Error('Backend returned non-JSON payload.');
  }
  return payload as GenerateDraftResponse;
}

export async function generateDraft(
  backendBaseUrl: string,
  payload: GenerateDraftRequest
): Promise<GenerateDraftResponse> {
  const client = createJsonClient({
    getBaseCandidates: () => [normalizeBaseUrl(backendBaseUrl)]
  });
  return client.post<GenerateDraftResponse>('/api/draft/generate', payload);
}

async function reconcileDraftWithFormPayload(
  backendBaseUrl: string,
  payload: GenerateDraftRequest,
  audioTracks: CapturedAudioTrack[] = []
): Promise<GenerateDraftResponse> {
  const body = createGenerateDraftFormData(payload, audioTracks);

  const response = await fetch(getEndpointUrl(backendBaseUrl, '/api/draft/generate'), {
    method: 'POST',
    headers: {
      Accept: 'application/json'
    },
    body
  });

  return parseGenerateDraftResponse(response);
}

function normalizeError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
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
  audioTracks: CapturedAudioTrack[] = []
): Promise<GenerateDraftResponse> {
  let lastError = streamError;
  handlers.onReconnect?.(streamError);

  for (let attempt = 0; attempt <= RECONCILE_RETRY_DELAYS_MS.length; attempt += 1) {
    if (attempt > 0) {
      await delay(RECONCILE_RETRY_DELAYS_MS[attempt - 1]);
      handlers.onReconnect?.(lastError);
    }

    try {
      return await reconcileDraftWithFormPayload(backendBaseUrl, payload, audioTracks);
    } catch (error) {
      const normalizedError = normalizeError(error);
      if (!isReconnectableGenerateError(normalizedError)) {
        throw normalizedError;
      }
      lastError = normalizedError;
    }
  }

  throw lastError;
}

function createGenerateDraftFormData(
  payload: GenerateDraftRequest,
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

async function generateDraftStreamCore(
  backendBaseUrl: string,
  payload: GenerateDraftRequest,
  handlers: GenerateDraftStreamHandlers,
  audioTracks: CapturedAudioTrack[] = []
): Promise<GenerateDraftResponse> {
  const hasAudioTracks = audioTracks.length > 0;
  const body = hasAudioTracks ? createGenerateDraftFormData(payload, audioTracks) : JSON.stringify(payload);
  const headers: Record<string, string> = {
    Accept: 'text/event-stream'
  };

  if (!hasAudioTracks) {
    headers['Content-Type'] = 'application/json';
  }

  const response = await fetch(getEndpointUrl(backendBaseUrl, '/api/draft/generate/stream'), {
    method: 'POST',
    headers,
    body
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

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });

    let separatorIndex = buffer.indexOf('\n\n');
    while (separatorIndex >= 0) {
      const block = buffer.slice(0, separatorIndex).trim();
      buffer = buffer.slice(separatorIndex + 2);
      if (block) {
        processEventBlock(block);
      }
      separatorIndex = buffer.indexOf('\n\n');
    }

    if (done) {
      break;
    }
  }

  const trailing = buffer.trim();
  if (trailing) {
    processEventBlock(trailing);
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
  audioTracks: CapturedAudioTrack[] = []
): Promise<GenerateDraftResponse> {
  try {
    return await generateDraftStreamCore(backendBaseUrl, payload, handlers, audioTracks);
  } catch (error) {
    if (error instanceof DraftStreamHttpError || error instanceof DraftStreamServerError || !payload.draftSessionId) {
      throw error;
    }

    const normalizedError = normalizeError(error);
    return reconcileDraftSession(backendBaseUrl, payload, handlers, normalizedError, audioTracks);
  }
}
