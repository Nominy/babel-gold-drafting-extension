import type {
  GenerateDraftErrorEvent,
  GenerateDraftRequest,
  GenerateDraftResponse,
  GenerateDraftRowEvent,
  GenerateDraftStartedEvent
} from './types';

function getEndpointUrl(backendBaseUrl: string, path: string): string {
  return `${backendBaseUrl.replace(/\/+$/, '')}${path}`;
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

export async function generateDraft(
  backendBaseUrl: string,
  payload: GenerateDraftRequest
): Promise<GenerateDraftResponse> {
  const response = await fetch(getEndpointUrl(backendBaseUrl, '/api/draft/generate'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload)
  });

  const data = await parseJsonResponse(response);

  if (!response.ok) {
    throw new Error(getErrorMessage(response.status, data));
  }

  if (!data || typeof data !== 'object') {
    throw new Error('Draft backend returned non-JSON payload.');
  }

  return data as GenerateDraftResponse;
}

export async function generateDraftStream(
  backendBaseUrl: string,
  payload: GenerateDraftRequest,
  handlers: {
    onStarted?: (event: GenerateDraftStartedEvent) => void;
    onRow?: (event: GenerateDraftRowEvent) => void;
    onDone?: (response: GenerateDraftResponse) => void;
  }
): Promise<GenerateDraftResponse> {
  const response = await fetch(getEndpointUrl(backendBaseUrl, '/api/draft/generate/stream'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream'
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    throw new Error(getErrorMessage(response.status, await parseJsonResponse(response)));
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
      throw new Error(errorPayload.error || 'Draft stream failed.');
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
