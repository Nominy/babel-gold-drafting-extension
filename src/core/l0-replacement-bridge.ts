import type { L0DraftRow } from './types';

export const L0_REPLACE_REQUEST_TYPE = 'babel-gold-drafting:l0-replace-request';
export const L0_REPLACE_RESPONSE_TYPE = 'babel-gold-drafting:l0-replace-response';
export const L0_REPLACE_READY_REQUEST_TYPE = 'babel-gold-drafting:l0-replace-ready-request';
export const L0_REPLACE_READY_RESPONSE_TYPE = 'babel-gold-drafting:l0-replace-ready-response';
const L0_REPLACE_PROTOCOL_VERSION = 1;
const L0_REPLACE_TIMEOUT_MS = 300000;
const L0_REPLACE_READY_TIMEOUT_MS = 5000;

export interface L0CreatedRowMapping {
  id: string;
  annotationId: string;
  lane: string;
  startSeconds: number;
  endSeconds: number;
}

type L0ReplaceResponse =
  | {
      type: typeof L0_REPLACE_RESPONSE_TYPE;
      version: 1;
      requestId: string;
      ok: true;
      created: L0CreatedRowMapping[];
    }
  | {
      type: typeof L0_REPLACE_RESPONSE_TYPE;
      version: 1;
      requestId: string;
      ok: false;
      reason: string;
      message: string;
    };

function isCreatedMapping(value: unknown): value is L0CreatedRowMapping {
  return Boolean(
    value &&
      typeof value === 'object' &&
      'id' in value &&
      typeof value.id === 'string' &&
      'annotationId' in value &&
      typeof value.annotationId === 'string' &&
      'lane' in value &&
      typeof value.lane === 'string' &&
      'startSeconds' in value &&
      typeof value.startSeconds === 'number' &&
      'endSeconds' in value &&
      typeof value.endSeconds === 'number'
  );
}

function isReplaceResponse(value: unknown, requestId: string): value is L0ReplaceResponse {
  if (
    !value ||
    typeof value !== 'object' ||
    !('type' in value) ||
    value.type !== L0_REPLACE_RESPONSE_TYPE ||
    !('version' in value) ||
    value.version !== L0_REPLACE_PROTOCOL_VERSION ||
    !('requestId' in value) ||
    value.requestId !== requestId ||
    !('ok' in value) ||
    typeof value.ok !== 'boolean'
  ) {
    return false;
  }
  if (value.ok) {
    return 'created' in value && Array.isArray(value.created) && value.created.every(isCreatedMapping);
  }
  return (
    'reason' in value &&
    typeof value.reason === 'string' &&
    'message' in value &&
    typeof value.message === 'string'
  );
}

export function requireL0ReplacementConsumer(): Promise<void> {
  const requestId = crypto.randomUUID();
  const { promise, resolve, reject } = Promise.withResolvers<void>();
  const cleanup = (): void => {
    window.clearTimeout(timeoutId);
    window.removeEventListener('message', onMessage);
  };
  const onMessage = (event: MessageEvent): void => {
    const data = event.data;
    if (
      event.source !== window
      || !data || typeof data !== 'object'
      || data.type !== L0_REPLACE_READY_RESPONSE_TYPE
      || data.version !== L0_REPLACE_PROTOCOL_VERSION
      || data.requestId !== requestId
    ) return;
    cleanup();
    resolve();
  };
  const timeoutId = window.setTimeout(() => {
    cleanup();
    reject(new Error('Babel Helper is required for L0 transcript replacement. Enable Babel Helper and reload this task.'));
  }, L0_REPLACE_READY_TIMEOUT_MS);
  window.addEventListener('message', onMessage);
  window.postMessage({
    type: L0_REPLACE_READY_REQUEST_TYPE,
    version: L0_REPLACE_PROTOCOL_VERSION,
    requestId
  }, '*');
  return promise;
}

export function replaceTranscriptWithL0Rows(rows: L0DraftRow[]): Promise<L0CreatedRowMapping[]> {
  if (!rows.length) {
    throw new Error('L0 replacement requires at least one returned row.');
  }
  const requestId =
    typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
      ? crypto.randomUUID()
      : `l0-${Date.now()}-${Math.random().toString(36).slice(2)}`;

  return new Promise((resolve, reject) => {
    const cleanup = (): void => {
      window.clearTimeout(timeoutId);
      window.removeEventListener('message', onMessage);
    };
    const onMessage = (event: MessageEvent): void => {
      if ((event.source && event.source !== window) || !isReplaceResponse(event.data, requestId)) {
        return;
      }
      cleanup();
      if (!event.data.ok) {
        reject(new Error(`L0 transcript replacement failed (${event.data.reason}): ${event.data.message}`));
        return;
      }
      if (event.data.created.length !== rows.length) {
        reject(new Error(`L0 transcript replacement created ${event.data.created.length} of ${rows.length} rows.`));
        return;
      }
      resolve(event.data.created);
    };
    const timeoutId = window.setTimeout(() => {
      cleanup();
      reject(new Error('Timed out waiting for Babel Helper to replace transcript segments.'));
    }, L0_REPLACE_TIMEOUT_MS);

    window.addEventListener('message', onMessage);
    window.postMessage(
      {
        type: L0_REPLACE_REQUEST_TYPE,
        version: L0_REPLACE_PROTOCOL_VERSION,
        requestId,
        rows: rows.map(({ id, lane, startSeconds, endSeconds, text }) => ({
          id,
          lane,
          startSeconds,
          endSeconds,
          text
        }))
      },
      '*'
    );
  });
}
