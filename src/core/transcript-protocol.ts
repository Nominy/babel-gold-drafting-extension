import type { ApplyDraftResult, DraftRowResult, TranscriptJob } from './types';

export const TRANSCRIPT_REQUEST_EVENT = 'babel-gold-drafting:transcript-request';
export const TRANSCRIPT_RESPONSE_EVENT = 'babel-gold-drafting:transcript-response';
export type TranscriptRequest = {
  version: 1;
  requestId: string;
} & ({ operation: 'capture' } | {
  operation: 'apply';
  expected: TranscriptJob;
  rows: DraftRowResult[];
});
export type TranscriptResponse = {
  version: 1;
  requestId: string;
} & ({ ok: true; job: TranscriptJob } | { ok: true; result: ApplyDraftResult } | { ok: false; error: string });

export function parseTranscriptEvent(event: Event): unknown {
  const detail = (event as CustomEvent<unknown>).detail;
  if (typeof detail !== 'string') return null;
  try { return JSON.parse(detail); } catch { return null; }
}

export function dispatchTranscriptEvent(documentRef: Document, type: string, payload: TranscriptRequest | TranscriptResponse): void {
  const event = documentRef.createEvent('CustomEvent');
  // A string crosses Chrome's isolated/MAIN world boundary without exporting
  // React callbacks or depending on shared JavaScript object identity.
  event.initCustomEvent(type, false, false, JSON.stringify(payload));
  documentRef.dispatchEvent(event);
}
