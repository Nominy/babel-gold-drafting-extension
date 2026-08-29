export const AUDIO_RESPONSE_MESSAGE_TYPE = 'babel-gold-drafting:audio-response';
export const AUDIO_SOURCE_MESSAGE_TYPE = 'babel-gold-drafting:audio-source';
export const AUDIO_FLUSH_REQUEST_MESSAGE_TYPE = 'babel-gold-drafting:audio-flush-request';
export const AUDIO_ENABLE_CAPTURE_MESSAGE_TYPE = 'babel-gold-drafting:audio-enable-capture';
export const PAGE_TASK_ID_ATTRIBUTE = 'data-babel-review-action-id';
export const PAGE_TASK_ID_REQUEST_MESSAGE_TYPE = 'babel-gold-drafting:page-task-id-request';
export const PAGE_TASK_ID_RESPONSE_MESSAGE_TYPE = 'babel-gold-drafting:page-task-id-response';

export type AudioInterceptSource = 'fetch' | 'xhr';

export interface AudioResponseMessage {
  type: typeof AUDIO_RESPONSE_MESSAGE_TYPE;
  url: string;
  trackId?: string;
  speakerKey?: string;
  trackLabel?: string;
  mappingSource?: string;
  mimeType: string;
  source: AudioInterceptSource;
  capturedAt: number;
  bytes: ArrayBuffer;
}

export interface AudioSourceMessage {
  type: typeof AUDIO_SOURCE_MESSAGE_TYPE;
  url: string;
  trackId?: string;
  speakerKey?: string;
  trackLabel?: string;
  mappingSource?: string;
  mimeType?: string;
  discoveredAt: number;
}

export interface AudioEnableCaptureMessage {
  type: typeof AUDIO_ENABLE_CAPTURE_MESSAGE_TYPE;
}

export interface PageTaskIdRequestMessage {
  type: typeof PAGE_TASK_ID_REQUEST_MESSAGE_TYPE;
  requestId: string;
}

export interface PageTaskIdResponseMessage {
  type: typeof PAGE_TASK_ID_RESPONSE_MESSAGE_TYPE;
  requestId: string;
  reviewActionId: string;
}
