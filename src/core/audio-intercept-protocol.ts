export const AUDIO_RESPONSE_MESSAGE_TYPE = 'babel-gold-drafting:audio-response';
export const AUDIO_SOURCE_MESSAGE_TYPE = 'babel-gold-drafting:audio-source';
export const AUDIO_FLUSH_REQUEST_MESSAGE_TYPE = 'babel-gold-drafting:audio-flush-request';
export const AUDIO_ENABLE_CAPTURE_MESSAGE_TYPE = 'babel-gold-drafting:audio-enable-capture';

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
