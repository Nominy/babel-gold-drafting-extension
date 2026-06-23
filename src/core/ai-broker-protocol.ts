import type {
  AiBrokerProvider,
  BrokerRedistributionGroup,
  BrokerRedistributeTextResult,
  BrokerTranscriptSegment
} from './types';

export const AI_BROKER_EXTERNAL_MESSAGE_TYPE = 'babel-gold-drafting:ai-broker';
export const AI_BROKER_INTERNAL_MESSAGE_TYPE = 'babel-gold-drafting:ai-broker-tab-request';
export const AI_BROKER_PORT_NAME = 'babel-gold-drafting:ai-broker-port';
export const AI_BROKER_INTERNAL_PORT_NAME = 'babel-gold-drafting:ai-broker-tab-port';
export const AI_BROKER_EXTENSION_ID_ATTR = 'data-babel-gold-drafting-extension-id';

export type AiBrokerOperation = 'ping' | 'transcribeSegment' | 'redistributeText';

export interface AiBrokerBaseRequest {
  type: typeof AI_BROKER_EXTERNAL_MESSAGE_TYPE;
  version: 1;
  operation: AiBrokerOperation;
  requestId?: string;
}

export interface AiBrokerPingRequest extends AiBrokerBaseRequest {
  operation: 'ping';
}

export interface AiBrokerTranscribeSegmentRequest extends AiBrokerBaseRequest {
  operation: 'transcribeSegment';
  segment: BrokerTranscriptSegment;
}

export interface AiBrokerRedistributeTextRequest extends AiBrokerBaseRequest {
  operation: 'redistributeText';
  groups: BrokerRedistributionGroup[];
}

export type AiBrokerExternalRequest =
  | AiBrokerPingRequest
  | AiBrokerTranscribeSegmentRequest
  | AiBrokerRedistributeTextRequest;

type AiBrokerInternalRequestFor<T extends AiBrokerExternalRequest> = Omit<T, 'type'> & {
  type: typeof AI_BROKER_INTERNAL_MESSAGE_TYPE;
};

export type AiBrokerInternalRequest =
  | AiBrokerInternalRequestFor<AiBrokerPingRequest>
  | AiBrokerInternalRequestFor<AiBrokerTranscribeSegmentRequest>
  | AiBrokerInternalRequestFor<AiBrokerRedistributeTextRequest>;

export interface AiBrokerPingResponse {
  ok: true;
  provider: AiBrokerProvider;
  remoteConfigured: boolean;
  capabilities: {
    transcribeSegment: boolean;
    redistributeText: boolean;
  };
}

export interface AiBrokerTranscribeSegmentResponse {
  ok: true;
  provider: 'remote-openrouter';
  text: string;
  model: string;
}

export interface AiBrokerRedistributeTextResponse {
  ok: true;
  provider: 'remote-openrouter';
  results: BrokerRedistributeTextResult[];
  model: string;
}

export interface AiBrokerUnavailableResponse {
  ok: false;
  reason:
    | 'invalid-request'
    | 'provider-local-gemini-nano'
    | 'remote-not-configured'
    | 'missing-tab'
    | 'tab-broker-unavailable'
    | 'broker-error';
  message?: string;
  fallbackAllowed: boolean;
}

export type AiBrokerResponse =
  | AiBrokerPingResponse
  | AiBrokerTranscribeSegmentResponse
  | AiBrokerRedistributeTextResponse
  | AiBrokerUnavailableResponse;

export interface AiBrokerPortEventMessage {
  type: 'event';
  event: 'capturing-audio' | 'calling-backend' | 'backend-waiting' | 'accepted';
  operation: AiBrokerOperation;
  message?: string;
  elapsedMs?: number;
}

export interface AiBrokerPortResultMessage {
  type: 'result';
  response: AiBrokerResponse;
}

export interface AiBrokerPortErrorMessage {
  type: 'error';
  response: AiBrokerUnavailableResponse;
}

export type AiBrokerPortMessage =
  | AiBrokerPortEventMessage
  | AiBrokerPortResultMessage
  | AiBrokerPortErrorMessage;

export function shouldUseRemoteBroker(provider: AiBrokerProvider): boolean {
  return provider === 'auto' || provider === 'remote-openrouter';
}

export function providerAllowsLocalFallback(provider: AiBrokerProvider): boolean {
  return provider !== 'remote-openrouter';
}
