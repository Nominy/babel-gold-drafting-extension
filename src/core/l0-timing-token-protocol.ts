export const L0_TIMING_TOKEN_MESSAGE_TYPE = 'babel-gold-drafting:l0-timing-token';
export const L0_TIMING_TOKEN_VERSION = 1 as const;

export interface L0TimingTokenRequest {
  type: typeof L0_TIMING_TOKEN_MESSAGE_TYPE;
  version: typeof L0_TIMING_TOKEN_VERSION;
  taskId: string;
}

export type L0TimingTokenResponse = L0TimingTokenRequest & (
  | { ok: true; token: string }
  | { ok: false; error: string }
);

export function isL0TimingAccessToken(value: unknown): value is string {
  return typeof value === 'string' && /^[A-Za-z0-9_-]{43}$/.test(value);
}

export function isL0TimingTokenRequest(value: unknown): value is L0TimingTokenRequest {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  return 'type' in value && value.type === L0_TIMING_TOKEN_MESSAGE_TYPE &&
    'version' in value && value.version === L0_TIMING_TOKEN_VERSION &&
    'taskId' in value && typeof value.taskId === 'string' && Boolean(value.taskId.trim());
}

export function isL0TimingTokenResponse(
  value: unknown,
  request: L0TimingTokenRequest
): value is L0TimingTokenResponse {
  if (!isL0TimingTokenRequest(value) || value.taskId !== request.taskId || !('ok' in value)) return false;
  return value.ok === true
    ? 'token' in value && isL0TimingAccessToken(value.token)
    : value.ok === false && 'error' in value && typeof value.error === 'string' && Boolean(value.error);
}
