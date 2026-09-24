export type VolunteerState = 'disabled' | 'connecting' | 'connected' | 'busy' | 'error';
export interface VolunteerStatus {
  state: VolunteerState;
  detail?: string;
}

export type VolunteerMessage =
  | { type: 'babel-l0-volunteer'; target: 'background'; action: 'status' | 'settings' }
  | { type: 'babel-l0-volunteer'; target: 'offscreen'; action: 'start' | 'stop' | 'status' };

export function isVolunteerMessage(value: unknown, target: VolunteerMessage['target']): value is VolunteerMessage {
  if (!value || typeof value !== 'object') return false;
  const message = value as Partial<VolunteerMessage>;
  return message.type === 'babel-l0-volunteer' && message.target === target &&
    (message.action === 'start' || message.action === 'stop' || message.action === 'status' ||
      message.action === 'settings') &&
    (target !== 'background' || message.action === 'status' || message.action === 'settings') &&
    (target !== 'offscreen' || message.action !== 'settings');
}
