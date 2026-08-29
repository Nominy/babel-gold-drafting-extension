export type L0TimingAvailability =
  | { taskId: string; status: 'available' }
  | { taskId: string; status: 'unavailable' }
  | { taskId: string; status: 'preparing' }
  | { taskId: string; status: 'queued'; position: number }
  | { taskId: string; status: 'running' }
  | { taskId: string; status: 'retrying' };

const listeners = new Set<(availability: L0TimingAvailability) => void>();
let currentAvailability: L0TimingAvailability | null = null;
let retryHandler: (() => boolean) | null = null;

export function getL0TimingAvailability(): L0TimingAvailability | null {
  return currentAvailability;
}

export function publishL0TimingAvailability(availability: L0TimingAvailability): void {
  if (
    currentAvailability?.taskId === availability.taskId
    && currentAvailability.status === availability.status
    && (
      availability.status !== 'queued'
      || (currentAvailability.status === 'queued' && currentAvailability.position === availability.position)
    )
  ) {
    return;
  }
  currentAvailability = availability;
  for (const listener of listeners) listener(availability);
}

export function subscribeL0TimingAvailability(
  listener: (availability: L0TimingAvailability) => void
): () => void {
  listeners.add(listener);
  if (currentAvailability) listener(currentAvailability);
  return () => listeners.delete(listener);
}

export function setL0TimingRetryHandler(handler: (() => boolean) | null): void {
  retryHandler = handler;
}

export function requestL0TimingRegeneration(): boolean {
  return retryHandler?.() === true;
}
