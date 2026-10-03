import {
  isL0TimingAccessToken,
  isL0TimingTokenRequest,
  type L0TimingTokenResponse
} from '../core/l0-timing-token-protocol';

interface TimingTokenStorage {
  get: (key: string) => Promise<Record<string, unknown>>;
  set: (items: Record<string, string>) => Promise<void>;
}

export type L0TimingTokenHandler = (
  message: unknown,
  sender: Pick<chrome.runtime.MessageSender, 'id'>,
  sendResponse: (response: L0TimingTokenResponse) => void
) => boolean;
export function createL0TimingTokenHandler(storage: TimingTokenStorage, extensionId: string): L0TimingTokenHandler {
  // Only allocations in progress live in memory; persisted credentials survive worker restarts.
  const allocations = new Map<string, Promise<string>>();

  function getOrCreate(taskId: string): Promise<string> {
    const existing = allocations.get(taskId);
    if (existing) return existing;
    const allocation = (async () => {
      const key = `babel-gold-drafting:l0-timing-token:${taskId}`;
      const stored = await storage.get(key);
      if (isL0TimingAccessToken(stored[key])) return stored[key];
      const bytes = crypto.getRandomValues(new Uint8Array(32));
      const token = btoa(String.fromCharCode(...bytes)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
      await storage.set({ [key]: token });
      return token;
    })();
    allocations.set(taskId, allocation);
    void allocation.then(
      () => allocations.delete(taskId),
      () => allocations.delete(taskId)
    );
    return allocation;
  }

  return (
    message: unknown,
    sender: Pick<chrome.runtime.MessageSender, 'id'>,
    sendResponse: (response: L0TimingTokenResponse) => void
  ): boolean => {
    if (sender.id !== extensionId || !isL0TimingTokenRequest(message)) return false;
    const envelope = { type: message.type, version: message.version, taskId: message.taskId };
    void getOrCreate(message.taskId).then(
      (token) => sendResponse({ ...envelope, ok: true, token }),
      (error: unknown) => sendResponse({
        ...envelope,
        ok: false,
        error: error instanceof Error ? error.message : String(error)
      })
    );
    return true;
  };
}

const runtime = globalThis.chrome?.runtime;
if (runtime?.onMessage && globalThis.chrome?.storage?.local) {
  runtime.onMessage.addListener(createL0TimingTokenHandler(chrome.storage.local, runtime.id));
}
