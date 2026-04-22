import { DraftingOverlayController } from './overlay';

let controller: DraftingOverlayController | null = null;

function isTranscriptionRoute(): boolean {
  return /^\/transcription(?:\/|$)/.test(window.location.pathname || '');
}

function syncController(): void {
  if (isTranscriptionRoute()) {
    controller ??= new DraftingOverlayController();
    controller.mount();
    return;
  }

  controller?.unmount();
}

function patchHistoryMethod(methodName: 'pushState' | 'replaceState'): void {
  const original = window.history[methodName];
  if (typeof original !== 'function') {
    return;
  }

  window.history[methodName] = function patchedHistoryMethod(this: History, ...args: Parameters<History['pushState']>) {
    const result = original.apply(this, args);
    window.setTimeout(syncController, 0);
    return result;
  } as History['pushState'];
}

function boot(): void {
  syncController();
  patchHistoryMethod('pushState');
  patchHistoryMethod('replaceState');
  window.addEventListener('popstate', () => window.setTimeout(syncController, 0), true);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot, { once: true });
} else {
  boot();
}
