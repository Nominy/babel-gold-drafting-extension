import { DraftingOverlayController } from './overlay';
import { registerLifecycle } from '../core/lifecycle';

declare global {
  interface Window {
    __babelGoldDraftingInstalled?: boolean;
  }
}

function boot(): void {
  if (window.__babelGoldDraftingInstalled) {
    return;
  }

  window.__babelGoldDraftingInstalled = true;
  const controller = new DraftingOverlayController();
  controller.mount();
  registerLifecycle(controller);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot, { once: true });
} else {
  boot();
}
