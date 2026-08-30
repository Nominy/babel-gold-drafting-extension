import { DraftingOverlayController } from './overlay';
import { publishGoldDraftingExtensionId, registerAiBrokerContentHandler } from './ai-broker-content';
import { registerLifecycle } from '../core/lifecycle';
import { enableL0TimingAudioCapture, registerL0TimingService } from './l0-timing-service';
import { refreshPageTaskIdentity } from './page-task-identity';
import { maybeShowLocalModelSuggestion } from './local-model-suggestion';

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
  publishGoldDraftingExtensionId();
  registerAiBrokerContentHandler();
  const timingService = registerL0TimingService();
  const controller = new DraftingOverlayController();
  controller.mount();
  void maybeShowLocalModelSuggestion();
  registerLifecycle(controller, () => {
    void refreshPageTaskIdentity().then(() => timingService.onLifecycleOpportunity());
  });
}


enableL0TimingAudioCapture();

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot, { once: true });
} else {
  boot();
}
