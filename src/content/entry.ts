import { DraftingOverlayController } from './overlay';
import { registerLifecycle } from '../core/lifecycle';
import { loadSettings } from '../core/settings';
import { AUDIO_ENABLE_CAPTURE_MESSAGE_TYPE } from '../core/audio-intercept-protocol';

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

function enableAudioCaptureIfConfigured(): void {
  void loadSettings()
    .then((settings) => {
      if (settings.audioInputEnabled) {
        window.postMessage({ type: AUDIO_ENABLE_CAPTURE_MESSAGE_TYPE }, '*');
      }
    })
    .catch(() => undefined);
}

enableAudioCaptureIfConfigured();

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot, { once: true });
} else {
  boot();
}
