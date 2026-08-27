import { DraftingOverlayController } from './overlay';
import { publishGoldDraftingExtensionId, registerAiBrokerContentHandler } from './ai-broker-content';
import { registerLifecycle } from '../core/lifecycle';
import { loadSettings } from '../core/settings';
import { shouldUseRemoteBroker } from '../core/ai-broker-protocol';
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
  publishGoldDraftingExtensionId();
  registerAiBrokerContentHandler();
  const controller = new DraftingOverlayController();
  controller.mount();
  registerLifecycle(controller);
}

function enableAudioCaptureIfConfigured(): void {
  void loadSettings()
    .then((settings) => {
      if (
        settings.audioInputEnabled ||
        settings.l0ReplacementPreviewEnabled ||
        shouldUseRemoteBroker(settings.aiBrokerProvider)
      ) {
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
