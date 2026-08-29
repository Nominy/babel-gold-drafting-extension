import { registerDomLifecycle } from '@nominy/babel-babel-runtime';
import type { DraftingMountController } from './types';

export function registerLifecycle(
  controller: DraftingMountController,
  onDomChange: () => void
): void {
  registerDomLifecycle(() => {
    controller.ensureMagicButton();
    onDomChange();
  });
}
