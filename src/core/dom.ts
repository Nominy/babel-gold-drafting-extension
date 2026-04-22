import { getReactFiber, normalizeText, setEditableValue } from '@nominy/babel-babel-runtime';

export const TRANSCRIPT_ROW_SELECTOR = 'tbody tr';
export const ROW_TEXTAREA_SELECTOR = 'textarea[placeholder^="What was said"]';

export { getReactFiber, normalizeText };

export function parseTimeValue(value: string): number | null {
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }

  const match = trimmed.match(/-?\d+(?::\d+)+(?:\.\d+)?/);
  if (!match) {
    return null;
  }

  return match[0].split(':').reduce<number | null>((total, part) => {
    if (total === null) {
      return null;
    }

    const numeric = Number(part);
    return Number.isFinite(numeric) ? total * 60 + numeric : null;
  }, 0);
}

export function setControlledTextareaValue(textarea: HTMLTextAreaElement, value: string): void {
  setEditableValue(textarea, value);
  textarea.dispatchEvent(new Event('change', { bubbles: true, cancelable: false }));
}

export function getTranscriptRowElements(root: ParentNode = document): HTMLTableRowElement[] {
  return Array.from(root.querySelectorAll<HTMLTableRowElement>(TRANSCRIPT_ROW_SELECTOR)).filter((row) =>
    row.querySelector(ROW_TEXTAREA_SELECTOR)
  );
}
