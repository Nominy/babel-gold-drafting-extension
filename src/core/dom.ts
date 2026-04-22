export const TRANSCRIPT_ROW_SELECTOR = 'tbody tr';
export const ROW_TEXTAREA_SELECTOR = 'textarea[placeholder^="What was said"]';

export function normalizeText(element: Element | null | undefined): string {
  if (!(element instanceof HTMLElement)) {
    return '';
  }

  const rawText = typeof element.innerText === 'string' ? element.innerText : element.textContent || '';
  return rawText.replace(/\s+/g, ' ').trim();
}

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

export function getReactFiber(element: HTMLElement | null): unknown {
  if (!(element instanceof HTMLElement)) {
    return null;
  }

  for (const name of Object.getOwnPropertyNames(element)) {
    if (name.startsWith('__reactFiber$')) {
      return (element as unknown as Record<string, unknown>)[name];
    }
  }

  return null;
}

export function setControlledTextareaValue(textarea: HTMLTextAreaElement, value: string): void {
  const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set;
  if (typeof setter === 'function') {
    setter.call(textarea, value);
  } else {
    textarea.value = value;
  }

  try {
    textarea.dispatchEvent(
      new InputEvent('input', {
        bubbles: true,
        cancelable: false,
        data: null,
        inputType: 'insertText'
      })
    );
  } catch {
    textarea.dispatchEvent(new Event('input', { bubbles: true, cancelable: false }));
  }

  textarea.dispatchEvent(new Event('change', { bubbles: true, cancelable: false }));
}

export function getTranscriptRowElements(root: ParentNode = document): HTMLTableRowElement[] {
  return Array.from(root.querySelectorAll<HTMLTableRowElement>(TRANSCRIPT_ROW_SELECTOR)).filter((row) =>
    row.querySelector(ROW_TEXTAREA_SELECTOR)
  );
}
