import { expect } from '@nominy/babel-extension-e2e/test';
import { ROW, TEXT, focusTranscript } from '@nominy/babel-extension-e2e/native-helpers';

export async function expectNativeTexts(page, expected, { sorted = false } = {}) {
  // Finish Slate editing so the captured native renderer exposes source offsets.
  await focusTranscript(page);
  await expect.poll(() => page.evaluate(({ ROW, TEXT }) => {
    const annotations = new Map(window.__BABEL_E2E__.snapshot().annotations.map(row => [row.id, row.content]));
    return [...document.querySelectorAll(ROW)].map(row => {
      const content = annotations.get(row.dataset.annotationId);
      const editor = row.querySelector(TEXT);
      if (typeof content !== 'string' || !editor || editor.isContentEditable) return { content, renderedMatches: false };
      // Recognized native tags render as chips without their enclosing []/<>.
      // Compare raw React state exactly, and independently check visible text.
      let rendered = '';
      let offset = 0;
      for (const chip of editor.querySelectorAll('[data-offset][data-chip]')) {
        const start = Number(chip.dataset.offset);
        const end = start + Number(chip.dataset.chip);
        rendered += content.slice(offset, start) + content.slice(start + 1, end - 1);
        offset = end;
      }
      rendered += content.slice(offset);
      return { content, renderedMatches: editor.textContent === rendered };
    });
  }, { ROW, TEXT }).then(rows => sorted ? rows.sort((left, right) => String(left.content).localeCompare(String(right.content))) : rows))
    .toEqual((sorted ? [...expected].sort((left, right) => left.localeCompare(right)) : expected)
      .map(content => ({ content, renderedMatches: true })));
}
