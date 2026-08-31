import { generateDraftStream } from '../core/backend-client';
import { generateL0Draft } from '../core/l0-client';
import { generateLocalL0Draft } from '../core/local-model-client';
import { replaceTranscriptWithL0Rows } from '../core/l0-replacement-bridge';
import { matchL0CreatedRows } from '../core/l0-created-row-matcher';
import { assessAudioCaptureForDrafting, type AudioCaptureIssue } from '../core/audio-capture-guard';
import { captureAudioTracksForDrafting } from '../core/audio-cues';
import { loadSettings } from '../core/settings';
import { applyDraftRows, buildDiffPreviewItems, captureTranscriptJob, restoreCapturedRows } from '../core/transcript';
import {
  getL0TimingAvailability,
  requestL0TimingRegeneration,
  subscribeL0TimingAvailability,
  type L0TimingAvailability
} from './l0-timing-availability';
import type {
  CapturedAudioTrack,
  DraftRowResult,
  DraftSessionState,
  DraftSummary,
  ExtensionSettings,
  GenerateDraftResponse,
  TranscriptJob
} from '../core/types';

const STYLE_ID = 'babel-gold-drafting-style';
const BUTTON_ID = 'babel-gold-drafting-magic-button';
const OVERLAY_ID = 'babel-gold-drafting-overlay';
const TOOLBAR_BUTTON_SELECTOR = 'button[aria-label="Play all tracks"]';
export type L0DraftGenerators = {
  remote: typeof generateL0Draft;
  local: typeof generateLocalL0Draft;
};

const DEFAULT_L0_DRAFT_GENERATORS: L0DraftGenerators = {
  remote: generateL0Draft,
  local: generateLocalL0Draft
};

export function generateConfiguredL0Draft(
  settings: ExtensionSettings,
  job: TranscriptJob,
  tracks: CapturedAudioTrack[],
  generators: L0DraftGenerators = DEFAULT_L0_DRAFT_GENERATORS
) {
  return settings.localModelsEnabled
    ? generators.local(settings, job, tracks)
    : generators.remote(settings, job, tracks);
}


function createDraftSessionId(jobId: string): string {
  const randomId =
    typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  return `${jobId}:${randomId}`;
}

function createElement<K extends keyof HTMLElementTagNameMap>(
  tagName: K,
  className?: string,
  textContent?: string
): HTMLElementTagNameMap[K] {
  const element = document.createElement(tagName);
  if (className) {
    element.className = className;
  }
  if (typeof textContent === 'string') {
    element.textContent = textContent;
  }
  return element;
}

function ensureStyles(): void {
  if (document.getElementById(STYLE_ID)) {
    return;
  }

  const style = document.createElement('style');
  style.id = STYLE_ID;
  style.textContent = `
    :root {
      --bgd-surface: #ffffff;
      --bgd-surface-muted: #faf5ff;
      --bgd-ink: #221b2d;
      --bgd-muted: #7c6995;
      --bgd-line: #eadff7;
      --bgd-accent: #7c3aed;
      --bgd-accent-hover: #6d28d9;
      --bgd-accent-soft: rgba(124, 58, 237, 0.08);
      --bgd-danger: #dc2626;
      --bgd-success: #16a34a;
      --bgd-radius: 10px;
      --bgd-font: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .bgd-toolbar-button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0;
      width: 36px;
      height: 36px;
      border: 1px solid #d8b4fe;
      background: #faf5ff;
      color: var(--bgd-accent);
      border-radius: 8px;
      padding: 0;
      font: 600 13px/1 var(--bgd-font);
      cursor: pointer;
      flex: 0 0 auto;
    }

    .bgd-toolbar-button:hover {
      background: #f3e8ff;
      border-color: #c084fc;
    }

    .bgd-toolbar-button[data-timing-open="true"] {
      border-radius: 8px 0 0 8px;
      background: #f3e8ff;
      border-color: #c084fc;
    }

    .bgd-toolbar-button[data-state="loading"] {
      cursor: wait;
      opacity: 0.92;
    }

    .bgd-toolbar-button[data-state="done"] {
      background: var(--bgd-success);
      border-color: var(--bgd-success);
    }

    .bgd-toolbar-button[data-state="error"] {
      background: var(--bgd-danger);
      border-color: var(--bgd-danger);
    }

    .bgd-toolbar-button .bgd-icon {
      font-size: 16px;
      line-height: 1;
    }

    .bgd-toolbar-button .bgd-spinner {
      width: 14px;
      height: 14px;
      border-radius: 50%;
      border: 2px solid currentColor;
      border-right-color: transparent;
      display: none;
    }

    .bgd-toolbar-button[data-state="loading"] .bgd-spinner {
      display: inline-block;
      animation: bgd-spin 0.7s linear infinite;
    }

    .bgd-toolbar-button[data-state="loading"] .bgd-icon {
      display: none;
    }

    .bgd-timing-hover-panel {
      position: fixed;
      z-index: 2147483647;
      display: flex;
      align-items: center;
      width: max-content;
      max-width: min(360px, calc(100vw - 24px));
      height: 36px;
      box-sizing: border-box;
      padding: 0 12px;
      border: 1px solid #c084fc;
      border-left: 0;
      border-radius: 0 8px 8px 0;
      background: #f3e8ff;
      color: var(--bgd-accent);
      box-shadow: none;
      font: 600 12px/1.2 var(--bgd-font);
      white-space: nowrap;
      opacity: 0;
      pointer-events: none;
      transform: translateX(-6px);
      transition: opacity 130ms ease, transform 130ms ease;
    }
    .bgd-timing-hover-panel[data-open="true"] {
      opacity: 1;
      pointer-events: auto;
      transform: translateX(0);
    }
    .bgd-timing-hover-panel .bgd-timing-state {
      display: flex;
      align-items: center;
    }
    .bgd-timing-hover-panel .bgd-timing-dot {
      display: none;
    }
    .bgd-timing-hover-panel .bgd-timing-retry {
      width: auto;
      margin: 0 0 0 10px;
      padding: 0;
      border: 0;
      background: transparent;
      color: var(--bgd-accent-hover);
      font: 700 12px/1.2 var(--bgd-font);
      text-decoration: underline;
      cursor: pointer;
    }

    #${OVERLAY_ID} {
      position: fixed;
      inset: 0;
      z-index: 2147483647;
      font: 13px/1.5 var(--bgd-font);
    }

    #${OVERLAY_ID}[hidden] {
      display: none;
    }

    #${OVERLAY_ID} .bgd-backdrop {
      position: absolute;
      inset: 0;
      background: rgba(15, 10, 25, 0.42);
      backdrop-filter: blur(2px);
    }

    #${OVERLAY_ID} .bgd-shell {
      position: relative;
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 16px;
    }

    #${OVERLAY_ID} .bgd-dialog {
      width: min(980px, calc(100vw - 24px));
      max-height: calc(100vh - 24px);
      overflow: auto;
      background: var(--bgd-surface);
      border: 1px solid var(--bgd-line);
      border-radius: var(--bgd-radius);
      box-shadow: 0 30px 80px rgba(15, 10, 25, 0.28);
      color: var(--bgd-ink);
    }

    #${OVERLAY_ID} .bgd-header {
      padding: 12px 16px;
      border-bottom: 1px solid var(--bgd-line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      position: sticky;
      top: 0;
      background: var(--bgd-surface);
      z-index: 1;
    }

    #${OVERLAY_ID} .bgd-header-title {
      font-weight: 700;
      font-size: 14px;
    }

    #${OVERLAY_ID} .bgd-header-title-row {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }

    #${OVERLAY_ID} .bgd-support-link {
      color: var(--bgd-accent);
      font-size: 11px;
      font-weight: 700;
      text-decoration: none;
      white-space: normal;
    }

    #${OVERLAY_ID} .bgd-support-link:hover {
      text-decoration: underline;
    }

    #${OVERLAY_ID} .bgd-header-subtitle {
      color: var(--bgd-muted);
      font-size: 12px;
      margin-top: 2px;
    }

    #${OVERLAY_ID} .bgd-close {
      border: 1px solid var(--bgd-line);
      border-radius: 8px;
      padding: 7px 10px;
      background: #fff;
      color: var(--bgd-ink);
      font: inherit;
      cursor: pointer;
    }

    #${OVERLAY_ID} .bgd-main {
      padding: 14px 16px 16px;
      display: grid;
      gap: 12px;
    }

    #${OVERLAY_ID} .bgd-status {
      font-size: 12px;
      color: var(--bgd-muted);
    }

    #${OVERLAY_ID} .bgd-status[data-error="true"] {
      color: var(--bgd-danger);
    }

    #${OVERLAY_ID} .bgd-audio-guard {
      border: 1px solid #f0d9a8;
      border-radius: 8px;
      background: #fffbeb;
      color: #5f4312;
      padding: 10px 12px;
      display: grid;
      gap: 10px;
      font-size: 12px;
    }

    #${OVERLAY_ID} .bgd-audio-guard[hidden] {
      display: none;
    }

    #${OVERLAY_ID} .bgd-audio-actions {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }

    #${OVERLAY_ID} .bgd-summary {
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
      font-size: 12px;
      color: var(--bgd-muted);
    }

    #${OVERLAY_ID} .bgd-summary-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }

    #${OVERLAY_ID} .bgd-summary-dot {
      width: 7px;
      height: 7px;
      border-radius: 999px;
      background: var(--bgd-accent);
    }

    #${OVERLAY_ID} .bgd-block {
      border: 1px solid var(--bgd-line);
      border-radius: var(--bgd-radius);
      background: var(--bgd-surface-muted);
      overflow: hidden;
    }

    #${OVERLAY_ID} .bgd-block-header {
      padding: 10px 12px;
      border-bottom: 1px solid var(--bgd-line);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--bgd-muted);
    }

    #${OVERLAY_ID} .bgd-block-body {
      padding: 12px;
    }

    #${OVERLAY_ID} .bgd-empty {
      color: var(--bgd-muted);
      font-size: 12px;
      text-align: center;
      padding: 12px;
      background: #fff;
      border-radius: 8px;
    }

    #${OVERLAY_ID} .bgd-diff-list {
      display: grid;
      gap: 10px;
      max-height: 56vh;
      overflow: auto;
    }

    #${OVERLAY_ID} .bgd-card {
      border: 1px solid var(--bgd-line);
      border-radius: 8px;
      background: #fff;
      padding: 10px;
      display: grid;
      gap: 8px;
    }

    #${OVERLAY_ID} .bgd-card.failed {
      border-color: #fecaca;
      background: #fff5f5;
    }

    #${OVERLAY_ID} .bgd-card-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      flex-wrap: wrap;
    }

    #${OVERLAY_ID} .bgd-card-title {
      font-size: 12px;
      font-weight: 600;
    }

    #${OVERLAY_ID} .bgd-badge {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 11px;
      font-weight: 600;
      background: var(--bgd-accent-soft);
      color: var(--bgd-accent);
    }

    #${OVERLAY_ID} .bgd-diff-view {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      background: #fcfbff;
    }

    #${OVERLAY_ID} .bgd-diff-pane {
      min-width: 0;
    }

    #${OVERLAY_ID} .bgd-diff-label {
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      color: var(--bgd-muted);
      margin-bottom: 4px;
    }

    #${OVERLAY_ID} .bgd-diff-content {
      border: 1px solid var(--bgd-line);
      border-radius: 8px;
      background: #fff;
      padding: 8px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 11px;
      white-space: pre-wrap;
      word-break: break-word;
      min-height: 58px;
    }

    #${OVERLAY_ID} .bgd-warning-list {
      margin: 0;
      padding-left: 18px;
      font-size: 12px;
      color: var(--bgd-muted);
    }

    #${OVERLAY_ID} .bgd-footer {
      padding: 12px 16px 16px;
      border-top: 1px solid var(--bgd-line);
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 8px;
      flex-wrap: wrap;
      background: var(--bgd-surface);
      position: sticky;
      bottom: 0;
    }

    #${OVERLAY_ID} .bgd-button {
      border: 1px solid var(--bgd-line);
      border-radius: 8px;
      padding: 8px 12px;
      background: #fff;
      color: var(--bgd-ink);
      font: inherit;
      font-weight: 600;
      cursor: pointer;
    }

    #${OVERLAY_ID} .bgd-button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    #${OVERLAY_ID} .bgd-button[data-variant="primary"] {
      border-color: transparent;
      background: var(--bgd-accent);
      color: #fff;
    }

    @keyframes bgd-spin {
      to { transform: rotate(360deg); }
    }

    @media (max-width: 720px) {
      #${OVERLAY_ID} .bgd-shell {
        padding: 8px;
      }

      #${OVERLAY_ID} .bgd-dialog {
        width: calc(100vw - 16px);
        max-height: calc(100vh - 16px);
      }

      #${OVERLAY_ID} .bgd-diff-view {
        grid-template-columns: 1fr;
      }
    }
  `;

  document.documentElement.appendChild(style);
}

export class DraftingOverlayController {
  private button: HTMLButtonElement | null = null;
  private overlay: HTMLDivElement | null = null;
  private dialogEl: HTMLDivElement | null = null;
  private statusEl: HTMLDivElement | null = null;
  private audioGuardEl: HTMLDivElement | null = null;
  private summaryEl: HTMLDivElement | null = null;
  private previewEl: HTMLDivElement | null = null;
  private applyButton: HTMLButtonElement | null = null;
  private restoreButton: HTMLButtonElement | null = null;
  private closeButton: HTMLButtonElement | null = null;
  private timingPanel: HTMLDivElement | null = null;
  private timingAvailability: L0TimingAvailability | null = null;
  private timingAvailabilityDispose: (() => void) | null = null;
  private timingPanelHideTimer: number | null = null;
  private state: DraftSessionState = {
    capturedJob: null,
    draftResponse: null,
    lastApplyResult: null
  };
  private streamedRows: DraftRowResult[] = [];
  private streamedSummary: DraftSummary | null = null;
  private l0WarningsByRowId = new Map<string, string[]>();
  private streamedCompletedRows = 0;
  private streamedTotalRows = 0;
  private activeDraftLabel = 'Gold / OpenRouter';
  private busy = false;
  private pendingAudioDraft: {
    capturedJob: TranscriptJob;
    settings: ExtensionSettings;
    audioTracks: CapturedAudioTrack[];
    issue: AudioCaptureIssue;
  } | null = null;

  mount(): void {
    ensureStyles();
    if (!this.timingAvailabilityDispose) {
      this.timingAvailability = getL0TimingAvailability();
      this.timingAvailabilityDispose = subscribeL0TimingAvailability((availability) => {
        this.timingAvailability = availability;
        this.renderTimingPanel();
      });
    }
    this.ensureMagicButton();
    this.ensureOverlay();
    this.render();
  }

  ensureMagicButton(): void {
    ensureStyles();
    this.ensureButton();
  }

  unmount(): void {
    this.button?.remove();
    this.overlay?.remove();
    this.timingPanel?.remove();
    this.timingAvailabilityDispose?.();
    if (this.timingPanelHideTimer !== null) window.clearTimeout(this.timingPanelHideTimer);
    this.button = null;
    this.overlay = null;
    this.dialogEl = null;
    this.statusEl = null;
    this.audioGuardEl = null;
    this.summaryEl = null;
    this.previewEl = null;
    this.applyButton = null;
    this.restoreButton = null;
    this.closeButton = null;
    this.timingPanel = null;
    this.timingAvailabilityDispose = null;
    this.timingPanelHideTimer = null;
  }

  private ensureButton(): void {
    const host = this.findToolbarHost();
    if (!host) {
      return;
    }

    if (this.button?.isConnected && this.button.parentElement === host) {
      return;
    }

    this.button?.remove();

    const button = document.createElement('button');
    button.id = BUTTON_ID;
    button.type = 'button';
    button.className = 'bgd-toolbar-button';
    button.setAttribute('aria-label', 'Gold Draft');
    button.title = 'Gold Draft';
    button.innerHTML = `
      <span class="bgd-icon">\u{1FA84}</span>
      <span class="bgd-spinner"></span>
    `;
    button.addEventListener('click', () => {
      void this.runMagicDraft();
    });
    button.addEventListener('mouseenter', () => this.showTimingPanel());
    button.addEventListener('mouseleave', () => this.scheduleTimingPanelHide());
    button.addEventListener('focus', () => this.showTimingPanel());
    button.addEventListener('blur', () => this.scheduleTimingPanelHide());

    host.appendChild(button);
    this.button = button;
  }

  private ensureTimingPanel(): HTMLDivElement {
    if (this.timingPanel?.isConnected) return this.timingPanel;
    const panel = createElement('div', 'bgd-timing-hover-panel');
    panel.dataset.open = 'false';
    panel.setAttribute('role', 'status');
    panel.setAttribute('aria-live', 'polite');
    const state = createElement('div', 'bgd-timing-state');
    state.append(
      createElement('span', 'bgd-timing-dot'),
      createElement('span', 'bgd-timing-copy')
    );
    const retry = createElement('button', 'bgd-timing-retry', 'Regenerate timestamp data');
    retry.type = 'button';
    retry.addEventListener('click', () => {
      if (requestL0TimingRegeneration()) {
        this.timingAvailability = getL0TimingAvailability();
        this.renderTimingPanel();
      }
    });
    panel.addEventListener('mouseenter', () => this.cancelTimingPanelHide());
    panel.addEventListener('mouseleave', () => this.scheduleTimingPanelHide());
    panel.append(state, retry);
    document.body.appendChild(panel);
    this.timingPanel = panel;
    this.renderTimingPanel();
    return panel;
  }

  private showTimingPanel(): void {
    this.cancelTimingPanelHide();
    const panel = this.ensureTimingPanel();
    const buttonRect = this.button?.getBoundingClientRect();
    if (buttonRect) {
      panel.style.top = `${buttonRect.top}px`;
      panel.style.left = `${buttonRect.right - 1}px`;
    }
    if (this.button) this.button.dataset.timingOpen = 'true';
    panel.dataset.open = 'true';
  }

  private scheduleTimingPanelHide(): void {
    this.cancelTimingPanelHide();
    this.timingPanelHideTimer = window.setTimeout(() => {
      this.timingPanelHideTimer = null;
      if (this.button) this.button.dataset.timingOpen = 'false';
      if (this.timingPanel) this.timingPanel.dataset.open = 'false';
    }, 180);
  }

  private cancelTimingPanelHide(): void {
    if (this.timingPanelHideTimer === null) return;
    window.clearTimeout(this.timingPanelHideTimer);
    this.timingPanelHideTimer = null;
  }

  private renderTimingPanel(): void {
    if (!this.timingPanel) return;
    const availability = this.timingAvailability;
    const status = availability?.status ?? 'unavailable';
    const copyByStatus: Record<L0TimingAvailability['status'], string> = {
      available: 'Timestamp data available',
      unavailable: 'Timestamp data not available',
      preparing: 'Generating timestamp data…',
      queued:
        availability?.status === 'queued'
          ? `Timestamp generation queued · #${availability.position}`
          : 'Timestamp generation queued',
      running: 'Generating timestamp data…',
      retrying: 'Retrying timestamp generation…'
    };
    this.timingPanel.dataset.status = status;
    this.timingPanel.dataset.taskId = availability?.taskId ?? '';
    const copy = this.timingPanel.querySelector<HTMLElement>('.bgd-timing-copy');
    if (copy) copy.textContent = copyByStatus[status];
    const retry = this.timingPanel.querySelector<HTMLButtonElement>('.bgd-timing-retry');
    if (retry) retry.hidden = status !== 'unavailable';
  }


  private findToolbarHost(): HTMLElement | null {
    const anchor = document.querySelector(TOOLBAR_BUTTON_SELECTOR);
    if (!(anchor instanceof HTMLButtonElement)) {
      return null;
    }

    return anchor.parentElement instanceof HTMLElement ? anchor.parentElement : null;
  }

  private ensureOverlay(): void {
    if (this.overlay?.isConnected) {
      return;
    }

    const overlay = createElement('div');
    overlay.id = OVERLAY_ID;
    overlay.hidden = true;

    const backdrop = createElement('div', 'bgd-backdrop');
    backdrop.addEventListener('click', () => this.closeDialog());

    const shell = createElement('div', 'bgd-shell');
    const dialog = createElement('div', 'bgd-dialog');
    dialog.addEventListener('click', (event) => event.stopPropagation());

    const header = createElement('div', 'bgd-header');
    const titleWrap = createElement('div');
    const titleRow = createElement('div', 'bgd-header-title-row');
    const supportLink = createElement(
      'a',
      'bgd-support-link',
      'if this extension saves you time, consider supporting development on Ko-Fi',
    );
    supportLink.href = 'https://ko-fi.com/naftsan';
    supportLink.target = '_blank';
    supportLink.rel = 'noopener noreferrer';
    titleRow.append(createElement('div', 'bgd-header-title', 'Gold Draft'), supportLink);
    titleWrap.append(titleRow, createElement('div', 'bgd-header-subtitle', 'Silver -> Gold draft preview before apply'));
    const closeButton = createElement('button', 'bgd-close', 'Close');
    closeButton.type = 'button';
    closeButton.addEventListener('click', () => this.closeDialog());
    header.append(titleWrap, closeButton);

    const main = createElement('div', 'bgd-main');
    const statusEl = createElement('div', 'bgd-status', 'Click the wand to generate a draft.');
    const audioGuardEl = createElement('div', 'bgd-audio-guard');
    audioGuardEl.hidden = true;

    const summaryBlock = createElement('section', 'bgd-block');
    summaryBlock.append(createElement('div', 'bgd-block-header', 'Summary'));
    const summaryBody = createElement('div', 'bgd-block-body');
    const summaryEl = createElement('div', 'bgd-empty', 'No draft generated yet.');
    summaryBody.append(summaryEl);
    summaryBlock.append(summaryBody);

    const previewBlock = createElement('section', 'bgd-block');
    previewBlock.append(createElement('div', 'bgd-block-header', 'Diff Preview'));
    const previewBody = createElement('div', 'bgd-block-body');
    const previewEl = createElement('div', 'bgd-empty', 'The diff will appear here after generation.');
    previewBody.append(previewEl);
    previewBlock.append(previewBody);

    main.append(statusEl, audioGuardEl, summaryBlock, previewBlock);

    const footer = createElement('div', 'bgd-footer');
    const restoreButton = createElement('button', 'bgd-button', 'Restore Original');
    restoreButton.type = 'button';
    restoreButton.addEventListener('click', () => void this.restoreOriginal());
    const applyButton = createElement('button', 'bgd-button', 'Apply Draft');
    applyButton.type = 'button';
    applyButton.dataset.variant = 'primary';
    applyButton.addEventListener('click', () => void this.applyDraft());
    footer.append(restoreButton, applyButton);

    dialog.append(header, main, footer);
    shell.append(dialog);
    overlay.append(backdrop, shell);
    document.documentElement.appendChild(overlay);

    this.overlay = overlay;
    this.dialogEl = dialog;
    this.statusEl = statusEl;
    this.audioGuardEl = audioGuardEl;
    this.summaryEl = summaryEl;
    this.previewEl = previewEl;
    this.applyButton = applyButton;
    this.restoreButton = restoreButton;
    this.closeButton = closeButton;
  }

  private openDialog(): void {
    if (this.overlay) {
      this.overlay.hidden = false;
    }
  }

  private closeDialog(): void {
    if (!this.busy && this.overlay) {
      this.overlay.hidden = true;
    }
  }

  private setButtonState(mode: 'idle' | 'loading' | 'done' | 'error', label: string): void {
    if (!(this.button instanceof HTMLButtonElement)) {
      return;
    }

    this.button.dataset.state = mode;
    this.button.disabled = mode === 'loading';
    this.button.title = label;
    this.button.setAttribute('aria-label', label);
  }

  private setBusy(nextBusy: boolean): void {
    this.busy = nextBusy;
    if (this.applyButton) {
      this.applyButton.disabled = nextBusy || !this.state.draftResponse;
    }
    if (this.restoreButton) {
      this.restoreButton.disabled = nextBusy || !this.state.capturedJob;
    }
    if (this.closeButton) {
      this.closeButton.disabled = nextBusy;
    }
    this.audioGuardEl
      ?.querySelectorAll<HTMLButtonElement>('button[data-bgd-audio-action]')
      .forEach((button) => {
        button.disabled = nextBusy;
      });
  }

  private setStatus(message: string, isError = false): void {
    if (this.statusEl) {
      this.statusEl.textContent = message;
      this.statusEl.dataset.error = isError ? 'true' : 'false';
    }
  }

  private clearAudioGuard(): void {
    this.pendingAudioDraft = null;
    if (this.audioGuardEl) {
      this.audioGuardEl.hidden = true;
      this.audioGuardEl.replaceChildren();
    }
  }

  private showAudioGuard(
    capturedJob: TranscriptJob,
    settings: ExtensionSettings,
    audioTracks: CapturedAudioTrack[],
    issue: AudioCaptureIssue
  ): void {
    if (!this.audioGuardEl) {
      return;
    }

    this.pendingAudioDraft = {
      capturedJob,
      settings,
      audioTracks,
      issue
    };

    const problem =
      issue.kind === 'missing'
        ? `No speaker-lane audio was captured. ${issue.capturedTracks} generic audio source(s) were ignored.`
        : `Only ${issue.capturedSpeakerLanes} of ${issue.expectedSpeakerLanes} speaker lane(s) were captured.`;
    const message = createElement(
      'div',
      '',
      `${problem} Audio cues may miss laughter or other events if the draft starts now.`
    );
    const actions = createElement('div', 'bgd-audio-actions');
    const retryButton = createElement('button', 'bgd-button', 'Retry Audio');
    retryButton.type = 'button';
    retryButton.dataset.bgdAudioAction = 'retry';
    retryButton.addEventListener('click', () => void this.retryPendingAudioCapture());

    actions.append(retryButton);

    if (issue.kind === 'partial' && audioTracks.length) {
      const usePartialButton = createElement('button', 'bgd-button', 'Use Captured Audio');
      usePartialButton.type = 'button';
      usePartialButton.dataset.bgdAudioAction = 'use-partial';
      usePartialButton.addEventListener('click', () => void this.continuePendingDraft('captured-audio'));
      actions.append(usePartialButton);
    }

    const textOnlyButton = createElement('button', 'bgd-button', 'Continue Text Only');
    textOnlyButton.type = 'button';
    textOnlyButton.dataset.bgdAudioAction = 'text-only';
    textOnlyButton.addEventListener('click', () => void this.continuePendingDraft('text-only'));
    actions.append(textOnlyButton);

    this.audioGuardEl.hidden = false;
    this.audioGuardEl.replaceChildren(message, actions);
    this.setStatus('Audio capture needs review before generating.');
  }

  private logCapturedAudioTracks(tracks: CapturedAudioTrack[]): void {
    console.info(
      '[Babel Gold Drafting] captured audio tracks',
      tracks.map((track) => ({
        trackId: track.trackId,
        speakerKey: track.speakerKey || '',
        trackLabel: track.trackLabel || '',
        source: track.source,
        bytes: track.blob.size
      }))
    );
  }

  private async captureAudioTracksWithStatus(): Promise<CapturedAudioTrack[]> {
    this.setStatus('Capturing available task audio...');
    const tracks = await captureAudioTracksForDrafting().catch(() => []);
    this.logCapturedAudioTracks(tracks);
    return tracks;
  }

  private renderSummary(draftResponse: GenerateDraftResponse | null): void {
    if (!this.summaryEl) {
      return;
    }

    if (!this.state.capturedJob) {
      this.summaryEl.className = 'bgd-empty';
      this.summaryEl.textContent = 'No transcript captured yet.';
      return;
    }

    const summary = createElement('div', 'bgd-summary');
    summary.append(this.createSummaryPill(this.activeDraftLabel));
    summary.append(
      this.createSummaryPill(`Job ${this.state.capturedJob.jobId}`),
      this.createSummaryPill(`${this.state.capturedJob.rows.length} rows captured`)
    );

    if (draftResponse) {
      summary.append(
        this.createSummaryPill(`${draftResponse.summary.rewrittenRows} rewritten`),
        this.createSummaryPill(`${draftResponse.summary.unchangedRows} unchanged`),
        this.createSummaryPill(`${draftResponse.summary.failedRows} failed`)
      );
    } else if (this.streamedTotalRows > 0) {
      summary.append(this.createSummaryPill(`${this.streamedCompletedRows}/${this.streamedTotalRows} rows complete`));
      if (this.streamedSummary) {
        summary.append(
          this.createSummaryPill(`${this.streamedSummary.rewrittenRows} rewritten`),
          this.createSummaryPill(`${this.streamedSummary.unchangedRows} unchanged`),
          this.createSummaryPill(`${this.streamedSummary.failedRows} failed`)
        );
      }
    }

    this.summaryEl.className = '';
    this.summaryEl.replaceChildren(summary);
  }

  private createSummaryPill(text: string): HTMLDivElement {
    const pill = createElement('div', 'bgd-summary-pill');
    pill.append(createElement('span', 'bgd-summary-dot'), createElement('span', '', text));
    return pill;
  }

  private renderPreview(): void {
    if (!this.previewEl) {
      return;
    }

    const previousList =
      this.previewEl.firstElementChild instanceof HTMLDivElement &&
      this.previewEl.firstElementChild.classList.contains('bgd-diff-list')
        ? this.previewEl.firstElementChild
        : null;
    const previousListScrollTop = previousList?.scrollTop ?? 0;

    const captured = this.state.capturedJob;
    const draft = this.state.draftResponse;
    if (!captured) {
      this.previewEl.className = 'bgd-empty';
      this.previewEl.textContent = 'The diff will appear here after generation.';
      return;
    }

    const sourceRows = draft ? draft.draftRows : this.streamedRows;
    if (!sourceRows.length) {
      this.previewEl.className = 'bgd-empty';
      this.previewEl.textContent = 'Waiting for the first completed row...';
      return;
    }

    const diffItems = buildDiffPreviewItems(captured.rows, sourceRows);
    if (!diffItems.length) {
      this.previewEl.className = 'bgd-empty';
      this.previewEl.textContent = draft ? 'No row text changed.' : 'Completed rows have no visible text changes yet.';
      return;
    }

    const list = createElement('div', 'bgd-diff-list');
    for (const item of diffItems.slice(0, 40)) {
      const card = createElement('article', `bgd-card ${item.status === 'failed' ? 'failed' : ''}`);
      const top = createElement('div', 'bgd-card-top');
      top.append(
        createElement('div', 'bgd-card-title', `Row ${item.index + 1}`),
        createElement('div', 'bgd-badge', item.status)
      );
      card.append(top);

      const diffView = createElement('div', 'bgd-diff-view');
      const beforePane = createElement('div', 'bgd-diff-pane');
      beforePane.append(
        createElement('div', 'bgd-diff-label', 'Before'),
        this.createDiffContent(item.before)
      );
      const afterPane = createElement('div', 'bgd-diff-pane');
      afterPane.append(
        createElement('div', 'bgd-diff-label', 'After'),
        this.createDiffContent(item.after)
      );
      diffView.append(beforePane, afterPane);
      card.append(diffView);

      if (item.warnings.length) {
        const warnings = createElement('ul', 'bgd-warning-list');
        for (const warning of item.warnings) {
          warnings.append(createElement('li', '', warning));
        }
        card.append(warnings);
      }

      list.append(card);
    }

    if (diffItems.length > 40) {
      list.append(createElement('div', 'bgd-empty', `Showing first 40 of ${diffItems.length} changed rows.`));
    }

    this.previewEl.className = '';
    this.previewEl.replaceChildren(list);
    list.scrollTop = previousListScrollTop;
  }

  private createDiffContent(text: string): HTMLDivElement {
    const content = createElement('div', 'bgd-diff-content');
    content.textContent = text || '(empty)';
    return content;
  }

  private render(): void {
    const dialogScrollTop = this.dialogEl?.scrollTop ?? 0;
    this.renderSummary(this.state.draftResponse);
    this.renderPreview();
    this.setBusy(this.busy);
    if (this.dialogEl) {
      this.dialogEl.scrollTop = dialogScrollTop;
    }
  }

  private async runMagicDraft(): Promise<void> {
    this.openDialog();
    this.activeDraftLabel = 'Gold / OpenRouter';
    this.clearAudioGuard();
    this.l0WarningsByRowId.clear();
    this.state = {
      capturedJob: null,
      draftResponse: null,
      lastApplyResult: null
    };
    this.streamedRows = [];
    this.streamedSummary = null;
    this.streamedCompletedRows = 0;
    this.streamedTotalRows = 0;
    this.render();

    try {
      this.setBusy(true);
      this.setButtonState('loading', 'Generating...');
      this.setStatus('Capturing transcript...');

      let capturedJob = captureTranscriptJob();
      if (!capturedJob.rows.length) {
        throw new Error('No transcript rows detected on this page.');
      }
      this.state.capturedJob = capturedJob;
      this.streamedTotalRows = capturedJob.rows.length;
      this.render();

      const settings = await loadSettings();
      if (settings.l0ReplacementPreviewEnabled) {
        this.activeDraftLabel = settings.l0DontRunLlm
          ? 'L0 replacement / no LLM'
          : 'L0 replacement -> Gold / OpenRouter';
        capturedJob = await this.runL0Replacement(capturedJob, settings);
        this.state.capturedJob = capturedJob;
        if (settings.l0DontRunLlm) {
          this.setButtonState('done', 'L0 Replacement Ready');
          window.setTimeout(() => this.setButtonState('idle', 'Gold Draft'), 1600);
          return;
        }
        this.state.draftResponse = null;
        this.streamedRows = [];
        this.streamedSummary = null;
        this.streamedCompletedRows = 0;
        this.streamedTotalRows = capturedJob.rows.length;
        this.render();
      }

      if (!settings.openRouterApiKey) {
        throw new Error(
          'OpenRouter API key is required. Add your key in the Babel Gold Drafting extension options. Setup guide: https://youtu.be/F-p45lvkzyU?si=2glvFn-iJnKEs8MI'
        );
      }
      const audioTracks = settings.audioInputEnabled ? await this.captureAudioTracksWithStatus() : [];
      const audioIssue = settings.audioInputEnabled ? assessAudioCaptureForDrafting(capturedJob, audioTracks) : null;
      if (audioIssue) {
        this.showAudioGuard(capturedJob, settings, audioTracks, audioIssue);
        this.setButtonState('error', 'Audio Capture Needs Review');
        window.setTimeout(() => this.setButtonState('idle', 'Gold Draft'), 2200);
        return;
      }

      await this.generateDraftFromCapture(capturedJob, settings, audioTracks);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.setStatus(message, true);
      this.setButtonState('error', 'Draft Failed');
      window.setTimeout(() => this.setButtonState('idle', 'Gold Draft'), 2200);
    } finally {
      this.setBusy(false);
      this.render();
    }
  }

  private async runL0Replacement(
    capturedJob: TranscriptJob,
    settings: ExtensionSettings
  ): Promise<TranscriptJob> {
    this.setStatus('Capturing exactly two WAV speaker tracks for L0 replacement...');
    const audioTracks = await captureAudioTracksForDrafting();
    this.logCapturedAudioTracks(audioTracks);
    this.setStatus(
      settings.localModelsEnabled
        ? 'Generating replacement segments with local browser models...'
        : 'Generating replacement segments with the self-hosted L0 endpoint...'
    );
    const response = await generateConfiguredL0Draft(settings, capturedJob, audioTracks);

    this.setStatus(`Replacing current transcript with ${response.rows.length} L0 segment(s) through Babel Helper...`);
    const created = await replaceTranscriptWithL0Rows(response.rows);
    const createdIds = new Set(created.map((mapping) => mapping.id));
    if (createdIds.size !== response.rows.length || response.rows.some((row) => !createdIds.has(row.id))) {
      throw new Error('Babel Helper returned incomplete or duplicate L0 row mappings.');
    }
    const populatedJob = captureTranscriptJob();
    const matchedRows = matchL0CreatedRows(response.rows, populatedJob.rows);
    const visibleTextMismatchCount = matchedRows.filter(({ warnings }) => warnings.length > 0).length;
    this.l0WarningsByRowId = new Map(
      matchedRows.map(({ capturedRow, warnings }) => [capturedRow.rowId, [...new Set(warnings)]])
    );
    const visibleTextMismatchStatus =
      visibleTextMismatchCount > 0
        ? ` ${visibleTextMismatchCount} visible text mismatch warning${visibleTextMismatchCount === 1 ? '' : 's'}.`
        : '';
    this.state.capturedJob = populatedJob;
    this.state.draftResponse = {
      draftRows: matchedRows.map(({ engineRow, capturedRow, warnings }) => ({
        rowId: capturedRow.rowId,
        rewrittenText: engineRow.text,
        status: 'rewritten',
        warnings: [...warnings]
      })),
      summary: {
        totalRows: created.length,
        rewrittenRows: created.length,
        unchangedRows: 0,
        failedRows: 0,
        anomalyCounts:
          visibleTextMismatchCount > 0 ? { l0VisibleTextMismatch: visibleTextMismatchCount } : {}
      },
      generationMeta: {
        model: Object.keys(response.models).join(' + ') || 'L0 two-model engine',
        rulePackVersion: 'l0-replacement',
        generatedAt: new Date().toISOString()
      }
    };
    this.setStatus(
      settings.l0DontRunLlm
        ? `L0 replacement complete: ${created.length} segment(s). LLM drafting was skipped.${visibleTextMismatchStatus}`
        : `L0 replacement complete: ${created.length} segment(s). Recaptured transcript for Gold LLM drafting.${visibleTextMismatchStatus}`
    );
    this.render();
    return populatedJob;
  }


  private async generateDraftFromCapture(
    capturedJob: TranscriptJob,
    settings: ExtensionSettings,
    audioTracks: CapturedAudioTrack[]
  ): Promise<void> {
    this.clearAudioGuard();
    const streamStatusLabel = audioTracks.length ? 'Streaming Gold draft with audio cues' : 'Streaming Gold draft';
    this.setStatus(
      `Starting Gold draft stream for ${capturedJob.rows.length} rows${
        audioTracks.length ? ` with ${audioTracks.length} audio track(s)` : ''
      }...`
    );

    const draftResponse = await generateDraftStream(settings.backendBaseUrl, {
      projectPreset: settings.projectPreset,
      jobId: capturedJob.jobId,
      draftSessionId: createDraftSessionId(capturedJob.jobId),
      rows: capturedJob.rows,
      openRouterApiKey: settings.openRouterApiKey,
      model: settings.model || undefined,
      serviceTier: settings.serviceTier,
      reasoningEffort: settings.reasoningEffort
    }, {
      onStarted: ({ totalRows }) => {
        this.streamedTotalRows = totalRows;
        this.setStatus(`${streamStatusLabel}... 0 / ${totalRows} rows complete.`);
        this.render();
      },
      onRow: ({ row, completedRows, totalRows, summary }) => {
        const existingIndex = this.streamedRows.findIndex((candidate) => candidate.rowId === row.rowId);
        if (existingIndex >= 0) {
          this.streamedRows[existingIndex] = row;
        } else {
          this.streamedRows.push(row);
        }
        this.streamedCompletedRows = completedRows;
        this.streamedTotalRows = totalRows;
        this.streamedSummary = summary;
        this.setStatus(`${streamStatusLabel}... ${completedRows} / ${totalRows} rows complete.`);
        this.render();
      },
      onDone: (response) => {
        this.streamedRows = response.draftRows;
        this.streamedSummary = response.summary;
        this.streamedCompletedRows = response.summary.totalRows;
        this.streamedTotalRows = response.summary.totalRows;
      },
      onReconnect: () => {
        this.setStatus('Stream connection lost. Reconciling final draft response...');
        this.render();
      }
    }, audioTracks);

    const l0VisibleTextMismatchCount = [...this.l0WarningsByRowId.values()].filter(
      (warnings) => warnings.length > 0
    ).length;
    const finalDraftResponse =
      l0VisibleTextMismatchCount > 0
        ? {
            ...draftResponse,
            draftRows: draftResponse.draftRows.map((row) => ({
              ...row,
              warnings: [...new Set([...row.warnings, ...(this.l0WarningsByRowId.get(row.rowId) ?? [])])]
            })),
            summary: {
              ...draftResponse.summary,
              anomalyCounts: {
                ...draftResponse.summary.anomalyCounts,
                l0VisibleTextMismatch:
                  (draftResponse.summary.anomalyCounts.l0VisibleTextMismatch ?? 0) +
                  l0VisibleTextMismatchCount
              }
            }
          }
        : draftResponse;
    this.streamedRows = finalDraftResponse.draftRows;
    this.streamedSummary = finalDraftResponse.summary;
    this.streamedCompletedRows = finalDraftResponse.summary.totalRows;
    this.streamedTotalRows = finalDraftResponse.summary.totalRows;
    this.state.draftResponse = finalDraftResponse;
    this.setStatus(
      `Draft ready. ${finalDraftResponse.summary.rewrittenRows} rewritten, ${finalDraftResponse.summary.failedRows} failed fallback rows.${
        l0VisibleTextMismatchCount > 0
          ? ` ${l0VisibleTextMismatchCount} visible text mismatch warning${l0VisibleTextMismatchCount === 1 ? '' : 's'}.`
          : ''
      }`
    );
    this.setButtonState('done', 'Draft Ready');
    window.setTimeout(() => this.setButtonState('idle', 'Gold Draft'), 1600);
  }

  private async retryPendingAudioCapture(): Promise<void> {
    const pending = this.pendingAudioDraft;
    if (!pending) {
      this.setStatus('No pending audio capture to retry.', true);
      return;
    }

    try {
      this.setBusy(true);
      this.setButtonState('loading', 'Retrying audio...');
      const audioTracks = await this.captureAudioTracksWithStatus();
      const audioIssue = assessAudioCaptureForDrafting(pending.capturedJob, audioTracks);
      if (audioIssue) {
        this.showAudioGuard(pending.capturedJob, pending.settings, audioTracks, audioIssue);
        this.setButtonState('error', 'Audio Capture Needs Review');
        window.setTimeout(() => this.setButtonState('idle', 'Gold Draft'), 2200);
        return;
      }

      await this.generateDraftFromCapture(pending.capturedJob, pending.settings, audioTracks);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.setStatus(message, true);
      this.setButtonState('error', 'Draft Failed');
      window.setTimeout(() => this.setButtonState('idle', 'Gold Draft'), 2200);
    } finally {
      this.setBusy(false);
      this.render();
    }
  }

  private async continuePendingDraft(mode: 'captured-audio' | 'text-only'): Promise<void> {
    const pending = this.pendingAudioDraft;
    if (!pending) {
      this.setStatus('No pending draft to continue.', true);
      return;
    }

    try {
      this.setBusy(true);
      this.setButtonState('loading', 'Generating...');
      await this.generateDraftFromCapture(
        pending.capturedJob,
        pending.settings,
        mode === 'captured-audio' ? pending.audioTracks : []
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.setStatus(message, true);
      this.setButtonState('error', 'Draft Failed');
      window.setTimeout(() => this.setButtonState('idle', 'Gold Draft'), 2200);
    } finally {
      this.setBusy(false);
      this.render();
    }
  }

  private async applyDraft(): Promise<void> {
    if (!this.state.draftResponse) {
      this.setStatus('No draft available yet.', true);
      return;
    }

    const result = applyDraftRows(this.state.draftResponse.draftRows);
    this.state.lastApplyResult = result;
    const missingNote = result.missingRowIds.length ? ` Missing ${result.missingRowIds.length} rows during apply.` : '';
    this.setStatus(`Applied draft to ${result.appliedCount} rows.${missingNote}`);
    this.setButtonState('done', 'Applied');
    this.closeDialog();
    window.setTimeout(() => this.setButtonState('idle', 'Gold Draft'), 1600);
    this.render();
  }

  private async restoreOriginal(): Promise<void> {
    if (!this.state.capturedJob) {
      this.setStatus('No captured snapshot to restore.', true);
      return;
    }

    const result = restoreCapturedRows(this.state.capturedJob);
    this.state.lastApplyResult = result;
    const missingNote = result.missingRowIds.length ? ` Missing ${result.missingRowIds.length} rows during restore.` : '';
    this.setStatus(`Restored ${result.appliedCount} original rows.${missingNote}`);
    this.setButtonState('done', 'Restored');
    window.setTimeout(() => this.setButtonState('idle', 'Gold Draft'), 1600);
    this.render();
  }
}
