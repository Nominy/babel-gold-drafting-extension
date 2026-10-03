import { ensureUiStyles, themeRoot, applyComponent } from '@nominy/babel-extension-frontend';
import { generateDraftStream } from '../core/backend-client';
import { generateL0Draft } from '../core/l0-client';
import { generateMaiL0Draft, MaiBridgeError } from '../core/mai-client';
import { replaceTranscriptWithL0Rows, requireL0ReplacementConsumer } from '../core/l0-replacement-bridge';
import { matchL0CreatedRows } from '../core/l0-created-row-matcher';
import { assessAudioCaptureForDrafting, type AudioCaptureIssue } from '../core/audio-capture-guard';
import { captureAudioTracksForDrafting } from '../core/audio-cues';
import { isBrowserLocalMode, loadSettings } from '../core/settings';
import { applyDraftRows, buildCanonicalTaskIdentity, buildDiffPreviewItems, captureTranscriptJob, restoreCapturedRows } from '../core/transcript';
import { generateCurrentLocalL0Draft, getCurrentL0TimingGeneration, waitForCurrentL0Timing } from './l0-timing-service';
import {
  getL0TimingAvailability,
  requestL0TimingRegeneration,
  subscribeL0TimingAvailability,
  type L0TimingAvailability
} from './l0-timing-availability';
import { readPublishedPageTaskId } from './page-task-identity';
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
export function shouldRunGoldLlmAfterL0(settings: ExtensionSettings): boolean {
  return settings.mode !== 'simple' && !settings.l0DontRunLlm &&
    (settings.mode === 'advanced' || Boolean(settings.openRouterApiKey.trim()));
}

export type L0DraftGenerators = {
  remote: typeof generateL0Draft;
  local: typeof generateCurrentLocalL0Draft;
  mai: typeof generateMaiL0Draft;
};

const DEFAULT_L0_DRAFT_GENERATORS: L0DraftGenerators = {
  remote: generateL0Draft,
  local: generateCurrentLocalL0Draft,
  mai: generateMaiL0Draft
};

export function generateConfiguredL0Draft(
  settings: ExtensionSettings,
  job: TranscriptJob,
  generators: L0DraftGenerators = DEFAULT_L0_DRAFT_GENERATORS
) {
  if (settings.mode === 'simple') return generators.mai(settings, job);
  return isBrowserLocalMode(settings)
    ? generators.local(settings, job)
    : generators.remote(settings, job);
}


function createDraftSessionId(jobId: string): string {
  const randomId =
    typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  return `${jobId}:${randomId}`;
}

export interface DialogTask {
  pathname: string;
  reviewActionId: string;
}

/**
 * A dialog belongs to the task it was opened on. Search and hash are router
 * noise on live; only a different pathname or a different, non-empty published
 * review action ID means the user is now on another task.
 */
export function hasDialogTaskChanged(
  task: DialogTask,
  location: Pick<Location, 'pathname'>,
  publishedReviewActionId: string
): boolean {
  if (task.pathname !== location.pathname) return true;
  return Boolean(task.reviewActionId && publishedReviewActionId && publishedReviewActionId !== task.reviewActionId);
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
  ensureUiStyles();
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement('style');
  style.id = STYLE_ID;
  style.textContent = `
    .bgd-toolbar-button[data-state="loading"] { cursor: wait; }
    .bgd-toolbar-button .bgd-spinner { display: none; }
    .bgd-toolbar-button[data-state="loading"] .bgd-spinner { display: inline-block; }
    .bgd-toolbar-button[data-state="loading"] .bgd-icon { display: none; }
    .bgd-toolbar-button .bgd-icon { font-size: 16px; line-height: 1; }
    .bgd-timing-hover-panel { position: fixed; z-index: 2147483647; display: flex; align-items: center;
      opacity: 0; pointer-events: none; }
    .bgd-timing-hover-panel[data-open="true"] { opacity: 1; pointer-events: auto; }
    .bgd-timing-dot { display: none; }
    .bgd-timing-retry { margin-left: 10px; }
    .bgd-diff-list { max-height: 56vh; overflow: auto; }
    .bgd-card-top { justify-content: space-between; }
    .bgd-card.failed { border-color: var(--bui-danger); }
  `;
  document.documentElement.appendChild(style);
}

export class DraftingOverlayController {
  private button: HTMLButtonElement | null = null;
  private overlay: HTMLDivElement | null = null;
  private dialogEl: HTMLDivElement | null = null;
  private dialogTask: DialogTask | null = null;
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
  };
  private streamedRows: DraftRowResult[] = [];
  private streamedSummary: DraftSummary | null = null;
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
    if (this.dialogTask && hasDialogTaskChanged(this.dialogTask, window.location, readPublishedPageTaskId())) {
      this.dialogTask = null;
      if (this.overlay) this.overlay.hidden = true;
      this.cancelTimingPanelHide();
      if (this.button) { this.button.dataset.timingOpen = 'false'; this.button.dataset.attachedOpen = 'false'; }
      if (this.timingPanel) this.timingPanel.dataset.open = 'false';
    }
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
    this.dialogTask = null;
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
    button.className = 'bgd-toolbar-button bui-icon-button';
    themeRoot(button, 'purple');
    applyComponent(button, 'icon-button', { variant: 'soft' });
    button.setAttribute('aria-label', 'Gold Draft');
    button.innerHTML = `
      <span class="bgd-icon">\u{1FA84}</span>
      <span class="bgd-spinner bui-spinner"></span>
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
    const panel = createElement('div', 'bgd-timing-hover-panel bui-attached-status');
    themeRoot(panel, 'purple');
    panel.dataset.open = 'false';
    panel.setAttribute('role', 'status');
    panel.setAttribute('aria-live', 'polite');
    const state = createElement('div', 'bgd-timing-state bui-row');
    state.append(
      createElement('span', 'bgd-timing-dot'),
      createElement('span', 'bgd-timing-copy bui-status-copy')
    );
    const activity = createElement('div', 'bgd-timing-activity bui-progress');
    activity.dataset.indeterminate = 'true';
    activity.style.width = '140px';
    activity.setAttribute('aria-hidden', 'true');
    activity.append(createElement('div', 'bui-progress-fill'));
    activity.hidden = true;
    state.append(activity);
    const retry = createElement('button', 'bgd-timing-retry bui-button', 'Regenerate timestamp data');
    retry.type = 'button';
    retry.addEventListener('click', () => {
      void loadSettings().then(async (settings) => {
        if (settings.mode === 'simple') {
          await waitForCurrentL0Timing(captureTranscriptJob(), settings);
        } else if (requestL0TimingRegeneration()) {
          this.timingAvailability = getL0TimingAvailability();
          this.renderTimingPanel();
        }
      }).catch((error) => {
        this.openDialog();
        this.setStatus(error instanceof Error ? error.message : String(error), true);
      });
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
    if (this.button) { this.button.dataset.timingOpen = 'true'; this.button.dataset.attachedOpen = 'true'; }
    panel.dataset.open = 'true';
  }

  private scheduleTimingPanelHide(): void {
    this.cancelTimingPanelHide();
    this.timingPanelHideTimer = window.setTimeout(() => {
      this.timingPanelHideTimer = null;
      if (this.button) { this.button.dataset.timingOpen = 'false'; this.button.dataset.attachedOpen = 'false'; }
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
      unavailable: availability?.status === 'unavailable' && availability.error
        ? `Timestamp data not available: ${availability.error}`
        : 'Timestamp data not available',
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
    const loading = ['preparing', 'queued', 'running', 'retrying'].includes(status);
    if (copy) { copy.textContent = copyByStatus[status]; copy.hidden = loading; }
    this.timingPanel.setAttribute('aria-label', copyByStatus[status]);
    const activity = this.timingPanel.querySelector<HTMLElement>('.bgd-timing-activity');
    if (activity) activity.hidden = !loading;
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
    themeRoot(overlay, 'purple');
    applyComponent(overlay, 'overlay', { variant: 'tinted' });
    overlay.hidden = true;

    const backdrop = createElement('div', 'bgd-backdrop bui-backdrop');
    backdrop.addEventListener('click', () => this.closeDialog());

    const shell = createElement('div', 'bgd-shell bui-dialog-position');
    const dialog = createElement('div', 'bgd-dialog bui-dialog');
    dialog.addEventListener('click', (event) => event.stopPropagation());

    const header = createElement('div', 'bgd-header bui-header');
    const titleWrap = createElement('div');
    const titleRow = createElement('div', 'bgd-header-title-row bui-row');
    const supportLink = createElement(
      'a',
      'bgd-support-link bui-link',
      'if this extension saves you time, consider supporting development on Ko-Fi',
    );
    supportLink.href = 'https://ko-fi.com/naftsan';
    supportLink.target = '_blank';
    supportLink.rel = 'noopener noreferrer';
    titleRow.append(createElement('div', 'bgd-header-title bui-title', 'Gold Draft'), supportLink);
    titleWrap.append(titleRow, createElement('div', 'bgd-header-subtitle bui-subtitle', 'Generated transcript; review before saving or submitting.'));
    const closeButton = createElement('button', 'bgd-close bui-button', 'Close');
    closeButton.type = 'button';
    closeButton.addEventListener('click', () => this.closeDialog());
    header.append(titleWrap, closeButton);

    const main = createElement('div', 'bgd-main bui-body');
    const statusEl = createElement('div', 'bgd-status bui-status', 'Click the wand to generate a draft.');
    const activity = createElement('div', 'bgd-activity bui-progress');
    activity.dataset.indeterminate = 'true';
    activity.setAttribute('aria-hidden', 'true');
    activity.hidden = !this.busy;
    activity.append(createElement('div', 'bui-progress-fill'));
    const audioGuardEl = createElement('div', 'bgd-audio-guard bui-notice');
    audioGuardEl.dataset.tone = 'warning';
    audioGuardEl.hidden = true;

    const summaryBlock = createElement('section', 'bgd-block bui-panel');
    summaryBlock.append(createElement('div', 'bgd-block-header bui-section-title', 'Summary'));
    const summaryBody = createElement('div', 'bgd-block-body bui-body');
    const summaryEl = createElement('div', 'bgd-empty bui-empty', 'No draft generated yet.');
    summaryBody.append(summaryEl);
    summaryBlock.append(summaryBody);

    const previewBlock = createElement('section', 'bgd-block bui-panel');
    previewBlock.append(createElement('div', 'bgd-block-header bui-section-title', 'Diff Preview'));
    const previewBody = createElement('div', 'bgd-block-body bui-body');
    const previewEl = createElement('div', 'bgd-empty bui-empty', 'The diff will appear here after generation.');
    previewBody.append(previewEl);
    previewBlock.append(previewBody);

    main.append(statusEl, activity, audioGuardEl, summaryBlock, previewBlock);

    const footer = createElement('div', 'bgd-footer bui-footer');
    const restoreButton = createElement('button', 'bgd-button bui-button', 'Restore Original');
    restoreButton.type = 'button';
    restoreButton.addEventListener('click', () => void this.restoreOriginal());
    const applyButton = createElement('button', 'bgd-button bui-button', 'Apply Draft');
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
    this.button.setAttribute('aria-label', label);
  }

  private setBusy(nextBusy: boolean): void {
    this.busy = nextBusy;
    const activity = this.overlay?.querySelector<HTMLElement>('.bgd-activity');
    if (activity) activity.hidden = !nextBusy;
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
    const actions = createElement('div', 'bgd-audio-actions bui-row');
    const retryButton = createElement('button', 'bgd-button bui-button', 'Retry Audio');
    retryButton.type = 'button';
    retryButton.dataset.bgdAudioAction = 'retry';
    retryButton.addEventListener('click', () => void this.retryPendingAudioCapture());

    actions.append(retryButton);

    if (issue.kind === 'partial' && audioTracks.length) {
      const usePartialButton = createElement('button', 'bgd-button bui-button', 'Use Captured Audio');
      usePartialButton.type = 'button';
      usePartialButton.dataset.bgdAudioAction = 'use-partial';
      usePartialButton.addEventListener('click', () => void this.continuePendingDraft('captured-audio'));
      actions.append(usePartialButton);
    }

    const textOnlyButton = createElement('button', 'bgd-button bui-button', 'Continue Text Only');
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
      this.summaryEl.className = 'bgd-empty bui-empty';
      this.summaryEl.textContent = 'No transcript captured yet.';
      return;
    }

    const summary = createElement('div', 'bgd-summary bui-row');
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
    const pill = createElement('div', 'bgd-summary-pill bui-row');
    pill.append(createElement('span', 'bgd-summary-dot bui-dot'), createElement('span', '', text));
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
      this.previewEl.className = 'bgd-empty bui-empty';
      this.previewEl.textContent = 'The diff will appear here after generation.';
      return;
    }

    const sourceRows = draft ? draft.draftRows : this.streamedRows;
    if (!sourceRows.length) {
      this.previewEl.className = 'bgd-empty bui-empty';
      this.previewEl.textContent = 'Waiting for the first completed row...';
      return;
    }

    const diffItems = buildDiffPreviewItems(captured.rows, sourceRows);
    if (!diffItems.length) {
      this.previewEl.className = 'bgd-empty bui-empty';
      this.previewEl.textContent = draft ? 'No row text changed.' : 'Completed rows have no visible text changes yet.';
      return;
    }

    const list = createElement('div', 'bgd-diff-list bui-stack');
    for (const item of diffItems.slice(0, 40)) {
      const card = createElement('article', `bgd-card bui-card ${item.status === 'failed' ? 'failed' : ''}`);
      const top = createElement('div', 'bgd-card-top bui-row');
      top.append(
        createElement('div', 'bgd-card-title bui-title', `Row ${item.index + 1}`),
        createElement('div', 'bgd-badge bui-badge', item.status)
      );
      card.append(top);

      const diffView = createElement('div', 'bgd-diff-view bui-diff');
      const beforePane = createElement('div', 'bgd-diff-pane bui-diff-pane');
      beforePane.append(
        createElement('div', 'bgd-diff-label bui-diff-label', 'Before'),
        this.createDiffContent(item.before)
      );
      const afterPane = createElement('div', 'bgd-diff-pane bui-diff-pane');
      afterPane.append(
        createElement('div', 'bgd-diff-label bui-diff-label', 'After'),
        this.createDiffContent(item.after)
      );
      diffView.append(beforePane, afterPane);
      card.append(diffView);

      if (item.warnings.length) {
        const warnings = createElement('ul', 'bgd-warning-list bui-list');
        for (const warning of item.warnings) {
          warnings.append(createElement('li', '', warning));
        }
        card.append(warnings);
      }

      list.append(card);
    }

    if (diffItems.length > 40) {
      list.append(createElement('div', 'bgd-empty bui-empty', `Showing first 40 of ${diffItems.length} changed rows.`));
    }

    this.previewEl.className = '';
    this.previewEl.replaceChildren(list);
    list.scrollTop = previousListScrollTop;
  }

  private createDiffContent(text: string): HTMLDivElement {
    const content = createElement('div', 'bgd-diff-content bui-diff-text');
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
    this.dialogTask = { pathname: window.location.pathname, reviewActionId: readPublishedPageTaskId() };
    this.openDialog();
    this.activeDraftLabel = 'Gold / OpenRouter';
    this.clearAudioGuard();
    this.state = {
      capturedJob: null,
      draftResponse: null,
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
      const settings = await loadSettings();
      if (!capturedJob.rows.length && settings.mode === 'advanced' && !settings.l0ReplacementPreviewEnabled) {
        throw new Error('No transcript rows detected on this page.');
      }
      this.state.capturedJob = capturedJob;
      if (capturedJob.taskScoped) this.dialogTask = { pathname: window.location.pathname, reviewActionId: capturedJob.jobId };
      this.streamedTotalRows = capturedJob.rows.length;
      this.render();

      if (settings.mode !== 'advanced' || settings.l0ReplacementPreviewEnabled) {
        if (settings.mode === 'simple' && !settings.openRouterApiKey.trim()) {
          throw new Error('Simple mode requires an OpenRouter API key. Add your key in the Babel Gold Drafting extension options, then try again.');
        }
        this.activeDraftLabel = settings.mode === 'local'
          ? 'Local / GigaAM + C-denoise (WebGPU)'
          : settings.mode === 'simple'
            ? 'Cloud / MAI native transcription'
            : settings.l0DontRunLlm ? 'L0 replacement / no LLM' : 'L0 replacement -> Gold / OpenRouter';
        capturedJob = await this.runL0Replacement(capturedJob, settings);
        this.state.capturedJob = capturedJob;
        if (!shouldRunGoldLlmAfterL0(settings)) {
          this.setButtonState('done', settings.mode === 'local' ? 'Local C-denoise Draft Ready' : settings.mode === 'simple' ? 'Cloud Draft Ready' : 'L0 Replacement Ready');
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

  private requireCurrentTask(capturedJob: TranscriptJob): void {
    if (buildCanonicalTaskIdentity(captureTranscriptJob()) !== buildCanonicalTaskIdentity(capturedJob)) {
      throw new Error('The task changed while drafting. Discard this draft and generate again for the current task.');
    }
  }

  private async requireCurrentDraft(capturedJob: TranscriptJob, settings: ExtensionSettings, timingGeneration?: number): Promise<void> {
    this.requireCurrentTask(capturedJob);
    const saved = await loadSettings();
    if (saved.mode !== settings.mode ||
      (saved.mode === 'advanced' && saved.localModelsEnabled !== settings.localModelsEnabled) ||
      (timingGeneration !== undefined && getCurrentL0TimingGeneration() !== timingGeneration)) {
      throw new Error('The drafting mode changed. Discard this draft and generate again in the current mode.');
    }
    this.requireCurrentTask(capturedJob);
  }

  private async runL0Replacement(
    capturedJob: TranscriptJob,
    settings: ExtensionSettings
  ): Promise<TranscriptJob> {
    this.setStatus('Checking Babel Helper availability...');
    const helperConfirmed = await requireL0ReplacementConsumer();
    if (settings.mode !== 'advanced' && !helperConfirmed) {
      throw new Error('Babel Helper is required for native editor integration. Install or enable Babel Helper and reload this task before transcribing.');
    }
    if (!helperConfirmed) {
      this.setStatus('Babel Helper did not confirm readiness; continuing with a direct replacement request...');
    }
    await this.requireCurrentDraft(capturedJob, settings);
    this.setStatus(settings.mode === 'local'
      ? 'Preparing local GigaAM + C-denoise on WebGPU; no audio cloud fallback...'
      : settings.mode === 'simple'
        ? 'Transcribing with MAI native punctuation and word timing (cached results are reused)...'
        : 'Waiting for shared L0 word timing...');
    await waitForCurrentL0Timing(capturedJob, settings);
    const timingGeneration = getCurrentL0TimingGeneration();
    await this.requireCurrentDraft(capturedJob, settings, timingGeneration);
    this.setStatus(settings.mode === 'simple'
      ? 'Loading the cached native MAI draft; no Gold rewriting...'
      : isBrowserLocalMode(settings)
        ? 'Punctuating cached full-lane words with C-denoise on WebGPU...'
        : 'Punctuating cached segments with the self-hosted L0 endpoint...');
    let response;
    try {
      response = await generateConfiguredL0Draft(settings, capturedJob);
    } catch (error) {
      if (settings.mode !== 'simple' || !(error instanceof MaiBridgeError) || error.code !== 'timing-unavailable') throw error;
      await waitForCurrentL0Timing(capturedJob, settings);
      await this.requireCurrentDraft(capturedJob, settings, timingGeneration);
      response = await generateConfiguredL0Draft(settings, capturedJob);
    }
    await this.requireCurrentDraft(capturedJob, settings, timingGeneration);

    this.setStatus(`Replacing current transcript with ${response.rows.length} L0 segment(s) through Babel Helper...`);
    const created = await replaceTranscriptWithL0Rows(response.rows);
    await this.requireCurrentDraft(capturedJob, settings, timingGeneration);
    const createdIds = new Set(created.map((mapping) => mapping.id));
    if (createdIds.size !== response.rows.length || response.rows.some((row) => !createdIds.has(row.id))) {
      throw new Error('Babel Helper returned incomplete or duplicate L0 row mappings.');
    }
    const populatedJob = captureTranscriptJob();
    const matchedRows = matchL0CreatedRows(response.rows, populatedJob.rows);
    this.state.capturedJob = populatedJob;
    this.state.draftResponse = {
      draftRows: matchedRows.map(({ engineRow, capturedRow }) => ({
        rowId: capturedRow.rowId,
        rewrittenText: engineRow.text,
        status: 'rewritten',
        warnings: []
      })),
      summary: {
        totalRows: created.length,
        rewrittenRows: created.length,
        unchangedRows: 0,
        failedRows: 0,
        anomalyCounts: {}
      },
      generationMeta: {
        model: isBrowserLocalMode(settings) ? 'GigaAM + C-denoise / WebGPU' : settings.mode === 'simple' ? 'microsoft/mai-transcribe-2' : Object.keys(response.models).join(' + ') || 'L0 two-model engine',
        rulePackVersion: 'l0-replacement',
        generatedAt: new Date().toISOString()
      }
    };
    this.setStatus(
      settings.mode === 'local'
        ? `Local C-denoise complete: ${created.length} segment(s), computed on WebGPU. ${shouldRunGoldLlmAfterL0(settings) ? 'Continuing with Gold LLM drafting.' : 'Gold LLM drafting was skipped.'}`
        : settings.mode === 'simple'
          ? `Cloud transcription complete: ${created.length} native MAI segment(s). No Gold rewriting was run.`
          : settings.l0DontRunLlm
            ? `L0 replacement complete: ${created.length} segment(s). LLM drafting was skipped.`
            : `L0 replacement complete: ${created.length} segment(s). Recaptured transcript for Gold LLM drafting.`
    );
    this.render();
    return populatedJob;
  }


  private async generateDraftFromCapture(
    capturedJob: TranscriptJob,
    settings: ExtensionSettings,
    audioTracks: CapturedAudioTrack[]
  ): Promise<void> {
    this.requireCurrentTask(capturedJob);
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
    this.requireCurrentTask(capturedJob);

    this.state.draftResponse = draftResponse;
    this.setStatus(
      `Draft ready. ${draftResponse.summary.rewrittenRows} rewritten, ${draftResponse.summary.failedRows} failed fallback rows.`
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
    if (!this.state.capturedJob) return;
    try {
      this.requireCurrentTask(this.state.capturedJob);
    } catch (error) {
      this.setStatus(error instanceof Error ? error.message : String(error), true);
      return;
    }

    const result = applyDraftRows(this.state.draftResponse.draftRows);
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
    try {
      this.requireCurrentTask(this.state.capturedJob);
    } catch (error) {
      this.setStatus(error instanceof Error ? error.message : String(error), true);
      return;
    }

    const result = restoreCapturedRows(this.state.capturedJob);
    const missingNote = result.missingRowIds.length ? ` Missing ${result.missingRowIds.length} rows during restore.` : '';
    this.setStatus(`Restored ${result.appliedCount} original rows.${missingNote}`);
    this.setButtonState('done', 'Restored');
    window.setTimeout(() => this.setButtonState('idle', 'Gold Draft'), 1600);
    this.render();
  }
}
