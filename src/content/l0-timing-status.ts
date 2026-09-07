import { ensureUiStyles, themeRoot, applyComponent } from '@nominy/babel-extension-frontend';
import type { L0TimingQueueStatus } from '../core/l0-timing-client';

const STATUS_ID = 'babel-gold-l0-timing-status';
const STYLE_ID = 'babel-gold-l0-timing-status-style';
const SHOW_DELAY_MS = 1_500;

export type L0TimingDisplayStatus = L0TimingQueueStatus | { status: 'retrying' };

function ensureStatusStyles(documentRef: Document): void {
    ensureUiStyles(documentRef);
    if (documentRef.getElementById(STYLE_ID)) return;
    const style = documentRef.createElement('style');
    style.id = STYLE_ID;
    style.textContent = `#${STATUS_ID} { position: fixed; right: 18px; bottom: 18px; pointer-events: none; }`;
    (documentRef.head ?? documentRef.documentElement).append(style);
  }

function statusCopy(status: L0TimingDisplayStatus): string | null {
  switch (status.status) {
    case 'preparing':
      return 'Preparing word timing…';
    case 'queued':
      return `In transcription queue · #${status.position}`;
    case 'running':
      return 'Generating word timing…';
    case 'retrying':
      return 'Retrying word timing…';
    case 'completed':
      return null;
  }
}

export class L0TimingStatusPill {
  private activeTaskId: string | null = null;
  private latestStatus: L0TimingDisplayStatus | null = null;
  private showTimer: number | null = null;

  constructor(
    private readonly documentRef: Document = document,
    private readonly delayMs = SHOW_DELAY_MS,
    private readonly schedule: (callback: () => void, delayMs: number) => number = (callback, delayMs) =>
      window.setTimeout(callback, delayMs),
    private readonly cancel: (timerId: number) => void = (timerId) => window.clearTimeout(timerId)
  ) {}

  activateTask(taskId: string): void {
    if (this.activeTaskId === taskId) return;
    this.removeVisibleStatus();
    this.cancelPendingShow();
    this.latestStatus = null;
    this.activeTaskId = taskId;
  }

  update(taskId: string, status: L0TimingDisplayStatus): void {
    this.activateTask(taskId);
    if (status.status === 'completed') return;
    this.latestStatus = status;
    if (this.documentRef.getElementById(STATUS_ID)) {
      this.renderLatestStatus();
      return;
    }
    if (this.showTimer === null) {
      this.showTimer = this.schedule(() => {
        this.showTimer = null;
        this.renderLatestStatus();
      }, this.delayMs);
    }
  }

  clear(taskId?: string): void {
    if (taskId !== undefined && taskId !== this.activeTaskId) return;
    this.cancelPendingShow();
    this.removeVisibleStatus();
    this.latestStatus = null;
    this.activeTaskId = null;
  }

  private renderLatestStatus(): void {
    if (!this.latestStatus) return;
    const copy = statusCopy(this.latestStatus);
    if (!copy) return;
    ensureStatusStyles(this.documentRef);
    let pill = this.documentRef.getElementById(STATUS_ID);
    if (!pill) {
      pill = this.documentRef.createElement('div');
      pill.id = STATUS_ID;
      themeRoot(pill, 'purple');
      applyComponent(pill, 'toast');
      pill.setAttribute('role', 'status');
      pill.setAttribute('aria-live', 'polite');
      pill.setAttribute('aria-atomic', 'true');
      const copyNode = this.documentRef.createElement('span');
      copyNode.className = 'babel-gold-l0-timing-copy';
      const activity = this.documentRef.createElement('div');
      activity.className = 'babel-gold-l0-timing-activity bui-progress';
      activity.dataset.indeterminate = 'true';
      const fill = this.documentRef.createElement('div');
      fill.className = 'bui-progress-fill';
      activity.append(fill);
      activity.setAttribute('aria-hidden', 'true');
      pill.append(copyNode, activity);
      (this.documentRef.body ?? this.documentRef.documentElement).append(pill);
    }
    const copyNode = pill.querySelector<HTMLElement>('.babel-gold-l0-timing-copy');
    if (copyNode) copyNode.textContent = copy;
  }

  private cancelPendingShow(): void {
    if (this.showTimer === null) return;
    this.cancel(this.showTimer);
    this.showTimer = null;
  }

  private removeVisibleStatus(): void {
    this.documentRef.getElementById(STATUS_ID)?.remove();
  }
}
