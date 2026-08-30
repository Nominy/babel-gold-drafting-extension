import { captureAudioTracksForDrafting } from '../core/audio-cues';
import { AUDIO_ENABLE_CAPTURE_MESSAGE_TYPE } from '../core/audio-intercept-protocol';
import { generateL0Timing, type L0TimingQueueStatus, type L0TimingRequestCallbacks } from '../core/l0-timing-client';
import { generateLocalL0Timing } from '../core/local-model-client';
import { loadSettings } from '../core/settings';
import { buildCanonicalTaskIdentity, captureTranscriptJob } from '../core/transcript';
import type { CapturedAudioTrack, ExtensionSettings, L0TimingResponse, TranscriptJob } from '../core/types';
import { L0TimingStatusPill, type L0TimingDisplayStatus } from './l0-timing-status';
import {
  publishL0TimingAvailability,
  setL0TimingRetryHandler
} from './l0-timing-availability';

export const L0_TIMING_UPDATE_MESSAGE_TYPE = 'babel-gold-drafting:l0-timing-update';
const INITIAL_RETRY_DELAY_MS = 5_000;
const MAX_RETRY_DELAY_MS = 120_000;
const MAX_AUTOMATIC_RETRIES = 3;

type TimingTaskState = {
  completed: boolean;
  inFlight: boolean;
  failureCount: number;
  retryNotBefore: number;
  retryScheduled: boolean;
};
export type L0TimingGenerators = {
  remote: typeof generateL0Timing;
  local: typeof generateLocalL0Timing;
};

const DEFAULT_L0_TIMING_GENERATORS: L0TimingGenerators = {
  remote: generateL0Timing,
  local: generateLocalL0Timing
};

export function requestConfiguredL0Timing(
  settings: ExtensionSettings,
  job: TranscriptJob,
  tracks: CapturedAudioTrack[],
  callbacks: L0TimingRequestCallbacks,
  generators: L0TimingGenerators = DEFAULT_L0_TIMING_GENERATORS
): Promise<L0TimingResponse> {
  return settings.localModelsEnabled
    ? generators.local(settings, job, tracks, callbacks)
    : generators.remote(settings, job, tracks, callbacks);
}


export interface L0TimingServiceDependencies {
  captureTranscript: () => TranscriptJob;
  currentTaskId: () => string;
  captureAudio: () => Promise<CapturedAudioTrack[]>;
  getSettings: () => Promise<ExtensionSettings>;
  requestTiming: (
    settings: ExtensionSettings,
    job: TranscriptJob,
    tracks: CapturedAudioTrack[],
    callbacks: L0TimingRequestCallbacks
  ) => Promise<L0TimingResponse>;
  publish: (message: {
    type: typeof L0_TIMING_UPDATE_MESSAGE_TYPE;
    version: 1;
    taskId: string;
    tracks: L0TimingResponse['tracks'];
  }) => void;
  activateStatusTask: (taskId: string) => void;
  updateStatus: (taskId: string, status: L0TimingDisplayStatus) => void;
  clearStatus: (taskId: string) => void;
  now: () => number;
  schedule: (callback: () => void, delayMs: number) => void;
}

export function isUsableL0TimingJob(job: TranscriptJob): boolean {
  if (!job.jobId.trim() || job.rows.length === 0) {
    return false;
  }
  return job.rows.some((row) => Boolean(row.speakerKey.trim()));
}

export class L0TimingService {
  private readonly taskStates = new Map<string, TimingTaskState>();

  constructor(private readonly dependencies: L0TimingServiceDependencies) {}

  onLifecycleOpportunity(): void {
    let job: TranscriptJob;
    let taskId: string;
    try {
      job = this.dependencies.captureTranscript();
      taskId = buildCanonicalTaskIdentity(job);
      if (this.dependencies.currentTaskId() !== taskId) {
        return;
      }
      this.dependencies.activateStatusTask(taskId);
      if (!isUsableL0TimingJob(job)) {
        publishL0TimingAvailability({ taskId, status: 'unavailable' });
        return;
      }
    } catch {
      return;
    }

    const state = this.getTaskState(taskId);
    if (state.completed) {
      publishL0TimingAvailability({ taskId, status: 'available' });
      return;
    }
    if (state.inFlight) {
      publishL0TimingAvailability({ taskId, status: 'preparing' });
      return;
    }
    if (state.retryScheduled || this.dependencies.now() < state.retryNotBefore) {
      publishL0TimingAvailability({ taskId, status: 'retrying' });
      return;
    }

    state.inFlight = true;
    publishL0TimingAvailability({ taskId, status: 'preparing' });
    void this.runAttempt(job, taskId, state).catch(() => undefined);
  }

  private getTaskState(taskId: string): TimingTaskState {
    const existing = this.taskStates.get(taskId);
    if (existing) {
      return existing;
    }
    const created: TimingTaskState = {
      completed: false,
      inFlight: false,
      failureCount: 0,
      retryNotBefore: 0,
      retryScheduled: false
    };
    this.taskStates.set(taskId, created);
    return created;
  }

  private isTaskCurrent(taskId: string): boolean {
    try {
      return this.dependencies.currentTaskId() === taskId;
    } catch {
      return false;
    }
  }

  private scheduleRetry(taskId: string, state: TimingTaskState): void {
    if (!this.isTaskCurrent(taskId)) {
      return;
    }
    this.dependencies.updateStatus(taskId, { status: 'retrying' });
    publishL0TimingAvailability({ taskId, status: 'retrying' });
    state.failureCount += 1;
    if (state.failureCount > MAX_AUTOMATIC_RETRIES) {
      state.retryNotBefore = Number.POSITIVE_INFINITY;
      publishL0TimingAvailability({ taskId, status: 'unavailable' });
      return;
    }
    const delayMs = Math.min(INITIAL_RETRY_DELAY_MS * 2 ** (state.failureCount - 1), MAX_RETRY_DELAY_MS);
    if (state.retryScheduled) {
      return;
    }
    state.retryScheduled = true;
    try {
      this.dependencies.schedule(() => {
        state.retryScheduled = false;
        this.onLifecycleOpportunity();
      }, delayMs);
    } catch {
      state.retryScheduled = false;
    }
  }

  private async runAttempt(job: TranscriptJob, taskId: string, state: TimingTaskState): Promise<void> {
    try {
      const settings = await this.dependencies.getSettings();
      const audioTracks = await this.dependencies.captureAudio();
      if (!this.isTaskCurrent(taskId)) {
        return;
      }
      const response = await this.dependencies.requestTiming(settings, job, audioTracks, {
        onQueueStatus: (status: L0TimingQueueStatus) => {
          if (!this.isTaskCurrent(taskId)) return;
          this.dependencies.updateStatus(taskId, status);
          if (status.status === 'queued') {
            publishL0TimingAvailability({
              taskId,
              status: 'queued',
              position: status.position
            });
          } else if (status.status === 'running') {
            publishL0TimingAvailability({ taskId, status: 'running' });
          } else if (status.status === 'preparing') {
            publishL0TimingAvailability({ taskId, status: 'preparing' });
          }
        }
      });
      if (!this.isTaskCurrent(taskId) || response.taskId !== taskId) {
        return;
      }
      this.dependencies.clearStatus(taskId);
      this.dependencies.publish({
        type: L0_TIMING_UPDATE_MESSAGE_TYPE,
        version: 1,
        taskId,
        tracks: response.tracks
      });
      publishL0TimingAvailability({ taskId, status: 'available' });
      state.completed = true;
    } catch (error) {
      console.error(
        `[Babel Gold] word timing attempt ${state.failureCount + 1} failed for task ${taskId}.`,
        error
      );
      this.scheduleRetry(taskId, state);
    } finally {
      state.inFlight = false;
    }
  }

  retryCurrentTask(): boolean {
    let job: TranscriptJob;
    let taskId: string;
    try {
      job = this.dependencies.captureTranscript();
      taskId = buildCanonicalTaskIdentity(job);
    } catch {
      return false;
    }
    if (!isUsableL0TimingJob(job) || !this.isTaskCurrent(taskId)) return false;
    const state = this.getTaskState(taskId);
    if (state.completed || state.inFlight || state.retryScheduled) return false;
    state.failureCount = 0;
    state.retryNotBefore = 0;
    publishL0TimingAvailability({ taskId, status: 'preparing' });
    this.onLifecycleOpportunity();
    return true;
  }
}

export function enableL0TimingAudioCapture(): void {
  try {
    window.postMessage({ type: AUDIO_ENABLE_CAPTURE_MESSAGE_TYPE }, '*');
  } catch {
    // Timing capture is intentionally invisible and must not affect the drafting surface.
  }
}

export function registerL0TimingService(): L0TimingService {
  const statusPill = new L0TimingStatusPill();
  const service = new L0TimingService({
    captureTranscript: () => captureTranscriptJob(),
    currentTaskId: () => buildCanonicalTaskIdentity(captureTranscriptJob()),
    captureAudio: () => captureAudioTracksForDrafting(),
    getSettings: () => loadSettings(),
    requestTiming: requestConfiguredL0Timing,
    publish: (message) => window.postMessage(message, '*'),
    activateStatusTask: (taskId) => statusPill.activateTask(taskId),
    updateStatus: (taskId, status) => statusPill.update(taskId, status),
    clearStatus: (taskId) => statusPill.clear(taskId),
    now: () => Date.now(),
    schedule: (callback, delayMs) => {
      window.setTimeout(callback, delayMs);
    }
  });
  setL0TimingRetryHandler(() => service.retryCurrentTask());
  return service;
}
