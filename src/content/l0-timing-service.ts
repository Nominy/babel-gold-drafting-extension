import { captureAudioTracksForDrafting } from '../core/audio-cues';
import { AUDIO_ENABLE_CAPTURE_MESSAGE_TYPE } from '../core/audio-intercept-protocol';
import { generateL0Timing, lookupL0Timing, prepareL0TimingTracks, type L0TimingQueueStatus, type L0TimingRequestCallbacks } from '../core/l0-timing-client';
import { generateLocalL0Draft, generateLocalL0Timing, LocalModelBridgeError } from '../core/local-model-client';
import { loadSettings } from '../core/settings';
import { buildCanonicalTaskIdentity, captureTranscriptJob } from '../core/transcript';
import type { CapturedAudioTrack, ExtensionSettings, L0TimingResponse, TranscriptJob } from '../core/types';
import {
  publishL0TimingAvailability,
  setL0TimingRetryHandler,
  subscribeL0TimingAvailability,
  type L0TimingAvailability
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
  audioWaitCount: number;
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
  currentPathname: () => string;
  captureAudio: () => Promise<CapturedAudioTrack[]>;
  getSettings: () => Promise<ExtensionSettings>;
  lookupTiming: (settings: ExtensionSettings, taskId: string) => Promise<L0TimingResponse | null>;
  requestLocalDraft: typeof generateLocalL0Draft;
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
    tracks: Array<Pick<L0TimingResponse['tracks'][number], 'lane' | 'tokens'>>;
  }) => void;
  now: () => number;
  schedule: (callback: () => void, delayMs: number) => void;
}

export function isUsableL0TimingJob(job: TranscriptJob): boolean {
  if (!job.jobId.trim()) return false;
  if (!job.rows.length) return job.taskScoped === true;
  return job.rows.some((row) => Boolean(row.speakerKey.trim()));
}

export class L0TimingService {
  private readonly taskStates = new Map<string, TimingTaskState>();
  private readonly taskChecks = new Set<() => void>();

  constructor(private readonly dependencies: L0TimingServiceDependencies) {}

  onLifecycleOpportunity(): void {
    for (const check of this.taskChecks) check();
    let job: TranscriptJob;
    let taskId: string;
    try {
      job = this.dependencies.captureTranscript();
      taskId = buildCanonicalTaskIdentity(job);
      if (this.dependencies.currentTaskId() !== taskId) {
        return;
      }
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
      return;
    }
    if (state.failureCount > MAX_AUTOMATIC_RETRIES) {
      publishL0TimingAvailability({ taskId, status: 'unavailable' });
      return;
    }
    if (state.retryScheduled || this.dependencies.now() < state.retryNotBefore) {
      publishL0TimingAvailability({ taskId, status: state.audioWaitCount ? 'preparing' : 'retrying' });
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
      retryScheduled: false,
      audioWaitCount: 0
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
      if (!settings.localModelsEnabled) {
        const cached = await this.dependencies.lookupTiming(settings, taskId);
        if (cached) {
          if (this.isTaskCurrent(taskId)) this.publishTiming(taskId, cached, state);
          return;
        }
      }
      if (!this.isTaskCurrent(taskId)) return;
      const audioTracks = (await this.dependencies.captureAudio()).filter((track) => track.blob.size > 0);
      if (!this.isTaskCurrent(taskId)) {
        return;
      }
      // Transcript rows can mount before WaveSurfer has registered both lanes.
      // Wait for capture readiness before starting a model or spending an ASR retry.
      try {
        prepareL0TimingTracks(job, audioTracks);
      } catch {
        state.audioWaitCount += 1;
        publishL0TimingAvailability({ taskId, status: 'preparing' });
        state.retryScheduled = true;
        try {
          this.dependencies.schedule(() => {
            state.retryScheduled = false;
            if (this.isTaskCurrent(taskId)) this.onLifecycleOpportunity();
          }, Math.min(INITIAL_RETRY_DELAY_MS * state.audioWaitCount, 30_000));
        } catch {
          state.retryScheduled = false;
        }
        return;
      }
      state.audioWaitCount = 0;
      const response = await this.dependencies.requestTiming(settings, job, audioTracks, {
        onQueueStatus: (status: L0TimingQueueStatus) => {
          if (!this.isTaskCurrent(taskId)) return;
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
      this.publishTiming(taskId, response, state);
    } catch (error) {
      if (!this.isTaskCurrent(taskId)) return;
      console.error(
        `[Babel Gold] word timing attempt ${state.failureCount + 1} failed for task ${taskId}.`,
        error
      );
      this.scheduleRetry(taskId, state);
    } finally {
      state.inFlight = false;
    }
  }

  private publishTiming(taskId: string, response: L0TimingResponse, state: TimingTaskState): void {
    this.dependencies.publish({
      type: L0_TIMING_UPDATE_MESSAGE_TYPE,
      version: 1,
      taskId,
      tracks: response.tracks.map(({ lane, tokens }) => ({ lane, tokens }))
    });
    publishL0TimingAvailability({ taskId, status: 'available' });
    state.completed = true;
  }

  waitForTiming(job: TranscriptJob, settings: ExtensionSettings): Promise<void> {
    const taskId = buildCanonicalTaskIdentity(job);
    const pathname = this.dependencies.currentPathname();
    const isCurrent = () => this.dependencies.currentPathname() === pathname && this.isTaskCurrent(taskId);
    const changedError = () => new Error('The task changed while waiting for L0 timing.');
    if (!isCurrent()) return Promise.reject(changedError());
    return new Promise<void>((resolve, reject) => {
      let settled = false;
      let subscribing = true;
      let unsubscribe: (() => void) | undefined;
      const finish = (error?: Error) => {
        if (settled) return;
        settled = true;
        unsubscribe?.();
        this.taskChecks.delete(checkTask);
        if (error) reject(error);
        else resolve();
      };
      const checkTask = () => {
        if (!isCurrent()) finish(changedError());
      };
      const onAvailability = (availability: L0TimingAvailability) => {
        if (availability.taskId !== taskId) {
          // The subscription replays the last task before this one has had a lifecycle opportunity.
          if (!subscribing) finish(changedError());
          return;
        }
        if (!isCurrent()) {
          finish(changedError());
        } else if (availability.status === 'unavailable') {
          finish(new Error(`L0 timing for task ${taskId} is unavailable.`));
        } else if (availability.status === 'available' ||
            (!settings.localModelsEnabled && (availability.status === 'queued' || availability.status === 'running'))) {
          finish();
        }
      };
      this.taskChecks.add(checkTask);
      unsubscribe = subscribeL0TimingAvailability(onAvailability);
      subscribing = false;
      if (settled) unsubscribe();
      else this.onLifecycleOpportunity();
    });
  }

  async generateLocalDraft(settings: ExtensionSettings, job: TranscriptJob) {
    const taskId = buildCanonicalTaskIdentity(job);
    if (!this.isTaskCurrent(taskId)) throw new Error('The task changed while drafting.');
    try {
      return await this.dependencies.requestLocalDraft(settings, job);
    } catch (error) {
      if (!(error instanceof LocalModelBridgeError) || error.code !== 'timing-unavailable') throw error;
      if (!this.isTaskCurrent(taskId)) throw new Error('The task changed while drafting.');
      // Offscreen timing is bounded and is lost on document restart. Reuse the normal
      // capture/timing lifecycle instead of treating the content-side completed flag as durable.
      const state = this.getTaskState(taskId);
      state.completed = false;
      state.failureCount = 0;
      state.audioWaitCount = 0;
      state.retryNotBefore = 0;
      publishL0TimingAvailability({ taskId, status: 'preparing' });
      await this.waitForTiming(job, settings);
      if (!this.isTaskCurrent(taskId)) throw new Error('The task changed while drafting.');
      return this.dependencies.requestLocalDraft(settings, job);
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
    state.audioWaitCount = 0;
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

let activeTimingService: L0TimingService | null = null;

export function waitForCurrentL0Timing(job: TranscriptJob, settings: ExtensionSettings): Promise<void> {
  if (!activeTimingService) throw new Error('L0 timing service is not initialized.');
  return activeTimingService.waitForTiming(job, settings);
}

export function generateCurrentLocalL0Draft(settings: ExtensionSettings, job: TranscriptJob) {
  if (!activeTimingService) throw new Error('L0 timing service is not initialized.');
  return activeTimingService.generateLocalDraft(settings, job);
}

export function registerL0TimingService(): L0TimingService {
  const service = new L0TimingService({
    captureTranscript: () => captureTranscriptJob(),
    currentTaskId: () => buildCanonicalTaskIdentity(captureTranscriptJob()),
    currentPathname: () => window.location.pathname,
    captureAudio: () => captureAudioTracksForDrafting(),
    getSettings: () => loadSettings(),
    lookupTiming: lookupL0Timing,
    requestLocalDraft: generateLocalL0Draft,
    requestTiming: requestConfiguredL0Timing,
    publish: (message) => window.postMessage(message, '*'),
    now: () => Date.now(),
    schedule: (callback, delayMs) => {
      window.setTimeout(callback, delayMs);
    }
  });
  setL0TimingRetryHandler(() => service.retryCurrentTask());
  activeTimingService = service;
  return service;
}
