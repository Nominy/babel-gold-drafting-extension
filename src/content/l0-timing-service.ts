import { captureAudioTracksForDrafting } from '../core/audio-cues';
import { AUDIO_ENABLE_CAPTURE_MESSAGE_TYPE } from '../core/audio-intercept-protocol';
import { generateL0Timing, lookupL0Timing, prepareL0TimingTracks, type L0TimingQueueStatus, type L0TimingRequestCallbacks } from '../core/l0-timing-client';
import { generateLocalL0Draft, generateLocalL0Timing, LocalModelBridgeError } from '../core/local-model-client';
import { generateMaiL0Timing, lookupMaiL0Timing } from '../core/mai-client';
import { isBrowserLocalMode, LOCAL_MODEL_BASE_URL, loadSettings, normalizeSettings, SETTINGS_STORAGE_KEY } from '../core/settings';
import { buildCanonicalTaskIdentity, captureTranscriptJob } from '../core/transcript';
import type { CapturedAudioTrack, ExtensionSettings, L0TimingResponse, TranscriptJob } from '../core/types';
import { getLocalModelStatus } from '../core/local-model-bundle';
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
  generation: number;
  attempt: Promise<void> | null;
  explicitAttempt: boolean;
  error: Error | null;
};
export type L0TimingGenerators = {
  remote: typeof generateL0Timing;
  local: typeof generateLocalL0Timing;
  mai: typeof generateMaiL0Timing;
};

const DEFAULT_L0_TIMING_GENERATORS: L0TimingGenerators = {
  remote: generateL0Timing,
  local: generateLocalL0Timing,
  mai: generateMaiL0Timing
};

export function requestConfiguredL0Timing(
  settings: ExtensionSettings,
  job: TranscriptJob,
  tracks: CapturedAudioTrack[],
  callbacks: L0TimingRequestCallbacks,
  generators: L0TimingGenerators = DEFAULT_L0_TIMING_GENERATORS
): Promise<L0TimingResponse> {
  if (settings.mode === 'simple') return generators.mai(settings, job, tracks, callbacks);
  return isBrowserLocalMode(settings)
    ? generators.local(settings, job, tracks, callbacks)
    : generators.remote(settings, job, tracks, callbacks);
}


export interface L0TimingServiceDependencies {
  captureTranscript: () => TranscriptJob;
  currentTaskId: () => string;
  currentPathname: () => string;
  captureAudio: () => Promise<CapturedAudioTrack[]>;
  getSettings: () => Promise<ExtensionSettings>;
  localModelStatus?: typeof getLocalModelStatus;
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
  private activeEngine = '';
  private generation = 0;
  private currentSettings: ExtensionSettings | null = null;
  private readonly manualRequests = new Set<string>();

  constructor(private readonly dependencies: L0TimingServiceDependencies) {}

  get generationId(): number {
    return this.generation;
  }

  onSettingsChanged(settings: ExtensionSettings): void {
    this.activateSettings(settings);
    this.onLifecycleOpportunity();
  }

  private engine(settings: ExtensionSettings): string {
    return settings.mode === 'simple' ? 'mai' : isBrowserLocalMode(settings) ? 'local' : 'remote';
  }

  private activateSettings(settings: ExtensionSettings): void {
    this.currentSettings = settings;
    const engine = this.engine(settings);
    if (engine === this.activeEngine) return;
    this.activeEngine = engine;
    this.generation += 1;
    for (const check of this.taskChecks) check();
  }

  onLifecycleOpportunity(): void {
    for (const check of this.taskChecks) check();
    void this.lifecycleOpportunity().catch(() => undefined);
  }

  private async lifecycleOpportunity(): Promise<void> {
    const job = this.dependencies.captureTranscript();
    const taskId = buildCanonicalTaskIdentity(job);
    const settings = await this.dependencies.getSettings();
    this.activateSettings(settings);
    if (!this.isTaskCurrent(taskId)) return;
    const state = this.getTaskState(taskId, settings);
    // Babel temporarily redraws rows without speaker labels. The canonical task
    // remains the same; its valid request/cache must survive that DOM transition.
    if (state.inFlight) return;
    if (!isUsableL0TimingJob(job) && !(state.completed && settings.mode !== 'simple')) {
      publishL0TimingAvailability({ taskId, status: 'unavailable' });
      return;
    }
    // Simple lifecycle only inspects the durable background cache. Even a previously
    // completed content state is not evidence that the session cache still exists.
    if (settings.mode === 'simple') {
      await this.startAttempt(job, taskId, settings, state, false);
      return;
    }
    if (state.completed) {
      publishL0TimingAvailability({ taskId, status: 'available' });
      return;
    }
    if (state.failureCount > MAX_AUTOMATIC_RETRIES) {
      publishL0TimingAvailability({ taskId, status: 'unavailable', ...(state.error ? { error: state.error.message } : {}) });
      return;
    }
    if (state.retryScheduled || this.dependencies.now() < state.retryNotBefore) {
      publishL0TimingAvailability({ taskId, status: state.audioWaitCount ? 'preparing' : 'retrying' });
      return;
    }
    await this.startAttempt(job, taskId, settings, state, false);
  }

  private getTaskState(taskId: string, settings: ExtensionSettings): TimingTaskState {
    const key = `${this.engine(settings)}:${taskId}`;
    const existing = this.taskStates.get(key);
    if (existing?.generation === this.generation) return existing;
    const created: TimingTaskState = {
      completed: false, inFlight: false, failureCount: 0, retryNotBefore: 0,
      retryScheduled: false, audioWaitCount: 0, generation: this.generation,
      attempt: null, error: null, explicitAttempt: false
    };
    this.taskStates.set(key, created);
    return created;
  }

  private isTaskCurrent(taskId: string): boolean {
    try {
      return this.dependencies.currentTaskId() === taskId;
    } catch {
      return false;
    }
  }

  private isOwnerCurrent(taskId: string, state: TimingTaskState): boolean {
    return state.generation === this.generation && this.isTaskCurrent(taskId);
  }

  private async verifyOwner(taskId: string, state: TimingTaskState): Promise<boolean> {
    this.activateSettings(await this.dependencies.getSettings());
    return this.isOwnerCurrent(taskId, state);
  }

  private scheduleRetry(taskId: string, state: TimingTaskState): void {
    if (!this.isOwnerCurrent(taskId, state)) return;
    publishL0TimingAvailability({ taskId, status: 'retrying' });
    state.failureCount += 1;
    if (state.failureCount > MAX_AUTOMATIC_RETRIES) {
      state.retryNotBefore = Number.POSITIVE_INFINITY;
      publishL0TimingAvailability({ taskId, status: 'unavailable', ...(state.error ? { error: state.error.message } : {}) });
      return;
    }
    const delayMs = Math.min(INITIAL_RETRY_DELAY_MS * 2 ** (state.failureCount - 1), MAX_RETRY_DELAY_MS);
    if (state.retryScheduled) return;
    state.retryScheduled = true;
    try {
      this.dependencies.schedule(() => {
        state.retryScheduled = false;
        if (this.isOwnerCurrent(taskId, state)) this.onLifecycleOpportunity();
      }, delayMs);
    } catch {
      state.retryScheduled = false;
    }
  }

  private startAttempt(
    job: TranscriptJob, taskId: string, settings: ExtensionSettings,
    state: TimingTaskState, explicit: boolean
  ): Promise<void> {
    if (state.attempt) return state.attempt;
    state.inFlight = true;
    state.explicitAttempt = explicit;
    state.error = null;
    if (explicit || settings.mode === 'advanced') publishL0TimingAvailability({ taskId, status: 'preparing' });
    const attempt = this.runAttempt(job, taskId, settings, state, explicit);
    state.attempt = attempt;
    return attempt;
  }

  private async runAttempt(
    job: TranscriptJob, taskId: string, settings: ExtensionSettings,
    state: TimingTaskState, explicit: boolean
  ): Promise<void> {
    try {
      if (settings.mode === 'simple' || !isBrowserLocalMode(settings)) {
        const cached = await this.dependencies.lookupTiming(settings, taskId);
        if (!(await this.verifyOwner(taskId, state))) return;
        if (cached?.taskId === taskId) {
          this.publishTiming(taskId, cached, state);
          return;
        }
        state.completed = false;
      }
      if (settings.mode === 'simple' && !explicit) {
        publishL0TimingAvailability({ taskId, status: 'unavailable' });
        return;
      }
      if (settings.mode === 'simple' && !settings.openRouterApiKey.trim()) {
        throw new Error('Simple mode requires an OpenRouter API key. Add your key in the Babel Gold Drafting extension options, then try again.');
      }
      if (settings.mode === 'local') {
        const status = await (this.dependencies.localModelStatus ?? getLocalModelStatus)(LOCAL_MODEL_BASE_URL);
        if (status.state !== 'ready' || status.tested !== true) {
          throw new Error('Local WebGPU C-denoise is not ready. Download and test the C-denoise bundle in extension Options, then retry. No cloud fallback was used.');
        }
      }
      if (!(await this.verifyOwner(taskId, state))) return;
      const audioTracks = (await this.dependencies.captureAudio()).filter((track) => track.blob.size > 0);
      if (!(await this.verifyOwner(taskId, state))) return;
      try {
        prepareL0TimingTracks(job, audioTracks);
      } catch (error) {
        if (settings.mode === 'simple') throw error;
        // Advanced retains its existing audio-readiness lifecycle.
        state.audioWaitCount += 1;
        publishL0TimingAvailability({ taskId, status: 'preparing' });
        state.retryScheduled = true;
        try {
          this.dependencies.schedule(() => {
            state.retryScheduled = false;
            if (this.isOwnerCurrent(taskId, state)) this.onLifecycleOpportunity();
          }, Math.min(INITIAL_RETRY_DELAY_MS * state.audioWaitCount, 30_000));
        } catch {
          state.retryScheduled = false;
        }
        return;
      }
      state.audioWaitCount = 0;
      const response = await this.dependencies.requestTiming(settings, job, audioTracks, {
        onQueueStatus: (status: L0TimingQueueStatus) => {
          if (!this.isOwnerCurrent(taskId, state)) return;
          if (status.status === 'queued') {
            publishL0TimingAvailability({ taskId, status: 'queued', position: status.position });
          } else if (status.status === 'running' || status.status === 'preparing') {
            publishL0TimingAvailability({ taskId, status: status.status });
          }
        }
      });
      if (!(await this.verifyOwner(taskId, state)) || response.taskId !== taskId) return;
      this.publishTiming(taskId, response, state);
    } catch (error) {
      if (!(await this.verifyOwner(taskId, state))) return;
      state.error = error instanceof Error ? error : new Error(String(error));
      if (settings.mode === 'simple') {
        publishL0TimingAvailability({ taskId, status: 'unavailable', error: state.error.message });
      } else if (settings.mode === 'local') {
        state.failureCount = MAX_AUTOMATIC_RETRIES + 1;
        publishL0TimingAvailability({ taskId, status: 'unavailable', error: state.error.message });
      } else {
        console.error(`[Babel Gold] word timing attempt ${state.failureCount + 1} failed for task ${taskId}.`, error);
        this.scheduleRetry(taskId, state);
      }
    } finally {
      state.inFlight = false;
      state.attempt = null;
    }
  }

  private publishTiming(taskId: string, response: L0TimingResponse, state: TimingTaskState): void {
    this.dependencies.publish({
      type: L0_TIMING_UPDATE_MESSAGE_TYPE, version: 1, taskId,
      tracks: response.tracks.map(({ lane, tokens }) => ({ lane, tokens }))
    });
    state.completed = true;
    publishL0TimingAvailability({ taskId, status: 'available' });
  }

  async waitForTiming(job: TranscriptJob, settings: ExtensionSettings): Promise<void> {
    const taskId = buildCanonicalTaskIdentity(job);
    const pathname = this.dependencies.currentPathname();
    const saved = await this.dependencies.getSettings();
    if (this.engine(saved) !== this.engine(settings)) throw new Error('The drafting mode changed while waiting for L0 timing.');
    if (settings.mode === 'local') {
      const status = await (this.dependencies.localModelStatus ?? getLocalModelStatus)(LOCAL_MODEL_BASE_URL);
      if (status.state !== 'ready' || status.tested !== true) {
        throw new Error('Local WebGPU C-denoise is not ready. Download and test the C-denoise bundle in extension Options, then retry.');
      }
    }
    this.activateSettings(saved);
    const state = this.getTaskState(taskId, settings);
    const isCurrent = () => this.dependencies.currentPathname() === pathname && this.isOwnerCurrent(taskId, state);
    const changedError = () => new Error('The task changed or drafting mode changed while waiting for L0 timing.');
    if (!isCurrent()) throw changedError();
    if (settings.mode === 'local' && state.completed) return;
    if (settings.mode === 'simple' || settings.mode === 'local') {
      // A free lifecycle lookup may already be running. Await it, then explicitly
      // recheck durable cache and generate only if absent; never trust completed.
      const existing = state.attempt;
      const alreadyExplicit = state.explicitAttempt;
      if (existing) {
        await this.waitWithOwnership(existing, isCurrent, changedError);
        if (!isCurrent()) throw changedError();
        if (alreadyExplicit) {
          if (state.error) throw state.error;
          if (!state.completed) throw new Error(`L0 timing for task ${taskId} is unavailable.`);
          return;
        }
      }
      if (!isCurrent()) throw changedError();
      const attempt = this.startAttempt(job, taskId, saved, state, true);
      await this.waitWithOwnership(attempt, isCurrent, changedError);
      if (!isCurrent()) throw changedError();
      if (state.error) throw state.error;
      if (!state.completed) throw new Error(`L0 timing for task ${taskId} is unavailable.`);
      return;
    }
    return new Promise<void>((resolve, reject) => {
      let settled = false;
      let unsubscribe: (() => void) | undefined;
      const finish = (error?: Error) => {
        if (settled) return;
        settled = true;
        unsubscribe?.();
        this.taskChecks.delete(checkTask);
        if (error) reject(error); else resolve();
      };
      const checkTask = () => { if (!isCurrent()) finish(changedError()); };
      const onAvailability = (availability: L0TimingAvailability) => {
        if (!isCurrent() || availability.taskId !== taskId) return checkTask();
        if (availability.status === 'unavailable') finish(state.error ?? new Error(`L0 timing for task ${taskId} is unavailable.`));
        else if (availability.status === 'available' ||
          (!isBrowserLocalMode(settings) && (availability.status === 'queued' || availability.status === 'running'))) finish();
      };
      this.taskChecks.add(checkTask);
      // Reset an unrelated engine's replay before subscribing.
      if (!state.completed && !state.inFlight) publishL0TimingAvailability({ taskId, status: 'preparing' });
      unsubscribe = subscribeL0TimingAvailability(onAvailability);
      if (settled) unsubscribe();
      else this.onLifecycleOpportunity();
    });
  }

  private waitWithOwnership(
    attempt: Promise<void>, isCurrent: () => boolean, changedError: () => Error
  ): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      let settled = false;
      const check = () => { if (!isCurrent()) finish(changedError()); };
      const finish = (error?: Error) => {
        if (settled) return;
        settled = true;
        this.taskChecks.delete(check);
        unsubscribe();
        if (error) reject(error); else resolve();
      };
      const unsubscribe = subscribeL0TimingAvailability(() => {
        if (!isCurrent()) queueMicrotask(check);
      });
      this.taskChecks.add(check);
      void attempt.then(() => finish(), (error: Error) => finish(error));
    });
  }

  async recoverLocalTiming(settings: ExtensionSettings, job: TranscriptJob): Promise<void> {
    const taskId = buildCanonicalTaskIdentity(job);
    if (!isBrowserLocalMode(settings) || !this.isTaskCurrent(taskId)) throw new Error('The local task changed while recovering its acoustic cache.');
    const state = this.getTaskState(taskId, settings);
    state.completed = false;
    state.failureCount = 0;
    state.audioWaitCount = 0;
    state.retryNotBefore = 0;
    publishL0TimingAvailability({ taskId, status: 'preparing' });
    await this.waitForTiming(job, settings);
    if (!this.isTaskCurrent(taskId)) throw new Error('The task changed while recovering local timing.');
  }

  async generateLocalDraft(settings: ExtensionSettings, job: TranscriptJob) {
    const taskId = buildCanonicalTaskIdentity(job);
    if (!this.isTaskCurrent(taskId)) throw new Error('The task changed while drafting.');
    try {
      return await this.dependencies.requestLocalDraft(settings, job);
    } catch (error) {
      if (!(error instanceof LocalModelBridgeError) || error.code !== 'timing-unavailable') throw error;
      await this.recoverLocalTiming(settings, job);
      return this.dependencies.requestLocalDraft(settings, job);
    }
  }

  retryCurrentTask(): boolean {
    let job: TranscriptJob;
    try { job = this.dependencies.captureTranscript(); } catch { return false; }
    const taskId = buildCanonicalTaskIdentity(job);
    if (this.manualRequests.has(taskId)) return false;
    if (!isUsableL0TimingJob(job) || !this.isTaskCurrent(taskId)) return false;
    const settings = this.currentSettings;
    if (settings) {
      const state = this.getTaskState(taskId, settings);
      if (state.inFlight || state.retryScheduled || (settings.mode === 'advanced' && state.completed)) return false;
      state.failureCount = 0;
      state.audioWaitCount = 0;
      state.retryNotBefore = 0;
    }
    this.manualRequests.add(taskId);
    publishL0TimingAvailability({ taskId, status: 'preparing' });
    void this.dependencies.getSettings().then(async (saved) => {
      this.activateSettings(saved);
      if (saved.mode === 'simple') await this.waitForTiming(job, saved);
      else this.onLifecycleOpportunity();
    }).catch((error) => {
      if (this.isTaskCurrent(taskId)) {
        publishL0TimingAvailability({ taskId, status: 'unavailable' });
        console.error('[Babel Gold] explicit timing request failed.', error);
      }
    }).finally(() => this.manualRequests.delete(taskId));
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

export function getCurrentL0TimingGeneration(): number {
  if (!activeTimingService) throw new Error('L0 timing service is not initialized.');
  return activeTimingService.generationId;
}

export function waitForCurrentL0Timing(job: TranscriptJob, settings: ExtensionSettings): Promise<void> {
  if (!activeTimingService) throw new Error('L0 timing service is not initialized.');
  return activeTimingService.waitForTiming(job, settings);
}

export function recoverCurrentLocalL0Timing(settings: ExtensionSettings, job: TranscriptJob): Promise<void> {
  if (!activeTimingService) throw new Error('L0 timing service is not initialized.');
  return activeTimingService.recoverLocalTiming(settings, job);
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
    lookupTiming: (settings, taskId) => settings.mode === 'simple'
      ? lookupMaiL0Timing(settings, taskId)
      : lookupL0Timing(settings, taskId),
    requestLocalDraft: generateLocalL0Draft,
    requestTiming: requestConfiguredL0Timing,
    publish: (message) => window.postMessage(message, '*'),
    now: () => Date.now(),
    schedule: (callback, delayMs) => {
      window.setTimeout(callback, delayMs);
    }
  });
  globalThis.chrome?.storage?.onChanged?.addListener((changes, areaName) => {
    if (areaName === 'local' && changes[SETTINGS_STORAGE_KEY]) {
      service.onSettingsChanged(normalizeSettings(changes[SETTINGS_STORAGE_KEY].newValue));
    }
  });
  setL0TimingRetryHandler(() => service.retryCurrentTask());
  activeTimingService = service;
  return service;
}
