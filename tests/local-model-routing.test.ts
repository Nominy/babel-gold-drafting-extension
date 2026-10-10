import test from 'node:test';
import assert from 'node:assert/strict';
import { resolveBrokerCapabilities } from '../src/background/ai-broker';
import {
  generateConfiguredL0SegmentText,
  type L0SegmentGenerators
} from '../src/content/ai-broker-content';
import {
  requestConfiguredL0Timing,
  type L0TimingGenerators
} from '../src/content/l0-timing-service';
import {
  generateConfiguredL0Draft,
  type L0DraftGenerators
} from '../src/content/overlay';
import { DEFAULT_SETTINGS, LOCAL_MODEL_BASE_URL } from '../src/core/settings';
import type {
  CapturedAudioTrack,
  ExtensionSettings,
  L0DraftResponse,
  L0TimingResponse,
  TranscriptJob,
  TranscriptRow
} from '../src/core/types';

const settings: ExtensionSettings = { ...DEFAULT_SETTINGS, mode: 'advanced' };
const localSettings: ExtensionSettings = { ...DEFAULT_SETTINGS, mode: 'advanced', localModelsEnabled: true };
const targetRow: TranscriptRow = {
  rowId: 'row-1',
  speakerKey: 'Speaker A',
  startSeconds: 10,
  endSeconds: 14,
  text: '',
  index: 0
};
const job: TranscriptJob = { jobId: 'task-1', rows: [targetRow] };
const tracks: CapturedAudioTrack[] = [
  {
    trackId: 'track-a',
    speakerKey: 'Speaker A',
    source: 'capture',
    blob: new Blob(['audio-a'], { type: 'audio/wav' }),
    mimeType: 'audio/wav'
  },
  {
    trackId: 'track-b',
    speakerKey: 'Speaker B',
    source: 'capture',
    blob: new Blob(['audio-b'], { type: 'audio/wav' }),
    mimeType: 'audio/wav'
  }
];
const timingResponse: L0TimingResponse = {
  taskId: 'task-1',
  tracks: [],
  summary: {},
  models: {}
};
const draftResponse: L0DraftResponse = {
  rows: [
    { id: 'generated-1', lane: 'speaker a', startSeconds: 10.5, endSeconds: 12, text: 'local text' }
  ],
  summary: {},
  models: {}
};

test('Advanced timing uses the configured remote or local engine without cloud fallback', async () => {
  const calls: string[] = [];
  const generators: L0TimingGenerators = {
    mai: async () => { assert.fail('Advanced must not use MAI'); },
    remote: async () => {
      calls.push('remote');
      return timingResponse;
    },
    local: async () => {
      calls.push('local');
      return timingResponse;
    }
  };

  assert.equal(await requestConfiguredL0Timing(settings, job, tracks, {}, generators), timingResponse);
  assert.deepEqual(calls, ['remote']);

  calls.length = 0;
  assert.equal(await requestConfiguredL0Timing(localSettings, job, tracks, {}, generators), timingResponse);
  assert.deepEqual(calls, ['local']);
  calls.length = 0;
  assert.equal(
    await requestConfiguredL0Timing(
      { ...localSettings, volunteerInferenceEnabled: false }, job, tracks, {}, generators
    ),
    timingResponse
  );
  assert.deepEqual(calls, ['local']);
});

test('Advanced drafting surfaces a local failure without falling back to remote or MAI', async () => {
  let remoteCalls = 0;
  const remote = async () => {
    remoteCalls += 1;
    return draftResponse;
  };
  const successGenerators: L0DraftGenerators = {
    remote,
    mai: async () => { assert.fail('Advanced must not use MAI'); },
    local: async () => draftResponse
  };

  assert.equal(await generateConfiguredL0Draft(settings, job, successGenerators), draftResponse);
  assert.equal(remoteCalls, 1);

  const localError = new Error('local inference failed');
  const failureGenerators: L0DraftGenerators = {
    mai: async () => { assert.fail('Advanced must not use MAI'); },
    remote,
    local: async () => {
      throw localError;
    }
  };
  await assert.rejects(
    generateConfiguredL0Draft(localSettings, job, failureGenerators),
    (error) => error === localError
  );
  assert.equal(remoteCalls, 1);
});


test('L0 broker capability uses the fixed supplier readiness for opt-in and remote availability otherwise', async () => {
  let statusCalls = 0;
  const defaultCapabilities = await resolveBrokerCapabilities(settings, async () => {
    statusCalls += 1;
    return { state: 'not-installed', completedBytes: 0, totalBytes: 1 };
  });
  assert.equal(defaultCapabilities.transcribeSegmentL0, true);
  assert.equal(statusCalls, 0);

  const notInstalledCapabilities = await resolveBrokerCapabilities(localSettings, async () => ({
    state: 'not-installed',
    completedBytes: 0,
    totalBytes: 1
  }));
  assert.equal(notInstalledCapabilities.transcribeSegmentL0, false);

  const readyCapabilities = await resolveBrokerCapabilities(localSettings, async (baseUrl) => {
    assert.equal(baseUrl, LOCAL_MODEL_BASE_URL);
    return {
      state: 'ready',
      completedBytes: 1,
      totalBytes: 1,
      tested: true
    };
  });
  assert.equal(readyCapabilities.transcribeSegmentL0, true);

  const failedStatusCapabilities = await resolveBrokerCapabilities(localSettings, async () => {
    throw new Error('cache unavailable');
  });
  assert.equal(failedStatusCapabilities.transcribeSegmentL0, false);

  const ownTaskSettings = { ...DEFAULT_SETTINGS, mode: 'local' as const, openRouterApiKey: '', aiBrokerProvider: 'local-gemini-nano' as const };
  const untested = await resolveBrokerCapabilities(ownTaskSettings, async () => ({
    state: 'ready', completedBytes: 1, totalBytes: 1, tested: false
  }));
  assert.deepEqual(untested, { transcribeSegment: false, transcribeSegmentL0: false, redistributeText: false, enhanceAudio: true });
  const tested = await resolveBrokerCapabilities(ownTaskSettings, async () => ({
    state: 'ready', completedBytes: 1, totalBytes: 1, tested: true
  }));
  assert.deepEqual(tested, { transcribeSegment: true, transcribeSegmentL0: true, redistributeText: false, enhanceAudio: true });
  for (const mode of ['simple', 'advanced', 'local'] as const) {
    const capabilities = await resolveBrokerCapabilities({ ...DEFAULT_SETTINGS, mode, openRouterApiKey: '' }, async () => {
      throw new Error('ASR is not installed');
    });
    assert.equal(capabilities.enhanceAudio, true, `Packaged enhancement is independent of ${mode} ASR setup and paid keys`);
  }
});

test('Simple native transcription ignores Advanced engine preferences and never falls back after a cloud failure', async () => {
  const simple: ExtensionSettings = {
    ...localSettings, mode: 'simple', l0ReplacementPreviewEnabled: false, l0DontRunLlm: false
  };
  const forbidden = async (): Promise<never> => { assert.fail('Simple must not run an Advanced engine'); };
  const cloudFailure = new Error('MAI provider rejected transcription');
  const timingGenerators: L0TimingGenerators = { remote: forbidden, local: forbidden, mai: async () => timingResponse };
  const draftGenerators: L0DraftGenerators = { remote: forbidden, local: forbidden, mai: async () => draftResponse };
  const segmentGenerators: L0SegmentGenerators = { remote: forbidden, local: forbidden, mai: async () => 'Ну, я… да!' };
  assert.equal(await requestConfiguredL0Timing(simple, job, [], {}, timingGenerators), timingResponse);
  assert.equal(await generateConfiguredL0Draft(simple, job, draftGenerators), draftResponse);
  assert.equal(await generateConfiguredL0SegmentText(simple, 'task-1', targetRow, segmentGenerators), 'Ну, я… да!');
  timingGenerators.mai = async () => { throw cloudFailure; };
  await assert.rejects(requestConfiguredL0Timing(simple, job, [], {}, timingGenerators), (error) => error === cloudFailure);
});
