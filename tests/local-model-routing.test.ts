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

const settings: ExtensionSettings = { ...DEFAULT_SETTINGS };
const localSettings: ExtensionSettings = { ...DEFAULT_SETTINGS, localModelsEnabled: true };
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

test('timing routing keeps the remote generator as the default and uses local only after opt-in', async () => {
  const calls: string[] = [];
  const generators: L0TimingGenerators = {
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
});

test('draft routing preserves the remote path by default and surfaces an opted-in local failure without fallback', async () => {
  let remoteCalls = 0;
  const remote = async () => {
    remoteCalls += 1;
    return draftResponse;
  };
  const successGenerators: L0DraftGenerators = {
    remote,
    local: async () => draftResponse
  };

  assert.equal(await generateConfiguredL0Draft(settings, job, tracks, successGenerators), draftResponse);
  assert.equal(remoteCalls, 1);

  const localError = new Error('local inference failed');
  const failureGenerators: L0DraftGenerators = {
    remote,
    local: async () => {
      throw localError;
    }
  };
  await assert.rejects(
    generateConfiguredL0Draft(localSettings, job, tracks, failureGenerators),
    (error) => error === localError
  );
  assert.equal(remoteCalls, 1);
});

test('segment routing keeps the existing prepared-track remote call when disabled', async () => {
  let localCalls = 0;
  let receivedTaskId = '';
  let receivedTrackCount = 0;
  const generators: L0SegmentGenerators = {
    remote: async (_settings, taskId, row, preparedTracks) => {
      receivedTaskId = taskId;
      receivedTrackCount = preparedTracks.length;
      assert.equal(row, targetRow);
      return 'remote text';
    },
    local: async () => {
      localCalls += 1;
      return 'local text';
    }
  };

  assert.equal(
    await generateConfiguredL0SegmentText(settings, 'task-1', targetRow, tracks, generators),
    'remote text'
  );
  assert.equal(receivedTaskId, 'task-1');
  assert.equal(receivedTrackCount, 2);
  assert.equal(localCalls, 0);
});

test('opted-in segment routing uses the exact cropped local result without remote fallback', async () => {
  let remoteCalls = 0;
  let receivedTrackCount = 0;
  const generators: L0SegmentGenerators = {
    remote: async () => {
      remoteCalls += 1;
      return 'remote text';
    },
    local: async (_settings, taskId, row, preparedTracks) => {
      assert.equal(taskId, 'task-1');
      assert.equal(row, targetRow);
      receivedTrackCount = preparedTracks.length;
      return 'exact local text';
    }
  };

  assert.equal(
    await generateConfiguredL0SegmentText(localSettings, 'task-1', targetRow, tracks, generators),
    'exact local text'
  );
  assert.equal(receivedTrackCount, 2);
  assert.equal(remoteCalls, 0);

  const localError = new Error('browser model crashed');
  generators.local = async () => {
    throw localError;
  };
  await assert.rejects(
    generateConfiguredL0SegmentText(localSettings, 'task-1', targetRow, tracks, generators),
    (error) => error === localError
  );
  assert.equal(remoteCalls, 0);
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
    assert.equal(LOCAL_MODEL_BASE_URL, 'https://reviewgen.ovh/browser-model');
    assert.equal(baseUrl, LOCAL_MODEL_BASE_URL);
    return {
      state: 'ready',
      completedBytes: 1,
      totalBytes: 1
    };
  });
  assert.equal(readyCapabilities.transcribeSegmentL0, true);

  const failedStatusCapabilities = await resolveBrokerCapabilities(localSettings, async () => {
    throw new Error('cache unavailable');
  });
  assert.equal(failedStatusCapabilities.transcribeSegmentL0, false);
});
