import type { AudioEnhancementChunk, AudioEnhancementProgress } from '@nominy/babel-babel-runtime';
import type { EnhancedAudioBatch } from '../src/core/audio-enhancement-runtime';
import { enhanceZipSamples } from '../src/core/audio-enhancement-dsp';
import test from 'node:test';
import assert from 'node:assert/strict';
import { Buffer } from 'node:buffer';

import { createLocalModelOffscreenBridge } from '../src/background/local-model-offscreen';
import { LocalModelBridgeError, createLocalModelClient } from '../src/core/local-model-client';
import { LocalTimingUnavailableError } from '../src/core/local-model-runtime';
import {
  LOCAL_MODEL_AUDIO_CHUNK_BYTES,
  LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE,
  LOCAL_MODEL_OFFSCREEN_VERSION,
  createLocalModelFailure,
  decodeAudioChunk,
  encodeAudioChunk,
  isLocalModelOffscreenRequest,
  type LocalModelEnhanceAudioRequest,
  type LocalModelEnhancementProgressMessage,
  type LocalModelOffscreenRequest,
  type LocalModelOffscreenResponse,
  type LocalModelUploadRequest,
  type WireCapturedAudioTrack
} from '../src/core/local-model-offscreen-protocol';
import { DEFAULT_SETTINGS } from '../src/core/settings';
import { buildCanonicalTaskIdentity } from '../src/core/transcript';
import type { PreparedL0Track } from '../src/core/l0-client';
import type {
  CapturedAudioTrack,
  L0DraftResponse,
  L0TimingResponse,
  TranscriptJob
} from '../src/core/types';
import { createLocalModelHost, type LocalModelHost } from '../src/offscreen/local-model-host';
import { L0TimingService } from '../src/content/l0-timing-service';
import { getL0TimingAvailability } from '../src/content/l0-timing-availability';

const row = {
  rowId: 'row-1',
  speakerKey: 'Speaker 1',
  startSeconds: 1,
  endSeconds: 2,
  text: 'source',
  index: 0
};
const job: TranscriptJob = { jobId: 'task-1', rows: [row] };
const audioBytes = new Uint8Array(LOCAL_MODEL_AUDIO_CHUNK_BYTES + 37);
for (let index = 0; index < audioBytes.length; index += 1) audioBytes[index] = index % 251;
const audioTracks: CapturedAudioTrack[] = [
  {
    trackId: 'track-1',
    speakerKey: 'Speaker 1',
    trackLabel: 'Left microphone',
    source: 'captured.wav',
    blob: new Blob([audioBytes], { type: 'audio/x-babel' }),
    mimeType: 'audio/wav'
  }
];
const preparedTracks: PreparedL0Track[] = [
  { lane: 'Speaker 1', fieldName: 'audio:1', audio: audioTracks[0] }
];
const timingResult: L0TimingResponse = {
  taskId: buildCanonicalTaskIdentity(job),
  tracks: [
    {
      lane: 'Speaker 1',
      pcmSha256: 'a'.repeat(64),
      sampleRate: 16000,
      tokens: [{ id: 'token-1', text: 'hello', startSeconds: 1, endSeconds: 2 }],
      segments: [{ id: 'segment-1', startSeconds: 1, endSeconds: 2, startSample: 16000, endSample: 32000, sampleRate: 16000 }]
    }
  ],
  summary: {},
  models: {}
};
const draftResult: L0DraftResponse = {
  rows: [{ id: 'draft-1', lane: 'Speaker 1', startSeconds: 1, endSeconds: 2, text: 'Hello.' }],
  summary: {},
  models: {}
};

function successResponse(request: LocalModelOffscreenRequest): LocalModelOffscreenResponse {
  const envelope = {
    type: LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE,
    version: LOCAL_MODEL_OFFSCREEN_VERSION,
    requestId: request.requestId
  } as const;
  if (request.operation === 'upload') {
    return {
      ...envelope,
      operation: 'upload',
      ok: true,
      result: {
        transferId: request.transferId,
        nextChunkIndex: request.chunkIndex + 1,
        complete: request.chunkIndex === request.chunkCount - 1
      }
    };
  }
  if (request.operation === 'timing') {
    return { ...envelope, operation: 'timing', ok: true, result: timingResult };
  }
  if (request.operation === 'draft') {
    return { ...envelope, operation: 'draft', ok: true, result: draftResult };
  }
  return { ...envelope, operation: 'segment', ok: true, result: 'Exact cropped text.' };
}

function wireTrack(transferId: string): WireCapturedAudioTrack {
  return {
    trackId: 'track-1',
    speakerKey: 'Speaker 1',
    trackLabel: 'Left microphone',
    source: 'captured.wav',
    audioTransferId: transferId,
    mimeType: 'audio/wav'
  };
}

function timingRequest(requestId: string, transferId = `transfer:${requestId}`): LocalModelOffscreenRequest {
  return {
    type: LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE,
    version: LOCAL_MODEL_OFFSCREEN_VERSION,
    target: 'background',
    requestId,
    operation: 'timing',
    settings: DEFAULT_SETTINGS,
    job,
    audioTracks: [wireTrack(transferId)]
  };
}

function uploadRequest(
  transferId: string,
  chunkIndex: number,
  chunkCount: number,
  totalBytes: number,
  bytes: Uint8Array,
  requestId = `${transferId}:${chunkIndex}`
): LocalModelUploadRequest {
  return {
    type: LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE,
    version: LOCAL_MODEL_OFFSCREEN_VERSION,
    target: 'offscreen',
    requestId,
    operation: 'upload',
    transferId,
    chunkIndex,
    chunkCount,
    totalBytes,
    mimeType: 'audio/x-babel',
    dataBase64: encodeAudioChunk(bytes)
  };
}

async function uploadBlob(
  host: LocalModelHost,
  transferId: string,
  blob: Blob
): Promise<WireCapturedAudioTrack> {
  const chunkCount = Math.max(1, Math.ceil(blob.size / LOCAL_MODEL_AUDIO_CHUNK_BYTES));
  for (let chunkIndex = 0; chunkIndex < chunkCount; chunkIndex += 1) {
    const start = chunkIndex * LOCAL_MODEL_AUDIO_CHUNK_BYTES;
    const end = Math.min(start + LOCAL_MODEL_AUDIO_CHUNK_BYTES, blob.size);
    const bytes = new Uint8Array(await blob.slice(start, end).arrayBuffer());
    const request = uploadRequest(transferId, chunkIndex, chunkCount, blob.size, bytes);
    request.mimeType = blob.type;
    const roundTripped = JSON.parse(JSON.stringify(request)) as LocalModelUploadRequest;
    const response = await host.handleRequest(roundTripped);
    assert.equal(response.ok, true);
  }
  return wireTrack(transferId);
}

test('background deduplicates concurrent offscreen creation and forwards transfer references unchanged', async () => {
  let documentExists = false;
  let createCount = 0;
  const creationGate = Promise.withResolvers<void>();
  const creationStarted = Promise.withResolvers<void>();
  const forwarded: LocalModelOffscreenRequest[] = [];
  const bridge = createLocalModelOffscreenBridge({
    hasDocument: async () => documentExists,
    createDocument: async () => {
      createCount += 1;
      creationStarted.resolve();
      await creationGate.promise;
      documentExists = true;
    },
    closeDocument: async () => {
      documentExists = false;
    },
    sendMessage: async (message) => {
      const roundTripped = JSON.parse(JSON.stringify(message)) as LocalModelOffscreenRequest;
      assert.deepEqual(roundTripped, message);
      forwarded.push(roundTripped);
      return successResponse(roundTripped);
    },
    workersReason: 'WORKERS' as chrome.offscreen.Reason
  });

  const first = bridge.forwardRequest(timingRequest('request-1'));
  const second = bridge.forwardRequest(timingRequest('request-2'));
  await creationStarted.promise;
  assert.equal(createCount, 1);
  creationGate.resolve();
  await Promise.all([first, second]);

  assert.equal(createCount, 1);
  assert.equal(forwarded.length, 2);
  assert.ok(forwarded.every((request) => request.target === 'offscreen'));
  const timing = forwarded[0] as Extract<LocalModelOffscreenRequest, { operation: 'timing' }>;
  assert.equal('blob' in timing.audioTracks[0], false);
  assert.equal('audioDataUrl' in timing.audioTracks[0], false);
  assert.equal(timing.audioTracks[0].audioTransferId, 'transfer:request-1');
});

test('background closes and recreates a broken offscreen document before retrying', async () => {
  let documentExists = true;
  let createCount = 0;
  let closeCount = 0;
  let sendCount = 0;
  const bridge = createLocalModelOffscreenBridge({
    hasDocument: async () => documentExists,
    createDocument: async () => {
      createCount += 1;
      documentExists = true;
    },
    closeDocument: async () => {
      closeCount += 1;
      documentExists = false;
    },
    sendMessage: async (message) => {
      sendCount += 1;
      return sendCount === 1 ? { ok: true, result: timingResult } : successResponse(message);
    },
    workersReason: 'WORKERS' as chrome.offscreen.Reason
  });

  const response = await bridge.forwardRequest(timingRequest('recover-me'));
  assert.equal(response.ok, true);
  assert.equal(sendCount, 2);
  assert.equal(closeCount, 1);
  assert.equal(createCount, 1);
});

test('offscreen host serializes heavyweight inference and turns runtime failures into actionable responses', async () => {
  const timingGate = Promise.withResolvers<void>();
  const timingStarted = Promise.withResolvers<void>();
  const calls: string[] = [];
  const host = createLocalModelHost(async () => ({
    isLocalTimingCurrent: async () => true,
    generateLocalL0Timing: async () => {
      calls.push('timing:start');
      timingStarted.resolve();
      await timingGate.promise;
      calls.push('timing:end');
      return timingResult;
    },
    generateLocalL0DraftFromTiming: async () => {
      calls.push('draft');
      throw new Error('ONNX model file is missing');
    },
    generateLocalL0SegmentDraft: async () => 'segment'
  }));
  const timingTrack = await uploadBlob(host, 'host-timing-audio', audioTracks[0].blob);

  const timing = host.handleRequest({
    ...timingRequest('host-timing', timingTrack.audioTransferId),
    target: 'offscreen'
  });
  const draft = host.handleRequest({
    type: LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE,
    version: LOCAL_MODEL_OFFSCREEN_VERSION,
    target: 'offscreen',
    requestId: 'host-draft',
    operation: 'draft',
    settings: DEFAULT_SETTINGS,
    taskId: timingResult.taskId
  });
  await timingStarted.promise;
  assert.deepEqual(calls, ['timing:start']);
  timingGate.resolve();
  assert.equal((await timing).ok, true);
  const failed = await draft;
  assert.deepEqual(calls, ['timing:start', 'timing:end', 'draft']);
  assert.equal(failed.ok, false);
  if (!failed.ok) {
    assert.equal(failed.error.code, 'inference-failed');
    assert.match(failed.error.message, /ONNX model file is missing/);
  }
});

test('all operations JSON-roundtrip bounded chunks and restore exact Blob bytes, MIME, and track metadata', async () => {
  const received: LocalModelOffscreenRequest[] = [];
  const runtimeTimingTracks: CapturedAudioTrack[][] = [];
  const runtimeDraftTimings: L0TimingResponse[] = [];
  const runtimeSegmentTracks: PreparedL0Track[][] = [];
  const host = createLocalModelHost(async () => ({
    isLocalTimingCurrent: async () => true,
    generateLocalL0Timing: async (_settings, _job, tracks) => {
      runtimeTimingTracks.push(tracks);
      return timingResult;
    },
    generateLocalL0DraftFromTiming: async (timing) => {
      runtimeDraftTimings.push(timing);
      return draftResult;
    },
    generateLocalL0SegmentDraft: async (_settings, _taskId, _row, tracks) => {
      runtimeSegmentTracks.push(tracks);
      return 'Exact cropped text.';
    }
  }));
  const bridge = createLocalModelOffscreenBridge({
    hasDocument: async () => true,
    createDocument: async () => undefined,
    closeDocument: async () => undefined,
    sendMessage: async (message) =>
      host.handleRequest(JSON.parse(JSON.stringify(message)) as LocalModelOffscreenRequest),
    workersReason: 'WORKERS' as chrome.offscreen.Reason
  });
  const client = createLocalModelClient(async (request) => {
    const serialized = JSON.stringify(request);
    assert.doesNotMatch(serialized, /"blob"\s*:/);
    assert.doesNotMatch(serialized, /audioDataUrl/);
    const roundTripped = JSON.parse(serialized) as LocalModelOffscreenRequest;
    assert.deepEqual(roundTripped, request);
    assert.equal(isLocalModelOffscreenRequest(roundTripped, 'background'), true);
    received.push(roundTripped);
    return bridge.forwardRequest(roundTripped);
  });
  const statuses: string[] = [];

  assert.equal(
    await client.generateLocalL0Timing(DEFAULT_SETTINGS, job, audioTracks, {
      onQueueStatus: (status) => statuses.push(status.status)
    }),
    timingResult
  );
  assert.equal(await client.generateLocalL0Draft(DEFAULT_SETTINGS, job), draftResult);
  assert.equal(
    await client.generateLocalL0SegmentDraft(DEFAULT_SETTINGS, 'task-1', row, preparedTracks),
    'Exact cropped text.'
  );

  assert.deepEqual(received.map((request) => request.operation), [
    'upload',
    'upload',
    'timing',
    'draft',
    'upload',
    'upload',
    'segment'
  ]);
  assert.ok(received.every((request) => request.target === 'background'));
  assert.deepEqual(statuses, ['preparing', 'running', 'completed']);
  const uploads = received.filter(
    (request): request is LocalModelUploadRequest => request.operation === 'upload'
  );
  assert.equal(uploads.length, 4);
  assert.ok(uploads.every((request) => decodeAudioChunk(request.dataBase64).byteLength <= LOCAL_MODEL_AUDIO_CHUNK_BYTES));
  assert.ok(uploads.every((request) => decodeAudioChunk(request.dataBase64).byteLength < audioTracks[0].blob.size));
  assert.ok(
    received
      .filter((request) => request.operation !== 'upload')
      .every((request) => !('dataBase64' in request))
  );

  assert.deepEqual(runtimeDraftTimings, [timingResult]);
  const audio = runtimeTimingTracks[0][0];
  assert.ok(audio.blob instanceof Blob);
  assert.equal(audio.blob.type, 'audio/x-babel');
  assert.deepEqual(new Uint8Array(await audio.blob.arrayBuffer()), audioBytes);
  assert.deepEqual(
    {
      trackId: audio.trackId,
      speakerKey: audio.speakerKey,
      trackLabel: audio.trackLabel,
      source: audio.source,
      mimeType: audio.mimeType
    },
    {
      trackId: 'track-1',
      speakerKey: 'Speaker 1',
      trackLabel: 'Left microphone',
      source: 'captured.wav',
      mimeType: 'audio/wav'
    }
  );
  assert.equal(runtimeSegmentTracks[0][0].lane, 'Speaker 1');
  assert.equal(runtimeSegmentTracks[0][0].fieldName, 'audio:1');
  assert.equal(runtimeSegmentTracks[0][0].audio.blob.type, 'audio/x-babel');
  assert.deepEqual(new Uint8Array(await runtimeSegmentTracks[0][0].audio.blob.arrayBuffer()), audioBytes);
  assert.equal(runtimeSegmentTracks[0][0].audio.trackLabel, 'Left microphone');

  const consumedTiming = received.find(
    (request): request is Extract<LocalModelOffscreenRequest, { operation: 'timing' }> =>
      request.operation === 'timing'
  );
  assert.ok(consumedTiming);
  const reused = await host.handleRequest({ ...consumedTiming, target: 'offscreen', requestId: 'reuse' });
  assert.equal(reused.ok, true);
  assert.equal(runtimeTimingTracks.length, 1, 'a second timing request reuses the offscreen ASR result');
});

test('uploads do not initialize runtime and reject duplicates, gaps, buffer overflow, and missing transfers', async () => {
  let timestamp = 0;
  let runtimeLoads = 0;
  const host = createLocalModelHost(
    async () => {
      runtimeLoads += 1;
      return {
        isLocalTimingCurrent: async () => true,
        generateLocalL0Timing: async () => timingResult,
        generateLocalL0DraftFromTiming: async () => draftResult,
        generateLocalL0SegmentDraft: async () => 'segment'
      };
    },
    { now: () => timestamp, maxBufferedBytes: 3, staleTransferMs: 100 }
  );

  const first = uploadRequest('partial', 0, 2, 4, new Uint8Array([1, 2]));
  assert.equal((await host.handleRequest(first)).ok, true);
  const duplicate = await host.handleRequest({ ...first, requestId: 'duplicate' });
  assert.equal(duplicate.ok, false);
  if (!duplicate.ok) assert.match(duplicate.error.message, /duplicate/);
  const gap = await host.handleRequest(uploadRequest('gap', 1, 2, 4, new Uint8Array([3, 4])));
  assert.equal(gap.ok, false);
  if (!gap.ok) assert.match(gap.error.message, /out-of-order/);
  const capped = await host.handleRequest(
    uploadRequest('capped', 0, 1, 2, new Uint8Array([5, 6]))
  );
  assert.equal(capped.ok, false);
  if (!capped.ok) assert.match(capped.error.message, /buffer limit/);

  timestamp = 100;
  assert.equal(
    (await host.handleRequest(uploadRequest('capped', 0, 1, 2, new Uint8Array([5, 6])))).ok,
    true
  );
  const missing = await host.handleRequest({
    ...timingRequest('missing-transfer', 'not-uploaded'),
    target: 'offscreen'
  });
  assert.equal(missing.ok, false);
  if (!missing.ok) {
    assert.equal(missing.error.code, 'invalid-request');
    assert.match(missing.error.message, /missing or incomplete/);
  }
  assert.equal(runtimeLoads, 0);
});

test('protocol rejects legacy whole-Blob/data-URL requests and audio chunks above 512 KiB', () => {
  const oldBlobRequest = {
    type: LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE,
    version: LOCAL_MODEL_OFFSCREEN_VERSION,
    target: 'background',
    requestId: 'old-blob-request',
    operation: 'timing',
    settings: DEFAULT_SETTINGS,
    job,
    audioTracks
  };
  assert.equal(isLocalModelOffscreenRequest(oldBlobRequest, 'background'), false);
  const roundTripped = JSON.parse(JSON.stringify(oldBlobRequest)) as {
    audioTracks: Array<{ blob: unknown }>;
  };
  assert.deepEqual(roundTripped.audioTracks[0].blob, {});
  assert.equal(isLocalModelOffscreenRequest(roundTripped, 'background'), false);

  const oldDataUrlRequest = {
    ...timingRequest('old-data-url'),
    audioTracks: [
      {
        trackId: 'track-1',
        source: 'captured.wav',
        mimeType: 'audio/wav',
        audioDataUrl: 'data:audio/wav;base64,UklGRg=='
      }
    ]
  };
  assert.equal(isLocalModelOffscreenRequest(oldDataUrlRequest, 'background'), false);

  const oversized = {
    ...uploadRequest('oversized', 0, 1, LOCAL_MODEL_AUDIO_CHUNK_BYTES + 1, new Uint8Array()),
    dataBase64: Buffer.alloc(LOCAL_MODEL_AUDIO_CHUNK_BYTES + 1).toString('base64')
  };
  assert.equal(isLocalModelOffscreenRequest(oversized, 'offscreen'), false);
});

test('client rejects mismatched responses and propagates host errors through the parent operation', async () => {
  const invalidClient = createLocalModelClient(async (request) => ({
    ...successResponse(request),
    requestId: 'different-request'
  }));
  await assert.rejects(
    invalidClient.generateLocalL0Draft(DEFAULT_SETTINGS, job),
    (error: unknown) =>
      error instanceof LocalModelBridgeError &&
      error.operation === 'draft' &&
      error.code === 'invalid-response' &&
      /invalid or mismatched response/.test(error.message)
  );

  const failureClient = createLocalModelClient(async (request) =>
    createLocalModelFailure(request, 'inference-failed', new Error('WASM backend initialization failed'))
  );
  await assert.rejects(
    failureClient.generateLocalL0SegmentDraft(DEFAULT_SETTINGS, 'task-1', row, preparedTracks),
    (error: unknown) =>
      error instanceof LocalModelBridgeError &&
      error.operation === 'segment' &&
      error.code === 'inference-failed' &&
      /WASM backend initialization failed/.test(error.message)
  );
});

for (const loss of ['eviction', 'restart'] as const) {
  test(`local drafting recaptures completed timing after offscreen ${loss} without growing the two-task cache`, async () => {
    const settings = { ...DEFAULT_SETTINGS, mode: 'advanced' as const, localModelsEnabled: true };
    const captured = ['Speaker 1', 'Speaker 2'].map((speakerKey, index) => ({
      ...audioTracks[0], speakerKey, trackId: `small-${index}`, blob: new Blob([new Uint8Array([index + 1])])
    }));
    const generatedTaskIds: string[] = [];
    const loadRuntime = async () => ({
      isLocalTimingCurrent: async () => true,
      generateLocalL0Timing: async (_settings: unknown, requestedJob: TranscriptJob) => {
        const taskId = buildCanonicalTaskIdentity(requestedJob);
        generatedTaskIds.push(taskId);
        return { ...timingResult, taskId };
      },
      generateLocalL0DraftFromTiming: async (timing: L0TimingResponse) => ({
        ...draftResult,
        rows: draftResult.rows.map((row) => ({ ...row, text: timing.taskId }))
      }),
      generateLocalL0SegmentDraft: async () => 'segment'
    });
    let host = createLocalModelHost(loadRuntime);
    const client = createLocalModelClient((message) =>
      host.handleRequest(JSON.parse(JSON.stringify({ ...message, target: 'offscreen' }))));
    let currentJob = job;
    let captures = 0;
    const service = new L0TimingService({
      captureTranscript: () => currentJob,
      currentTaskId: () => buildCanonicalTaskIdentity(currentJob),
      currentPathname: () => '/tasks/current',
      captureAudio: async () => { captures += 1; return captured; },
      getSettings: async () => settings,
      lookupTiming: async () => { throw new Error('Local timing must not use the remote cache'); },
      requestTiming: client.generateLocalL0Timing,
      requestLocalDraft: client.generateLocalL0Draft,
      publish: () => undefined,
      now: () => 0,
      schedule: () => { assert.fail('Recovery must not require retry timers'); }
    });
    const jobs = loss === 'eviction' ? [job, { ...job, jobId: 'task-2' }, { ...job, jobId: 'task-3' }] : [job];
    for (const nextJob of jobs) {
      currentJob = nextJob;
      service.onLifecycleOpportunity();
      await service.waitForTiming(currentJob, settings);
    }
    if (loss === 'restart') host = createLocalModelHost(loadRuntime);
    currentJob = job;
    service.onLifecycleOpportunity();
    assert.equal(captures, jobs.length, 'the content-side completed flag still predates the cache loss');
    await assert.rejects(client.generateLocalL0Draft(settings, job),
      (error: unknown) => error instanceof LocalModelBridgeError && error.code === 'timing-unavailable');
    const recovered = await service.generateLocalDraft(settings, job);
    assert.equal(recovered.rows[0].text, buildCanonicalTaskIdentity(job));
    assert.equal(captures, jobs.length + 1);
    assert.deepEqual(generatedTaskIds, [...jobs, job].map(buildCanonicalTaskIdentity));
    assert.deepEqual(getL0TimingAvailability(), { taskId: buildCanonicalTaskIdentity(job), status: 'available' });
    assert.deepEqual(await service.generateLocalDraft(settings, job), recovered);
    assert.equal(captures, jobs.length + 1, 'a cache hit must not capture or run ASR again');
  });
}

test('model replacement invalidates private timing before draft reuse and segment cache loss requests full capture', async () => {
  let current = true;
  const host = createLocalModelHost(async () => ({
    isLocalTimingCurrent: async () => current,
    generateLocalL0Timing: async () => timingResult,
    generateLocalL0DraftFromTiming: async () => draftResult,
    generateLocalL0SegmentDraft: async (_settings, taskId) => { throw new LocalTimingUnavailableError(taskId); }
  }));
  const uploaded = await uploadBlob(host, 'versioned-audio', audioTracks[0].blob);
  assert.equal((await host.handleRequest({ ...timingRequest('before-update', uploaded.audioTransferId), target: 'offscreen' })).ok, true);
  current = false;
  const draft = await host.handleRequest({
    type: LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE, version: LOCAL_MODEL_OFFSCREEN_VERSION, target: 'offscreen',
    operation: 'draft', requestId: 'after-update', settings: DEFAULT_SETTINGS, taskId: timingResult.taskId
  });
  assert.equal(draft.ok, false);
  if (!draft.ok) assert.equal(draft.error.code, 'timing-unavailable');
  const segmentAudio = await uploadBlob(host, 'segment-versioned-audio', audioTracks[0].blob);
  const segment = await host.handleRequest({
    type: LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE, version: LOCAL_MODEL_OFFSCREEN_VERSION, target: 'offscreen',
    operation: 'segment', requestId: 'after-private-eviction', settings: DEFAULT_SETTINGS, taskId: timingResult.taskId, row,
    tracks: [{ lane: 'Speaker 1', fieldName: 'audio:1', audio: segmentAudio }]
  });
  assert.equal(segment.ok, false);
  if (!segment.ok) assert.equal(segment.error.code, 'timing-unavailable');
});

test('enhancement streams two complete tracks in bounded downloads independently of ASR and releases every transfer', async () => {
  const bytes = new Uint8Array(LOCAL_MODEL_AUDIO_CHUNK_BYTES + 44);
  for (let index = 0; index < bytes.length; index++) bytes[index] = index % 251;
  const tracks = [audioTracks[0], { ...audioTracks[0], trackId: 'track-2', speakerKey: 'Speaker 2' }];
  const batch: EnhancedAudioBatch = {
    provider: 'browser-local', model: 'zipenhancer-verified', modelSha256: 'e'.repeat(64),
    tracks: tracks.map((track) => ({ bytes, metadata: {
      trackId: track.trackId, speakerKey: track.speakerKey!, trackLabel: track.trackLabel!,
      mimeType: 'audio/wav', sampleRate: 22050, frameCount: (bytes.length - 44) / 2,
      sourceSha256: 'a'.repeat(64), wavSha256: 'b'.repeat(64),
      totalBytes: bytes.length, chunkCount: 2
    } }))
  };
  let enhancements = 0;
  const host = createLocalModelHost(async () => { throw new Error('Enhancement must not initialize ASR'); }, {
    enhanceAudio: async (originals) => {
      enhancements++;
      assert.deepEqual(originals.map((track) => track.trackId), ['track-1', 'track-2']);
      for (const original of originals) assert.deepEqual(new Uint8Array(await original.blob.arrayBuffer()), audioBytes);
      return batch;
    }
  });
  const requests: LocalModelOffscreenRequest[] = [];
  const retainedOutputIds: string[] = [];
  const client = createLocalModelClient(async (request) => {
    assert.equal(isLocalModelOffscreenRequest(request, 'background'), true);
    assert.ok(JSON.stringify(request).length < 1024 * 1024, 'Chrome messages carry only a bounded chunk or metadata');
    requests.push(request);
    const response = await host.handleRequest(JSON.parse(JSON.stringify({ ...request, target: 'offscreen' })));
    if (response.ok && response.operation === 'enhanceAudio') retainedOutputIds.push(...response.result.tracks.map((track) => track.audioTransferId));
    return JSON.parse(JSON.stringify(response));
  });
  const chunks: AudioEnhancementChunk[] = [];
  const result = await client.enhanceAudio('native-review-1', tracks, { isCurrent: () => true, onAudioChunk: (chunk) => { chunks.push(chunk); } });
  assert.equal(enhancements, 1);
  assert.equal(result.taskId, 'native-review-1');
  assert.equal(result.modelSha256, batch.modelSha256);
  assert.equal(result.tracks.length, 2);
  assert.ok(result.tracks.every((track) => !('audioTransferId' in track)));
  assert.equal(chunks.length, 4);
  for (const track of tracks) {
    const downloaded = chunks.filter((chunk) => chunk.trackId === track.trackId);
    assert.deepEqual(downloaded.map((chunk) => chunk.chunkIndex), [0, 1]);
    assert.deepEqual(Buffer.concat(downloaded.map((chunk) => Buffer.from(decodeAudioChunk(chunk.dataBase64)))), Buffer.from(bytes));
  }
  assert.ok(requests.filter((request) => request.operation === 'enhanceAudio').every((request) => !('settings' in request)));
  assert.equal(requests.at(-1)?.operation, 'release');
  for (const transferId of retainedOutputIds) {
    const missing = await host.handleRequest({
      type: LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE, version: LOCAL_MODEL_OFFSCREEN_VERSION,
      target: 'offscreen', requestId: `after-cleanup:${transferId}`, operation: 'download', transferId, chunkIndex: 0
    });
    assert.equal(missing.ok, false, 'Completed output buffers are no longer retained');
  }
});

test('stale enhancement stops streaming and cleans retained output even when a caller rejects a chunk', async () => {
  const bytes = new Uint8Array(LOCAL_MODEL_AUDIO_CHUNK_BYTES + 44);
  const batch: EnhancedAudioBatch = { provider: 'browser-local', model: 'zipenhancer-verified', modelSha256: 'e'.repeat(64), tracks: [{
    bytes, metadata: { trackId: 'track-1', speakerKey: 'Speaker 1', trackLabel: 'Left microphone',
      mimeType: 'audio/wav', sampleRate: 16000, frameCount: (bytes.length - 44) / 2,
      sourceSha256: 'a'.repeat(64), wavSha256: 'b'.repeat(64), totalBytes: bytes.length, chunkCount: 2 }
  }] };
  const host = createLocalModelHost(async () => { throw new Error('ASR must remain unloaded'); }, { enhanceAudio: async () => batch });
  let outputId = '', current = true, emitted = 0;
  const client = createLocalModelClient(async (request) => {
    const response = await host.handleRequest({ ...request, target: 'offscreen' });
    if (response.ok && response.operation === 'enhanceAudio') outputId = response.result.tracks[0].audioTransferId;
    return response;
  });
  await assert.rejects(client.enhanceAudio('native-review-1', audioTracks, {
    isCurrent: () => current,
    onAudioChunk: () => { emitted++; current = false; }
  }), (error) => error instanceof LocalModelBridgeError && error.code === 'stale-task');
  assert.equal(emitted, 1);
  const missing = await host.handleRequest({
    type: LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE, version: LOCAL_MODEL_OFFSCREEN_VERSION,
    target: 'offscreen', requestId: 'after-stale', operation: 'download', transferId: outputId, chunkIndex: 1
  });
  assert.equal(missing.ok, false);
});

test('all-zero enhancement lanes reject undefined reference RMS rather than inventing enhanced silence', async () => {
  let neuralCalls = 0;
  await assert.rejects(enhanceZipSamples(new Float32Array(16000), 16000, async () => {
    neuralCalls++;
    throw new Error('An undefined source scale must fail before neural execution');
  }), /all-zero source lane.*RMS normalization is undefined/);
  assert.equal(neuralCalls, 0);
});

test('enhancement progress is isolated by request, native task and track and unsubscribes after terminal failures', async () => {
  const listeners = new Set<(message: unknown) => void>();
  const pending: Array<{
    request: LocalModelEnhanceAudioRequest;
    result: PromiseWithResolvers<LocalModelOffscreenResponse>;
  }> = [];
  const bothStarted = Promise.withResolvers<void>();
  const released: string[][] = [];
  const client = createLocalModelClient(async (request) => {
    if (request.operation === 'enhanceAudio') {
      const result = Promise.withResolvers<LocalModelOffscreenResponse>();
      pending.push({ request, result });
      if (pending.length === 2) bothStarted.resolve();
      return result.promise;
    }
    if (request.operation === 'release') {
      released.push(request.transferIds);
      return { type: LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE, version: LOCAL_MODEL_OFFSCREEN_VERSION,
        requestId: request.requestId, operation: 'release', ok: true, result: { released: true } };
    }
    return successResponse(request);
  }, (listener) => {
    listeners.add(listener);
    return () => { listeners.delete(listener); };
  });
  let firstCurrent = true;
  const firstProgress: AudioEnhancementProgress[] = [], secondProgress: AudioEnhancementProgress[] = [];
  const settled = Promise.allSettled([
    client.enhanceAudio('native-review-1', audioTracks, {
      isCurrent: () => firstCurrent, onAudioChunk: () => assert.fail('Failed inference cannot stream audio'),
      onProgress: (progress) => { firstProgress.push(progress); }
    }),
    client.enhanceAudio('native-review-2', audioTracks, {
      isCurrent: () => true, onAudioChunk: () => assert.fail('Failed inference cannot stream audio'),
      onProgress: (progress) => { secondProgress.push(progress); }
    })
  ]);
  await bothStarted.promise;
  const first = pending.find(({ request }) => request.taskId === 'native-review-1')!;
  const second = pending.find(({ request }) => request.taskId === 'native-review-2')!;
  const event: LocalModelEnhancementProgressMessage = {
    type: LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE, version: LOCAL_MODEL_OFFSCREEN_VERSION,
    target: 'content', event: 'enhancement-progress', operation: 'enhanceAudio',
    requestId: first.request.requestId, taskId: first.request.taskId,
    progress: { phase: 'enhancing', trackId: 'track-1', trackIndex: 0, trackCount: 1, completedChunks: 1, totalChunks: 3 }
  };
  for (const listener of listeners) {
    listener({ ...event, taskId: second.request.taskId });
    listener({ ...event, requestId: 'expired-request' });
    listener({ ...event, progress: { ...event.progress, trackId: 'another-track' } });
    listener({ ...event, progress: { ...event.progress, trackCount: 2 } });
    listener({ ...event, dataBase64: 'user-audio-must-not-enter-progress' });
  }
  assert.deepEqual(firstProgress, []);
  assert.deepEqual(secondProgress, []);
  for (const listener of listeners) listener(event);
  for (const listener of listeners) listener({ ...event, requestId: second.request.requestId, taskId: second.request.taskId });
  assert.deepEqual(firstProgress, [event.progress]);
  assert.deepEqual(secondProgress, [event.progress]);
  firstCurrent = false;
  for (const listener of listeners) listener({ ...event, progress: { ...event.progress, completedChunks: 2 } });
  assert.equal(firstProgress.length, 1, 'late progress for a no-longer-current native task is ignored');
  for (const { request, result } of pending) {
    result.resolve(createLocalModelFailure(request, 'inference-failed', new Error('WebGPU inference failed')));
  }
  const results = await settled;
  assert.ok(results.every((result) => result.status === 'rejected' &&
    result.reason instanceof LocalModelBridgeError && result.reason.code === 'inference-failed'));
  assert.equal(listeners.size, 0, 'every request-scoped listener is removed after failure');
  assert.equal(released.length, 2);
});

for (const terminal of ['success', 'failure'] as const) {
test(`background routes only owned offscreen progress and forgets requests after ${terminal}`, async () => {
  const inference = Promise.withResolvers<LocalModelOffscreenResponse>();
  const started = Promise.withResolvers<void>();
  const deliveries: Array<{ tabId: number; frameId: number; message: LocalModelEnhancementProgressMessage }> = [];
  const extensionId = 'gold-extension', offscreenDocumentUrl = `chrome-extension://${extensionId}/offscreen.html`;
  const bridge = createLocalModelOffscreenBridge({
    hasDocument: async () => true, createDocument: async () => undefined, closeDocument: async () => undefined,
    sendMessage: async () => { started.resolve(); return inference.promise; },
    workersReason: 'WORKERS' as chrome.offscreen.Reason,
    extensionId, offscreenDocumentUrl,
    sendProgressToTab: async (tabId, message, frameId) => { deliveries.push({ tabId, message, frameId }); }
  });
  const request: LocalModelEnhanceAudioRequest = {
    type: LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE, version: LOCAL_MODEL_OFFSCREEN_VERSION,
    target: 'background', requestId: 'owned-request', operation: 'enhanceAudio',
    taskId: 'native-review-1', audioTracks: [wireTrack('original-transfer')]
  };
  const result = bridge.handleRequest(request, { id: extensionId, tab: { id: 42 } as chrome.tabs.Tab, frameId: 3 });
  await started.promise;
  const event: LocalModelEnhancementProgressMessage = {
    type: LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE, version: LOCAL_MODEL_OFFSCREEN_VERSION,
    target: 'background', event: 'enhancement-progress', operation: 'enhanceAudio',
    requestId: request.requestId, taskId: request.taskId,
    progress: { phase: 'enhancing', trackId: 'track-1', trackIndex: 0, trackCount: 1, completedChunks: 1, totalChunks: 3 }
  };
  const offscreenSender = { id: extensionId, url: offscreenDocumentUrl };
  assert.equal(await bridge.handleProgress(event, { ...offscreenSender, id: 'another-extension' }), false);
  assert.equal(await bridge.handleProgress(event, { ...offscreenSender, url: `chrome-extension://${extensionId}/options.html` }), false);
  assert.equal(await bridge.handleProgress(event, { ...offscreenSender, tab: { id: 99 } as chrome.tabs.Tab }), false);
  assert.equal(await bridge.handleProgress({ ...event, taskId: 'another-task' }, offscreenSender), false);
  assert.equal(await bridge.handleProgress({ ...event, requestId: 'another-request' }, offscreenSender), false);
  const duplicate = await bridge.handleRequest(request, { id: extensionId, tab: { id: 99 } as chrome.tabs.Tab });
  assert.equal(duplicate.ok, false, 'another tab cannot steal an active request ID');
  assert.equal(await bridge.handleProgress(event, offscreenSender), true);
  assert.deepEqual(deliveries, [{ tabId: 42, frameId: 3, message: { ...event, target: 'content' } }]);
  inference.resolve(terminal === 'failure'
    ? createLocalModelFailure(request, 'inference-failed', new Error('WebGPU device lost'))
    : {
      type: LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE, version: LOCAL_MODEL_OFFSCREEN_VERSION,
      requestId: request.requestId, operation: 'enhanceAudio', ok: true,
      result: {
        ok: true, provider: 'browser-local', taskId: request.taskId, model: 'zipenhancer-verified', modelSha256: 'e'.repeat(64),
        tracks: [{ trackId: 'track-1', speakerKey: 'Speaker 1', trackLabel: 'Left microphone',
          mimeType: 'audio/wav', sampleRate: 16000, frameCount: 1,
          sourceSha256: 'a'.repeat(64), wavSha256: 'b'.repeat(64), totalBytes: 46, chunkCount: 1,
          audioTransferId: 'enhanced-output' }]
      }
    });
  const response = await result;
  assert.equal(response.ok, terminal === 'success');
  assert.equal(await bridge.handleProgress(event, offscreenSender), false, 'terminal requests no longer have an owner');
  assert.equal(deliveries.length, 1);
});
}

