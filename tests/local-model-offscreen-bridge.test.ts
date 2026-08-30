import test from 'node:test';
import assert from 'node:assert/strict';
import { Buffer } from 'node:buffer';

import { createLocalModelOffscreenBridge } from '../src/background/local-model-offscreen';
import { LocalModelBridgeError, createLocalModelClient } from '../src/core/local-model-client';
import {
  LOCAL_MODEL_AUDIO_CHUNK_BYTES,
  LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE,
  LOCAL_MODEL_OFFSCREEN_VERSION,
  createLocalModelFailure,
  decodeAudioChunk,
  encodeAudioChunk,
  isLocalModelOffscreenRequest,
  type LocalModelOffscreenRequest,
  type LocalModelOffscreenResponse,
  type LocalModelUploadRequest,
  type WireCapturedAudioTrack
} from '../src/core/local-model-offscreen-protocol';
import { DEFAULT_SETTINGS } from '../src/core/settings';
import type { PreparedL0Track } from '../src/core/l0-client';
import type {
  CapturedAudioTrack,
  L0DraftResponse,
  L0TimingResponse,
  TranscriptJob
} from '../src/core/types';
import { createLocalModelHost, type LocalModelHost } from '../src/offscreen/local-model-host';

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
  taskId: 'task-1',
  tracks: [
    {
      lane: 'Speaker 1',
      tokens: [{ id: 'token-1', text: 'hello', startSeconds: 1, endSeconds: 2 }]
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
    generateLocalL0Timing: async () => {
      calls.push('timing:start');
      timingStarted.resolve();
      await timingGate.promise;
      calls.push('timing:end');
      return timingResult;
    },
    generateLocalL0Draft: async () => {
      calls.push('draft');
      throw new Error('ONNX model file is missing');
    },
    generateLocalL0SegmentDraft: async () => 'segment'
  }));
  const timingTrack = await uploadBlob(host, 'host-timing-audio', audioTracks[0].blob);
  const draftTrack = await uploadBlob(host, 'host-draft-audio', audioTracks[0].blob);

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
    job,
    audioTracks: [draftTrack]
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
  const runtimeDraftTracks: CapturedAudioTrack[][] = [];
  const runtimeSegmentTracks: PreparedL0Track[][] = [];
  const host = createLocalModelHost(async () => ({
    generateLocalL0Timing: async (_settings, _job, tracks) => {
      runtimeTimingTracks.push(tracks);
      return timingResult;
    },
    generateLocalL0Draft: async (_settings, _job, tracks) => {
      runtimeDraftTracks.push(tracks);
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
  assert.equal(await client.generateLocalL0Draft(DEFAULT_SETTINGS, job, audioTracks), draftResult);
  assert.equal(
    await client.generateLocalL0SegmentDraft(DEFAULT_SETTINGS, 'task-1', row, preparedTracks),
    'Exact cropped text.'
  );

  assert.deepEqual(received.map((request) => request.operation), [
    'upload',
    'upload',
    'timing',
    'upload',
    'upload',
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
  assert.equal(uploads.length, 6);
  assert.ok(uploads.every((request) => decodeAudioChunk(request.dataBase64).byteLength <= LOCAL_MODEL_AUDIO_CHUNK_BYTES));
  assert.ok(uploads.every((request) => decodeAudioChunk(request.dataBase64).byteLength < audioTracks[0].blob.size));
  assert.ok(
    received
      .filter((request) => request.operation !== 'upload')
      .every((request) => !('dataBase64' in request))
  );

  for (const tracks of [runtimeTimingTracks[0], runtimeDraftTracks[0]]) {
    const audio = tracks[0];
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
  }
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
  assert.equal(reused.ok, false);
  if (!reused.ok) assert.equal(reused.error.code, 'invalid-request');
});

test('uploads do not initialize runtime and reject duplicates, gaps, buffer overflow, and missing transfers', async () => {
  let timestamp = 0;
  let runtimeLoads = 0;
  const host = createLocalModelHost(
    async () => {
      runtimeLoads += 1;
      return {
        generateLocalL0Timing: async () => timingResult,
        generateLocalL0Draft: async () => draftResult,
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
    invalidClient.generateLocalL0Draft(DEFAULT_SETTINGS, job, audioTracks),
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
