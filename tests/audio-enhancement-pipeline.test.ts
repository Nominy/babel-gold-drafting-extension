import test, { after, before, type TestContext } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Worker as ThreadWorker } from 'node:worker_threads';
import { build } from 'esbuild';
import {
  encodeEnhancementWav, enhanceZipSamples, readEnhancementWavHeader,
  ZIP_CHUNK_SAMPLES, ZIP_SAMPLE_RATE, ZIP_STRIDE_SAMPLES
} from '../src/core/audio-enhancement-dsp';
import { enhanceZipSamplesInWorker } from '../src/core/audio-enhancement-pipeline';
import { ZipDspWorkerClient } from '../src/core/audio-enhancement-worker-client';
import { createZipDspWorkerHandler } from '../src/core/audio-enhancement-worker';
import { parseZipDspResponse, type ZipDspRequest, type ZipDspResponse, type ZipDspWorkerPort } from '../src/core/audio-enhancement-worker-protocol';
import { runExclusiveGpuInference } from '../src/core/local-gpu-run-queue';

let temporaryDirectory: string;
let workerPath: string;
before(async () => {
  temporaryDirectory = await mkdtemp(path.join(tmpdir(), 'babel-zip-dsp-'));
  workerPath = path.join(temporaryDirectory, 'worker.mjs');
  await build({ entryPoints: [fileURLToPath(new URL('./fixtures/audio-enhancement-worker-thread.ts', import.meta.url))],
    outfile: workerPath, bundle: true, platform: 'node', format: 'esm', target: 'node22', logLevel: 'silent' });
});
after(async () => { await rm(temporaryDirectory, { recursive: true, force: true }); });

class WorkerFailureEvent extends Event {
  constructor(readonly message: string) { super('error'); }
}
class RealDspWorker extends EventTarget implements ZipDspWorkerPort {
  readonly thread = new ThreadWorker(workerPath);
  readonly exited = Promise.withResolvers<number>();
  readonly requests: Array<{ type: ZipDspRequest['type']; index?: number }> = [];
  readonly responses: ZipDspResponse[] = [];
  readonly transferredLengths: number[] = [];
  readonly outstanding = new Set<number>();
  maximumPending = 0;
  terminateCount = 0;
  onRequest?: (request: ZipDspRequest) => void;
  onResponse?: (response: ZipDspResponse) => void;
  constructor(t: TestContext) {
    super();
    this.thread.on('message', (data: unknown) => {
      const response = parseZipDspResponse(data);
      this.outstanding.delete(response.id);
      this.responses.push(response);
      this.onResponse?.(response);
      this.dispatchEvent(new MessageEvent('message', { data }));
    });
    this.thread.on('error', (error) => this.dispatchEvent(new WorkerFailureEvent(error.message)));
    this.thread.on('exit', (code) => this.exited.resolve(code));
    t.after(async () => { if (!this.terminateCount) this.terminate(); await this.exited.promise; });
  }
  postMessage(request: ZipDspRequest, transfer: ArrayBuffer[]): void {
    this.requests.push({ type: request.type, ...('index' in request ? { index: request.index } : {}) });
    this.outstanding.add(request.id);
    this.maximumPending = Math.max(this.maximumPending, this.outstanding.size);
    this.onRequest?.(request);
    this.thread.postMessage(request, transfer);
    this.transferredLengths.push(...transfer.map((buffer) => buffer.byteLength));
  }
  terminate(): void { this.terminateCount++; void this.thread.terminate(); }
}

function sourceLane(frameCount: number, phase = 0): Float32Array<ArrayBuffer> {
  return Float32Array.from({ length: frameCount }, (_, index) =>
    Math.fround((index % 997 < 31 ? 0.00000013 : 0.21) * Math.sin(index / 13 + phase) + 0.00002 * Math.cos(index / 79)));
}
function transformed(magnitude: Float32Array, phase: Float32Array) {
  for (let index = 0; index < magnitude.length; index++) {
    magnitude[index] *= 0.9;
    phase[index] += 0.015;
  }
  return { magnitude, phase };
}
async function oracle(source: Float32Array, sampleRate: number) {
  const output = await enhanceZipSamples(source, sampleRate, async (magnitude, phase) => transformed(magnitude, phase));
  return encodeEnhancementWav(output, sampleRate, source.length);
}

for (const [sampleRate, frameCount] of [
  [ZIP_SAMPLE_RATE, 400], [ZIP_SAMPLE_RATE, ZIP_CHUNK_SAMPLES],
  [ZIP_SAMPLE_RATE, ZIP_CHUNK_SAMPLES + ZIP_STRIDE_SAMPLES + 137],
  [48_000, 3 * ZIP_CHUNK_SAMPLES + 412], [44_100, 4 * 44_100 + 137]
] as const) {
  test(`real worker matches sequential arithmetic and full first/tail clock at ${sampleRate} Hz/${frameCount} frames`, async (t) => {
    const source = sourceLane(frameCount), expected = await oracle(source.slice(), sampleRate);
    const worker = new RealDspWorker(t), completed: Array<[number, number]> = [];
    const result = await enhanceZipSamplesInWorker(source, sampleRate, async (magnitude, phase) => transformed(magnitude, phase), {
      createWorker: () => worker,
      onChunkCompleted: (count, total) => { completed.push([count, total]); }
    });
    assert.equal(source.byteLength, 0, 'owned original PCM was transferred, not copied');
    assert.deepEqual(result.bytes, expected);
    assert.deepEqual(readEnhancementWavHeader(result.bytes.buffer).frameCount, frameCount);
    assert.equal(result.sampleRate, sampleRate);
    assert.deepEqual(completed, Array.from({ length: result.totalChunks }, (_, index) => [index + 1, result.totalChunks]));
    assert.ok(worker.transferredLengths.length > 2 && worker.transferredLengths.every((length) => length === 0));
    assert.ok(worker.maximumPending <= 2);
    assert.equal(worker.terminateCount, 1);
    assert.equal(worker.requests.at(-1)?.type, 'finish');
  });
}

test('lookahead and acknowledged prior ISTFT run during the next sole neural flight; callbacks gate further flights', async (t) => {
  const worker = new RealDspWorker(t), started = [Promise.withResolvers<void>(), Promise.withResolvers<void>(), Promise.withResolvers<void>()];
  const gates = [Promise.withResolvers<void>(), Promise.withResolvers<void>(), Promise.withResolvers<void>()];
  const lookedAhead = Promise.withResolvers<void>(), firstCompleted = Promise.withResolvers<void>();
  const callbackGate = Promise.withResolvers<void>(), secondFinished = Promise.withResolvers<void>();
  const secondCompleted = Promise.withResolvers<void>();
  const completed: number[] = [];
  let calls = 0, active = 0, maximumActive = 0;
  worker.onResponse = (response) => {
    if (response.ok && response.type === 'spectrum' && response.index === 1) lookedAhead.resolve();
  };
  worker.onRequest = (request) => {
    if (request.type === 'reconstruct') assert.equal(active, request.index < 2 ? 1 : 0, 'prior reconstruction overlaps the next flight, final reconstruction follows its flight');
  };
  const enhancement = enhanceZipSamplesInWorker(sourceLane(ZIP_CHUNK_SAMPLES + ZIP_STRIDE_SAMPLES + 137), ZIP_SAMPLE_RATE, async (magnitude, phase) => {
    const index = calls++;
    active++; maximumActive = Math.max(maximumActive, active);
    started[index]?.resolve();
    if (gates[index]) await gates[index].promise;
    active--;
    if (index === 1) secondFinished.resolve();
    return transformed(magnitude, phase);
  }, { createWorker: () => worker, onChunkCompleted: async (count) => {
    completed.push(count);
    if (count === 1) { firstCompleted.resolve(); await callbackGate.promise; }
    if (count === 2) secondCompleted.resolve();
  } });
  await started[0].promise;
  await lookedAhead.promise;
  assert.deepEqual(completed, [], 'prepared/submitted spectra are not completed');
  gates[0].resolve();
  await started[1].promise;
  await firstCompleted.promise;
  assert.equal(active, 1);
  assert.deepEqual(completed, [1]);
  gates[1].resolve();
  await secondFinished.promise;
  assert.equal(calls, 2);
  callbackGate.resolve();
  await started[2].promise;
  await secondCompleted.promise;
  assert.equal(active, 1);
  gates[2].resolve();
  const result = await enhancement;
  assert.equal(result.totalChunks, 3);
  assert.equal(maximumActive, 1);
  assert.deepEqual(completed, [1, 2, 3]);
  assert.deepEqual(worker.requests.map(({ type, index }) => `${type}:${index ?? ''}`),
    ['initialize:', 'spectrum:0', 'spectrum:1', 'reconstruct:0', 'spectrum:2', 'reconstruct:1', 'reconstruct:2', 'finish:']);
  assert.equal(worker.maximumPending, 2);
});

test('one batch-owned worker resets RMS/STFT state between different lanes without retaining output buffers', async (t) => {
  const worker = new RealDspWorker(t), client = new ZipDspWorkerClient(worker);
  try {
    for (const [sampleRate, frameCount, phase] of [[16_000, 64_137, 0], [48_000, 192_412, 1.7]] as const) {
      const source = sourceLane(frameCount, phase), expected = await oracle(source.slice(), sampleRate);
      const result = await enhanceZipSamplesInWorker(source, sampleRate, async (magnitude, phase) => transformed(magnitude, phase), { client });
      assert.deepEqual(result.bytes, expected);
      assert.equal(worker.terminateCount, 0, 'batch owns worker lifetime across lanes');
    }
  } finally { client.close(); }
  assert.equal(worker.requests.filter(({ type }) => type === 'initialize').length, 2);
  assert.equal(worker.terminateCount, 1);
});

for (const failure of ['neural', 'inverse-STFT', 'progress'] as const) {
  test(`${failure} rejection terminates speculation, drains the real outstanding flight and never publishes a partial lane`, async (t) => {
    const worker = new RealDspWorker(t), secondStarted = Promise.withResolvers<void>();
    const secondGate = Promise.withResolvers<void>(), callbackStarted = Promise.withResolvers<void>();
    const completed: number[] = [];
    let calls = 0, settled = false;
    const enhancement = enhanceZipSamplesInWorker(sourceLane(ZIP_CHUNK_SAMPLES + ZIP_STRIDE_SAMPLES + 137), ZIP_SAMPLE_RATE, async (magnitude, phase) => {
      const index = calls++;
      if (index === 1) {
        secondStarted.resolve();
        await secondGate.promise;
        if (failure === 'neural') throw new Error('WebGPU device lost during window 2');
      }
      if (index === 0 && failure === 'inverse-STFT') phase[0] = Number.NaN;
      return transformed(magnitude, phase);
    }, { createWorker: () => worker, onChunkCompleted: (count) => {
      callbackStarted.resolve();
      if (failure === 'progress') throw new Error('Owning tab rejected progress');
      completed.push(count);
    } });
    const observed = enhancement.then(() => { settled = true; }, (error: unknown) => { settled = true; throw error; });
    const rejected = assert.rejects(observed, failure === 'neural' ? /WebGPU device lost/ : failure === 'inverse-STFT' ? /invalid spectral values/ : /rejected progress/);
    await secondStarted.promise;
    if (failure === 'neural') {
      await callbackStarted.promise;
      assert.deepEqual(completed, [1]);
    } else {
      await worker.exited.promise;
      assert.equal(settled, false, 'failure cannot release the outstanding neural/profiler flight early');
      assert.deepEqual(completed, []);
    }
    secondGate.resolve();
    await rejected;
    assert.equal(calls, 2);
    assert.equal(worker.terminateCount, 1);
    assert.ok(worker.maximumPending <= 2);
    assert.equal(worker.requests.some(({ type }) => type === 'finish'), false);
    assert.equal(worker.requests.some(({ type, index }) => type === 'reconstruct' && index === 1), false);
  });
}

test('cancellation terminates pending worker work but keeps neural admission until its flight settles', async (t) => {
  const worker = new RealDspWorker(t), controller = new AbortController();
  const started = Promise.withResolvers<void>(), gate = Promise.withResolvers<void>();
  const completed: number[] = [];
  let settled = false;
  const enhancement = enhanceZipSamplesInWorker(sourceLane(ZIP_CHUNK_SAMPLES + 137), ZIP_SAMPLE_RATE, async (magnitude, phase) => {
    started.resolve(); await gate.promise;
    return transformed(magnitude, phase);
  }, { signal: controller.signal, createWorker: () => worker, onChunkCompleted: (count) => { completed.push(count); } });
  const observed = enhancement.finally(() => { settled = true; });
  const rejected = assert.rejects(observed, /Task cancelled/);
  await started.promise;
  controller.abort(new Error('Task cancelled'));
  await worker.exited.promise;
  assert.equal(settled, false);
  gate.resolve();
  await rejected;
  assert.deepEqual(completed, []);
  assert.equal(worker.terminateCount, 1);
  assert.equal(worker.requests.some(({ type }) => type === 'finish'), false);
});

test('pre-cancelled lanes allocate no worker', async () => {
  const controller = new AbortController(); controller.abort(new Error('Already cancelled'));
  let created = 0;
  await assert.rejects(enhanceZipSamplesInWorker(sourceLane(400), ZIP_SAMPLE_RATE, async (magnitude, phase) => transformed(magnitude, phase), {
    signal: controller.signal, createWorker: () => { created++; throw new Error('Must not allocate'); }
  }), /Already cancelled/);
  assert.equal(created, 0);
});

test('real worker rejects out-of-order reconstruction and rejects every queued acknowledgement', async (t) => {
  const worker = new RealDspWorker(t), client = new ZipDspWorkerClient(worker);
  await client.initialize(sourceLane(ZIP_CHUNK_SAMPLES + 137), ZIP_SAMPLE_RATE);
  const first = await client.nextSpectrum(0);
  const badReconstruction = client.reconstruct(1, first);
  const queuedSpectrum = client.nextSpectrum(1);
  const results = await Promise.allSettled([badReconstruction, queuedSpectrum]);
  assert.ok(results.every((result) => result.status === 'rejected' && result.reason instanceof Error && /out of order/.test(result.reason.message)));
  assert.equal(first.magnitude.byteLength, 0);
  assert.equal(first.phase.byteLength, 0);
  assert.equal(worker.terminateCount, 1);
  assert.equal(worker.maximumPending, 2);
});

test('real worker enforces a single speculative spectrum and cannot finish an unfinished lane', async (t) => {
  for (const operation of ['lookahead', 'finish'] as const) {
    const worker = new RealDspWorker(t), client = new ZipDspWorkerClient(worker);
    await client.initialize(sourceLane(ZIP_CHUNK_SAMPLES + ZIP_STRIDE_SAMPLES + 137), ZIP_SAMPLE_RATE);
    await client.nextSpectrum(0);
    await client.nextSpectrum(1);
    await assert.rejects(operation === 'lookahead' ? client.nextSpectrum(2) : client.finish(),
      operation === 'lookahead' ? /one-window lookahead/ : /partial source lane/);
    assert.equal(worker.terminateCount, 1);
  }
});

test('worker processor transfers detached spectra without invalidating its compact overlap history', async () => {
  const replies: ZipDspResponse[] = [];
  let detachedBuffers = 0, closed = 0;
  const handle = createZipDspWorkerHandler((response, transfer) => {
    replies.push(structuredClone(response, { transfer }));
    for (const buffer of transfer) { assert.equal(buffer.byteLength, 0); detachedBuffers++; }
  }, () => { closed++; });
  const source = sourceLane(ZIP_CHUNK_SAMPLES + 137), expected = await oracle(source.slice(), ZIP_SAMPLE_RATE);
  const frameCount = source.length;
  handle(structuredClone({ id: 1, type: 'initialize', samples: source, sampleRate: ZIP_SAMPLE_RATE, frameCount }, { transfer: [source.buffer] }));
  assert.equal(source.byteLength, 0);
  let id = 2;
  for (let index = 0; index < 2; index++) {
    handle({ id: id++, type: 'spectrum', index });
    const reply = replies.at(-1);
    assert.ok(reply?.ok && reply.type === 'spectrum');
    transformed(reply.spectrum.magnitude, reply.spectrum.phase);
    handle(structuredClone({ id: id++, type: 'reconstruct', index, spectrum: reply.spectrum },
      { transfer: [reply.spectrum.magnitude.buffer, reply.spectrum.phase.buffer] }));
    assert.equal(reply.spectrum.magnitude.byteLength, 0);
    assert.equal(reply.spectrum.phase.byteLength, 0);
  }
  handle({ id: id++, type: 'finish' });
  const result = replies.at(-1);
  assert.ok(result?.ok && result.type === 'finished');
  assert.deepEqual(result.bytes, expected);
  assert.equal(detachedBuffers, 5);
  assert.equal(closed, 0, 'batch client, not lane completion, owns successful worker shutdown');
});

test('shared GPU admission serializes observer ownership and recovers after a rejected audited run', async () => {
  const gate = Promise.withResolvers<void>(), started = Promise.withResolvers<void>();
  const events: string[] = [];
  const first = runExclusiveGpuInference(async () => { events.push('zip:start'); started.resolve(); await gate.promise; events.push('zip:restore'); throw new Error('audit rejected'); });
  const rejected = assert.rejects(first, /audit rejected/);
  const second = runExclusiveGpuInference(async () => { events.push('asr:start'); events.push('asr:restore'); return 7; });
  await started.promise;
  assert.deepEqual(events, ['zip:start']);
  gate.resolve();
  await rejected;
  assert.equal(await second, 7);
  assert.deepEqual(events, ['zip:start', 'zip:restore', 'asr:start', 'asr:restore']);
});

test('transfers copy only nonowned PCM/ORT subviews and leave unrelated buffer owners intact', async (t) => {
  const pool = sourceLane(802), source = pool.subarray(201, 601), originalPool = pool.slice();
  const expected = await oracle(source.slice(), ZIP_SAMPLE_RATE), worker = new RealDspWorker(t);
  const outputOwners: ArrayBuffer[] = [];
  const result = await enhanceZipSamplesInWorker(source, ZIP_SAMPLE_RATE, async (magnitude, phase) => {
    transformed(magnitude, phase);
    const magOwner = new Float32Array(magnitude.length + 2), phaOwner = new Float32Array(phase.length + 2);
    magOwner.set(magnitude, 1); phaOwner.set(phase, 1);
    outputOwners.push(magOwner.buffer, phaOwner.buffer);
    return { magnitude: magOwner.subarray(1, magOwner.length - 1), phase: phaOwner.subarray(1, phaOwner.length - 1) };
  }, { createWorker: () => worker });
  assert.deepEqual(result.bytes, expected);
  assert.deepEqual(pool, originalPool);
  assert.ok(outputOwners.every((buffer) => buffer.byteLength > 0), 'ORT subviews must not detach a larger allocator-owned buffer');
  assert.ok(worker.transferredLengths.every((length) => length === 0), 'the owned transfer copies themselves detach');
});

for (const stage of ['ready', 'encoding'] as const) {
  test(`${stage} callback rejection closes the real worker without returning or encoding a partial lane`, async (t) => {
    const worker = new RealDspWorker(t);
    let neuralCalls = 0;
    await assert.rejects(enhanceZipSamplesInWorker(sourceLane(400), ZIP_SAMPLE_RATE, async (magnitude, phase) => {
      neuralCalls++;
      return transformed(magnitude, phase);
    }, { createWorker: () => worker,
      onChunksReady: () => { if (stage === 'ready') throw new Error('Initial progress rejected'); },
      onEncoding: () => { if (stage === 'encoding') throw new Error('Encoding progress rejected'); }
    }), stage === 'ready' ? /Initial progress rejected/ : /Encoding progress rejected/);
    assert.equal(neuralCalls, stage === 'ready' ? 0 : 1);
    assert.equal(worker.terminateCount, 1);
    assert.equal(worker.requests.some(({ type }) => type === 'finish'), false);
  });
}

for (const sourceError of ['zero', 'nonfinite'] as const) {
  test(`failed whole-lane ${sourceError} normalization closes the worker before any neural run or completion`, async (t) => {
    const worker = new RealDspWorker(t), source = sourceError === 'zero' ? new Float32Array(400) : sourceLane(400);
    if (sourceError === 'nonfinite') source[201] = Number.NaN;
    let neuralCalls = 0, completions = 0;
    await assert.rejects(enhanceZipSamplesInWorker(source, ZIP_SAMPLE_RATE, async (magnitude, phase) => {
      neuralCalls++;
      return transformed(magnitude, phase);
    }, { createWorker: () => worker, onChunkCompleted: () => { completions++; } }),
    sourceError === 'zero' ? /all-zero source lane/ : /source PCM is nonfinite/);
    assert.equal(neuralCalls, 0);
    assert.equal(completions, 0);
    assert.equal(worker.terminateCount, 1);
    assert.deepEqual(worker.requests.map(({ type }) => type), ['initialize']);
  });
}
