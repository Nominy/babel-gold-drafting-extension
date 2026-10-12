import assert from 'node:assert/strict';
import test, { type TestContext } from 'node:test';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'esbuild';
import { IDBFactory } from 'fake-indexeddb';
import modelArtifact from '../src/core/audio-enhancement-webgpu-model.json';
import { enhancementSha256 } from '../src/core/audio-enhancement-cache';
import { encodeEnhancementWav, readEnhancementWavHeader } from '../src/core/audio-enhancement-dsp';
import { createZipDspWorkerHandler } from '../src/core/audio-enhancement-worker';
import type { ZipDspRequest, ZipDspWorkerPort } from '../src/core/audio-enhancement-worker-protocol';
import type { ZipWebGpuEngine } from '../src/core/zipenhancer-webgpu';
import type { AudioEnhancementProgress } from '@nominy/babel-babel-runtime';
import type * as EnhancementRuntime from '../src/core/audio-enhancement-runtime';
import {
  isLocalModelEnhancementProgress, LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE, LOCAL_MODEL_OFFSCREEN_VERSION,
  type LocalModelEnhanceAudioRequest
} from '../src/core/local-model-offscreen-protocol';

type Output = { magnitude: Float32Array; phase: Float32Array };

function installGlobal(t: TestContext, name: string, value: unknown): void {
  const previous = Object.getOwnPropertyDescriptor(globalThis, name);
  Object.defineProperty(globalThis, name, { configurable: true, value });
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, name, previous);
    else Reflect.deleteProperty(globalThis, name);
  });
}

function enhanced(magnitude: Float32Array, phase: Float32Array): Output {
  return { magnitude: Float32Array.from(magnitude, value => value * 0.8), phase: phase.slice() };
}
function invalid(magnitude: Float32Array, phase: Float32Array): Output {
  const output = enhanced(magnitude, phase);
  output.phase[0] = Number.NaN;
  return output;
}
function tracks(prefix: string) {
  return [400, 537].map((frames, index) => {
    const pcm = Float32Array.from({ length: frames }, (_, sample) => 0.2 * Math.sin(sample / 13 + index));
    return { trackId: `${prefix}:${index}`, speakerKey: `speaker:${index}`, source: 'native', mimeType: 'audio/wav',
      blob: new Blob([encodeEnhancementWav(pcm, 16_000, frames)], { type: 'audio/wav' }) };
  });
}

let fixtureId = 0;
async function runtimeFixture(t: TestContext, options: {
  artifactPrecision?: string;
  precision?: string;
  tensors?: readonly { dataType: number }[];
  nodes?: readonly { op: string; attributes: Readonly<Record<string, unknown>> }[];
} = {}) {
  const directory = await mkdtemp(path.join(tmpdir(), 'babel-zip-fp32-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const weights = new Uint8Array([1, 2, 3, 4]), weightsSha256 = await enhancementSha256(weights.buffer);
  const plan = {
    checkpointSha256: modelArtifact.checkpointSha256, sourceGraphSha256: modelArtifact.sourceGraphSha256,
    kernelAbi: modelArtifact.kernelAbi, precision: options.precision ?? 'float32',
    tensors: options.tensors ?? [{ dataType: 1 }], nodes: options.nodes ?? [],
    weights: { file: 'zipenhancer-webgpu.weights.bin', sha256: weightsSha256, byteLength: weights.byteLength }
  };
  const planBytes = new TextEncoder().encode(JSON.stringify(plan));
  const artifact = { ...modelArtifact, sha256: 'a'.repeat(64), precision: options.artifactPrecision ?? 'float32',
    planSha256: await enhancementSha256(planBytes.buffer), weightsSha256, weightsByteLength: weights.byteLength
  };
  const key = `__zipFp32Fixture${++fixtureId}`, outfile = path.join(directory, 'runtime.mjs');
  await build({ entryPoints: [fileURLToPath(new URL('../src/core/audio-enhancement-runtime.ts', import.meta.url))],
    outfile, bundle: true, platform: 'node', format: 'esm', target: 'node22', logLevel: 'silent', plugins: [{
      name: 'controlled-neural-boundary', setup(builder) {
        builder.onResolve({ filter: /^\.\/zipenhancer-webgpu$/ }, () => ({ path: 'engine', namespace: 'controlled-neural-boundary' }));
        builder.onLoad({ filter: /.*/, namespace: 'controlled-neural-boundary' }, () => ({
          contents: `export const createZipWebGpuEngine = (...args) => globalThis[${JSON.stringify(key)}](...args);`, loader: 'js'
        }));
        builder.onLoad({ filter: /audio-enhancement-webgpu-model\.json$/ }, () => ({ contents: JSON.stringify(artifact), loader: 'json' }));
      }
    }] });
  const events: string[] = [], requests: ZipDspRequest[] = [], fetches: string[] = [];
  const engines: Array<{ disposed: boolean; disposal: PromiseWithResolvers<void> }> = [];
  let idle: (() => void) | undefined;
  const controls = {
    initialize: async (): Promise<void> => {},
    infer: async (magnitude: Float32Array, phase: Float32Array): Promise<Output> => enhanced(magnitude, phase),
    fetch: undefined as ((url: string, signal?: AbortSignal | null) => Promise<Response | undefined>) | undefined
  };
  installGlobal(t, key, async (plan: { precision: string }, _weights: Uint8Array, options: { adapter: GPUAdapter }) => {
    assert.equal(plan.precision, 'float32');
    assert.ok(engines.every(engine => engine.disposed), 'only one model/device may be resident');
    await options.adapter.requestDevice();
    const engine = { disposed: false, disposal: Promise.withResolvers<void>() };
    engines.push(engine);
    events.push('create:float32');
    await controls.initialize();
    let running = false;
    return {
      infer: async (magnitude: Float32Array, phase: Float32Array) => {
        assert.equal(running, false); assert.equal(engine.disposed, false);
        running = true;
        events.push('infer:float32');
        try { return await controls.infer(magnitude, phase); }
        finally { running = false; events.push('settled:float32'); }
      },
      dispose: async () => {
        assert.equal(running, false, 'actual inference must settle before resources are released');
        if (engine.disposed) return;
        events.push('dispose:float32');
        engine.disposed = true;
        engine.disposal.resolve();
      }
    } as ZipWebGpuEngine;
  });
  class OwnedDspWorker extends EventTarget implements ZipDspWorkerPort {
    closed = false;
    readonly handle = createZipDspWorkerHandler((response, transfer) => {
      const data = structuredClone(response, { transfer });
      queueMicrotask(() => { if (!this.closed) this.dispatchEvent(new MessageEvent('message', { data })); });
    }, () => {});
    postMessage(request: ZipDspRequest, transfer: ArrayBuffer[]): void {
      requests.push(request);
      const owned = structuredClone(request, { transfer });
      queueMicrotask(() => { if (!this.closed) this.handle(owned); });
    }
    terminate(): void { this.closed = true; events.push('worker:terminate'); }
  }
  installGlobal(t, 'Worker', OwnedDspWorker);
  installGlobal(t, 'chrome', { runtime: { getURL: (file: string) => `chrome-extension://fp32/${file}` } });
  installGlobal(t, 'navigator', { gpu: { requestAdapter: async () => {
    events.push('adapter:acquire');
    let consumed = false;
    return {
      info: { isFallbackAdapter: false }, features: { has: (feature: string) => feature === 'timestamp-query' },
      limits: { maxComputeWorkgroupStorageSize: 32768, maxComputeInvocationsPerWorkgroup: 256,
        maxComputeWorkgroupSizeX: 256, maxStorageBuffersPerShaderStage: 8 },
      requestDevice: async () => {
        if (consumed) throw new DOMException('GPUAdapter has already been consumed', 'OperationError');
        consumed = true;
        events.push('adapter:device');
        return {} as GPUDevice;
      }
    };
  } } });
  installGlobal(t, 'indexedDB', new IDBFactory());
  installGlobal(t, 'setTimeout', (callback: () => void, milliseconds: number) => {
    assert.equal(milliseconds, 120_000); idle = callback; return 1;
  });
  installGlobal(t, 'clearTimeout', () => { idle = undefined; });
  t.mock.method(globalThis, 'fetch', async (input: string | URL | Request, options?: RequestInit) => {
    const url = String(input);
    fetches.push(url);
    assert.match(url, /^chrome-extension:\/\/fp32\/dist\/models\/zipenhancer-webgpu\.(plan\.json|weights\.bin)$/, 'only primary packaged assets may be requested');
    const override = await controls.fetch?.(url, options?.signal);
    if (override) return override;
    if (url.endsWith('.weights.bin')) return new Response(weights);
    return new Response(planBytes);
  });
  // Each test imports its independently bundled runtime with a controlled neural boundary.
  const runtime = await import(pathToFileURL(outfile).href) as typeof EnhancementRuntime;
  const disposeIdle = async () => {
    if (!idle) return;
    const callback = idle; idle = undefined;
    const engine = engines.at(-1);
    callback();
    if (engine && !engine.disposed) await engine.disposal.promise;
  };
  t.after(disposeIdle);
  return { runtime, controls, events, engines, requests, fetches, disposeIdle };
}

test('owner, cache and localOnly consumers use one FP32 engine and retain exact Original clocks', async t => {
  const fixture = await runtimeFixture(t), originals = tracks('owner'), progress: AudioEnhancementProgress[] = [];
  const request: LocalModelEnhanceAudioRequest = {
    type: LOCAL_MODEL_OFFSCREEN_MESSAGE_TYPE, version: LOCAL_MODEL_OFFSCREEN_VERSION,
    target: 'offscreen', operation: 'enhanceAudio', requestId: 'owned-fp32', taskId: 'native-task',
    audioTracks: originals.map(({ blob: _blob, ...track }, index) => ({ ...track, audioTransferId: `transfer-${index}` }))
  };
  const batch = await fixture.runtime.enhanceAudioTracks(originals, notice => {
    const message = {
      type: request.type, version: request.version, target: 'content', event: 'enhancement-progress',
      operation: request.operation, requestId: request.requestId, taskId: request.taskId, progress: notice
    };
    assert.equal(isLocalModelEnhancementProgress(message, 'content', request), true,
      `The owning content tab must accept actual enhancement progress: ${JSON.stringify(notice)}`);
    assert.equal(notice.backendReason, undefined, 'local progress cannot claim remote routing');
    progress.push(notice);
  });
  assert.equal(batch.cacheStatus, 'stored');
  assert.equal(batch.tracks.length, 2);
  assert.deepEqual(fixture.events.filter(event => /^(create|dispose|infer):/.test(event)), [
    'create:float32', 'infer:float32', 'infer:float32'
  ]);
  assert.equal(fixture.events.filter(event => event === 'adapter:acquire').length, 1);
  assert.equal(fixture.events.filter(event => event === 'adapter:device').length, 1);
  assert.equal(fixture.fetches.length, 2, 'the primary plan and weights load exactly once');
  let previous: AudioEnhancementProgress | undefined;
  for (const notice of progress.filter(value => ['loading-model', 'enhancing', 'encoding'].includes(value.phase))) {
    assert.equal(notice.backend, 'webgpu');
    if (previous?.trackId === notice.trackId) {
      assert.ok(notice.completedChunks >= previous.completedChunks);
      if (previous.totalChunks > 0) assert.equal(notice.totalChunks, previous.totalChunks);
      assert.ok(['loading-model', 'enhancing', 'encoding'].indexOf(notice.phase) >=
        ['loading-model', 'enhancing', 'encoding'].indexOf(previous.phase));
    }
    previous = notice;
  }
  for (let index = 0; index < batch.tracks.length; index++) {
    const output = batch.tracks[index], original = await originals[index].blob.arrayBuffer();
    const sourceClock = readEnhancementWavHeader(original), outputClock = readEnhancementWavHeader(output.bytes.buffer);
    assert.equal(outputClock.frameCount, sourceClock.frameCount); assert.equal(outputClock.sampleRate, sourceClock.sampleRate);
    assert.equal(output.metadata.sourceSha256, await enhancementSha256(original));
    assert.notDeepEqual(output.bytes, new Uint8Array(original), 'Enhanced is neural output, never relabeled Original');
  }
  const fetches = fixture.fetches.length, runs = fixture.events.filter(event => event.startsWith('infer:')).length;
  const probes = fixture.events.filter(event => event === 'adapter:acquire').length;
  const cached = await fixture.runtime.enhanceAudioTracks(originals);
  assert.equal(cached.cacheStatus, 'hit'); assert.equal(fixture.fetches.length, fetches);
  assert.equal(fixture.events.filter(event => event.startsWith('infer:')).length, runs);
  assert.equal(fixture.events.filter(event => event === 'adapter:acquire').length, probes);
  await fixture.runtime.enhanceAudioTracks(tracks('volunteer'), undefined, { localOnly: true, cache: false });
  assert.equal(fixture.engines.length, 1, 'the sole engine remains resident across lanes and consumers until idle disposal');
  assert.equal(fixture.events.filter(event => event === 'infer:float32').length, 4);
  await fixture.disposeIdle();
  assert.ok(fixture.engines.every(engine => engine.disposed));
  await fixture.runtime.enhanceAudioTracks(tracks('after-idle'), undefined, { cache: false });
  assert.equal(fixture.engines.length, 2, 'new work after idle release creates one fresh engine');
});

for (const failure of ['device', 'audit', 'numerical', 'later-lane', 'asset', 'progress'] as const) {
  test(`${failure} rejects the complete pair without cache publication or fallback`, async t => {
    const fixture = await runtimeFixture(t), originals = tracks(failure);
    const snapshots = await Promise.all(originals.map(track => track.blob.arrayBuffer()));
    const reason = new Error(`${failure} rejected`);
    let calls = 0;
    fixture.controls.infer = async (magnitude, phase) => {
      calls++;
      if (failure === 'device' || failure === 'audit') throw reason;
      if (failure === 'numerical' || failure === 'later-lane' && calls === 2) return invalid(magnitude, phase);
      return enhanced(magnitude, phase);
    };
    if (failure === 'asset') fixture.controls.fetch = async () => { throw reason; };
    await assert.rejects(fixture.runtime.enhanceAudioTracks(originals, progress => {
      if (failure === 'progress' && progress.phase === 'enhancing' && progress.completedChunks > 0) throw reason;
    }), error => failure === 'numerical' || failure === 'later-lane'
      ? error instanceof Error && /invalid spectral values/.test(error.message) : error === reason);
    assert.ok(fixture.engines.every(engine => engine.disposed));
    assert.equal(fixture.requests.filter(request => request.type === 'finish').length, failure === 'later-lane' ? 1 : 0);
    assert.equal(fixture.engines.length, failure === 'asset' ? 0 : 1);
    assert.equal(calls, failure === 'asset' ? 0 : failure === 'later-lane' ? 2 : 1);
    assert.equal(fixture.events.filter(event => event === 'adapter:acquire').length, 1, 'failure never reacquires or reroutes');
    for (let index = 0; index < originals.length; index++) assert.deepEqual(await originals[index].blob.arrayBuffer(), snapshots[index]);
    fixture.controls.fetch = undefined;
    fixture.controls.infer = async (magnitude, phase) => enhanced(magnitude, phase);
    const retried = await fixture.runtime.enhanceAudioTracks(originals);
    assert.notEqual(retried.cacheStatus, 'hit', 'a rejected pair must never become a cache hit');
    assert.equal(retried.cacheStatus, 'stored');
    assert.equal(fixture.requests.filter(request => request.type === 'finish').length, failure === 'later-lane' ? 3 : 2);
  });
}

for (const stage of ['load', 'initialize', 'flight'] as const) {
  test(`cancellation during ${stage} drains owned work and releases the sole engine without publication`, async t => {
    const fixture = await runtimeFixture(t), controller = new AbortController(), reason = new Error('Owner cancelled enhancement');
    const originals = tracks(stage), started = Promise.withResolvers<void>(), gate = Promise.withResolvers<void>();
    if (stage === 'load') fixture.controls.fetch = async (_url, signal) => {
      started.resolve();
      await gate.promise;
      signal?.throwIfAborted();
      return undefined;
    };
    if (stage === 'initialize') fixture.controls.initialize = async () => { started.resolve(); await gate.promise; };
    if (stage === 'flight') fixture.controls.infer = async (magnitude, phase) => {
      started.resolve(); await gate.promise;
      return enhanced(magnitude, phase);
    };
    let settled = false;
    const result = fixture.runtime.enhanceAudioTracks(originals, undefined, { signal: controller.signal })
      .finally(() => { settled = true; });
    const rejected = assert.rejects(result, error => error === reason);
    await started.promise;
    controller.abort(reason);
    await Promise.resolve();
    assert.equal(settled, false);
    if (stage !== 'load') assert.equal(fixture.engines.at(-1)?.disposed, false);
    gate.resolve();
    await rejected;
    assert.ok(fixture.engines.every(engine => engine.disposed));
    assert.equal(fixture.requests.some(request => request.type === 'finish'), false);
    assert.equal(fixture.engines.length, stage === 'load' ? 0 : 1);
    fixture.controls.fetch = undefined;
    fixture.controls.initialize = async () => {};
    fixture.controls.infer = async (magnitude, phase) => enhanced(magnitude, phase);
    const retried = await fixture.runtime.enhanceAudioTracks(originals);
    assert.equal(retried.cacheStatus, 'stored', 'cancellation cannot publish a cache hit');
  });
}

for (const laneIndex of [0, 1]) {
  test(`cancellation before encoding lane ${laneIndex} never commits or returns an incomplete pair`, async t => {
    const fixture = await runtimeFixture(t), controller = new AbortController(), reason = new Error('Owner cancelled completion');
    const originals = tracks(`encoding-${laneIndex}`);
    await assert.rejects(fixture.runtime.enhanceAudioTracks(originals, progress => {
      if (progress.phase === 'encoding' && progress.trackIndex === laneIndex) controller.abort(reason);
    }, { signal: controller.signal }), error => error === reason);
    const runs = fixture.events.filter(event => event.startsWith('infer:')).length;
    const retried = await fixture.runtime.enhanceAudioTracks(originals);
    assert.equal(retried.cacheStatus, 'stored');
    assert.equal(fixture.events.filter(event => event.startsWith('infer:')).length, runs + 2);
  });
}

for (const [name, options] of [
  ['mixed artifact', { artifactPrecision: 'mixed-float16' }],
  ['mixed plan', { precision: 'mixed-float16' }],
  ['half tensor', { tensors: [{ dataType: 10 }] }],
  ['half Cast', { nodes: [{ op: 'Cast', attributes: { to: 10 } }] }],
  ['half fused Cast', { nodes: [{ op: 'ZipSwoosh', attributes: { steps: [{ op: 'Cast', to: 10 }] } }] }],
  ['nested half fused Cast', { nodes: [{ op: 'ZipSwoosh', attributes: { steps: [{ op: 'Where', steps: [{ op: 'Cast', to: 10 }] }] } }] }],
  ['half fused output', { nodes: [{ op: 'ZipSwoosh', attributes: { outputType: 10 } }] }],
] as const) {
  test(`${name} is rejected before any engine or worker allocation`, async t => {
    const fixture = await runtimeFixture(t, options);
    await assert.rejects(fixture.runtime.enhanceAudioTracks(tracks(name), undefined, { cache: false }));
    assert.equal(fixture.engines.length, 0);
    assert.equal(fixture.events.filter(event => event === 'adapter:device').length, 0);
    assert.equal(fixture.requests.length, 0);
    assert.equal(fixture.fetches.length, name === 'mixed artifact' ? 0 : 2);
  });
}
