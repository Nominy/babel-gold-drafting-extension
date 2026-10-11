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

type Precision = 'mixed-float16' | 'float32';
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
async function runtimeFixture(t: TestContext) {
  const directory = await mkdtemp(path.join(tmpdir(), 'babel-zip-recovery-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const weights = new Uint8Array([1, 2, 3, 4]), weightsSha256 = await enhancementSha256(weights.buffer);
  const plans = (['mixed-float16', 'float32'] as const).map(precision => ({
    checkpointSha256: modelArtifact.checkpointSha256, sourceGraphSha256: modelArtifact.sourceGraphSha256,
    kernelAbi: modelArtifact.kernelAbi, precision, tensors: [{ dataType: 1 }], nodes: [],
    weights: { file: precision === 'float32' ? 'zipenhancer-webgpu-fp32.weights.bin' : 'zipenhancer-webgpu.weights.bin',
      sha256: weightsSha256, byteLength: weights.byteLength }
  }));
  const planBytes = plans.map(plan => new TextEncoder().encode(JSON.stringify(plan)));
  const artifact = { ...modelArtifact, sha256: 'a'.repeat(64), precision: 'mixed-float16',
    planSha256: await enhancementSha256(planBytes[0].buffer), weightsSha256, weightsByteLength: weights.byteLength,
    recovery: { policy: 'invalid-spectra-float32-v1', precision: 'float32',
      planPath: 'models/zipenhancer-webgpu-fp32.plan.json', weightsPath: 'models/zipenhancer-webgpu-fp32.weights.bin',
      planSha256: await enhancementSha256(planBytes[1].buffer), weightsSha256, weightsByteLength: weights.byteLength }
  };
  const key = `__zipRecoveryFixture${++fixtureId}`, outfile = path.join(directory, 'runtime.mjs');
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
  const engines: Array<{ precision: Precision; disposed: boolean; disposal: PromiseWithResolvers<void> }> = [];
  let idle: (() => void) | undefined;
  const controls = {
    recoveryAdapterAvailable: true,
    infer: async (precision: Precision, magnitude: Float32Array, phase: Float32Array): Promise<Output> =>
      precision === 'mixed-float16' ? invalid(magnitude, phase) : enhanced(magnitude, phase),
    fetch: undefined as ((url: string, signal?: AbortSignal | null) => Promise<Response | undefined>) | undefined
  };
  installGlobal(t, key, async (plan: { precision: Precision }, _weights: Uint8Array, options: { adapter: GPUAdapter }) => {
    assert.ok(engines.every(engine => engine.disposed), 'only one model/device may be resident');
    await options.adapter.requestDevice();
    const engine = { precision: plan.precision, disposed: false, disposal: Promise.withResolvers<void>() };
    engines.push(engine);
    events.push(`create:${engine.precision}`);
    let running = false;
    return {
      infer: async (magnitude: Float32Array, phase: Float32Array) => {
        assert.equal(running, false); assert.equal(engine.disposed, false);
        running = true;
        events.push(`infer:${engine.precision}`);
        try { return await controls.infer(engine.precision, magnitude, phase); }
        finally { running = false; events.push(`settled:${engine.precision}`); }
      },
      dispose: async () => {
        assert.equal(running, false, 'actual inference must settle before resources are released');
        if (engine.disposed) return;
        events.push(`dispose:${engine.precision}`);
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
  installGlobal(t, 'chrome', { runtime: { getURL: (file: string) => `chrome-extension://recovery/${file}` } });
  let adapterCount = 0;
  installGlobal(t, 'navigator', { gpu: { requestAdapter: async () => {
    adapterCount++;
    events.push('adapter:acquire');
    if (adapterCount > 1 && !controls.recoveryAdapterAvailable) return null;
    let consumed = false;
    return {
      info: { isFallbackAdapter: false }, features: { has: (feature: string) => feature === 'shader-f16' || feature === 'timestamp-query' },
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
    assert.match(url, /^chrome-extension:\/\/recovery\/dist\/models\//, 'fixtures never upload originals');
    const override = await controls.fetch?.(url, options?.signal);
    if (override) return override;
    if (url.endsWith('.weights.bin')) return new Response(weights);
    return new Response(planBytes[url.includes('-fp32.') ? 1 : 0]);
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

test('local owner, cache and localOnly consumers share one memory-safe latched GPU recovery and retain exact Original clocks', async t => {
  const fixture = await runtimeFixture(t), originals = tracks('owner'), progress: AudioEnhancementProgress[] = [];
  let firstInput: Output | undefined;
  fixture.controls.infer = async (precision, magnitude, phase) => {
    if (precision === 'mixed-float16') { firstInput = { magnitude, phase }; return invalid(magnitude, phase); }
    if (firstInput) {
      assert.equal(magnitude, firstInput.magnitude); assert.equal(phase, firstInput.phase);
      firstInput = undefined;
    }
    return enhanced(magnitude, phase);
  };
  const batch = await fixture.runtime.enhanceAudioTracks(originals, notice => { progress.push(notice); });
  assert.equal(batch.cacheStatus, 'stored');
  assert.equal(batch.tracks.length, 2);
  assert.deepEqual(fixture.events.filter(event => /^(create|dispose|infer):/.test(event)), [
    'create:mixed-float16', 'infer:mixed-float16', 'dispose:mixed-float16', 'create:float32', 'infer:float32', 'infer:float32'
  ]);
  assert.equal(fixture.events.filter(event => event === 'adapter:device').length, 2, 'mixed and FP32 initialization consume distinct adapters');
  const recovery = progress.find(notice => notice.backendReason?.includes('full FP32'));
  assert.ok(recovery); assert.equal(recovery.phase, 'enhancing'); assert.equal(recovery.completedChunks, 0); assert.equal(recovery.totalChunks, 1);
  let previous: AudioEnhancementProgress | undefined;
  for (const notice of progress.filter(value => ['loading-model', 'enhancing', 'encoding'].includes(value.phase))) {
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
  const cached = await fixture.runtime.enhanceAudioTracks(originals);
  assert.equal(cached.cacheStatus, 'hit'); assert.equal(fixture.fetches.length, fetches);
  assert.equal(fixture.events.filter(event => event.startsWith('infer:')).length, runs);
  await fixture.runtime.enhanceAudioTracks(tracks('volunteer'), undefined, { localOnly: true, cache: false });
  assert.equal(fixture.engines.length, 2, 'FP32 remains latched across tracks and consumers until idle disposal');
  assert.equal(fixture.events.filter(event => event === 'infer:mixed-float16').length, 1);
  await fixture.disposeIdle();
  assert.ok(fixture.engines.every(engine => engine.disposed));
});

for (const failure of ['device', 'fp32-numerical', 'fp32-later-lane', 'fp32-asset', 'fp32-adapter'] as const) {
  test(`${failure} rejects the complete pair without cache publication or further fallback`, async t => {
    const fixture = await runtimeFixture(t), originals = tracks(failure);
    let fullCalls = 0;
    fixture.controls.infer = async (precision, magnitude, phase) => {
      if (failure === 'device') throw new Error('GPU device lost');
      if (precision === 'float32' && failure === 'fp32-later-lane' && ++fullCalls === 1) return enhanced(magnitude, phase);
      return invalid(magnitude, phase);
    };
    if (failure === 'fp32-asset') fixture.controls.fetch = async url => url.includes('-fp32.') ? new Response(null, { status: 503 }) : undefined;
    if (failure === 'fp32-adapter') fixture.controls.recoveryAdapterAvailable = false;
    await assert.rejects(fixture.runtime.enhanceAudioTracks(originals),
      failure === 'device' ? /GPU device lost/ : failure === 'fp32-asset' ? /assets could not be loaded/
        : failure === 'fp32-adapter' ? /fresh hardware WebGPU adapter/ : /invalid spectral values/);
    assert.ok(fixture.engines.every(engine => engine.disposed));
    assert.equal(fixture.requests.filter(request => request.type === 'finish').length, failure === 'fp32-later-lane' ? 1 : 0);
    assert.equal(fixture.engines.length, failure === 'fp32-numerical' || failure === 'fp32-later-lane' ? 2 : 1);
    if (failure === 'device' || failure === 'fp32-adapter') assert.equal(fixture.fetches.some(url => url.includes('-fp32.')), false);
    fixture.controls.fetch = undefined;
    fixture.controls.recoveryAdapterAvailable = true;
    fixture.controls.infer = async (_precision, magnitude, phase) => enhanced(magnitude, phase);
    const retried = await fixture.runtime.enhanceAudioTracks(originals);
    assert.notEqual(retried.cacheStatus, 'hit', 'a rejected pair must never become a cache hit');
    assert.equal(retried.cacheStatus, 'stored');
  });
}

for (const stage of ['recovery-load', 'recovery-flight'] as const) {
  test(`cancellation during ${stage} drains owned work, publishes nothing and disposes the single resident engine`, async t => {
    const fixture = await runtimeFixture(t), controller = new AbortController(), reason = new Error('Owner cancelled recovery');
    const started = Promise.withResolvers<void>(), gate = Promise.withResolvers<void>();
    if (stage === 'recovery-load') fixture.controls.fetch = async (url, signal) => {
      if (!url.includes('-fp32.')) return undefined;
      started.resolve();
      await gate.promise;
      signal?.throwIfAborted();
      return undefined;
    };
    fixture.controls.infer = async (precision, magnitude, phase) => {
      if (precision === 'mixed-float16') return invalid(magnitude, phase);
      started.resolve(); await gate.promise;
      return enhanced(magnitude, phase);
    };
    let settled = false;
    const result = fixture.runtime.enhanceAudioTracks(tracks(stage), undefined, { signal: controller.signal, cache: false })
      .finally(() => { settled = true; });
    const rejected = assert.rejects(result, error => error === reason);
    await started.promise;
    controller.abort(reason);
    await Promise.resolve();
    assert.equal(settled, false);
    if (stage === 'recovery-flight') assert.equal(fixture.engines.at(-1)?.disposed, false);
    gate.resolve();
    await rejected;
    assert.ok(fixture.engines.every(engine => engine.disposed));
    assert.equal(fixture.requests.some(request => request.type === 'finish'), false);
    assert.equal(fixture.engines.length, stage === 'recovery-flight' ? 2 : 1);
  });
}
