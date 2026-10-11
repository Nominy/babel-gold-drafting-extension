import type { AudioEnhancementCacheStatus, AudioEnhancementProgress, AudioEnhancementTrackMetadata } from '@nominy/babel-babel-runtime';
import modelArtifact from './audio-enhancement-webgpu-model.json';
import type { ZipWebGpuEngine } from './zipenhancer-webgpu';
import type { ZipWebGpuPlan } from './zipenhancer-webgpu-plan';
import { enhanceAudioThroughSwarm } from './audio-enhancement-swarm';
import type { EnhancementModelDescriptor } from './audio-enhancement-swarm-protocol';
import { selectZipEnhancementBackend } from './audio-enhancement-backend';
import type { ZipEnhancementBackend } from './audio-enhancement-backend';
import { LOCAL_MODEL_AUDIO_CHUNK_BYTES } from './local-model-offscreen-protocol';
import { decodeEnhancementWav, ZIP_BINS } from './audio-enhancement-dsp';
import { enhanceZipSamplesInWorker } from './audio-enhancement-pipeline';
import { createPackagedZipDspWorker, throwIfZipCancelled, ZipDspWorkerClient } from './audio-enhancement-worker-client';
import { runExclusiveGpuInference } from './local-gpu-run-queue';
import { createAudioEnhancementPairCache, enhancementSha256, runWithAudioEnhancementPairCache } from './audio-enhancement-cache';
import type { CapturedAudioTrack, ExtensionSettings } from './types';
import type { AudioEnhancementPairCache, PreparedEnhancementSource } from './audio-enhancement-cache';

const CHECKPOINT_SHA256 = 'b18896915e27a821585584221d0c0820f35e12145315ae3f1e73ccd5a68d195f';
const SOURCE_GRAPH_SHA256 = '2f18c8f7ff10a2702d6243ce1230db9e73e6804dd6cd7b20d8e191ee06924016';
const MODEL_IDLE_MS = 2 * 60 * 1000;
const INPUT_FRAMES = 641;
const TENSOR_ELEMENTS = ZIP_BINS * INPUT_FRAMES;
interface ZipModelAssets {
  planSha256: string;
  weightsSha256: string;
  weightsByteLength: number;
  planPath: string;
  weightsPath: string;
  precision: 'float32' | 'mixed-float16';
}
interface ZipModelArtifact extends ZipModelAssets {
  id: string;
  sha256: string;
  checkpointSha256: string;
  sourceGraphSha256: string;
  recovery: ZipModelAssets & { policy: 'invalid-spectra-float32-v1'; precision: 'float32' };
  kernelAbi: string;
  sampleRate: number;
  fftSize: number;
  hopSize: number;
  winSize: number;
  compressFactor: number;
  chunkSeconds: number;
  strideSeconds: number;
  inputFrames: number;
}
export interface EnhancedAudioTrack {
  metadata: AudioEnhancementTrackMetadata;
  bytes: Uint8Array<ArrayBuffer>;
}
export interface EnhancedAudioBatch {
  provider: 'browser-local' | 'swarm';
  model: string;
  modelSha256: string;
  tracks: EnhancedAudioTrack[];
  cacheStatus?: AudioEnhancementCacheStatus;
  cacheMessage?: string;
}
interface ZipResources { engine: ZipWebGpuEngine; precision: ZipModelAssets['precision'] }
let resourcesPromise: Promise<ZipResources> | null = null;
let idleTimer: number | NodeJS.Timeout | null = null;
let enhancementTail: Promise<void> = Promise.resolve();
let ownedRequestTail: Promise<void> = Promise.resolve();
let activeRequests = 0;
let pairCache: AudioEnhancementPairCache | undefined;

function artifact(): ZipModelArtifact {
  const value = modelArtifact as ZipModelArtifact;
  const recovery = value.recovery;
  if (!value.id || ![value.sha256, value.planSha256, value.weightsSha256].every(hash => /^[a-f0-9]{64}$/.test(hash)) ||
    value.checkpointSha256 !== CHECKPOINT_SHA256 || value.sourceGraphSha256 !== SOURCE_GRAPH_SHA256 ||
    value.sampleRate !== 16000 || value.fftSize !== 400 || value.winSize !== 400 || value.hopSize !== 100 ||
    value.compressFactor !== 0.3 || value.chunkSeconds !== 4 || value.strideSeconds !== 3 || value.inputFrames !== INPUT_FRAMES ||
    value.kernelAbi !== 'zip-relative-attention-v1+swoosh-v1' || !['float32', 'mixed-float16'].includes(value.precision) ||
    !Number.isSafeInteger(value.weightsByteLength) || value.weightsByteLength <= 0 ||
    value.planPath !== 'models/zipenhancer-webgpu.plan.json' || value.weightsPath !== 'models/zipenhancer-webgpu.weights.bin' ||
    recovery?.policy !== 'invalid-spectra-float32-v1' || recovery.precision !== 'float32' ||
    ![recovery.planSha256, recovery.weightsSha256].every(hash => /^[a-f0-9]{64}$/.test(hash)) ||
    !Number.isSafeInteger(recovery.weightsByteLength) || recovery.weightsByteLength <= 0 ||
    recovery.planPath !== 'models/zipenhancer-webgpu-fp32.plan.json' || recovery.weightsPath !== 'models/zipenhancer-webgpu-fp32.weights.bin') {
    throw new Error('The packaged ZipEnhancer browser assets are not bound to the verified checkpoint and DSP contract.');
  }
  return value;
}
export function getAudioEnhancementModel(): EnhancementModelDescriptor {
  const metadata = artifact();
  return { id: metadata.id, sha256: metadata.sha256, sourceGraphSha256: metadata.sourceGraphSha256 };
}


async function initialize(
  selection: Extract<ZipEnhancementBackend, { backend: 'webgpu' }>, recovery: boolean, signal?: AbortSignal
): Promise<ZipResources> {
  throwIfZipCancelled(signal);
  const metadata = artifact(), assets = recovery ? metadata.recovery : metadata, runtime = globalThis.chrome?.runtime;
  if (typeof runtime?.getURL !== 'function') throw new Error('ZipEnhancer cannot locate its packaged browser model assets.');
  const [planResponse, weightsResponse] = await Promise.all([
    fetch(runtime.getURL(`dist/${assets.planPath}`), { signal }),
    fetch(runtime.getURL(`dist/${assets.weightsPath}`), { signal }),
  ]);
  if (!planResponse.ok || !weightsResponse.ok) throw new Error(`The packaged ZipEnhancer assets could not be loaded (plan HTTP ${planResponse.status}, weights HTTP ${weightsResponse.status}).`);
  const [planBytes, weightsBytes] = await Promise.all([planResponse.arrayBuffer(), weightsResponse.arrayBuffer()]);
  throwIfZipCancelled(signal);
  if (await enhancementSha256(planBytes) !== assets.planSha256 || weightsBytes.byteLength !== assets.weightsByteLength) throw new Error('ZipEnhancer browser asset digest/length mismatch.');
  const plan = JSON.parse(new TextDecoder().decode(planBytes)) as ZipWebGpuPlan;
  if (plan.checkpointSha256 !== metadata.checkpointSha256 || plan.sourceGraphSha256 !== metadata.sourceGraphSha256 ||
      plan.precision !== assets.precision || plan.kernelAbi !== metadata.kernelAbi ||
      plan.weights.sha256 !== assets.weightsSha256 || plan.weights.byteLength !== assets.weightsByteLength ||
      plan.weights.file !== assets.weightsPath.slice('models/'.length) ||
      recovery && (plan.tensors.some(tensor => tensor.dataType === 10) || plan.nodes.some(node => node.op === 'Cast' && node.attributes.to === 10))) {
    throw new Error('ZipEnhancer browser plan and packaged identity disagree.');
  }
  // Platform-specific import: JSEP evaluates browser-only GPU APIs, unavailable
  // to GPU-ineligible browsers and Node offscreen-host tests. Cache hits never load it.
  const { createZipWebGpuEngine } = await import('./zipenhancer-webgpu');
  throwIfZipCancelled(signal);
  return { engine: await createZipWebGpuEngine(plan, new Uint8Array(weightsBytes), { adapter: selection.adapter }), precision: assets.precision };
}

async function resources(
  selection: Extract<ZipEnhancementBackend, { backend: 'webgpu' }>, signal?: AbortSignal
): Promise<ZipResources> {
  if (!resourcesPromise) {
    const pending: Promise<ZipResources> = runExclusiveGpuInference(() => initialize(selection, false, signal)).catch((error) => {
      if (resourcesPromise === pending) resourcesPromise = null;
      throw error;
    });
    resourcesPromise = pending;
  }
  return resourcesPromise;
}

async function promoteToFullPrecision(
  resource: ZipResources, signal?: AbortSignal
): Promise<void> {
  await runExclusiveGpuInference(async () => {
    throwIfZipCancelled(signal);
    // The failed mixed flight has settled. Never keep two model/device allocations resident.
    await resource.engine.dispose();
    throwIfZipCancelled(signal);
    // requestDevice consumes its adapter; recovery must acquire a fresh hardware adapter, never delegate.
    const selection = await selectZipEnhancementBackend(undefined, 'auto');
    throwIfZipCancelled(signal);
    if (selection.backend !== 'webgpu') throw new Error('ZipEnhancer full-FP32 recovery could not acquire a fresh hardware WebGPU adapter.');
    const recovery = await initialize(selection, true, signal);
    resource.engine = recovery.engine;
    resource.precision = recovery.precision;
  });
}

async function releaseResources(): Promise<void> {
  const previous = resourcesPromise;
  resourcesPromise = null;
  if (previous) {
    try {
      const { engine } = await previous;
      await runExclusiveGpuInference(() => engine.dispose());
    } catch { /* Preserve the originating task/device failure; never accept partial output. */ }
  }
}

async function infer(resource: ZipResources, magnitude: Float32Array, phase: Float32Array, frames: number, signal?: AbortSignal): Promise<{ magnitude: Float32Array; phase: Float32Array }> {
  if (frames !== INPUT_FRAMES || magnitude.length !== TENSOR_ELEMENTS || phase.length !== TENSOR_ELEMENTS) {
    throw new Error(`ZipEnhancer requires exactly [1, ${ZIP_BINS}, ${INPUT_FRAMES}] magnitude/phase before inference.`);
  }
  return runExclusiveGpuInference(async () => {
    throwIfZipCancelled(signal);
    const output = await resource.engine.infer(magnitude, phase);
    throwIfZipCancelled(signal);
    return output;
  });
}

export interface AudioEnhancementRunOptions {
  signal?: AbortSignal;
  taskId?: string;
  /** Leased work must never delegate again or persist another user's recordings. */
  localOnly?: boolean;
  cache?: boolean;
  /** Trusted saved preference/coordinator snapshot supplied by the offscreen host. */
  backend?: ExtensionSettings['audioEnhancementBackend'];
  swarmBaseUrl?: string;
  /** Host admission is acquired only for local execution, never for cache or remote waits. */
  runLocal?: <T>(action: () => Promise<T>) => Promise<T>;
}

async function enhanceAudioLocally(
  sources: PreparedEnhancementSource[],
  selection: Extract<ZipEnhancementBackend, { backend: 'webgpu' }>,
  reportProgress: (progress: AudioEnhancementProgress) => Promise<void>,
  options: AudioEnhancementRunOptions
): Promise<EnhancedAudioTrack[]> {
  let worker: ZipDspWorkerClient | undefined;
  try {
    throwIfZipCancelled(options.signal);
    const enhanced: EnhancedAudioTrack[] = [];
    for (let trackIndex = 0; trackIndex < sources.length; trackIndex++) {
      const { track, bytes: source, sourceSha256 } = sources[trackIndex];
      const { samples, sampleRate } = decodeEnhancementWav(source);
      const frameCount = samples.length; // Save before worker transfer detaches source PCM.
      const trackProgress = { trackId: track.trackId, trackIndex, trackCount: sources.length };
      let completed = 0;
      if (!resourcesPromise) {
        await reportProgress({ ...trackProgress, phase: 'loading-model', completedChunks: 0, totalChunks: 0 });
      }
      const resource = await resources(selection, options.signal);
      throwIfZipCancelled(options.signal);
      worker ??= new ZipDspWorkerClient(createPackagedZipDspWorker(), options.signal);
      const { bytes } = await enhanceZipSamplesInWorker(samples, sampleRate, (magnitude, phase, frames) => infer(resource, magnitude, phase, frames, options.signal), {
        client: worker, signal: options.signal,
        ...(resource.precision === 'mixed-float16' ? {
          canRecoverNumericalFailure: () => resource.precision === 'mixed-float16',
          recoverNumericalFailure: async (magnitude: Float32Array, phase: Float32Array, frames: number) => {
            await promoteToFullPrecision(resource, options.signal);
            return infer(resource, magnitude, phase, frames, options.signal);
          },
          // backendReason describes remote routing in the existing wire contract.
          // Recovery stays local and must remain valid for already-installed Helper clients.
          onNumericalRecovery: (_index: number, totalChunks: number) => reportProgress({
            ...trackProgress, phase: 'enhancing', completedChunks: completed, totalChunks
          })
        } : {}),
        onChunkCompleted: (completedChunks, totalChunks) => {
          completed = completedChunks;
          return reportProgress({ ...trackProgress, phase: 'enhancing', completedChunks, totalChunks });
        },
        onChunksReady: (totalChunks) => reportProgress({ ...trackProgress, phase: 'enhancing', completedChunks: 0, totalChunks }),
        onEncoding: (totalChunks) => reportProgress({ ...trackProgress, phase: 'encoding', completedChunks: totalChunks, totalChunks })
      });
      enhanced.push({ bytes, metadata: {
        trackId: track.trackId, speakerKey: track.speakerKey ?? track.trackId,
        trackLabel: track.trackLabel ?? track.speakerKey ?? track.trackId, mimeType: 'audio/wav',
        sampleRate, frameCount, sourceSha256, wavSha256: await enhancementSha256(bytes.buffer),
        totalBytes: bytes.byteLength, chunkCount: Math.ceil(bytes.byteLength / LOCAL_MODEL_AUDIO_CHUNK_BYTES)
      } });
    }
    throwIfZipCancelled(options.signal);
    return enhanced;
  } catch (error) {
    worker?.close();
    await releaseResources();
    throw error;
  } finally { worker?.close(); }
}

function runLocalEnhancement(action: () => Promise<EnhancedAudioTrack[]>): Promise<EnhancedAudioTrack[]> {
  if (idleTimer !== null) { globalThis.clearTimeout(idleTimer); idleTimer = null; }
  activeRequests++;
  const result = enhancementTail.then(action);
  enhancementTail = result.then(() => undefined, () => undefined);
  return result.finally(() => {
    activeRequests--;
    if (!activeRequests && resourcesPromise) idleTimer = globalThis.setTimeout(() => {
      idleTimer = null;
      void releaseResources();
    }, MODEL_IDLE_MS);
  });
}

async function enhanceAudioTracksSelected(
  tracks: CapturedAudioTrack[],
  onProgress: ((progress: AudioEnhancementProgress) => void | Promise<void>) | undefined,
  options: AudioEnhancementRunOptions
): Promise<EnhancedAudioBatch> {
  throwIfZipCancelled(options.signal);
  const metadata = artifact();
  let backend: ZipEnhancementBackend | undefined;
  let provider: EnhancedAudioBatch['provider'] = 'browser-local';
  const reportProgress = async (progress: AudioEnhancementProgress): Promise<void> => {
    throwIfZipCancelled(options.signal);
    await onProgress?.({ ...progress, ...(backend ? {
      backend: backend.backend, ...(backend.backend === 'swarm' ? { backendReason: progress.backendReason ?? backend.reason } : {})
    } : {}) });
    throwIfZipCancelled(options.signal);
  };
  const cache = options.localOnly || options.cache === false ? null : (pairCache ??= createAudioEnhancementPairCache());
  const batch = await runWithAudioEnhancementPairCache(tracks, metadata, cache, async sources => {
    const selection = await selectZipEnhancementBackend(undefined, options.localOnly ? 'auto' : options.backend);
    backend = selection;
    throwIfZipCancelled(options.signal);
    if (selection.backend === 'swarm') {
      if (options.localOnly) throw new Error('This swarm worker has no usable hardware WebGPU adapter.');
      if (!options.taskId) throw new Error('Remote enhancement requires the current native task identity.');
      provider = 'swarm';
      return enhanceAudioThroughSwarm(sources, getAudioEnhancementModel(), {
        taskId: options.taskId, signal: options.signal, onProgress: reportProgress, baseUrl: options.swarmBaseUrl
      });
    }
    const run = () => runLocalEnhancement(() => enhanceAudioLocally(sources, selection, reportProgress, options));
    return options.runLocal ? options.runLocal(run) : run();
  }, reportProgress, options.signal);
  return { ...batch, provider };
}

export async function enhanceAudioTracks(
  tracks: CapturedAudioTrack[],
  onProgress?: (progress: AudioEnhancementProgress) => void | Promise<void>,
  options: AudioEnhancementRunOptions = {}
): Promise<EnhancedAudioBatch> {
  throwIfZipCancelled(options.signal);
  if (!tracks.length || tracks.some((track) => !track.trackId) || new Set(tracks.map((track) => track.trackId)).size !== tracks.length) return Promise.reject(new Error('ZipEnhancer requires distinct original source tracks.'));
  const run = () => enhanceAudioTracksSelected(tracks, onProgress, options);
  // Serialize the owner's pair-cache selection/commit, not remote waits with GPU work.
  // Volunteer leases neither touch this cache nor wait on an owner's off-device request.
  if (options.localOnly || options.cache === false) return run();
  const result = ownedRequestTail.then(run);
  ownedRequestTail = result.then(() => undefined, () => undefined);
  return result;
}
