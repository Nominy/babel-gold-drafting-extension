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
import type { CapturedAudioTrack } from './types';
import type { AudioEnhancementPairCache } from './audio-enhancement-cache';
import { requireReviewGraderAccess, reviewGraderAccess } from './review-grader-access';

const CHECKPOINT_SHA256 = 'b18896915e27a821585584221d0c0820f35e12145315ae3f1e73ccd5a68d195f';
const SOURCE_GRAPH_SHA256 = '2f18c8f7ff10a2702d6243ce1230db9e73e6804dd6cd7b20d8e191ee06924016';
const MODEL_IDLE_MS = 2 * 60 * 1000;
const INPUT_FRAMES = 641;
const TENSOR_ELEMENTS = ZIP_BINS * INPUT_FRAMES;
interface ZipModelArtifact {
  id: string;
  sha256: string;
  checkpointSha256: string;
  sourceGraphSha256: string;
  planSha256: string;
  weightsSha256: string;
  weightsByteLength: number;
  planPath: string;
  weightsPath: string;
  precision: 'float32' | 'mixed-float16';
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
interface ZipResources { engine: ZipWebGpuEngine }
let resourcesPromise: Promise<ZipResources> | null = null;
let idleTimer: number | NodeJS.Timeout | null = null;
let enhancementTail: Promise<void> = Promise.resolve();
let activeRequests = 0;
let pairCache: AudioEnhancementPairCache | undefined;
let observeAccess: (() => void) | undefined;

function artifact(): ZipModelArtifact {
  const value = modelArtifact as ZipModelArtifact;
  if (!value.id || ![value.sha256, value.planSha256, value.weightsSha256].every(hash => /^[a-f0-9]{64}$/.test(hash)) ||
    value.checkpointSha256 !== CHECKPOINT_SHA256 || value.sourceGraphSha256 !== SOURCE_GRAPH_SHA256 ||
    value.sampleRate !== 16000 || value.fftSize !== 400 || value.winSize !== 400 || value.hopSize !== 100 ||
    value.compressFactor !== 0.3 || value.chunkSeconds !== 4 || value.strideSeconds !== 3 || value.inputFrames !== INPUT_FRAMES ||
    value.kernelAbi !== 'zip-relative-attention-v1+swoosh-v1' || !['float32', 'mixed-float16'].includes(value.precision) ||
    !Number.isSafeInteger(value.weightsByteLength) || value.weightsByteLength <= 0 ||
    value.planPath !== 'models/zipenhancer-webgpu.plan.json' || value.weightsPath !== 'models/zipenhancer-webgpu.weights.bin') {
    throw new Error('The packaged ZipEnhancer browser assets are not bound to the verified checkpoint and DSP contract.');
  }
  return value;
}
export function getAudioEnhancementModel(): EnhancementModelDescriptor {
  const metadata = artifact();
  return { id: metadata.id, sha256: metadata.sha256, sourceGraphSha256: metadata.sourceGraphSha256 };
}


async function initialize(selection: Extract<ZipEnhancementBackend, { backend: 'webgpu' }>): Promise<ZipResources> {
  const metadata = artifact(), runtime = globalThis.chrome?.runtime;
  if (typeof runtime?.getURL !== 'function') throw new Error('ZipEnhancer cannot locate its packaged browser model assets.');
  const [planResponse, weightsResponse] = await Promise.all([
    fetch(runtime.getURL(`dist/${metadata.planPath}`)),
    fetch(runtime.getURL(`dist/${metadata.weightsPath}`)),
  ]);
  if (!planResponse.ok || !weightsResponse.ok) throw new Error(`The packaged ZipEnhancer assets could not be loaded (plan HTTP ${planResponse.status}, weights HTTP ${weightsResponse.status}).`);
  const [planBytes, weightsBytes] = await Promise.all([planResponse.arrayBuffer(), weightsResponse.arrayBuffer()]);
  if (await enhancementSha256(planBytes) !== metadata.planSha256 || weightsBytes.byteLength !== metadata.weightsByteLength) throw new Error('ZipEnhancer browser asset digest/length mismatch.');
  const plan = JSON.parse(new TextDecoder().decode(planBytes)) as ZipWebGpuPlan;
  if (plan.checkpointSha256 !== metadata.checkpointSha256 || plan.sourceGraphSha256 !== metadata.sourceGraphSha256 ||
      plan.precision !== metadata.precision || plan.kernelAbi !== metadata.kernelAbi ||
      plan.weights.sha256 !== metadata.weightsSha256 || plan.weights.byteLength !== metadata.weightsByteLength) throw new Error('ZipEnhancer browser plan and packaged identity disagree.');
  // Platform-specific import: JSEP evaluates browser-only GPU APIs, unavailable
  // to GPU-ineligible browsers and Node offscreen-host tests. Cache hits never load it.
  const { createZipWebGpuEngine } = await import('./zipenhancer-webgpu');
  return { engine: await createZipWebGpuEngine(plan, new Uint8Array(weightsBytes), { adapter: selection.adapter }) };
}

async function resources(selection: Extract<ZipEnhancementBackend, { backend: 'webgpu' }>): Promise<ZipResources> {
  if (!resourcesPromise) {
    const pending: Promise<ZipResources> = runExclusiveGpuInference(() => initialize(selection)).catch((error) => {
      if (resourcesPromise === pending) resourcesPromise = null;
      throw error;
    });
    resourcesPromise = pending;
  }
  return resourcesPromise;
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

async function infer(resource: ZipResources, magnitude: Float32Array, phase: Float32Array, frames: number): Promise<{ magnitude: Float32Array; phase: Float32Array }> {
  if (frames !== INPUT_FRAMES || magnitude.length !== TENSOR_ELEMENTS || phase.length !== TENSOR_ELEMENTS) {
    throw new Error(`ZipEnhancer requires exactly [1, ${ZIP_BINS}, ${INPUT_FRAMES}] magnitude/phase before inference.`);
  }
  return runExclusiveGpuInference(() => resource.engine.infer(magnitude, phase));
}

export interface AudioEnhancementRunOptions {
  signal?: AbortSignal;
  taskId?: string;
  /** Leased work must never delegate again or persist another user's recordings. */
  localOnly?: boolean;
  cache?: boolean;
}

async function enhanceAudioTracksExclusive(
  tracks: CapturedAudioTrack[],
  onProgress: ((progress: AudioEnhancementProgress) => void | Promise<void>) | undefined,
  options: AudioEnhancementRunOptions
): Promise<EnhancedAudioBatch> {
  let worker: ZipDspWorkerClient | undefined;
  try {
    throwIfZipCancelled(options.signal);
    const metadata = artifact();
    const selection = await selectZipEnhancementBackend();
    if (selection.backend === 'swarm' && options.localOnly) throw new Error('This swarm worker has no usable hardware WebGPU adapter.');
    const reportProgress = async (progress: AudioEnhancementProgress): Promise<void> => {
      throwIfZipCancelled(options.signal);
      await onProgress?.({ ...progress, backend: selection.backend, ...(selection.backend === 'swarm' ? { backendReason: selection.reason } : {}) });
      throwIfZipCancelled(options.signal);
    };
    const batch = await runWithAudioEnhancementPairCache(tracks, metadata, options.cache === false ? null : (pairCache ??= createAudioEnhancementPairCache()), async (sources) => {
      if (selection.backend === 'swarm') {
        if (!options.taskId) throw new Error('Swarm enhancement requires the current native task identity.');
        return enhanceAudioThroughSwarm(sources, getAudioEnhancementModel(), { taskId: options.taskId, signal: options.signal, onProgress: reportProgress });
      }
      const enhanced: EnhancedAudioTrack[] = [];
      for (let trackIndex = 0; trackIndex < sources.length; trackIndex++) {
        const { track, bytes: source, sourceSha256 } = sources[trackIndex];
        const { samples, sampleRate } = decodeEnhancementWav(source);
        const frameCount = samples.length; // Save before worker transfer detaches source PCM.
        const trackProgress = { trackId: track.trackId, trackIndex, trackCount: tracks.length };
        if (!resourcesPromise) {
          await reportProgress({ ...trackProgress, phase: 'loading-model', completedChunks: 0, totalChunks: 0 });
        }
        const resource = await resources(selection);
        throwIfZipCancelled(options.signal);
        worker ??= new ZipDspWorkerClient(createPackagedZipDspWorker(), options.signal);
        const { bytes } = await enhanceZipSamplesInWorker(samples, sampleRate, (magnitude, phase, frames) => infer(resource, magnitude, phase, frames), {
          client: worker, signal: options.signal,
          onChunkCompleted: (completedChunks, totalChunks) => reportProgress({ ...trackProgress, phase: 'enhancing', completedChunks, totalChunks }),
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
      worker?.close(); worker = undefined;
      return enhanced;
    }, reportProgress, options.signal);
    return { ...batch, provider: selection.backend === 'swarm' ? 'swarm' : 'browser-local' };
  } catch (error) {
    worker?.close();
    await releaseResources();
    throw error;
  } finally { worker?.close(); }
}

export async function enhanceAudioTracks(
  tracks: CapturedAudioTrack[],
  onProgress?: (progress: AudioEnhancementProgress) => void | Promise<void>,
  options: AudioEnhancementRunOptions = {}
): Promise<EnhancedAudioBatch> {
  const grant = await requireReviewGraderAccess();
  const signal = options.signal ? AbortSignal.any([options.signal, grant]) : grant;
  signal.throwIfAborted();
  options = { ...options, signal };
  observeAccess ??= reviewGraderAccess().subscribe(available => {
    if (!available) {
      pairCache?.close();
      pairCache = undefined;
      void releaseResources();
    }
  });
  if (!tracks.length || tracks.some((track) => !track.trackId) || new Set(tracks.map((track) => track.trackId)).size !== tracks.length) return Promise.reject(new Error('ZipEnhancer requires distinct original source tracks.'));
  if (idleTimer !== null) { globalThis.clearTimeout(idleTimer); idleTimer = null; }
  activeRequests++;
  // Offscreen already admits ASR/enhancement tasks; this also covers direct callers
  // and makes pair-cache selection/commit atomic with respect to other Zip tasks.
  const result = enhancementTail.then(() => enhanceAudioTracksExclusive(tracks, onProgress, options));
  enhancementTail = result.then(() => undefined, () => undefined);
  return result.finally(() => {
    activeRequests--;
    if (!activeRequests && resourcesPromise) idleTimer = globalThis.setTimeout(() => {
      idleTimer = null;
      void releaseResources();
    }, MODEL_IDLE_MS);
  });
}
