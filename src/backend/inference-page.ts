import { setupLocalModels, getCachedBundleDescriptor, markLocalModelWebGpuTested } from '../core/local-model-bundle';
import { INFERENCE_RELEASE, assertReleasedGraphs } from '../core/inference-release';
import { LOCAL_MODEL_BASE_URL, LOCAL_MODEL_SAMPLE_URL, DEFAULT_SETTINGS } from '../core/settings';
import { generateLocalL0Timing, generateLocalL0DraftFromTiming, transcribeLocalAudio } from '../core/local-model-runtime';
import type { L0TimingResponse, TranscriptJob, TranscriptRow } from '../core/types';
import { enhanceAudioTracks, getAudioEnhancementModel } from '../core/audio-enhancement-runtime';
import { parseEnhancementPayload, parseEnhancementTrackMetadata, parseEnhancementWorkerProgress, sameEnhancementModel, verifyEnhancementWav } from '../core/audio-enhancement-swarm-protocol';
import type { EnhancementWorkerProgress } from '../core/audio-enhancement-swarm-protocol';

type BackendEvent = { progress: EnhancementWorkerProgress } | { fieldName: string; offset: number; audio: string };
type BackendEmit = (event: BackendEvent) => Promise<void>;

async function prepare() {
  let descriptor = await getCachedBundleDescriptor(LOCAL_MODEL_BASE_URL);
  if (!descriptor) {
    await setupLocalModels(LOCAL_MODEL_BASE_URL);
    descriptor = await getCachedBundleDescriptor(LOCAL_MODEL_BASE_URL);
  }
  if (!descriptor) throw new Error('Backend model installation failed.');
  assertReleasedGraphs(descriptor.files);
  const response = await fetch(LOCAL_MODEL_SAMPLE_URL);
  if (!response.ok) throw new Error('Backend GPU preflight sample is unavailable.');
  const sample = await response.blob();
  await transcribeLocalAudio(sample, { allowUntestedBundle: true });
  await markLocalModelWebGpuTested(LOCAL_MODEL_BASE_URL, descriptor.identity);
  // Execute the packaged graph and DSP, not merely an adapter availability probe.
  const enhanced = await enhanceAudioTracks([0, 1].map(index => ({
    trackId: `preflight:${index}`, speakerKey: `preflight:${index}`, trackLabel: 'GPU preflight',
    source: 'trusted-backend', mimeType: 'audio/wav', blob: sample
  })), undefined, { localOnly: true, cache: false, taskId: 'backend-preflight' });
  if (enhanced.provider !== 'browser-local' || enhanced.tracks.length !== 2 ||
      enhanced.tracks.some(track => track.bytes.byteLength !== track.metadata.totalBytes)) {
    throw new Error('ZipEnhancer GPU preflight did not return a complete pair.');
  }
  return { ...getAudioEnhancementModel(), provider: 'webgpu' };
}

async function transcribe(payload: { taskId: string; tracks: Array<{ lane: string; fieldName: string }> }, audio: Record<string, string>): Promise<L0TimingResponse> {
  const job: TranscriptJob = {
    jobId: payload.taskId,
    rows: payload.tracks.map((track, index) => ({ rowId: `backend:${index}`, speakerKey: track.lane, index, text: '', startSeconds: null, endSeconds: null }))
  };
  const tracks = payload.tracks.map(track => ({
    trackId: track.fieldName, speakerKey: track.lane, trackLabel: track.lane, source: 'trusted-backend', mimeType: 'audio/wav',
    blob: new Blob([Uint8Array.from(atob(audio[track.fieldName]), char => char.charCodeAt(0))], { type: 'audio/wav' })
  }));
  return generateLocalL0Timing(DEFAULT_SETTINGS, job, tracks, undefined, payload.taskId);
}

async function enhance(value: unknown, audio: Record<string, string>, emit: BackendEmit) {
  const payload = parseEnhancementPayload(value);
  if (!sameEnhancementModel(payload.model, getAudioEnhancementModel())) throw new Error('Backend enhancement model identity mismatch.');
  const tracks = await Promise.all(payload.tracks.map(async track => {
    const bytes = Uint8Array.from(atob(audio[track.fieldName]), char => char.charCodeAt(0));
    await verifyEnhancementWav(bytes.buffer, track, track.sourceSha256);
    return { trackId: track.trackId, speakerKey: track.speakerKey, trackLabel: track.trackLabel,
      source: 'trusted-backend', mimeType: 'audio/wav', blob: new Blob([bytes], { type: 'audio/wav' }) };
  }));
  let previous: EnhancementWorkerProgress | undefined;
  const batch = await enhanceAudioTracks(tracks, async progress => {
    if (!['loading-model', 'enhancing', 'encoding'].includes(progress.phase)) return;
    const { phase, trackId, trackIndex, trackCount, completedChunks, totalChunks } = progress;
    previous = parseEnhancementWorkerProgress({ phase, trackId, trackIndex, trackCount, completedChunks, totalChunks }, payload, previous);
    await emit({ progress: previous });
  }, { localOnly: true, cache: false, taskId: payload.taskId });
  if (batch.provider !== 'browser-local' || batch.model !== payload.model.id ||
      batch.modelSha256 !== payload.model.sha256 || batch.tracks.length !== 2) throw new Error('Invalid backend enhancement result.');
  for (let index = 0; index < batch.tracks.length; index++) {
    const track = batch.tracks[index], source = payload.tracks[index];
    parseEnhancementTrackMetadata(track.metadata, source);
    await verifyEnhancementWav(track.bytes.buffer, source, track.metadata.wavSha256, true);
    for (let offset = 0; offset < track.bytes.byteLength; offset += 512 * 1024) {
      const chunk = track.bytes.subarray(offset, offset + 512 * 1024);
      let binary = '';
      for (let start = 0; start < chunk.length; start += 8192) binary += String.fromCharCode(...chunk.subarray(start, start + 8192));
      await emit({ fieldName: source.fieldName, offset, audio: btoa(binary) });
    }
  }
  return { model: batch.model, modelSha256: batch.modelSha256, tracks: batch.tracks.map(track => track.metadata) };
}

Object.assign(globalThis, { babelInference: {
  release: INFERENCE_RELEASE.id,
  prepare,
  transcribe,
  enhance,
  draft: (timing: L0TimingResponse, rows?: TranscriptRow[]) => generateLocalL0DraftFromTiming(timing, rows)
} });
