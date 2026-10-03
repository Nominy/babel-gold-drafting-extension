import { setupLocalModels, getCachedBundleDescriptor, markLocalModelWebGpuTested } from '../core/local-model-bundle';
import { INFERENCE_RELEASE, assertReleasedGraphs } from '../core/inference-release';
import { LOCAL_MODEL_BASE_URL, LOCAL_MODEL_SAMPLE_URL, DEFAULT_SETTINGS } from '../core/settings';
import { generateLocalL0Timing, generateLocalL0DraftFromTiming, transcribeLocalAudio } from '../core/local-model-runtime';
import type { L0TimingResponse, TranscriptJob, TranscriptRow } from '../core/types';

async function prepare(): Promise<void> {
  let descriptor = await getCachedBundleDescriptor(LOCAL_MODEL_BASE_URL);
  if (!descriptor) {
    await setupLocalModels(LOCAL_MODEL_BASE_URL);
    descriptor = await getCachedBundleDescriptor(LOCAL_MODEL_BASE_URL);
  }
  if (!descriptor) throw new Error('Backend model installation failed.');
  assertReleasedGraphs(descriptor.files);
  const response = await fetch(LOCAL_MODEL_SAMPLE_URL);
  if (!response.ok) throw new Error('Backend GPU preflight sample is unavailable.');
  await transcribeLocalAudio(await response.blob(), { allowUntestedBundle: true });
  await markLocalModelWebGpuTested(LOCAL_MODEL_BASE_URL, descriptor.identity);
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

Object.assign(globalThis, { babelInference: {
  release: INFERENCE_RELEASE.id,
  prepare,
  transcribe,
  draft: (timing: L0TimingResponse, rows?: TranscriptRow[]) => generateLocalL0DraftFromTiming(timing, rows)
} });
