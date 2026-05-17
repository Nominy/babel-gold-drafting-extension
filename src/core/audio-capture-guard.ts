import type { CapturedAudioTrack, TranscriptJob } from './types';

export type AudioCaptureIssueKind = 'missing' | 'partial';

export interface AudioCaptureIssue {
  kind: AudioCaptureIssueKind;
  expectedSpeakerLanes: number;
  capturedSpeakerLanes: number;
  capturedTracks: number;
}

function normalizeLaneKey(value: string | undefined): string {
  return typeof value === 'string' ? value.trim().toLocaleLowerCase() : '';
}

function getExpectedSpeakerLaneCount(job: TranscriptJob): number {
  const speakers = new Set(job.rows.map((row) => normalizeLaneKey(row.speakerKey)).filter(Boolean));
  return speakers.size || 1;
}

function getCapturedSpeakerLaneCount(tracks: CapturedAudioTrack[]): number {
  const lanes = new Set<string>();
  for (const track of tracks) {
    const lane = normalizeLaneKey(track.speakerKey) || normalizeLaneKey(track.trackLabel);
    if (lane) {
      lanes.add(lane);
    }
  }
  return lanes.size;
}

export function assessAudioCaptureForDrafting(job: TranscriptJob, tracks: CapturedAudioTrack[]): AudioCaptureIssue | null {
  const expectedSpeakerLanes = getExpectedSpeakerLaneCount(job);
  const capturedSpeakerLanes = getCapturedSpeakerLaneCount(tracks);

  if (capturedSpeakerLanes === 0) {
    return {
      kind: 'missing',
      expectedSpeakerLanes,
      capturedSpeakerLanes,
      capturedTracks: tracks.length
    };
  }

  if (capturedSpeakerLanes < expectedSpeakerLanes) {
    return {
      kind: 'partial',
      expectedSpeakerLanes,
      capturedSpeakerLanes,
      capturedTracks: tracks.length
    };
  }

  return null;
}
