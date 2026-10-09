import type { Readable } from 'node:stream';
export const PROFILE_UPLOAD_CHUNK_BYTES: number;
export const profileArtifactPattern: RegExp;
export interface ProfileArtifactUploadAcknowledgement { nextOffset: number; total: number; complete: boolean }
export interface ProfileArtifactStore {
  writeChunk(name: string, request: Readable, offset: number, total: number): Promise<ProfileArtifactUploadAcknowledgement>;
}
export function createProfileArtifactStore(directory: string): ProfileArtifactStore;
