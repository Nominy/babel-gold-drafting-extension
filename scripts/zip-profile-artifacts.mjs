import path from 'node:path';
import { createWriteStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import { Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';

export const PROFILE_UPLOAD_CHUNK_BYTES = 4 * 1024 * 1024;
export const profileArtifactPattern = /^(?:config-\d+-(?:input-\d+|uniform)\.bin|spectrum-\d+-(?:mag|pha)\.bin|capture-manifest-\d+\.json|replay-\d+\.json|run-\d+(?:-speaker-\d+\.wav|\.json)|(?:configurations|occurrences|report|coverage-failure|summary)\.json|enhanced-\d+\.wav)$/;

// Requests are bounded independently of the artifact size. Keep state until the
// private server closes so completed files cannot be overwritten or appended.
export function createProfileArtifactStore(directory) {
  const states = new Map();
  return {
    async writeChunk(name, request, offset, total) {
      if (!profileArtifactPattern.test(name) || !Number.isSafeInteger(offset) || !Number.isSafeInteger(total) ||
          offset < 0 || total < 0 || offset > total) throw new Error('Invalid artifact upload name or range');
      let state = states.get(name);
      if (!state) {
        if (offset !== 0) throw new Error(`Artifact ${name} must begin at offset zero`);
        state = { total, offset: 0, writing: false, complete: false, failed: false, writeError: null };
        states.set(name, state);
      }
      if (state.total !== total || state.offset !== offset || state.writing || state.complete || state.failed) {
        throw new Error(`Artifact ${name} has an invalid, overlapping, or completed upload range: ${JSON.stringify({
          existingOffset: state.offset, existingTotal: state.total, newOffset: offset, newTotal: total,
          writing: state.writing, complete: state.complete, failed: state.failed, originatingWriteError: state.writeError,
        })}`);
      }
      state.writing = true;
      const expected = Math.min(PROFILE_UPLOAD_CHUNK_BYTES, total - offset);
      let received = 0;
      const counter = new Transform({ transform(chunk, encoding, callback) {
        received += chunk.byteLength;
        if (received > expected) callback(new Error(`Artifact ${name} upload chunk exceeds its exact range`));
        else callback(null, chunk);
      } });
      const filename = path.join(directory, name);
      try {
        await pipeline(request, counter, createWriteStream(filename, { flags: offset === 0 ? 'wx' : 'r+', start: offset, mode: 0o600 }));
        if (received !== expected) throw new Error(`Artifact ${name} upload is incomplete: expected ${expected}, received ${received}`);
        state.offset += received;
        if (state.offset === total) {
          if ((await stat(filename)).size !== total) throw new Error(`Artifact ${name} assembled byte length is incorrect`);
          state.complete = true;
        }
        return { nextOffset: state.offset, total, complete: state.complete };
      } catch (error) {
        state.failed = true;
        state.writeError = {
          code: error && typeof error === 'object' && typeof error.code === 'string' ? error.code : null,
          message: error instanceof Error ? error.message : String(error),
        };
        throw error;
      } finally { state.writing = false; }
    },
  };
}
