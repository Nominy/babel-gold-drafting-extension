import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { PassThrough, Readable } from 'node:stream';
import test from 'node:test';
import { createProfileArtifactStore, PROFILE_UPLOAD_CHUNK_BYTES } from '../scripts/zip-profile-artifacts.mjs';

test('bounded artifact ranges assemble every binary byte and UTF8 JSON across chunk boundaries', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'zip-profile-upload-'));
  try {
    const store = createProfileArtifactStore(directory);
    const binary = Buffer.alloc(PROFILE_UPLOAD_CHUNK_BYTES * 2 + 137);
    for (let index = 0; index < binary.length; index++) binary[index] = index % 251;
    const json = Buffer.from(JSON.stringify({ sample: '猫é'.repeat(PROFILE_UPLOAD_CHUNK_BYTES) }), 'utf8');
    for (const [name, bytes] of [['config-0-input-0.bin', binary], ['configurations.json', json]] as const) {
      for (let offset = 0; offset < bytes.length; offset += PROFILE_UPLOAD_CHUNK_BYTES) {
        const end = Math.min(bytes.length, offset + PROFILE_UPLOAD_CHUNK_BYTES);
        const ack = await store.writeChunk(name, Readable.from(bytes.subarray(offset, end)), offset, bytes.length);
        assert.deepEqual(ack, { nextOffset: end, total: bytes.length, complete: end === bytes.length });
      }
      assert.deepEqual(await readFile(path.join(directory, name)), bytes);
      await assert.rejects(store.writeChunk(name, Readable.from(bytes.subarray(0, PROFILE_UPLOAD_CHUNK_BYTES)), 0, bytes.length), /invalid.*completed/);
    }
    const empty = await store.writeChunk('config-1-input-0.bin', Readable.from([]), 0, 0);
    assert.deepEqual(empty, { nextOffset: 0, total: 0, complete: true });
    assert.equal((await readFile(path.join(directory, 'config-1-input-0.bin'))).length, 0);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('artifact ranges reject gaps, changed lengths, short/oversized chunks and unsafe filenames', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'zip-profile-upload-'));
  try {
    const store = createProfileArtifactStore(directory), chunk = Buffer.alloc(PROFILE_UPLOAD_CHUNK_BYTES, 23), total = chunk.length + 1;
    await assert.rejects(store.writeChunk('../report.json', Readable.from([]), 0, 0), /Invalid/);
    await assert.rejects(store.writeChunk('config-0-input-0.bin', Readable.from(chunk), 1, total), /offset zero/);
    await store.writeChunk('config-0-input-0.bin', Readable.from(chunk), 0, total);
    await assert.rejects(store.writeChunk('config-0-input-0.bin', Readable.from(Buffer.of(1)), chunk.length - 1, total), /invalid/);
    await assert.rejects(store.writeChunk('config-0-input-0.bin', Readable.from(Buffer.of(1)), chunk.length, total + 1), /invalid/);
    await store.writeChunk('config-0-input-0.bin', Readable.from(Buffer.of(99)), chunk.length, total);
    assert.equal((await readFile(path.join(directory, 'config-0-input-0.bin')))[total - 1], 99);
    await assert.rejects(store.writeChunk('config-1-input-0.bin', Readable.from(Buffer.of(1)), 0, 2), /incomplete/);
    await assert.rejects(store.writeChunk('config-1-input-0.bin', Readable.from(Buffer.of(2)), 1, 2), /invalid/);
    await assert.rejects(store.writeChunk('config-2-input-0.bin', Readable.from(Buffer.alloc(chunk.length + 1)), 0, chunk.length + 1), /exceeds/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('an in-flight artifact range cannot be overwritten by a concurrent request', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'zip-profile-upload-'));
  try {
    const store = createProfileArtifactStore(directory), stream = new PassThrough();
    const pending = store.writeChunk('config-0-input-0.bin', stream, 0, 1);
    await assert.rejects(store.writeChunk('config-0-input-0.bin', Readable.from(Buffer.of(2)), 0, 1), /overlapping/);
    stream.end(Buffer.of(1));
    assert.deepEqual(await pending, { nextOffset: 1, total: 1, complete: true });
    assert.deepEqual(await readFile(path.join(directory, 'config-0-input-0.bin')), Buffer.of(1));
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('originating stream/write failure code and message survive subsequent rejected requests', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'zip-profile-upload-'));
  try {
    const store = createProfileArtifactStore(directory), stream = new PassThrough();
    const failure = Object.assign(new Error('No space left on device'), { code: 'ENOSPC' });
    const pending = store.writeChunk('config-0-input-0.bin', stream, 0, 1);
    const rejected = assert.rejects(pending, error => error === failure);
    stream.destroy(failure); await rejected;
    await assert.rejects(store.writeChunk('config-0-input-0.bin', Readable.from(Buffer.of(1)), 0, 1),
      /originatingWriteError.*ENOSPC.*No space left on device/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
