#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { readFile, realpath, stat } from 'node:fs/promises';
import { createServer } from 'node:http';
import path from 'node:path';

const args = process.argv.slice(2);
const options = {};
for (let index = 0; index < args.length; index += 2) {
  const key = args[index];
  if (!['--bundle', '--port', '--sample-dir'].includes(key) || !args[index + 1] || options[key]) {
    throw new Error('Usage: serve-dev-cdenoise.mjs --bundle DIRECTORY [--port 8798] [--sample-dir AUTHORIZED_SAMPLE_DIRECTORY]');
  }
  options[key] = args[index + 1];
}
if (!options['--bundle']) throw new Error('--bundle is required');
const port = Number(options['--port'] ?? 8798);
if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('--port must be 1..65535');
const bundleRoot = await realpath(path.resolve(options['--bundle']));
const sampleRoot = await realpath(path.resolve(options['--sample-dir'] ?? bundleRoot));
const allowed = new Map();
const required = [
  'asr/v3_ctc.onnx', 'asr/v3_ctc.yaml', 'punctuation/context.fp16.onnx',
  'punctuation/denoise.fp16.onnx', 'punctuation/c-denoise.json', 'punctuation/config.json',
  'punctuation/gpu-placement.json',
  'punctuation/tokenizer.json', 'punctuation/tokenizer_config.json',
  'punctuation/special_tokens_map.json', 'punctuation/vocab.txt'
];
const manifestBytes = await readFile(path.join(bundleRoot, 'manifest.json'));
const manifest = JSON.parse(manifestBytes.toString('utf8'));
if (manifest.schema !== 'babel-browser-model-bundle-v3' || manifest.pass !== true ||
    manifest.validation?.numericExport?.pass !== true || manifest.targetBytes !== 1_500_000_000 ||
    !Array.isArray(manifest.files) || !Number.isSafeInteger(manifest.totalBytes) ||
    manifest.totalBytes <= 0 || manifest.totalBytes > manifest.targetBytes) {
  throw new Error('Bundle must contain the validated C-denoise v3 manifest within its byte target');
}
if (manifest.source?.asrCheckpointSha256 !== '02cea9973d0e839f6a3eeca101b83a93f93a066c2da2e3ebfa176d57e61d84d3' ||
    manifest.source?.cDenoiseCheckpointSha256 !== '0a01c3535fb66627b13f266bc59ab7b95c2aa85f7413a051d1e17294515ded5a') {
  throw new Error('Bundle checkpoint provenance is not the accepted speech and C-denoise pair');
}
const numeric = manifest.validation.numericExport;
if (!/^[a-f0-9]{64}$/.test(numeric.reportSha256 ?? '') ||
    !numeric.limits || typeof numeric.limits !== 'object' || Array.isArray(numeric.limits) || !Object.keys(numeric.limits).length ||
    !Array.isArray(numeric.cases) || !numeric.cases.length || !Array.isArray(numeric.asr) || !numeric.asr.length) {
  throw new Error('Bundle is missing numeric export parity evidence');
}
for (const row of [...numeric.cases, ...numeric.asr]) {
  if (row?.pass !== true || typeof row.id !== 'string' || !row.id ||
      !row.comparisons || typeof row.comparisons !== 'object' || Array.isArray(row.comparisons) ||
      !Object.keys(row.comparisons).length) throw new Error('Bundle has failed or empty numeric export cases');
  for (const metric of Object.values(row.comparisons)) {
    if (metric?.pass !== true || metric.finite !== true || !Number.isFinite(metric.maxAbs) ||
        metric.maxAbs < 0 || !Number.isFinite(metric.rmse) || metric.rmse < 0) throw new Error('Bundle has failed or non-finite numeric export metrics');
  }
}
if (!manifest.source.baseModelFiles || !Object.keys(manifest.source.baseModelFiles).length ||
    Object.values(manifest.source.baseModelFiles).some((sha) => typeof sha !== 'string' || !/^[a-f0-9]{64}$/.test(sha))) {
  throw new Error('Bundle is missing base-model SHA-256 provenance');
}
async function addFile(root, relative, expected) {
  if (typeof relative !== 'string' || !relative || /[\\%?#\x00-\x1f\x7f:]/.test(relative) ||
      relative.split('/').some((part) => !part || part === '.' || part === '..')) {
    throw new Error(`Unsafe supplier file path: ${relative}`);
  }
  if (allowed.has(relative)) throw new Error(`Duplicate supplier path: ${relative}`);
  const filename = await realpath(path.join(root, relative));
  const inside = path.relative(root, filename);
  if (!inside || inside.startsWith(`..${path.sep}`) || path.isAbsolute(inside)) throw new Error(`Supplier path escapes source directory: ${relative}`);
  const info = await stat(filename);
  if (!info.isFile() || !Number.isSafeInteger(info.size) || info.size <= 0) throw new Error(`Supplier file is not a non-empty regular file: ${relative}`);
  const digest = createHash('sha256');
  for await (const chunk of createReadStream(filename)) digest.update(chunk);
  const sha256 = digest.digest('hex');
  if (expected && (expected.bytes !== info.size || expected.sha256 !== sha256)) throw new Error(`Supplier byte size or SHA-256 mismatch: ${relative}`);
  allowed.set(relative, { filename, bytes: info.size, etag: `"${sha256}"` });
}
let totalBytes = 0;
for (const file of manifest.files) {
  if (!required.includes(file.path) && !/^(asr|punctuation)\/[^/]+\.onnx\.data(?:\.\d+)?$/.test(file.path)) throw new Error(`Unsupported model file: ${file.path}`);
  await addFile(bundleRoot, file.path, file);
  totalBytes += file.bytes;
}
if (totalBytes !== manifest.totalBytes || required.some((name) => !allowed.has(name))) throw new Error('Manifest does not describe the complete exact C-denoise bundle');
await addFile(bundleRoot, 'manifest.json');
await addFile(sampleRoot, 'sample-russian-15s.wav');
await addFile(sampleRoot, 'sample-russian-15s-license.txt');
const sampleBytes = await readFile(allowed.get('sample-russian-15s.wav').filename);
if (sampleBytes.toString('ascii', 0, 4) !== 'RIFF' || sampleBytes.toString('ascii', 8, 12) !== 'WAVE') throw new Error('Authorized sample must be a PCM WAV');
let byteRate = 0;
let dataBytes = 0;
for (let cursor = 12; cursor + 8 <= sampleBytes.length;) {
  const id = sampleBytes.toString('ascii', cursor, cursor + 4);
  const bytes = sampleBytes.readUInt32LE(cursor + 4);
  if (cursor + 8 + bytes > sampleBytes.length) throw new Error('Truncated sample WAV');
  if (id === 'fmt ') {
    if (bytes < 16 || sampleBytes.readUInt16LE(cursor + 8) !== 1 || sampleBytes.readUInt16LE(cursor + 10) !== 1 || sampleBytes.readUInt32LE(cursor + 12) !== 16000 || sampleBytes.readUInt16LE(cursor + 22) !== 16) throw new Error('Authorized sample must be mono 16 kHz PCM16');
    byteRate = sampleBytes.readUInt32LE(cursor + 16);
  }
  if (id === 'data') dataBytes += bytes;
  cursor += 8 + bytes + (bytes % 2);
}
if (byteRate !== 32000 || dataBytes <= 0 || dataBytes / byteRate > 15) throw new Error('Authorized sample must contain no more than 15 seconds of audio');
const license = await readFile(allowed.get('sample-russian-15s-license.txt').filename, 'utf8');
if (!/Public Domain/.test(license) || !license.includes('https://archive.org/details/aesops_fables_russian_0905_librivox')) throw new Error('Sample requires the existing LibriVox public-domain source/license declaration');
const types = { '.json': 'application/json', '.txt': 'text/plain; charset=utf-8', '.yaml': 'text/plain; charset=utf-8', '.wav': 'audio/wav' };
const server = createServer(async (request, response) => {
  response.setHeader('Access-Control-Allow-Origin', '*');
  response.setHeader('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS');
  response.setHeader('Access-Control-Allow-Headers', 'Range');
  response.setHeader('Access-Control-Expose-Headers', 'Content-Length, Content-Range, Accept-Ranges, ETag');
  if (request.method === 'OPTIONS') { response.writeHead(204); response.end(); return; }
  if (!['GET', 'HEAD'].includes(request.method)) { response.setHeader('Allow', 'GET, HEAD, OPTIONS'); response.writeHead(405); response.end('Static model supplier only'); return; }
  let pathname;
  try {
    const rawPath = (request.url ?? '').split('?')[0];
    pathname = decodeURIComponent(rawPath);
    if (pathname.includes('%') || /[\\\x00-\x1f\x7f]/.test(pathname) || pathname.split('/').some((part) => part === '.' || part === '..')) throw new Error('Unsafe path');
  } catch { response.writeHead(400); response.end('Invalid supplier path'); return; }
  const relative = pathname.startsWith('/c-denoise/') ? pathname.slice('/c-denoise/'.length) : '';
  let file = allowed.get(relative);
  if (!file) { response.writeHead(404); response.end('No model file at this path'); return; }
  let currentManifest;
  if (relative === 'manifest.json') {
    try {
      // Parent may append actual-WebGPU proof after startup; serve the real current bytes.
      currentManifest = await readFile(file.filename);
      file = { ...file, bytes: currentManifest.length,
        etag: `"${createHash('sha256').update(currentManifest).digest('hex')}"` };
    } catch { response.writeHead(503); response.end('Manifest is unavailable'); return; }
  }
  response.setHeader('Accept-Ranges', 'bytes');
  response.setHeader('ETag', file.etag);
  response.setHeader('Cache-Control', 'no-store');
  response.setHeader('Content-Type', types[path.extname(relative)] ?? 'application/octet-stream');
  let start = 0;
  let end = file.bytes - 1;
  let status = 200;
  if (request.headers.range) {
    const match = /^bytes=(\d*)-(\d*)$/.exec(request.headers.range);
    if (!match || (!match[1] && !match[2])) { response.setHeader('Content-Range', `bytes */${file.bytes}`); response.writeHead(416); response.end(); return; }
    if (!match[1]) {
      const suffix = Number(match[2]);
      start = Math.max(0, file.bytes - suffix);
      if (!Number.isSafeInteger(suffix) || suffix <= 0) start = file.bytes;
    } else {
      start = Number(match[1]);
      end = match[2] ? Math.min(Number(match[2]), end) : end;
    }
    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || start >= file.bytes || end < start) { response.setHeader('Content-Range', `bytes */${file.bytes}`); response.writeHead(416); response.end(); return; }
    status = 206;
    response.setHeader('Content-Range', `bytes ${start}-${end}/${file.bytes}`);
  }
  response.setHeader('Content-Length', end - start + 1);
  response.writeHead(status);
  if (request.method === 'HEAD') { response.end(); return; }
  if (currentManifest) { response.end(currentManifest.subarray(start, end + 1)); return; }
  const stream = createReadStream(file.filename, { start, end });
  stream.on('error', () => response.destroy());
  response.on('close', () => stream.destroy());
  stream.pipe(response);
});
server.on('error', (error) => { console.error(error.message); process.exitCode = 1; });
server.listen(port, '127.0.0.1', () => console.log(`C-denoise static supplier http://127.0.0.1:${port}/c-denoise — ${totalBytes} verified model bytes; no inference/upload API`));
