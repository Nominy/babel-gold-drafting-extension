import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { mkdir, readFile, writeFile, stat } from 'node:fs/promises';
import { pipeline } from 'node:stream/promises';
import { build } from 'esbuild';
import { chromium } from 'playwright';
import { createOrtJsepBuildOptions } from './ort-jsep-options.mjs';
import { createProfileArtifactStore } from './zip-profile-artifacts.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const workspace = path.resolve(root, '../..');
const audio = [], references = [];
let plan, out, executable, capture = false, runs = 3;
for (let i = 2; i < process.argv.length; i++) {
  const arg = process.argv[i];
  if (arg === '--capture') { capture = true; continue; }
  const value = process.argv[++i];
  if (!value) throw new Error(`Missing value for ${arg}`);
  if (arg === '--plan') plan = path.resolve(value);
  else if (arg === '--out') out = path.resolve(value);
  else if (arg === '--audio') audio.push(path.resolve(value));
  else if (arg === '--reference') references.push(path.resolve(value));
  else if (arg === '--executable') executable = path.resolve(value);
  else if (arg === '--runs') { runs = Number(value); if (!Number.isSafeInteger(runs) || runs < 1) throw new Error('Invalid run count'); }
  else throw new Error(`Unknown option ${arg}`);
}
if (!out || !plan) throw new Error('Supply --out NEW_PRIVATE_DIRECTORY --plan PATH [--runs 3] [--capture]');
if (!audio.length) for (const i of [1, 2]) audio.push(path.join(workspace, `audio/.private/noisy-live-20261008/speaker-${i}.wav`));
if (!references.length) for (const i of [1, 2]) references.push(path.resolve(workspace, `../babel_experiment/artifacts/maxine-afx-live-20261008/network-performance/static-only-warm-speaker-${i}.wav`));
if (references.length !== audio.length) throw new Error('Every source needs an accepted browser WAV reference');
const parsedPlan = JSON.parse(await readFile(plan, 'utf8'));
if (parsedPlan.weights.file !== 'zipenhancer-webgpu.weights.bin') throw new Error('Unexpected weights filename');
const weights = path.join(path.dirname(plan), parsedPlan.weights.file);
for (const name of [plan, weights, ...audio, ...references]) {
  if (!(await stat(name)).isFile()) throw new Error(`Not a regular file: ${name}`);
}
await mkdir(path.dirname(out), { recursive: true }); await mkdir(out, { recursive: false });
const artifactStore = createProfileArtifactStore(out);
const common = { absWorkingDir: root, bundle: true, platform: 'browser', format: 'iife', target: 'chrome114', write: false, logLevel: 'warning' };
const [main, worker] = await Promise.all([
  build({ ...common, ...createOrtJsepBuildOptions(), entryPoints: [path.join(root, 'scripts/zip-webgpu-benchmark-browser.ts')] }),
  build({ ...common, entryPoints: [path.join(root, 'src/workers/audio-enhancement.ts')] }),
]);
await writeFile(path.join(out, 'benchmark-bundle.js'), main.outputFiles[0].contents);
const server = createServer(async (request, response) => {
  response.setHeader('Content-Security-Policy', "default-src 'none'; script-src 'self'; connect-src 'self'; worker-src 'self'; base-uri 'none'; frame-src 'none'");
  try {
    const url = new URL(request.url, 'http://127.0.0.1'), route = url.pathname;
    if (route === '/' && request.method === 'GET') { response.setHeader('Content-Type', 'text/html'); response.end('<!doctype html><title>Private ZipEnhancer WebGPU benchmark</title><body>Preparing browser inference…<script src="/benchmark.js"></script>'); return; }
    const bundle = route === '/benchmark.js' ? main : route === '/dsp-worker.js' ? worker : undefined;
    if (bundle && request.method === 'GET') {
      response.setHeader('Content-Type', 'text/javascript'); response.end(bundle.outputFiles[0].contents); return;
    }
    if (route.startsWith('/result/') && request.method === 'POST') {
      const name = route.slice('/result/'.length);
      if (!/^(?:run-\d+(?:-speaker-\d+\.wav|\.json)|summary\.json)$/.test(name)) { response.writeHead(400).end(); return; }
      const offset = url.searchParams.get('offset'), total = url.searchParams.get('total');
      if (offset === null || total === null || !/^\d+$/.test(offset) || !/^\d+$/.test(total)) { response.writeHead(400).end('Missing artifact range'); return; }
      const result = await artifactStore.writeChunk(name, request, Number(offset), Number(total));
      response.setHeader('Content-Type', 'application/json'); response.end(JSON.stringify(result)); return;
    }
    let file;
    if (route === '/plan') file = plan;
    else if (route === '/weights') file = weights;
    else if (/^\/audio\/\d+$/.test(route)) file = audio[Number(route.split('/')[2])];
    else if (/^\/reference\/\d+$/.test(route)) file = references[Number(route.split('/')[2])];
    if (!file || request.method !== 'GET') { response.writeHead(404).end(); return; }
    response.setHeader('Content-Type', file.endsWith('.json') ? 'application/json' : 'application/octet-stream');
    response.setHeader('Content-Length', (await stat(file)).size); await pipeline(createReadStream(file), response);
  } catch (error) { if (!response.headersSent) response.writeHead(500); response.end(String(error)); }
});
let browser;
try {
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
  browser = await chromium.launch({ ...(executable ? { executablePath: executable } : { channel: 'chrome' }), headless: true });
  const page = await browser.newPage();
  const logs = [];
  page.on('console', value => {
    const text = value.text();
    logs.push({ type: value.type(), text });
  });
  page.on('pageerror', error => logs.push({ type: 'pageerror', text: String(error) }));
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  await page.waitForFunction(() => typeof globalThis.runZipWebBenchmark === 'function');
  await writeFile(path.join(out, 'provenance.json'), JSON.stringify({
    backend: 'webgpu', plan, weights, unsafeGpuFlags: false, audio, references, capture, runs, browserVersion: browser.version(),
  }, null, 2));
  try { console.log(JSON.stringify(await page.evaluate(options => globalThis.runZipWebBenchmark(options), {
    capture, runs, audioCount: audio.length,
  }), null, 2)); }
  finally { await writeFile(path.join(out, 'browser-log.json'), JSON.stringify(logs, null, 2)); }
} finally {
  if (browser) await browser.close();
  await new Promise(resolve => server.close(resolve));
}
