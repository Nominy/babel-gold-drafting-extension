import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'node:http';
import { createReadStream, constants } from 'node:fs';
import { mkdir, open, readFile, writeFile, unlink, stat, link, copyFile } from 'node:fs/promises';
import { pipeline } from 'node:stream/promises';
import { createHash } from 'node:crypto';
import { build } from 'esbuild';
import { chromium } from 'playwright';
import { createOrtJsepBuildOptions } from './ort-jsep-options.mjs';
import { kernelSourceHashes } from './ort-kernel-transforms.mjs';
import { createProfileArtifactStore, profileArtifactPattern } from './zip-profile-artifacts.mjs';
import { hashCampaignFile, verifyCaptureCampaign } from './zip-profile-campaign.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const args = process.argv.slice(2), audio = [];
const options = { repeats: 12, warmups: 4, captureBudgetMiB: 256, audioCount: 0, suiteOnly: false, recover: false };
let out, model = path.join(root, 'models/zipenhancer.onnx'), executable, baseline = false, captureDir, compareWaves, recoveryManifest, verifiedCampaign;
for (let index = 0; index < args.length; index++) {
  const arg = args[index];
  if (arg === '--baseline') { baseline = true; continue; }
  if (arg === '--help') {
    console.log('node scripts/profile-zip-kernels.mjs --audio lane1.wav --audio lane2.wav --out PRIVATE_NEW_DIR [--baseline] [--compare-waves BASELINE_DIR] [--model graph.onnx] [--executable chrome.exe] [--repeats 12] [--warmups 4] [--capture-budget-mib 256]\nnode scripts/profile-zip-kernels.mjs --suite-only PRIOR_CAPTURE_DIR --out PRIVATE_NEW_DIR [--repeats 24] [--warmups 6]\nnode scripts/profile-zip-kernels.mjs --recover-manifest PRIOR_DIR/capture-manifest-N.json --out PRIVATE_NEW_DIR [--capture-budget-mib 256]\nRecovery validates model/audio/helper hashes, every complete buffer/spectrum/WAV hash, every configuration identity and original GPU source-node proofs before using prior files. Complete original-lane inference need not be repeated; only pending real configurations are captured and all kernels replayed. Loose files without an identity manifest cannot be recovered safely.\nOutputs cover every configuration, shader variant and source-node occurrence. Immutable capture-manifest-N.json checkpoints contain full configurations/occurrences, original introducing spectra and missing-config details before readback/recovery. Complete finite and structural/index data are actual captured buffers, never synthetic substitutes. Loopback/CSP only. No top-N truncation. Dedicated fresh browser, never a live browser. Capture wall time is not production benchmark evidence.');
    process.exit(0);
  }
  const value = args[++index]; if (!value) throw new Error(`Missing value for ${arg}`);
  if (arg === '--audio') audio.push(path.resolve(value));
  else if (arg === '--out') out = path.resolve(value);
  else if (arg === '--model') model = path.resolve(value);
  else if (arg === '--executable') executable = path.resolve(value);
  else if (arg === '--suite-only') { options.suiteOnly = true; captureDir = path.resolve(value); }
  else if (arg === '--compare-waves') compareWaves = path.resolve(value);
  else if (arg === '--recover-manifest') { recoveryManifest = path.resolve(value); options.recover = true; }
  else if (arg === '--repeats' || arg === '--warmups' || arg === '--capture-budget-mib') {
    const number = Number(value);
    if (!Number.isSafeInteger(number) || number < (arg === '--warmups' ? 0 : 1) || (arg === '--repeats' && number > 512)) throw new Error(`Invalid ${arg}`);
    options[arg === '--capture-budget-mib' ? 'captureBudgetMiB' : arg.slice(2)] = number;
  } else throw new Error(`Unknown option ${arg}`);
}
if (recoveryManifest) {
  if (options.suiteOnly) throw new Error('--recover-manifest and --suite-only are distinct campaigns');
  const header = JSON.parse(await readFile(recoveryManifest, 'utf8'));
  if (header?.schema !== 'babel-zip-kernel-capture-v1' || !Array.isArray(header.sourcePaths) || header.sourcePaths.some(p => typeof p !== 'string') ||
      typeof header.identity?.baseline !== 'boolean') throw new Error('Recovery requires a recorded capture manifest, not loose buffer files');
  if (baseline && !header.identity.baseline) throw new Error('Recovery kernel mode differs from --baseline');
  baseline = header.identity.baseline;
  if (!audio.length) audio.push(...header.sourcePaths);
}
if (!out || (!options.suiteOnly && !audio.length) || (options.suiteOnly && (audio.length || baseline))) throw new Error('Supply --out and either --audio paths (optionally --baseline) or --suite-only prior capture directory');
if (compareWaves && options.suiteOnly) throw new Error('--compare-waves requires full WAV inference');
options.audioCount = audio.length;
for (const file of options.suiteOnly ? [path.join(captureDir, 'configurations.json')] : [model, ...audio]) if (!(await stat(file)).isFile()) throw new Error(`Not a regular file: ${file}`);
if (!options.suiteOnly) {
  const helperFiles = ['src/core/webgpu-binary-address.ts', 'src/core/webgpu-fp32-accumulation.ts', 'src/core/audio-enhancement-dsp.ts', 'src/core/c-denoise-acoustic.ts',
    'scripts/ort-kernel-transforms.mjs', 'scripts/ort-jsep-options.mjs'];
  const kernelHelperHashes = Object.fromEntries(await Promise.all(helperFiles.map(async name => [name, await hashCampaignFile(path.join(root, name))])));
  options.identity = { modelSha256: await hashCampaignFile(model), audioSha256: await Promise.all(audio.map(hashCampaignFile)),
    baseline, ortSourceHashes: kernelSourceHashes, kernelHelperHashes };
  options.sourcePaths = audio;
  if (recoveryManifest) {
    const metadata = JSON.parse(await readFile(path.join(root, 'src/core/audio-enhancement-model.json'), 'utf8'));
    verifiedCampaign = await verifyCaptureCampaign(recoveryManifest, options.identity, metadata.placementGraph);
  }
}
// Prevent concurrent instances of this external GPU tool without disturbing any other browser.
const lockPath = path.join(root, '.zip-gpu-profiler.lock');
const lock = await open(lockPath, 'wx');
let browser, server;
try {
  await lock.writeFile(JSON.stringify({ pid: process.pid, out, started: new Date().toISOString() }));
  await mkdir(path.dirname(out), { recursive: true }); await mkdir(out, { recursive: false, mode: 0o700 });
  if (verifiedCampaign) {
    const files = new Set(verifiedCampaign.spectra.flatMap(s => [s.mag.file, s.pha.file]));
    for (const config of verifiedCampaign.configurations.filter(c => c.captureComplete)) {
      for (const input of config.inputs) files.add(input.file);
      if (config.uniformBytes) files.add(config.uniformFile);
    }
    if (verifiedCampaign.modelInputsComplete) for (const wave of verifiedCampaign.waves) files.add(`enhanced-${wave.index}.wav`);
    for (const name of files) {
      const previous = path.join(path.dirname(recoveryManifest), name), destination = path.join(out, name);
      try { await link(previous, destination); }
      catch (error) { if (error.code !== 'EXDEV') throw error; await copyFile(previous, destination, constants.COPYFILE_EXCL); }
    }
  }
  const bundle = await build({ ...createOrtJsepBuildOptions({ profile: true, baseline }), entryPoints: [path.join(root, 'scripts/zip-kernel-profiler-browser.ts')],
    absWorkingDir: root, bundle: true, format: 'esm', platform: 'browser', target: 'chrome120', write: false, logLevel: 'warning' });
  const source = bundle.outputFiles[0].contents;
  await writeFile(path.join(out, 'profiler-bundle.js'), source);
  const artifactStore = createProfileArtifactStore(out);
  const ortDist = path.dirname(fileURLToPath(import.meta.resolve('onnxruntime-web')));
  server = createServer(async (request, response) => {
    response.setHeader('Cross-Origin-Opener-Policy', 'same-origin'); response.setHeader('Cross-Origin-Embedder-Policy', 'require-corp'); response.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
    // Enforce the loopback origin without CDP Fetch interception: paused POST
    // events serialize large binary captures into the browser-control transport.
    // ORT's shipped Emscripten emval method callers use new Function locally.
    response.setHeader('Content-Security-Policy', "default-src 'none'; script-src 'self' 'unsafe-eval'; connect-src 'self'; worker-src 'self' blob:; base-uri 'none'; frame-src 'none'; form-action 'none'");
    try {
      const url = new URL(request.url, 'http://127.0.0.1'), route = url.pathname;
      if (route === '/' && request.method === 'GET') { response.setHeader('Content-Type', 'text/html'); response.end('<!doctype html><meta charset="utf-8"><title>Private GPU profiler</title><script type="module" src="/profiler.js"></script>'); return; }
      if (route === '/profiler.js' && request.method === 'GET') { response.setHeader('Content-Type', 'text/javascript'); response.end(source); return; }
      let file;
      if (route.startsWith('/artifact/')) {
        const name = route.slice('/artifact/'.length); if (!profileArtifactPattern.test(name)) { response.writeHead(400).end(); return; }
        if (request.method === 'POST') {
          const offsetText = url.searchParams.get('offset'), totalText = url.searchParams.get('total');
          if (offsetText === null || totalText === null || !/^\d+$/.test(offsetText) || !/^\d+$/.test(totalText)) {
            response.writeHead(400).end('Artifact offset and total are required'); return;
          }
          const acknowledgement = await artifactStore.writeChunk(name, request, Number(offsetText), Number(totalText));
          response.setHeader('Content-Type', 'application/json'); response.end(JSON.stringify(acknowledgement)); return;
        }
        file = path.join(options.suiteOnly ? captureDir : out, name);
      } else if (route === '/capture-configurations' && options.suiteOnly) file = path.join(captureDir, 'configurations.json');
      else if (route === '/recovery-manifest' && verifiedCampaign) file = recoveryManifest;
      else if (route === '/model' && !options.suiteOnly) file = model;
      else if (/^\/audio\/\d+$/.test(route)) file = audio[Number(route.split('/')[2])];
      else if (/^\/ort\/ort-wasm-simd-threaded\.jsep\.(?:mjs|wasm)$/.test(route)) file = path.join(ortDist, route.slice('/ort/'.length));
      if (!file || request.method !== 'GET') { response.writeHead(404).end(); return; }
      const metadata = await stat(file); response.setHeader('Content-Length', metadata.size);
      response.setHeader('Content-Type', file.endsWith('.mjs') ? 'text/javascript' : file.endsWith('.wasm') ? 'application/wasm' : 'application/octet-stream');
      await pipeline(createReadStream(file), response);
    } catch (error) { if (!response.headersSent) response.writeHead(500); response.end(String(error)); }
  });
  const { promise: listening, resolve: ready, reject: failed } = Promise.withResolvers(); server.once('error', failed); server.listen(0, '127.0.0.1', ready); await listening;
  const origin = `http://127.0.0.1:${server.address().port}`;
  browser = await chromium.launch({ ...(executable ? { executablePath: executable } : { channel: 'chrome' }), headless: true,
    args: ['--enable-unsafe-webgpu', '--enable-features=WebGPUDeveloperFeatures'] });
  const context = await browser.newContext();
  const page = await context.newPage(); const logs = [];
  page.on('console', message => logs.push({ type: message.type(), text: message.text() })); page.on('pageerror', error => logs.push({ type: 'pageerror', text: String(error) }));
  await page.goto(origin); await page.waitForFunction(() => typeof globalThis.runZipKernelProfiler === 'function');
  await writeFile(path.join(out, 'provenance.json'), JSON.stringify({ baseline, suiteOnly: options.suiteOnly, captureDir, recoveryManifest, audio, model, options,
    kernelSourceHashes, bundleSha256: createHash('sha256').update(source).digest('hex'), browserVersion: browser.version(), started: new Date().toISOString() }, null, 2));
  try {
    const result = await page.evaluate(options => globalThis.runZipKernelProfiler(options), options);
    if (compareWaves) {
      const comparisons = [];
      for (const wave of result.waves) {
        const baselineFile = path.join(compareWaves, `enhanced-${wave.index}.wav`);
        const bytes = await readFile(baselineFile);
        const baselineSha256 = createHash('sha256').update(bytes).digest('hex');
        comparisons.push({ index: wave.index, baselineFile, baselineSha256, optimizedSha256: wave.wavSha256, exact: baselineSha256 === wave.wavSha256 });
      }
      await writeFile(path.join(out, 'wave-comparison.json'), JSON.stringify(comparisons, null, 2));
      if (comparisons.some(comparison => !comparison.exact)) throw new Error('Full-wave SHA256 mismatch against actual baseline WAVs; artifacts retained');
    }
    console.log(JSON.stringify({ out, ...result }, null, 2));
  } finally { await writeFile(path.join(out, 'browser-log.json'), JSON.stringify(logs, null, 2)); }
} finally {
  if (browser) await browser.close();
  if (server) { const { promise, resolve } = Promise.withResolvers(); server.close(resolve); await promise; }
  await lock.close(); await unlink(lockPath);
}
