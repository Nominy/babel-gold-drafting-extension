import { copyFile, mkdir, rm } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildExtension, defineExtensionBuild } from '@nominy/babel-extension-build';
import { ortJsepBuildOptions } from './scripts/ort-jsep-options.mjs';
import { prepareZipWebGpuArtifact, zipWebGpuAssets } from './scripts/zip-webgpu-artifact.mjs';

const watch = process.argv.includes('--watch');
const rootDir = path.dirname(fileURLToPath(import.meta.url));
const requestedDevModelUrl = process.env.BABEL_DEV_C_DENOISE_MODEL_URL?.trim();
let devModelUrl = '';
if (requestedDevModelUrl) {
  const url = new URL(requestedDevModelUrl);
  if (
    !['http:', 'https:'].includes(url.protocol) ||
    !['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname) ||
    url.username || url.password || url.search || url.hash ||
    url.pathname.replace(/\/+$/, '') !== '/c-denoise'
  ) {
    throw new Error('BABEL_DEV_C_DENOISE_MODEL_URL must be a loopback HTTP(S) /c-denoise URL');
  }
  devModelUrl = url.toString().replace(/\/+$/, '');
}
const ortRuntimeOutputDir = path.join(rootDir, 'dist/vendor/ort');
const zipModelOutputDir = path.join(rootDir, 'dist/models');
const offscreenPageSourcePath = path.join(rootDir, 'src/offscreen/offscreen.html');
const offscreenPageOutputPath = path.join(rootDir, 'offscreen.html');
const ortRuntimeAssetNames = [
  'ort-wasm-simd-threaded.mjs',
  'ort-wasm-simd-threaded.wasm',
  'ort-wasm-simd-threaded.asyncify.mjs',
  'ort-wasm-simd-threaded.asyncify.wasm',
  'ort-wasm-simd-threaded.jsep.mjs',
  'ort-wasm-simd-threaded.jsep.wasm',
  'ort-wasm-simd-threaded.jspi.mjs',
  'ort-wasm-simd-threaded.jspi.wasm'
];

async function prepareExtensionAssets() {
  await prepareZipWebGpuArtifact(rootDir);
  await mkdir(ortRuntimeOutputDir, { recursive: true });
  await mkdir(zipModelOutputDir, { recursive: true });
  await Promise.all([
    copyFile(offscreenPageSourcePath, offscreenPageOutputPath),
    ...zipWebGpuAssets.map((name) =>
      copyFile(path.join(rootDir, 'models', name), path.join(zipModelOutputDir, name))
    ),
    ...ortRuntimeAssetNames.map((assetName) =>
      copyFile(fileURLToPath(import.meta.resolve(`onnxruntime-web/${assetName}`)), path.join(ortRuntimeOutputDir, assetName))
    )
  ]);
  await Promise.all(['models/zipenhancer.onnx', 'models/zipenhancer-cpu.onnx', 'workers/zipenhancer-wasm.js', 'workers/zipenhancer-wasm.js.map']
    .map(name => rm(path.join(rootDir, 'dist', name), { force: true })));
}

const config = defineExtensionBuild({
  watch,
  prepare: prepareExtensionAssets,
  sharedOptions: {
    minify: false,
    sourcemap: true,
    target: 'chrome114',
    conditions: ['onnxruntime-web-use-extern-wasm'],
    alias: ortJsepBuildOptions.alias,
    plugins: ortJsepBuildOptions.plugins,
    external: ortJsepBuildOptions.external,
    define: { ...ortJsepBuildOptions.define, __BABEL_DEV_C_DENOISE_MODEL_URL__: JSON.stringify(devModelUrl) },
    format: 'iife',
    logLevel: 'info'
  },
  tasks: [
    {
      entryPoints: ['src/content/entry.ts'],
      outfile: 'dist/content/entry.js'
    },
    {
      entryPoints: ['src/content/audio-request-interceptor.ts'],
      outfile: 'dist/content/audio-request-interceptor.js'
    },
    {
      entryPoints: ['src/background/ai-broker.ts'],
      outfile: 'dist/background/ai-broker.js'
    },
    {
      entryPoints: ['src/options/options.ts'],
      outfile: 'dist/options/options.js'
    },
    {
      entryPoints: ['src/offscreen/local-model-host.ts'],
      outfile: 'dist/offscreen/local-model-host.js'
    },
    {
      entryPoints: ['src/workers/audio-enhancement.ts'],
      outfile: 'dist/workers/audio-enhancement.js'
    }
  ],
  watchMessage: 'Watching gold drafting extension bundles...'
});

await buildExtension(config);
