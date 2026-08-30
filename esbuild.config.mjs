import { copyFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildExtension, defineExtensionBuild } from '@nominy/babel-extension-build';

const watch = process.argv.includes('--watch');
const rootDir = path.dirname(fileURLToPath(import.meta.url));
const ortRuntimeSourceDir = path.join(rootDir, 'node_modules/onnxruntime-web/dist');
const ortRuntimeOutputDir = path.join(rootDir, 'dist/vendor/ort');
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
  await mkdir(ortRuntimeOutputDir, { recursive: true });
  await Promise.all([
    copyFile(offscreenPageSourcePath, offscreenPageOutputPath),
    ...ortRuntimeAssetNames.map((assetName) =>
      copyFile(path.join(ortRuntimeSourceDir, assetName), path.join(ortRuntimeOutputDir, assetName))
    )
  ]);
}

const config = defineExtensionBuild({
  watch,
  prepare: prepareExtensionAssets,
  sharedOptions: {
    minify: false,
    sourcemap: true,
    target: 'chrome114',
    conditions: ['onnxruntime-web-use-extern-wasm'],
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
    }
  ],
  watchMessage: 'Watching gold drafting extension bundles...'
});

await buildExtension(config);
