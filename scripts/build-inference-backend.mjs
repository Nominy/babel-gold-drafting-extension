import { build } from 'esbuild';
import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { ortJsepBuildOptions } from './ort-jsep-options.mjs';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const { version } = JSON.parse(await readFile(path.join(root, 'manifest.json'), 'utf8'));
const destination = path.join(root, '.artifacts/inference-backend/extension');
await mkdir(destination, { recursive: true });
await build({ ...ortJsepBuildOptions, entryPoints: [path.join(root, 'src/backend/inference-page.ts')], outfile: path.join(destination, 'inference.js'),
  bundle: true, platform: 'browser', format: 'esm', target: 'chrome114', conditions: ['onnxruntime-web-use-extern-wasm'],
  define: { ...ortJsepBuildOptions.define, __BABEL_DEV_C_DENOISE_MODEL_URL__: JSON.stringify(process.env.BABEL_DEV_C_DENOISE_MODEL_URL || '') }
});
await writeFile(path.join(destination, 'manifest.json'), JSON.stringify({ manifest_version: 3, name: 'Babel trusted C-denoise runtime', version,
  permissions: ['storage', 'unlimitedStorage'], host_permissions: ['https://reviewgen.ovh/*', 'http://127.0.0.1/*', 'http://localhost/*'],
  background: { service_worker: 'background.js' },
  content_security_policy: { extension_pages: "script-src 'self' 'wasm-unsafe-eval'; object-src 'self'; connect-src 'self' https://reviewgen.ovh http://127.0.0.1:* http://localhost:*" }
}, null, 2));
await writeFile(path.join(destination, 'background.js'), 'chrome.runtime.onInstalled.addListener(() => {});');
await writeFile(path.join(destination, 'inference.html'), '<!doctype html><meta charset="utf-8"><title>Babel inference</title><script type="module" src="inference.js"></script>');
await mkdir(path.join(destination, 'dist/vendor/ort'), { recursive: true });
for (const name of ['ort-wasm-simd-threaded.jsep.wasm', 'ort-wasm-simd-threaded.jsep.mjs']) {
  await copyFile(fileURLToPath(import.meta.resolve(`onnxruntime-web/${name}`)), path.join(destination, 'dist/vendor/ort', name));
}
await copyFile(path.join(root, 'model-release.json'), path.join(root, '.artifacts/inference-backend/model-release.json'));
await copyFile(path.join(root, 'scripts/inference-backend-worker.mjs'), path.join(root, '.artifacts/inference-backend/worker.mjs'));
console.log(`Trusted backend runtime: ${destination}`);
