import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { readFile } from 'node:fs/promises';
import { assertKernelSource, kernelSourceHashes, transformBinarySource, transformProgramManagerSource, transformBackendProfileSource } from './ort-kernel-transforms.mjs';

// ORT 1.29's default WebGPU bundle uses the native EP, which does not emit
// the public JS dispatch profiler used by the neural-placement acceptance gate.
// Build the shipped JSEP source against its matching shipped .jsep WASM instead.
const ortDist = path.dirname(fileURLToPath(import.meta.resolve('onnxruntime-web')));
// Profiling hooks are compiled only into the private external harness.
export function createOrtJsepBuildOptions({ profile = false, baseline = false } = {}) {
  return {
  plugins: [{ name: 'babel-webgpu-pinned-kernels', setup(build) {
    build.onLoad({ filter: /[\\/]onnxruntime-web[\\/]lib[\\/]wasm[\\/]jsep[\\/](?:webgpu[\\/]program-manager|webgpu[\\/]ops[\\/]binary-op|backend-webgpu)\.ts$/ }, async ({ path: filename }) => {
      const source = await readFile(filename, 'utf8');
      const root = path.resolve(ortDist, '../lib/wasm/jsep');
      const relative = path.relative(root, filename).replaceAll('\\', '/');
      assertKernelSource(relative, source);
      if (relative === 'webgpu/program-manager.ts') {
        await Promise.all(Object.keys(kernelSourceHashes).map(async (name) =>
          assertKernelSource(name, await readFile(path.resolve(root, name)))));
        return { loader: 'ts', contents: transformProgramManagerSource(source, { profile }) };
      }
      if (relative === 'webgpu/ops/binary-op.ts') return { loader: 'ts', contents: transformBinarySource(source, { profile, baseline }) };
      return { loader: 'ts', contents: profile ? transformBackendProfileSource(source) : source };
    });
  } }],
  alias: { 'onnxruntime-web/webgpu': path.resolve(ortDist, '../lib/index.ts') },
  // These guarded Node-only branches are never entered by extension pages.
  external: ['node:os', 'node:fs'],
  define: {
    'BUILD_DEFS.DISABLE_WEBGL': 'true',
    'BUILD_DEFS.DISABLE_JSEP': 'false',
    'BUILD_DEFS.DISABLE_WEBGPU': 'true',
    'BUILD_DEFS.DISABLE_WEBNN': 'true',
    'BUILD_DEFS.DISABLE_WASM': 'false',
    'BUILD_DEFS.DISABLE_WASM_PROXY': 'true',
    'BUILD_DEFS.ENABLE_JSPI': 'false',
    'BUILD_DEFS.ENABLE_BUNDLE_WASM_JS': 'false',
    'BUILD_DEFS.IS_ESM': 'false',
    'BUILD_DEFS.ESM_IMPORT_META_URL': 'undefined',
    'BUILD_DEFS.BUNDLE_FILENAME': '""'
  }
  };
}
export const ortJsepBuildOptions = createOrtJsepBuildOptions();
