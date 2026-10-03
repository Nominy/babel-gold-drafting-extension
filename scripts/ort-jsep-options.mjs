import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';

// ORT 1.29's default WebGPU bundle uses the native EP, which does not emit
// the public JS dispatch profiler used by the neural-placement acceptance gate.
// Build the shipped JSEP source against its matching shipped .jsep WASM instead.
const ortDist = path.dirname(fileURLToPath(import.meta.resolve('onnxruntime-web')));
export const ortJsepBuildOptions = {
  plugins: [{ name: 'babel-webgpu-fp32-accumulation', setup(build) {
    build.onLoad({ filter: /[\\/]onnxruntime-web[\\/]lib[\\/]wasm[\\/]jsep[\\/]webgpu[\\/]program-manager\.ts$/ }, async ({ path: filename }) => {
      const source = await readFile(filename, 'utf8');
      const pinned = {
        'program-manager.ts': '194b4b5d5afc55402dc840c8db4570945e0fc0651f39cb2d1955208898d5e331',
        'ops/3rd-party/matmul_packed_webgpu.ts': '78aa576c8cd162b38b30e76895453caca4870fbc2a404306ff625b2f869313f4',
        'ops/conv-grouped.ts': '23c29bcf97313acbf010e30c85109fe8cd7d8a62fc752b153532c34eb81ef288'
      };
      await Promise.all(Object.entries(pinned).map(async ([relative, expected]) => {
        const bytes = await readFile(path.resolve(path.dirname(filename), relative));
        if (createHash('sha256').update(bytes).digest('hex') !== expected) {
          throw new Error(`ORT 1.29 mixed precision shader source changed: ${relative}. Revalidate WebGPU kernels before building.`);
        }
      }));
      const original = 'const userCode = programInfo.getShaderSource(shaderHelper);';
      if (source.split(original).length !== 2) throw new Error('ORT JSEP shader compilation contract changed');
      const helper = fileURLToPath(new URL('../src/core/webgpu-fp32-accumulation.ts', import.meta.url));
      return { loader: 'ts', contents: `import { promoteFp16Accumulation } from ${JSON.stringify(helper)};\n` +
        source.replace(original, 'const userCode = promoteFp16Accumulation(programInfo.getShaderSource(shaderHelper));') };
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
