import { createHash } from 'node:crypto';
import { readFile, readdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { kernelSourceHashes } from './ort-kernel-transforms.mjs';

const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
const checkpoint = 'b18896915e27a821585584221d0c0820f35e12145315ae3f1e73ccd5a68d195f';
const graph = '2f18c8f7ff10a2702d6243ce1230db9e73e6804dd6cd7b20d8e191ee06924016';
export const zipWebGpuAssets = ['zipenhancer-webgpu.plan.json', 'zipenhancer-webgpu.weights.bin', 'zipenhancer.LICENSE', 'zipenhancer.NOTICE'];

export async function prepareZipWebGpuArtifact(root) {
  const planBytes = await readFile(path.join(root, 'models', zipWebGpuAssets[0]));
  const plan = JSON.parse(planBytes);
  if (plan.schema !== 'babel-zip-webgpu-v1' || plan.checkpointSha256 !== checkpoint || plan.sourceGraphSha256 !== graph ||
      plan.kernelAbi !== 'zip-relative-attention-v1+swoosh-v1' || plan.weights?.file !== zipWebGpuAssets[1] ||
      !['float32', 'mixed-float16'].includes(plan.precision)) throw new Error('The browser ZipEnhancer plan is not bound to the verified model/ABI');
  const weights = await readFile(path.join(root, 'models', zipWebGpuAssets[1]));
  if (weights.byteLength !== plan.weights.byteLength || digest(weights) !== plan.weights.sha256) throw new Error('Browser ZipEnhancer weight digest/length mismatch');
  const core = path.join(root, 'src/core');
  const runtimeFiles = (await readdir(core)).filter(name => /^zipenhancer-webgpu.*\.(?:ts|mjs)$/.test(name)).map(name => `src/core/${name}`);
  if (!runtimeFiles.includes('src/core/zipenhancer-webgpu.ts') || !runtimeFiles.includes('src/core/zipenhancer-webgpu-attention.ts')) throw new Error('Browser ZipEnhancer runtime sources are incomplete');
  const sourceFiles = [...runtimeFiles, 'src/core/audio-enhancement-runtime.ts', 'src/core/audio-enhancement-backend.ts', 'src/core/audio-enhancement-dsp.ts',
    'src/core/audio-enhancement-swarm.ts', 'src/core/audio-enhancement-swarm-protocol.ts',
    'src/core/audio-enhancement-worker.ts', 'src/core/audio-enhancement-worker-client.ts', 'src/core/audio-enhancement-worker-protocol.ts',
    'src/core/audio-enhancement-pipeline.ts', 'src/core/c-denoise-acoustic.ts', 'scripts/ort-jsep-options.mjs', 'scripts/ort-kernel-transforms.mjs'].sort();
  const implementation = Object.fromEntries(await Promise.all(sourceFiles.map(async name => [name, digest(await readFile(path.join(root, name)))])));
  const planSha256 = digest(planBytes);
  const identity = { schema: 'babel-zip-webgpu-artifact-v1', planSha256, weightsSha256: plan.weights.sha256,
    kernelAbi: plan.kernelAbi, precision: plan.precision, implementation, vendorSources: kernelSourceHashes };
  const metadata = {
    id: 'zipenhancer-webgpu-2026-10-09-r1', sha256: digest(JSON.stringify(identity)),
    checkpointSha256: checkpoint, sourceGraphSha256: graph, planSha256,
    weightsSha256: plan.weights.sha256, weightsByteLength: weights.byteLength,
    planPath: `models/${zipWebGpuAssets[0]}`, weightsPath: `models/${zipWebGpuAssets[1]}`,
    precision: plan.precision, kernelAbi: plan.kernelAbi, sampleRate: 16000,
    fftSize: 400, hopSize: 100, winSize: 400, compressFactor: 0.3,
    chunkSeconds: 4, strideSeconds: 3, inputFrames: 641, implementation,
  };
  const destination = path.join(core, 'audio-enhancement-webgpu-model.json');
  const text = `${JSON.stringify(metadata, null, 2)}\n`;
  let previous;
  try { previous = await readFile(destination, 'utf8'); } catch (error) { if (error.code !== 'ENOENT') throw error; }
  if (previous !== text) await writeFile(destination, text);
  return metadata;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  console.log(JSON.stringify(await prepareZipWebGpuArtifact(path.resolve(import.meta.dirname, '..')), null, 2));
}
