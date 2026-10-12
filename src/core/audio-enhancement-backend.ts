import type { ExtensionSettings } from './types';

export type ZipEnhancementBackend =
  | { backend: 'webgpu'; adapter: GPUAdapter }
  | { backend: 'swarm'; reason: string };

interface BackendEnvironment {
  gpu?: Pick<GPU, 'requestAdapter'>;
}

/** Admission only: GPU inference failures never trigger an off-device retry. */
export async function selectZipEnhancementBackend(
  environment: BackendEnvironment = globalThis.navigator ?? {},
  preference: ExtensionSettings['audioEnhancementBackend'] = 'auto',
): Promise<ZipEnhancementBackend> {
  const swarm = (reason: string): ZipEnhancementBackend => ({ backend: 'swarm', reason });
  if (preference === 'swarm') return swarm('Remote inference selected; on a cache miss, Original recordings leave this device for official GPU inference first, then the GPU swarm.');
  if (!environment.gpu) return swarm('WebGPU is unavailable; enhancement will run remotely, using official GPU inference first, then the GPU swarm.');
  let adapter: GPUAdapter | null;
  try {
    adapter = await environment.gpu.requestAdapter({ powerPreference: 'high-performance', forceFallbackAdapter: false });
  } catch {
    return swarm('The browser could not acquire WebGPU; enhancement will run remotely, using official GPU inference first, then the GPU swarm.');
  }
  if (!adapter || adapter.info?.isFallbackAdapter !== false) return swarm('No confirmed hardware WebGPU adapter; enhancement will run remotely, using official GPU inference first, then the GPU swarm.');
  if (!adapter.features.has('timestamp-query')) {
    return swarm('Required GPU features are unavailable; enhancement will run remotely, using official GPU inference first, then the GPU swarm.');
  }
  if (adapter.limits.maxComputeWorkgroupStorageSize < 21504 || adapter.limits.maxComputeInvocationsPerWorkgroup < 128 ||
      adapter.limits.maxComputeWorkgroupSizeX < 128 || adapter.limits.maxStorageBuffersPerShaderStage < 8) {
    return swarm('The GPU cannot fit the inference kernels; enhancement will run remotely, using official GPU inference first, then the GPU swarm.');
  }
  return { backend: 'webgpu', adapter };
}
