export type ZipEnhancementBackend =
  | { backend: 'webgpu'; adapter: GPUAdapter }
  | { backend: 'swarm'; reason: string };

interface BackendEnvironment {
  gpu?: Pick<GPU, 'requestAdapter'>;
}

/** Admission only: GPU inference failures never trigger an off-device retry. */
export async function selectZipEnhancementBackend(
  environment: BackendEnvironment = globalThis.navigator ?? {},
): Promise<ZipEnhancementBackend> {
  const swarm = (reason: string): ZipEnhancementBackend => ({ backend: 'swarm', reason });
  if (!environment.gpu) return swarm('WebGPU is unavailable; enhancement will run remotely on a swarm GPU.');
  let adapter: GPUAdapter | null;
  try {
    adapter = await environment.gpu.requestAdapter({ powerPreference: 'high-performance', forceFallbackAdapter: false });
  } catch {
    return swarm('The browser could not acquire WebGPU; enhancement will run remotely on a swarm GPU.');
  }
  if (!adapter || adapter.info?.isFallbackAdapter !== false) return swarm('No confirmed hardware WebGPU adapter; enhancement will run remotely on a swarm GPU.');
  if (!adapter.features.has('shader-f16') || !adapter.features.has('timestamp-query')) {
    return swarm('Required GPU features are unavailable; enhancement will run remotely on a swarm GPU.');
  }
  if (adapter.limits.maxComputeWorkgroupStorageSize < 21504 || adapter.limits.maxComputeInvocationsPerWorkgroup < 128 ||
      adapter.limits.maxComputeWorkgroupSizeX < 128 || adapter.limits.maxStorageBuffersPerShaderStage < 8) {
    return swarm('The GPU cannot fit the inference kernels; enhancement will run remotely on a swarm GPU.');
  }
  return { backend: 'webgpu', adapter };
}
