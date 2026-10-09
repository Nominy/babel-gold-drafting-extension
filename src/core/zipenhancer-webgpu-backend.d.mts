import type { AdapterInfoLike, ComputeContextLike, ProgramInfoLike, TensorViewLike, ZipGpuEnvironment } from './zipenhancer-webgpu-types';

export function registerOperator(name: string, entry: (context: ComputeContextLike, attributes: unknown) => void): void;
export function createAttributeWithCacheKey<T extends Record<string, unknown>>(attributes: T): T & { readonly cacheKey: string };

export class WebGpuBackend {
  device: GPUDevice;
  adapterInfo: AdapterInfoLike;
  maxDispatchNumber: number;
  queryType: 'none' | 'inside-passes' | 'at-passes';
  currentKernelCustomData: Record<string, unknown>;
  onFailure?: (error: Error) => void;
  allocatedBufferBytes: number;
  peakAllocatedBufferBytes: number;
  gpuDataManager: { refreshPendingBuffers(): void };
  initialize(env: ZipGpuEnvironment, adapter: GPUAdapter): Promise<void>;
  createKernel(type: string, id: number, attributes: unknown, name: string): void;
  releaseKernel(id: number): void;
  computeKernel(id: number, context: ComputeContextLike, errors: Promise<string | null>[]): number;
  run(program: ProgramInfoLike, inputs: readonly TensorViewLike[], outputIndices: readonly number[],
    createOutput: (index: number, dataType: number, dims: readonly number[]) => TensorViewLike,
    createIntermediate: (dataType: number, dims: readonly number[]) => TensorViewLike, outputCount: number): TensorViewLike[];
  alloc(size: number): number;
  free(id: number): number;
  upload(id: number, data: Uint8Array): void;
  download(id: number, getTarget: () => Uint8Array): Promise<void>;
  registerBuffer(sessionId: number, index: number, buffer: GPUBuffer, logicalSize: number): number;
  unregisterBuffers(sessionId: number): void;
  onCreateSession(): void;
  onReleaseSession(sessionId: number): void;
  onRunStart(sessionId: number): void;
  captureBegin(): void;
  captureEnd(): void;
  replay(): void;
  flush(): void;
  drainProfiles(): Promise<void>;
  dispose(): void;
}
