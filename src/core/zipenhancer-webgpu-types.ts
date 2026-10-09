/// <reference types="@webgpu/types" />

// Deliberately structural: application typechecking must not traverse ORT's private TypeScript sources.
export interface TensorViewLike {
  readonly data: number;
  readonly dataType: number;
  readonly dims: readonly number[];
  getFloat32Array(): Float32Array;
  getBigInt64Array(): BigInt64Array;
  getInt32Array(): Int32Array;
  getUint16Array(): Uint16Array;
  reshape(dims: readonly number[]): TensorViewLike;
}

export interface ShaderHelperLike {
  mainStart(workgroupSize?: number | [number, number, number]): string;
}

export interface ProgramInfoLike {
  name: string;
  shaderCache?: {
    hint?: string;
    inputDependencies?: ('none' | 'type' | 'rank' | 'dims')[];
  };
  getShaderSource(helper: ShaderHelperLike): string;
  getRunData(inputs: readonly TensorViewLike[]): {
    outputs: readonly { dims: readonly number[]; dataType: number }[];
    dispatchGroup: { x: number; y?: number; z?: number };
    programUniforms?: readonly { type: number; data: number | readonly number[] }[];
  };
}

export interface AdapterInfoLike {
  isArchitecture(architecture: 'ampere' | 'gen-12lp'): boolean;
  isVendor(vendor: 'amd' | 'intel' | 'nvidia'): boolean;
}

export interface ComputeMappingLike {
  readonly inputs?: readonly (TensorViewLike | number)[];
  readonly outputs?: readonly number[];
}

export interface ComputeContextLike {
  readonly adapterInfo: AdapterInfoLike;
  readonly opKernelContext: number;
  readonly inputs: readonly TensorViewLike[];
  readonly kernelCustomData: Record<string, unknown>;
  readonly customDataBuffer: Uint8Array;
  readonly outputCount: number;
  compute(program: ProgramInfoLike, mapping?: ComputeMappingLike): TensorViewLike[];
  output(index: number, dims: readonly number[]): number;
}

export interface ZipGpuProfileEvent {
  version: number;
  kernelId: number;
  kernelName: string;
  kernelType: string;
  programName: string;
  startTime: number;
  endTime: number;
  inputsMetadata: readonly { dims: readonly number[]; dataType: string }[];
  outputsMetadata: readonly { dims: readonly number[]; dataType: string }[];
}

export interface ZipGpuEnvironment {
  wasm: Record<string, never>;
  webgpu: { profiling: { mode: 'default'; ondata(event: ZipGpuProfileEvent): void } };
  logLevel: 'error';
  debug: boolean;
}
