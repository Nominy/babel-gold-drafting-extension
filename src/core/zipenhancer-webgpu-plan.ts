export interface ZipGpuTensor {
  name: string;
  dataType: number;
  dims: number[];
  initializer?: { offset: number; byteLength: number };
  aliasOf?: string;
  slot?: number;
}

export interface ZipGpuNode {
  id: number;
  name: string;
  kind: 'compute' | 'alias';
  op: string;
  inputs: string[];
  outputs: string[];
  attributes: Record<string, unknown>;
  /** Original source nodes implemented by this operation, never synthetic dispatches. */
  sources: string[];
  reason?: 'precision-cast';
}

export interface ZipGpuAttentionGroup {
  id: string;
  projections: { q: string; k: string; p: string; pos: string };
  scoreSources: string[];
  consumers: { node: string; stage: 'nonlinear' | 'self1' | 'self2'; heads: number; output: string; sources: string[] }[];
}

export interface ZipWebGpuPlan {
  schema: 'babel-zip-webgpu-v1';
  checkpointSha256: string;
  sourceGraphSha256: string;
  precision: 'float32' | 'mixed-float16';
  kernelAbi: 'zip-relative-attention-v1+swoosh-v1';
  weights: { file: string; sha256: string; byteLength: number };
  inputs: string[];
  outputs: string[];
  tensors: ZipGpuTensor[];
  nodes: ZipGpuNode[];
  slots: { id: number; byteLength: number }[];
  sources: { name: string; op: string; placement: 'gpu' | 'tensor-alias' }[];
  attentionGroups: ZipGpuAttentionGroup[];
  statistics: { sourceNodes: number; sourceGpuNodes: number; fusedSourceNodes: number; gpuGroups: number; plannedBytes: number };
}
