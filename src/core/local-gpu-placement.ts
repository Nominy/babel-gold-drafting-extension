// ORT 1.29's public WebGPU profiler reports dispatched GPU programs, not CPU kernels.
// Source identities and disabled graph optimizations make missing neural dispatches
// a rejection, rather than an unobservable/accepted CPU fallback.
export interface PlacementTensor { name: string; dtype: string; ancestry: string }
export interface PlacementNode {
  name: string;
  opType: string;
  placement: 'gpu' | 'host-metadata' | 'tensor-alias' | 'constant';
  inputs: PlacementTensor[];
  outputs: PlacementTensor[];
  reason: string;
}
export interface PlacementGraph { path: string; sha256: string; nodes: PlacementNode[] }
export interface PlacementPolicy {
  schema: 'babel-c-denoise-gpu-placement-v1';
  sources: { asrCheckpointSha256: string; cDenoiseCheckpointSha256: string };
  graphs: Record<'asr' | 'context' | 'denoise', PlacementGraph>;
}
export interface GpuDispatch {
  version: number;
  kernelName: string;
  kernelType: string;
  programName: string;
  startTime: number;
  endTime: number;
}
export interface GraphPlacementDiagnostic {
  readonly path: string;
  readonly sha256: string;
  readonly verifiedRuns: number;
  readonly requiredGpuNodes: number;
  readonly verifiedGpuNodes: number;
  readonly gpuPrograms: number;
  readonly allowedHostMetadataNodes: number;
  readonly storageAliasNodes: number;
  readonly constantNodes: number;
}

// Deliberately conservative preflight for this model family, pinned to ORT 1.29
// lib/wasm/jsep/webgpu/op-resolve-rules.ts. Unknown compute operators fail closed.
const GPU_OPS: Record<string, true> = {
  Abs: true, Add: true, ArgMax: true, ArgMin: true, AveragePool: true, BatchNormalization: true, Cast: true,
  Ceil: true, Clip: true, Concat: true, Conv: true, ConvTranspose: true, Cos: true, Div: true, Einsum: true,
  Elu: true, Equal: true, Erf: true, Exp: true, Expand: true, Floor: true, Gather: true, GatherElements: true,
  GatherND: true, Gelu: true, Gemm: true, GlobalAveragePool: true, GlobalMaxPool: true, Greater: true,
  GreaterOrEqual: true, InstanceNormalization: true, LayerNormalization: true, LeakyRelu: true,
  Less: true, LessOrEqual: true, Log: true, MatMul: true, MaxPool: true, Mul: true, Neg: true, Not: true, Pad: true,
  Pow: true, Range: true, Reciprocal: true, ReduceMin: true, ReduceMean: true, ReduceMax: true,
  ReduceSum: true, ReduceProd: true, ReduceL1: true, ReduceL2: true, ReduceLogSum: true,
  ReduceLogSumExp: true, ReduceSumSquare: true, Relu: true, Resize: true, Sigmoid: true, Sin: true,
  Slice: true, Split: true, Sqrt: true, Softmax: true, Sub: true, Tan: true, Tanh: true, Tile: true, Transpose: true, Where: true
};
const ALIAS_OPS: Record<string, true> = { Identity: true, Reshape: true, Squeeze: true, Unsqueeze: true };

export function validatePlacementPolicy(value: unknown, sources: PlacementPolicy['sources']): PlacementPolicy {
  const policy = value as PlacementPolicy | null;
  if (!policy || policy.schema !== 'babel-c-denoise-gpu-placement-v1' ||
    policy.sources?.asrCheckpointSha256 !== sources.asrCheckpointSha256 ||
    policy.sources?.cDenoiseCheckpointSha256 !== sources.cDenoiseCheckpointSha256) {
    throw new Error('GPU placement policy does not match the accepted source checkpoints. Reinstall the verified bundle.');
  }
  for (const key of ['asr', 'context', 'denoise'] as const) {
    const graph = policy.graphs?.[key];
    if (!graph || typeof graph.path !== 'string' || !/^[a-f0-9]{64}$/.test(graph.sha256) ||
      !Array.isArray(graph.nodes) || !graph.nodes.length) throw new Error(`Invalid ${key} GPU placement graph.`);
    const names = new Set<string>();
    let required = 0;
    for (const node of graph.nodes) {
      if (!node || typeof node.name !== 'string' || !node.name || names.has(node.name) ||
        typeof node.opType !== 'string' || !Array.isArray(node.inputs) || !Array.isArray(node.outputs) ||
        !node.outputs.length || typeof node.reason !== 'string' || !node.reason ||
        [...node.inputs, ...node.outputs].some((tensor) => !tensor || typeof tensor.name !== 'string' ||
          !tensor.name || typeof tensor.dtype !== 'string' || typeof tensor.ancestry !== 'string')) {
        throw new Error(`Invalid or duplicate placement node in ${graph.path}.`);
      }
      names.add(node.name);
      if (node.placement === 'gpu') {
        if (!Object.hasOwn(GPU_OPS, node.opType)) throw new Error(`Unsupported WebGPU neural operator ${node.opType} (${node.name}). No accepted CPU fallback.`);
        if (node.outputs.some((tensor) => tensor.dtype === 'unknown')) throw new Error(`Unproven tensor dtype at ${node.name}.`);
        required += 1;
      } else if (node.placement === 'tensor-alias') {
        const sameTypeCast = node.opType === 'Cast' && node.inputs.length === 1 && node.outputs.length === 1 &&
          node.inputs[0].dtype !== 'unknown' && node.inputs[0].dtype === node.outputs[0].dtype;
        if (!Object.hasOwn(ALIAS_OPS, node.opType) && !sameTypeCast) throw new Error(`Non-alias operator ${node.name} cannot bypass the GPU compute audit.`);
      } else if (node.placement === 'constant') {
        if (node.opType !== 'Constant') throw new Error(`Nonconstant operator ${node.name} cannot bypass the GPU compute audit.`);
      } else if (node.placement === 'host-metadata') {
        if (node.outputs.some((tensor) => tensor.ancestry !== 'metadata') ||
          !['Shape', 'Size'].includes(node.opType) && node.inputs.some((tensor) => !['metadata', 'constant'].includes(tensor.ancestry))) {
          throw new Error(`Unproven host metadata ancestry at ${node.name}; neural tensor values require GPU dispatch.`);
        }
      } else {
        throw new Error(`Unrecognized placement at ${node.name}.`);
      }
    }
    if (!required) throw new Error(`${graph.path} has no required neural GPU dispatches.`);
  }
  return policy;
}

export async function bindPlacementGraph(graph: PlacementGraph, path: string, bytes: ArrayBuffer): Promise<void> {
  const sha256 = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)),
    (byte) => byte.toString(16).padStart(2, '0')).join('');
  if (graph.path !== path || graph.sha256 !== sha256) throw new Error(`GPU placement policy is not bound to the loaded graph ${path}. Reinstall the verified bundle.`);
}

export class GpuRunAudit {
  private readonly required: Map<string, PlacementNode>;
  private readonly observed = new Set<string>();
  private programs = 0;
  private error: Error | null = null;

  constructor(private readonly graph: PlacementGraph) {
    this.required = new Map(graph.nodes.filter((node) => node.placement === 'gpu').map((node) => [node.name, node]));
  }

  observe(data: GpuDispatch): void {
    const node = this.required.get(data.kernelName);
    if (!node) return;
    if (data.version !== 1 || data.kernelType !== node.opType || !data.programName ||
      !Number.isSafeInteger(data.startTime) || !Number.isSafeInteger(data.endTime) ||
      data.endTime < data.startTime) {
      this.error = new Error(`Invalid GPU dispatch evidence for ${node.name}.`);
      return;
    }
    this.observed.add(node.name);
    this.programs += 1;
  }

  finish(verifiedRuns: number): GraphPlacementDiagnostic {
    if (this.error) throw this.error;
    const missing = [...this.required.keys()].filter((name) => !this.observed.has(name));
    if (missing.length) throw new Error(`Strict WebGPU placement failed for ${this.graph.path}: ${missing.length} required neural nodes had no GPU dispatch (${missing.slice(0, 8).join(', ')}). CPU neural results are not accepted.`);
    return Object.freeze({ path: this.graph.path, sha256: this.graph.sha256, verifiedRuns,
      requiredGpuNodes: this.required.size, verifiedGpuNodes: this.observed.size, gpuPrograms: this.programs,
      allowedHostMetadataNodes: this.graph.nodes.filter((node) => node.placement === 'host-metadata').length,
      storageAliasNodes: this.graph.nodes.filter((node) => node.placement === 'tensor-alias').length,
      constantNodes: this.graph.nodes.filter((node) => node.placement === 'constant').length });
  }
}
