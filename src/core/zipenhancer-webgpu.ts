import { WebGpuBackend, createAttributeWithCacheKey, registerOperator } from './zipenhancer-webgpu-backend.mjs';
import { zipRelativeAttention } from './zipenhancer-webgpu-attention';
import type { ZipRelativeAttentionAttributes } from './zipenhancer-webgpu-attention';
import { zipSwoosh } from './zipenhancer-webgpu-swoosh';
import type { ZipSwooshAttributes } from './zipenhancer-webgpu-swoosh';
import { specializeProgram } from './zipenhancer-webgpu-specialize.mjs';
import type { ZipGpuNode, ZipGpuTensor, ZipWebGpuPlan } from './zipenhancer-webgpu-plan';
import type { ComputeContextLike, ComputeMappingLike, ProgramInfoLike, TensorViewLike, ZipGpuProfileEvent } from './zipenhancer-webgpu-types';

const SOURCE_SHA = '2f18c8f7ff10a2702d6243ce1230db9e73e6804dd6cd7b20d8e191ee06924016';
const CHECKPOINT_SHA = 'b18896915e27a821585584221d0c0820f35e12145315ae3f1e73ccd5a68d195f';
const INVENTORY_SHA = 'ce89c8aca78f1053a1b8f356a7f6ecdd9679b2bee08de13dd0a1d9f0079674f8';
const INPUT_NAMES = ['noisy_mag', 'noisy_pha'];
const OUTPUT_NAMES = ['amp_g', 'pha_g'];
const IO_SHAPE = [1, 201, 641];
const TYPE_NAMES: Readonly<Record<number, string>> = { 1: 'float32', 6: 'int32', 7: 'int64', 9: 'bool', 10: 'float16' };
const TYPE_BYTES: Readonly<Record<number, number>> = { 1: 4, 6: 4, 7: 8, 9: 1, 10: 2 };
const SESSION_ID = 1;
const EMPTY_BYTES = new Uint8Array(0);

type Output = { magnitude: Float32Array; phase: Float32Array };
export interface ZipWebGpuAudit {
  run: number;
  mode: 'execute' | 'capture' | 'replay';
  expectedPrograms: number;
  observedPrograms: number;
  computeGroups: number;
  sourceGpuNodes: number;
  fusionConsumers: number;
  sourceInventorySha256: string;
  events: readonly ZipGpuProfileEvent[];
}
export interface ZipWebGpuDiagnostic {
  provider: 'webgpu-jsep-direct';
  mode: 'static' | 'capture-unproven' | 'capture-verified';
  precision: ZipWebGpuPlan['precision'];
  plannedBytes: number;
  peakPlannedBytes: number;
  peakAllocatedBufferBytes: number;
  computeGroups: number;
  activationSlots: number;
  initializerCount: number;
  captureDistinctInputsVerified: boolean;
  lastAudit?: ZipWebGpuAudit;
}
export interface ZipWebGpuEngine {
  readonly diagnostic: ZipWebGpuDiagnostic;
  infer(magnitude: Float32Array, phase: Float32Array): Promise<Output>;
  dispose(): Promise<void>;
}

function requireCondition(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(`ZipEnhancer WebGPU: ${message}`);
}
function equalNumbers(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}
function elements(dims: readonly number[]): number {
  let size = 1;
  for (const dim of dims) {
    requireCondition(Number.isSafeInteger(dim) && dim >= 0, 'invalid static dimension');
    size *= dim;
    requireCondition(Number.isSafeInteger(size), 'tensor element count overflow');
  }
  return size;
}
function tensorBytes(tensor: Pick<ZipGpuTensor, 'dims' | 'dataType'>): number {
  const width = TYPE_BYTES[tensor.dataType];
  requireCondition(width !== undefined, `unsupported tensor dtype ${tensor.dataType}`);
  const bytes = elements(tensor.dims) * width;
  requireCondition(Number.isSafeInteger(bytes), 'tensor byte count overflow');
  return bytes;
}
function padded(bytes: number): number { return Math.ceil(bytes / 16) * 16; }
async function sha256(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', bytes as Uint8Array<ArrayBuffer>);
  return Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, '0')).join('');
}
function freezePlan(value: unknown): void {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.freeze(value);
    for (const child of Object.values(value)) freezePlan(child);
  }
}

interface ValidatedPlan {
  tensors: Map<string, ZipGpuTensor>;
  roots: Map<string, string>;
  computeNodes: ZipGpuNode[];
  sourceGpuNames: Set<string>;
  plannedBytes: number;
}
async function validatePlan(plan: ZipWebGpuPlan, weights: Uint8Array): Promise<ValidatedPlan> {
  requireCondition(plan.schema === 'babel-zip-webgpu-v1' && plan.kernelAbi === 'zip-relative-attention-v1+swoosh-v1', 'unsupported plan schema/kernel ABI');
  requireCondition(plan.sourceGraphSha256 === SOURCE_SHA && plan.checkpointSha256 === CHECKPOINT_SHA, 'source graph/checkpoint identity mismatch');
  requireCondition(plan.precision === 'float32' || plan.precision === 'mixed-float16', 'unsupported precision');
  requireCondition(JSON.stringify(plan.inputs) === JSON.stringify(INPUT_NAMES) && JSON.stringify(plan.outputs) === JSON.stringify(OUTPUT_NAMES), 'unexpected model boundaries');
  requireCondition(plan.weights.byteLength === weights.byteLength && await sha256(weights) === plan.weights.sha256, 'binary weights digest/length mismatch');
  const sourceRows = plan.sources.map(source => `${source.name}\0${source.op}\0${source.placement}`).sort();
  requireCondition(await sha256(new TextEncoder().encode(sourceRows.join('\n'))) === INVENTORY_SHA, 'source inventory is not the pinned production graph');
  const sources = new Map(plan.sources.map(source => [source.name, source]));
  requireCondition(sources.size === 2329 && plan.sources.length === sources.size, 'invalid source inventory');
  const tensors = new Map<string, ZipGpuTensor>();
  const slots = new Map<number, number>();
  for (const slot of plan.slots) {
    requireCondition(Number.isSafeInteger(slot.id) && slot.id >= 0 && !slots.has(slot.id), 'duplicate/invalid activation slot');
    requireCondition(Number.isSafeInteger(slot.byteLength) && slot.byteLength > 0, 'invalid activation slot extent');
    slots.set(slot.id, slot.byteLength);
  }
  for (const tensor of plan.tensors) {
    requireCondition(tensor.name.length > 0 && !tensors.has(tensor.name), 'duplicate/empty tensor name');
    const bytes = tensorBytes(tensor);
    requireCondition(plan.precision !== 'float32' || tensor.dataType !== 10, 'half tensor in float32 plan');
    if (tensor.initializer) {
      const { offset, byteLength } = tensor.initializer;
      requireCondition(Number.isSafeInteger(offset) && offset >= 0 && offset % 16 === 0 && byteLength === bytes && offset + bytes <= weights.byteLength, `invalid initializer bytes: ${tensor.name}`);
      requireCondition(tensor.aliasOf === undefined && tensor.slot === undefined, 'initializer cannot alias an activation');
    }
    tensors.set(tensor.name, tensor);
  }
  const roots = new Map<string, string>();
  const visiting = new Set<string>();
  const rootOf = (name: string): string => {
    const known = roots.get(name);
    if (known !== undefined) return known;
    const tensor = tensors.get(name);
    requireCondition(tensor && !visiting.has(name), `missing tensor or alias cycle: ${name}`);
    visiting.add(name);
    let root = name;
    if (tensor.aliasOf !== undefined) {
      root = rootOf(tensor.aliasOf);
      const original = tensors.get(root)!;
      requireCondition(original.dataType === tensor.dataType && elements(original.dims) === elements(tensor.dims), `invalid storage alias: ${name}`);
      requireCondition(tensor.slot === undefined || tensor.slot === original.slot, `alias slot differs from root: ${name}`);
    }
    visiting.delete(name);
    roots.set(name, root);
    return root;
  };
  for (const name of tensors.keys()) rootOf(name);
  const boundary = new Set([...plan.inputs, ...plan.outputs]);
  for (const name of boundary) {
    const tensor = tensors.get(name);
    requireCondition(tensor && tensor.dataType === 1 && equalNumbers(tensor.dims, IO_SHAPE) && !tensor.initializer && tensor.slot === undefined, `invalid FP32 boundary: ${name}`);
    requireCondition(!plan.inputs.includes(name) || !tensor.aliasOf, 'graph inputs cannot alias existing storage');
    requireCondition(tensors.get(rootOf(name))!.slot === undefined, 'boundary root cannot occupy an activation slot');
  }
  const boundaryRoots = new Set([...boundary].map(rootOf));
  requireCondition(boundaryRoots.size === boundary.size, 'graph boundaries must have distinct storage');
  const producers = new Map<string, number>();
  const lastUses = new Map<string, number>();
  const available = new Set(plan.inputs);
  for (const tensor of tensors.values()) if (tensor.initializer) available.add(tensor.name);
  for (const name of available) producers.set(rootOf(name), -1);
  const nodeNames = new Map<string, ZipGpuNode>();
  const nodeIds = new Set<number>();
  const covered = new Set<string>();
  const sourceClaims = new Map<string, number>();
  const computeNodes: ZipGpuNode[] = [];
  plan.nodes.forEach((node, index) => {
    requireCondition(node.name.length > 0 && !nodeNames.has(node.name) && Number.isSafeInteger(node.id) && node.id >= 0 && !nodeIds.has(node.id), 'duplicate/invalid node identity');
    nodeNames.set(node.name, node);
    nodeIds.add(node.id);
    requireCondition(node.outputs.length > 0 && new Set(node.outputs).size === node.outputs.length, `invalid node outputs: ${node.name}`);
    for (const input of node.inputs) {
      if (!input) continue;
      requireCondition(available.has(input), `input precedes its producer: ${node.name}/${input}`);
      lastUses.set(rootOf(input), index);
    }
    requireCondition(new Set(node.sources).size === node.sources.length, `duplicate source claim: ${node.name}`);
    for (const source of node.sources) {
      requireCondition(sources.has(source), `unknown source claim: ${source}`);
      covered.add(source);
      sourceClaims.set(source, (sourceClaims.get(source) ?? 0) + 1);
    }
    if (node.kind === 'alias') {
      requireCondition(node.op === 'Alias' && node.sources.length > 0 && node.sources.every(name => sources.get(name)!.placement === 'tensor-alias'), `GPU source claimed by alias: ${node.name}`);
    } else {
      requireCondition(node.kind === 'compute' && node.op !== 'Alias', `invalid node kind: ${node.name}`);
      computeNodes.push(node);
      if (node.op !== 'ZipRelativeAttention' && node.op !== 'ZipSwoosh') {
        requireCondition(node.reason === 'precision-cast'
          ? node.op === 'Cast' && node.sources.length === 0
          : node.sources.length === 1 && sources.get(node.sources[0])?.op === node.op && sources.get(node.sources[0])?.placement === 'gpu', `unbound compute source: ${node.name}`);
      }
    }
    for (const output of node.outputs) {
      const tensor = tensors.get(output);
      requireCondition(tensor && !available.has(output), `missing/duplicate output: ${output}`);
      const root = rootOf(output);
      if (node.kind === 'alias') {
        requireCondition(tensor.aliasOf && available.has(tensor.aliasOf) && node.inputs.includes(tensor.aliasOf), `alias is not an input view: ${output}`);
      } else {
        requireCondition(root === output && !tensor.initializer && !producers.has(root), `compute output aliases existing storage: ${output}`);
        producers.set(root, index);
      }
      lastUses.set(root, Math.max(lastUses.get(root) ?? -1, index));
      available.add(output);
    }
  });
  requireCondition(covered.size === sources.size, 'incomplete source coverage');
  for (const output of plan.outputs) {
    requireCondition(available.has(output), `unproduced graph output: ${output}`);
    lastUses.set(rootOf(output), plan.nodes.length);
  }
  const fusedNodes = new Set<string>();
  const sharedScoreSources = new Set<string>();
  const fusedSources = new Set<string>();
  requireCondition(plan.attentionGroups.length === 8 && new Set(plan.attentionGroups.map(group => group.id)).size === 8, 'invalid attention group inventory');
  for (const group of plan.attentionGroups) {
    requireCondition(new Set(group.scoreSources).size === group.scoreSources.length, 'duplicate attention score source');
    for (const name of group.scoreSources) {
      requireCondition(!sharedScoreSources.has(name), 'score source shared across unrelated attention groups');
      sharedScoreSources.add(name);
    }
    requireCondition(group.scoreSources.length > 0 && group.consumers.length === 3 && new Set(group.consumers.map(consumer => consumer.stage)).size === 3, `incomplete attention consumers: ${group.id}`);
    for (const consumer of group.consumers) {
      requireCondition(['nonlinear', 'self1', 'self2'].includes(consumer.stage), 'invalid attention stage');
      const node = nodeNames.get(consumer.node);
      const expected = new Set([...group.scoreSources, ...consumer.sources]);
      requireCondition(node?.kind === 'compute' && node.op === 'ZipRelativeAttention' && !fusedNodes.has(node.name), `missing/repeated fused consumer: ${consumer.node}`);
      requireCondition(node.sources.length === expected.size && node.sources.every(name => expected.has(name)), `incomplete fusion provenance: ${node.name}`);
      requireCondition(node.inputs.length === 5 && node.inputs[0] === group.projections.q && node.inputs[1] === group.projections.k && node.inputs[2] === group.projections.p && node.inputs[3] === group.projections.pos, `fusion projections differ: ${node.name}`);
      requireCondition(node.outputs.length === 1 && node.outputs[0] === consumer.output && consumer.heads === (consumer.stage === 'nonlinear' ? 1 : 4) && node.attributes.heads === consumer.heads, `fusion consumer boundary mismatch: ${node.name}`);
      fusedNodes.add(node.name);
      for (const name of expected) fusedSources.add(name);
    }
  }
  requireCondition(computeNodes.filter(node => node.op === 'ZipRelativeAttention').length === fusedNodes.size && fusedNodes.size === 24, 'unregistered attention dispatch group');
  const swooshNodes = computeNodes.filter(node => node.op === 'ZipSwoosh');
  requireCondition(swooshNodes.length === 40, 'incomplete pointwise fusion inventory');
  for (const node of swooshNodes) {
    const attrs = node.attributes as unknown as ZipSwooshAttributes;
    requireCondition(Array.isArray(attrs.steps) && attrs.steps.length === 10 && node.sources.length === attrs.steps.length &&
      attrs.steps.every((step, index) => step && step.source === node.sources[index] &&
        sources.get(step.source)?.placement === 'gpu' && sources.get(step.source)?.op === step.op),
    `unbound pointwise expression sources: ${node.name}`);
    requireCondition(node.inputs.length >= 1 && node.inputs.length <= 7 && node.outputs.length === 1, `invalid pointwise boundary: ${node.name}`);
    const activation = tensors.get(node.inputs[0])!, result = tensors.get(node.outputs[0])!;
    requireCondition((activation.dataType === 1 || activation.dataType === 10) && result.dataType === attrs.outputType &&
      equalNumbers(activation.dims, result.dims), `pointwise output changes shape/storage: ${node.name}`);
    for (const input of node.inputs.slice(1)) {
      const scalar = tensors.get(input)!;
      requireCondition(scalar.initializer && (scalar.dataType === 1 || scalar.dataType === 10) && elements(scalar.dims) === 1, `nonconstant pointwise scalar: ${node.name}/${input}`);
    }
    for (const name of node.sources) fusedSources.add(name);
  }
  for (const source of sources.keys()) requireCondition(sourceClaims.get(source) === (sharedScoreSources.has(source) ? 3 : 1), `source has incorrect fusion multiplicity: ${source}`);
  const sourceGpuNames = new Set(plan.sources.filter(source => source.placement === 'gpu').map(source => source.name));
  const gpuCoverage = new Set(computeNodes.flatMap(node => node.sources).filter(name => sourceGpuNames.has(name)));
  requireCondition(sourceGpuNames.size === 2115 && gpuCoverage.size === sourceGpuNames.size, 'incomplete GPU source coverage');
  const intervals = new Map<number, { start: number; end: number; name: string }[]>();
  let plannedBytes = 0;
  for (const bytes of slots.values()) plannedBytes += padded(bytes);
  for (const tensor of tensors.values()) {
    if (rootOf(tensor.name) !== tensor.name) continue;
    const bytes = tensorBytes(tensor);
    requireCondition(producers.has(tensor.name), `unused/unproduced root: ${tensor.name}`);
    if (tensor.slot !== undefined) {
      requireCondition(!boundaryRoots.has(tensor.name) && !tensor.initializer && slots.has(tensor.slot) && slots.get(tensor.slot)! >= bytes, `invalid slot capacity: ${tensor.name}`);
      const values = intervals.get(tensor.slot) ?? [];
      values.push({ start: producers.get(tensor.name)!, end: lastUses.get(tensor.name) ?? producers.get(tensor.name)!, name: tensor.name });
      intervals.set(tensor.slot, values);
    } else {
      requireCondition(boundaryRoots.has(tensor.name) || tensor.initializer || bytes === 0, `activation lacks static slot: ${tensor.name}`);
      plannedBytes += padded(bytes);
    }
  }
  for (const [id, values] of intervals) {
    values.sort((a, b) => a.start - b.start);
    for (let i = 1; i < values.length; i++) requireCondition(values[i - 1].end < values[i].start, `overlapping slot ${id}: ${values[i - 1].name}/${values[i].name}`);
  }
  requireCondition(intervals.size === slots.size, 'unused activation slot');
  requireCondition(plan.statistics.sourceNodes === 2329 && plan.statistics.sourceGpuNodes === 2115 && plan.statistics.gpuGroups === computeNodes.length && plan.statistics.fusedSourceNodes === fusedSources.size, 'plan statistics differ from coverage');
  requireCondition(plan.statistics.plannedBytes === plannedBytes, 'planned byte accounting mismatch');
  return { tensors, roots, computeNodes, plannedBytes, sourceGpuNames };
}

class JsTensorView implements TensorViewLike {
  constructor(readonly dataType: number, readonly data: number, readonly dims: readonly number[], private readonly host?: Uint8Array) {}
  private bytes(type: number): Uint8Array {
    requireCondition(this.dataType === type && this.host !== undefined, 'CPU tensor reads are restricted to immutable initializer bytes');
    return this.host;
  }
  getFloat32Array(): Float32Array { const bytes = this.bytes(1); return new Float32Array(bytes.buffer, bytes.byteOffset, elements(this.dims)); }
  getBigInt64Array(): BigInt64Array { const bytes = this.bytes(7); return new BigInt64Array(bytes.buffer, bytes.byteOffset, elements(this.dims)); }
  getInt32Array(): Int32Array { const bytes = this.bytes(6); return new Int32Array(bytes.buffer, bytes.byteOffset, elements(this.dims)); }
  getUint16Array(): Uint16Array { const bytes = this.bytes(10); return new Uint16Array(bytes.buffer, bytes.byteOffset, elements(this.dims)); }
  reshape(dims: readonly number[]): TensorViewLike {
    requireCondition(elements(dims) === elements(this.dims), 'reshape changes tensor storage');
    return new JsTensorView(this.dataType, this.data, dims, this.host);
  }
}
const MISSING_TENSOR = new JsTensorView(1, 0, [0]);

function numberAttribute(node: ZipGpuNode, key: string, fallback: number): number {
  const value = node.attributes[key] ?? fallback;
  requireCondition(typeof value === 'number' && Number.isFinite(value), `invalid ${node.op}.${key}`);
  return value;
}
function arrayAttribute(node: ZipGpuNode, key: string, fallback: number[]): number[] {
  const value = node.attributes[key] ?? fallback;
  requireCondition(Array.isArray(value) && value.every(item => Number.isSafeInteger(item)), `invalid ${node.op}.${key}`);
  return value.slice() as number[];
}
function enumAttribute(node: ZipGpuNode, key: string, values: string[], fallback: string): number {
  const value = node.attributes[key] ?? fallback;
  const index = typeof value === 'string' ? values.indexOf(value) : value;
  requireCondition(typeof index === 'number' && Number.isInteger(index) && index >= 0 && index < values.length, `invalid ${node.op}.${key}`);
  return index;
}
function normalizeAttributes(node: ZipGpuNode, tensors: Map<string, ZipGpuTensor>): Record<string, unknown> {
  switch (node.op) {
    case 'Conv': {
      const weight = tensors.get(node.inputs[1])!;
      const rank = weight.dims.length - 2;
      return { format: 'NCHW', auto_pad: enumAttribute(node, 'auto_pad', ['NOTSET', 'VALID', 'SAME_UPPER', 'SAME_LOWER'], 'NOTSET'),
        dilations: arrayAttribute(node, 'dilations', Array(rank).fill(1)), group: numberAttribute(node, 'group', 1),
        kernel_shape: arrayAttribute(node, 'kernel_shape', weight.dims.slice(2)), pads: arrayAttribute(node, 'pads', Array(rank * 2).fill(0)),
        strides: arrayAttribute(node, 'strides', Array(rank).fill(1)), w_is_const: () => Boolean(weight.initializer), activation: '', activation_params: [] };
    }
    case 'Cast': return { to: numberAttribute(node, 'to', -1) };
    case 'Concat': case 'Gather': return { axis: numberAttribute(node, 'axis', 0) };
    case 'Transpose': return { perm: arrayAttribute(node, 'perm', [...tensors.get(node.inputs[0])!.dims.keys()].reverse()) };
    case 'Slice': return { starts: [], ends: [], axes: [] };
    case 'ReduceMax': case 'ReduceMean': case 'ReduceSum':
      return createAttributeWithCacheKey({ axes: arrayAttribute(node, 'axes', []), keepDims: numberAttribute(node, 'keepdims', 1) !== 0, noopWithEmptyAxes: numberAttribute(node, 'noop_with_empty_axes', 0) !== 0 });
    case 'Pad': return createAttributeWithCacheKey({ mode: enumAttribute(node, 'mode', ['constant', 'reflect', 'edge', 'wrap'], 'constant'), value: numberAttribute(node, 'value', 0), pads: arrayAttribute(node, 'pads', []) });
    case 'ZipRelativeAttention': case 'ZipSwoosh': return { ...node.attributes };
    default: return {};
  }
}

const attentionEntry = (context: ComputeContextLike, attributes: unknown): void => {
  requireCondition(attributes !== null && typeof attributes === 'object', 'invalid attention attributes');
  const attrs = attributes as Record<string, unknown>;
  requireCondition((attrs.heads === 1 || attrs.heads === 4) && attrs.queryHeadDim === 12 && attrs.posHeadDim === 4 && (attrs.valueHeadDim === 8 || attrs.valueHeadDim === 48), 'unsupported attention specialization');
  zipRelativeAttention(context, attrs as unknown as ZipRelativeAttentionAttributes);
};
const swooshEntry = (context: ComputeContextLike, attributes: unknown): void => {
  requireCondition(attributes !== null && typeof attributes === 'object', 'invalid pointwise expression attributes');
  zipSwoosh(context, attributes as ZipSwooshAttributes);
};

interface ExpectedProgram {
  kernelId: number;
  kernelName: string;
  kernelType: string;
  programName: string;
  inputsMetadata: ZipGpuProfileEvent['inputsMetadata'];
  outputsMetadata: ZipGpuProfileEvent['outputsMetadata'];
}
function programKey(program: ExpectedProgram): string {
  return JSON.stringify([program.kernelId, program.kernelName, program.kernelType, program.programName, program.inputsMetadata, program.outputsMetadata]);
}
class DispatchAudit {
  readonly expected: ExpectedProgram[] = [];
  readonly events: ZipGpuProfileEvent[] = [];
  private readonly remaining = new Map<string, number>();
  private readonly observedGroups = new Set<number>();
  expect(program: ExpectedProgram): void {
    this.expected.push(program);
    const key = programKey(program);
    this.remaining.set(key, (this.remaining.get(key) ?? 0) + 1);
  }
  observe(event: ZipGpuProfileEvent): void {
    requireCondition(event.version === 1 && Number.isSafeInteger(event.startTime) && Number.isSafeInteger(event.endTime) && event.endTime >= event.startTime, 'invalid real GPU timestamps');
    const key = programKey(event);
    const count = this.remaining.get(key) ?? 0;
    requireCondition(count > 0, `unexpected GPU dispatch: ${event.kernelName}/${event.programName}`);
    this.remaining.set(key, count - 1);
    this.observedGroups.add(event.kernelId);
    this.events.push(event);
  }
  finish(run: number, mode: ZipWebGpuAudit['mode'], nodes: readonly ZipGpuNode[], sourceGpuNames: ReadonlySet<string>): ZipWebGpuAudit {
    requireCondition(this.events.length === this.expected.length && [...this.remaining.values()].every(count => count === 0), 'missing GPU timestamp callbacks');
    requireCondition(nodes.every(node => this.observedGroups.has(node.id)) && this.observedGroups.size === nodes.length, 'compute/fusion group did not execute on GPU');
    const covered = new Set(nodes.flatMap(node => node.sources).filter(name => sourceGpuNames.has(name)));
    const fusionConsumers = nodes.filter(node => node.op === 'ZipRelativeAttention' && this.observedGroups.has(node.id)).length;
    requireCondition(covered.size === sourceGpuNames.size && fusionConsumers === 24, 'missing source or attention consumer dispatch');
    return Object.freeze({ run, mode, expectedPrograms: this.expected.length, observedPrograms: this.events.length,
      computeGroups: this.observedGroups.size, sourceGpuNodes: covered.size, fusionConsumers, sourceInventorySha256: INVENTORY_SHA, events: Object.freeze(this.events) });
  }
}

class JsComputeContext implements ComputeContextLike {
  readonly adapterInfo;
  readonly opKernelContext: number;
  readonly inputs: readonly TensorViewLike[];
  readonly outputCount: number;
  readonly customDataBuffer = EMPTY_BYTES;
  get kernelCustomData(): Record<string, unknown> { return this.backend.currentKernelCustomData; }
  constructor(private readonly backend: WebGpuBackend, private readonly node: ZipGpuNode,
    private readonly views: Map<string, TensorViewLike>, private readonly getAudit: () => DispatchAudit) {
    this.adapterInfo = backend.adapterInfo;
    this.opKernelContext = node.id;
    const lastInput = node.inputs.findLastIndex(name => name !== '');
    this.inputs = node.inputs.slice(0, lastInput + 1).map(name => name ? views.get(name)! : MISSING_TENSOR);
    this.outputCount = node.outputs.length;
  }
  private outputView(index: number, dataType: number, dims: readonly number[]): TensorViewLike {
    const view = this.views.get(this.node.outputs[index]);
    requireCondition(view && view.dataType === dataType && equalNumbers(view.dims, dims), `kernel output differs from static plan: ${this.node.name}[${index}]`);
    return view;
  }
  output(index: number, dims: readonly number[]): number {
    const view = this.views.get(this.node.outputs[index]);
    requireCondition(view, `invalid output index: ${this.node.name}[${index}]`);
    return this.outputView(index, view.dataType, dims).data;
  }
  compute(program: ProgramInfoLike, mapping?: ComputeMappingLike): TensorViewLike[] {
    const inputs = mapping?.inputs?.map(input => typeof input === 'number' ? this.inputs[input] : input) ?? this.inputs;
    requireCondition(inputs.every(Boolean), `invalid program inputs: ${this.node.name}`);
    const outputs = this.backend.run(specializeProgram(program, inputs), inputs, mapping?.outputs ?? [],
      (index, type, dims) => this.outputView(index, type, dims),
      (dataType, dims) => {
        const bytes = tensorBytes({ dataType, dims: [...dims] });
        return new JsTensorView(dataType, bytes ? this.backend.alloc(bytes) : 0, dims);
      }, this.outputCount);
    // backend.run skips only programs whose outputs are all zero-sized. Count actual submitted programs,
    // including weight transforms and kernel temporaries, never one fabricated event per source node.
    if (outputs.some(output => output.data !== 0)) this.getAudit().expect({ kernelId: this.node.id, kernelName: this.node.name,
      kernelType: this.node.op, programName: program.name,
      inputsMetadata: inputs.map(view => ({ dims: view.dims, dataType: TYPE_NAMES[view.dataType] })),
      outputsMetadata: outputs.map(view => ({ dims: view.dims, dataType: TYPE_NAMES[view.dataType] })) });
    return outputs;
  }
}

function compareCapture(actual: Output, expected: Output): void {
  for (const key of ['magnitude', 'phase'] as const) {
    const a = actual[key];
    const b = expected[key];
    requireCondition(a.length === b.length, 'captured output clock changed');
    for (let i = 0; i < a.length; i++) requireCondition(Number.isFinite(a[i]) && Number.isFinite(b[i]) && Math.abs(a[i] - b[i]) <= 1e-6 + 1e-5 * Math.abs(b[i]), `captured ${key} differs from uncaptured execution at ${i}`);
  }
}

export async function createZipWebGpuEngine(
  inputPlan: ZipWebGpuPlan, inputWeights: Uint8Array,
  options: { adapter?: GPUAdapter; capture?: boolean; onDiagnostic?: (diagnostic: ZipWebGpuDiagnostic) => void } = {},
): Promise<ZipWebGpuEngine> {
  // Own these bytes/metadata: callers must not be able to mutate a validated execution plan or host constants.
  const plan = structuredClone(inputPlan);
  const weights = new Uint8Array(inputWeights);
  const validated = await validatePlan(plan, weights);
  freezePlan(plan);
  requireCondition(navigator.gpu, 'WebGPU is unavailable');
  const adapter = options.adapter ?? await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  requireCondition(adapter && !adapter.info.isFallbackAdapter, 'a hardware WebGPU adapter is required');
  requireCondition(adapter.features.has('timestamp-query') || adapter.features.has('chromium-experimental-timestamp-query-inside-passes' as GPUFeatureName), 'GPU timestamp queries are required for dispatch proof');
  requireCondition(plan.precision !== 'mixed-float16' || adapter.features.has('shader-f16'), 'mixed precision requires shader-f16');
  const backend = new WebGpuBackend();
  backend.maxDispatchNumber = 128;
  const buffers: GPUBuffer[] = [];
  const kernels: number[] = [];
  const views = new Map<string, TensorViewLike>();
  const contexts: JsComputeContext[] = [];
  let sessionCreated = false;
  let activeAudit: DispatchAudit | undefined;
  let failed: Error | undefined;
  let closing = false;
  let activeInference: Promise<Output> | undefined;
  let disposePromise: Promise<void> | undefined;
  let cleanupPromise: Promise<void> | undefined;
  let ready = false;
  let initializationScopes = 0;
  let capturePrograms: readonly ExpectedProgram[] | undefined;
  let firstCapturedInput: string | undefined;
  let captureDistinctInputsVerified = false;
  let runCount = 0;
  let diagnostic: ZipWebGpuDiagnostic = {
    provider: 'webgpu-jsep-direct', mode: options.capture ? 'capture-unproven' : 'static', precision: plan.precision,
    plannedBytes: validated.plannedBytes, peakPlannedBytes: validated.plannedBytes, peakAllocatedBufferBytes: 0,
    computeGroups: validated.computeNodes.length, activationSlots: plan.slots.length,
    initializerCount: plan.tensors.filter(tensor => tensor.initializer).length, captureDistinctInputsVerified: false,
  };
  const fatal = (reason: unknown): Error => {
    failed ??= reason instanceof Error ? reason : new Error(String(reason));
    if (ready && !activeInference && !closing) {
      closing = true;
      cleanupPromise ??= cleanup();
      void cleanupPromise.catch(() => undefined);
    }
    return failed;
  };
  const assertLive = (): void => {
    if (failed) throw failed;
    requireCondition(!closing, 'engine is disposed');
  };
  const notify = (audit?: ZipWebGpuAudit): void => {
    diagnostic = Object.freeze({ ...diagnostic, mode: options.capture ? captureDistinctInputsVerified ? 'capture-verified' : 'capture-unproven' : 'static',
      peakAllocatedBufferBytes: backend.peakAllocatedBufferBytes, captureDistinctInputsVerified,
      ...(audit ? { lastAudit: audit } : {}) });
    options.onDiagnostic?.(diagnostic);
  };
  const cleanup = async (): Promise<void> => {
    let cleanupFailure: unknown;
    try {
      if (backend.device) {
        try { backend.flush(); } catch (error) { cleanupFailure ??= error; }
        try { await backend.device.queue.onSubmittedWorkDone(); } catch (error) { cleanupFailure ??= error; }
        try { await backend.drainProfiles(); } catch (error) { cleanupFailure ??= error; }
      }
      for (const id of kernels) {
        try { backend.releaseKernel(id); } catch (error) { cleanupFailure ??= error; }
      }
      kernels.length = 0;
      if (sessionCreated) {
        try { backend.onReleaseSession(SESSION_ID); } catch (error) { cleanupFailure ??= error; }
        sessionCreated = false;
      }
      for (const buffer of buffers) buffer.destroy();
      buffers.length = 0;
      if (backend.gpuDataManager) {
        backend.gpuDataManager.refreshPendingBuffers();
        backend.dispose();
      }
    } finally {
      backend.device?.destroy();
      views.clear();
      contexts.length = 0;
      capturePrograms = undefined;
    }
    if (cleanupFailure && !failed) throw cleanupFailure;
  };
  backend.onFailure = fatal;
  try {
    await backend.initialize({ wasm: {}, webgpu: { profiling: { mode: 'default', ondata: event => {
      requireCondition(activeAudit, 'GPU callback outside an active run');
      activeAudit.observe(event);
    } } }, logLevel: 'error', debug: true }, adapter);
    requireCondition(backend.queryType !== 'none', 'dedicated device lacks GPU profiling');
    requireCondition(backend.device.limits.maxComputeWorkgroupStorageSize >= 21504 &&
      backend.device.limits.maxComputeInvocationsPerWorkgroup >= 128 &&
      backend.device.limits.maxComputeWorkgroupSizeX >= 128 &&
      backend.device.limits.maxStorageBuffersPerShaderStage >= 6, 'device limits cannot execute the full attention tile');
    backend.device.addEventListener('uncapturederror', event => { fatal(event.error); });
    void backend.device.lost.then(info => { if (!closing) fatal(new Error(`ZipEnhancer GPU device lost: ${info.reason}: ${info.message}`)); });
    backend.device.pushErrorScope('out-of-memory');
    backend.device.pushErrorScope('internal');
    backend.device.pushErrorScope('validation');
    initializationScopes = 3;
    backend.onCreateSession();
    sessionCreated = true;
    backend.onRunStart(SESSION_ID);
    const slots = new Map<number, number>();
    let registration = 0;
    const allocate = (bytes: number): number => {
      if (!bytes) return 0;
      const size = padded(bytes);
      requireCondition(size <= backend.device.limits.maxBufferSize && size <= backend.device.limits.maxStorageBufferBindingSize, 'tensor exceeds device buffer limits');
      const buffer = backend.device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      buffers.push(buffer);
      return backend.registerBuffer(SESSION_ID, registration++, buffer, bytes);
    };
    for (const slot of plan.slots) slots.set(slot.id, allocate(slot.byteLength));
    for (const tensor of plan.tensors) {
      if (tensor.aliasOf) continue;
      const data = tensor.slot === undefined ? allocate(tensorBytes(tensor)) : slots.get(tensor.slot)!;
      const host = tensor.initializer ? weights.subarray(tensor.initializer.offset, tensor.initializer.offset + tensor.initializer.byteLength) : undefined;
      views.set(tensor.name, new JsTensorView(tensor.dataType, data, tensor.dims, host));
      if (host && host.byteLength) backend.upload(data, host);
    }
    for (const tensor of plan.tensors) if (tensor.aliasOf) views.set(tensor.name, views.get(validated.roots.get(tensor.name)!)!.reshape(tensor.dims));
    registerOperator('ZipRelativeAttention', attentionEntry);
    registerOperator('ZipSwoosh', swooshEntry);
    for (const node of validated.computeNodes) {
      // Include partially created kernel records in cleanup if its eager parser rejects.
      kernels.push(node.id);
      backend.createKernel(node.op, node.id, normalizeAttributes(node, validated.tensors), node.name);
      contexts.push(new JsComputeContext(backend, node, views, () => {
        requireCondition(activeAudit, 'program outside active dispatch audit');
        return activeAudit;
      }));
    }
    const initializationErrors: Promise<GPUError | null>[] = [];
    while (initializationScopes > 0) {
      initializationScopes--;
      initializationErrors.push(backend.device.popErrorScope());
    }
    const initialized = await Promise.allSettled(initializationErrors);
    await backend.device.queue.onSubmittedWorkDone();
    for (const result of initialized) {
      if (result.status === 'rejected') throw result.reason;
      requireCondition(!result.value, `GPU initialization failed: ${result.value?.message}`);
    }
    if (failed) throw failed;
    const execute = async (mode: ZipWebGpuAudit['mode']): Promise<Output> => {
      if (failed) throw failed;
      const audit = new DispatchAudit();
      activeAudit = audit;
      backend.onRunStart(SESSION_ID);
      const errors: Promise<string | null>[] = [];
      let failure: unknown;
      let capturing = false;
      backend.device.pushErrorScope('out-of-memory');
      backend.device.pushErrorScope('internal');
      backend.device.pushErrorScope('validation');
      try {
        if (mode === 'replay') {
          requireCondition(capturePrograms, 'capture is not initialized');
          for (const program of capturePrograms) audit.expect(program);
          backend.replay();
        } else {
          if (mode === 'capture') { backend.captureBegin(); capturing = true; }
          for (let i = 0; i < validated.computeNodes.length; i++) {
            if (failed) throw failed;
            if (backend.computeKernel(validated.computeNodes[i].id, contexts[i], errors) !== 0) {
              const details = (await Promise.all(errors)).filter((value): value is string => Boolean(value));
              throw new Error(`ZipEnhancer WebGPU kernel ${validated.computeNodes[i].name} failed: ${details.join('; ')}`);
            }
          }
          if (capturing) { backend.captureEnd(); capturing = false; }
          else backend.flush();
        }
      } catch (error) {
        failure = error;
      } finally {
        if (capturing) {
          try { backend.captureEnd(); } catch (error) { failure ??= error; }
        }
        for (let i = 0; i < 3; i++) errors.push(backend.device.popErrorScope().then(error => error?.message ?? null));
      }
      const validation = await Promise.allSettled(errors);
      for (const result of validation) {
        if (result.status === 'rejected') failure ??= result.reason;
        else if (result.value) {
          const prior = failure instanceof Error ? failure : failed;
          failure = new Error(prior ? `${result.value}\n${prior.message}` : result.value, { cause: prior });
        }
      }
      try {
        backend.flush();
        await backend.device.queue.onSubmittedWorkDone();
        await backend.drainProfiles();
      } catch (error) { failure ??= error; }
      if (failure) {
        failed = failure instanceof Error ? failure : new Error(String(failure));
        throw fatal(failed);
      }
      if (failed) throw failed;
      const proof = audit.finish(++runCount, mode, validated.computeNodes, validated.sourceGpuNames);
      if (mode === 'capture') capturePrograms = audit.expected;
      const magnitude = new Float32Array(elements(IO_SHAPE));
      const phase = new Float32Array(elements(IO_SHAPE));
      // Only final outputs are ever read back, and the manager's originalSize is the exact logical extent.
      const downloads = await Promise.allSettled([
        backend.download(views.get(plan.outputs[0])!.data, () => new Uint8Array(magnitude.buffer)),
        backend.download(views.get(plan.outputs[1])!.data, () => new Uint8Array(phase.buffer)),
      ]);
      for (const download of downloads) if (download.status === 'rejected') throw download.reason;
      await backend.drainProfiles();
      if (failed) throw failed;
      activeAudit = undefined;
      notify(proof);
      return { magnitude, phase };
    };
    notify();
    ready = true;
    const infer = async (magnitude: Float32Array, phase: Float32Array): Promise<Output> => {
      assertLive();
      requireCondition(!activeInference, 'concurrent infer calls are not allowed');
      requireCondition(magnitude instanceof Float32Array && phase instanceof Float32Array && magnitude.length === elements(IO_SHAPE) && phase.length === elements(IO_SHAPE), 'input clock/shape mismatch');
      const work = async (): Promise<Output> => {
        try {
          // Capture proof hashes the exact owned snapshots uploaded, not caller-mutable arrays across an await.
          const magnitudeBytes = new Uint8Array(magnitude.buffer, magnitude.byteOffset, magnitude.byteLength);
          const phaseBytes = new Uint8Array(phase.buffer, phase.byteOffset, phase.byteLength);
          const magUpload = options.capture ? new Uint8Array(magnitudeBytes) : magnitudeBytes;
          const phaUpload = options.capture ? new Uint8Array(phaseBytes) : phaseBytes;
          // Each earlier invocation completes submissions/readbacks before these fixed input IDs are overwritten.
          backend.upload(views.get(plan.inputs[0])!.data, magUpload);
          backend.upload(views.get(plan.inputs[1])!.data, phaUpload);
          if (!options.capture) return await execute('execute');
          const inputDigest = `${await sha256(magUpload)}:${await sha256(phaUpload)}`;
          if (!capturePrograms) {
            const normal = await execute('execute');
            const captured = await execute('capture');
            compareCapture(captured, normal);
            firstCapturedInput = inputDigest;
            return captured;
          }
          if (!captureDistinctInputsVerified && inputDigest !== firstCapturedInput) {
            const normal = await execute('execute');
            const replayed = await execute('replay');
            compareCapture(replayed, normal);
            captureDistinctInputsVerified = true;
            notify();
            return replayed;
          }
          return await execute('replay');
        } catch (error) {
          fatal(error);
          closing = true;
          cleanupPromise ??= cleanup();
          await cleanupPromise;
          throw failed;
        }
      };
      const pending = work();
      activeInference = pending;
      try { return await pending; }
      finally { activeInference = undefined; }
    };
    return {
      get diagnostic() { return diagnostic; },
      infer,
      dispose: () => {
        if (!disposePromise) {
          closing = true;
          const running = activeInference;
          disposePromise = (async () => {
            if (running) await running.catch(() => undefined);
            cleanupPromise ??= cleanup();
            await cleanupPromise;
          })();
        }
        return disposePromise;
      },
    };
  } catch (error) {
    fatal(error);
    closing = true;
    while (initializationScopes > 0) {
      initializationScopes--;
      await backend.device.popErrorScope().catch(() => null);
    }
    cleanupPromise ??= cleanup();
    await cleanupPromise;
    throw failed;
  }
}
