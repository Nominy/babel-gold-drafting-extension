import type {
  ComputeContextLike,
  ProgramInfoLike,
  TensorViewLike,
} from './zipenhancer-webgpu-types';

export interface ZipRelativeAttentionAttributes {
  readonly heads: 1 | 4;
  readonly queryHeadDim: 12;
  readonly posHeadDim: 4;
  readonly valueHeadDim: 8 | 48;
}

const QUERY_TILE = 16;
const KEY_TILE = 64;
const ROW_LANES = 8;
const WORKGROUP_SIZE = QUERY_TILE * ROW_LANES;
const KERNEL_ABI = 'zip-relative-attention-wgsl-v1-native-half-probabilities';

/**
 * Standard WGSL counterpart of scripts/zipenhancer_tiled_attention.cu.
 *
 * Unlike CUDA's packed inputs, ONNX K is [4,B,12,N] and pos is
 * [4,1,4,2N-1]. Q/P retain four physical heads even for the head-zero
 * nonlinear consumer. There is no sqrt(D) scaling or attention window.
 *
 * Half policy matches the native experiment: FP32 dots, round QK and P*pos
 * separately to half, then round their sum to half. Online softmax max/sum
 * stay FP32; only unnormalized tile probabilities round to half before the
 * FP32 P*V accumulation. Output rounds to the input storage type. Diagnostic
 * float32 removes all these half boundaries. Reduction order and exp differ
 * from CUDA, so waveform admission remains necessary; this is not bitwise
 * emulation of WMMA or source half-rounded normalized probabilities.
 *
 * All workgroup arrays are FP32, with 16 queries and 64 streamed keys:
 * Q 192 + P 64 + K 768 + V (64*Dv) + probabilities 1024 + reductions 256
 * scalars = 11264 bytes (Dv=8) / 21504 bytes (Dv=48). Six storage bindings,
 * no uniforms, no quadratic scratch, no subgroup/matrix feature required.
 * The standard backend enables f16 only on a shader-f16 device; the executor
 * must reject a half plan before dispatch if that feature is unavailable.
 */
export function createZipRelativeAttentionProgram(
  inputs: readonly TensorViewLike[],
  attributes: ZipRelativeAttentionAttributes,
): ProgramInfoLike {
  if (inputs.length !== 5) {
    throw new Error('ZipRelativeAttention requires Q, K, P, pos and V.');
  }
  const q = inputs[0];
  const v = inputs[4];
  const { heads, queryHeadDim, posHeadDim, valueHeadDim } = attributes;
  if (
    (heads !== 1 && heads !== 4) || queryHeadDim !== 12 || posHeadDim !== 4 ||
    (heads === 1 ? valueHeadDim !== 48 : valueHeadDim !== 8)
  ) {
    throw new Error('ZipRelativeAttention requires head-zero Dv=48 or four-head Dv=8.');
  }
  if (q.dims.length !== 4) {
    throw new Error('ZipRelativeAttention Q must have rank four.');
  }
  const batch = q.dims[1];
  const length = q.dims[2];
  if (!Number.isSafeInteger(batch) || batch <= 0 || !Number.isSafeInteger(length) || length <= 0) {
    throw new Error('ZipRelativeAttention requires positive static batch and sequence dimensions.');
  }
  const expectedShapes = [
    [4, batch, length, 12],
    [4, batch, 12, length],
    [4, batch, length, 4],
    [4, 1, 4, 2 * length - 1],
    [heads, batch, length, valueHeadDim],
  ];
  const dataType = q.dataType;
  if (dataType !== 1 && dataType !== 10) {
    throw new Error('ZipRelativeAttention supports only float32 and float16 storage.');
  }
  for (let index = 0; index < inputs.length; ++index) {
    const input = inputs[index];
    const expected = expectedShapes[index];
    if (
      input.dataType !== dataType || input.dims.length !== expected.length ||
      expected.some((dimension, axis) => input.dims[axis] !== dimension)
    ) {
      throw new Error(`ZipRelativeAttention input ${index} has an incompatible shape or storage type.`);
    }
    const elements = expected.reduce((product, dimension) => product * dimension, 1);
    if (!Number.isSafeInteger(elements) || elements > 0xffffffff) {
      throw new Error('ZipRelativeAttention tensor offsets exceed WGSL u32 addressing.');
    }
  }

  const queryTiles = Math.ceil(length / QUERY_TILE);
  const workgroups = queryTiles * heads * batch;
  const valueSlots = valueHeadDim / ROW_LANES;
  const half = dataType === 10;
  const storageType = half ? 'f16' : 'f32';
  const roundHalf = (expression: string): string => half ? `f32(f16(${expression}))` : expression;

  return {
    name: 'ZipRelativeAttention',
    shaderCache: {
      // Every shape, type and attribute affecting generated WGSL is explicit.
      hint: `${KERNEL_ABI};${dataType};${batch};${length};${heads};${queryHeadDim};${posHeadDim};${valueHeadDim};${QUERY_TILE};${KEY_TILE};${ROW_LANES}`,
      inputDependencies: ['type', 'type', 'type', 'type', 'type'],
    },
    getRunData: () => ({
      outputs: [{ dims: v.dims, dataType }],
      dispatchGroup: { x: queryTiles, y: heads * batch, z: 1 },
    }),
    getShaderSource: (shaderHelper) => `
@group(0) @binding(0) var<storage, read> q: array<${storageType}>;
@group(0) @binding(1) var<storage, read> k: array<${storageType}>;
@group(0) @binding(2) var<storage, read> p: array<${storageType}>;
@group(0) @binding(3) var<storage, read> pos: array<${storageType}>;
@group(0) @binding(4) var<storage, read> v: array<${storageType}>;
@group(0) @binding(5) var<storage, read_write> output: array<${storageType}>;

var<workgroup> queries: array<f32, ${QUERY_TILE * 12}>;
var<workgroup> position_queries: array<f32, ${QUERY_TILE * 4}>;
var<workgroup> keys: array<f32, ${KEY_TILE * 12}>;
var<workgroup> values: array<f32, ${KEY_TILE * valueHeadDim}>;
var<workgroup> probabilities: array<f32, ${QUERY_TILE * KEY_TILE}>;
// Separate reductions prevent a fast lane overwriting a maximum before its
// row peers read it. The final tile barrier protects both arrays on reuse.
var<workgroup> maxima: array<f32, ${WORKGROUP_SIZE}>;
var<workgroup> sums: array<f32, ${WORKGROUP_SIZE}>;

${shaderHelper.mainStart(WORKGROUP_SIZE)}
  // Backend dispatch normalization may add whole workgroups. This condition
  // is workgroup-uniform; individual padded query lanes never return early.
  if (workgroup_index >= ${workgroups}u) { return; }
  let tid = local_id.x;
  let row = tid / ${ROW_LANES}u;
  let lane = tid % ${ROW_LANES}u;
  let query_start = (workgroup_index % ${queryTiles}u) * ${QUERY_TILE}u;
  let head_batch = workgroup_index / ${queryTiles}u;
  let head = head_batch / ${batch}u;
  let query_index = query_start + row;
  let sequence_base = head_batch * ${length}u;
  let position_base = head * ${4 * (2 * length - 1)}u;
  // WGSL rejects nonfinite constant expressions. Every row has valid keys;
  // this finite minimum gives exactly zero exponentials for padding/seeding.
  let negative_infinity = -3.4028234e38;

  for (var index = tid; index < ${QUERY_TILE * 12}u; index += ${WORKGROUP_SIZE}u) {
    let source_row = query_start + index / 12u;
    var value = 0.0;
    if (source_row < ${length}u) {
      value = f32(q[(sequence_base + source_row) * 12u + index % 12u]);
    }
    queries[index] = value;
  }
  if (tid < ${QUERY_TILE * 4}u) {
    let source_row = query_start + tid / 4u;
    var value = 0.0;
    if (source_row < ${length}u) {
      value = f32(p[(sequence_base + source_row) * 4u + tid % 4u]);
    }
    position_queries[tid] = value;
  }
  workgroupBarrier();

  var accumulated: array<f32, ${valueSlots}>;
  var running_max = negative_infinity;
  var denominator = 0.0;
  for (var key_start = 0u; key_start < ${length}u; key_start += ${KEY_TILE}u) {
    // Channel-major tile loads preserve the source transposed-K addressing.
    for (var index = tid; index < ${KEY_TILE * 12}u; index += ${WORKGROUP_SIZE}u) {
      let channel = index / ${KEY_TILE}u;
      let source_key = key_start + index % ${KEY_TILE}u;
      var value = 0.0;
      if (source_key < ${length}u) {
        value = f32(k[(head_batch * 12u + channel) * ${length}u + source_key]);
      }
      keys[index] = value;
    }
    for (var index = tid; index < ${KEY_TILE * valueHeadDim}u; index += ${WORKGROUP_SIZE}u) {
      let source_key = key_start + index / ${valueHeadDim}u;
      var value = 0.0;
      if (source_key < ${length}u) {
        value = f32(v[(sequence_base + source_key) * ${valueHeadDim}u + index % ${valueHeadDim}u]);
      }
      values[index] = value;
    }
    workgroupBarrier();

    var logits: array<f32, ${KEY_TILE / ROW_LANES}>;
    var lane_max = negative_infinity;
    for (var slot = 0u; slot < ${KEY_TILE / ROW_LANES}u; slot++) {
      let column = lane + slot * ${ROW_LANES}u;
      let key_index = key_start + column;
      var logit = negative_infinity;
      if (key_index < ${length}u) {
        // Padded queries use finite dummy logits and participate in all
        // barriers without reading an invalid query or relative position.
        logit = 0.0;
        if (query_index < ${length}u) {
          var content = 0.0;
          for (var channel = 0u; channel < 12u; channel++) {
            content = fma(queries[row * 12u + channel], keys[channel * ${KEY_TILE}u + column], content);
          }
          let relative_index = ${length - 1}u - query_index + key_index;
          var positional = 0.0;
          for (var channel = 0u; channel < 4u; channel++) {
            positional = fma(position_queries[row * 4u + channel],
              f32(pos[position_base + channel * ${2 * length - 1}u + relative_index]), positional);
          }
          let content_rounded = ${roundHalf('content')};
          let positional_rounded = ${roundHalf('positional')};
          logit = ${roundHalf('content_rounded + positional_rounded')};
        }
      }
      logits[slot] = logit;
      lane_max = max(lane_max, logit);
    }
    maxima[tid] = lane_max;
    workgroupBarrier();
    for (var stride = ${ROW_LANES / 2}u; stride > 0u; stride /= 2u) {
      if (lane < stride) {
        maxima[tid] = max(maxima[tid], maxima[tid + stride]);
      }
      workgroupBarrier();
    }
    let new_max = max(running_max, maxima[row * ${ROW_LANES}u]);
    let rescale = exp(running_max - new_max);
    var lane_sum = 0.0;
    for (var slot = 0u; slot < ${KEY_TILE / ROW_LANES}u; slot++) {
      let column = lane + slot * ${ROW_LANES}u;
      let weight = exp(logits[slot] - new_max);
      probabilities[row * ${KEY_TILE}u + column] = ${roundHalf('weight')};
      lane_sum += weight;
    }
    sums[tid] = lane_sum;
    workgroupBarrier();
    for (var stride = ${ROW_LANES / 2}u; stride > 0u; stride /= 2u) {
      if (lane < stride) {
        sums[tid] += sums[tid + stride];
      }
      workgroupBarrier();
    }
    denominator = denominator * rescale + sums[row * ${ROW_LANES}u];
    running_max = new_max;
    for (var slot = 0u; slot < ${valueSlots}u; slot++) {
      let channel = lane + slot * ${ROW_LANES}u;
      var tile_value = 0.0;
      for (var column = 0u; column < ${KEY_TILE}u; column++) {
        tile_value = fma(probabilities[row * ${KEY_TILE}u + column],
          values[column * ${valueHeadDim}u + channel], tile_value);
      }
      accumulated[slot] = accumulated[slot] * rescale + tile_value;
    }
    // No key/probability/value/reduction tile can be reused until all its
    // readers finish, including lanes belonging to padded query rows.
    workgroupBarrier();
  }
  if (query_index < ${length}u) {
    let inverse_denominator = 1.0 / denominator;
    for (var slot = 0u; slot < ${valueSlots}u; slot++) {
      let channel = lane + slot * ${ROW_LANES}u;
      output[(sequence_base + query_index) * ${valueHeadDim}u + channel] =
        ${storageType}(accumulated[slot] * inverse_denominator);
    }
  }
}`,
  };
}

/** Use the normal backend path so each consumer is dispatched/audited/captured. */
export function zipRelativeAttention(
  context: ComputeContextLike,
  attributes: ZipRelativeAttentionAttributes,
): void {
  context.compute(createZipRelativeAttentionProgram(context.inputs, attributes));
}
