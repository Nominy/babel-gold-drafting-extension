import type {
  ComputeContextLike,
  ProgramInfoLike,
  TensorViewLike,
} from './zipenhancer-webgpu-types';

export interface ZipSwooshAttributes {
  readonly steps: readonly {
    readonly source: string;
    readonly op: 'Add' | 'Sub' | 'Mul' | 'Div' | 'Exp' | 'Log' | 'Cast' | 'Equal' | 'Where';
    readonly inputs: readonly number[];
    readonly to?: number;
  }[];
  readonly output: number;
  readonly outputType: 1 | 10;
}

type ExpressionType = 'float' | 'bool';
const WORKGROUP_SIZE = 64;
const KERNEL_ABI = 'zip-swoosh-expression-wgsl-v1';
const LANES = ['x', 'y', 'z', 'w'] as const;

function requireCondition(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(`ZipSwoosh ${message}`);
}

/** Compile the original source DAG, not an algebraically simplified activation. */
export function createZipSwooshProgram(
  inputs: readonly TensorViewLike[],
  attributes: ZipSwooshAttributes,
): ProgramInfoLike {
  requireCondition(inputs.length >= 1 && inputs.length <= 7, 'requires one activation and at most six GPU scalars.');
  requireCondition(attributes && typeof attributes === 'object', 'requires expression attributes.');
  requireCondition(Array.isArray(attributes.steps) && attributes.steps.length > 0, 'requires a nonempty expression DAG.');
  requireCondition(attributes.outputType === 1 || attributes.outputType === 10, 'requires float32 or float16 output storage.');
  requireCondition(Number.isSafeInteger(attributes.output) && attributes.output >= 0 && attributes.output < attributes.steps.length,
    'has an invalid output reference.');

  // Own all metadata used by the closures and cache key; callers cannot mutate
  // a compiled program into a different expression or output allocation.
  const inputShapes: number[][] = [];
  const inputTypes: number[] = [];
  let elements = 0;
  for (let index = 0; index < inputs.length; ++index) {
    const input = inputs[index];
    requireCondition(input && (input.dataType === 1 || input.dataType === 10), `input ${index} requires float32 or float16 storage.`);
    requireCondition(Array.isArray(input.dims), `input ${index} requires static dimensions.`);
    let count = 1;
    for (const dimension of input.dims) {
      requireCondition(Number.isSafeInteger(dimension) && dimension > 0, `input ${index} has an invalid dimension.`);
      count *= dimension;
      requireCondition(Number.isSafeInteger(count) && count <= Math.floor(0xffffffff / (input.dataType === 10 ? 2 : 4)),
        `input ${index} exceeds safe WGSL byte addressing.`);
    }
    if (index === 0) {
      elements = count;
    } else {
      requireCondition(count === 1 && input.dims.length <= inputShapes[0].length,
        `input ${index} must broadcast as a scalar without changing the activation shape.`);
    }
    inputShapes.push([...input.dims]);
    inputTypes.push(input.dataType);
  }
  const outputType = attributes.outputType;
  requireCondition(elements <= Math.floor(0xffffffff / (outputType === 10 ? 2 : 4)),
    'output exceeds safe WGSL byte addressing.');
  const outputIndex = attributes.output;
  const expressionTypes: ExpressionType[] = [];
  const steps: { op: ZipSwooshAttributes['steps'][number]['op']; inputs: number[]; to?: number }[] = [];
  const sources = new Set<string>();
  for (let index = 0; index < attributes.steps.length; ++index) {
    const step: ZipSwooshAttributes['steps'][number] = attributes.steps[index];
    requireCondition(step && typeof step === 'object', `step ${index} is invalid.`);
    requireCondition(typeof step.source === 'string' && step.source.length > 0 && !sources.has(step.source),
      `step ${index} requires a unique source node name.`);
    sources.add(step.source);
    requireCondition(Array.isArray(step.inputs), `step ${index} requires input references.`);
    for (const ref of step.inputs) {
      requireCondition(Number.isSafeInteger(ref) && ref >= -inputs.length && ref < index,
        `step ${index} has a missing, forward or cyclic input reference.`);
    }
    requireCondition(step.op === 'Cast' || step.to === undefined, `step ${index} has a Cast attribute on a non-Cast operation.`);
    const types = step.inputs.map((ref): ExpressionType => ref < 0 ? 'float' : expressionTypes[ref]);
    let resultType: ExpressionType;
    switch (step.op) {
      case 'Add':
      case 'Sub':
      case 'Mul':
      case 'Div':
        requireCondition(types.length === 2 && types.every((type) => type === 'float'),
          `step ${index} requires two floating-point operands.`);
        resultType = 'float';
        break;
      case 'Exp':
      case 'Log':
        requireCondition(types.length === 1 && types[0] === 'float', `step ${index} requires one floating-point operand.`);
        resultType = 'float';
        break;
      case 'Cast':
        requireCondition(types.length === 1 && (step.to === 1 || step.to === 9 || step.to === 10),
          `step ${index} supports only unary Cast to float32, bool or float16.`);
        resultType = step.to === 9 ? 'bool' : 'float';
        break;
      case 'Equal':
        requireCondition(types.length === 2 && types[0] === types[1], `step ${index} requires Equal operands of the same type.`);
        resultType = 'bool';
        break;
      case 'Where':
        requireCondition(types.length === 3 && types[0] === 'bool' && types[1] === types[2],
          `step ${index} requires a boolean condition and matching Where branches.`);
        resultType = types[1];
        break;
      default:
        throw new Error(`ZipSwoosh step ${index} has an unsupported operation.`);
    }
    expressionTypes.push(resultType);
    steps.push({ op: step.op, inputs: [...step.inputs], to: step.to });
  }
  requireCondition(expressionTypes[outputIndex] === 'float', 'output must be floating-point.');

  // Every claimed source and binding must contribute to the final activation.
  // This also establishes that the output has the activation's broadcast shape.
  const liveSteps = new Set<number>();
  const liveInputs = new Set<number>();
  const pending = [outputIndex];
  while (pending.length > 0) {
    const index = pending.pop()!;
    if (liveSteps.has(index)) continue;
    liveSteps.add(index);
    for (const ref of steps[index].inputs) {
      if (ref < 0) liveInputs.add(-ref - 1);
      else pending.push(ref);
    }
  }
  requireCondition(liveSteps.size === steps.length && liveInputs.size === inputs.length && liveInputs.has(0),
    'requires all source steps and GPU inputs to contribute to the activation output.');

  const packed = elements % 4 === 0;
  const vectors = Math.ceil(elements / 4);
  const outputStorage = outputType === 10 ? 'f16' : 'f32';
  const floatFromBool = (value: string): string => `select(vec4<f32>(0.0), vec4<f32>(1.0), ${value})`;
  const statements = steps.map((step, index) => {
    const args = step.inputs.map((ref) => ref < 0 ? `input_${-ref - 1}` : `step_${ref}`);
    const firstType = step.inputs[0] < 0 ? 'float' : expressionTypes[step.inputs[0]];
    let expression: string;
    switch (step.op) {
      case 'Add': expression = `${args[0]} + ${args[1]}`; break;
      case 'Sub': expression = `${args[0]} - ${args[1]}`; break;
      case 'Mul': expression = `${args[0]} * ${args[1]}`; break;
      case 'Div': expression = `${args[0]} / ${args[1]}`; break;
      case 'Exp': expression = `exp(${args[0]})`; break;
      case 'Log': expression = `log(${args[0]})`; break;
      case 'Equal': expression = `${args[0]} == ${args[1]}`; break;
      case 'Cast':
        if (step.to === 9) {
          expression = firstType === 'bool' ? args[0] : `${args[0]} != vec4<f32>(0.0)`;
        } else {
          expression = firstType === 'bool' ? floatFromBool(args[0]) : args[0];
          if (step.to === 10 && firstType !== 'bool') expression = `round_to_half(${expression})`;
        }
        break;
      case 'Where':
        // WGSL select operates on numeric values, not boolean vectors.
        expression = expressionTypes[index] === 'bool'
          ? `select(${floatFromBool(args[2])}, ${floatFromBool(args[1])}, ${args[0]}) != vec4<f32>(0.0)`
          : `select(${args[2]}, ${args[1]}, ${args[0]})`;
        break;
    }
    return `  let step_${index}: vec4<${expressionTypes[index] === 'bool' ? 'bool' : 'f32'}> = ${expression};`;
  }).join('\n');
  const declarations = inputTypes.map((type, index) => {
    const scalar = type === 10 ? 'f16' : 'f32';
    const storage = index === 0 && packed ? `vec4<${scalar}>` : scalar;
    return `@group(0) @binding(${index}) var<storage, read> buffer_${index}: array<${storage}>;`;
  }).join('\n');
  const scalarLoads = inputTypes.slice(1).map((_, index) =>
    `  let input_${index + 1}: vec4<f32> = vec4<f32>(f32(buffer_${index + 1}[0u]));`).join('\n');
  const activationLoad = packed
    ? '  let input_0: vec4<f32> = vec4<f32>(buffer_0[vector_index]);'
    : `  let base = vector_index * 4u;\n  var input_0 = vec4<f32>(0.0);\n${LANES.map((lane, index) =>
      `  if (base + ${index}u < ${elements}u) { input_0.${lane} = f32(buffer_0[base + ${index}u]); }`).join('\n')}`;
  const outputStore = packed
    ? `  output[vector_index] = vec4<${outputStorage}>(step_${outputIndex});`
    : LANES.map((lane, index) =>
      `  if (base + ${index}u < ${elements}u) { output[base + ${index}u] = ${outputStorage}(step_${outputIndex}.${lane}); }`).join('\n');
  // Packing/unpacking models an explicit Cast to half without requiring the
  // f16 language extension when all input/output storage is float32.
  const halfCastHelper = steps.some((step) => step.op === 'Cast' && step.to === 10)
    ? `fn round_to_half(value: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(unpack2x16float(pack2x16float(value.xy)), unpack2x16float(pack2x16float(value.zw)));
}`
    : '';

  return {
    name: 'ZipSwoosh',
    shaderCache: {
      // Source names are audit metadata, never WGSL or shader-cache identity.
      hint: JSON.stringify([KERNEL_ABI, WORKGROUP_SIZE, inputShapes, inputTypes, steps, outputIndex, outputType]),
      inputDependencies: inputTypes.map(() => 'dims'),
    },
    getRunData: () => ({
      outputs: [{ dims: inputShapes[0], dataType: outputType }],
      dispatchGroup: { x: Math.ceil(vectors / WORKGROUP_SIZE) },
    }),
    getShaderSource: (shaderHelper) => `
${declarations}
@group(0) @binding(${inputTypes.length}) var<storage, read_write> output: array<${packed ? `vec4<${outputStorage}>` : outputStorage}>;
${halfCastHelper}
${shaderHelper.mainStart(WORKGROUP_SIZE)}
  // Flatten normalized multidimensional backend dispatches, including padding.
  let vector_index = workgroup_index * ${WORKGROUP_SIZE}u + local_id.x;
  if (vector_index >= ${vectors}u) { return; }
${activationLoad}
${scalarLoads}
${statements}
${outputStore}
}`,
  };
}

/** Keep dispatch, source attribution and graph capture on the normal GPU path. */
export function zipSwoosh(context: ComputeContextLike, attributes: ZipSwooshAttributes): void {
  context.compute(createZipSwooshProgram(context.inputs, attributes));
}
