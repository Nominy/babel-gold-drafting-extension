// Literal address arithmetic only: FP32 operations and their evaluation order stay upstream.
// A dims cache dependency is required whenever these expressions are selected.
export function broadcastOffsetExpression(input: readonly number[], output: readonly number[], index: string): string {
  if (input.length > output.length) throw new Error('Invalid broadcast rank');
  const aligned = [...Array(output.length - input.length).fill(1), ...input];
  let inputStride = 1, outputStride = 1;
  const groups: { size: number; inputStride: number; outputStride: number }[] = [];
  let group: typeof groups[number] | null = null;
  for (let axis = output.length - 1; axis >= 0; axis--) {
    const dimension = aligned[axis], extent = output[axis];
    if (!Number.isSafeInteger(extent) || extent < 1 || (dimension !== 1 && dimension !== extent)) throw new Error('Invalid broadcast dimension');
    if (dimension > 1) {
      if (group) group.size *= extent;
      else { group = { size: extent, inputStride, outputStride }; groups.unshift(group); }
    } else if (extent > 1) {
      group = null;
    }
    inputStride *= dimension; outputStride *= extent;
  }
  if (inputStride === outputStride) return `(${index})`;
  // Fold whole contiguous non-broadcast runs instead of decomposing every coordinate.
  // Singleton axes of extent one neither contribute an index nor interrupt a run.
  return groups.length ? groups.map(({ size, inputStride, outputStride: stride }) => {
    const coordinate = stride === 1 ? `(${index})` : `((${index}) / ${stride}u)`;
    const bounded = stride * size === outputStride ? coordinate : `(${coordinate} % ${size}u)`;
    return inputStride === 1 ? bounded : `(${bounded} * ${inputStride}u)`;
  }).join(' + ') : '0u';
}

export function specializeBinaryAssignment(
  dimsA: readonly number[], dimsB: readonly number[], dimsOutput: readonly number[],
  vectorize: boolean, sharedDimensionDivisibleBy4: boolean,
  expressionScalar: (a: string, b: string) => string,
  expressionVector: (a: string, b: string) => string,
): string {
  const size = (dims: readonly number[]) => dims.reduce((a, b) => a * b, 1);
  if (vectorize) {
    const value = (name: string, dims: readonly number[]) => {
      if (size(dims) === 1) return `vec4<f32>(${name}[0u].x)`;
      if (size(dims) === size(dimsOutput)) return `${name}[global_idx]`;
      const offset = broadcastOffsetExpression(dims, dimsOutput, 'global_idx * 4u');
      return sharedDimensionDivisibleBy4 || dims[dims.length - 1] % 4 === 0
        ? `${name}[(${offset}) / 4u]`
        : `vec4<f32>(${name}[(${offset}) / 4u][(${offset}) % 4u])`;
    };
    return `outputData[global_idx] = ${expressionVector(value('aData', dimsA), value('bData', dimsB))};`;
  }
  return [0, 1, 2, 3].map((lane) => {
    const index = `global_idx * 4u + ${lane}u`;
    const a = broadcastOffsetExpression(dimsA, dimsOutput, index), b = broadcastOffsetExpression(dimsB, dimsOutput, index);
    // Padding lanes are not logical tensor elements, and must not form invalid broadcast indices.
    return `if (${index} < ${size(dimsOutput)}u) { let offsetA = ${a}; let offsetB = ${b}; outputData[global_idx][${lane}] = ${expressionScalar('aData[offsetA / 4u][offsetA % 4u]', 'bData[offsetB / 4u][offsetB % 4u]')}; }`;
  }).join('\n');
}
