import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';

export const kernelSourceHashes = Object.freeze({
  'webgpu/program-manager.ts': '194b4b5d5afc55402dc840c8db4570945e0fc0651f39cb2d1955208898d5e331',
  'webgpu/ops/binary-op.ts': 'd3d231328b8c9fbc1daab7c550d6468114cd85c83cad4f163c96e06943cf283f',
  'webgpu/ops/common.ts': '2eb7d63a1e932d30ef246d6e01c77503b9705d06fcbbdc6b8c27f3500412a84d',
  'backend-webgpu.ts': '4232f0cb7d18942f11b72f10376b88dbda636980c7abfd0c1da7a7df48fe0735',
  'webgpu/ops/3rd-party/matmul_packed_webgpu.ts': '78aa576c8cd162b38b30e76895453caca4870fbc2a404306ff625b2f869313f4',
  'webgpu/ops/conv-grouped.ts': '23c29bcf97313acbf010e30c85109fe8cd7d8a62fc752b153532c34eb81ef288',
});
export function assertKernelSource(relative, bytes) {
  if (createHash('sha256').update(bytes).digest('hex') !== kernelSourceHashes[relative]) {
    throw new Error(`ORT 1.29 kernel source changed: ${relative}. Revalidate address, precision and profiling contracts before building.`);
  }
}
function replaceOnce(source, anchor, replacement) {
  if (source.split(anchor).length !== 2) throw new Error(`ORT kernel transform contract changed: ${anchor}`);
  return source.replace(anchor, replacement);
}
export function transformBinarySource(source, { profile = false, baseline = false } = {}) {
  const helper = fileURLToPath(new URL('../src/core/webgpu-binary-address.ts', import.meta.url));
  source = `import { specializeBinaryAssignment } from ${JSON.stringify(helper)};\n` + source;
  source = replaceOnce(source, '  additionalImplementation?: string,\n) => {', '  additionalImplementation?: string,\n  specializeAddresses = false,\n) => {');
  source = replaceOnce(source, '  return `\n        ${shaderHelper.registerUniform', `  if (specializeAddresses && doBroadcast) {\n    assignment = specializeBinaryAssignment(dimsA, dimsB, dimsOutput, vectorize, sharedDimensionDivisibleBy4, expressionScalar, expressionVector);\n  }\n\n  return \`\n        \${shaderHelper.registerUniform`);
  source = replaceOnce(source, '  return {\n    name,\n    shaderCache:', `  const specializeAddresses = ${!baseline} && ['Add', 'Sub', 'Mul', 'Div'].includes(name) && a.dataType === DataType.float && b.dataType === DataType.float && outputDataType === DataType.float;\n  return {\n    name,\n    shaderCache:`);
  source = replaceOnce(source, "inputDependencies: ['rank', 'rank'],", "inputDependencies: specializeAddresses ? ['dims', 'dims'] : ['rank', 'rank'],");
  source = replaceOnce(source, '        additionalImplementation,\n      ),', '        additionalImplementation,\n        specializeAddresses,\n      ),');
  if (profile) {
    source = replaceOnce(source, '): ProgramInfo => {', "): ProgramInfo & { getBaselineShaderSource?: ProgramInfo['getShaderSource'] } => {");
    source = replaceOnce(source, '    getRunData: () => ({', `    getBaselineShaderSource: (shaderHelper) => createBinaryOpProgramShader(shaderHelper, aDims, bDims, outputShape, vectorize, isBroadcast, sharedDimensionDivisibleBy4, funcCall, a.dataType, b.dataType, outputDataType, additionalImplementation, false),\n    getRunData: () => ({`);
  }
  return source;
}
export function transformProgramManagerSource(source, { profile = false } = {}) {
  const helper = fileURLToPath(new URL('../src/core/webgpu-fp32-accumulation.ts', import.meta.url));
  source = `import { promoteFp16Accumulation } from ${JSON.stringify(helper)};\n` + source;
  source = replaceOnce(source, 'const userCode = programInfo.getShaderSource(shaderHelper);', 'const userCode = promoteFp16Accumulation(programInfo.getShaderSource(shaderHelper));');
  if (profile) {
    source = replaceOnce(source, '    const shaderModule = device.createShaderModule', `    const baselineHelper = createShaderHelper(normalizedDispatchGroupSize, this.backend.device.limits);\n    const baselineUserCode = (programInfo as ProgramInfo & { getBaselineShaderSource?: ProgramInfo['getShaderSource'] }).getBaselineShaderSource?.(baselineHelper);\n    const baselineCode = baselineUserCode === undefined ? code : \`\${enableDirectives.join('\\n')}\\n\${baselineHelper.additionalImplementations}\\n\${promoteFp16Accumulation(baselineUserCode)}\`;\n    const shaderModule = device.createShaderModule`);
    source = replaceOnce(source, 'return { programInfo, computePipeline, uniformVariablesInfo: shaderHelper.variablesInfo };', 'return { programInfo, computePipeline, uniformVariablesInfo: shaderHelper.variablesInfo, code, baselineCode } as Artifact;');
  }
  return source;
}
export function transformBackendProfileSource(source) {
  source = replaceOnce(source, '    let uniformBufferBinding: GPUBindingResource | undefined;', '    let profilerPackedUniforms: ArrayBuffer | undefined;\n    let uniformBufferBinding: GPUBindingResource | undefined;');
  source = replaceOnce(source, '      const arrayBuffer = new ArrayBuffer(currentOffset);', '      const arrayBuffer = new ArrayBuffer(currentOffset);\n      profilerPackedUniforms = arrayBuffer;');
  source = replaceOnce(source, "    if (this.queryType !== 'none' || this.sessionStatus === 'capturing') {", `    const profilerConfigId = (globalThis as typeof globalThis & { __babelKernelProfiler?: { capture: (context: unknown) => number } }).__babelKernelProfiler?.capture({ backend: this, artifact, key, inputDatas, outputDatas, inputTensorViews, outputTensorViews, programUniforms, packedUniforms: profilerPackedUniforms, dispatchGroup: normalizedDispatchGroup, kernelId: this.currentKernelId });\n    if (this.queryType !== 'none' || this.sessionStatus === 'capturing') {`);
  source = replaceOnce(source, '        programName: artifact.programInfo.name,', '        profilerConfigId,\n        programName: artifact.programInfo.name,');
  source = replaceOnce(source, '      const pendingKernelInfo: PendingKernelInfo = {', '      const pendingKernelInfo: PendingKernelInfo & { profilerConfigId?: number } = {');
  source = replaceOnce(source, '              version: 1,', '              ...{ profilerConfigId: (pendingKernelInfo as PendingKernelInfo & { profilerConfigId?: number }).profilerConfigId },\n              version: 1,');
  return source;
}
