import * as ort from 'onnxruntime-web/webgpu';
import artifact from '../src/core/audio-enhancement-model.json';
import { decodeEnhancementWav, encodeEnhancementWav, enhanceZipSamples } from '../src/core/audio-enhancement-dsp';
import { bindPlacementGraph, GpuRunAudit, type PlacementGraph, type GraphPlacementDiagnostic } from '../src/core/local-gpu-placement';
import type { Artifact, GpuData, ProgramUniform, TensorInfo } from '../node_modules/onnxruntime-web/lib/wasm/jsep/webgpu/types';
import type { WebGpuBackend } from '../node_modules/onnxruntime-web/lib/wasm/jsep/backend-webgpu';

import type { CampaignIdentity, CampaignTensor, CampaignConfiguration, CampaignSpectrum, CampaignWave, CampaignOccurrence, CaptureCampaign } from './zip-profile-campaign.mjs';
type TensorDescription = CampaignTensor;
type Configuration = CampaignConfiguration;
type ReplayConfiguration = Pick<Configuration, 'id' | 'program' | 'cacheKey' | 'shader' | 'baselineShader' | 'inputs' | 'outputs' | 'uniforms' | 'uniformBytes' | 'uniformFile' | 'uniformSha256' | 'dispatchGroup' | 'count'>;
interface CaptureContext { backend: WebGpuBackend; artifact: Artifact & { code: string; baselineCode: string }; key: string; inputDatas: GpuData[];
  inputTensorViews: readonly TensorInfo[]; outputTensorViews: readonly TensorInfo[]; programUniforms?: readonly ProgramUniform[];
  packedUniforms?: ArrayBuffer; dispatchGroup: [number, number, number] }
interface Options { captureBudgetMiB: number; repeats: number; warmups: number; audioCount: number; suiteOnly: boolean;
  identity: CampaignIdentity; sourcePaths: string[]; recover: boolean }
const host = globalThis as typeof globalThis & { __babelKernelProfiler?: { capture: (context: CaptureContext) => number }; runZipKernelProfiler?: (options: Options) => Promise<unknown> };
const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
const bytesPerElement: Record<number, number> = { 1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8, 12: 4, 13: 8, 16: 2 };
const configurations: Configuration[] = [], configurationKeys = new Map<string, number>();
let captureBytes = 0, captureBudget = 256 * 1024 * 1024, measurement = true, submittedDispatches = 0, captureActive = false;
const captures: { config: Configuration; buffers: GPUBuffer[]; uniform: ArrayBuffer | null }[] = [];
const occurrences: CampaignOccurrence[] = [];
const savedArtifactNames = new Set<string>();
let currentSpectrumId: number | null = null, introductionPhase: 'initialization' | 'inference' = 'initialization', verifiedFullRecovery = false;
async function captureSha256(data: ArrayBuffer | Uint8Array<ArrayBuffer>) {
  const digest = await crypto.subtle.digest('SHA-256', data);
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
}
function tensorDescription(tensor: TensorInfo): TensorDescription {
  const elements = tensor.dims.reduce((a: number, b: number) => a * b, 1);
  const width = bytesPerElement[tensor.dataType];
  if (!width) throw new Error(`Unsupported replay dtype ${tensor.dataType}`);
  return { dims: [...tensor.dims], dataType: tensor.dataType, elements, logicalBytes: elements * width, bufferBytes: Math.max(16, Math.ceil(elements * width / 16) * 16) };
}
host.__babelKernelProfiler = {
  capture({ backend, artifact: compiled, key, inputDatas, inputTensorViews, outputTensorViews, programUniforms, packedUniforms, dispatchGroup }) {
    if (!captureActive) return -1;
    submittedDispatches++;
    const inputs = inputTensorViews.map(tensorDescription), outputs = outputTensorViews.map(tensorDescription);
    const identity = JSON.stringify([compiled.code, inputs, outputs, programUniforms ?? [], dispatchGroup]);
    let id = configurationKeys.get(identity);
    if (id === undefined) {
      if (verifiedFullRecovery) throw new Error('Recovery encountered a shader/configuration absent from the verified full-model manifest');
      id = configurations.length; configurationKeys.set(identity, id);
      configurations.push({ id, program: compiled.programInfo.name, cacheKey: key, shader: compiled.code, baselineShader: compiled.baselineCode,
        inputs, outputs, uniforms: structuredClone(programUniforms ?? []), uniformBytes: packedUniforms?.byteLength ?? 0, dispatchGroup: [...dispatchGroup],
        count: 0, gpuDurationNs: 0, sources: {}, dataOrigin: 'actual full intermediate input buffers captured before dispatch; no synthetic tensor values',
        captured: false, captureComplete: false, introducedPhase: introductionPhase, introducingSpectrumId: currentSpectrumId });
    }
    const config = configurations[id];
    const size = inputs.reduce((n, t) => n + t.bufferBytes, 0) + config.uniformBytes;
    if (!config.captured && (captureBytes === 0 || captureBytes + size <= captureBudget)) {
      config.captured = true; captureBytes += size;
      backend.endComputePass();
      const snapshot = (source: GPUBuffer, size: number, offset = 0) => {
        const staging = backend.device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        backend.getCommandEncoder().copyBufferToBuffer(source, offset, staging, 0, size);
        return staging;
      };
      captures.push({ config, buffers: inputDatas.map((data, index) => snapshot(data.buffer, inputs[index].bufferBytes)),
        uniform: packedUniforms ?? null });
    }
    return id;
  },
};
async function save(name: string, data: ArrayBuffer | Uint8Array | string) {
  if (savedArtifactNames.has(name)) throw new Error(`Duplicate logical artifact save: ${name}`);
  savedArtifactNames.add(name);
  const bytes = typeof data === 'string' ? new TextEncoder().encode(data) : data instanceof Uint8Array ? data : new Uint8Array(data);
  // Bound every HTTP/CDP POST body, including JSON inventories and large tensors.
  // The server assembles exact consecutive ranges; no sample or byte is omitted.
  const chunkBytes = 4 * 1024 * 1024;
  let offset = 0;
  do {
    const end = Math.min(bytes.byteLength, offset + chunkBytes);
    const response = await fetch(`/artifact/${name}?offset=${offset}&total=${bytes.byteLength}`, {
      method: 'POST', body: bytes.subarray(offset, end),
    });
    if (!response.ok) throw new Error(`Artifact chunk write failed: ${name}@${offset}: ${await response.text()}`);
    const acknowledgement: unknown = await response.json();
    if (!acknowledgement || typeof acknowledgement !== 'object' ||
        !('nextOffset' in acknowledgement) || acknowledgement.nextOffset !== end ||
        !('total' in acknowledgement) || acknowledgement.total !== bytes.byteLength ||
        !('complete' in acknowledgement) || acknowledgement.complete !== (end === bytes.byteLength)) {
      throw new Error(`Artifact upload acknowledgement mismatch: ${name}@${offset}`);
    }
    offset = end;
  } while (offset < bytes.byteLength);
}
async function drainCaptures(device: GPUDevice) {
  await device.queue.onSubmittedWorkDone();
  for (const capture of captures.splice(0)) {
    for (let index = 0; index < capture.buffers.length; index++) {
      const buffer = capture.buffers[index]; await buffer.mapAsync(GPUMapMode.READ);
      const raw = buffer.getMappedRange();
      const name = `config-${capture.config.id}-input-${index}.bin`;
      await save(name, raw);
      capture.config.inputs[index].file = name;
      capture.config.inputs[index].sha256 = await captureSha256(raw);
      if (capture.config.inputs[index].dataType === 1) {
        const values = new Float32Array(raw, 0, capture.config.inputs[index].elements);
        let min = Infinity, max = -Infinity, finite = 0, nonfinite = 0;
        for (const value of values) { if (Number.isFinite(value)) { min = Math.min(min, value); max = Math.max(max, value); finite++; } else nonfinite++; }
        capture.config.inputs[index].range = { min: finite ? min : null, max: finite ? max : null, finite, nonfinite,
          sample: Array.from(values.subarray(0, Math.min(32, values.length))).map(v => Number.isFinite(v) ? v : String(v)) };
      }
      buffer.unmap(); buffer.destroy();
    }
    if (capture.uniform) {
      const name = `config-${capture.config.id}-uniform.bin`;
      await save(name, capture.uniform);
      capture.config.uniformFile = name;
      capture.config.uniformSha256 = await captureSha256(capture.uniform);
    }
    capture.config.captureComplete = true;
  }
  captureBytes = 0;
}
async function rawBuffer(device: GPUDevice, file: string, size: number, usage: number, expectedSha256: string | undefined) {
  const response = await fetch(`/artifact/${file}`);
  if (!response.ok) throw new Error(`Missing captured buffer ${file}`);
  const data = await response.arrayBuffer();
  if (data.byteLength !== size) throw new Error(`Replay size mismatch ${file}`);
  if (!expectedSha256 || await captureSha256(data) !== expectedSha256) throw new Error(`Replay captured-data identity failed: ${file}`);
  const buffer = device.createBuffer({ size, usage: usage | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}
function validateConfiguration(value: unknown): ReplayConfiguration {
  if (!value || typeof value !== 'object' || !('id' in value) || !('program' in value) || !('cacheKey' in value) ||
      !('shader' in value) || !('baselineShader' in value) || !('inputs' in value) || !('outputs' in value) ||
      !('dispatchGroup' in value) || !('count' in value) || !('uniformBytes' in value) ||
      typeof value.id !== 'number' || !Number.isSafeInteger(value.id) || value.id < 0 ||
      typeof value.count !== 'number' || !Number.isSafeInteger(value.count) || value.count < 0 ||
      typeof value.program !== 'string' || typeof value.cacheKey !== 'string' || typeof value.shader !== 'string' || typeof value.baselineShader !== 'string' ||
      !Array.isArray(value.inputs) || !Array.isArray(value.outputs) || !Array.isArray(value.dispatchGroup) || value.dispatchGroup.length !== 3 ||
      value.dispatchGroup.some(n => !Number.isSafeInteger(n) || n < 1) || typeof value.uniformBytes !== 'number' ||
      !Number.isSafeInteger(value.uniformBytes) || value.uniformBytes < 0 || value.uniformBytes % 4 !== 0) throw new Error('Invalid replay configuration');
  const tensors = (items: unknown[], requireFile: boolean): TensorDescription[] => items.map(item => {
    if (!item || typeof item !== 'object' || !('dims' in item) || !('dataType' in item) ||
        !('bufferBytes' in item) || !('logicalBytes' in item) || !Array.isArray(item.dims) ||
        item.dims.some(n => !Number.isSafeInteger(n) || n < 1) || typeof item.dataType !== 'number') throw new Error('Invalid replay tensor');
    const description = tensorDescription({ dims: item.dims, dataType: item.dataType });
    if (description.bufferBytes !== item.bufferBytes || description.logicalBytes !== item.logicalBytes) throw new Error('Replay tensor byte size mismatch');
    if (requireFile) {
      if (!('file' in item) || typeof item.file !== 'string' || !/^config-\d+-input-\d+\.bin$/.test(item.file)) throw new Error('Invalid replay tensor file');
      description.file = item.file;
      if (!('sha256' in item) || typeof item.sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(item.sha256)) throw new Error('Replay tensor lacks its original capture hash');
      description.sha256 = item.sha256;
    }
    return description;
  });
  let uniformFile: string | undefined, uniformSha256: string | undefined;
  if (value.uniformBytes > 0) {
    if (!('uniformFile' in value) || typeof value.uniformFile !== 'string' || !/^config-\d+-uniform\.bin$/.test(value.uniformFile)) throw new Error('Invalid replay uniform file');
    uniformFile = value.uniformFile;
    if (!('uniformSha256' in value) || typeof value.uniformSha256 !== 'string' || !/^[a-f0-9]{64}$/.test(value.uniformSha256)) throw new Error('Replay uniform lacks its original capture hash');
    uniformSha256 = value.uniformSha256;
  }
  return { id: value.id, program: value.program, cacheKey: value.cacheKey, shader: value.shader, baselineShader: value.baselineShader,
    inputs: tensors(value.inputs, true), outputs: tensors(value.outputs, false), dispatchGroup: [value.dispatchGroup[0], value.dispatchGroup[1], value.dispatchGroup[2]],
    count: value.count, uniformBytes: value.uniformBytes, uniformFile, uniformSha256, uniforms: [] };
}

async function replay(device: GPUDevice, config: ReplayConfiguration, repeats: number, warmups: number) {
  const retained: GPUBuffer[] = [];
  device.pushErrorScope('validation');
  try {
    const inputs = [];
    for (const input of config.inputs) {
      if (!input.file) throw new Error(`Uncaptured input for config ${config.id}`);
      inputs.push(await rawBuffer(device, input.file, input.bufferBytes, GPUBufferUsage.STORAGE, input.sha256));
    }
    retained.push(...inputs);
    const uniform = config.uniformFile ? await rawBuffer(device, config.uniformFile, config.uniformBytes, GPUBufferUsage.UNIFORM, config.uniformSha256) : null;
    if (uniform) retained.push(uniform);
    const variants: { pipeline: GPUComputePipeline; outputs: GPUBuffer[]; bindGroup: GPUBindGroup }[] = [];
    for (const code of [config.baselineShader, config.shader]) {
      const module = device.createShaderModule({ code });
      const info = await module.getCompilationInfo();
      if (info.messages.some(m => m.type === 'error')) throw new Error(`Replay WGSL compilation failed: ${JSON.stringify(info.messages)}`);
      const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
      const outputs = config.outputs.map(t => device.createBuffer({ size: t.bufferBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }));
      retained.push(...outputs);
      const entries = [...inputs, ...outputs, ...(uniform ? [uniform] : [])].map((buffer, binding) => ({ binding, resource: { buffer } }));
      const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
      variants.push({ pipeline, outputs, bindGroup });
    }
    const dispatch = (encoder: GPUCommandEncoder, variant: typeof variants[number], timestampWrites?: GPUComputePassTimestampWrites) => {
      const pass = encoder.beginComputePass(timestampWrites ? { timestampWrites } : {});
      pass.setPipeline(variant.pipeline); pass.setBindGroup(0, variant.bindGroup); pass.dispatchWorkgroups(...config.dispatchGroup); pass.end();
    };
    for (let i = 0; i < warmups; i++) { const encoder = device.createCommandEncoder(); for (const v of variants) dispatch(encoder, v); device.queue.submit([encoder.finish()]); }
    await device.queue.onSubmittedWorkDone();
    const query = device.createQuerySet({ type: 'timestamp', count: repeats * 4 });
    const resolved = device.createBuffer({ size: repeats * 32, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({ size: repeats * 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    retained.push(resolved, readback);
    const encoder = device.createCommandEncoder();
    for (let i = 0; i < repeats; i++) {
      // Alternate order to avoid attributing warm-cache advantage to one variant.
      for (const v of i % 2 ? [1, 0] : [0, 1]) dispatch(encoder, variants[v], { querySet: query, beginningOfPassWriteIndex: i * 4 + v * 2, endOfPassWriteIndex: i * 4 + v * 2 + 1 });
    }
    encoder.resolveQuerySet(query, 0, repeats * 4, resolved, 0); encoder.copyBufferToBuffer(resolved, 0, readback, 0, repeats * 32);
    device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
    const timestamps = new BigUint64Array(readback.getMappedRange());
    const durations = [0, 1].map(v => Array.from({ length: repeats }, (_, i) => Number(timestamps[i * 4 + v * 2 + 1] - timestamps[i * 4 + v * 2])));
    readback.unmap(); query.destroy();
    const comparisons = [];
    for (let index = 0; index < config.outputs.length; index++) {
      const bytes = config.outputs[index].logicalBytes;
      const words = Math.ceil(bytes / 4);
      const counters = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const result = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      retained.push(counters, result);
      const tailMask = bytes % 4 === 0 ? '0xffffffffu' : `0x${((2 ** ((bytes % 4) * 8)) - 1).toString(16)}u`;
      const groupX = Math.min(Math.ceil(words / 256), device.limits.maxComputeWorkgroupsPerDimension);
      const groupY = Math.ceil(words / (groupX * 256));
      const compareModule = device.createShaderModule({ code: `@group(0) @binding(0) var<storage, read> a: array<u32>; @group(0) @binding(1) var<storage, read> b: array<u32>; @group(0) @binding(2) var<storage, read_write> count: atomic<u32>; @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>) { let offset = id.x + id.y * ${groupX * 256}u; if (offset >= ${words}u) { return; } let mask = select(0xffffffffu, ${tailMask}, offset == ${words - 1}u); if ((a[offset] & mask) != (b[offset] & mask)) { atomicAdd(&count, 1u); } }` });
      const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: compareModule, entryPoint: 'main' } });
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [variants[0].outputs[index], variants[1].outputs[index], counters].map((buffer, binding) => ({ binding, resource: { buffer } })) });
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(groupX, groupY); pass.end(); encoder.copyBufferToBuffer(counters, 0, result, 0, 16); device.queue.submit([encoder.finish()]);
      await result.mapAsync(GPUMapMode.READ); const mismatchWords = new Uint32Array(result.getMappedRange())[0]; result.unmap();
      comparisons.push({ output: index, logicalBytes: bytes, mismatchWords, bitExact: mismatchWords === 0 });
    }
    const error = await device.popErrorScope(); if (error) throw new Error(error.message);
    const mean = durations.map(values => values.reduce((a, b) => a + b, 0) / values.length);
    return { id: config.id, repeats, warmups, baselineNs: mean[0], optimizedNs: mean[1], baselineSamplesNs: durations[0], optimizedSamplesNs: durations[1],
      baselineWeightedNs: mean[0] * config.count, optimizedWeightedNs: mean[1] * config.count, changedShader: config.shader !== config.baselineShader, comparisons };
  } finally { for (const buffer of retained) buffer.destroy(); }
}

host.runZipKernelProfiler = async (options: Options) => {
  captureBudget = options.captureBudgetMiB * 1024 * 1024;
  const adapter = await gpu?.requestAdapter({ powerPreference: 'high-performance', forceFallbackAdapter: false });
  if (!adapter || (adapter.info.isFallbackAdapter ?? (adapter as GPUAdapter & { isFallbackAdapter?: boolean }).isFallbackAdapter) !== false || !adapter.features.has('timestamp-query')) throw new Error('A confirmed hardware timestamp-query GPU is required; no CPU/cloud fallback');
  if (options.suiteOnly) {
    const response = await fetch('/capture-configurations'); if (!response.ok) throw new Error('Cannot read prior capture configurations');
    const previous: unknown = await response.json();
    if (!Array.isArray(previous)) throw new Error('Invalid prior capture configuration inventory');
    const device = await adapter.requestDevice({ requiredFeatures: ['timestamp-query', ...(['shader-f16', 'subgroups'] as GPUFeatureName[]).filter(f => adapter.features.has(f))],
      requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize, maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage } });
    const results = [];
    try {
      for (const item of previous) {
        const config = validateConfiguration(item);
        const result = await replay(device, config, options.repeats, options.warmups);
        results.push(result); await save(`replay-${config.id}.json`, JSON.stringify(result));
      }
      const report = { suiteOnly: true, configurations: results.length, replays: results, correctness: results.every(r => r.comparisons.every(c => c.bitExact)),
        baselineWeightedNs: results.reduce((n, r) => n + r.baselineWeightedNs, 0), optimizedWeightedNs: results.reduce((n, r) => n + r.optimizedWeightedNs, 0),
        ranking: [...results].sort((a, b) => b.baselineWeightedNs - a.baselineWeightedNs) };
      await save('report.json', JSON.stringify(report));
      if (!report.correctness) throw new Error('Kernel output mismatch; artifacts retained');
      return report;
    } finally { device.destroy(); }
  }
  ort.env.webgpu.adapter = adapter; ort.env.wasm.numThreads = 1; ort.env.wasm.wasmPaths = '/ort/'; ort.env.webgpu.profiling = { mode: 'default' };
  const modelResponse = await fetch('/model'); if (!modelResponse.ok) throw new Error('Model read failed');
  const modelBytes = await modelResponse.arrayBuffer(), graph = artifact.placementGraph as PlacementGraph;
  await bindPlacementGraph(graph, 'models/zipenhancer.onnx', modelBytes);
  const audits: GraphPlacementDiagnostic[] = [], waves: CampaignWave[] = [];
  const spectra = new Map<number, CampaignSpectrum>();
  let audit: GpuRunAudit | null = null, dispatches = 0, observedDispatches = 0, runNumber = 0, manifestNumber = 0, modelInputsComplete = false, wavesFromRecovery = false;
  if (options.recover) {
    const response = await fetch('/recovery-manifest'); if (!response.ok) throw new Error('Verified capture campaign is unavailable');
    // Node verifies identity, complete file hashes, and all source-node audits.
    const recovery: CaptureCampaign = await response.json();
    for (const config of recovery.configurations) {
      config.captured = config.captureComplete;
      if (!config.captureComplete) {
        for (const input of config.inputs) { delete input.file; delete input.sha256; }
        delete config.uniformFile; delete config.uniformSha256;
      }
      configurations.push(config);
      configurationKeys.set(JSON.stringify([config.shader, config.inputs.map(tensorDescription), config.outputs.map(tensorDescription), config.uniforms, config.dispatchGroup]), config.id);
      if (!recovery.modelInputsComplete) { config.count = 0; config.gpuDurationNs = 0; config.sources = {}; }
    }
    for (const spectrum of recovery.spectra) spectra.set(spectrum.id, spectrum);
    if (recovery.modelInputsComplete) {
      for (const occurrence of recovery.occurrences) occurrences.push(occurrence);
      audits.push(...recovery.audits); waves.push(...recovery.waves);
      modelInputsComplete = true; measurement = false; verifiedFullRecovery = true; wavesFromRecovery = true;
    }
  }
  async function checkpoint(stage: string) {
    const missing = configurations.filter(c => !c.captureComplete).map(c => ({
      id: c.id, program: c.program, inputs: c.inputs, uniforms: c.uniforms, introducedPhase: c.introducedPhase,
      introducingSpectrumId: c.introducingSpectrumId, count: c.count,
    }));
    const manifest: CaptureCampaign = { schema: 'babel-zip-kernel-capture-v1', identity: options.identity, sourcePaths: options.sourcePaths,
      configurations, spectra: [...spectra.values()], occurrences, audits, waves, modelInputsComplete, stage, missing };
    await save(`capture-manifest-${manifestNumber++}.json`, JSON.stringify(manifest));
  }
  async function timestampsComplete() {
    const deadline = performance.now() + 10000;
    while (observedDispatches < submittedDispatches) {
      if (performance.now() > deadline) throw new Error('Dispatch timestamps did not complete');
      const { promise, resolve } = Promise.withResolvers<void>(); setTimeout(resolve, 5); await promise;
    }
  }
  ort.env.webgpu.profiling.ondata = (publicData) => {
    const data = publicData as typeof publicData & { profilerConfigId: number };
    if (data.profilerConfigId < 0) return;
    audit?.observe(data); observedDispatches++;
    if (!measurement) return;
    const config = configurations[data.profilerConfigId]; if (!config) throw new Error('Missing dispatch configuration identity');
    config.count++; config.gpuDurationNs += data.endTime - data.startTime;
    const name = `${data.kernelType}|${data.kernelName}`; config.sources[name] = (config.sources[name] ?? 0) + 1;
    occurrences.push({ configId: config.id, phase: introductionPhase, run: currentSpectrumId, ...data });
  };
  captureActive = true;
  let session = await ort.InferenceSession.create(modelBytes, { executionProviders: [{ name: 'webgpu', preferredLayout: 'NCHW' }], graphOptimizationLevel: 'disabled' });
  const device = await ort.env.webgpu.device as GPUDevice;
  await device.queue.onSubmittedWorkDone(); await timestampsComplete();
  await checkpoint('initialized-before-buffer-readback');
  await drainCaptures(device); await checkpoint('initialized');
  const infer = async (magnitude: Float32Array, phase: Float32Array, frames: number, originId?: number) => {
    currentSpectrumId = originId ?? runNumber++; introductionPhase = 'inference';
    const a = new ort.Tensor('float32', magnitude, [1, 201, frames]), b = new ort.Tensor('float32', phase, [1, 201, frames]);
    audit = new GpuRunAudit(graph);
    let outputs: ort.InferenceSession.ReturnType | undefined;
    try {
      outputs = await session.run({ noisy_mag: a, noisy_pha: b }); await device.queue.onSubmittedWorkDone();
      const proof = await audit.finishAfterDispatches(1); if (measurement) audits.push(proof);
      await timestampsComplete();
      if (!spectra.has(currentSpectrumId) && configurations.some(c => c.introducingSpectrumId === currentSpectrumId)) {
        if (!(magnitude.buffer instanceof ArrayBuffer) || !(phase.buffer instanceof ArrayBuffer)) throw new Error('Original spectrum ownership is not a plain ArrayBuffer');
        const magFile = `spectrum-${currentSpectrumId}-mag.bin`, phaFile = `spectrum-${currentSpectrumId}-pha.bin`;
        const magBytes = new Uint8Array(magnitude.buffer, magnitude.byteOffset, magnitude.byteLength), phaBytes = new Uint8Array(phase.buffer, phase.byteOffset, phase.byteLength);
        await save(magFile, magBytes); await save(phaFile, phaBytes);
        spectra.set(currentSpectrumId, { id: currentSpectrumId, frames, mag: { file: magFile, sha256: await captureSha256(magBytes) },
          pha: { file: phaFile, sha256: await captureSha256(phaBytes) } });
      }
      if (currentSpectrumId === 0 || !measurement) await checkpoint('inference-before-buffer-readback');
      await drainCaptures(device);
      if (currentSpectrumId === 0 || !measurement) await checkpoint('inference-captured');
      return { magnitude: outputs.amp_g.data as Float32Array, phase: outputs.pha_g.data as Float32Array };
    } finally { audit = null; a.dispose(); b.dispose(); if (outputs) for (const output of Object.values(outputs)) output.dispose(); }
  };
  async function loadSpectrum(id: number) {
    const spectrum = spectra.get(id); if (!spectrum) throw new Error(`Missing original introducing spectrum ${id}`);
    const channels = [];
    for (const channel of [spectrum.mag, spectrum.pha]) {
      const response = await fetch(`/artifact/${channel.file}`); if (!response.ok) throw new Error(`Original spectrum read failed: ${channel.file}`);
      const bytes = await response.arrayBuffer();
      if (bytes.byteLength !== 201 * spectrum.frames * 4 || await captureSha256(bytes) !== channel.sha256) throw new Error(`Original spectrum identity changed: ${channel.file}`);
      channels.push(new Float32Array(bytes));
    }
    return { magnitude: channels[0], phase: channels[1], frames: spectrum.frames };
  }
  try {
    if (!modelInputsComplete) {
      for (let index = 0; index < options.audioCount; index++) {
        const response = await fetch(`/audio/${index}`); if (!response.ok) throw new Error('Audio read failed');
        const source = await response.arrayBuffer();
        if (await captureSha256(source) !== options.identity.audioSha256[index]) throw new Error('Original audio changed during the capture campaign');
        const { samples, sampleRate } = decodeEnhancementWav(source);
        const start = performance.now(), waveform = await enhanceZipSamples(samples, sampleRate, infer);
        const wav = encodeEnhancementWav(waveform, sampleRate, samples.length), hash = await captureSha256(wav.buffer);
        await save(`enhanced-${index}.wav`, wav); waves.push({ index, sampleRate, frames: samples.length, wavSha256: hash, wallMs: performance.now() - start });
        await checkpoint(`lane-${index}-complete`);
      }
      modelInputsComplete = true;
    }
    dispatches = occurrences.length; measurement = false; verifiedFullRecovery = true;
    await checkpoint('full-model-before-capture-recovery');
    await save('occurrences.json', JSON.stringify(occurrences));
    while (configurations.some(config => !config.captureComplete)) {
      const pending = configurations.filter(config => !config.captureComplete), before = configurations.filter(c => c.captureComplete).length;
      const origins = [...new Set(pending.map(c => c.introducingSpectrumId))];
      for (const original of origins) {
        const id = original ?? spectra.keys().next().value;
        if (id === undefined) throw new Error('No original full-model spectrum is recorded for cold capture');
        const spectral = await loadSpectrum(id);
        const warmBefore = configurations.filter(c => c.captureComplete).length;
        await infer(spectral.magnitude, spectral.phase, spectral.frames, id);
        if (configurations.filter(c => c.captureComplete).length === warmBefore) {
          // Cold weight transforms can first execute inside session.run.
          await session.release(); introductionPhase = 'initialization'; currentSpectrumId = null;
          session = await ort.InferenceSession.create(modelBytes, { executionProviders: [{ name: 'webgpu', preferredLayout: 'NCHW' }], graphOptimizationLevel: 'disabled' });
          await device.queue.onSubmittedWorkDone(); await timestampsComplete();
          await checkpoint('cold-session-before-buffer-readback'); await drainCaptures(device);
          await infer(spectral.magnitude, spectral.phase, spectral.frames, id);
        }
      }
      await checkpoint('capture-recovery-pass-complete');
      if (configurations.filter(c => c.captureComplete).length <= before) {
        const missing = configurations.filter(c => !c.captureComplete).map(c => ({ id: c.id, program: c.program, inputs: c.inputs, uniforms: c.uniforms,
          introducedPhase: c.introducedPhase, introducingSpectrumId: c.introducingSpectrumId, count: c.count }));
        await save('coverage-failure.json', JSON.stringify({ missing, latestManifest: `capture-manifest-${manifestNumber - 1}.json` }));
        await save('configurations.json', JSON.stringify(configurations));
        throw new Error(`Cannot capture every configuration; missing ${JSON.stringify(missing)}. Exact manifests and occurrences retained.`);
      }
    }
    for (const config of configurations) {
      config.shaderSha256 = await captureSha256(new TextEncoder().encode(config.shader).buffer);
      config.baselineShaderSha256 = await captureSha256(new TextEncoder().encode(config.baselineShader).buffer);
    }
    await checkpoint('all-configurations-captured');
    await session.release();
    await save('configurations.json', JSON.stringify(configurations));
    // ORT can use an internal timestamp capability without enabling the public
    // timestamp-query feature. External replay owns an explicitly enabled device.
    const replayAdapter = await gpu?.requestAdapter({ powerPreference: 'high-performance', forceFallbackAdapter: false });
    if (!replayAdapter || replayAdapter.info.isFallbackAdapter !== false || !replayAdapter.features.has('timestamp-query')) throw new Error('Hardware timestamp-query adapter unavailable for external replay');
    const replayDevice = await replayAdapter.requestDevice({
      requiredFeatures: ['timestamp-query', ...(['shader-f16', 'subgroups'] as GPUFeatureName[]).filter(f => replayAdapter.features.has(f))],
      requiredLimits: { maxBufferSize: replayAdapter.limits.maxBufferSize, maxStorageBufferBindingSize: replayAdapter.limits.maxStorageBufferBindingSize,
        maxComputeWorkgroupStorageSize: replayAdapter.limits.maxComputeWorkgroupStorageSize, maxStorageBuffersPerShaderStage: replayAdapter.limits.maxStorageBuffersPerShaderStage },
    });
    const replays = [];
    try {
      for (const config of configurations) {
        const result = await replay(replayDevice, config, options.repeats, options.warmups); replays.push(result);
        await save(`replay-${config.id}.json`, JSON.stringify(result));
      }
    } finally { replayDevice.destroy(); }
    const families = Object.fromEntries([...new Set(configurations.map(c => c.program))].map(program => {
      const configs = configurations.filter(c => c.program === program);
      return [program, { configurations: configs.length, shaderVariants: new Set(configs.map(c => c.shaderSha256)).size,
        count: configs.reduce((n, c) => n + c.count, 0), gpuDurationNs: configs.reduce((n, c) => n + c.gpuDurationNs, 0) }];
    }));
    const sourceNodes: Record<string, { kernelName: string; operator: string; count: number; gpuDurationNs: number; configurations: number[] }> = {};
    for (const occurrence of occurrences) {
      const key = `${occurrence.kernelType}|${occurrence.kernelName}`;
      const node = sourceNodes[key] ??= { kernelName: occurrence.kernelName, operator: occurrence.kernelType, count: 0, gpuDurationNs: 0, configurations: [] };
      node.count++; node.gpuDurationNs += occurrence.endTime - occurrence.startTime;
      if (!node.configurations.includes(occurrence.configId)) node.configurations.push(occurrence.configId);
    }
    const report = { modelSha256: graph.sha256, adapter: adapter.info, dispatches, configurations: configurations.length, waves, replays, families, audits,
      waveInferenceOrigin: wavesFromRecovery ? 'identity-validated capture campaign' : 'current full-model run',
      initializationDispatches: occurrences.filter(o => o.phase === 'initialization').length,
      inferenceDispatches: occurrences.filter(o => o.phase === 'inference').length,
      sourceNodes: Object.values(sourceNodes).sort((a, b) => b.gpuDurationNs - a.gpuDurationNs),
      baselineWeightedNs: replays.reduce((n, r) => n + r.baselineWeightedNs, 0), optimizedWeightedNs: replays.reduce((n, r) => n + r.optimizedWeightedNs, 0),
      correctness: replays.every(r => r.comparisons.every(c => c.bitExact)), ranking: [...replays].sort((a, b) => b.baselineWeightedNs - a.baselineWeightedNs),
      operators: Object.fromEntries([...new Set(occurrences.map(d => d.kernelType))].map(op => [op, { count: occurrences.filter(d => d.kernelType === op).length, gpuDurationNs: occurrences.filter(d => d.kernelType === op).reduce((n, d) => n + d.endTime - d.startTime, 0) }])) };
    await save('report.json', JSON.stringify(report));
    if (!report.correctness) throw new Error('Kernel baseline/optimized output mismatch; artifacts retained');
    return { dispatches, configurations: configurations.length, correctness: report.correctness, waves };
  } finally {
    delete host.__babelKernelProfiler;
    for (const capture of captures.splice(0)) for (const buffer of capture.buffers) buffer.destroy();
  }
};
