import test from 'node:test';
import assert from 'node:assert/strict';
import {
  bindPlacementGraph, GpuRunAudit, validatePlacementPolicy,
  type PlacementNode, type PlacementPolicy
} from '../src/core/local-gpu-placement';

const sources = { asrCheckpointSha256: 'a'.repeat(64), cDenoiseCheckpointSha256: 'b'.repeat(64) };
function node(name: string, opType: string, placement: PlacementNode['placement'], dtype = 'float16'): PlacementNode {
  return { name, opType, placement, reason: 'Fixture tensor dependency proof',
    inputs: [{ name: `${name}/input`, dtype, ancestry: placement === 'host-metadata' ? 'metadata' : 'neural-data' }],
    outputs: [{ name: `${name}/output`, dtype, ancestry: placement === 'host-metadata' ? 'metadata' : 'neural-data' }] };
}
function policy(nodes: PlacementNode[]): PlacementPolicy {
  return { schema: 'babel-c-denoise-gpu-placement-v1', sources,
    graphs: Object.fromEntries(['asr', 'context', 'denoise'].map((key) => [key,
      { path: `${key}.onnx`, sha256: 'c'.repeat(64), nodes }])) as PlacementPolicy['graphs'] };
}
const dispatch = (kernelName: string, kernelType: string) => ({ version: 1, kernelName, kernelType,
  programName: `${kernelType}/shader`, startTime: 100, endTime: 200 });

test('empty, partial, and metadata-only GPU traces cannot certify learned inference', () => {
  const graph = policy([node('shape', 'Shape', 'host-metadata', 'int64'),
    node('embedding', 'Gather', 'gpu'), node('projection', 'MatMul', 'gpu')]).graphs.context;
  const audit = new GpuRunAudit(graph);
  assert.throws(() => audit.finish(1), /2 required neural nodes/);
  audit.observe(dispatch('shape', 'Shape'));
  assert.throws(() => audit.finish(1), /2 required neural nodes/);
  audit.observe(dispatch('embedding', 'Gather'));
  assert.throws(() => audit.finish(1), /projection/);
  audit.observe(dispatch('projection', 'MatMul'));
  const result = audit.finish(1);
  assert.equal(result.verifiedGpuNodes, 2);
  assert.equal(result.allowedHostMetadataNodes, 1);
  assert.equal(result.gpuPrograms, 2);
  assert.throws(() => new GpuRunAudit(graph).finish(2), /2 required neural nodes/);
});

test('renamed nodes, wrong operators, and invalid timestamps fail closed', () => {
  const graph = policy([node('projection', 'MatMul', 'gpu')]).graphs.context;
  const renamed = new GpuRunAudit(graph);
  renamed.observe(dispatch('fused_projection', 'MatMul'));
  assert.throws(() => renamed.finish(1), /no GPU dispatch/);
  for (const evidence of [dispatch('projection', 'Add'),
    { ...dispatch('projection', 'MatMul'), startTime: 201 },
    { ...dispatch('projection', 'MatMul'), endTime: Number.NaN }]) {
    const audit = new GpuRunAudit(graph);
    audit.observe(evidence);
    audit.observe(dispatch('projection', 'MatMul'));
    assert.throws(() => audit.finish(1), /Invalid GPU dispatch evidence/);
  }
});

test('dtype and dependency proof distinguish metadata Gather from learned embedding Gather', () => {
  const metadata = node('shape_gather', 'Gather', 'host-metadata', 'int64');
  const embedding = node('embedding', 'Gather', 'gpu');
  const valid = policy([metadata, embedding]);
  assert.equal(validatePlacementPolicy(valid, sources), valid);
  const forged = policy([metadata, { ...embedding, placement: 'host-metadata' }]);
  assert.throws(() => validatePlacementPolicy(forged, sources), /Unproven host metadata ancestry/);
  const inputForged = policy([embedding, { ...metadata, inputs: [{ name: 'learned_integer_lookup', dtype: 'int64', ancestry: 'model-initializer' }] }]);
  assert.throws(() => validatePlacementPolicy(inputForged, sources), /Unproven host metadata ancestry/);
});

test('only storage-preserving aliases bypass neural arithmetic dispatch checks', () => {
  const compute = node('projection', 'MatMul', 'gpu');
  validatePlacementPolicy(policy([compute, node('reshape', 'Reshape', 'tensor-alias'),
    node('same_type', 'Cast', 'tensor-alias')]), sources);
  assert.throws(() => validatePlacementPolicy(policy([compute, node('fake_alias', 'Add', 'tensor-alias')]), sources), /Non-alias/);
  const cast = node('precision_change', 'Cast', 'tensor-alias');
  cast.outputs[0].dtype = 'float32';
  assert.throws(() => validatePlacementPolicy(policy([compute, cast]), sources), /Non-alias/);
});

test('unknown neural operators and stale source checkpoint policies are rejected before session creation', () => {
  assert.throws(() => validatePlacementPolicy(policy([node('neural', 'UnsupportedNeuralOp', 'gpu')]), sources), /Unsupported WebGPU neural operator/);
  assert.throws(() => validatePlacementPolicy(policy([node('projection', 'MatMul', 'gpu')]), { ...sources, asrCheckpointSha256: 'd'.repeat(64) }), /accepted source checkpoints/);
});

test('placement policy binds loaded graph path and exact file bytes, not operator names alone', async () => {
  const bytes = Uint8Array.of(1, 2, 3).buffer;
  const sha256 = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), (byte) => byte.toString(16).padStart(2, '0')).join('');
  const graph = { ...policy([node('projection', 'MatMul', 'gpu')]).graphs.context, sha256 };
  await bindPlacementGraph(graph, graph.path, bytes);
  await assert.rejects(bindPlacementGraph(graph, 'different.onnx', bytes), /not bound/);
  await assert.rejects(bindPlacementGraph(graph, graph.path, Uint8Array.of(1, 2, 4).buffer), /not bound/);
});
