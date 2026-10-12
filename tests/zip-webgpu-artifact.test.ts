import assert from 'node:assert/strict';
import test from 'node:test';
import { assertZipWebGpuFloat32Plan } from '../scripts/zip-webgpu-artifact.mjs';

const plan = (nodes: object[] = [{ op: 'Cast', attributes: { to: 1 } }]) => ({
  precision: 'float32', tensors: [1, 6, 7, 9].map(dataType => ({ dataType })), nodes,
});


test('artifact guard rejects mixed labels, half tensors and nested fused half casts', () => {
  assert.throws(() => assertZipWebGpuFloat32Plan({ ...plan(), precision: 'mixed-float16' }), /full-FP32/);
  for (const dataType of [10, 11, 16]) {
    assert.throws(() => assertZipWebGpuFloat32Plan({ ...plan(), tensors: [{ dataType }] }), /full-FP32/);
  }
  for (const node of [
    { op: 'Cast', attributes: { to: 10 } },
    { op: 'ZipSwoosh', attributes: { outputType: 10 } },
    { op: 'ZipSwoosh', attributes: { steps: [{ op: 'Cast', to: 10 }] } },
    { op: 'Fusion', attributes: { steps: [{ op: 'Fusion', attributes: { steps: [{ op: 'Cast', to: 10 }] } }] } },
  ]) assert.throws(() => assertZipWebGpuFloat32Plan(plan([node])), /full-FP32/);
});
