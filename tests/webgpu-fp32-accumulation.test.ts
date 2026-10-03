import test from 'node:test';
import assert from 'node:assert/strict';
import { promoteFp16Accumulation } from '../src/core/webgpu-fp32-accumulation';

test('vector GEMM widens products and bias while preserving FP16 buffers and one output conversion', () => {
  const shader = `
var<workgroup> mm_Asub: array<array<vec4<f16>, 8>, 32>;
var<workgroup> mm_Bsub: array<array<vec4<f16>, 8>, 32>;
fn mm_write(batch: i32, row: i32, colIn: i32, valueIn: vec4<f16>) {
  var value = valueIn;
  value = value + bias[colIn];
  set_resultByIndices(vec3<u32>(batch, row, colIn), value);
}
var acc: array<vec4<f16>, rowPerThread>;
let ACached = mm_Asub[tileRow + i][k];
let BCached0 = mm_Bsub[k * innerElementSize][tileCol];
acc[i] = BCached0 * ACached.x + acc[i];
mm_write(batch, globalRow + innerRow, globalCol, acc[innerRow]);`;
  const promoted = promoteFp16Accumulation(shader);
  assert.match(promoted, /mm_Asub: array<array<vec4<f16>/);
  assert.match(promoted, /var acc: array<vec4<f32>/);
  assert.match(promoted, /valueIn: vec4<f32>/);
  assert.match(promoted, /value = value \+ vec4<f32>\(bias\[colIn\]\)/);
  assert.match(promoted, /fma\(BCached0, vec4<f32>\(ACached.x\), acc\[i\]\)/);
  assert.match(promoted, /set_resultByIndices\(vec3<u32>\(batch, row, colIn\), vec4<f16>\(value\)\)/);
  assert.match(promoted, /globalCol, acc\[innerRow\]\)/);
});

test('scalar GEMM widens cached operands and preserves convolution output conversion', () => {
  const shader = `
var<workgroup> mm_Asub: array<array<f16, 32>, 32>;
fn mm_write(batch: i32, row: i32, colIn: i32, valueIn: f16) {
  var value = valueIn;
  value += getBiasByOutputCoords(coords);
  setOutputAtCoords(coords[0], coords[1], coords[2], coords[3], value);
}
var acc: array<array<f16, colPerThread>, rowPerThread>;
var BCached: array<f16, colPerThread>;
BCached[inner] = mm_Bsub[k][tileCol + inner];
let ACached = mm_Asub[tileRow + innerRow][k];
acc[innerRow][innerCol] = acc[innerRow][innerCol] + ACached * BCached[innerCol];`;
  const promoted = promoteFp16Accumulation(shader);
  assert.match(promoted, /var acc: array<array<f32/);
  assert.match(promoted, /BCached\[inner\] = f32\(mm_Bsub/);
  assert.match(promoted, /valueIn: f32/);
  assert.match(promoted, /value \+= f32\(getBiasByOutputCoords\(coords\)\)/);
  assert.match(promoted, /coords\[3\], f16\(value\)\)/);
});

test('unrelated FP16 elementwise shaders and existing FP32 accumulators stay unchanged', () => {
  for (const source of ['var value: f16 = f16(0); output[0] = value;', 'var acc: array<vec4<f32>, 4>;']) {
    assert.equal(promoteFp16Accumulation(source), source);
  }
});
