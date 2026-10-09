import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import ts from 'typescript';
import { broadcastOffsetExpression, specializeBinaryAssignment } from '../src/core/webgpu-binary-address';
import { assertKernelSource, kernelSourceHashes, transformBinarySource, transformProgramManagerSource, transformBackendProfileSource } from '../scripts/ort-kernel-transforms.mjs';


// Evaluate the restricted emitted u32 address grammar, with integer (not JS float) division.
function address(expression: string, globalIndex: number): number {
  const tokens = expression.match(/global_idx|\d+u?|[()+*/%]/g) ?? [];
  let cursor = 0;
  function atom(): number {
    const token = tokens[cursor++];
    if (token === '(') { const value = sum(); assert.equal(tokens[cursor++], ')'); return value; }
    if (token === 'global_idx') return globalIndex;
    assert.match(token, /^\d+u?$/); return Number(token.replace('u', ''));
  }
  function product(): number {
    let result = atom();
    while (['*', '/', '%'].includes(tokens[cursor])) {
      const operator = tokens[cursor++], rhs = atom();
      result = operator === '*' ? result * rhs : operator === '/' ? Math.floor(result / rhs) : result % rhs;
    }
    return result;
  }
  function sum(): number { let result = product(); while (tokens[cursor] === '+') { cursor++; result += product(); } return result; }
  const result = sum(); assert.equal(cursor, tokens.length); return result;
}
function referenceOffset(input: readonly number[], output: readonly number[], index: number): number {
  const coordinates = Array(output.length).fill(0);
  for (let axis = output.length - 1; axis >= 0; axis--) { coordinates[axis] = index % output[axis]; index = Math.floor(index / output[axis]); }
  let result = 0;
  for (let axis = 0; axis < input.length; axis++) result = result * input[axis] + (input[axis] === 1 ? 0 : coordinates[output.length - input.length + axis]);
  return result;
}

test('literal broadcast address agrees for all logical elements, ranks, leading axes and prime tails', () => {
  const outputShapes = [[], [1], [3], [7], [2, 3], [2, 3, 5], [2, 3, 4], [2, 3, 2, 2], [3, 1, 5, 7], [1, 201, 641]];
  for (const output of outputShapes) {
    const size = output.reduce((a, b) => a * b, 1);
    for (let rank = 0; rank <= output.length; rank++) for (let mask = 0; mask < 2 ** rank; mask++) {
      const input = output.slice(output.length - rank).map((d, axis) => mask & (1 << axis) ? d : 1);
      const expression = broadcastOffsetExpression(input, output, 'global_idx');
      // Exhaust all small cases; all model endpoints and deterministic interior probes for large cases.
      const indices = size < 1000 ? Array.from({ length: size }, (_, i) => i) : [0, 1, 3, 4, 640, 641, size - 4, size - 1, ...Array.from({ length: 97 }, (_, i) => Math.floor(i * (size - 1) / 96))];
      for (const index of indices) assert.equal(address(expression, index), referenceOffset(input, output, index), JSON.stringify({ input, output, index, expression }));
    }
  }
});

test('packed vector layout preserves lane mapping for scalar, trailing, leading and shared-four broadcasts', () => {
  for (const [a, b, output, shared] of [
    [[2, 3, 4], [1, 3, 1], [2, 3, 4], false],
    [[1], [2, 3, 7], [2, 3, 7], false],
    [[2, 3, 2, 2], [1, 1, 2, 2], [2, 3, 2, 2], true],
    [[4], [3, 1], [3, 4], false],
    [[1, 4], [3, 1], [3, 4], false],
  ] as const) {
    const count = output.reduce((a: number, b: number) => a * b, 1);
    for (const input of [a, b]) for (let base = 0; base < count; base += 4) {
      const offset = address(broadcastOffsetExpression(input, output, 'global_idx * 4u'), base / 4);
      const vector = input.reduce((a: number, b: number) => a * b, 1) === count || shared || input[input.length - 1] % 4 === 0;
      for (let lane = 0; lane < Math.min(4, count - base); lane++) assert.equal(offset + (vector ? lane : 0), referenceOffset(input, output, base + lane));
    }
  }
  const source = specializeBinaryAssignment([1], [2, 3, 7], [2, 3, 7], true, false, (a, b) => `${a}-${b}`, (a, b) => `${a}-${b}`);
  assert.match(source, /vec4<f32>\(aData\[0u\]\.x\)-bData\[global_idx\]/);
});

test('scalar tails are guarded and the exact upstream expression receives original operand ordering', () => {
  for (const operation of ['+', '-', '*', '/']) {
    const expressions: [string, string][] = [];
    const scalar = (a: string, b: string) => { expressions.push([a, b]); return `${a}${operation}${b}`; };
    const source = specializeBinaryAssignment([2, 3, 5], [1, 3, 1], [2, 3, 5], false, false, scalar, scalar);
    assert.equal(expressions.length, 4);
    assert.equal((source.match(/< 30u/g) ?? []).length, 4);
    for (const [a, b] of expressions) { assert.equal(a, 'aData[offsetA / 4u][offsetA % 4u]'); assert.equal(b, 'bData[offsetB / 4u][offsetB % 4u]'); }
  }
  assert.throws(() => broadcastOffsetExpression([3], [4], 'global_idx'), /dimension/);
  assert.throws(() => broadcastOffsetExpression([1, 3], [3], 'global_idx'), /rank/);
});

test('pinned transforms parse and reject upstream drift; precision promotion remains present', async () => {
  const root = new URL('../node_modules/onnxruntime-web/lib/wasm/jsep/', import.meta.url);
  for (const relative of Object.keys(kernelSourceHashes)) {
    const source = await readFile(new URL(relative, root), 'utf8');
    assertKernelSource(relative, source);
    assert.throws(() => assertKernelSource(relative, source + '\n'), /changed/);
    for (const profile of [false, true]) {
      const transformed = relative === 'webgpu/ops/binary-op.ts' ? transformBinarySource(source, { profile }) :
        relative === 'webgpu/program-manager.ts' ? transformProgramManagerSource(source, { profile }) :
        relative === 'backend-webgpu.ts' && profile ? transformBackendProfileSource(source) : source;
      const result = ts.transpileModule(transformed, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext }, reportDiagnostics: true });
      assert.deepEqual(result.diagnostics?.filter(d => d.category === ts.DiagnosticCategory.Error), []);
      if (relative === 'webgpu/program-manager.ts') assert.match(transformed, /promoteFp16Accumulation\(programInfo\.getShaderSource/);
    }
  }
});

test('real transformed program builder keys every specialized dimension and leaves other dtypes/operators unchanged', async () => {
  const source = await readFile(new URL('../node_modules/onnxruntime-web/lib/wasm/jsep/webgpu/ops/binary-op.ts', import.meta.url), 'utf8');
  const utilSource = await readFile(new URL('../node_modules/onnxruntime-web/lib/wasm/jsep/util.ts', import.meta.url), 'utf8');
  const utilParsed = ts.createSourceFile('util.ts', utilSource, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS);
  const classes = utilParsed.statements.filter((statement): statement is ts.ClassDeclaration => ts.isClassDeclaration(statement) &&
    Boolean(statement.name && ['BroadcastUtil', 'ShapeUtil'].includes(statement.name.text)))
    .map(statement => ts.factory.updateClassDeclaration(statement, undefined, statement.name, statement.typeParameters, statement.heritageClauses, statement.members));
  assert.equal(classes.length, 2);
  const utilPrinted = ts.createPrinter().printFile(ts.factory.updateSourceFile(utilParsed, classes));
  const utilCode = ts.transpileModule(utilPrinted, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText;
  interface ShapeUtilities {
    ShapeUtil: { size: (dims: readonly number[]) => number; areEqual: (a: readonly number[], b: readonly number[]) => boolean };
    BroadcastUtil: { calcShape: (a: readonly number[], b: readonly number[], isMatMul: false) => readonly number[] | undefined };
  }
  // Use the actual upstream class bodies without Node's package-level CJS/ESM named-export ambiguity.
  const { ShapeUtil, BroadcastUtil }: ShapeUtilities = new Function(`${utilCode}\nreturn { ShapeUtil, BroadcastUtil };`)();
  for (const baseline of [false, true]) {
    const transformed = transformBinarySource(source, { baseline });
    const parsed = ts.createSourceFile('binary-op.ts', transformed, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS);
    // Execute the actual two upstream builder declarations, not a restatement of the cache policy.
    const declarations = parsed.statements.filter(statement => ts.isVariableStatement(statement) &&
      statement.declarationList.declarations.some(declaration => ts.isIdentifier(declaration.name) &&
        ['createBinaryOpProgramShader', 'createBinaryOpProgramInfo'].includes(declaration.name.text)));
    const printed = ts.createPrinter().printFile(ts.factory.updateSourceFile(parsed, declarations));
    const code = ts.transpileModule(printed, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText;
    type Builder = (name: string, key: string, a: { dims: number[]; dataType: number }, b: { dims: number[]; dataType: number }, expression: (a: string, b: string) => string) =>
      { shaderCache: { hint: string; inputDependencies: string[] } };
    const builder: Builder = new Function('DataType', 'ShapeUtil', 'BroadcastUtil', 'specializeBinaryAssignment',
      `${code}\nreturn createBinaryOpProgramInfo;`)({ float: 1 }, ShapeUtil, BroadcastUtil, specializeBinaryAssignment);
    for (const name of ['Add', 'Sub', 'Mul', 'Div', 'Pow', 'Greater']) for (const dataType of [1, 10, 6]) {
      const first = builder(name, '', { dims: [2, 3, 5], dataType }, { dims: [1, 3, 1], dataType }, (a, b) => `${a}+${b}`);
      const second = builder(name, '', { dims: [2, 7, 5], dataType }, { dims: [1, 7, 1], dataType }, (a, b) => `${a}+${b}`);
      const specialized = !baseline && ['Add', 'Sub', 'Mul', 'Div'].includes(name) && dataType === 1;
      assert.deepEqual(first.shaderCache.inputDependencies, specialized ? ['dims', 'dims'] : ['rank', 'rank']);
      // Identical hint/rank must not alias distinct shape-specialized artifacts.
      if (specialized) {
        assert.equal(first.shaderCache.hint, second.shaderCache.hint);
        assert.equal(first.shaderCache.inputDependencies[0], 'dims');
      }
    }
  }
});
