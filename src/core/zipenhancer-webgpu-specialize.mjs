// Only the direct fixed-shape engine uses this adapter. ORT's general ASR
// backend remains dynamic. ShaderHelperImpl layout is pinned by the build.
const shaderTypes = { 12: 'u32', 6: 'i32', 1: 'f32', 10: 'f16' };
function halfValue(bits) {
  const sign = bits & 0x8000 ? -1 : 1, exponent = (bits >>> 10) & 31, fraction = bits & 1023;
  return exponent === 31 ? NaN : sign * (exponent ? 1 + fraction / 1024 : fraction / 1024) * 2 ** (exponent ? exponent - 15 : -14);
}
function scalar(type, value) {
  if (type === 'u32') {
    if (!Number.isInteger(value) || value < 0 || value > 0xffffffff) throw new Error('Invalid static u32 uniform');
    return `${value}u`;
  }
  if (type === 'i32') {
    if (!Number.isInteger(value) || value < -0x80000000 || value > 0x7fffffff) throw new Error('Invalid static i32 uniform');
    return `bitcast<i32>(${value >>> 0}u)`;
  }
  const number = type === 'f16' ? halfValue(value) : Math.fround(value);
  if (!Number.isFinite(number)) throw new Error('Nonfinite static uniform');
  return `${type}(${Object.is(number, -0) ? '-0.0' : String(number)})`;
}
function member(type, values) {
  if (values.length === 1) return scalar(type, values[0]);
  if (values.length <= 4) return `vec${values.length}<${type}>(${values.map(value => scalar(type, value)).join(',')})`;
  if (type === 'f16') {
    const matrices = [];
    for (let start = 0; start < values.length; start += 8) {
      const row = Array.from({ length: 8 }, (_, index) => values[start + index] ?? 0);
      matrices.push(`mat2x4<f16>(${member(type, row.slice(0, 4))},${member(type, row.slice(4))})`);
    }
    return `array<mat2x4<f16>,${matrices.length}>(${matrices.join(',')})`;
  }
  const vectors = [];
  for (let start = 0; start < values.length; start += 4) vectors.push(member(type, Array.from({ length: 4 }, (_, index) => values[start + index] ?? 0)));
  return `array<vec4<${type}>,${vectors.length}>(${vectors.join(',')})`;
}

export function specializeProgram(program, inputs) {
  const run = program.getRunData(inputs), uniforms = run.programUniforms;
  if (!uniforms?.length) return program;
  // Nonfinite constants cannot be represented in WGSL constant expressions.
  // Retain the original GPU uniform path for those programs, without altering values.
  if (uniforms.some(uniform => {
    const values = typeof uniform.data === 'number' ? [uniform.data] : uniform.data;
    return !shaderTypes[uniform.type] || !values.length || values.some(value => !Number.isFinite(uniform.type === 10 ? halfValue(value) : uniform.type === 1 ? Math.fround(value) : value));
  })) return program;
  const { programUniforms, ...staticRun } = run;
  return {
    ...program,
    shaderCache: { ...program.shaderCache, hint: `${program.shaderCache?.hint ?? ''};zip-static-uniforms-v1:${JSON.stringify(uniforms)}` },
    getRunData: () => staticRun,
    getShaderSource(helper) {
      const source = program.getShaderSource(helper);
      if (!Array.isArray(helper.uniforms) || helper.uniforms.length !== uniforms.length || typeof helper.uniformDeclaration !== 'function') throw new Error('Pinned ShaderHelper uniform layout changed');
      const fields = helper.uniforms;
      const original = helper.uniformDeclaration.bind(helper);
      const expressions = uniforms.map((uniform, index) => {
        const field = fields[index], values = typeof uniform.data === 'number' ? [uniform.data] : Array.from(uniform.data);
        if (field.type !== shaderTypes[uniform.type] || (field.length ?? 1) !== values.length) throw new Error('Static GPU uniform type/length mismatch');
        return member(field.type, values);
      });
      helper.uniformDeclaration = () => {
        const declaration = original();
        const binding = /@group\(0\)\s*@binding\(\d+\)\s*var<uniform>\s+uniforms:\s*Uniforms;/g;
        if ([...declaration.matchAll(binding)].length !== 1) throw new Error('Pinned GPU uniform declaration changed');
        return declaration.replace(binding, `const uniforms: Uniforms = Uniforms(${expressions.join(',')});`);
      };
      return source;
    },
  };
}
