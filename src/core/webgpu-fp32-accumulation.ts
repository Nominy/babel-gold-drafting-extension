/** ORT 1.29 JSEP keeps FP16 storage but also accumulates GEMM/Conv in FP16.
 * Match native mixed precision: widen operands and reductions, round on output.
 * Applied to the pinned generated shaders before WebGPU compilation.
 */
export function promoteFp16Accumulation(source: string): string {
  if (source.includes('mm_Asub') && /var acc\s*:\s*array<.*f16/.test(source)) {
    const vector = /mm_Asub\s*:\s*array<array<vec(\d)<f16>/.exec(source);
    const castA = vector ? `vec${vector[1]}<f32>` : 'f32';
    const castB = vector ? 'vec4<f32>' : 'f32';
    const castOutput = vector ? 'vec4<f16>' : 'f16';
    const computeOutput = vector ? 'vec4<f32>' : 'f32';
    source = source
      .replace(/(var acc\s*:\s*array<[^;]+);/, (_, declaration: string) => `${declaration.replaceAll('f16', 'f32')};`)
      .replace(/let (ACached\d*) = ([^;]+);/g, `let $1 = ${castA}($2);`)
      .replace(/let (BCached\d*) = ([^;]+);/g, `let $1 = ${castB}($2);`)
      .replace(/var BCached\s*:\s*array<f16,/g, 'var BCached: array<f32,')
      .replace(/(BCached\[inner\] = )([^;]+);/g, '$1f32($2);')
      .replace(/acc\[i\] = (BCached\d*) \* (ACached(?:\d*\[i\]|\.[xyzw])) \+ acc\[i\];/g, 'acc[i] = fma($1, vec4<f32>($2), acc[i]);')
      .replace(/acc\[innerRow\]\[innerCol\] = acc\[innerRow\]\[innerCol\] \+ ACached \* BCached\[innerCol\];/g,
        'acc[innerRow][innerCol] = fma(ACached, BCached[innerCol], acc[innerRow][innerCol]);');
    const start = source.indexOf('fn mm_write(');
    if (start < 0) throw new Error('Pinned FP16 GEMM shader has no output function');
    const brace = source.indexOf('{', start);
    let end = brace + 1, depth = 1;
    while (depth && end < source.length) { const c = source[end++]; if (c === '{') depth++; else if (c === '}') depth--; }
    let write = source.slice(start, end);
    write = write
      .replace(/(valueIn\s*:\s*)(?:f16|vec4<f16>)/, `$1${computeOutput}`)
      .replace(/value = value \+ ([^;]+);/g, `value = value + ${computeOutput}($1);`)
      .replace(/value \+= ([^;]+);/g, `value += ${computeOutput}($1);`)
      .replace(/((?:output|result)\[[^;]+?\]\s*=\s*)value;/g, `$1${castOutput}(value);`)
      .replace(/(setOutputAtCoords\([^;]+,\s*)value(\);)/g, `$1${castOutput}(value)$2`);
    write = write.replace(/(set_\w+\([^;]+,\s*)value(\);)/g, `$1${castOutput}(value)$2`);
    source = source.slice(0, start) + write + source.slice(end);
  }
  const accumulator = /var value:\s*(f16|vec\d<f16>)\s*=/.exec(source);
  const vectorAccumulator = /var values:\s*array<(f16|vec\d<f16>),/.exec(source);
  const storageType = accumulator?.[1] ?? vectorAccumulator?.[1];
  if (storageType && (source.includes('w_val') || source.includes('wVal'))) {
    const computeType = storageType.replace('f16', 'f32');
    source = source
      .replace(`var value: ${storageType} = ${storageType}(0)`, `var value: ${computeType} = ${computeType}(0)`)
      .replace(`var values: array<${storageType},`, `var values: array<${computeType},`)
      .replace(/value \+= xVal \* wVal;/g, `value += f32(xVal) * ${computeType}(wVal);`)
      .replace(/values\[i\] = fma\(([^;]+), w_val, values\[i\]\);/g, `values[i] = fma(${computeType}($1), ${computeType}(w_val), values[i]);`)
      .replace(/value \+= b\[output_channel\];/g, `value += ${computeType}(b[output_channel]);`)
      .replace(/(output\[[^;]+?\]\s*=\s*)value;/g, `$1${storageType}(value);`);
  }
  return source;
}
