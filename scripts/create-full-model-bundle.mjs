#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { constants, copyFileSync, createReadStream, existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';

const [publishedBundle, asrGraph, punctuationGraph, destination] = process.argv.slice(2);
if (!publishedBundle || !asrGraph || !punctuationGraph || !destination) {
  throw new Error('Usage: create-full-model-bundle.mjs OLD_BUNDLE FULL_ASR_ONNX FULL_PUNCTUATION_ONNX OUTPUT_DIRECTORY');
}

const output = resolve(destination);
if (existsSync(output) && readdirSync(output).length) {
  throw new Error(`Output directory must be empty: ${output}`);
}
const previous = JSON.parse(readFileSync(join(publishedBundle, 'manifest.json'), 'utf8'));
const expectedGraphs = {
  'asr/v3_ctc.onnx': {
    source: asrGraph,
    sha256: '761dbc0ca4d55f1ee0643c739af29616d1ee601283d12f6e1a573fcf63f74304'
  },
  'punctuation/model.fp16.onnx': {
    source: punctuationGraph,
    sha256: '75e1308d2d15847467b63339a658bec1aef66e9858c09013dfa576b9ea8d4923'
  }
};
const files = [];
for (const file of previous.files) {
  const path = file.path === 'punctuation/model.int8.onnx' ? 'punctuation/model.fp16.onnx' : file.path;
  if (path === 'punctuation/distillation_metrics.json') continue;
  const graph = expectedGraphs[path];
  const source = graph?.source ?? join(publishedBundle, file.path);
  const target = join(output, path);
  mkdirSync(dirname(target), { recursive: true });
  copyFileSync(source, target, constants.COPYFILE_FICLONE);
  const digest = createHash('sha256');
  for await (const chunk of createReadStream(target)) digest.update(chunk);
  const sha256 = digest.digest('hex');
  if (graph && sha256 !== graph.sha256) {
    throw new Error(`Full model ${path} has unexpected SHA-256 ${sha256}`);
  }
  files.push({ path, bytes: statSync(target).size, sha256, role: file.role });
}
for (const path of Object.keys(expectedGraphs)) {
  if (!files.some((file) => file.path === path)) throw new Error(`Missing model graph ${path}`);
}
// The Options smoke-test fetches this public-domain WAV outside the model manifest.
copyFileSync(
  join(publishedBundle, 'sample-russian-15s.wav'),
  join(output, 'sample-russian-15s.wav'),
  constants.COPYFILE_FICLONE
);
copyFileSync(
  join(publishedBundle, 'sample-russian-15s-license.txt'),
  join(output, 'sample-russian-15s-license.txt')
);
const totalBytes = files.reduce((sum, file) => sum + file.bytes, 0);
const targetBytes = 1_500_000_000;
if (totalBytes > targetBytes) throw new Error(`Full model bundle exceeds ${targetBytes} bytes`);
writeFileSync(join(output, 'manifest.json'), JSON.stringify({
  schema: 'babel-browser-model-bundle-v2',
  targetBytes,
  totalBytes,
  pass: true,
  files,
  models: {
    asr: { graph: 'asr/v3_ctc.onnx', checkpointSha256: '02cea9973d0e839f6a3eeca101b83a93f93a066c2da2e3ebfa176d57e61d84d3', weights: 'float16' },
    punctuation: { graph: 'punctuation/model.fp16.onnx', source: 'punctuation-production-spacing', weights: 'float16' }
  }
}, null, 2) + '\n');
console.log(`Verified ${files.length} files, ${totalBytes} bytes in ${output}`);
