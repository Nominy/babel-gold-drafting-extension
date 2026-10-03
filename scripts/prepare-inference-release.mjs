#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { copyFile, cp, mkdir, readFile, readdir, realpath, stat, writeFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const argumentsMap = new Map();
for (let index = 2; index < process.argv.length; index += 2) {
  const key = process.argv[index], value = process.argv[index + 1];
  if (!['--bundle', '--sample-dir', '--out', '--evidence', '--coordinator-evidence'].includes(key) || !value || argumentsMap.has(key)) throw new Error('Use --bundle DIR --sample-dir DIR --out DIR --evidence JSON --coordinator-evidence JSON');
  argumentsMap.set(key, value);
}
for (const key of ['--bundle', '--sample-dir', '--out', '--evidence', '--coordinator-evidence']) if (!argumentsMap.has(key)) throw new Error(`Missing ${key}`);
const bundle = await realpath(argumentsMap.get('--bundle'));
const sample = await realpath(argumentsMap.get('--sample-dir'));
const output = path.resolve(argumentsMap.get('--out'));
const release = JSON.parse(await readFile(path.join(root, 'model-release.json'), 'utf8'));
const version = JSON.parse(await readFile(path.join(root, 'manifest.json'), 'utf8')).version;
if (version !== release.minimumExtensionVersion) throw new Error('Bump the extension to the reviewed release version before packaging.');
try { if ((await readdir(output)).length) throw new Error('Release destination must be empty; existing releases are never overwritten.'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
if (output === bundle || output.startsWith(`${bundle}${path.sep}`)) throw new Error('Release output must be separate from source model files.');
const manifest = JSON.parse(await readFile(path.join(bundle, 'manifest.json'), 'utf8'));
if (manifest.schema !== release.bundleSchema || !manifest.pass || !manifest.validation?.actualWebGPU?.pass) throw new Error('Only a numerically and hardware-validated v3 bundle can be prepared.');
const backendEvidence = JSON.parse(await readFile(argumentsMap.get('--evidence'), 'utf8'));
if (!backendEvidence.pass || !backendEvidence.labelsSurviveRestart || backendEvidence.health?.release !== release.id || backendEvidence.health?.provider !== 'webgpu') throw new Error('A successful actual backend/restart report is required.');
const coordinatorEvidence = JSON.parse(await readFile(argumentsMap.get('--coordinator-evidence'), 'utf8'));
if (!coordinatorEvidence.pass || coordinatorEvidence.release !== release.id || coordinatorEvidence.oldClientStatus !== 426 || !coordinatorEvidence.trustedFallback || !coordinatorEvidence.labelsSurviveCoordinatorRestart) throw new Error('A successful coordinator release-gate and restart report is required.');
for (const [name, expected] of Object.entries(release.graphs)) {
  if (manifest.files.find(file => file.path === name)?.sha256 !== expected) throw new Error(`Wrong released graph: ${name}`);
}
async function hash(filename) {
  const digest = createHash('sha256');
  for await (const bytes of createReadStream(filename)) digest.update(bytes);
  return digest.digest('hex');
}
for (const file of manifest.files) {
  if (typeof file.path !== 'string' || path.isAbsolute(file.path) || file.path.split(/[\\/]/).some(part => part === '..' || !part)) throw new Error('Unsafe bundle path');
  const source = await realpath(path.join(bundle, file.path));
  if (!source.startsWith(`${bundle}${path.sep}`) || (await stat(source)).size !== file.bytes || await hash(source) !== file.sha256) throw new Error(`Bundle integrity failed: ${file.path}`);
}
const environment = { ...process.env };
delete environment.BABEL_DEV_C_DENOISE_MODEL_URL;
function run(script, args = []) { execFileSync(process.execPath, [path.join(root, script), ...args], { cwd: root, env: environment, stdio: 'inherit' }); }
run('esbuild.config.mjs');
run('scripts/pack.mjs', ['--no-build']);
run('scripts/build-inference-backend.mjs');
await mkdir(output, { recursive: true });
const models = path.join(output, 'models', release.id);
for (const file of manifest.files) {
  const destination = path.join(models, file.path);
  await mkdir(path.dirname(destination), { recursive: true });
  await copyFile(path.join(bundle, file.path), destination);
}
// Private corpus sequences stay in local validation artifacts, not public model metadata.
function publicCase(row) {
  return { id: row.id, pass: row.pass, comparisons: row.comparisons,
    ...(row.ctcTokenAgreement === undefined ? {} : { ctcTokenAgreement: row.ctcTokenAgreement }),
    ...(row.encodedLengthsEqual === undefined ? {} : { encodedLengthsEqual: row.encodedLengthsEqual }) };
}
manifest.releaseId = release.id;
manifest.validation.numericExport.cases = manifest.validation.numericExport.cases.map(publicCase);
manifest.validation.numericExport.asr = manifest.validation.numericExport.asr.map(publicCase);
for (const key of ['asrCases', 'punctuationCases']) {
  if (Array.isArray(manifest.validation.actualWebGPU[key])) manifest.validation.actualWebGPU[key] = manifest.validation.actualWebGPU[key].map(publicCase);
}
await writeFile(path.join(models, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
for (const name of ['sample-russian-15s.wav', 'sample-russian-15s-license.txt']) await copyFile(path.join(sample, name), path.join(models, name));
await copyFile(path.join(root, 'model-release.json'), path.join(output, 'release.json'));
await copyFile(path.join(root, '.artifacts', `babel-gold-drafting-extension-${version}.zip`), path.join(output, `babel-gold-drafting-extension-${version}.zip`));
const backend = path.join(output, 'backend');
await cp(path.join(root, '.artifacts/inference-backend'), backend, { recursive: true });
await writeFile(path.join(backend, 'package.json'), JSON.stringify({ name: 'babel-c-denoise-trusted-backend', version, private: true, type: 'module', dependencies: { playwright: '1.63.0' } }, null, 2));
const npmCli = process.env.npm_execpath;
if (!npmCli) throw new Error('Run this preparation command through npm so the backend dependency lock can be generated.');
execFileSync(process.execPath, [npmCli, 'install', '--package-lock-only', '--ignore-scripts'], { cwd: backend, env: environment, stdio: 'inherit' });
const engine = path.resolve(root, '../l0-draft-engine');
await cp(path.join(engine, 'l0_draft_engine'), path.join(backend, 'l0_draft_engine'), { recursive: true,
  filter: source => !source.includes('__pycache__') && !source.endsWith('.pyc') });
for (const name of ['requirements-webgpu.txt', 'requirements-webgpu.lock', 'pyproject.toml']) await copyFile(path.join(engine, name), path.join(backend, name));
await copyFile(path.join(engine, 'requirements-webgpu.txt'), path.join(backend, 'requirements.txt'));
await mkdir(path.join(backend, 'scripts'), { recursive: true });
for (const name of ['Preflight-WebGPU.py', 'Smoke-WebGPU.py', 'Smoke-Coordinator-WebGPU.py']) await copyFile(path.join(engine, 'scripts', name), path.join(backend, 'scripts', name));
for (const name of ['LICENSE', 'COPYING.LGPLv2.1']) await copyFile(path.join(root, name), path.join(backend, name));
await cp(path.join(root, 'src/core'), path.join(backend, 'source/core'), { recursive: true });
await cp(path.join(root, 'deploy/inference-release'), path.join(output, 'deployment'), { recursive: true });
await mkdir(path.join(output, 'validation'), { recursive: true });
await copyFile(argumentsMap.get('--evidence'), path.join(output, 'validation/trusted-backend-webgpu.json'));
await copyFile(argumentsMap.get('--coordinator-evidence'), path.join(output, 'validation/coordinator-release-webgpu.json'));
const checksums = [];
async function inventory(directory) {
  for (const item of await readdir(directory, { withFileTypes: true })) {
    const filename = path.join(directory, item.name);
    if (item.isDirectory()) await inventory(filename);
    else checksums.push({ path: path.relative(output, filename).replaceAll('\\', '/'), bytes: (await stat(filename)).size, sha256: await hash(filename) });
  }
}
await inventory(output);
await writeFile(path.join(output, 'checksums.json'), JSON.stringify({ release: release.id, files: checksums }, null, 2));
console.log(`Prepared ${release.id}: ${output}`);
