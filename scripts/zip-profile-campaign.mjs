import path from 'node:path';
import { createReadStream } from 'node:fs';
import { readFile, stat } from 'node:fs/promises';
import { createHash } from 'node:crypto';

export async function hashCampaignFile(filename) {
  const digest = createHash('sha256');
  for await (const chunk of createReadStream(filename)) digest.update(chunk);
  return digest.digest('hex');
}

// A recovery campaign never infers configuration identity from filenames alone.
// Complete buffers, spectra, and WAVs must have their original recorded hashes.
export async function verifyCaptureCampaign(filename, identity, graph) {
  const manifest = JSON.parse(await readFile(filename, 'utf8'));
  if (graph.sha256 !== identity.modelSha256) throw new Error('Capture campaign is not bound to the current GPU placement graph');
  if (!manifest || manifest.schema !== 'babel-zip-kernel-capture-v1' || !manifest.identity ||
      manifest.identity.modelSha256 !== identity.modelSha256 || manifest.identity.baseline !== identity.baseline ||
      JSON.stringify(manifest.identity.audioSha256) !== JSON.stringify(identity.audioSha256) ||
      !Array.isArray(manifest.sourcePaths) || manifest.sourcePaths.length !== identity.audioSha256.length ||
      !Array.isArray(manifest.configurations) || !Array.isArray(manifest.spectra) || !Array.isArray(manifest.occurrences) ||
      !Array.isArray(manifest.audits) || !Array.isArray(manifest.waves) || typeof manifest.modelInputsComplete !== 'boolean') {
    throw new Error('Capture recovery requires a recorded manifest bound to the exact model, audio sources and kernel mode');
  }
  for (const key of ['ortSourceHashes', 'kernelHelperHashes']) {
    if (!manifest.identity[key] || Object.keys(manifest.identity[key]).length !== Object.keys(identity[key]).length ||
        Object.entries(identity[key]).some(([name, hash]) => manifest.identity[key][name] !== hash)) {
      throw new Error(`Capture recovery ${key} changed; previous data cannot be reidentified safely`);
    }
  }
  const directory = path.dirname(filename), verified = new Set();
  async function verifyFile(name, bytes, hash) {
    if (typeof name !== 'string' || !/^(?:config-\d+-(?:input-\d+|uniform)\.bin|spectrum-\d+-(?:mag|pha)\.bin|enhanced-\d+\.wav)$/.test(name) ||
        !Number.isSafeInteger(bytes) || bytes < 0 || typeof hash !== 'string' || !/^[a-f0-9]{64}$/.test(hash)) throw new Error('Capture file lacks a safe name, byte length or original SHA256');
    if (verified.has(name)) throw new Error(`Capture manifest assigns a file twice: ${name}`);
    verified.add(name);
    const file = path.join(directory, name);
    if ((await stat(file)).size !== bytes || await hashCampaignFile(file) !== hash) throw new Error(`Capture file identity failed: ${name}`);
  }
  const spectra = new Map();
  for (const spectrum of manifest.spectra) {
    if (!spectrum || !Number.isSafeInteger(spectrum.id) || spectrum.id < 0 || spectrum.frames !== 641 || spectra.has(spectrum.id)) throw new Error('Invalid original spectrum inventory');
    spectra.set(spectrum.id, spectrum);
    for (const channel of ['mag', 'pha']) {
      if (spectrum[channel]?.file !== `spectrum-${spectrum.id}-${channel}.bin`) throw new Error('Original spectrum file identity changed');
      await verifyFile(spectrum[channel].file, 201 * 641 * 4, spectrum[channel].sha256);
    }
  }
  const widths = { 1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8, 12: 4, 13: 8, 16: 2 };
  for (let id = 0; id < manifest.configurations.length; id++) {
    const config = manifest.configurations[id];
    if (!config || config.id !== id || typeof config.program !== 'string' || typeof config.shader !== 'string' || typeof config.baselineShader !== 'string' ||
        typeof config.cacheKey !== 'string' || !Array.isArray(config.inputs) || !Array.isArray(config.outputs) || !Array.isArray(config.uniforms) ||
        !Array.isArray(config.dispatchGroup) || config.dispatchGroup.length !== 3 || config.dispatchGroup.some(n => !Number.isSafeInteger(n) || n < 1) ||
        !Number.isSafeInteger(config.count) || config.count < 0 || !Number.isSafeInteger(config.gpuDurationNs) || config.gpuDurationNs < 0 ||
        !Number.isSafeInteger(config.uniformBytes) || config.uniformBytes < 0 || config.uniformBytes % 4 !== 0 || typeof config.captureComplete !== 'boolean' ||
        typeof config.captured !== 'boolean' || typeof config.dataOrigin !== 'string' || !config.sources || typeof config.sources !== 'object' ||
        !['initialization', 'inference'].includes(config.introducedPhase) ||
        (config.introducedPhase === 'inference' && config.introducingSpectrumId === null) ||
        (config.introducingSpectrumId !== null && !spectra.has(config.introducingSpectrumId))) throw new Error(`Invalid capture configuration ${id}`);
    for (const uniform of config.uniforms) {
      if (![1, 6, 10, 12].includes(uniform.type) || !(typeof uniform.data === 'number' && Number.isFinite(uniform.data) ||
          Array.isArray(uniform.data) && uniform.data.every(Number.isFinite))) throw new Error(`Invalid captured uniform ${id}`);
    }
    for (const tensor of [...config.inputs, ...config.outputs]) {
      if (!tensor || !Array.isArray(tensor.dims) || tensor.dims.some(n => !Number.isSafeInteger(n) || n < 1) || !widths[tensor.dataType]) throw new Error(`Invalid captured tensor ${id}`);
      const elements = tensor.dims.reduce((a, b) => a * b, 1), bytes = elements * widths[tensor.dataType];
      if (tensor.elements !== elements || tensor.logicalBytes !== bytes || tensor.bufferBytes !== Math.max(16, Math.ceil(bytes / 16) * 16)) throw new Error(`Captured tensor size changed: ${id}`);
    }
    if (config.captureComplete) {
      for (let input = 0; input < config.inputs.length; input++) {
        const tensor = config.inputs[input];
        if (tensor.file !== `config-${id}-input-${input}.bin`) throw new Error(`Captured input name changed: ${id}`);
        await verifyFile(tensor.file, tensor.bufferBytes, tensor.sha256);
      }
      if (config.uniformBytes) {
        if (config.uniformFile !== `config-${id}-uniform.bin`) throw new Error(`Captured uniform name changed: ${id}`);
        await verifyFile(config.uniformFile, config.uniformBytes, config.uniformSha256);
      }
    }
  }
  const counts = new Array(manifest.configurations.length).fill(0), durations = counts.slice(), runs = new Map();
  const required = new Map(graph.nodes.filter(n => n.placement === 'gpu').map(n => [n.name, n.opType]));
  for (const occurrence of manifest.occurrences) {
    if (!occurrence || occurrence.version !== 1 || !manifest.configurations[occurrence.configId] || !Number.isSafeInteger(occurrence.startTime) ||
        !Number.isSafeInteger(occurrence.endTime) || occurrence.endTime < occurrence.startTime || typeof occurrence.kernelName !== 'string' ||
        typeof occurrence.kernelType !== 'string' || !['initialization', 'inference'].includes(occurrence.phase)) throw new Error('Invalid captured GPU occurrence');
    counts[occurrence.configId]++; durations[occurrence.configId] += occurrence.endTime - occurrence.startTime;
    if (occurrence.phase === 'inference') {
      if (!Number.isSafeInteger(occurrence.run) || occurrence.run < 0) throw new Error('Captured inference has no original run identity');
      const observed = runs.get(occurrence.run) ?? new Set(); runs.set(occurrence.run, observed);
      if (required.get(occurrence.kernelName) === occurrence.kernelType) observed.add(occurrence.kernelName);
    }
  }
  for (const config of manifest.configurations) if (counts[config.id] !== config.count || durations[config.id] !== config.gpuDurationNs) throw new Error(`Captured cumulative count/duration changed: ${config.id}`);
  if (manifest.modelInputsComplete) {
    if (manifest.waves.length !== identity.audioSha256.length || runs.size !== manifest.audits.length || !runs.size ||
        [...runs.values()].some(observed => observed.size !== required.size) || manifest.audits.some(a => a.sha256 !== identity.modelSha256 ||
          a.verifiedGpuNodes !== required.size || a.requiredGpuNodes !== required.size || a.verifiedRuns !== 1 || a.path !== graph.path)) {
      throw new Error('Capture campaign lacks complete original-lane GPU source-node proofs');
    }
    for (let index = 0; index < manifest.waves.length; index++) {
      const wave = manifest.waves[index];
      if (wave.index !== index || !Number.isSafeInteger(wave.frames) || wave.frames < 1) throw new Error('Invalid captured original-lane WAV');
      await verifyFile(`enhanced-${index}.wav`, 44 + wave.frames * 2, wave.wavSha256);
    }
  }
  return manifest;
}
