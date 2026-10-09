import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import test from 'node:test';
import { encodeEnhancementWav } from '../src/core/audio-enhancement-dsp';
import type { PlacementGraph } from '../src/core/local-gpu-placement';
import { hashCampaignFile, verifyCaptureCampaign, type CaptureCampaign, type CampaignIdentity } from '../scripts/zip-profile-campaign.mjs';

async function captureFixture(directory: string) {
  // Metadata-validation fixture only; no synthetic result is used as real inference evidence.
  const modelSha256 = createHash('sha256').update('unit-test model identity').digest('hex');
  const identity: CampaignIdentity = { modelSha256, audioSha256: [createHash('sha256').update('unit-test source identity').digest('hex')], baseline: false,
    ortSourceHashes: { upstream: 'recorded-source-hash' }, kernelHelperHashes: { helper: 'recorded-helper-hash' } };
  const graph: PlacementGraph = { path: 'models/zipenhancer.onnx', sha256: modelSha256,
    nodes: [{ name: 'fixture/Add', opType: 'Add', placement: 'gpu', inputs: [], outputs: [], reason: 'metadata validator fixture' }] };
  const data = Buffer.alloc(16, 7), spectral = Buffer.alloc(201 * 641 * 4), uniform = Buffer.alloc(16);
  for (const [name, bytes] of [['config-0-input-0.bin', data], ['config-0-uniform.bin', uniform], ['spectrum-0-mag.bin', spectral], ['spectrum-0-pha.bin', spectral]] as const) {
    await writeFile(path.join(directory, name), bytes);
  }
  const wav = encodeEnhancementWav(new Float32Array([0.25]), 16000, 1);
  await writeFile(path.join(directory, 'enhanced-0.wav'), wav);
  const tensor = { dims: [1], dataType: 1, elements: 1, logicalBytes: 4, bufferBytes: 16 };
  const manifest: CaptureCampaign = { schema: 'babel-zip-kernel-capture-v1', identity, sourcePaths: ['fixture.wav'], stage: 'unit-test metadata fixture', missing: [], modelInputsComplete: true,
    configurations: [{ id: 0, program: 'Add', cacheKey: 'fixture', shader: '@compute @workgroup_size(1) fn main() {}', baselineShader: '@compute @workgroup_size(1) fn main() {}',
      inputs: [{ ...tensor, file: 'config-0-input-0.bin', sha256: await hashCampaignFile(path.join(directory, 'config-0-input-0.bin')) }], outputs: [tensor],
      uniforms: [{ type: 12, data: 1 }], uniformBytes: 16, uniformFile: 'config-0-uniform.bin', uniformSha256: await hashCampaignFile(path.join(directory, 'config-0-uniform.bin')),
      dispatchGroup: [1, 1, 1], count: 1, gpuDurationNs: 10, sources: { 'Add|fixture/Add': 1 }, dataOrigin: 'unit-test fixture, not model inference', captured: true, captureComplete: true,
      introducedPhase: 'inference', introducingSpectrumId: 0 }],
    spectra: [{ id: 0, frames: 641, mag: { file: 'spectrum-0-mag.bin', sha256: await hashCampaignFile(path.join(directory, 'spectrum-0-mag.bin')) },
      pha: { file: 'spectrum-0-pha.bin', sha256: await hashCampaignFile(path.join(directory, 'spectrum-0-pha.bin')) } }],
    occurrences: [{ version: 1, configId: 0, phase: 'inference', run: 0, kernelName: 'fixture/Add', kernelType: 'Add', programName: 'Add', startTime: 0, endTime: 10 }],
    audits: [{ path: graph.path, sha256: modelSha256, verifiedRuns: 1, requiredGpuNodes: 1, verifiedGpuNodes: 1, gpuPrograms: 1, allowedHostMetadataNodes: 0, storageAliasNodes: 0, constantNodes: 0 }],
    waves: [{ index: 0, sampleRate: 16000, frames: 1, wavSha256: await hashCampaignFile(path.join(directory, 'enhanced-0.wav')), wallMs: 1 }] };
  const filename = path.join(directory, 'capture-manifest-0.json');
  await writeFile(filename, JSON.stringify(manifest));
  return { filename, manifest, identity, graph };
}

test('campaign recovery validates full file bytes, exact source identities and complete per-run GPU source coverage', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'zip-campaign-'));
  try {
    const fixture = await captureFixture(directory);
    assert.deepEqual(await verifyCaptureCampaign(fixture.filename, fixture.identity, fixture.graph), fixture.manifest);
    await assert.rejects(verifyCaptureCampaign(fixture.filename, { ...fixture.identity, audioSha256: ['changed'] }, fixture.graph), /exact model, audio/);
    await assert.rejects(verifyCaptureCampaign(fixture.filename, { ...fixture.identity, kernelHelperHashes: { helper: 'changed' } }, fixture.graph), /kernelHelperHashes changed/);
    const bytes = await readFile(path.join(directory, 'config-0-input-0.bin')); bytes[7] ^= 1;
    await writeFile(path.join(directory, 'config-0-input-0.bin'), bytes);
    await assert.rejects(verifyCaptureCampaign(fixture.filename, fixture.identity, fixture.graph), /file identity failed/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('campaign recovery rejects altered counts, missing original spectra and incomplete source-node audits', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'zip-campaign-'));
  try {
    const fixture = await captureFixture(directory);
    for (const mutation of [
      (manifest: CaptureCampaign) => { manifest.configurations[0].count = 2; },
      (manifest: CaptureCampaign) => { manifest.configurations[0].introducingSpectrumId = 9; },
      (manifest: CaptureCampaign) => { manifest.occurrences[0].kernelName = 'unknown'; },
      (manifest: CaptureCampaign) => { manifest.audits[0] = { ...manifest.audits[0], verifiedGpuNodes: 0 }; },
    ]) {
      const manifest = structuredClone(fixture.manifest); mutation(manifest);
      await writeFile(fixture.filename, JSON.stringify(manifest));
      await assert.rejects(verifyCaptureCampaign(fixture.filename, fixture.identity, fixture.graph));
    }
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('pending captures remain pending rather than promoting partial buffers to verified model evidence', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'zip-campaign-'));
  try {
    const fixture = await captureFixture(directory), manifest = structuredClone(fixture.manifest);
    manifest.configurations[0].captureComplete = false;
    delete manifest.configurations[0].inputs[0].file; delete manifest.configurations[0].inputs[0].sha256;
    delete manifest.configurations[0].uniformFile; delete manifest.configurations[0].uniformSha256;
    await writeFile(fixture.filename, JSON.stringify(manifest));
    const recovered = await verifyCaptureCampaign(fixture.filename, fixture.identity, fixture.graph);
    assert.equal(recovered.configurations[0].captureComplete, false);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
