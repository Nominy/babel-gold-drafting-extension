import { readFile, mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { enhanceAudioThroughSwarm } from '../src/core/audio-enhancement-swarm';
import { enhancementSha256 } from '../src/core/audio-enhancement-cache';
import type { PreparedEnhancementSource } from '../src/core/audio-enhancement-cache';
import { readEnhancementWavHeader } from '../src/core/audio-enhancement-dsp';
import { parseEnhancementModel } from '../src/core/audio-enhancement-swarm-protocol';

const audio: string[] = [];
let coordinator = '', out = '';
for (let index = 2; index < process.argv.length; index += 2) {
  const key = process.argv[index], value = process.argv[index + 1];
  if (!value) throw new Error(`Missing value for ${key}`);
  if (key === '--audio') audio.push(value);
  else if (key === '--coordinator') coordinator = value;
  else if (key === '--out') out = value;
  else throw new Error(`Unknown argument ${key}`);
}
if (!coordinator || !out || audio.length !== 2) throw new Error('Supply --coordinator URL --audio WAV --audio WAV --out NEW_PRIVATE_DIRECTORY');
const artifact = JSON.parse(await readFile(new URL('../src/core/audio-enhancement-webgpu-model.json', import.meta.url), 'utf8'));
const model = parseEnhancementModel({ id: artifact.id, sha256: artifact.sha256, sourceGraphSha256: artifact.sourceGraphSha256 });
const sources: PreparedEnhancementSource[] = await Promise.all(audio.map(async (file, index) => {
  const bytes = Uint8Array.from(await readFile(file)).buffer;
  const clock = readEnhancementWavHeader(bytes), trackId = `smoke-speaker-${index + 1}`;
  return { trackId, bytes, sampleRate: clock.sampleRate, frameCount: clock.frameCount, sourceSha256: await enhancementSha256(bytes),
    track: { trackId, speakerKey: trackId, trackLabel: trackId, mimeType: 'audio/wav', source: 'explicit-smoke-audio', blob: new Blob([bytes], { type: 'audio/wav' }) } };
}));
await mkdir(out, { recursive: false });
const started = performance.now();
const result = await enhanceAudioThroughSwarm(sources, model, { taskId: `zip-smoke:${crypto.randomUUID()}`, baseUrl: coordinator,
  onProgress: progress => { console.log(JSON.stringify({ elapsedMs: performance.now() - started, ...progress })); },
});
const elapsedMs = performance.now() - started;
for (let index = 0; index < result.length; index++) await writeFile(path.join(out, `speaker-${index + 1}.wav`), result[index].bytes);
const report = { coordinator, model, elapsedMs, tracks: result.map(track => track.metadata) };
await writeFile(path.join(out, 'report.json'), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
