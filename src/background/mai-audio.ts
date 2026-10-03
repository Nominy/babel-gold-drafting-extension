import { MaiError } from '../core/mai-protocol';

export const MAI_PROVIDER_AUDIO_BYTES = 24 * 1024 * 1024;
export const MAI_CHUNK_SECONDS = 180;
export interface MaiPcmAudio {
  bytes: Uint8Array<ArrayBuffer>;
  pcm: Uint8Array<ArrayBuffer>;
  sampleRate: number;
  channels: number;
  blockAlign: number;
  frameCount: number;
  durationSeconds: number;
  pcmSha256: string;
}
export interface MaiAudioChunk {
  startSample: number;
  endSample: number;
  wav: Blob;
}
function tag(bytes: Uint8Array, offset: number): string {
  return String.fromCharCode(...bytes.subarray(offset, offset + 4));
}
export async function parseMaiPcmWav(bytes: Uint8Array<ArrayBuffer>): Promise<MaiPcmAudio> {
  if (bytes.length < 44 || tag(bytes, 0) !== 'RIFF' || tag(bytes, 8) !== 'WAVE') {
    throw new MaiError('invalid-audio', 'MAI transcription requires a real PCM16 WAV file.');
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const riffEnd = view.getUint32(4, true) + 8;
  if (riffEnd !== bytes.length) throw new MaiError('invalid-audio', 'WAV RIFF length does not match its bytes.');
  let sampleRate = 0;
  let channels = 0;
  let blockAlign = 0;
  let dataOffset = -1;
  let dataLength = 0;
  let formatSeen = false;
  for (let offset = 12; offset < riffEnd;) {
    if (offset + 8 > riffEnd) throw new MaiError('invalid-audio', 'WAV chunk header is truncated.');
    const length = view.getUint32(offset + 4, true);
    const start = offset + 8;
    if (start + length > riffEnd) throw new MaiError('invalid-audio', 'WAV chunk extends beyond the actual file.');
    const kind = tag(bytes, offset);
    if (kind === 'fmt ') {
      if (formatSeen || length < 16) throw new MaiError('invalid-audio', 'WAV format chunk is invalid.');
      formatSeen = true;
      const format = view.getUint16(start, true);
      channels = view.getUint16(start + 2, true);
      sampleRate = view.getUint32(start + 4, true);
      blockAlign = view.getUint16(start + 12, true);
      const bits = view.getUint16(start + 14, true);
      if (format !== 1 || bits !== 16 || channels < 1 || channels > 2 || sampleRate < 8000 || sampleRate > 192000 ||
          blockAlign !== channels * 2 || view.getUint32(start + 8, true) !== sampleRate * blockAlign) {
        throw new MaiError('invalid-audio', 'WAV must contain supported PCM16 sample geometry.');
      }
    } else if (kind === 'data') {
      if (dataOffset !== -1) throw new MaiError('invalid-audio', 'Multiple WAV data chunks are not supported.');
      dataOffset = start;
      dataLength = length;
    }
    offset = start + length + (length & 1);
    if (offset > riffEnd) throw new MaiError('invalid-audio', 'WAV chunk padding is truncated.');
  }
  if (!formatSeen || dataOffset < 0 || dataLength === 0 || dataLength % blockAlign !== 0) {
    throw new MaiError('invalid-audio', 'WAV has no complete PCM frames.');
  }
  const pcm = bytes.subarray(dataOffset, dataOffset + dataLength);
  const digest = await crypto.subtle.digest('SHA-256', pcm);
  const pcmSha256 = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
  const frameCount = dataLength / blockAlign;
  return { bytes, pcm, sampleRate, channels, blockAlign, frameCount, durationSeconds: frameCount / sampleRate, pcmSha256 };
}
function chunkWav(audio: MaiPcmAudio, start: number, end: number): Blob {
  const pcm = audio.pcm.subarray(start * audio.blockAlign, end * audio.blockAlign);
  const header = new ArrayBuffer(44);
  const view = new DataView(header);
  for (const [offset, name] of [[0, 'RIFF'], [8, 'WAVE'], [12, 'fmt '], [36, 'data']] as const) {
    for (let index = 0; index < name.length; index += 1) view.setUint8(offset + index, name.charCodeAt(index));
  }
  view.setUint32(4, 36 + pcm.length, true);
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, audio.channels, true);
  view.setUint32(24, audio.sampleRate, true);
  view.setUint32(28, audio.sampleRate * audio.blockAlign, true);
  view.setUint16(32, audio.blockAlign, true);
  view.setUint16(34, 16, true);
  view.setUint32(40, pcm.length, true);
  return new Blob([header, pcm], { type: 'audio/wav' });
}
export function splitMaiAudio(audio: MaiPcmAudio): MaiAudioChunk[] {
  const maxFrames = Math.min(Math.floor((MAI_PROVIDER_AUDIO_BYTES - 44) / audio.blockAlign), audio.sampleRate * MAI_CHUNK_SECONDS);
  const chunks: MaiAudioChunk[] = [];
  const pcmView = new DataView(audio.pcm.buffer, audio.pcm.byteOffset, audio.pcm.byteLength);
  for (let start = 0; start < audio.frameCount;) {
    let end = Math.min(audio.frameCount, start + maxFrames);
    if (end < audio.frameCount) {
      // Disjoint ownership: every real sample occurs once. Prefer a quiet 20ms frame
      // near the bound, rather than duplicate/remove recognized boundary words.
      const windowFrames = Math.floor(audio.sampleRate / 50);
      const searchStart = Math.max(start + Math.floor(maxFrames / 2), end - audio.sampleRate * 5);
      let lowestEnergy = Infinity;
      let quietEnd = end;
      for (let candidate = searchStart; candidate + windowFrames <= end; candidate += windowFrames) {
        let energy = 0;
        for (let frame = candidate; frame < candidate + windowFrames; frame += 1) {
          for (let channel = 0; channel < audio.channels; channel += 1) {
            const sample = pcmView.getInt16(frame * audio.blockAlign + channel * 2, true);
            energy += sample * sample;
          }
        }
        if (energy <= lowestEnergy) {
          lowestEnergy = energy;
          quietEnd = candidate + Math.floor(windowFrames / 2);
        }
      }
      if (lowestEnergy / (windowFrames * audio.channels) <= 64 * 64) end = quietEnd;
    }
    chunks.push({ startSample: start, endSample: end, wav: chunkWav(audio, start, end) });
    start = end;
  }
  return chunks;
}
