#!/usr/bin/env node

import { cpSync, existsSync, mkdirSync, rmSync } from 'node:fs';
import { join, resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const target = join(root, 'babel-gold-drafting-extension');
const entries = ['manifest.json', 'options.html', 'offscreen.html', 'icons', 'dist', 'LICENSE', 'COPYING.LGPLv2.1'];
const sources = ['ffmpeg-audio-raw.ts', 'ffmpeg-audio-denoise.ts'];

for (const entry of [...entries, ...sources.map((name) => `src/core/${name}`)]) {
  if (!existsSync(join(root, entry))) throw new Error(`Required build input is missing: ${entry}`);
}
rmSync(target, { recursive: true, force: true });
mkdirSync(join(target, 'source'), { recursive: true });
for (const entry of entries) {
  cpSync(join(root, entry), join(target, entry), { recursive: true });
}
for (const name of sources) {
  cpSync(join(root, 'src/core', name), join(target, 'source', name));
}
console.log(`Load unpacked: ${target}`);
