#!/usr/bin/env node

import { execSync } from 'node:child_process';
import { readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { createDeflateRaw } from 'node:zlib';

const ROOT = resolve(import.meta.dirname, '..');
const skipBuild = process.argv.includes('--no-build');

if (!skipBuild) {
  console.log('Building...');
  execSync('npm run build', { cwd: ROOT, stdio: 'inherit' });
}

const manifestPath = join(ROOT, 'manifest.json');
const manifest = JSON.parse(readFileSync(manifestPath, 'utf-8').replace(/^\uFEFF/, ''));
const version = manifest.version;
const zipName = `babel-gold-drafting-extension-${version}.zip`;
const zipPath = resolve(ROOT, '..', zipName);

function collectFiles(dir, base) {
  const results = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    const rel = join(base, entry.name);
    if (entry.isDirectory()) {
      results.push(...collectFiles(full, rel));
    } else {
      results.push({ full, rel });
    }
  }
  return results;
}

const storeManifest = {
  ...manifest,
  host_permissions: [
    'https://dashboard.babel.audio/*',
    'https://reviewgen.ovh/*'
  ]
};

const tempManifestPath = join(ROOT, '.tmp.store.manifest.json');
writeFileSync(tempManifestPath, `${JSON.stringify(storeManifest, null, 2)}\n`);

const files = [
  { full: tempManifestPath, rel: 'manifest.json' },
  { full: join(ROOT, 'options.html'), rel: 'options.html' }
];

for (const { full, rel } of collectFiles(join(ROOT, 'icons'), 'icons')) {
  files.push({ full, rel });
}

for (const { full, rel } of collectFiles(join(ROOT, 'dist'), 'dist')) {
  if (full.endsWith('.js')) {
    files.push({ full, rel });
  }
}

function crc32(buf) {
  let crc = 0xffffffff;
  for (let i = 0; i < buf.length; i += 1) {
    crc ^= buf[i];
    for (let j = 0; j < 8; j += 1) {
      crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0);
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function dosDateTime(date) {
  const time =
    ((date.getHours() & 0x1f) << 11) |
    ((date.getMinutes() & 0x3f) << 5) |
    ((date.getSeconds() >> 1) & 0x1f);
  const d =
    (((date.getFullYear() - 1980) & 0x7f) << 9) |
    (((date.getMonth() + 1) & 0x0f) << 5) |
    (date.getDate() & 0x1f);
  return { time, date: d };
}

function writeUInt32LE(buf, val, off) {
  buf.writeUInt32LE(val >>> 0, off);
}

function writeUInt16LE(buf, val, off) {
  buf.writeUInt16LE(val & 0xffff, off);
}

async function deflate(data) {
  const chunks = [];
  const deflater = createDeflateRaw({ level: 9 });
  deflater.on('data', (chunk) => chunks.push(chunk));
  deflater.end(data);
  await new Promise((resolve, reject) => {
    deflater.on('end', resolve);
    deflater.on('error', reject);
  });
  return Buffer.concat(chunks);
}

async function createZip(outPath, entries) {
  const centralHeaders = [];
  const parts = [];
  let offset = 0;
  const now = new Date();
  const { time: dosTime, date: dosDate } = dosDateTime(now);

  for (const { rel, full } of entries) {
    const raw = readFileSync(full);
    const crc = crc32(raw);
    const compressed = await deflate(raw);
    const useDeflate = compressed.length < raw.length;
    const method = useDeflate ? 8 : 0;
    const stored = useDeflate ? compressed : raw;
    const nameBytes = Buffer.from(rel.replace(/\\/g, '/'), 'utf-8');

    const local = Buffer.alloc(30 + nameBytes.length);
    writeUInt32LE(local, 0x04034b50, 0);
    writeUInt16LE(local, 20, 4);
    writeUInt16LE(local, 0, 6);
    writeUInt16LE(local, method, 8);
    writeUInt16LE(local, dosTime, 10);
    writeUInt16LE(local, dosDate, 12);
    writeUInt32LE(local, crc, 14);
    writeUInt32LE(local, stored.length, 18);
    writeUInt32LE(local, raw.length, 22);
    writeUInt16LE(local, nameBytes.length, 26);
    writeUInt16LE(local, 0, 28);
    nameBytes.copy(local, 30);

    parts.push(local, stored);

    const central = Buffer.alloc(46 + nameBytes.length);
    writeUInt32LE(central, 0x02014b50, 0);
    writeUInt16LE(central, 20, 4);
    writeUInt16LE(central, 20, 6);
    writeUInt16LE(central, 0, 8);
    writeUInt16LE(central, method, 10);
    writeUInt16LE(central, dosTime, 12);
    writeUInt16LE(central, dosDate, 14);
    writeUInt32LE(central, crc, 16);
    writeUInt32LE(central, stored.length, 20);
    writeUInt32LE(central, raw.length, 24);
    writeUInt16LE(central, nameBytes.length, 28);
    writeUInt16LE(central, 0, 30);
    writeUInt16LE(central, 0, 32);
    writeUInt16LE(central, 0, 34);
    writeUInt16LE(central, 0, 36);
    writeUInt32LE(central, 0, 38);
    writeUInt32LE(central, offset, 42);
    nameBytes.copy(central, 46);
    centralHeaders.push(central);

    offset += local.length + stored.length;
  }

  const centralStart = offset;
  let centralSize = 0;
  for (const central of centralHeaders) {
    parts.push(central);
    centralSize += central.length;
  }

  const eocd = Buffer.alloc(22);
  writeUInt32LE(eocd, 0x06054b50, 0);
  writeUInt16LE(eocd, 0, 4);
  writeUInt16LE(eocd, 0, 6);
  writeUInt16LE(eocd, entries.length, 8);
  writeUInt16LE(eocd, entries.length, 10);
  writeUInt32LE(eocd, centralSize, 12);
  writeUInt32LE(eocd, centralStart, 16);
  writeUInt16LE(eocd, 0, 20);
  parts.push(eocd);

  writeFileSync(outPath, Buffer.concat(parts));
}

console.log(`Packing ${files.length} files...`);
for (const file of files) {
  console.log(`  ${file.rel}`);
}

try {
  await createZip(zipPath, files);
  const stat = statSync(zipPath);
  console.log(`\nCreated ${zipName} (${(stat.size / 1024).toFixed(1)} KB)`);
} finally {
  rmSync(tempManifestPath, { force: true });
}
