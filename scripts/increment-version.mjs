#!/usr/bin/env node

import { readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

const rootDir = resolve(import.meta.dirname, '..');

function readJson(relativePath) {
  const text = readFileSync(join(rootDir, relativePath), 'utf8').replace(/^\uFEFF/, '');
  return JSON.parse(text);
}

function writeJson(relativePath, value) {
  writeFileSync(join(rootDir, relativePath), `${JSON.stringify(value, null, 2)}\n`);
}

function incrementPatchVersion(version) {
  const match = /^(\d+)\.(\d+)\.(\d+)$/.exec(version);
  if (!match) {
    throw new Error(`Unsupported version format: ${version}`);
  }

  const [, major, minor, patch] = match;
  return `${major}.${minor}.${Number(patch) + 1}`;
}

const packageJson = readJson('package.json');
const manifestJson = readJson('manifest.json');
const packageLockJson = readJson('package-lock.json');

const currentVersion = packageJson.version;
if (typeof currentVersion !== 'string' || !currentVersion.trim()) {
  throw new Error('package.json is missing a valid version field');
}

const nextVersion = incrementPatchVersion(currentVersion);

packageJson.version = nextVersion;
manifestJson.version = nextVersion;

if (packageLockJson && typeof packageLockJson === 'object') {
  packageLockJson.version = nextVersion;
  if (packageLockJson.packages && packageLockJson.packages['']) {
    packageLockJson.packages[''].version = nextVersion;
  }
}

writeJson('package.json', packageJson);
writeJson('manifest.json', manifestJson);
writeJson('package-lock.json', packageLockJson);

console.log(`Version bumped: ${currentVersion} -> ${nextVersion}`);
