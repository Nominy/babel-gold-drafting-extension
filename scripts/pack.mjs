#!/usr/bin/env node

import { readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { collectFiles, packExtension } from '@nominy/babel-extension-build';

const ROOT = resolve(import.meta.dirname, '..');
const tempManifestPath = join(ROOT, '.tmp.store.manifest.json');
const STORE_EXTERNALLY_CONNECTABLE_IDS = ['dldjgploldmldipplklepcpjdjhehald'];

try {
  await packExtension({
    rootDir: ROOT,
    skipBuild: process.argv.includes('--no-build'),
    buildCommand: {
      command: 'npm',
      args: ['run', 'build']
    },
    collectPackResult() {
      const manifest = JSON.parse(readFileSync(join(ROOT, 'manifest.json'), 'utf-8').replace(/^\uFEFF/, ''));
      const storeManifest = {
        ...manifest,
        externally_connectable: {
          ids: STORE_EXTERNALLY_CONNECTABLE_IDS
        },
        host_permissions: ['https://dashboard.babel.audio/*', 'https://reviewgen.ovh/*']
      };
      delete storeManifest.optional_host_permissions;
      writeFileSync(tempManifestPath, `${JSON.stringify(storeManifest, null, 2)}\n`);

      const files = [
        { full: tempManifestPath, rel: 'manifest.json' },
        { full: join(ROOT, 'options.html'), rel: 'options.html' }
      ];

      for (const entry of collectFiles(join(ROOT, 'icons'), 'icons')) {
        files.push(entry);
      }

      for (const entry of collectFiles(join(ROOT, 'dist'), 'dist')) {
        if (entry.full.endsWith('.js')) {
          files.push(entry);
        }
      }

      return {
        entries: files,
        zipName: `babel-gold-drafting-extension-${manifest.version}.zip`,
        zipOutputDir: process.env.BABEL_EXTENSION_ZIP_DIR ?? '.artifacts',
        zipPath: process.env.BABEL_EXTENSION_ZIP_PATH
      };
    }
  });
} finally {
  rmSync(tempManifestPath, { force: true });
}
