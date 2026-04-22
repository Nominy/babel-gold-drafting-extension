#!/usr/bin/env node

import { resolve } from 'node:path';
import { runSetupGithubSecrets } from '@nominy/babel-extension-build';

await runSetupGithubSecrets({
  rootDir: resolve(import.meta.dirname, '..')
});
