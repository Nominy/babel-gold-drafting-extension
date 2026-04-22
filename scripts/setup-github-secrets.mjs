#!/usr/bin/env node

import { execFileSync } from 'node:child_process';
import { resolve } from 'node:path';
import { getDefaultEnvFiles, loadCwsEnvironment, parseItemUrl } from './cws-env.mjs';

const rootDir = resolve(import.meta.dirname, '..');
const args = parseArgs(process.argv.slice(2));

if (args.flags.has('help')) {
  printHelp();
  process.exit(0);
}

const repo = args.values.get('repo') ?? args.positionals[0];
if (!repo) {
  throw new Error('Missing target repository. Pass OWNER/REPO as the first argument or via --repo.');
}

const envFile = args.values.get('env-file') ?? args.values.get('file');
const loaded = await loadCwsEnvironment(rootDir, envFile);
const values = loaded.values;
const itemUrl = values.CWS_ITEM_URL?.trim();
const publisherId = values.CWS_PUBLISHER_ID?.trim();
const extensionId = values.CWS_EXTENSION_ID?.trim();
const clientSecret = values.CWS_CLIENT_SECRET?.trim();
const refreshToken = values.CWS_REFRESH_TOKEN?.trim();
const accessToken = values.CWS_ACCESS_TOKEN?.trim();

if (!clientSecret || !refreshToken) {
  throw new Error(
    `Missing local Chrome Web Store credentials. Expected CWS_CLIENT_SECRET and CWS_REFRESH_TOKEN in ${
      loaded.filePath ?? 'the environment'
    }.`
  );
}

const itemTarget =
  publisherId && extensionId
    ? { publisherId, extensionId }
    : itemUrl
      ? parseItemUrl(itemUrl, loaded.filePath ?? 'CWS_ITEM_URL')
      : null;

if (!itemTarget) {
  throw new Error(
    `Missing Chrome Web Store item target. Set CWS_ITEM_URL or both CWS_PUBLISHER_ID and CWS_EXTENSION_ID in ${
      loaded.filePath ?? 'the environment'
    }.`
  );
}

const clientId = values.CWS_CLIENT_ID?.trim() ?? (await resolveClientId(accessToken, loaded.filePath));

const secrets = {
  CWS_CLIENT_ID: clientId,
  CWS_CLIENT_SECRET: clientSecret,
  CWS_REFRESH_TOKEN: refreshToken,
  CWS_PUBLISHER_ID: itemTarget.publisherId,
  CWS_EXTENSION_ID: itemTarget.extensionId
};

if (accessToken) {
  secrets.CWS_ACCESS_TOKEN = accessToken;
}

for (const [name, value] of Object.entries(secrets)) {
  execFileSync('gh', ['secret', 'set', name, '--repo', repo], {
    input: value,
    stdio: ['pipe', 'inherit', 'inherit']
  });
  console.log(`Set ${name} on ${repo}`);
}

console.log('GitHub Actions secrets updated successfully.');

function parseArgs(argv) {
  const values = new Map();
  const flags = new Set();
  const positionals = [];

  for (let index = 0; index < argv.length; index += 1) {
    const entry = argv[index];
    if (!entry.startsWith('--')) {
      positionals.push(entry);
      continue;
    }

    const trimmed = entry.slice(2);
    const equalsIndex = trimmed.indexOf('=');
    if (equalsIndex >= 0) {
      values.set(trimmed.slice(0, equalsIndex), trimmed.slice(equalsIndex + 1));
      continue;
    }

    const next = argv[index + 1];
    if (!next || next.startsWith('--')) {
      flags.add(trimmed);
      continue;
    }

    values.set(trimmed, next);
    index += 1;
  }

  return { values, flags, positionals };
}

function printHelp() {
  console.log(`Usage: node scripts/setup-github-secrets.mjs OWNER/REPO [options]

Options:
  --repo OWNER/REPO   Target GitHub repository
  --env-file PATH     Local dotenv file. Defaults to first readable file in:
                      ${getDefaultEnvFiles().join(', ')}
  --file PATH         Alias for --env-file
  --help              Show this help

Supported dotenv variables:
  CWS_CLIENT_SECRET
  CWS_REFRESH_TOKEN
  CWS_ITEM_URL

Optional:
  CWS_CLIENT_ID
  CWS_ACCESS_TOKEN
  CWS_PUBLISHER_ID
  CWS_EXTENSION_ID

Legacy data-deploy files are still supported as a fallback.
`);
}

async function resolveClientId(accessToken, sourceLabel) {
  if (!accessToken) {
    throw new Error(
      `Missing CWS_CLIENT_ID and CWS_ACCESS_TOKEN in ${sourceLabel ?? 'the environment'}. ` +
        'Set CWS_CLIENT_ID directly in your local dotenv file to avoid relying on token introspection.'
    );
  }

  const url = new URL('https://www.googleapis.com/oauth2/v1/tokeninfo');
  url.searchParams.set('access_token', accessToken);

  const response = await fetch(url);
  const payload = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(
      'Unable to recover the OAuth client ID from the current access token. ' +
        'Add CWS_CLIENT_ID to your local dotenv file and retry.\n' +
        JSON.stringify(payload, null, 2)
    );
  }

  const clientId = payload.issued_to;
  if (typeof clientId !== 'string' || clientId.length === 0) {
    throw new Error(
      'Google token info did not return "issued_to". Add CWS_CLIENT_ID to your local dotenv file and retry.'
    );
  }

  return clientId;
}
