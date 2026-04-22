#!/usr/bin/env node

import { readFile, access } from 'node:fs/promises';
import { constants } from 'node:fs';
import { join, resolve } from 'node:path';
import { getDefaultEnvFiles, loadCwsEnvironment, parseItemUrl } from './cws-env.mjs';

const rootDir = resolve(import.meta.dirname, '..');
const args = parseArgs(process.argv.slice(2));

if (args.flags.has('help')) {
  printHelp();
  process.exit(0);
}

const envFile = args.values.get('env-file') ?? args.values.get('file');
const loaded = await loadCwsEnvironment(rootDir, envFile);

const manifest = await readManifest();
const zipPath = resolve(
  rootDir,
  args.values.get('zip') ??
    process.env.CWS_ZIP_PATH ??
    resolve(rootDir, '.artifacts', `babel-gold-drafting-extension-${manifest.version}.zip`)
);
const publishType = normalizePublishType(
  args.values.get('publish-type') ?? process.env.CWS_PUBLISH_TYPE ?? 'DEFAULT_PUBLISH'
);
const skipReview = parseBoolean(
  args.values.get('skip-review') ?? process.env.CWS_SKIP_REVIEW ?? 'false'
);
const pollIntervalMs = parsePositiveInteger(
  args.values.get('poll-interval-ms') ?? process.env.CWS_POLL_INTERVAL_MS ?? '5000',
  'poll interval'
);
const pollTimeoutMs = parsePositiveInteger(
  args.values.get('poll-timeout-ms') ?? process.env.CWS_POLL_TIMEOUT_MS ?? '120000',
  'poll timeout'
);
const { publisherId, extensionId } = resolveItemTarget();

await ensureReadableFile(zipPath);

console.log(`Preparing Chrome Web Store upload for ${extensionId} (${manifest.version})`);
console.log(`ZIP: ${zipPath}`);

const accessToken = await getAccessToken();
const uploadUrl = createUploadUrl(publisherId, extensionId);
const publishUrl = createPublishUrl(publisherId, extensionId);
const statusUrl = createStatusUrl(publisherId, extensionId);
const zipBuffer = await readFile(zipPath);

const uploadResult = await requestJson(uploadUrl, {
  method: 'POST',
  headers: {
    Authorization: `Bearer ${accessToken}`,
    'Content-Type': 'application/zip'
  },
  body: zipBuffer
});
ensureApiSuccess(uploadResult, 'upload');

const finalUploadResult = await waitForUploadIfNeeded(
  uploadResult,
  accessToken,
  statusUrl,
  pollIntervalMs,
  pollTimeoutMs
);
ensureApiSuccess(finalUploadResult, 'upload status');

console.log(`Upload response:\n${formatJson(finalUploadResult)}`);

const publishBody = {};
if (publishType !== 'DEFAULT_PUBLISH') {
  publishBody.publishType = publishType;
}
if (skipReview) {
  publishBody.skipReview = true;
}

const publishHeaders = {
  Authorization: `Bearer ${accessToken}`
};
const publishOptions = {
  method: 'POST',
  headers: publishHeaders
};

if (Object.keys(publishBody).length > 0) {
  publishHeaders['Content-Type'] = 'application/json';
  publishOptions.body = JSON.stringify(publishBody);
}

const publishResult = await requestJson(publishUrl, publishOptions);
ensureApiSuccess(publishResult, 'publish');

console.log(`Publish response:\n${formatJson(publishResult)}`);
console.log('Chrome Web Store publish request completed successfully.');

function parseArgs(argv) {
  const values = new Map();
  const flags = new Set();

  for (let index = 0; index < argv.length; index += 1) {
    const entry = argv[index];
    if (!entry.startsWith('--')) {
      throw new Error(`Unexpected argument: ${entry}`);
    }

    const trimmed = entry.slice(2);
    const equalsIndex = trimmed.indexOf('=');
    if (equalsIndex >= 0) {
      const key = trimmed.slice(0, equalsIndex);
      const value = trimmed.slice(equalsIndex + 1);
      values.set(key, value);
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

  return { values, flags };
}

function printHelp() {
  console.log(`Usage: node scripts/publish-cws.mjs [options]

Options:
  --zip PATH               ZIP to upload. Defaults to .artifacts/babel-gold-drafting-extension-<version>.zip
  --env-file PATH          Local dotenv file. Defaults to first readable file in:
                           ${getDefaultEnvFiles().join(', ')}
  --file PATH              Alias for --env-file
  --publish-type TYPE      DEFAULT_PUBLISH or STAGED_PUBLISH
  --skip-review            Request skipReview=true
  --poll-interval-ms N     Upload-status polling interval in milliseconds
  --poll-timeout-ms N      Upload-status polling timeout in milliseconds
  --help                   Show this help

Environment:
  CWS_CLIENT_ID
  CWS_CLIENT_SECRET
  CWS_REFRESH_TOKEN
  CWS_ACCESS_TOKEN         Optional short-lived fallback when refresh credentials are unavailable
  CWS_PUBLISHER_ID
  CWS_EXTENSION_ID
  CWS_ITEM_URL             Alternative to CWS_PUBLISHER_ID + CWS_EXTENSION_ID
  CWS_ZIP_PATH
  CWS_PUBLISH_TYPE
  CWS_SKIP_REVIEW
`);
}

async function readManifest() {
  const raw = await readFile(join(rootDir, 'manifest.json'), 'utf8');
  return JSON.parse(raw.replace(/^\uFEFF/, ''));
}

function resolveItemTarget() {
  const itemUrl = process.env.CWS_ITEM_URL?.trim();
  if (itemUrl) {
    return parseItemUrl(itemUrl, loaded.filePath ?? 'CWS_ITEM_URL');
  }

  const publisherId = process.env.CWS_PUBLISHER_ID?.trim();
  const extensionId = process.env.CWS_EXTENSION_ID?.trim();

  if (!publisherId || !extensionId) {
    throw new Error(
      'Missing Chrome Web Store item target. Set CWS_ITEM_URL or both CWS_PUBLISHER_ID and CWS_EXTENSION_ID.'
    );
  }

  return { publisherId, extensionId };
}

async function ensureReadableFile(filePath) {
  try {
    await access(filePath, constants.R_OK);
  } catch (error) {
    throw new Error(
      `Cannot read ZIP file at ${filePath}. Run "npm run build:zip" first or pass --zip PATH.\n` +
        `${error instanceof Error ? error.message : String(error)}`
    );
  }
}

async function getAccessToken() {
  const clientId = process.env.CWS_CLIENT_ID?.trim();
  const clientSecret = process.env.CWS_CLIENT_SECRET?.trim();
  const refreshToken = process.env.CWS_REFRESH_TOKEN?.trim();

  if (clientId && clientSecret && refreshToken) {
    const body = new URLSearchParams({
      client_id: clientId,
      client_secret: clientSecret,
      refresh_token: refreshToken,
      grant_type: 'refresh_token'
    });

    const tokenResult = await requestJson('https://oauth2.googleapis.com/token', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded'
      },
      body
    });

    const accessToken = tokenResult?.access_token;
    if (typeof accessToken !== 'string' || accessToken.length === 0) {
      throw new Error(`Google token endpoint did not return an access token.\n${formatJson(tokenResult)}`);
    }

    return accessToken;
  }

  const accessToken = process.env.CWS_ACCESS_TOKEN?.trim();
  if (accessToken) {
    console.warn(
      'Using CWS_ACCESS_TOKEN directly. This is short-lived; prefer CWS_CLIENT_ID + CWS_CLIENT_SECRET + CWS_REFRESH_TOKEN for GitHub Actions.'
    );
    return accessToken;
  }

  throw new Error(
    'Missing Chrome Web Store authentication. Set CWS_CLIENT_ID, CWS_CLIENT_SECRET, and CWS_REFRESH_TOKEN, or provide CWS_ACCESS_TOKEN as a short-lived fallback.'
  );
}

async function requestJson(url, options) {
  const response = await fetch(url, options);
  const text = await response.text();
  const payload = parseJson(text);

  if (!response.ok) {
    throw new Error(
      `${options.method ?? 'GET'} ${url} failed with ${response.status} ${response.statusText}.\n${formatJson(
        payload ?? text
      )}`
    );
  }

  return payload ?? {};
}

async function waitForUploadIfNeeded(initialResult, accessToken, statusUrl, intervalMs, timeoutMs) {
  const initialState = findDeepProperty(initialResult, 'uploadState');
  if (initialState !== 'UPLOAD_IN_PROGRESS') {
    return initialResult;
  }

  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    await sleep(intervalMs);
    const statusResult = await requestJson(statusUrl, {
      method: 'GET',
      headers: {
        Authorization: `Bearer ${accessToken}`
      }
    });
    ensureApiSuccess(statusResult, 'fetch status');

    const state = findDeepProperty(statusResult, 'uploadState');
    if (state !== 'UPLOAD_IN_PROGRESS') {
      return statusResult;
    }
  }

  throw new Error(`Upload stayed in UPLOAD_IN_PROGRESS for longer than ${timeoutMs}ms.`);
}

function ensureApiSuccess(payload, stage) {
  const apiError = findDeepProperty(payload, 'error');
  if (apiError) {
    throw new Error(`Chrome Web Store ${stage} returned an API error.\n${formatJson(apiError)}`);
  }

  const itemError = findDeepProperty(payload, 'itemError');
  if (itemError) {
    throw new Error(`Chrome Web Store ${stage} returned an item error.\n${formatJson(itemError)}`);
  }

  const uploadState = findDeepProperty(payload, 'uploadState');
  if (typeof uploadState === 'string' && /(FAIL|ERROR|INVALID|REJECT)/i.test(uploadState)) {
    throw new Error(`Chrome Web Store ${stage} failed with uploadState=${uploadState}.\n${formatJson(payload)}`);
  }
}

function createUploadUrl(publisherId, extensionId) {
  return `https://chromewebstore.googleapis.com/upload/v2/publishers/${encodeURIComponent(
    publisherId
  )}/items/${encodeURIComponent(extensionId)}:upload`;
}

function createPublishUrl(publisherId, extensionId) {
  return `https://chromewebstore.googleapis.com/v2/publishers/${encodeURIComponent(
    publisherId
  )}/items/${encodeURIComponent(extensionId)}:publish`;
}

function createStatusUrl(publisherId, extensionId) {
  return `https://chromewebstore.googleapis.com/v2/publishers/${encodeURIComponent(
    publisherId
  )}/items/${encodeURIComponent(extensionId)}:fetchStatus`;
}

function parseJson(text) {
  if (!text) {
    return null;
  }

  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function findDeepProperty(value, key) {
  if (!value || typeof value !== 'object') {
    return undefined;
  }

  if (Object.prototype.hasOwnProperty.call(value, key)) {
    return value[key];
  }

  for (const child of Object.values(value)) {
    const found = findDeepProperty(child, key);
    if (found !== undefined) {
      return found;
    }
  }

  return undefined;
}

function formatJson(value) {
  if (typeof value === 'string') {
    return value;
  }
  return JSON.stringify(value, null, 2);
}

function normalizePublishType(value) {
  const normalized = String(value).trim().toUpperCase();
  if (!normalized || normalized === 'DEFAULT' || normalized === 'DEFAULT_PUBLISH') {
    return 'DEFAULT_PUBLISH';
  }
  if (normalized === 'STAGED' || normalized === 'STAGED_PUBLISH') {
    return 'STAGED_PUBLISH';
  }
  throw new Error(`Unsupported publish type: ${value}`);
}

function parseBoolean(value) {
  if (typeof value === 'boolean') {
    return value;
  }

  const normalized = String(value).trim().toLowerCase();
  if (normalized === '' || normalized === '0' || normalized === 'false' || normalized === 'no') {
    return false;
  }
  if (normalized === '1' || normalized === 'true' || normalized === 'yes') {
    return true;
  }
  throw new Error(`Unsupported boolean value: ${value}`);
}

function parsePositiveInteger(value, label) {
  const number = Number.parseInt(String(value), 10);
  if (!Number.isFinite(number) || number <= 0) {
    throw new Error(`Invalid ${label}: ${value}`);
  }
  return number;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
