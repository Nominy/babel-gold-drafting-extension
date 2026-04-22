#!/usr/bin/env node

import { access, readFile } from 'node:fs/promises';
import { constants } from 'node:fs';
import { resolve } from 'node:path';

const DEFAULT_ENV_FILES = ['.env.cws.local', '.env.local', '.env', 'data-deploy'];

export async function loadCwsEnvironment(rootDir, explicitFilePath) {
  const filePath = explicitFilePath
    ? resolve(rootDir, explicitFilePath)
    : await findFirstReadableFile(rootDir, DEFAULT_ENV_FILES);

  if (!filePath) {
    return {
      filePath: null,
      format: null,
      values: { ...process.env }
    };
  }

  const source = await readFile(filePath, 'utf8');
  const format = looksLikeLegacyDeployData(source) ? 'legacy-deploy-data' : 'dotenv';
  const parsedValues = format === 'dotenv' ? parseDotenv(source) : parseLegacyDeployData(source);
  const values = { ...process.env, ...parsedValues };

  for (const [key, value] of Object.entries(parsedValues)) {
    if (!(key in process.env)) {
      process.env[key] = value;
    }
  }

  return { filePath, format, values };
}

export function parseItemUrl(itemUrl, label = 'CWS item URL') {
  let url;
  try {
    url = new URL(itemUrl);
  } catch (error) {
    throw new Error(`Invalid ${label}: ${error instanceof Error ? error.message : String(error)}`);
  }

  const match = url.pathname.match(/\/v2\/publishers\/([^/]+)\/items\/([^/]+)/);
  if (!match) {
    throw new Error(`Unexpected Chrome Web Store item URL in ${label}: ${itemUrl}`);
  }

  return {
    publisherId: decodeURIComponent(match[1]),
    extensionId: decodeURIComponent(match[2])
  };
}

export function getDefaultEnvFiles() {
  return [...DEFAULT_ENV_FILES];
}

async function findFirstReadableFile(rootDir, candidates) {
  for (const candidate of candidates) {
    const filePath = resolve(rootDir, candidate);
    if (await isReadableFile(filePath)) {
      return filePath;
    }
  }

  return null;
}

async function isReadableFile(filePath) {
  try {
    await access(filePath, constants.R_OK);
    return true;
  } catch {
    return false;
  }
}

function looksLikeLegacyDeployData(source) {
  return /(^|\r?\n)\s*(secret|refresh-token|access-token|the extension|client-id)\s*:\s*($|\r?\n)/i.test(
    source
  );
}

function parseDotenv(source) {
  const result = {};

  for (const rawLine of source.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) {
      continue;
    }

    const normalized = line.startsWith('export ') ? line.slice(7).trim() : line;
    const equalsIndex = normalized.indexOf('=');
    if (equalsIndex <= 0) {
      continue;
    }

    const key = normalized.slice(0, equalsIndex).trim();
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key)) {
      continue;
    }

    let value = normalized.slice(equalsIndex + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    } else {
      const commentIndex = value.indexOf(' #');
      if (commentIndex >= 0) {
        value = value.slice(0, commentIndex).trimEnd();
      }
    }

    result[key] = value;
  }

  return result;
}

function parseLegacyDeployData(source) {
  const legacy = {};
  let currentKey = null;

  for (const rawLine of source.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) {
      continue;
    }

    if (line.endsWith(':')) {
      currentKey = line.slice(0, -1).trim().toLowerCase();
      continue;
    }

    if (!currentKey) {
      continue;
    }

    legacy[currentKey] = line;
    currentKey = null;
  }

  const result = {};

  if (legacy.secret) {
    result.CWS_CLIENT_SECRET = legacy.secret;
  }
  if (legacy['refresh-token']) {
    result.CWS_REFRESH_TOKEN = legacy['refresh-token'];
  }
  if (legacy['access-token']) {
    result.CWS_ACCESS_TOKEN = legacy['access-token'];
  }
  if (legacy['client-id']) {
    result.CWS_CLIENT_ID = legacy['client-id'];
  }
  if (legacy['the extension']) {
    result.CWS_ITEM_URL = legacy['the extension'];
  }

  return result;
}
