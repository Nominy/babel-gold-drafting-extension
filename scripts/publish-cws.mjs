#!/usr/bin/env node

import { resolve } from 'node:path';
import {
  loadCwsEnvironment,
  parseItemUrl,
  runPublishCws
} from '@nominy/babel-extension-build';

const rootDir = resolve(import.meta.dirname, '..');
const argv = process.argv.slice(2);

function readOption(...names) {
  for (let index = 0; index < argv.length; index += 1) {
    for (const name of names) {
      const option = `--${name}`;
      if (argv[index] === option) {
        return argv[index + 1]?.startsWith('--') ? undefined : argv[index + 1];
      }
      if (argv[index].startsWith(`${option}=`)) {
        return argv[index].slice(option.length + 1);
      }
    }
  }
  return undefined;
}

function resolveItemTarget() {
  const itemUrl = process.env.CWS_ITEM_URL?.trim();
  if (itemUrl) {
    return parseItemUrl(itemUrl);
  }

  const publisherId = process.env.CWS_PUBLISHER_ID?.trim();
  const extensionId = process.env.CWS_EXTENSION_ID?.trim();
  if (!publisherId || !extensionId) {
    throw new Error('Missing Chrome Web Store item target. Set CWS_ITEM_URL or both CWS_PUBLISHER_ID and CWS_EXTENSION_ID.');
  }
  return { publisherId, extensionId };
}

async function requestAccessToken() {
  const clientId = process.env.CWS_CLIENT_ID?.trim();
  const clientSecret = process.env.CWS_CLIENT_SECRET?.trim();
  const refreshToken = process.env.CWS_REFRESH_TOKEN?.trim();
  if (clientId && clientSecret && refreshToken) {
    const response = await fetch('https://oauth2.googleapis.com/token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        client_id: clientId,
        client_secret: clientSecret,
        refresh_token: refreshToken,
        grant_type: 'refresh_token'
      })
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || typeof payload.access_token !== 'string' || !payload.access_token) {
      throw new Error(`Could not obtain a Chrome Web Store access token (${response.status} ${response.statusText}).`);
    }
    return payload.access_token;
  }

  const accessToken = process.env.CWS_ACCESS_TOKEN?.trim();
  if (!accessToken) {
    throw new Error('Missing Chrome Web Store authentication.');
  }
  return accessToken;
}

async function cancelPendingSubmission() {
  const { publisherId, extensionId } = resolveItemTarget();
  const accessToken = await requestAccessToken();
  const response = await fetch(
    `https://chromewebstore.googleapis.com/v2/publishers/${encodeURIComponent(publisherId)}/items/${encodeURIComponent(extensionId)}:cancelSubmission`,
    {
      method: 'POST',
      headers: { Authorization: `Bearer ${accessToken}` }
    }
  );
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const detail = typeof payload?.error?.message === 'string' ? `: ${payload.error.message}` : '';
    throw new Error(`Chrome Web Store submission cancellation failed (${response.status} ${response.statusText})${detail}`);
  }
  console.log(`Cancelled pending Chrome Web Store submission for ${extensionId}.`);
}

await loadCwsEnvironment(rootDir, readOption('env-file', 'file'));
if (process.env.CWS_CANCEL_PENDING?.trim().toLowerCase() === 'true') {
  await cancelPendingSubmission();
}

await runPublishCws({
  rootDir,
  defaultZipPath(version) {
    return resolve(rootDir, '.artifacts', `babel-gold-drafting-extension-${version}.zip`);
  },
  usageZipLine: '.artifacts/babel-gold-drafting-extension-<version>.zip'
});
