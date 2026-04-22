# Babel Gold Drafting Extension

Standalone MV3 extension for Silver-to-Gold transcript drafting on the Babel transcription page.

## Build

1. `npm install`
2. `npm run build`
3. Load unpacked from `gold-drafting-extension/` in `chrome://extensions`

Each `npm run build` mirrors the helper extension flow and bumps the patch version in:
- [`package.json`](/C:/Users/User/Desktop/dev/babel/drafting/gold-drafting-extension/package.json)
- [`manifest.json`](/C:/Users/User/Desktop/dev/babel/drafting/gold-drafting-extension/manifest.json)
- [`package-lock.json`](/C:/Users/User/Desktop/dev/babel/drafting/gold-drafting-extension/package-lock.json)

## Store Package

Run:

```bash
npm run build:zip
```

This will:
- bump the patch version
- build the bundled scripts
- create a Chrome Web Store ZIP in `.artifacts/`

The ZIP includes only:
- `manifest.json`
- `options.html`
- `icons/*`
- `dist/*.js`

The packaged manifest strips local development host permissions and keeps only:
- `https://dashboard.babel.audio/*`
- `https://reviewgen.ovh/*`

## Behavior

- Captures the current Babel transcription rows as a locked job snapshot
- Sends the snapshot to the dedicated drafting backend
- Shows rewrite summary and row-level diff preview
- Applies the generated draft back into existing Babel textareas only
- Restores the captured original snapshot on demand

## Validation

- `npm run typecheck`
- `npm test`
- `npm run build`

## Release

- GitHub Releases are the canonical home for packaged ZIPs.
- `.github/workflows/deploy-gold-drafting-extension.yml` is the manual Chrome Web Store deployment workflow. It validates the extension, builds the release ZIP, publishes it to the Chrome Web Store, commits the version bump, and then publishes the GitHub Release asset.
- Required GitHub Actions secrets:
  - `CWS_CLIENT_ID`
  - `CWS_CLIENT_SECRET`
  - `CWS_REFRESH_TOKEN`
  - `CWS_PUBLISHER_ID`
  - `CWS_EXTENSION_ID`
- Optional GitHub Actions secret:
  - `CWS_ACCESS_TOKEN`
- For local publishing helpers, keep Chrome Web Store credentials in `.env.cws.local` and start from `.env.cws.example`.
- To seed the GitHub Actions secrets from the local dotenv file, run `node scripts/setup-github-secrets.mjs OWNER/REPO`.
