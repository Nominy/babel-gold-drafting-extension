# Babel Gold Drafting Extension

Standalone MV3 extension for Silver-to-Gold transcript drafting on the Babel transcription page.

## Build

1. `npm install`
2. `npm run build`
3. Load unpacked from `gold-drafting-extension/` in `chrome://extensions`

`npm run build` automatically advances the patch version in:
- [`package.json`](/C:/Users/User/Desktop/dev/babel/drafting/gold-drafting-extension/package.json)
- [`manifest.json`](/C:/Users/User/Desktop/dev/babel/drafting/gold-drafting-extension/manifest.json)
- [`package-lock.json`](/C:/Users/User/Desktop/dev/babel/drafting/gold-drafting-extension/package-lock.json)

Use `npm run build:core` when you need to rebuild bundles without changing version files.

## Store Package

Run:

```bash
npm run build:zip
```

This will:
- advance the patch version through `npm run build`
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
- Sends the snapshot to the dedicated drafting backend with the user's OpenRouter API key
- Captures the two Babel audio lanes in the background and sends them to the configured L0 `/v1/transcribe` endpoint for word timing metadata; failures are silent and do not block editing
- Audio-enhanced drafting is enabled by default and can be disabled in extension settings. When enabled, drafting requests may send audio tracks to the LLM backend for audible-event and vocal-style tags
- Requires BYOK for drafting; there is no shared backend key fallback for regular generation
- Lets the user choose a model, OpenRouter service tier, and reasoning effort; `google/gemini-3-flash-preview` with low reasoning is the default
- Shows rewrite summary and row-level diff preview
- Applies the generated draft back into existing Babel textareas only
- Restores the captured original snapshot on demand

## Validation

- `npm run typecheck`
- `npm test`
- `npm run build`

## Release

- GitHub Releases are the canonical home for packaged ZIPs.
- The version committed in `manifest.json`, `package.json` and `package-lock.json` is the release version. Bump it in the PR (`npm run version:patch`, or implicitly through `npm run build`); CI never bumps or commits versions. It must be greater than what the Chrome Web Store currently holds; the publish script checks the store's published and submitted versions and aborts otherwise.
- Push to `main` (`.github/workflows/deploy-gold-drafting-extension.yml`, job `prerelease`) validates, builds the ZIP with `build:core` (no bump), uploads it as a workflow artifact, and creates a GitHub *pre-release* tagged `v<version>` at that commit. It never publishes to the Chrome Web Store. If `v<version>` is already a full release the job fails until the version is bumped.
- Publishing to the Chrome Web Store is manual only: run the same workflow via `workflow_dispatch` (job `publish`) with `version` (must equal `manifest.json` on the selected ref, whose tag `v<version>` must point at that commit and still be a pre-release) and `confirm` set to `PUBLISH <version>`. `publish_type` defaults to `STAGED_PUBLISH`; `replace_pending_submission` cancels a pending review first. The job re-validates, rebuilds, uploads and publishes, then promotes the pre-release to a full release.
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
