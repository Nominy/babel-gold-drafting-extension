# Babel Gold Drafting Extension

## Install and build

Requires Node.js 22.14+ and the shared platform at `../../shared/babel-extension-platform`. Initialize the parent checkout with `git submodule update --init --recursive` first.

From this directory:

```sh
npm --prefix ../../shared/babel-extension-platform ci
npm ci
npm run build
```

Load this directory unpacked in `chrome://extensions`. Bundles and model runtime assets go to `dist/`; the build also writes `offscreen.html`. Reload the extension and refresh the Babel dashboard after rebuilding. Set the backend address and your OpenRouter key in extension settings. For local GPU transcription, install [L0 Draft Engine](../l0-draft-engine/README.md).

`build` bumps the patch version in `package.json`, `manifest.json`, and `package-lock.json`. Use `npm run build:core` for a no-bump rebuild.

## Checks

```sh
npm run typecheck
npm test
```

Browser checks use the [shared browser setup](../../shared/babel-extension-platform/README.md#browser-checks).

## Package and publish

```sh
npm run build:zip                     # rebuilds and bumps the version
npm run build:zip -- --no-build       # packages existing bundles
```

The store ZIP goes to `.artifacts/` and strips local development host permissions.

Commit the release version before merging; `npm run version:patch` bumps it without building. CI does not bump versions. The version must exceed the store's published and submitted versions.

Push to `main` creates a GitHub prerelease `v<version>`. To publish, manually run `.github/workflows/deploy-gold-drafting-extension.yml` on that commit with `version=<version>` and `confirm=PUBLISH <version>`. The tag must still be a prerelease pointing at the selected commit. `publish_type` defaults to `STAGED_PUBLISH`; `replace_pending_submission` cancels a pending review. A successful publish promotes the prerelease.

Required Actions secrets: `CWS_CLIENT_ID`, `CWS_CLIENT_SECRET`, `CWS_REFRESH_TOKEN`, `CWS_PUBLISHER_ID`, `CWS_EXTENSION_ID`. Optional fallback: `CWS_ACCESS_TOKEN`.

For local publishing, copy `.env.cws.example` to ignored `.env.cws.local` and fill the credentials. Seed repository secrets with `node scripts/setup-github-secrets.mjs OWNER/REPO`; publish locally with `npm run publish:cws`.
