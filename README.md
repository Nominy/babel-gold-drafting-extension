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
- create a Chrome Web Store ZIP one directory above the extension root

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
