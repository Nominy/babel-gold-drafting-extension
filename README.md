# Babel Gold Drafting Extension

## Install and build

Requires Node.js 22.14+ and the shared platform at `../../shared/babel-extension-platform`. Initialize the parent checkout with `git submodule update --init --recursive` first.

From this directory:

```sh
npm --prefix ../../shared/babel-extension-platform ci
npm ci
npm run build
```

Load `babel-gold-drafting-extension/` unpacked in `chrome://extensions`. Run `npm run build` whenever you want to rebuild that same folder, then reload the extension and refresh the Babel dashboard tab. Nothing watches or rebuilds automatically. Keep Babel Helper installed for native editor replacement and word navigation. Open extension Options and choose **Local**, **Cloud (Simple)**, or **Advanced** below.

`build` preserves the version. Use `npm run version:patch` explicitly when preparing a new release. The manifest includes the official public identity key so the unpacked extension keeps its normal broker identity.

## Local WebGPU mode

**Local** is the default for new installs. In Options, download the C-denoise v3 bundle and test the supplied public-domain sample or a speech recording of at most 15 seconds. A successful test verifies every required neural graph node on hardware WebGPU before enabling own-task inference. Local requires a hardware adapter with `shader-f16`, `timestamp-query`, and enough buffer capacity for the largest embedding. Local transcription needs no OpenRouter key or inference backend. With a configured key and **Don't run the LLM** unchecked, Gold drafting follows the local replacement using the saved backend/model and audio-input preference.

The bundle keeps the accepted GigaAM checkpoint and learned C-denoise weights. GigaAM runs transcription, timing, and acoustic-feature extraction; display rows follow the restored denoised audio activity and pause detector, rather than 32-word/12-second limits. Punctuation runs jointly over each complete audio segment that fits 256 tokens; longer segments use overlapping context windows while remaining intact in the editor. Each context pass is followed by four denoising steps. Bounded masks and indices use GPU-compatible types, and matrix/convolution reductions accumulate in FP32 with FP16 storage and output rounding. Every inference retains the SHA-bound node dispatch audit. Host audio preprocessing, dimension bookkeeping, and tensor views remain permitted; missing neural GPU dispatches reject the result.

Timing and punctuation labels remain together in the offscreen runtime for two tasks. Draft replacement and segment actions reuse those full-lane labels. Eviction, navigation, or offscreen restart can require recapturing the current task's audio. Other volunteers' timing alone cannot reconstruct the acoustic context; this development build disables shared coordinator participation.

The verified local bundle for this checkout is `D:/babel_experiment_runs/c-denoise-webgpu/gpu-bundle`; hardware and native parity evidence is in the adjacent `gpu-validation` directory. To serve it and build an own-task development extension, run from this extension directory:

```powershell
# Keep this static file supplier running in one terminal.
node scripts/serve-dev-cdenoise.mjs --bundle D:/babel_experiment_runs/c-denoise-webgpu/gpu-bundle --port 8808 --sample-dir D:/babel_experiment_runs/c-denoise-webgpu/public-sample

# In a second terminal:
$env:BABEL_DEV_C_DENOISE_MODEL_URL = 'http://127.0.0.1:8808/c-denoise'
npm run build
Remove-Item Env:\BABEL_DEV_C_DENOISE_MODEL_URL
```

Reload the unpacked extension, open Options, select Local, and download/test this bundle. Existing pre-lowering caches must be updated. An ordinary build without that environment variable uses the hosted supplier; it does not publish these model files. The local graph bundle and matching runtime must ship together.

Gold 0.2.41 uses the immutable release contract in `model-release.json`. The [release cutover guide](deploy/inference-release/README.md) covers staging the model, the shared WebGPU backend, protocol 3 workers, and mandatory client upgrades. `npm run prepare:inference-release -- --bundle DIR --sample-dir DIR --out EMPTY_DIR --evidence BACKEND_SMOKE_JSON --coordinator-evidence COORDINATOR_SMOKE_JSON` verifies the tested model and packages the production store ZIP, model files, backend dependency locks, deployment examples, and SHA-256 inventory. It requires successful real GPU/restart reports and keeps private corpus token sequences out of public metadata.

## Cloud (Simple) and Advanced modes

**Cloud (Simple)** uses explicit paid cloud transcription. Enter an OpenRouter API key, save, and reload the Babel task. No backend address, local model download, or GPU setup is needed. Use the transcription wand or a Helper transcription action to generate a draft.

- Transcription uses `microsoft/mai-transcribe-2` through OpenRouter's dedicated speech-to-text endpoint: Russian language hint, verbatim style, verbose JSON, and word timestamps.
- The two captured speaker lanes are transcribed independently; their existing identities are authoritative. Diarization is disabled for isolated lanes. Native words, punctuation, capitalization, and repetitions are retained rather than passed through L2/BERT or a Gold rewrite.
- Word timing and native draft text are retained together in extension-origin IndexedDB. Reloads, repeated actions, and Helper segment transcription reuse completed results. Only transcript/timing results are cached there, not the API key or audio bytes.
- Opening a task performs only a free cache lookup. Audio is uploaded and charged only after an explicit transcription or retry action. The key is read by the extension background and sent directly to OpenRouter; Simple requests do not send it to the page or the drafting backend.
- OpenRouter/provider audio processing is cloud-based. At the current approximate $0.10 per audio hour, two one-hour speaker lanes cost about $0.20; provider pricing may change. Long PCM16 WAVs are split into bounded, non-overlapping audio chunks, with offsets retained and a declared text separator between chunk outputs.
- The same native editor, replacement bridge, playback, word navigation, and preview/apply controls remain in use. Babel Helper must be available before Simple starts a paid transcription. As with existing L0 replacement, generating the draft replaces native rows; inspect the result before saving/submitting.
- Helper text redistribution is a separate, explicitly requested OpenRouter text-model action using `google/gemini-3.8-flash` and the same key. It reviews whole-sentence allocation moves; MAI is not used as a chat model. No extra text-model pass runs after transcription.
- MAI does not guarantee Babel-specific audible-event tags or Gold interruption notation. Review verbatim details, punctuation, and timings. Backwards or unusably long word timing metadata is bounded without reordering recognized text; missing word timestamps are reported without inventing timing or silently switching models.

**Advanced** exposes the existing backend, model, reasoning/service-tier, Helper provider, hosted/self-hosted L0, downloaded browser-model, and volunteering controls. Existing nonempty settings without a mode migrate to Advanced, so an installed user's workflow is not silently switched to paid cloud transcription. Switching modes preserves Advanced preferences; Simple ignores their local-model and volunteer flags. For a self-hosted GPU engine, install [L0 Draft Engine](../l0-draft-engine/README.md).

Provider failures are visible and never cause an automatic paid retry or hidden fallback in Simple. An interrupted request whose charge/result is uncertain first shows a warning; another explicit retry may repeat that unfinished chunk, while completed chunks/lanes stay reusable.

## Local models without swarm participation

Local mode uses downloaded models for your own tasks. Advanced also exposes **downloaded local browser models** and the explicit **Volunteer to process other users' L0 audio** preference. The development supplier build keeps shared jobs disabled regardless of a retained volunteer preference.

Local timing captures the complete speaker lanes and retains their acoustic punctuation labels in the offscreen runtime, publishing only lane/word tokens to Babel Helper. Advanced hosted timing retains its task capability and lookup/upload lifecycle. Navigation cancels pending timing waits and releases drafting controls, including while the next route has no transcript.

Replacement verification uses task-scoped recording IDs published by the page bridge, including when an initially empty transcript receives its first rows. Display labels alone are not treated as recording IDs; stale task metadata and ambiguous duplicate labels are rejected rather than guessing a lane.

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

Push to `main` creates a GitHub prerelease `v<version>`. To publish, manually run `.github/workflows/deploy-gold-drafting-extension.yml` on that commit with `version=<version>` and `confirm=PUBLISH <version>`. The tag must still be a prerelease pointing at the selected commit. `publish_type` defaults to `DEFAULT_PUBLISH`, so the Chrome Web Store publishes automatically after approval; select `STAGED_PUBLISH` only to publish manually after review. `replace_pending_submission` cancels a pending review. A successful submission promotes the GitHub prerelease; this does not mean the store version is already live.

Required Actions secrets: `CWS_CLIENT_ID`, `CWS_CLIENT_SECRET`, `CWS_REFRESH_TOKEN`, `CWS_PUBLISHER_ID`, `CWS_EXTENSION_ID`. Optional fallback: `CWS_ACCESS_TOKEN`.

For local publishing, copy `.env.cws.example` to ignored `.env.cws.local` and fill the credentials. Seed repository secrets with `node scripts/setup-github-secrets.mjs OWNER/REPO`; publish locally with `npm run publish:cws`.
