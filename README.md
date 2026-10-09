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

## Packaged ZipEnhancer runtime

With matching Babel Helper and **Review Grader 0.1.1 or newer** installed and enabled, default-enabled **ZipEnhancer audio** automatically enhances both loaded speaker lanes. Without the pinned Review Grader extension, enhancement UI/settings, capture, cache/model initialization and swarm work stay inactive; other Gold/Helper features remain available. A usable hardware WebGPU adapter runs the packaged model locally; otherwise a cache miss uploads the Original recordings to a compatible remote swarm GPU. Assets are `models/zipenhancer-webgpu.plan.json` and `zipenhancer-webgpu.weights.bin` (4.21 MB of weights). There is no CPU inference engine. The enhancement model is independent of the downloadable C-denoise/ASR bundle.

Gold verifies packaged asset hashes and actual local neural GPU dispatches without changing ASR's ORT environment. GPU admission requires a confirmed hardware adapter, FP16/timestamp support, and sufficient kernel limits. Only unavailable capabilities select the swarm: GPU model/allocation/inference errors remain failures rather than triggering an off-device retry. Both routes use the same model identity, full-context windows, source clocks, overlap and per-lane RMS restoration.

Access uses an authenticated Chrome external port to Review Grader (`geagfgdjmeojbkbdjmbchkhjjfpaffbe`), not page markers or page messages. Disabling/removing Grader aborts active enhancement, removes its controls/settings and restores Original audio. Re-enabling reconnects without refreshing the editor. Older content-only Grader builds do not grant access; update the addon, including its background bundle.

Request-scoped progress identifies **WebGPU (local)** or **Swarm GPU (remote)**, with explicit upload, queue, inference and transfer phases. Completed-window counts originate from the executing worker after neural inference and audio reconstruction. Sender/task/track/request ownership is checked throughout; cancellation never installs partial or stale audio.

Audio DSP runs in `dist/workers/audio-enhancement.js`, reusing overlapping STFT frames and reconstructing the previous window while the next GPU inference runs. Swarm workers invoke the same GPU-only runtime without reading or writing their persistent pair cache. Requesters retain the existing single-pair cache; a validated hit performs no upload or inference.

Helper owns the native strength slider and Original/Enhanced mixing. Drafting, segment transcription, and L0 capture consume its committed WAVs, including intermediate blends; enhancement inference always receives retained Originals. Every nonzero strength has a stable model/source/mix identity, while strength0 preserves the Original canonical identity. L0 waits for desired strength to commit and refuses unavailable or changed native buffers rather than substituting network Originals. Slider movement while held changes neither capture identity nor inference.

`audio-enhancement-cache.ts` retains one enhanced pair in extension-origin IndexedDB (`babel-gold-drafting-audio-enhancement`, store `current`, key `pair`). Identity binds the plan, weights, runtime/kernel/DSP implementation and sorted recording-ID/source-SHA pairs, independent of review action or transcript edits. Selection atomically replaces an old pair with a byte-free reservation; both verified WAVs commit only while its nonce remains current. Late results cannot repopulate an evicted pair. Hits verify clocks/digests, restore current request order/labels, and skip decoded-PCM allocation, GPU setup, and inference. Browser eviction is a cache miss; database/quota errors do not turn valid enhancement into failure, and model errors are never replaced by cache success.

Model provenance/license remain in `models/zipenhancer.NOTICE` and `.LICENSE`. The original ONNX and `audio-enhancement-model.json` remain source/export references, not shipped inference assets. `scripts/export-zipenhancer-webgpu.py` derives either a full-FP32 control or the selected mixed-FP16 plan; `scripts/zip-webgpu-artifact.mjs` binds its assets and implementation into `audio-enhancement-webgpu-model.json` at build time.

The fixed `[1, 201, 641]` engine reuses pinned ORT WGSL operators without a WASM session. It adds 24 tiled relative-attention consumers, 40 exact source-DAG Swoosh fusions, shape/uniform specialization, and lifetime-checked activation-buffer reuse. It preserves every key, learned parameter, window, and overlap; no low-rank compression is selected. FP16 storage uses FP32 matrix accumulation and protected nonlinear/statistics islands. Fusion provenance accounts for all 2,115 original GPU nodes; every window must observe all selected compute groups, not fabricated per-source dispatch events.

### Browser port measurements

Chrome 153 / NVIDIA Blackwell, both complete 63.63-second lanes, 44 neural windows, audio DSP included, no output-cache hit. Editor capture/transport is excluded:

| Runtime | Cold processing | Warm processing |
|---|---:|---:|
| Previous FP32 ONNX worker runtime | 20.15 s | 16.98–18.51 s |
| Direct mixed-FP16 WebGPU + tiled attention + static uniforms | 25.54 s | 10.21–10.33 s |
| Selected WebGPU engine + pointwise fusion | 12.20 s | 7.11–7.12 s |

The selected engine's six output WAVs were identical across its cold/two warm runs. Relative waveform RMSE versus accepted browser FP32 was 0.0842% / 0.0879%; energy ratios were 1.00014 / 0.99995. The FP32 fusion control stayed within one PCM16 quantization step. These checks establish numerical fidelity, not listening or ASR-quality certification. Peak tracked GPU-buffer allocation was 249,391,616 bytes; static planned storage was 220,865,696 bytes.

Optional `--capture` also passed distinct-input replay and full-wave checks, but measured only 7.03–7.07 s warm. It remains off in production: that small sample does not justify the extra replay state. **Native CUDA's 1.971 s is not the browser result.** The tested browser exposes FP16/subgroups/timestamps but no matrix/tensor-core API; the portable WGSL attention implementation does not execute CUDA WMMA.

```powershell
python scripts/export-zipenhancer-webgpu.py --precision mixed-float16 --out-dir models
node scripts/benchmark-zipenhancer-webgpu.mjs --plan models/zipenhancer-webgpu.plan.json --out PRIVATE_NEW_DIR --runs 3 --executable PATH_TO_CHROME
```

Use the existing Python environment with ONNX/ONNX Runtime for export. The benchmark serves only loopback assets, runs real hardware WebGPU, records full-window dispatch proofs and both WAVs, and uploads artifacts in verified 4 MiB chunks. Keep recording-derived artifacts private. Local evidence: `D:/babel_experiment_runs/zipenhancer-webgpu-port-20261009`.

The reconstructed editor smoke used the actual packaged Gold/Helper bundles: automatic replacement matched the benchmark WAV hashes, Original switching restored exact source hashes, both lanes played, native S split 20 rows into 21, and timed ghost-cursor projection survived Original/Enhanced switching using the retained deployed timing fixture. Reload restored the cached pair in 2.32 s from navigation, with cache-hit progress and no enhancement windows.

The older `profile-zip-kernels.mjs` / `watch-zip-kernel-profile.mjs` tools remain explicit original-ONNX kernel comparisons, not the new engine benchmark. Their default graph now comes from the retained source `models/zipenhancer.onnx`, which is no longer copied into the extension distribution.

### GPU swarm fallback and privacy

GPU-ineligible requesters use the saved custom L0 coordinator URL (the public coordinator by default). They upload both Original WAVs, never transcript text, only after the one-pair cache misses. The UI identifies **Swarm GPU (remote)** before upload. Disable Helper's Automatic Audio Enhancement to prevent these uploads. Swarm processing is not private on-device processing: a participating worker receives the audio and must be trusted; software cannot prevent a malicious volunteer from retaining it.

Workers remain explicitly opt-in in Gold's Local/Advanced local-model mode and require the updated Review Grader too. Enhancement-only workers need the packaged model and admitted hardware GPU, not the 1.31 GB ASR bundle. Registration advertises operation capabilities and an exact enhancement model identity; incompatible workers cannot lease a request. ASR and drafting keep their existing release and trusted-backend behavior.

The coordinator adds `/v1/enhance` to the existing worker/lease queue. Original and result audio travel as bounded binary multipart bodies. Owner capabilities protect status; worker and lease capabilities protect input downloads, progress and completion. Identities, model digests, WAV hashes and clocks are validated on both sides. Enhancement files are request-temporary and deleted after delivery, disconnect, failure or expiry; enhancement results are not persisted on the coordinator. Official workers bypass their local audio-pair cache for leased recordings. A missing/expired worker fails visibly with Originals intact, never falling back to CPU or silently forwarding to an ASR endpoint.

Dedicated GPU worker (separate browser/profile; never use a personal Chrome profile):

```powershell
node scripts/run-enhancement-swarm-worker.mjs --extension PATH_TO_BUILT_GOLD --grader PATH_TO_BUILT_GRADER --profile DEDICATED_PROFILE --coordinator COORDINATOR_URL
# Windows supervisor, using the same windowless scheduled-task pattern as the L0 services:
powershell -File scripts/Run-EnhancementSwarm.ps1 -Extension PATH_TO_BUILT_GOLD -Grader PATH_TO_BUILT_GRADER -Profile DEDICATED_PROFILE -Coordinator COORDINATOR_URL
node --import tsx scripts/smoke-enhancement-swarm.ts --coordinator COORDINATOR_URL --audio PUBLIC_SAMPLE.wav --audio PUBLIC_SAMPLE.wav --out NEW_PRIVATE_DIRECTORY
```

Deploy `coordinator.py` and `enhancement.py` together, then the matching built Gold worker, Grader and client. Existing ASR backends/tunnels need no restart or changes. The worker needs Node and the pinned Playwright browser; its supervisor restarts failed dedicated workers. Status logs contain lifecycle/error messages, not audio or bearer capabilities.

Local acceptance used the real reconstructed editor with WebGPU disabled, a separate actual GPU volunteer, and both complete private recordings. Swarm output exactly matched the verified GPU WAV hashes; native playback, Original/Enhanced switching, S splitting, and upload-free cache reuse passed. Removing the worker produced a bounded error with playable Originals. A GPU-enabled requester still completed locally without uploading. Grader-absent checks found no enhancement UI, bridge, offscreen runtime or model/swarm requests; actual Grader disable restored exact native Original PCM, canceled in-flight swarm work and removed settings. Re-enable recovered without a refresh. Coordinator tests: 64 passed; Gold: 360; Helper: 365; Grader: 13; shared runtime: 33.

### Native CUDA speed experiments (not the extension runtime)

`scripts/benchmark-zipenhancer-native.py` benchmarks the original verified checkpoint in PyTorch/CUDA. It does not replace Gold's WebGPU model, start a native inference service, or change the browser audio/cache path. The target is one warm second for **both complete 63.63-second recordings**, without an output-cache hit. Original four-second windows, three-second stride, per-lane normalization and output clocks are retained.

Run from the workspace root using the existing CUDA environment and private reference packages:

```powershell
$env:TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER = '0'
../babel_experiment/.venv-gigaam/Scripts/python.exe drafting/gold-drafting-extension/scripts/benchmark-zipenhancer-native.py --candidate autocastfp16 --batch-size 1 --compile --cuda-graph --fp16-residuals --repeats 5 --baseline-dir D:/babel_experiment_runs/zipenhancer-native-speed-20261009/baseline-v6 --out D:/babel_experiment_runs/zipenhancer-native-speed-20261009
node drafting/gold-drafting-extension/scripts/watch-zip-native-bench.mjs D:/babel_experiment_runs/zipenhancer-native-speed-20261009 53927
```

The launcher switch selects Triton's supported CUDA launcher; the installed PyTorch 2.9.1 Windows static launcher raised `OverflowError: Python int too large to convert to C long` in some compiled/captured variants. This is not a CPU fallback or a precision change. Compile/capture setup is reported separately and took several minutes for new graphs; warm timings must not be presented as first-use latency.

Timings include both source-PCM lanes already decoded in host RAM, resampling, CUDA STFT/model/ISTFT, transfers, source-clock restoration and in-memory WAV encoding. Disk I/O is excluded. Standalone network wall time and CUDA-event network time within the full pipeline are labeled separately. A baseline cache is only a numerical oracle: every measured candidate still executes every inference window on every repetition. Baseline reuse verifies checkpoint, original audio, configuration, DSP, implementation and WAV hashes.

Optional experiments include `--low-rank FRACTION` (actual two-matrix feedforward SVD), `--fused-softmax`, `--cudnn-tune`, `--channels-last`, and `--triton-norm` (custom anchored FP32-statistics CUDA kernels). `--fp16-residuals` stores parameters/residuals in FP16 while keeping the replaced nonlinear/normalization calculations in FP32. Every approximation records full-wave, quiet-window and speech-energy-proxy errors against untouched native FP32. These numerical checks are not listening tests or ASR-quality certification.

The first sweep measured approximately 9.1–9.7 s native FP32, 5.54 s FP16 CUDA graph, 2.46 s compiled FP16, 2.29 s compiled/manual graph, and 2.21 s with FP16 residual storage. One second was **not reached**. Half-rank feedforward compression measured 5.68 s and 6.2–6.6% relative waveform RMSE: no speed/quality win. Compiled batch two was slower and deviated by up to 8.3%, so it is not a recommended candidate. Faster isolated normalization kernels did not improve full-task latency. Complete private results, failed candidates and WAVs remain outside the repository; the SSE dashboard serves only experiment metadata.

`--backend tensorrt` and `--backend ort` exercise separate native GPU engines against the same DSP/reference contract; fixed batch one and FP32 or FP16 candidates only. Optional runtime packages can live under `--backend-packages` without modifying the existing Python environment. TensorRT FP32 was numerically close (relative RMSE below 0.000001) but took about 5.4 s. TensorRT FP16 variants collapsed the outputs to near-silence even after progressively stricter precision constraints: these are rejected experimental results, not valid speedups. ORT keeps CPU EP fallback disabled; unsupported neural placement is an explicit failure.
With ORT 1.23.2, verbose construction explicitly reported no CUDA kernel for `/model/phase_decoder/Atan` and attempted two host/device copies around it. The adapter rejects that CPU neural placement; `ZIPENHANCER_ORT_VERBOSE=1` exposes the diagnostic without enabling fallback.

### Controlled C++/CUDA attention experiment

`zipenhancer_tiled_attention.cu` and `zipenhancer_tiled_model.py` replace the materialized relative-attention matrices, not the entire host engine. The C ABI consumes contiguous GPU FP16 Q/K/position/value buffers on the caller's stream. A 16-query/64-key WMMA tile computes both matrix products with FP32 accumulation, online FP32 softmax statistics, and bounded shared scratch. Relative positions remain `N-1-query+key`; every key and all learned weights are retained. The nonlinear first-head consumer and both self-attention consumers use the same projected attention plan.

Build with CUDA 12.8 for the tested SM120 device from a compatible x64 Visual C++ developer environment (v142 was used):

```powershell
nvcc -std=c++17 -O3 -arch=sm_120 -lineinfo --shared -Xcompiler /MT drafting/gold-drafting-extension/scripts/zipenhancer_tiled_attention.cu -o D:/babel_experiment_runs/zipenhancer-native-speed-20261009/zipenhancer_tiled_attention-final.dll
```

Add `--tiled-attention-library D:/babel_experiment_runs/zipenhancer-native-speed-20261009/zipenhancer_tiled_attention-final.dll` to the native benchmark command above. Initial compilation/capture is excluded from warm timing and remains expensive. Five final full-pair warm runs measured 1.927–2.012 s, median 1.971 s, with approximately 0.090%/0.102% waveform relative RMSE versus native FP32. PyTorch-tracked peak allocated tensor memory fell from about 1.37 GB to 0.16 GB. Twenty deterministic head/batch/sequence-tail cases and CUDA graph replay were exercised. Larger query tiles and SFU exponentials did not improve the result and are not selected; batch four was slower at about 2.12 s.

Instruction profiling used a user-approved, one-off elevated Nsight Compute 2026.3.1 run, without changing global counter permissions. The full-counter command timed out: its retained report contains **525 launches, partial coverage of one real warmed window**, not a complete model profile. Assembly-correlated source counters were recovered. The slow attention reduction reached roughly 89% DRAM throughput with a long-scoreboard stall ratio near seven; this motivated eliminating the large intermediates. Counter-replay durations are not production latency measurements. `profile-zipenhancer-native-kernels.py` provides the real-input profiling range via CUDA profiler start/stop APIs.

The one-second full-task goal remains unmet. This native CUDA kernel is not bundled into the extension; the browser uses the separate WGSL implementation and measured timings above. Emscripten can port C++ host/planner code, but cannot make CUDA WMMA instructions execute through WebGPU.


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

Gold 0.2.42 uses the immutable release contract in `model-release.json`. The [release cutover guide](deploy/inference-release/README.md) covers staging the model, the shared WebGPU backend, protocol 3 workers, and mandatory client upgrades. `npm run prepare:inference-release -- --bundle DIR --sample-dir DIR --out EMPTY_DIR --evidence BACKEND_SMOKE_JSON --coordinator-evidence COORDINATOR_SMOKE_JSON` verifies the tested model and packages the production store ZIP, model files, backend dependency locks, deployment examples, and SHA-256 inventory. It requires successful real GPU/restart reports and keeps private corpus token sequences out of public metadata.

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
