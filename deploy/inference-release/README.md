# C-denoise v3 release cutover

This directory prepares release `c-denoise-v3-2026-10-03-r2` for Gold `0.2.42`.
Preparing it does not upload model files, submit the extension, restart a live service, or change the public proxy.

The model bundle is about 1.31 GB. Neural inference uses the pinned GigaAM/C-denoise graphs and the same WebGPU runtime on clients and trusted GPU backends. Completed punctuation labels travel with word timing, so rendering on another worker or after a restart does not need the original acoustic tensors. Legacy timing, v2 workers, and legacy backend results cannot satisfy the enforced release contract.

## Stage the model and backend first

1. Verify every file against `checksums.json`. Upload `models/c-denoise-v3-2026-10-03-r2/` to `/opt/babel/models/c-denoise-v3-2026-10-03-r2/` on the public asset host. Add the supplied Apache include to the existing TLS vhost and validate its configuration before reloading. This introduces a new immutable URL without changing the old model URL.
2. Copy `backend/` to a dedicated release directory on the trusted GPU machine. Use Node 22+ and Python 3.12. In `backend/`, run `npm ci --omit=dev`, `npx playwright install chromium`, and `python -m pip install -r requirements-webgpu.lock`. These Python versions were tested in a clean environment without Torch or Transformers. The public coordinator needs the Python dependencies only; it does not need Chromium or a GPU.
3. Set the variables in `trusted-backend.env.example`, adapting the two filesystem paths to that machine. On Windows, use a dedicated profile such as `D:\babel\c-denoise-v3\chromium` and the actual absolute `worker.mjs` path. Keep the old backend port/service untouched while staging port 8769. These variables must be passed by the service launcher; the application does not read this example file automatically.
4. Run `python scripts/Preflight-WebGPU.py`. It downloads and verifies the current model if needed, then proves actual hardware WebGPU execution. A failed GPU preflight blocks startup; there is no neural CPU fallback. The public sample and its license are included with the model assets.
5. Start the trusted service with `python -m uvicorn l0_draft_engine.app:app --host 127.0.0.1 --port 8769 --workers 1`. Startup repeats GPU preflight. `/health` must report `ok: true`, the current release ID, and `provider: webgpu`.
6. Keep the existing reverse SSH tunnel running and add a separate loopback tunnel from public port 28769 to GPU-machine port 8769. Start the new coordinator on a staging loopback port (for example 18769) with the supplied coordinator environment and `python -m uvicorn l0_draft_engine.coordinator:app --host 127.0.0.1 --port 18769 --workers 1`. One process owns its leases. Use the release-specific cache directory so the previous coordinator's data remains available for rollback.

Run two-lane transcription and draft tests against this staged coordinator with `X-Babel-Inference-Release: c-denoise-v3-2026-10-03-r2`. Worker registration requires protocol 3, bundle schema v3, and the exact model release ID. Verify that a request without the release header returns HTTP 426 before accepting an upload, that current workers can complete jobs, and that trusted fallback produces the same completed-label contract.

For durable Windows hosting, create `.venv` in the deployed backend directory, install the lock there, and run `scripts/Install-DurableService.ps1 -EngineTaskName 'Babel C-denoise v3 Engine' -TunnelTaskName 'Babel C-denoise v3 Tunnel' -EngineRoot FULL_BACKEND_PATH -EngineStartScript Start-WebGPU.ps1 -Port 8769 -TunnelTarget ethernetservers -TunnelRemotePort 28769`. These separate tasks preserve the old service and tunnel. The engine launcher uses the deployed `worker.mjs` and a dedicated `chromium` profile, enforces the current release, and repeats GPU preflight after restart. It runs at logon and heals failures while the user remains logged in; keep the GPU machine awake.

Before exposing the staged services, run `python scripts/Smoke-WebGPU.py --audio PATH_TO_PUBLIC_SAMPLE --report backend-smoke.json` and `python scripts/Smoke-Coordinator-WebGPU.py --audio PATH_TO_PUBLIC_SAMPLE --report coordinator-smoke.json` with the trusted backend environment. They exercise the real GPU runtime, backend and coordinator restarts, and the old-client rejection. Run them while the staged service is stopped: Chromium's dedicated profile supports one process at a time. The prepared `validation/` directory contains the local Windows GPU results; repeat preflight and smoke on the deployment GPU.

## Publish and activate

Submit the prepared Gold ZIP through the existing release workflow. Its production build points at the immutable model URL; localhost builds are prohibited in store packaging. The candidate version must still exceed every published or pending store version at submission time.

Once Gold 0.2.42 is available in the store, switch the existing public L0 proxy to the staged coordinator. The supplied coordinator setting enforces the current release. Chrome updates are asynchronous, so remaining old clients receive a clear update-required response until their extension updates and the task page is refreshed. Do not interrupt a draft in progress by forcibly reloading the extension.

New Local clients open model setup automatically. Updated Local clients also open setup if their current bundle is missing, outdated, or untested. SHA-matching cached files are reused across the old and new supplier URLs; changed files are downloaded and verified in a staging cache. Inference remains blocked until the current release passes the user's WebGPU test. Existing mode, API key, LLM, and volunteer preferences remain intact.

Local transcription is followed by Gold drafting when a key is configured and “Don't run the LLM” is unchecked. This uses the saved backend/model and audio-input preference. Volunteering remains opt-in.

Monitor model download failures, HTTP 426 responses, worker registrations, queue/fallback failures, and GPU preflight failures. Retain the old service, tunnel, and model files during the cutover window. If needed, restore the old public coordinator target while preparing a corrective extension release; a Chrome Store update cannot be rolled back instantly. Do not remove client caches or user settings during rollback.
