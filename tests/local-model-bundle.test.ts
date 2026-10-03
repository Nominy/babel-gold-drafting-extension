import { INFERENCE_RELEASE } from '../src/core/inference-release';
import test from 'node:test';
import assert from 'node:assert/strict';
import { webcrypto } from 'node:crypto';
import {
  LOCAL_MODEL_CACHE_NAME,
  getCachedLocalModelFile,
  getCachedBundleDescriptor,
  markLocalModelWebGpuTested,
  getLocalModelStatus,
  removeLocalModels,
  setupLocalModels,
  type LocalModelProgress
} from '../src/core/local-model-bundle';

const REQUIRED_PATHS = [
  'asr/v3_ctc.onnx',
  'asr/v3_ctc.yaml',
  'punctuation/context.fp16.onnx',
  'punctuation/denoise.fp16.onnx',
  'punctuation/c-denoise.json',
  'punctuation/gpu-placement.json',
  'punctuation/config.json',
  'punctuation/tokenizer.json',
  'punctuation/tokenizer_config.json',
  'punctuation/special_tokens_map.json',
  'punctuation/vocab.txt'
] as const;

type ManifestFile = {
  path: string;
  bytes: number;
  sha256: string;
  role: string;
};

type Manifest = {
  releaseId: string;
  schema: string;
  generatedAt: string;
  targetBytes: number;
  totalBytes: number;
  pass: boolean;
  runtimeLibrariesExcluded: boolean;
  files: ManifestFile[];
  models: Record<string, unknown>;
  source: { asrCheckpointSha256: string; cDenoiseCheckpointSha256: string; baseModelFiles: Record<string, string> };
  validation: Record<string, unknown>;
};

class MemoryCache {
  readonly entries = new Map<string, Response>();
  async keys(): Promise<Request[]> { return [...this.entries.keys()].map(url => new Request(url)); }

  async match(request: RequestInfo | URL): Promise<Response | undefined> {
    const response = this.entries.get(requestKey(request));
    return response?.clone();
  }

  async put(request: RequestInfo | URL, response: Response): Promise<void> {
    this.entries.set(requestKey(request), response.clone());
  }
}

class MemoryCacheStorage {
  readonly caches = new Map<string, MemoryCache>();

  async open(cacheName: string): Promise<Cache> {
    let cache = this.caches.get(cacheName);
    if (!cache) {
      cache = new MemoryCache();
      this.caches.set(cacheName, cache);
    }
    return cache as unknown as Cache;
  }

  async has(cacheName: string): Promise<boolean> {
    return this.caches.has(cacheName);
  }

  async delete(cacheName: string): Promise<boolean> {
    return this.caches.delete(cacheName);
  }

  async keys(): Promise<string[]> {
    return [...this.caches.keys()];
  }
}

class MemoryStorageArea {
  readonly values: Record<string, unknown> = {};

  async get(keys?: string | string[] | Record<string, unknown> | null): Promise<Record<string, unknown>> {
    if (keys == null) {
      return { ...this.values };
    }
    const requestedKeys = typeof keys === 'string' ? [keys] : Array.isArray(keys) ? keys : Object.keys(keys);
    return Object.fromEntries(
      requestedKeys
        .filter((key) => Object.hasOwn(this.values, key))
        .map((key) => [key, this.values[key]])
    );
  }

  async set(items: Record<string, unknown>): Promise<void> {
    // Match Chrome's storage serialization rather than preserving JS insertion order.
    Object.assign(this.values, JSON.parse(JSON.stringify(items, (_key, value) =>
      value && typeof value === 'object' && !Array.isArray(value)
        ? Object.fromEntries(Object.keys(value).sort().map((key) => [key, value[key]]))
        : value)));
  }

  async remove(keys: string | string[]): Promise<void> {
    for (const key of typeof keys === 'string' ? [keys] : keys) {
      delete this.values[key];
    }
  }
}

type Harness = {
  cacheStorage: MemoryCacheStorage;
  storageArea: MemoryStorageArea;
  requestedUrls: string[];
  setBundle(baseUrl: string, manifest: Manifest, contents: Record<string, Uint8Array>): void;
};

function requestKey(request: RequestInfo | URL): string {
  return typeof request === 'string' || request instanceof URL ? request.toString() : request.url;
}

function bytes(value: string): Uint8Array {
  return new TextEncoder().encode(value);
}

async function sha256(value: Uint8Array): Promise<string> {
  const digest = await webcrypto.subtle.digest('SHA-256', value);
  return Buffer.from(digest).toString('hex');
}

async function createBundle(
  label: string,
  mutate?: (manifest: Manifest, contents: Record<string, Uint8Array>) => void | Promise<void>
): Promise<{ manifest: Manifest; contents: Record<string, Uint8Array> }> {
  const contents = Object.fromEntries(
    REQUIRED_PATHS.map((path, index) => [path, bytes(`${label}:${index}:${path}`)])
  );
  const files: ManifestFile[] = [];
  for (const path of REQUIRED_PATHS) {
    files.push({
      path,
      bytes: contents[path].byteLength,
      sha256: await sha256(contents[path]),
      role: path.endsWith('.onnx') ? 'model' : 'metadata'
    });
  }
  const manifest: Manifest = {
    schema: 'babel-browser-model-bundle-v3', releaseId: INFERENCE_RELEASE.id,
    generatedAt: '2026-08-30T00:00:00.000Z',
    targetBytes: 1_500_000_000,
    totalBytes: files.reduce((total, file) => total + file.bytes, 0),
    pass: true,
    runtimeLibrariesExcluded: true,
    files,
    models: {},
    source: {
      asrCheckpointSha256: '02cea9973d0e839f6a3eeca101b83a93f93a066c2da2e3ebfa176d57e61d84d3',
      cDenoiseCheckpointSha256: '0a01c3535fb66627b13f266bc59ab7b95c2aa85f7413a051d1e17294515ded5a',
      baseModelFiles: { 'model.safetensors': '1'.repeat(64) }
    },
    validation: { numericExport: { pass: true, reportSha256: '2'.repeat(64), limits: { maxAbs: 0.2 },
      cases: [{ id: 'context-and-core', pass: true, comparisons: { logits: { pass: true, finite: true, maxAbs: 0.01, rmse: 0.001 } } }],
      asr: [{ id: 'asr', pass: true, comparisons: { logits: { pass: true, finite: true, maxAbs: 0.01, rmse: 0.001 } } }]
    } }
  };
  await mutate?.(manifest, contents);
  manifest.totalBytes = manifest.files.reduce((total, file) => total + file.bytes, 0);
  return { manifest, contents };
}

function installHarness(): Harness {
  const cacheStorage = new MemoryCacheStorage();
  const storageArea = new MemoryStorageArea();
  const requestedUrls: string[] = [];
  const responses = new Map<string, () => Response>();

  Object.defineProperty(globalThis, 'caches', {
    configurable: true,
    writable: true,
    value: cacheStorage as unknown as CacheStorage
  });
  Object.defineProperty(globalThis, 'crypto', {
    configurable: true,
    writable: true,
    value: webcrypto
  });
  Object.defineProperty(globalThis, 'chrome', {
    configurable: true,
    writable: true,
    value: {
      storage: {
        local: storageArea
      }
    }
  });
  Object.defineProperty(globalThis, 'fetch', {
    configurable: true,
    writable: true,
    value: async (input: string | URL | Request) => {
      const url = requestKey(input);
      requestedUrls.push(url);
      const response = responses.get(url);
      return response ? response() : new Response('not found', { status: 404 });
    }
  });

  return {
    cacheStorage,
    storageArea,
    requestedUrls,
    setBundle(baseUrl, manifest, contents) {
      responses.set(`${baseUrl}/manifest.json`, () =>
        new Response(JSON.stringify(manifest), {
          status: 200,
          headers: { 'content-type': 'application/json' }
        })
      );
      for (const [path, content] of Object.entries(contents)) {
        responses.set(`${baseUrl}/${path}`, () =>
          new Response(content.slice(), {
            status: 200,
            headers: { 'content-type': 'application/octet-stream' }
          })
        );
      }
    }
  };
}

function enterOffscreenEnvironment(): void {
  Object.defineProperty(globalThis, 'chrome', {
    configurable: true,
    writable: true,
    value: { runtime: {} }
  });
}

function installedModelCache(harness: Harness): MemoryCache {
  const modelCaches = [...harness.cacheStorage.caches.entries()].filter(([cacheName]) =>
    cacheName.startsWith(`${LOCAL_MODEL_CACHE_NAME}:bundle:`)
  );
  assert.equal(modelCaches.length, 1);
  return modelCaches[0][1];
}

test('installs a fully verified bundle and reads only manifest-listed files from the ready cache', async () => {
  const harness = installHarness();
  const baseUrl = 'https://models.example.test/gigaam';
  const bundle = await createBundle('ready');
  harness.setBundle(baseUrl, bundle.manifest, bundle.contents);

  const status = await setupLocalModels(
    ' https://user:password@models.example.test/gigaam///?token=secret#fragment '
  );
  assert.deepEqual(status, {
    state: 'ready',
    completedBytes: bundle.manifest.totalBytes,
    totalBytes: bundle.manifest.totalBytes,
    tested: false,
    source: bundle.manifest.source
  });
  assert.equal(harness.requestedUrls[0], `${baseUrl}/manifest.json`);
  assert.ok(harness.requestedUrls.every((url) => !url.includes('password') && !url.includes('token=')));

  const cached = await getCachedLocalModelFile('asr/v3_ctc.onnx', baseUrl);
  assert.ok(cached);
  assert.deepEqual(
    new Uint8Array(await cached.arrayBuffer()),
    bundle.contents['asr/v3_ctc.onnx']
  );
  assert.equal(await getCachedLocalModelFile('not-in-manifest.bin', baseUrl), null);
  assert.equal((await getLocalModelStatus(`${baseUrl}/`)).state, 'ready');
  assert.equal(harness.cacheStorage.caches.size, 1);
});

test('large model graphs install from verified byte ranges rather than one long transfer', async () => {
  const harness = installHarness();
  const baseUrl = 'https://models.example.test/ranged';
  const path = 'asr/v3_ctc.onnx';
  const graph = new Uint8Array(16 * 1024 * 1024 + 37);
  graph.fill(37);
  const bundle = await createBundle('ranged', async (manifest, contents) => {
    contents[path] = graph;
    const entry = manifest.files.find((file) => file.path === path);
    assert.ok(entry);
    entry.bytes = graph.byteLength;
    entry.sha256 = await sha256(graph);
  });
  harness.setBundle(baseUrl, bundle.manifest, bundle.contents);

  const fetchWithoutRanges = globalThis.fetch;
  const requestedRanges: string[] = [];
  globalThis.fetch = async (input, init) => {
    const url = requestKey(input);
    const range = new Headers(init?.headers).get('range');
    if (url !== `${baseUrl}/${path}` || !range) {
      return fetchWithoutRanges(input, init);
    }
    requestedRanges.push(range);
    const match = /^bytes=(\d+)-(\d+)$/.exec(range);
    assert.ok(match);
    const start = Number(match[1]);
    const end = Number(match[2]);
    return new Response(graph.slice(start, end + 1), {
      status: 206,
      headers: { 'content-range': `bytes ${start}-${end}/${graph.byteLength}` }
    });
  };
  const progress: LocalModelProgress[] = [];
  const status = await setupLocalModels(baseUrl, (update) => progress.push(update));

  assert.equal(status.state, 'ready');
  assert.deepEqual(requestedRanges, [
    'bytes=0-16777215',
    `bytes=16777216-${graph.byteLength - 1}`
  ]);
  assert.ok(progress.some((update) => update.currentPath === path && update.completedBytes === 16777216));
  const cached = await getCachedLocalModelFile(path, baseUrl);
  assert.ok(cached);
  assert.equal(await sha256(new Uint8Array(await cached.arrayBuffer())), await sha256(graph));
});

test('offscreen lookup discovers one complete manifest-backed bundle without chrome.storage', async () => {
  const harness = installHarness();
  const baseUrl = 'https://models.example.test/offscreen-valid';
  const bundle = await createBundle('offscreen-valid');
  harness.setBundle(baseUrl, bundle.manifest, bundle.contents);
  await setupLocalModels(baseUrl);
  enterOffscreenEnvironment();

  const cached = await getCachedLocalModelFile(
    'asr/v3_ctc.onnx',
    `${baseUrl}///?ignored=true#fragment`
  );
  assert.ok(cached);
  assert.deepEqual(
    new Uint8Array(await cached.arrayBuffer()),
    bundle.contents['asr/v3_ctc.onnx']
  );
  assert.equal(
    await getCachedLocalModelFile('asr/v3_ctc.onnx', 'https://models.example.test/other'),
    null
  );
});

test('offscreen lookup rejects bundles without a verified v3 manifest', async () => {
  const harness = installHarness();
  const baseUrl = 'https://models.example.test/offscreen-missing-manifest';
  const bundle = await createBundle('offscreen-missing-manifest');
  harness.setBundle(baseUrl, bundle.manifest, bundle.contents);
  await setupLocalModels(baseUrl);
  installedModelCache(harness).entries.delete(`${baseUrl}/manifest.json`);
  enterOffscreenEnvironment();

  assert.equal(await getCachedLocalModelFile('asr/v3_ctc.onnx', baseUrl), null);
});

test('offscreen lookup rejects ambiguous complete caches for the same bundle URL', async () => {
  const harness = installHarness();
  const baseUrl = 'https://models.example.test/offscreen-ambiguous';
  const bundle = await createBundle('offscreen-ambiguous');
  harness.setBundle(baseUrl, bundle.manifest, bundle.contents);
  await setupLocalModels(baseUrl);

  const source = installedModelCache(harness);
  const duplicate = (await harness.cacheStorage.open(
    `${LOCAL_MODEL_CACHE_NAME}:bundle:duplicate`
  )) as unknown as MemoryCache;
  for (const [url, response] of source.entries) {
    await duplicate.put(url, response);
  }
  const pointerResponse = duplicate.entries.get(`${baseUrl}/__bundle-pointer.json`);
  assert.ok(pointerResponse);
  const pointer = await pointerResponse.json();
  await duplicate.put(`${baseUrl}/__bundle-pointer.json`,
    new Response(JSON.stringify({ ...pointer, cacheName: `${LOCAL_MODEL_CACHE_NAME}:bundle:duplicate` })));
  enterOffscreenEnvironment();

  assert.equal(await getCachedLocalModelFile('asr/v3_ctc.onnx', baseUrl), null);
});

test('offscreen lookup rejects a cached file that is not listed in the cached manifest', async () => {
  const harness = installHarness();
  const baseUrl = 'https://models.example.test/offscreen-unlisted';
  const bundle = await createBundle('offscreen-unlisted');
  harness.setBundle(baseUrl, bundle.manifest, bundle.contents);
  await setupLocalModels(baseUrl);

  const cache = installedModelCache(harness);
  await cache.put(`${baseUrl}/unlisted.bin`, new Response('unlisted', { status: 200 }));
  enterOffscreenEnvironment();

  assert.equal(await getCachedLocalModelFile('unlisted.bin', baseUrl), null);
});

test('offscreen lookup rejects a cache missing any manifest-listed file', async () => {
  const harness = installHarness();
  const baseUrl = 'https://models.example.test/offscreen-incomplete';
  const bundle = await createBundle('offscreen-incomplete');
  harness.setBundle(baseUrl, bundle.manifest, bundle.contents);
  await setupLocalModels(baseUrl);

  const cache = installedModelCache(harness);
  cache.entries.delete(`${baseUrl}/punctuation/vocab.txt`);
  enterOffscreenEnvironment();

  assert.equal(await getCachedLocalModelFile('asr/v3_ctc.onnx', baseUrl), null);
});

test('offscreen lookup rejects a cached manifest with an invalid schema', async () => {
  const harness = installHarness();
  const baseUrl = 'https://models.example.test/offscreen-invalid-manifest';
  const bundle = await createBundle('offscreen-invalid-manifest');
  harness.setBundle(baseUrl, bundle.manifest, bundle.contents);
  await setupLocalModels(baseUrl);

  const cache = installedModelCache(harness);
  await cache.put(
    `${baseUrl}/manifest.json`,
    new Response(JSON.stringify({ ...bundle.manifest, schema: 'unsupported-schema' }), {
      status: 200,
      headers: { 'content-type': 'application/json' }
    })
  );
  enterOffscreenEnvironment();

  assert.equal(await getCachedLocalModelFile('asr/v3_ctc.onnx', baseUrl), null);
});

test('release migration reuses only verified cached bytes across supplier URLs and requires a new GPU test', async () => {
  const harness = installHarness();
  const oldUrl = 'https://models.example.test/previous';
  const newUrl = 'https://models.example.test/release';
  const bundle = await createBundle('same-verified-bytes');
  harness.setBundle(oldUrl, bundle.manifest, bundle.contents);
  await setupLocalModels(oldUrl);
  const pointer = harness.storageArea.values['babel_gold_local_model_bundle_pointer'] as { releaseId?: string };
  delete pointer.releaseId;
  assert.equal(await getCachedBundleDescriptor(oldUrl), null);
  harness.requestedUrls.length = 0;
  harness.setBundle(newUrl, bundle.manifest, bundle.contents);
  await setupLocalModels(newUrl);
  assert.deepEqual(harness.requestedUrls, [`${newUrl}/manifest.json`]);
  const descriptor = await getCachedBundleDescriptor(newUrl);
  assert.equal(descriptor?.releaseId, INFERENCE_RELEASE.id);
  assert.equal(descriptor?.tested, false);
  const graph = await getCachedLocalModelFile('asr/v3_ctc.onnx', newUrl);
  assert.ok(graph);
  assert.equal(await sha256(new Uint8Array(await graph.arrayBuffer())), bundle.manifest.files[0].sha256);
});

test('installer refuses the old quantized and distilled model bundle schema', async () => {
  const harness = installHarness();
  const baseUrl = 'https://models.example.test/old-bundle';
  const bundle = await createBundle('old-bundle');
  bundle.manifest.schema = 'babel-browser-model-bundle-v1';
  bundle.manifest.targetBytes = 500_000_000;
  harness.setBundle(baseUrl, bundle.manifest, bundle.contents);

  await assert.rejects(setupLocalModels(baseUrl), /manifest schema must be babel-browser-model-bundle-v3/);
  assert.deepEqual(await harness.cacheStorage.keys(), []);
});

test('a present storage area never falls back when its active pointer is missing', async () => {
  const harness = installHarness();
  const baseUrl = 'https://models.example.test/pointer-required';
  const bundle = await createBundle('pointer-required');
  harness.setBundle(baseUrl, bundle.manifest, bundle.contents);
  await setupLocalModels(baseUrl);
  delete harness.storageArea.values['babel_gold_local_model_bundle_pointer'];

  assert.equal(await getCachedLocalModelFile('asr/v3_ctc.onnx', baseUrl), null);
});

test('a replacement hash failure preserves cached data but stops readiness without fallback', async () => {
  const harness = installHarness();
  const baseUrl = 'https://models.example.test/bundle';
  const original = await createBundle('original');
  harness.setBundle(baseUrl, original.manifest, original.contents);
  await setupLocalModels(baseUrl);
  const originalCacheNames = await harness.cacheStorage.keys();

  const corrupt = await createBundle('replacement', (manifest) => {
    manifest.files[0].sha256 = '0'.repeat(64);
  });
  harness.setBundle(baseUrl, corrupt.manifest, corrupt.contents);
  await assert.rejects(setupLocalModels(baseUrl), /failed SHA-256 verification/);

  assert.equal(await getCachedLocalModelFile('asr/v3_ctc.onnx', baseUrl), null);
  for (const name of originalCacheNames) assert.equal(await harness.cacheStorage.has(name), true);
  assert.equal((await getLocalModelStatus(baseUrl)).state, 'error');
});

test('a replacement size failure also preserves the active cache', async () => {
  const harness = installHarness();
  const baseUrl = 'https://models.example.test/bundle';
  const original = await createBundle('original-size');
  harness.setBundle(baseUrl, original.manifest, original.contents);
  await setupLocalModels(baseUrl);

  const corrupt = await createBundle('replacement-size', (manifest) => {
    manifest.files[0].bytes += 1;
  });
  harness.setBundle(baseUrl, corrupt.manifest, corrupt.contents);
  await assert.rejects(setupLocalModels(baseUrl), /has size .* expected/);

  assert.equal(await getCachedLocalModelFile('asr/v3_ctc.onnx', baseUrl), null);
  assert.equal((await getLocalModelStatus(baseUrl)).state, 'error');
});

test('rejects traversal before fetching or caching any manifest file', async () => {
  const harness = installHarness();
  const baseUrl = 'https://models.example.test/traversal';
  const bundle = await createBundle('traversal', async (manifest, contents) => {
    const traversalContent = bytes('must never be fetched');
    manifest.files.push({
      path: '../outside.onnx',
      bytes: traversalContent.byteLength,
      sha256: await sha256(traversalContent),
      role: 'model'
    });
    contents['../outside.onnx'] = traversalContent;
  });
  harness.setBundle(baseUrl, bundle.manifest, bundle.contents);

  await assert.rejects(setupLocalModels(baseUrl), /Unsafe local model file path/);
  assert.deepEqual(harness.requestedUrls, [`${baseUrl}/manifest.json`]);
  assert.deepEqual(await harness.cacheStorage.keys(), []);
  assert.equal(await getCachedLocalModelFile('../outside.onnx', baseUrl), null);
});

test('an interrupted install clears stale progress while retaining resumable data', async () => {
  const harness = installHarness();
  const baseUrl = 'https://models.example.test/interrupted-install';
  const operationId = 'abandoned-first-install';
  const stagingCacheName = `${LOCAL_MODEL_CACHE_NAME}:bundle:${operationId}`;
  await harness.cacheStorage.open(stagingCacheName);
  await harness.storageArea.set({
    babel_gold_local_model_bundle_status: {
      baseUrl,
      operationId,
      startedAt: Date.now() - 31 * 60 * 1000,
      updatedAt: Date.now() - 31 * 60 * 1000,
      state: 'downloading',
      completedBytes: 123,
      totalBytes: 456,
      currentPath: REQUIRED_PATHS[0]
    }
  });

  assert.deepEqual(await getLocalModelStatus(baseUrl), {
    state: 'not-installed',
    completedBytes: 0,
    totalBytes: 0
  });
  assert.equal(await harness.cacheStorage.has(stagingCacheName), true);
  assert.equal(
    Object.hasOwn(harness.storageArea.values, 'babel_gold_local_model_bundle_status'),
    false
  );
});

test('a recent cross-context download remains live without an owner in this module realm', async () => {
  const harness = installHarness();
  const baseUrl = 'https://models.example.test/cross-context-download';
  const operationId = 'owned-by-another-extension-page';
  const stagingCacheName = `${LOCAL_MODEL_CACHE_NAME}:bundle:${operationId}`;
  await harness.cacheStorage.open(stagingCacheName);
  await harness.storageArea.set({
    babel_gold_local_model_bundle_status: {
      baseUrl,
      operationId,
      startedAt: Date.now(),
      updatedAt: Date.now(),
      state: 'downloading',
      completedBytes: 123,
      totalBytes: 456,
      currentPath: REQUIRED_PATHS[0]
    }
  });

  assert.deepEqual(await getLocalModelStatus(baseUrl), {
    state: 'downloading',
    completedBytes: 123,
    totalBytes: 456,
    currentPath: REQUIRED_PATHS[0],
    error: undefined
  });
  assert.equal(await harness.cacheStorage.has(stagingCacheName), true);
  assert.equal(
    Object.hasOwn(harness.storageArea.values, 'babel_gold_local_model_bundle_status'),
    true
  );
});

test('an interrupted replacement reports the complete active bundle and preserves its cache', async () => {
  const harness = installHarness();
  const baseUrl = 'https://models.example.test/interrupted-replacement';
  const original = await createBundle('active-before-interruption');
  harness.setBundle(baseUrl, original.manifest, original.contents);
  await setupLocalModels(baseUrl);
  const activePointer = harness.storageArea.values[
    'babel_gold_local_model_bundle_pointer'
  ] as { cacheName: string };
  const operationId = 'abandoned-replacement';
  const stagingCacheName = `${LOCAL_MODEL_CACHE_NAME}:bundle:${operationId}`;
  await harness.cacheStorage.open(stagingCacheName);
  await harness.storageArea.set({
    babel_gold_local_model_bundle_status: {
      baseUrl,
      operationId,
      startedAt: Date.now() - 31 * 60 * 1000,
      updatedAt: Date.now() - 31 * 60 * 1000,
      state: 'downloading',
      completedBytes: 321,
      totalBytes: 654,
      currentPath: REQUIRED_PATHS[1]
    }
  });

  assert.deepEqual(await getLocalModelStatus(baseUrl), {
    state: 'ready',
    completedBytes: original.manifest.totalBytes,
    totalBytes: original.manifest.totalBytes,
    tested: false,
    source: original.manifest.source
  });
  assert.equal(await harness.cacheStorage.has(activePointer.cacheName), true);
  assert.equal(await harness.cacheStorage.has(stagingCacheName), true);
  assert.equal(
    Object.hasOwn(harness.storageArea.values, 'babel_gold_local_model_bundle_status'),
    false
  );
});

test('a live current download continues to report its stored progress', async () => {
  const harness = installHarness();
  const baseUrl = 'https://models.example.test/live-progress';
  const bundle = await createBundle('live-progress');
  harness.setBundle(baseUrl, bundle.manifest, bundle.contents);
  const firstFileUrl = `${baseUrl}/${REQUIRED_PATHS[0]}`;
  const regularFetch = globalThis.fetch;
  let releaseFirstFile!: () => void;
  const firstFileReleased = new Promise<void>((resolve) => {
    releaseFirstFile = resolve;
  });
  let markFirstFileRequested!: () => void;
  const firstFileRequested = new Promise<void>((resolve) => {
    markFirstFileRequested = resolve;
  });
  globalThis.fetch = async (input: string | URL | Request, init?: RequestInit) => {
    if (requestKey(input) === firstFileUrl) {
      markFirstFileRequested();
      await firstFileReleased;
    }
    return regularFetch(input, init);
  };

  const setup = setupLocalModels(baseUrl);
  await firstFileRequested;
  assert.deepEqual(await getLocalModelStatus(baseUrl), {
    state: 'downloading',
    completedBytes: 0,
    totalBytes: bundle.manifest.totalBytes,
    currentPath: REQUIRED_PATHS[0],
    error: undefined
  });
  releaseFirstFile();
  await setup;
});

test('remove clears the active pointer, readiness, and every model cache', async () => {
  const harness = installHarness();
  const baseUrl = 'https://models.example.test/removable';
  const bundle = await createBundle('remove');
  harness.setBundle(baseUrl, bundle.manifest, bundle.contents);
  await setupLocalModels(baseUrl);
  await harness.cacheStorage.open(`${LOCAL_MODEL_CACHE_NAME}:orphaned-stage`);
  await harness.cacheStorage.open(LOCAL_MODEL_CACHE_NAME);

  await removeLocalModels();

  assert.deepEqual(await getLocalModelStatus(baseUrl), {
    state: 'not-installed',
    completedBytes: 0,
    totalBytes: 0
  });
  assert.equal(await getCachedLocalModelFile('asr/v3_ctc.onnx', baseUrl), null);
  assert.deepEqual(await harness.cacheStorage.keys(), []);
  assert.deepEqual(harness.storageArea.values, {});
});

test('download progress is monotonic and identifies the current sequential file', async () => {
  const harness = installHarness();
  const baseUrl = 'https://models.example.test/progress';
  const bundle = await createBundle('progress');
  harness.setBundle(baseUrl, bundle.manifest, bundle.contents);
  const progress: LocalModelProgress[] = [];

  await setupLocalModels(baseUrl, (update) => progress.push({ ...update }));

  assert.ok(progress.length >= REQUIRED_PATHS.length);
  assert.equal(progress[0].completedBytes, 0);
  assert.equal(progress.at(-1)?.completedBytes, bundle.manifest.totalBytes);
  assert.ok(progress.every((update) => update.totalBytes === bundle.manifest.totalBytes));
  assert.ok(progress.every((update) => REQUIRED_PATHS.includes(update.currentPath as never)));
  for (let index = 1; index < progress.length; index += 1) {
    assert.ok(progress[index].completedBytes >= progress[index - 1].completedBytes);
  }
  assert.deepEqual(
    [...new Set(progress.map((update) => update.currentPath))],
    bundle.manifest.files.map((file) => file.path)
  );
});

test('v2 readiness is invalidated without deleting old data before v3 installation succeeds', async () => {
  const harness = installHarness();
  const baseUrl = 'https://models.example.test/upgrade';
  const oldCacheName = `${LOCAL_MODEL_CACHE_NAME}:bundle:old-v2`;
  const oldCache = await harness.cacheStorage.open(oldCacheName);
  await oldCache.put(`${baseUrl}/punctuation/model.fp16.onnx`, new Response('old learned BERT graph'));
  harness.storageArea.values.babel_gold_local_model_bundle_pointer = {
    version: 2, cacheName: oldCacheName, baseUrl, totalBytes: 22,
    files: [{ path: 'punctuation/model.fp16.onnx', bytes: 22, sha256: '0'.repeat(64) }]
  };
  assert.equal((await getLocalModelStatus(baseUrl)).state, 'error');
  assert.equal(await getCachedLocalModelFile('punctuation/model.fp16.onnx', baseUrl), null);
  assert.equal(await harness.cacheStorage.has(oldCacheName), true);
  const bundle = await createBundle('upgrade');
  harness.setBundle(baseUrl, bundle.manifest, bundle.contents);
  await setupLocalModels(baseUrl);
  assert.equal(await harness.cacheStorage.has(oldCacheName), false);
  assert.equal((await getLocalModelStatus(baseUrl)).tested, false);
});

test('decorative pass cannot admit wrong provenance or failed numeric parity', async () => {
  for (const corrupt of [
    (manifest: Manifest) => { manifest.source.cDenoiseCheckpointSha256 = '0'.repeat(64); },
    (manifest: Manifest) => { manifest.validation = { numericExport: { pass: false } }; },
    (manifest: Manifest) => {
      manifest.validation = { numericExport: { pass: true, reportSha256: '1'.repeat(64), limits: { maxAbs: 0.2 },
        cases: [{ id: 'invalid', pass: true, comparisons: { logits: { pass: true, finite: true, maxAbs: null, rmse: 0 } } }],
        asr: [{ id: 'asr', pass: true, comparisons: { logits: { pass: true, finite: true, maxAbs: 0, rmse: 0 } } }] } };
    }
  ]) {
    const harness = installHarness();
    const baseUrl = 'https://models.example.test/unvalidated';
    const bundle = await createBundle('unvalidated', corrupt);
    harness.setBundle(baseUrl, bundle.manifest, bundle.contents);
    await assert.rejects(setupLocalModels(baseUrl), /provenance|numeric|parity/i);
    assert.equal((await getLocalModelStatus(baseUrl)).state, 'error');
    assert.equal(await getCachedBundleDescriptor(baseUrl), null);
    assert.deepEqual(harness.requestedUrls, [`${baseUrl}/manifest.json`]);
  }
});

test('explicit retry resumes checksum-verified whole files and never activates a partial bundle', async () => {
  const harness = installHarness();
  const baseUrl = 'https://models.example.test/resume';
  const bundle = await createBundle('resume');
  harness.setBundle(baseUrl, bundle.manifest, bundle.contents);
  const fetchNormally = globalThis.fetch;
  const interruptedPath = REQUIRED_PATHS[3];
  globalThis.fetch = async (input, init) => requestKey(input) === `${baseUrl}/${interruptedPath}`
    ? new Response('interrupted', { status: 503 }) : fetchNormally(input, init);
  await assert.rejects(setupLocalModels(baseUrl), /HTTP 503/);
  assert.equal(await getCachedBundleDescriptor(baseUrl), null);
  globalThis.fetch = fetchNormally;
  harness.requestedUrls.length = 0;
  await setupLocalModels(baseUrl);
  for (const path of REQUIRED_PATHS.slice(0, 3)) assert.equal(harness.requestedUrls.includes(`${baseUrl}/${path}`), false);
  assert.equal((await getLocalModelStatus(baseUrl)).tested, false);
  assert.equal((await harness.cacheStorage.keys()).length, 1);
});

test('WebGPU activation is bound to bundle identity and is visible in the offscreen cache', async () => {
  const harness = installHarness();
  const baseUrl = 'https://models.example.test/tested';
  const first = await createBundle('first');
  harness.setBundle(baseUrl, first.manifest, first.contents);
  await setupLocalModels(baseUrl);
  const descriptor = await getCachedBundleDescriptor(baseUrl);
  assert.ok(descriptor);
  await markLocalModelWebGpuTested(baseUrl, descriptor.identity);
  const tested = await getCachedBundleDescriptor(baseUrl);
  assert.equal(tested?.identity, descriptor.identity);
  assert.equal(tested?.tested, true);
  const second = await createBundle('second');
  harness.setBundle(baseUrl, second.manifest, second.contents);
  await setupLocalModels(baseUrl);
  await assert.rejects(markLocalModelWebGpuTested(baseUrl, descriptor.identity), /changed during its test/);
  const replacement = await getCachedBundleDescriptor(baseUrl);
  assert.ok(replacement);
  assert.notEqual(replacement.identity, descriptor.identity);
  assert.equal(replacement.tested, false);
  await markLocalModelWebGpuTested(baseUrl, replacement.identity);
  enterOffscreenEnvironment();
  const offscreen = await getCachedBundleDescriptor(baseUrl);
  assert.equal(offscreen?.identity, replacement.identity);
  assert.equal(offscreen?.tested, true);
});
