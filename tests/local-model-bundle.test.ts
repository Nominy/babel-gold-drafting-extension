import test from 'node:test';
import assert from 'node:assert/strict';
import { webcrypto } from 'node:crypto';
import {
  LOCAL_MODEL_CACHE_NAME,
  getCachedLocalModelFile,
  getLocalModelStatus,
  removeLocalModels,
  setupLocalModels,
  type LocalModelProgress
} from '../src/core/local-model-bundle';

const REQUIRED_PATHS = [
  'asr/v3_ctc.onnx',
  'asr/v3_ctc.yaml',
  'punctuation/model.int8.onnx',
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
  schema: string;
  generatedAt: string;
  targetBytes: number;
  totalBytes: number;
  pass: boolean;
  runtimeLibrariesExcluded: boolean;
  files: ManifestFile[];
  models: Record<string, unknown>;
};

class MemoryCache {
  readonly entries = new Map<string, Response>();

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
    Object.assign(this.values, items);
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
    schema: 'babel-browser-model-bundle-v1',
    generatedAt: '2026-08-30T00:00:00.000Z',
    targetBytes: 500_000_000,
    totalBytes: files.reduce((total, file) => total + file.bytes, 0),
    pass: true,
    runtimeLibrariesExcluded: true,
    files,
    models: {}
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
    totalBytes: bundle.manifest.totalBytes
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

test('offscreen lookup preserves complete bundles installed before manifest markers', async () => {
  const harness = installHarness();
  const baseUrl = 'https://models.example.test/offscreen-legacy';
  const bundle = await createBundle('offscreen-legacy');
  harness.setBundle(baseUrl, bundle.manifest, bundle.contents);
  await setupLocalModels(baseUrl);
  installedModelCache(harness).entries.delete(`${baseUrl}/manifest.json`);
  enterOffscreenEnvironment();

  const cached = await getCachedLocalModelFile('asr/v3_ctc.onnx', baseUrl);
  assert.ok(cached);
  assert.deepEqual(
    new Uint8Array(await cached.arrayBuffer()),
    bundle.contents['asr/v3_ctc.onnx']
  );
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

test('a present storage area never falls back when its active pointer is missing', async () => {
  const harness = installHarness();
  const baseUrl = 'https://models.example.test/pointer-required';
  const bundle = await createBundle('pointer-required');
  harness.setBundle(baseUrl, bundle.manifest, bundle.contents);
  await setupLocalModels(baseUrl);
  delete harness.storageArea.values['babel_gold_local_model_bundle_pointer'];

  assert.equal(await getCachedLocalModelFile('asr/v3_ctc.onnx', baseUrl), null);
});

test('a replacement hash failure leaves the previously ready bundle and pointer untouched', async () => {
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

  const cached = await getCachedLocalModelFile('asr/v3_ctc.onnx', baseUrl);
  assert.ok(cached);
  assert.deepEqual(
    new Uint8Array(await cached.arrayBuffer()),
    original.contents['asr/v3_ctc.onnx']
  );
  assert.deepEqual(await harness.cacheStorage.keys(), originalCacheNames);
  assert.equal((await getLocalModelStatus(baseUrl)).state, 'ready');
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

  const cached = await getCachedLocalModelFile('asr/v3_ctc.onnx', baseUrl);
  assert.ok(cached);
  assert.deepEqual(
    new Uint8Array(await cached.arrayBuffer()),
    original.contents['asr/v3_ctc.onnx']
  );
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

test('an interrupted first install is discarded instead of remaining stuck downloading', async () => {
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
  assert.equal(await harness.cacheStorage.has(stagingCacheName), false);
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
    totalBytes: original.manifest.totalBytes
  });
  assert.equal(await harness.cacheStorage.has(activePointer.cacheName), true);
  assert.equal(await harness.cacheStorage.has(stagingCacheName), false);
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
