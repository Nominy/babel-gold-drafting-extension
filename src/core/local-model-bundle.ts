import { INFERENCE_RELEASE } from './inference-release';
export const LOCAL_MODEL_CACHE_NAME = 'babel-gold-local-models';

const MANIFEST_SCHEMA = 'babel-browser-model-bundle-v3';
const MANIFEST_TARGET_BYTES = 1_500_000_000;
const POINTER_STORAGE_KEY = 'babel_gold_local_model_bundle_pointer';
const STATUS_STORAGE_KEY = 'babel_gold_local_model_bundle_status';
const POINTER_VERSION = 3;
const ASR_CHECKPOINT_SHA256 = '02cea9973d0e839f6a3eeca101b83a93f93a066c2da2e3ebfa176d57e61d84d3';
const C_DENOISE_CHECKPOINT_SHA256 = '0a01c3535fb66627b13f266bc59ab7b95c2aa85f7413a051d1e17294515ded5a';
const DOWNLOAD_STALE_AFTER_MS = 30 * 60 * 1000;
const DOWNLOAD_CHUNK_BYTES = 16 * 1024 * 1024;
const REQUIRED_FILES: Record<string, true> = {
  'asr/v3_ctc.onnx': true,
  'asr/v3_ctc.yaml': true,
  'punctuation/context.fp16.onnx': true,
  'punctuation/denoise.fp16.onnx': true,
  'punctuation/c-denoise.json': true,
  'punctuation/gpu-placement.json': true,
  'punctuation/config.json': true,
  'punctuation/tokenizer.json': true,
  'punctuation/tokenizer_config.json': true,
  'punctuation/special_tokens_map.json': true,
  'punctuation/vocab.txt': true
};

export type LocalModelProgress = {
  completedBytes: number;
  totalBytes: number;
  currentPath: string;
};

export type LocalModelStatus = {
  state: 'not-installed' | 'downloading' | 'ready' | 'error';
  completedBytes: number;
  totalBytes: number;
  currentPath?: string;
  error?: string;
  tested?: boolean;
  source?: ModelSource;
};

type ModelSource = {
  asrCheckpointSha256: string;
  cDenoiseCheckpointSha256: string;
  baseModelFiles: Record<string, string>;
};

type ManifestFile = {
  path: string;
  bytes: number;
  sha256: string;
};

type ModelManifest = {
  releaseId: string;
  files: ManifestFile[];
  totalBytes: number;
  source: ModelSource;
};

export type CachedBundleDescriptor = ModelManifest & {
  identity: string;
  baseUrl: string;
  tested: boolean;
};

type ActiveBundlePointer = {
  version: typeof POINTER_VERSION;
  releaseId: string;
  cacheName: string;
  baseUrl: string;
  totalBytes: number;
  files: ManifestFile[];
  source: ModelSource;
  webgpuTestedAt?: number;
};

type StoredStatus = LocalModelStatus & {
  baseUrl: string;
  operationId: string;
  startedAt?: number;
  updatedAt?: number;
};

function getStorageArea(): chrome.storage.StorageArea {
  const storage = globalThis.chrome?.storage?.local;
  if (!storage) {
    throw new Error('chrome.storage.local is unavailable');
  }
  return storage;
}

function getCacheStorage(): CacheStorage {
  const storage = globalThis.caches;
  if (!storage) {
    throw new Error('Cache Storage is unavailable');
  }
  return storage;
}

function normalizeBaseUrl(input: string): string {
  if (typeof input !== 'string' || !input.trim()) {
    throw new Error('A local model bundle URL is required');
  }

  let url: URL;
  try {
    url = new URL(input.trim());
  } catch {
    throw new Error('Local model bundle URL must be a valid HTTP or HTTPS URL');
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new Error('Local model bundle URL must use HTTP or HTTPS');
  }

  url.username = '';
  url.password = '';
  url.search = '';
  url.hash = '';
  url.pathname = url.pathname.replace(/\/+$/, '') || '/';
  return url.toString().replace(/\/+$/, '');
}

function assertSafeRelativePath(path: unknown): asserts path is string {
  if (typeof path !== 'string' || !path || path.startsWith('/') || /[\\%:?#\u0000-\u001f\u007f]/.test(path)) {
    throw new Error(`Unsafe local model file path: ${String(path)}`);
  }

  if (path.split('/').some((segment) => !segment || segment === '.' || segment === '..')) {
    throw new Error(`Unsafe local model file path: ${path}`);
  }
}

function fileUrl(baseUrl: string, path: string): string {
  return `${baseUrl}/${path}`;
}

function requireRecord(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
}

function validateSource(value: unknown): ModelSource {
  const source = requireRecord(value, 'Local model source');
  if (source.asrCheckpointSha256 !== ASR_CHECKPOINT_SHA256 ||
      source.cDenoiseCheckpointSha256 !== C_DENOISE_CHECKPOINT_SHA256) {
    throw new Error('Local C-denoise bundle uses unsupported speech or correction checkpoint provenance');
  }
  const baseModelFiles = requireRecord(source.baseModelFiles, 'Local model baseModelFiles');
  if (Object.keys(baseModelFiles).length === 0 ||
      Object.values(baseModelFiles).some((sha) => typeof sha !== 'string' || !/^[a-f0-9]{64}$/.test(sha))) {
    throw new Error('Local C-denoise bundle must identify its base model files by SHA-256');
  }
  return { asrCheckpointSha256: ASR_CHECKPOINT_SHA256, cDenoiseCheckpointSha256: C_DENOISE_CHECKPOINT_SHA256,
    baseModelFiles: baseModelFiles as Record<string, string> };
}

function validateManifest(value: unknown): ModelManifest {
  const manifest = requireRecord(value, 'Local model manifest');
  if (manifest.releaseId !== INFERENCE_RELEASE.id) {
    throw new Error('This model bundle is outdated. Download the current C-denoise release.');
  }
  if (manifest.schema !== MANIFEST_SCHEMA) {
    throw new Error(`Local model manifest schema must be ${MANIFEST_SCHEMA}`);
  }
  if (manifest.targetBytes !== MANIFEST_TARGET_BYTES) {
    throw new Error(`Local model manifest targetBytes must be ${MANIFEST_TARGET_BYTES}`);
  }
  if (manifest.pass !== true) {
    throw new Error('Local model manifest has not passed bundle validation');
  }
  const source = validateSource(manifest.source);
  const validation = requireRecord(manifest.validation, 'Local model validation');
  const numeric = requireRecord(validation.numericExport, 'Local model numeric export validation');
  if (numeric.pass !== true || typeof numeric.reportSha256 !== 'string' ||
      !/^[a-f0-9]{64}$/.test(numeric.reportSha256) ||
      !Array.isArray(numeric.cases) || numeric.cases.length === 0 ||
      !Array.isArray(numeric.asr) || numeric.asr.length === 0 ||
      Object.keys(requireRecord(numeric.limits, 'Numeric export limits')).length === 0) {
    throw new Error('Local C-denoise bundle requires successful numeric export parity evidence');
  }
  for (const result of [...numeric.cases, ...numeric.asr]) {
    const row = requireRecord(result, 'Numeric export case');
    const comparisons = requireRecord(row.comparisons, 'Numeric export comparisons');
    if (row.pass !== true || typeof row.id !== 'string' || !row.id ||
        Object.keys(comparisons).length === 0) {
      throw new Error('Local C-denoise numeric export contains a failed or empty case');
    }
    for (const comparison of Object.values(comparisons)) {
      const metric = requireRecord(comparison, 'Numeric export metric');
      if (metric.pass !== true || metric.finite !== true ||
          typeof metric.maxAbs !== 'number' || !Number.isFinite(metric.maxAbs) || metric.maxAbs < 0 ||
          typeof metric.rmse !== 'number' || !Number.isFinite(metric.rmse) || metric.rmse < 0) {
        throw new Error('Local C-denoise numeric export contains failed or non-finite parity metrics');
      }
    }
  }
  if (!Number.isSafeInteger(manifest.totalBytes) || (manifest.totalBytes as number) <= 0) {
    throw new Error('Local model manifest totalBytes must be a positive safe integer');
  }
  if ((manifest.totalBytes as number) > MANIFEST_TARGET_BYTES) {
    throw new Error('Local model manifest exceeds its declared byte target');
  }
  if (!Array.isArray(manifest.files) || manifest.files.length === 0) {
    throw new Error('Local model manifest files must be a non-empty array');
  }

  const paths = new Set<string>();
  const files: ManifestFile[] = [];
  let declaredTotal = 0;
  for (const rawFile of manifest.files) {
    const file = requireRecord(rawFile, 'Local model manifest file');
    assertSafeRelativePath(file.path);
    if (paths.has(file.path)) {
      throw new Error(`Duplicate local model file path: ${file.path}`);
    }
    if (!Number.isSafeInteger(file.bytes) || (file.bytes as number) <= 0) {
      throw new Error(`Invalid byte size for local model file: ${file.path}`);
    }
    if (typeof file.sha256 !== 'string' || !/^[a-fA-F0-9]{64}$/.test(file.sha256)) {
      throw new Error(`Invalid SHA-256 for local model file: ${file.path}`);
    }

    paths.add(file.path);
    if (file.path === 'punctuation/model.fp16.onnx' ||
        (!Object.hasOwn(REQUIRED_FILES, file.path) && !/^((asr|punctuation)\/)[^/]+\.onnx\.data(?:\.\d+)?$/.test(file.path))) {
      throw new Error(`Unsupported local C-denoise model file: ${file.path}`);
    }
    declaredTotal += file.bytes as number;
    if (!Number.isSafeInteger(declaredTotal)) {
      throw new Error('Local model manifest byte total is not a safe integer');
    }
    files.push({
      path: file.path,
      bytes: file.bytes as number,
      sha256: file.sha256.toLowerCase()
    });
  }

  if (declaredTotal !== manifest.totalBytes) {
    throw new Error('Local model manifest totalBytes does not equal its file byte total');
  }
  for (const requiredPath of Object.keys(REQUIRED_FILES)) {
    if (!paths.has(requiredPath)) {
      throw new Error(`Local model manifest is missing required file: ${requiredPath}`);
    }
  }

  return { releaseId: INFERENCE_RELEASE.id, files, totalBytes: manifest.totalBytes as number, source };
}

function isActiveBundlePointer(value: unknown): value is ActiveBundlePointer {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return false;
  }
  const pointer = value as Partial<ActiveBundlePointer>;
  if (
    pointer.version !== POINTER_VERSION || pointer.releaseId !== INFERENCE_RELEASE.id ||
    typeof pointer.cacheName !== 'string' ||
    !pointer.cacheName.startsWith(`${LOCAL_MODEL_CACHE_NAME}:bundle:`) ||
    typeof pointer.baseUrl !== 'string' ||
    !Number.isSafeInteger(pointer.totalBytes) ||
    !Array.isArray(pointer.files)
  ) {
    return false;
  }

  try {
    validateSource(pointer.source);
    if (pointer.webgpuTestedAt !== undefined &&
        (!Number.isFinite(pointer.webgpuTestedAt) || pointer.webgpuTestedAt <= 0)) return false;
    if (normalizeBaseUrl(pointer.baseUrl) !== pointer.baseUrl) {
      return false;
    }
    let total = 0;
    const paths = new Set<string>();
    for (const file of pointer.files) {
      assertSafeRelativePath(file?.path);
      if (
        paths.has(file.path) ||
        !Number.isSafeInteger(file.bytes) ||
        file.bytes <= 0 ||
        typeof file.sha256 !== 'string' ||
        !/^[a-f0-9]{64}$/.test(file.sha256)
      ) {
        return false;
      }
      paths.add(file.path);
      total += file.bytes;
    }
    return total === pointer.totalBytes && Object.keys(REQUIRED_FILES).every((path) => paths.has(path));
  } catch {
    return false;
  }
}

function isStoredStatus(value: unknown): value is StoredStatus {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return false;
  }
  const status = value as Partial<StoredStatus>;
  return (
    typeof status.baseUrl === 'string' &&
    typeof status.operationId === 'string' &&
    status.operationId.length > 0 &&
    (status.startedAt === undefined ||
      (typeof status.startedAt === 'number' && Number.isFinite(status.startedAt))) &&
    (status.updatedAt === undefined ||
      (typeof status.updatedAt === 'number' && Number.isFinite(status.updatedAt))) &&
    (status.state === 'downloading' || status.state === 'error') &&
    typeof status.completedBytes === 'number' &&
    typeof status.totalBytes === 'number'
  );
}

async function readPointer(): Promise<ActiveBundlePointer | null> {
  const stored = await getStorageArea().get(POINTER_STORAGE_KEY);
  const pointer = stored[POINTER_STORAGE_KEY];
  return isActiveBundlePointer(pointer) ? pointer : null;
}

// Chrome storage may reorder nested object keys. Identity and cache admission
// compare semantic metadata, not insertion order from either serializer.
function bundleMetadataJson(value: unknown): string {
  return JSON.stringify(value, (_key, item: unknown) =>
    item && typeof item === 'object' && !Array.isArray(item)
      ? Object.fromEntries(Object.keys(item).sort().map((key) => [key, (item as Record<string, unknown>)[key]]))
      : item);
}

async function cacheIsComplete(pointer: ActiveBundlePointer): Promise<boolean> {
  const cacheStorage = getCacheStorage();
  if (!(await cacheStorage.has(pointer.cacheName))) {
    return false;
  }
  const cache = await cacheStorage.open(pointer.cacheName);
  const manifestResponse = await cache.match(fileUrl(pointer.baseUrl, 'manifest.json'));
  if (!manifestResponse?.ok) return false;
  try {
    const manifest = validateManifest(await manifestResponse.json());
    if (manifest.totalBytes !== pointer.totalBytes ||
        bundleMetadataJson(manifest.source) !== bundleMetadataJson(pointer.source) ||
        bundleMetadataJson(manifest.files) !== bundleMetadataJson(pointer.files)) return false;
  } catch {
    return false;
  }
  for (const file of pointer.files) {
    if (!(await cache.match(fileUrl(pointer.baseUrl, file.path)))) {
      return false;
    }
  }
  return true;
}

async function cacheHasWebGpuTest(pointer: ActiveBundlePointer): Promise<boolean> {
  const cache = await getCacheStorage().open(pointer.cacheName);
  const response = await cache.match(fileUrl(pointer.baseUrl, '__bundle-pointer.json'));
  if (!response?.ok) return false;
  try {
    const metadata: unknown = await response.json();
    return isActiveBundlePointer(metadata) && typeof metadata.webgpuTestedAt === 'number' &&
      metadata.cacheName === pointer.cacheName && metadata.baseUrl === pointer.baseUrl &&
      metadata.totalBytes === pointer.totalBytes &&
      bundleMetadataJson(metadata.files) === bundleMetadataJson(pointer.files) &&
      bundleMetadataJson(metadata.source) === bundleMetadataJson(pointer.source);
  } catch {
    return false;
  }
}

async function findCachedPointer(baseUrl: string): Promise<ActiveBundlePointer | null> {
  const normalizedBaseUrl = normalizeBaseUrl(baseUrl);
  const cacheStorage = getCacheStorage();
  let found: ActiveBundlePointer | null = null;
  for (const cacheName of await cacheStorage.keys()) {
    if (!cacheName.startsWith(`${LOCAL_MODEL_CACHE_NAME}:bundle:`)) continue;
    const cache = await cacheStorage.open(cacheName);
    const response = await cache.match(fileUrl(normalizedBaseUrl, '__bundle-pointer.json'));
    if (!response?.ok) continue;
    try {
      const pointer: unknown = await response.json();
      if (!isActiveBundlePointer(pointer) || pointer.cacheName !== cacheName ||
          pointer.baseUrl !== normalizedBaseUrl || !(await cacheIsComplete(pointer))) continue;
      if (found) return null;
      found = pointer;
    } catch {
      continue;
    }
  }
  return found;
}

export async function getCachedBundleDescriptor(baseUrl: string): Promise<CachedBundleDescriptor | null> {
  const normalizedBaseUrl = normalizeBaseUrl(baseUrl);
  const pointer = globalThis.chrome?.storage?.local
    ? await readPointer()
    : await findCachedPointer(normalizedBaseUrl);
  if (!pointer || pointer.baseUrl !== normalizedBaseUrl || !(await cacheIsComplete(pointer))) return null;
  if (globalThis.chrome?.storage?.local && (await getLocalModelStatus(baseUrl)).state !== 'ready') return null;
  const files = pointer.files.toSorted((a, b) => a.path.localeCompare(b.path));
  const identity = await sha256Hex(new TextEncoder().encode(bundleMetadataJson({
    baseUrl: normalizedBaseUrl, source: pointer.source, files
  })).buffer);
  return { identity, releaseId: pointer.releaseId, baseUrl: normalizedBaseUrl, files, source: pointer.source,
    totalBytes: pointer.totalBytes, tested: await cacheHasWebGpuTest(pointer) };
}

function makeOperationId(): string {
  if (typeof globalThis.crypto?.randomUUID === 'function') {
    return globalThis.crypto.randomUUID();
  }
  const random = Math.random().toString(36).slice(2);
  return `${Date.now().toString(36)}-${random}`;
}

async function writeStatus(status: StoredStatus): Promise<void> {
  await getStorageArea().set({ [STATUS_STORAGE_KEY]: status });
}

async function clearStatus(operationId?: string): Promise<void> {
  const storage = getStorageArea();
  if (operationId) {
    const stored = await storage.get(STATUS_STORAGE_KEY);
    const status = stored[STATUS_STORAGE_KEY];
    if (!isStoredStatus(status) || status.operationId !== operationId) {
      return;
    }
  }
  await storage.remove(STATUS_STORAGE_KEY);
}

async function clearStaleDownload(status: StoredStatus): Promise<void> {
  // Keep verified files for an explicitly requested retry; never activate a partial cache.
  await clearStatus(status.operationId);
}

async function sha256Hex(bytes: ArrayBuffer): Promise<string> {
  if (!globalThis.crypto?.subtle) {
    throw new Error('Web Crypto SHA-256 is unavailable');
  }
  const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
}

async function downloadModelFile(
  url: string,
  file: ManifestFile,
  onChunk: (downloadedBytes: number) => Promise<void>
): Promise<{ bytes: ArrayBuffer; headers: Headers }> {
  if (file.bytes <= DOWNLOAD_CHUNK_BYTES) {
    const response = await globalThis.fetch(url, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`Failed to download local model file ${file.path}: HTTP ${response.status}`);
    }
    return { bytes: await response.arrayBuffer(), headers: new Headers(response.headers) };
  }

  const bytes = new ArrayBuffer(file.bytes);
  const target = new Uint8Array(bytes);
  let headers = new Headers();
  for (let offset = 0; offset < file.bytes; offset += DOWNLOAD_CHUNK_BYTES) {
    const end = Math.min(offset + DOWNLOAD_CHUNK_BYTES, file.bytes) - 1;
    const response = await globalThis.fetch(url, {
      cache: 'no-store',
      headers: { Range: `bytes=${offset}-${end}` }
    });
    if (offset === 0 && response.status === 200) {
      return { bytes: await response.arrayBuffer(), headers: new Headers(response.headers) };
    }
    if (response.status !== 206) {
      throw new Error(`Failed to download local model file ${file.path}: HTTP ${response.status}`);
    }
    if (response.headers.get('content-range') !== `bytes ${offset}-${end}/${file.bytes}`) {
      throw new Error(`Local model file ${file.path} returned an invalid byte range`);
    }
    const chunk = new Uint8Array(await response.arrayBuffer());
    if (chunk.byteLength !== end - offset + 1) {
      throw new Error(
        `Local model file ${file.path} has size ${chunk.byteLength}; expected ${end - offset + 1}`
      );
    }
    target.set(chunk, offset);
    if (offset === 0) {
      headers = new Headers(response.headers);
    }
    if (end + 1 < file.bytes) {
      await onChunk(end + 1);
    }
  }
  return { bytes, headers };
}

export async function getLocalModelStatus(baseUrl: string): Promise<LocalModelStatus> {
  let normalizedBaseUrl: string;
  try {
    normalizedBaseUrl = normalizeBaseUrl(baseUrl);
  } catch (error) {
    return {
      state: 'error',
      completedBytes: 0,
      totalBytes: 0,
      error: error instanceof Error ? error.message : String(error)
    };
  }

  const storage = getStorageArea();
  const stored = await storage.get([POINTER_STORAGE_KEY, STATUS_STORAGE_KEY]);
  const pendingStatus = stored[STATUS_STORAGE_KEY];
  const storedPointer = stored[POINTER_STORAGE_KEY];
  const pointer = isActiveBundlePointer(storedPointer) ? storedPointer : null;
  let staleDownloadCleared = false;
  if (
    isStoredStatus(pendingStatus) &&
    pendingStatus.baseUrl === normalizedBaseUrl &&
    pendingStatus.state === 'downloading'
  ) {
    const lastUpdateAt = pendingStatus.updatedAt ?? pendingStatus.startedAt;
    const downloadAge =
      typeof lastUpdateAt === 'number' ? Date.now() - lastUpdateAt : Number.POSITIVE_INFINITY;
    const isLive = downloadAge >= 0 && downloadAge <= DOWNLOAD_STALE_AFTER_MS;
    if (isLive) {
      const { state, completedBytes, totalBytes, currentPath, error } = pendingStatus;
      return { state, completedBytes, totalBytes, currentPath, error };
    }
    await clearStaleDownload(pendingStatus);
    staleDownloadCleared = true;
  }

  if (!staleDownloadCleared && isStoredStatus(pendingStatus) &&
      pendingStatus.baseUrl === normalizedBaseUrl && pendingStatus.state === 'error') {
    const { state, completedBytes, totalBytes, currentPath, error } = pendingStatus;
    return { state, completedBytes, totalBytes, currentPath, error };
  }
  if (pointer?.baseUrl === normalizedBaseUrl) {
    if (await cacheIsComplete(pointer)) {
      return {
        state: 'ready',
        completedBytes: pointer.totalBytes,
        totalBytes: pointer.totalBytes,
        tested: await cacheHasWebGpuTest(pointer),
        source: pointer.source
      };
    }
    return {
      state: 'error',
      completedBytes: 0,
      totalBytes: pointer.totalBytes,
      error: 'Installed local model cache is incomplete'
    };
  }

  if (
    !staleDownloadCleared &&
    isStoredStatus(pendingStatus) &&
    pendingStatus.baseUrl === normalizedBaseUrl
  ) {
    const { state, completedBytes, totalBytes, currentPath, error } = pendingStatus;
    return { state, completedBytes, totalBytes, currentPath, error };
  }
  if (storedPointer && typeof storedPointer === 'object' &&
      'version' in storedPointer && storedPointer.version !== POINTER_VERSION) {
    return { state: 'error', completedBytes: 0, totalBytes: 0,
      error: 'The cached legacy model bundle is not C-denoise v3. Click Download to install the verified new bundle; old files remain until installation succeeds or you remove them.' };
  }

  return { state: 'not-installed', completedBytes: 0, totalBytes: 0 };
}

export async function markLocalModelWebGpuTested(baseUrl: string, expectedIdentity: string): Promise<void> {
  const pointer = await readPointer();
  const descriptor = await getCachedBundleDescriptor(baseUrl);
  if (!pointer || !descriptor || descriptor.identity !== expectedIdentity) {
    throw new Error('The verified C-denoise bundle changed during its test. Test the current bundle again.');
  }
  const testedPointer = { ...pointer, webgpuTestedAt: Date.now() };
  const cache = await getCacheStorage().open(pointer.cacheName);
  await cache.put(fileUrl(pointer.baseUrl, '__bundle-pointer.json'),
    new Response(JSON.stringify(testedPointer), { headers: { 'content-type': 'application/json' } }));
}

export async function setupLocalModels(
  baseUrl: string,
  onProgress?: (progress: LocalModelProgress) => void
): Promise<LocalModelStatus> {
  const normalizedBaseUrl = normalizeBaseUrl(baseUrl);
  const cacheStorage = getCacheStorage();
  const storage = getStorageArea();
  const operationId = makeOperationId();
  const startedAt = Date.now();
  const cacheName = `${LOCAL_MODEL_CACHE_NAME}:bundle:${operationId}`;
  let completedBytes = 0;
  let totalBytes = 0;
  let currentPath = 'manifest.json';

  try {
    await writeStatus({
      baseUrl: normalizedBaseUrl,
      operationId,
      startedAt,
      updatedAt: startedAt,
      state: 'downloading',
      completedBytes,
      totalBytes,
      currentPath
    });

    const manifestResponse = await globalThis.fetch(`${normalizedBaseUrl}/manifest.json`, {
      cache: 'no-store'
    });
    if (!manifestResponse.ok) {
      throw new Error(`Failed to download local model manifest: HTTP ${manifestResponse.status}`);
    }
    const manifestDocument: unknown = await manifestResponse.json();
    const manifest = validateManifest(manifestDocument);
    totalBytes = manifest.totalBytes;
    const stageCache = await cacheStorage.open(cacheName);
    await stageCache.put(fileUrl(normalizedBaseUrl, 'manifest.json'), new Response(JSON.stringify(manifestDocument), {
      status: 200,
      headers: { 'content-type': 'application/json' }
    }));
    const previousCacheNames = (await cacheStorage.keys()).filter((name) =>
      name !== cacheName && (name === LOCAL_MODEL_CACHE_NAME || name.startsWith(`${LOCAL_MODEL_CACHE_NAME}:`)));

    for (const file of manifest.files) {
      currentPath = file.path;
      const beforeProgress = { completedBytes, totalBytes, currentPath };
      onProgress?.(beforeProgress);
      await writeStatus({
        baseUrl: normalizedBaseUrl,
        operationId,
        startedAt,
        updatedAt: Date.now(),
        state: 'downloading',
        ...beforeProgress
      });
      let resumed = false;
      for (const previousCacheName of previousCacheNames) {
        const previousCache = await cacheStorage.open(previousCacheName);
        const previousUrl = (await previousCache.keys()).find(request => new URL(request.url).pathname.endsWith(`/${file.path}`));
        const response = await previousCache.match(fileUrl(normalizedBaseUrl, file.path)) ??
          (previousUrl ? await previousCache.match(previousUrl) : undefined);
        if (!response?.ok) continue;
        const bytes = await response.clone().arrayBuffer();
        if (bytes.byteLength !== file.bytes || await sha256Hex(bytes) !== file.sha256) continue;
        await stageCache.put(fileUrl(normalizedBaseUrl, file.path), response);
        completedBytes += file.bytes;
        onProgress?.({ completedBytes, totalBytes, currentPath });
        resumed = true;
        break;
      }
      if (resumed) continue;

      const { bytes, headers } = await downloadModelFile(
        fileUrl(normalizedBaseUrl, file.path),
        file,
        async (downloadedBytes) => {
          const progress = { completedBytes: completedBytes + downloadedBytes, totalBytes, currentPath };
          onProgress?.(progress);
          await writeStatus({
            baseUrl: normalizedBaseUrl,
            operationId,
            startedAt,
            updatedAt: Date.now(),
            state: 'downloading',
            ...progress
          });
        }
      );
      if (bytes.byteLength !== file.bytes) {
        throw new Error(
          `Local model file ${file.path} has size ${bytes.byteLength}; expected ${file.bytes}`
        );
      }
      const actualSha256 = await sha256Hex(bytes);
      if (actualSha256 !== file.sha256) {
        throw new Error(`Local model file ${file.path} failed SHA-256 verification`);
      }

      headers.delete('content-range');
      headers.delete('content-length');
      // A single 850 MB Response chunk can exceed Chromium's cache transport
      // limits. Preserve the verified bytes while feeding CacheStorage in the
      // same bounded chunks used for downloads.
      let offset = 0;
      const body = new ReadableStream<Uint8Array>({
        pull(controller) {
          if (offset === bytes.byteLength) { controller.close(); return; }
          const end = Math.min(offset + DOWNLOAD_CHUNK_BYTES, bytes.byteLength);
          controller.enqueue(new Uint8Array(bytes, offset, end - offset));
          offset = end;
        }
      });
      await stageCache.put(
        fileUrl(normalizedBaseUrl, file.path),
        new Response(body, { status: 200, headers })
      );
      completedBytes += file.bytes;
      const afterProgress = { completedBytes, totalBytes, currentPath };
      onProgress?.(afterProgress);
      await writeStatus({
        baseUrl: normalizedBaseUrl,
        operationId,
        startedAt,
        updatedAt: Date.now(),
        state: 'downloading',
        ...afterProgress
      });
    }

    if (completedBytes !== totalBytes) {
      throw new Error('Downloaded local model byte total does not match the manifest');
    }

    const pointer: ActiveBundlePointer = {
      version: POINTER_VERSION,
      releaseId: INFERENCE_RELEASE.id,
      cacheName,
      baseUrl: normalizedBaseUrl,
      totalBytes,
      files: manifest.files,
      source: manifest.source
    };
    await stageCache.put(fileUrl(normalizedBaseUrl, '__bundle-pointer.json'),
      new Response(JSON.stringify(pointer), { headers: { 'content-type': 'application/json' } }));
    await storage.set({ [POINTER_STORAGE_KEY]: pointer });
    await clearStatus(operationId).catch(() => undefined);

    await Promise.all(previousCacheNames.map((name) => cacheStorage.delete(name).catch(() => false)));

    return { state: 'ready', completedBytes, totalBytes, tested: false, source: manifest.source };
  } catch (error) {
    // Retain only checksum-verified whole files for a later explicit resume.
    const message = error instanceof Error ? error.message : String(error);
    await writeStatus({
      baseUrl: normalizedBaseUrl,
      operationId,
      startedAt,
      updatedAt: Date.now(),
      state: 'error',
      completedBytes,
      totalBytes,
      currentPath,
      error: message
    }).catch(() => undefined);
    throw error;
  }
}

export async function removeLocalModels(): Promise<void> {
  const storage = getStorageArea();
  await storage.remove([POINTER_STORAGE_KEY, STATUS_STORAGE_KEY]);
  const cacheStorage = getCacheStorage();
  const cacheNames = await cacheStorage.keys();
  await Promise.all(
    cacheNames
      .filter(
        (cacheName) =>
          cacheName === LOCAL_MODEL_CACHE_NAME ||
          cacheName.startsWith(`${LOCAL_MODEL_CACHE_NAME}:`)
      )
      .map((cacheName) => cacheStorage.delete(cacheName))
  );
}

export async function getCachedLocalModelFile(
  path: string,
  baseUrl: string
): Promise<Response | null> {
  try {
    assertSafeRelativePath(path);
  } catch {
    return null;
  }

  const pointer = globalThis.chrome?.storage?.local
    ? await readPointer()
    : await findCachedPointer(baseUrl);
  if (!pointer || pointer.baseUrl !== normalizeBaseUrl(baseUrl) ||
      !pointer.files.some((file) => file.path === path) || !(await cacheIsComplete(pointer))) return null;
  if (globalThis.chrome?.storage?.local && (await getLocalModelStatus(baseUrl)).state !== 'ready') return null;
  const cache = await getCacheStorage().open(pointer.cacheName);
  return (await cache.match(fileUrl(pointer.baseUrl, path))) ?? null;
}
