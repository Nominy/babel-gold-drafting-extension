export const LOCAL_MODEL_CACHE_NAME = 'babel-gold-local-models';

const MANIFEST_SCHEMA = 'babel-browser-model-bundle-v1';
const MANIFEST_TARGET_BYTES = 500_000_000;
const POINTER_STORAGE_KEY = 'babel_gold_local_model_bundle_pointer';
const STATUS_STORAGE_KEY = 'babel_gold_local_model_bundle_status';
const POINTER_VERSION = 1;
const DOWNLOAD_STALE_AFTER_MS = 30 * 60 * 1000;
const REQUIRED_FILES: Record<string, true> = {
  'asr/v3_ctc.onnx': true,
  'asr/v3_ctc.yaml': true,
  'punctuation/model.int8.onnx': true,
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
};

type ManifestFile = {
  path: string;
  bytes: number;
  sha256: string;
};

type ModelManifest = {
  files: ManifestFile[];
  totalBytes: number;
};

type ActiveBundlePointer = {
  version: typeof POINTER_VERSION;
  cacheName: string;
  baseUrl: string;
  totalBytes: number;
  files: ManifestFile[];
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
  if (typeof path !== 'string' || !path || path.startsWith('/') || /[\\?#\u0000-\u001f\u007f]/.test(path)) {
    throw new Error(`Unsafe local model file path: ${String(path)}`);
  }

  let decoded = path;
  for (let pass = 0; pass < 3; pass += 1) {
    let next: string;
    try {
      next = decodeURIComponent(decoded);
    } catch {
      throw new Error(`Unsafe local model file path: ${path}`);
    }
    if (next === decoded) {
      break;
    }
    decoded = next;
  }

  if (
    decoded.startsWith('/') ||
    decoded.includes('\\') ||
    decoded.includes('?') ||
    decoded.includes('#') ||
    decoded.split('/').some((segment) => !segment || segment === '.' || segment === '..')
  ) {
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

function validateManifest(value: unknown): ModelManifest {
  const manifest = requireRecord(value, 'Local model manifest');
  if (manifest.schema !== MANIFEST_SCHEMA) {
    throw new Error(`Local model manifest schema must be ${MANIFEST_SCHEMA}`);
  }
  if (manifest.targetBytes !== MANIFEST_TARGET_BYTES) {
    throw new Error(`Local model manifest targetBytes must be ${MANIFEST_TARGET_BYTES}`);
  }
  if (manifest.pass !== true) {
    throw new Error('Local model manifest has not passed bundle validation');
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
    if (!Number.isSafeInteger(file.bytes) || (file.bytes as number) < 0) {
      throw new Error(`Invalid byte size for local model file: ${file.path}`);
    }
    if (typeof file.sha256 !== 'string' || !/^[a-fA-F0-9]{64}$/.test(file.sha256)) {
      throw new Error(`Invalid SHA-256 for local model file: ${file.path}`);
    }

    paths.add(file.path);
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

  return { files, totalBytes: manifest.totalBytes as number };
}

function isActiveBundlePointer(value: unknown): value is ActiveBundlePointer {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return false;
  }
  const pointer = value as Partial<ActiveBundlePointer>;
  if (
    pointer.version !== POINTER_VERSION ||
    typeof pointer.cacheName !== 'string' ||
    !pointer.cacheName.startsWith(`${LOCAL_MODEL_CACHE_NAME}:bundle:`) ||
    typeof pointer.baseUrl !== 'string' ||
    !Number.isSafeInteger(pointer.totalBytes) ||
    !Array.isArray(pointer.files)
  ) {
    return false;
  }

  try {
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
        file.bytes < 0 ||
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

async function cacheIsComplete(pointer: ActiveBundlePointer): Promise<boolean> {
  const cacheStorage = getCacheStorage();
  if (!(await cacheStorage.has(pointer.cacheName))) {
    return false;
  }
  const cache = await cacheStorage.open(pointer.cacheName);
  for (const file of pointer.files) {
    if (!(await cache.match(fileUrl(pointer.baseUrl, file.path)))) {
      return false;
    }
  }
  return true;
}

async function findCachedBundleFileWithoutPointer(
  path: string,
  baseUrl: string
): Promise<Response | null> {
  const normalizedBaseUrl = normalizeBaseUrl(baseUrl);
  const cacheStorage = getCacheStorage();
  const cacheNames = await cacheStorage.keys();
  let matchingResponse: Response | null = null;
  let matchingCaches = 0;

  for (const cacheName of cacheNames) {
    if (!cacheName.startsWith(`${LOCAL_MODEL_CACHE_NAME}:bundle:`)) {
      continue;
    }

    const cache = await cacheStorage.open(cacheName);
    const manifestResponse = await cache.match(fileUrl(normalizedBaseUrl, 'manifest.json'));
    let listedPaths: string[];
    if (manifestResponse) {
      if (!manifestResponse.ok) return null;
      try {
        listedPaths = validateManifest(await manifestResponse.json()).files.map((file) => file.path);
      } catch {
        return null;
      }
    } else {
      // Bundles installed before the offscreen cache marker was introduced
      // still contain every verified runtime file. Keep those installations
      // usable without forcing another ~479 MB download.
      listedPaths = Object.keys(REQUIRED_FILES);
    }
    if (!listedPaths.includes(path)) {
      continue;
    }
    let complete = true;
    for (const listedPath of listedPaths) {
      if (!(await cache.match(fileUrl(normalizedBaseUrl, listedPath)))) {
        complete = false;
        break;
      }
    }
    if (!complete) {
      continue;
    }
    const response = (await cache.match(fileUrl(normalizedBaseUrl, path))) ?? null;
    if (!response) {
      continue;
    }
    matchingCaches += 1;
    if (matchingCaches > 1) {
      return null;
    }
    matchingResponse = response;
  }

  return matchingCaches === 1 ? matchingResponse : null;
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

async function clearStaleDownload(
  status: StoredStatus,
  pointer: ActiveBundlePointer | null
): Promise<void> {
  const stagingCacheName = `${LOCAL_MODEL_CACHE_NAME}:bundle:${status.operationId}`;
  if (pointer?.cacheName !== stagingCacheName) {
    await getCacheStorage().delete(stagingCacheName).catch(() => false);
  }
  await clearStatus(status.operationId);
}

async function sha256Hex(bytes: ArrayBuffer): Promise<string> {
  if (!globalThis.crypto?.subtle) {
    throw new Error('Web Crypto SHA-256 is unavailable');
  }
  const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
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
    await clearStaleDownload(pendingStatus, pointer);
    staleDownloadCleared = true;
  }

  if (pointer?.baseUrl === normalizedBaseUrl) {
    if (await cacheIsComplete(pointer)) {
      return {
        state: 'ready',
        completedBytes: pointer.totalBytes,
        totalBytes: pointer.totalBytes
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

  return { state: 'not-installed', completedBytes: 0, totalBytes: 0 };
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
  const previousPointer = await readPointer();
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
    const cachedManifestResponse = new Response(JSON.stringify(manifestDocument), {
      status: 200,
      headers: { 'content-type': 'application/json' }
    });
    totalBytes = manifest.totalBytes;
    const stageCache = await cacheStorage.open(cacheName);

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

      const response = await globalThis.fetch(fileUrl(normalizedBaseUrl, file.path), {
        cache: 'no-store'
      });
      if (!response.ok) {
        throw new Error(`Failed to download local model file ${file.path}: HTTP ${response.status}`);
      }
      const bytes = await response.arrayBuffer();
      if (bytes.byteLength !== file.bytes) {
        throw new Error(
          `Local model file ${file.path} has size ${bytes.byteLength}; expected ${file.bytes}`
        );
      }
      const actualSha256 = await sha256Hex(bytes);
      if (actualSha256 !== file.sha256) {
        throw new Error(`Local model file ${file.path} failed SHA-256 verification`);
      }

      const headers = new Headers(response.headers);
      headers.delete('content-range');
      await stageCache.put(
        fileUrl(normalizedBaseUrl, file.path),
        new Response(bytes, { status: 200, headers })
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
    await stageCache.put(
      fileUrl(normalizedBaseUrl, 'manifest.json'),
      cachedManifestResponse
    );

    const pointer: ActiveBundlePointer = {
      version: POINTER_VERSION,
      cacheName,
      baseUrl: normalizedBaseUrl,
      totalBytes,
      files: manifest.files
    };
    await storage.set({ [POINTER_STORAGE_KEY]: pointer });
    await clearStatus(operationId).catch(() => undefined);

    if (previousPointer && previousPointer.cacheName !== cacheName) {
      await cacheStorage.delete(previousPointer.cacheName).catch(() => false);
    }

    return { state: 'ready', completedBytes, totalBytes };
  } catch (error) {
    await cacheStorage.delete(cacheName).catch(() => false);
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

  if (!globalThis.chrome?.storage?.local) {
    try {
      return await findCachedBundleFileWithoutPointer(path, baseUrl);
    } catch {
      return null;
    }
  }

  const pointer = await readPointer();
  if (!pointer || !pointer.files.some((file) => file.path === path)) {
    return null;
  }
  if (!(await cacheIsComplete(pointer))) {
    return null;
  }

  const cache = await getCacheStorage().open(pointer.cacheName);
  return (await cache.match(fileUrl(pointer.baseUrl, path))) ?? null;
}
