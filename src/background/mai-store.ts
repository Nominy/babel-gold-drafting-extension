export interface MaiStore {
  get<T>(key: string): Promise<T | undefined>;
  set<T>(key: string, value: T): Promise<void>;
}
// Extension-origin IndexedDB avoids the 10MiB session-storage quota for long
// transcripts. No credential or audio bytes are persisted in this result store.
export function createMaiResultStore(): MaiStore {
  let database: Promise<IDBDatabase> | undefined;
  function open(): Promise<IDBDatabase> {
    if (!database) {
      const { promise, resolve, reject } = Promise.withResolvers<IDBDatabase>();
      const request = indexedDB.open('babel-gold-drafting-mai', 1);
      request.onupgradeneeded = () => { request.result.createObjectStore('results'); };
      request.onsuccess = () => { resolve(request.result); };
      request.onerror = () => { reject(new Error('MAI result storage is unavailable.')); };
      database = promise;
      database.catch(() => { database = undefined; });
    }
    return database;
  }
  return {
    async get<T>(key: string): Promise<T | undefined> {
      const db = await open();
      const { promise, resolve, reject } = Promise.withResolvers<T | undefined>();
      const transaction = db.transaction('results', 'readonly');
      const request = transaction.objectStore('results').get(key);
      request.onsuccess = () => { resolve(request.result as T | undefined); };
      request.onerror = () => { reject(new Error('Could not read the MAI native result cache.')); };
      return promise;
    },
    async set<T>(key: string, value: T): Promise<void> {
      const db = await open();
      const { promise, resolve, reject } = Promise.withResolvers<void>();
      const transaction = db.transaction('results', 'readwrite');
      transaction.objectStore('results').put(value, key);
      transaction.oncomplete = () => { resolve(); };
      transaction.onerror = transaction.onabort = () => { reject(new Error('Could not retain the MAI native result cache.')); };
      return promise;
    }
  };
}
