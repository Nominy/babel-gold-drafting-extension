import { createZipDspWorkerHandler } from '../core/audio-enhancement-worker';
import type { ZipDspResponse } from '../core/audio-enhancement-worker-protocol';

// This bundle runs in a dedicated worker; the project otherwise uses DOM globals.
const scope = globalThis as unknown as {
  postMessage(response: ZipDspResponse, transfer: ArrayBuffer[]): void;
  close(): void;
  addEventListener(type: 'message', listener: (event: MessageEvent<unknown>) => void): void;
};

const handle = createZipDspWorkerHandler(
  (response, transfer) => scope.postMessage(response, transfer),
  () => scope.close()
);
scope.addEventListener('message', (event) => handle(event.data));
