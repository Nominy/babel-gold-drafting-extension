import { parentPort } from 'node:worker_threads';
import type { ZipDspResponse } from '../../src/core/audio-enhancement-worker-protocol';

if (!parentPort) throw new Error('The DSP fixture requires a real worker thread.');
const port = parentPort;
Object.defineProperties(globalThis, {
  postMessage: { value: (message: ZipDspResponse, transfer: ArrayBuffer[]) => port.postMessage(message, transfer) },
  close: { value: () => port.close() },
  addEventListener: { value: (type: string, listener: (event: MessageEvent<unknown>) => void) => {
    if (type === 'message') port.on('message', (data: unknown) => listener(new MessageEvent('message', { data })));
  } }
});
