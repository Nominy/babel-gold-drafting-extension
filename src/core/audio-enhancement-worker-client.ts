import type { ZipSpectrum } from './audio-enhancement-dsp';
import {
  ownedZipArray, parseZipDspResponse, zipSpectrumTransfers,
  type ZipDspRequest, type ZipDspSuccess, type ZipDspWorkerPort
} from './audio-enhancement-worker-protocol';

export function throwIfZipCancelled(signal?: AbortSignal): void {
  if (signal?.aborted) throw signal.reason instanceof Error ? signal.reason : new DOMException('ZipEnhancer was cancelled.', 'AbortError');
}

export function createPackagedZipDspWorker(): ZipDspWorkerPort {
  const runtime = globalThis.chrome?.runtime;
  if (typeof runtime?.getURL !== 'function' || typeof Worker !== 'function') throw new Error('ZipEnhancer cannot locate its packaged CPU DSP worker.');
  return new Worker(runtime.getURL('dist/workers/audio-enhancement.js'), { name: 'zipenhancer-cpu-dsp' });
}

type PendingRequest = {
  type: ZipDspSuccess['type']; index?: number;
  resolve: (response: ZipDspSuccess) => void; reject: (error: Error) => void;
};

export class ZipDspWorkerClient {
  private readonly pending = new Map<number, PendingRequest>();
  private nextId = 1;
  private acknowledgedId = 0;
  private failure: Error | undefined;
  private closed = false;

  constructor(private readonly worker: ZipDspWorkerPort, private readonly signal?: AbortSignal) {
    worker.addEventListener('message', this.onMessage);
    worker.addEventListener('error', this.onError);
    worker.addEventListener('messageerror', this.onMessageError);
    signal?.addEventListener('abort', this.onAbort, { once: true });
    globalThis.addEventListener?.('pagehide', this.onPageHide, { once: true });
    if (signal?.aborted) this.onAbort();
  }

  private readonly onMessage = (event: Event): void => {
    if (this.closed) return;
    try {
      if (!('data' in event)) throw new Error('ZipEnhancer DSP acknowledgement has no data.');
      const response = parseZipDspResponse(event.data);
      if (!response.ok) {
        const error = new Error(response.error.message); error.name = response.error.name;
        this.close(error); return;
      }
      const pending = this.pending.get(response.id);
      if (!pending || response.id !== this.acknowledgedId + 1 || response.type !== pending.type ||
        pending.index !== undefined && (!('index' in response) || response.index !== pending.index) ||
        response.type === 'reconstructed' && response.completedChunks !== response.index + 1) {
        throw new Error('ZipEnhancer DSP returned an out-of-order acknowledgement.');
      }
      this.pending.delete(response.id);
      this.acknowledgedId = response.id;
      pending.resolve(response);
    } catch (error) { this.close(error instanceof Error ? error : new Error(String(error))); }
  };
  private readonly onError = (event: Event): void => this.close(new Error('message' in event && typeof event.message === 'string' ? event.message : 'ZipEnhancer DSP worker failed.'));
  private readonly onMessageError = (): void => this.close(new Error('ZipEnhancer DSP worker transfer failed.'));
  private readonly onAbort = (): void => {
    try { throwIfZipCancelled(this.signal); } catch (error) { this.close(error instanceof Error ? error : new Error(String(error))); }
  };
  private readonly onPageHide = (): void => this.close(new Error('ZipEnhancer owning document was closed.'));

  private request(request: ZipDspRequest, type: ZipDspSuccess['type'], transfer: ArrayBuffer[] = []): Promise<ZipDspSuccess> {
    if (this.closed) return Promise.reject(this.failure ?? new Error('ZipEnhancer DSP worker is closed.'));
    if (this.pending.size >= 2) {
      const error = new Error('ZipEnhancer DSP exceeded its bounded request queue.');
      this.close(error);
      return Promise.reject(error);
    }
    const { promise, resolve, reject } = Promise.withResolvers<ZipDspSuccess>();
    this.pending.set(request.id, { type, ...('index' in request ? { index: request.index } : {}), resolve, reject });
    try { this.worker.postMessage(request, transfer); }
    catch (error) { this.close(error instanceof Error ? error : new Error(String(error))); }
    return promise;
  }

  async initialize(source: Float32Array, sampleRate: number) {
    const frameCount = source.length, samples = ownedZipArray(source);
    const response = await this.request({ id: this.nextId++, type: 'initialize', samples, sampleRate, frameCount }, 'ready', [samples.buffer]);
    if (response.type !== 'ready') throw new Error('ZipEnhancer DSP did not initialize.');
    return response;
  }
  async nextSpectrum(index: number) {
    const response = await this.request({ id: this.nextId++, type: 'spectrum', index }, 'spectrum');
    if (response.type !== 'spectrum') throw new Error('ZipEnhancer DSP did not prepare a spectrum.');
    return response.spectrum;
  }
  async reconstruct(index: number, enhanced: ZipSpectrum) {
    const spectrum = { magnitude: ownedZipArray(enhanced.magnitude), phase: ownedZipArray(enhanced.phase), frames: enhanced.frames };
    const response = await this.request({ id: this.nextId++, type: 'reconstruct', index, spectrum }, 'reconstructed', zipSpectrumTransfers(spectrum));
    if (response.type !== 'reconstructed') throw new Error('ZipEnhancer DSP did not reconstruct a window.');
    return response.completedChunks;
  }
  async finish() {
    const response = await this.request({ id: this.nextId++, type: 'finish' }, 'finished');
    if (response.type !== 'finished') throw new Error('ZipEnhancer DSP did not finish the source lane.');
    return response;
  }

  close(error = new Error('ZipEnhancer DSP worker is closed.')): void {
    if (this.closed) return;
    this.closed = true; this.failure = error;
    this.worker.removeEventListener('message', this.onMessage);
    this.worker.removeEventListener('error', this.onError);
    this.worker.removeEventListener('messageerror', this.onMessageError);
    this.signal?.removeEventListener('abort', this.onAbort);
    globalThis.removeEventListener?.('pagehide', this.onPageHide);
    this.worker.terminate();
    for (const request of this.pending.values()) request.reject(error);
    this.pending.clear();
  }
}
