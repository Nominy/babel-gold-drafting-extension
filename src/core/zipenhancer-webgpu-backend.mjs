import { WebGpuBackend as JsepWebGpuBackend } from '../../node_modules/onnxruntime-web/lib/wasm/jsep/backend-webgpu.ts';
import { WEBGPU_OP_RESOLVE_RULES } from '../../node_modules/onnxruntime-web/lib/wasm/jsep/webgpu/op-resolve-rules.ts';
import { tensorDataTypeEnumToString } from '../../node_modules/onnxruntime-web/lib/wasm/wasm-common.ts';
export { createAttributeWithCacheKey } from '../../node_modules/onnxruntime-web/lib/wasm/jsep/webgpu/attribute-with-cache-key.ts';

export function registerOperator(name, entry) {
  const previous = WEBGPU_OP_RESOLVE_RULES.get(name);
  if (previous && previous[0] !== entry) throw new Error(`Conflicting WebGPU operator: ${name}`);
  WEBGPU_OP_RESOLVE_RULES.set(name, [entry]);
}

// The source imports above MUST pass through scripts/ort-jsep-options.mjs's pinned transforms.
// No WASM session or upstream global environment is initialized by this bridge.
export class WebGpuBackend extends JsepWebGpuBackend {
  profileTasks = new Set();
  profileFailure;
  onFailure;
  allocatedBufferBytes = 0;
  peakAllocatedBufferBytes = 0;

  async initialize(env, adapter) {
    await super.initialize(env, adapter);
    const createBuffer = this.device.createBuffer.bind(this.device);
    const track = (buffer) => {
      this.allocatedBufferBytes += buffer.size;
      this.peakAllocatedBufferBytes = Math.max(this.peakAllocatedBufferBytes, this.allocatedBufferBytes);
      const destroy = buffer.destroy.bind(buffer);
      let destroyed = false;
      buffer.destroy = () => {
        if (!destroyed) {
          destroyed = true;
          this.allocatedBufferBytes -= buffer.size;
        }
        destroy();
      };
      return buffer;
    };
    if (this.queryResolveBuffer) track(this.queryResolveBuffer);
    this.device.createBuffer = (descriptor) => track(createBuffer(descriptor));
  }

  createKernel(type, id, attributes, name) {
    super.createKernel(type, id, attributes, name);
    // Upstream parses outside computeKernel's finally. Parse now, before entering a kernel.
    const kernel = this.kernels.get(id);
    if (kernel.attributes[0]) {
      kernel.attributes = [undefined, kernel.attributes[0](kernel.attributes[1])];
    }
  }

  // The pinned backend's flush starts an unobservable mapAsync and leaks its read buffers.
  // Keep exactly its real dispatch/query metadata, but own the completion and failure barrier.
  flush() {
    if (!this.commandEncoder) return;
    this.endComputePass();
    const count = this.pendingDispatchNumber;
    const pending = this.pendingKernels;
    let readBuffer;
    if (this.queryType !== 'none' && count > 0) {
      if (pending.length !== count) throw new Error('WebGPU dispatch/query count mismatch');
      this.commandEncoder.resolveQuerySet(this.querySet, 0, count * 2, this.queryResolveBuffer, 0);
      readBuffer = this.device.createBuffer({ size: count * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      this.commandEncoder.copyBufferToBuffer(this.queryResolveBuffer, 0, readBuffer, 0, count * 16);
    }
    this.device.queue.submit([this.commandEncoder.finish()]);
    this.gpuDataManager.refreshPendingBuffers();
    this.commandEncoder = null;
    this.pendingDispatchNumber = 0;
    this.pendingKernels = [];
    if (!readBuffer) return;
    const task = readBuffer.mapAsync(GPUMapMode.READ).then(() => {
      const times = new BigUint64Array(readBuffer.getMappedRange());
      for (let i = 0; i < count; i++) {
        const dispatch = pending[i];
        const kernel = this.kernels.get(dispatch.kernelId);
        if (!kernel) throw new Error('WebGPU kernel released before its timestamp callback');
        this.queryTimeBase ??= times[0];
        const startTime = Number(times[i * 2] - this.queryTimeBase);
        const endTime = Number(times[i * 2 + 1] - this.queryTimeBase);
        if (!Number.isSafeInteger(startTime) || !Number.isSafeInteger(endTime)) throw new Error('Invalid GPU timestamp range');
        const metadata = (views) => views.map((view) => ({ dims: view.dims, dataType: tensorDataTypeEnumToString(view.dataType) }));
        this.env.webgpu.profiling.ondata({
          version: 1,
          kernelId: dispatch.kernelId,
          kernelName: kernel.kernelName,
          kernelType: kernel.kernelType,
          programName: dispatch.programName,
          startTime,
          endTime,
          inputsMetadata: metadata(dispatch.inputTensorViews),
          outputsMetadata: metadata(dispatch.outputTensorViews),
        });
      }
    }).catch((error) => {
      this.profileFailure ??= error instanceof Error ? error : new Error(String(error));
      this.onFailure?.(this.profileFailure);
    }).finally(() => {
      readBuffer.destroy();
      this.profileTasks.delete(task);
    });
    this.profileTasks.add(task);
  }

  async drainProfiles() {
    while (this.profileTasks.size) await Promise.all(this.profileTasks);
    if (this.profileFailure) throw this.profileFailure;
  }

  dispose() {
    if (this.profileTasks.size) throw new Error('Drain WebGPU timestamp callbacks before disposal');
    this.queryResolveBuffer?.destroy();
    super.dispose();
    this.programManager?.repo?.clear();
  }
}
