// ORT's device and profiling observer are shared by every local neural model.
// Hold this admission through dispatch readback, audit and observer restoration.
let inferenceTail: Promise<void> = Promise.resolve();

export function runExclusiveGpuInference<T>(action: () => Promise<T>): Promise<T> {
  const result = inferenceTail.then(action);
  inferenceTail = result.then(() => undefined, () => undefined);
  return result;
}
