export const zipWebGpuAssets: string[];
export function assertZipWebGpuFloat32Plan(plan: unknown): void;
export interface ZipWebGpuArtifactMetadata {
  id: string;
  sha256: string;
  checkpointSha256: string;
  sourceGraphSha256: string;
  planSha256: string;
  weightsSha256: string;
  weightsByteLength: number;
  planPath: string;
  weightsPath: string;
  precision: 'float32';
  kernelAbi: string;
  sampleRate: number;
  fftSize: number;
  hopSize: number;
  winSize: number;
  compressFactor: number;
  chunkSeconds: number;
  strideSeconds: number;
  inputFrames: number;
  implementation: Record<string, string>;
}
export function prepareZipWebGpuArtifact(root: string): Promise<ZipWebGpuArtifactMetadata>;
