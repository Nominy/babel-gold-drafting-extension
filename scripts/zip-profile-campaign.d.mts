import type { PlacementGraph, GraphPlacementDiagnostic, GpuDispatch } from '../src/core/local-gpu-placement';
export interface CampaignUniform { type: number; data: number | readonly number[] }
export interface CampaignIdentity { modelSha256: string; audioSha256: string[]; baseline: boolean; ortSourceHashes: Record<string, string>; kernelHelperHashes: Record<string, string> }
export interface CampaignTensor { dims: number[]; dataType: number; elements: number; logicalBytes: number; bufferBytes: number; file?: string; sha256?: string;
  range?: { min: number | null; max: number | null; finite: number; nonfinite: number; sample: (number | string)[] } }
export interface CampaignConfiguration { id: number; program: string; cacheKey: string; shader: string; baselineShader: string; inputs: CampaignTensor[]; outputs: CampaignTensor[];
  uniforms: readonly CampaignUniform[]; uniformBytes: number; uniformFile?: string; uniformSha256?: string; dispatchGroup: [number, number, number]; count: number; gpuDurationNs: number;
  sources: Record<string, number>; dataOrigin: string; captured: boolean; captureComplete: boolean; shaderSha256?: string; baselineShaderSha256?: string;
  introducedPhase: 'initialization' | 'inference'; introducingSpectrumId: number | null }
export interface CampaignSpectrum { id: number; frames: number; mag: { file: string; sha256: string }; pha: { file: string; sha256: string } }
export interface CampaignWave { index: number; sampleRate: number; frames: number; wavSha256: string; wallMs: number }
export interface CampaignOccurrence extends GpuDispatch { configId: number; phase: 'initialization' | 'inference'; run: number | null; [field: string]: unknown }
export interface CaptureCampaign { schema: 'babel-zip-kernel-capture-v1'; identity: CampaignIdentity; sourcePaths: string[]; configurations: CampaignConfiguration[];
  spectra: CampaignSpectrum[]; occurrences: CampaignOccurrence[]; audits: GraphPlacementDiagnostic[]; waves: CampaignWave[]; modelInputsComplete: boolean;
  stage: string; missing: { id: number; program: string; inputs: CampaignTensor[]; uniforms: readonly CampaignUniform[]; introducedPhase: string; introducingSpectrumId: number | null; count: number }[] }
export function hashCampaignFile(filename: string): Promise<string>;
export function verifyCaptureCampaign(filename: string, identity: CampaignIdentity, graph: PlacementGraph): Promise<CaptureCampaign>;
