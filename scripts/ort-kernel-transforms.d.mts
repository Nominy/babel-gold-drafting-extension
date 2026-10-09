export declare const kernelSourceHashes: Readonly<Record<string, string>>;
export declare function assertKernelSource(relative: string, bytes: string | Uint8Array): void;
export interface KernelTransformOptions {
  profile?: boolean;
  baseline?: boolean;
}
export declare function transformBinarySource(source: string, options?: KernelTransformOptions): string;
export declare function transformProgramManagerSource(source: string, options?: Pick<KernelTransformOptions, 'profile'>): string;
export declare function transformBackendProfileSource(source: string): string;
