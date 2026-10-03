import release from '../../model-release.json';
export const INFERENCE_RELEASE = Object.freeze(release);
export const INFERENCE_HEADERS = Object.freeze({ 'X-Babel-Inference-Release': release.id });
export function assertReleasedGraphs(files: readonly { path: string; sha256: string }[]): void {
  for (const [path, sha] of Object.entries(release.graphs)) {
    if (files.find(file => file.path === path)?.sha256 !== sha) {
      throw new Error('A model update is required. Download and test the current C-denoise bundle in Options.');
    }
  }
}
