import type { BrokerRedistributionGroup } from './types';
import { LOCAL_MODEL_AUDIO_CHUNK_BYTES } from './local-model-offscreen-protocol';

export const MAI_MESSAGE_TYPE = 'babel-gold-drafting:mai';
export const MAI_PROTOCOL_VERSION = 1 as const;
export const MAI_MAX_AUDIO_BYTES = 512 * 1024 * 1024;
export type MaiOperation = 'upload' | 'timing' | 'lookup' | 'draft' | 'segment' | 'redistribute';
export interface MaiEnvelope {
  type: typeof MAI_MESSAGE_TYPE;
  version: typeof MAI_PROTOCOL_VERSION;
  requestId: string;
  operation: MaiOperation;
}
export interface MaiUploadRequest extends MaiEnvelope {
  operation: 'upload';
  transferId: string;
  chunkIndex: number;
  chunkCount: number;
  totalBytes: number;
  dataBase64: string;
}
export interface MaiSource {
  lane: string;
  trackId: string;
  transferId: string;
}
export interface MaiTimingRequest extends MaiEnvelope {
  operation: 'timing';
  taskId: string;
  sources: MaiSource[];
}
export interface MaiCacheRequest extends MaiEnvelope {
  operation: 'lookup' | 'draft';
  taskId: string;
}
export interface MaiSegmentRequest extends MaiEnvelope {
  operation: 'segment';
  taskId: string;
  lane: string;
  startSeconds: number;
  endSeconds: number;
}
export interface MaiRedistributeRequest extends MaiEnvelope {
  operation: 'redistribute';
  groups: BrokerRedistributionGroup[];
}
export type MaiRequest = MaiUploadRequest | MaiTimingRequest | MaiCacheRequest | MaiSegmentRequest | MaiRedistributeRequest;
export type MaiResponse = MaiEnvelope & (
  | { ok: true; result: unknown }
  | { ok: false; error: { code: string; message: string } }
);

function text(value: unknown, max = 4096): value is string {
  return typeof value === 'string' && value.length > 0 && value.length <= max;
}
function only(record: object, keys: string[]): boolean {
  return Object.keys(record).every((key) => keys.includes(key));
}
function finite(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}
function validGroups(value: unknown): value is BrokerRedistributionGroup[] {
  if (!Array.isArray(value) || value.length > 100) return false;
  const groups = value as BrokerRedistributionGroup[];
  return groups.every((group) => {
    if (!group || typeof group !== 'object' || Array.isArray(group) ||
        !only(group, ['groupId', 'speakerKey', 'fullText', 'segments', 'draftAllocations']) ||
        !text(group.groupId) || !text(group.speakerKey) || typeof group.fullText !== 'string' ||
        group.fullText.length > 500_000 || !Array.isArray(group.segments) || group.segments.length > 5000 ||
        !Array.isArray(group.draftAllocations) || group.draftAllocations.length !== group.segments.length) return false;
    return group.segments.every((row) => row !== null && typeof row === 'object' && !Array.isArray(row) &&
      only(row, ['id', 'index', 'speakerKey', 'startSeconds', 'endSeconds', 'text']) && text(row.id) &&
      Number.isSafeInteger(row.index) && text(row.speakerKey) &&
      (row.startSeconds === null || finite(row.startSeconds)) &&
      (row.endSeconds === null || finite(row.endSeconds)) && typeof row.text === 'string' && row.text.length <= 500_000) &&
      group.draftAllocations.every((row) => row !== null && typeof row === 'object' && !Array.isArray(row) &&
        only(row, ['segmentId', 'text']) && text(row.segmentId) && typeof row.text === 'string' && row.text.length <= 500_000);
  }) && JSON.stringify(groups).length <= 2_000_000;
}
type MaiRequestCandidate = Partial<Omit<MaiUploadRequest, 'operation'> & Omit<MaiTimingRequest, 'operation'> &
  Omit<MaiCacheRequest, 'operation'> & Omit<MaiSegmentRequest, 'operation'> & Omit<MaiRedistributeRequest, 'operation'>> &
  { operation?: unknown };
export function isMaiRequest(input: unknown): input is MaiRequest {
  if (!input || typeof input !== 'object' || Array.isArray(input)) return false;
  const value = input as MaiRequestCandidate;
  if (value.type !== MAI_MESSAGE_TYPE || value.version !== MAI_PROTOCOL_VERSION ||
      !text(value.requestId, 256)) return false;
  const base = ['type', 'version', 'requestId', 'operation'];
  if (value.operation === 'upload') {
    if (!only(value, [...base, 'transferId', 'chunkIndex', 'chunkCount', 'totalBytes', 'dataBase64']) ||
        !text(value.transferId, 256) || !Number.isSafeInteger(value.totalBytes) ||
        !Number.isSafeInteger(value.chunkCount) || !Number.isSafeInteger(value.chunkIndex) ||
        typeof value.dataBase64 !== 'string' || value.dataBase64.length > Math.ceil(LOCAL_MODEL_AUDIO_CHUNK_BYTES / 3) * 4 ||
        !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(value.dataBase64)) return false;
    const total = value.totalBytes as number;
    const count = value.chunkCount as number;
    const index = value.chunkIndex as number;
    const padding = value.dataBase64.endsWith('==') ? 2 : value.dataBase64.endsWith('=') ? 1 : 0;
    const bytes = value.dataBase64.length / 4 * 3 - padding;
    return total >= 44 && total <= MAI_MAX_AUDIO_BYTES && count === Math.ceil(total / LOCAL_MODEL_AUDIO_CHUNK_BYTES) &&
      index >= 0 && index < count && bytes === Math.min(LOCAL_MODEL_AUDIO_CHUNK_BYTES, total - index * LOCAL_MODEL_AUDIO_CHUNK_BYTES);
  }
  if (value.operation === 'redistribute') return only(value, [...base, 'groups']) && validGroups(value.groups);
  if (!text(value.taskId, 8192)) return false;
  if (value.operation === 'lookup' || value.operation === 'draft') return only(value, [...base, 'taskId']);
  if (value.operation === 'segment') return only(value, [...base, 'taskId', 'lane', 'startSeconds', 'endSeconds']) &&
    text(value.lane, 256) && finite(value.startSeconds) && finite(value.endSeconds) &&
    value.startSeconds >= 0 && value.endSeconds > value.startSeconds;
  if (value.operation !== 'timing' || !only(value, [...base, 'taskId', 'sources']) ||
      !Array.isArray(value.sources) || value.sources.length !== 2) return false;
  return value.sources.every((source) => source !== null && typeof source === 'object' && !Array.isArray(source) &&
    only(source, ['lane', 'trackId', 'transferId']) && text(source.lane, 256) && text(source.trackId, 256) && text(source.transferId, 256)) &&
    new Set(value.sources.map((source) => source.lane.trim().toLocaleLowerCase())).size === 2 &&
    new Set(value.sources.map((source) => source.transferId)).size === 2;
}
export class MaiError extends Error {
  constructor(readonly code: string, message: string) {
    super(message);
    this.name = 'MaiError';
  }
}
