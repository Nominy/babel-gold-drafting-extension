export type ProjectPresetId = 'ru-gold-2sp-v1';
export type OpenRouterServiceTier = 'default' | 'flex' | 'priority';
export type OpenRouterReasoningEffort = 'default' | 'none' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh';
export type AiBrokerProvider = 'auto' | 'remote-openrouter' | 'local-gemini-nano';

export interface ExtensionSettings {
  backendBaseUrl: string;
  projectPreset: ProjectPresetId;
  openRouterApiKey: string;
  model: string;
  serviceTier: OpenRouterServiceTier;
  reasoningEffort: OpenRouterReasoningEffort;
  aiBrokerProvider: AiBrokerProvider;
  l0ReplacementPreviewEnabled: boolean;
  l0CustomBaseUrl: string;
  l0DontRunLlm: boolean;
  audioInputEnabled: boolean;
  localModelsEnabled: boolean;
}

export interface CapturedAudioTrack {
  trackId: string;
  speakerKey?: string;
  trackLabel?: string;
  source: string;
  blob: Blob;
  mimeType: string;
}

export interface TranscriptRow {
  rowId: string;
  speakerKey: string;
  processedRecordingId?: string;
  startSeconds: number | null;
  endSeconds: number | null;
  text: string;
  index: number;
}

export interface TranscriptJob {
  jobId: string;
  taskScoped?: boolean;
  rows: TranscriptRow[];
}

export interface L0DraftTrackSpec {
  lane: string;
  fieldName: string;
}


export interface L0DraftPayload {
  taskId: string;
  tracks: [L0DraftTrackSpec, L0DraftTrackSpec];
}

export interface L0DraftRow {
  id: string;
  lane: string;
  startSeconds: number;
  endSeconds: number;
  text: string;
}

export interface L0DraftResponse {
  rows: L0DraftRow[];
  summary: Record<string, unknown>;
  models: Record<string, unknown>;
}

export interface L0TimingToken {
  id: string;
  text: string;
  startSeconds: number;
  endSeconds: number;
}

export interface L0TimingTrack {
  lane: string;
  tokens: L0TimingToken[];
}

export interface L0TimingResponse {
  taskId: string;
  tracks: L0TimingTrack[];
  summary: Record<string, unknown>;
  models: Record<string, unknown>;
}

export type DraftRowStatus = 'rewritten' | 'unchanged' | 'failed';

export interface DraftRowResult {
  rowId: string;
  rewrittenText: string;
  status: DraftRowStatus;
  warnings: string[];
}

export interface DraftSummary {
  totalRows: number;
  rewrittenRows: number;
  unchangedRows: number;
  failedRows: number;
  anomalyCounts: Record<string, number>;
}

export interface DraftGenerationMeta {
  model: string;
  rulePackVersion: string;
  generatedAt: string;
}

export interface GenerateDraftRequest {
  projectPreset: ProjectPresetId;
  jobId: string;
  draftSessionId?: string;
  rows: TranscriptRow[];
  openRouterApiKey: string;
  model?: string;
  serviceTier?: OpenRouterServiceTier;
  reasoningEffort?: OpenRouterReasoningEffort;
}

export interface GenerateDraftResponse {
  draftRows: DraftRowResult[];
  summary: DraftSummary;
  generationMeta: DraftGenerationMeta;
}

export interface BrokerTranscriptSegment {
  rowId: string;
  speakerKey: string;
  startSeconds: number;
  endSeconds: number;
}

export interface BrokerTranscribeSegmentRequest {
  openRouterApiKey: string;
  model?: string;
  serviceTier?: OpenRouterServiceTier;
  reasoningEffort?: OpenRouterReasoningEffort;
  segment: BrokerTranscriptSegment;
}

export interface BrokerTranscribeSegmentResponse {
  text: string;
  model: string;
}

export interface BrokerRedistributionSegment {
  id: string;
  index: number;
  speakerKey: string;
  startSeconds: number | null;
  endSeconds: number | null;
  text: string;
}

export interface BrokerRedistributionAllocation {
  segmentId: string;
  text: string;
}

export interface BrokerRedistributionGroup {
  groupId: string;
  speakerKey: string;
  fullText: string;
  segments: BrokerRedistributionSegment[];
  draftAllocations: BrokerRedistributionAllocation[];
}

export interface BrokerRedistributionMove {
  fromIndex: number;
  toIndex: number;
  sentenceCount: number;
}

export interface BrokerRedistributionReview {
  acceptDraft: boolean;
  moves: BrokerRedistributionMove[];
  notes?: string;
}

export interface BrokerRedistributeTextRequest {
  openRouterApiKey: string;
  model?: string;
  serviceTier?: OpenRouterServiceTier;
  reasoningEffort?: OpenRouterReasoningEffort;
  groups: BrokerRedistributionGroup[];
}

export type BrokerRedistributeTextResult =
  | {
      groupId: string;
      ok: true;
      review: BrokerRedistributionReview;
      model: string;
    }
  | {
      groupId: string;
      ok: false;
      error: string;
    };

export interface BrokerRedistributeTextResponse {
  model: string;
  results: BrokerRedistributeTextResult[];
}

export interface GenerateDraftStartedEvent {
  jobId: string;
  totalRows: number;
}

export interface GenerateDraftRowEvent {
  row: DraftRowResult;
  completedRows: number;
  totalRows: number;
  summary: DraftSummary;
}

export interface GenerateDraftErrorEvent {
  error: string;
}

export interface DiffPreviewItem {
  rowId: string;
  index: number;
  before: string;
  after: string;
  status: DraftRowStatus;
  warnings: string[];
}

export interface ApplyDraftResult {
  appliedCount: number;
  missingRowIds: string[];
}

export interface DraftSessionState {
  capturedJob: TranscriptJob | null;
  draftResponse: GenerateDraftResponse | null;
}

export interface DraftingMountController {
  ensureMagicButton(): void;
}
