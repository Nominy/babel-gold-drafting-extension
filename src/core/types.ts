export type ProjectPresetId = 'ru-gold-2sp-v1';

export interface ExtensionSettings {
  backendBaseUrl: string;
  projectPreset: ProjectPresetId;
}

export interface TranscriptRow {
  rowId: string;
  speakerKey: string;
  startSeconds: number | null;
  endSeconds: number | null;
  text: string;
  index: number;
}

export interface TranscriptJob {
  jobId: string;
  rows: TranscriptRow[];
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
  rows: TranscriptRow[];
}

export interface GenerateDraftResponse {
  draftRows: DraftRowResult[];
  summary: DraftSummary;
  generationMeta: DraftGenerationMeta;
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
  lastApplyResult: ApplyDraftResult | null;
}
