"use strict";
(() => {
  // ../../shared/babel-extension-platform/packages/babel-extension-frontend/src/index.mjs
  function normalizeBaseUrl(value) {
    return String(value || "").trim().replace(/\/+$/, "");
  }
  function createSettingsStore(config) {
    const getStorageArea2 = config.getStorageArea ?? defaultGetStorageArea;
    return {
      async loadSettings() {
        const storage = getStorageArea2();
        const fallback = config.normalize(config.defaults);
        if (!storage) {
          return fallback;
        }
        return new Promise((resolve) => {
          storage.get(config.storageKey, (items) => {
            const runtime = globalThis.chrome?.runtime;
            if (runtime?.lastError) {
              resolve(fallback);
              return;
            }
            resolve(config.normalize(items?.[config.storageKey]));
          });
        });
      },
      async saveSettings(value) {
        const normalized = config.normalize(value);
        const storage = getStorageArea2();
        if (!storage) {
          return normalized;
        }
        return new Promise((resolve) => {
          storage.set({ [config.storageKey]: normalized }, () => {
            resolve(normalized);
          });
        });
      }
    };
  }
  function defaultGetStorageArea() {
    return globalThis.chrome?.storage?.local ?? null;
  }

  // src/core/backend-client.ts
  function getEndpointUrl(backendBaseUrl, path) {
    return `${normalizeBaseUrl(backendBaseUrl)}${path}`;
  }
  async function parseJsonResponse(response) {
    const text = await response.text();
    try {
      return JSON.parse(text);
    } catch {
      return text;
    }
  }
  function getErrorMessage(status, payload) {
    if (payload && typeof payload === "object" && "error" in payload && typeof payload.error === "string") {
      return payload.error;
    }
    if (typeof payload === "string") {
      return `HTTP ${status}: ${payload.slice(0, 240)}`;
    }
    return `HTTP ${status}`;
  }
  async function generateDraftStream(backendBaseUrl, payload, handlers) {
    const response = await fetch(getEndpointUrl(backendBaseUrl, "/api/draft/generate/stream"), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream"
      },
      body: JSON.stringify(payload)
    });
    if (!response.ok) {
      throw new Error(getErrorMessage(response.status, await parseJsonResponse(response)));
    }
    if (!response.body) {
      throw new Error("Draft backend did not return a stream body.");
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finalResponse = null;
    const processEventBlock = (block) => {
      const lines = block.split(/\r?\n/);
      let eventName = "message";
      const dataLines = [];
      for (const line of lines) {
        if (!line) {
          continue;
        }
        if (line.startsWith("event:")) {
          eventName = line.slice("event:".length).trim();
          continue;
        }
        if (line.startsWith("data:")) {
          dataLines.push(line.slice("data:".length).trimStart());
        }
      }
      if (!dataLines.length) {
        return;
      }
      const payloadText = dataLines.join("\n");
      const parsed = JSON.parse(payloadText);
      if (eventName === "started") {
        handlers.onStarted?.(parsed);
        return;
      }
      if (eventName === "row") {
        handlers.onRow?.(parsed);
        return;
      }
      if (eventName === "done") {
        finalResponse = parsed;
        handlers.onDone?.(finalResponse);
        return;
      }
      if (eventName === "error") {
        const errorPayload = parsed;
        throw new Error(errorPayload.error || "Draft stream failed.");
      }
    };
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      let separatorIndex = buffer.indexOf("\n\n");
      while (separatorIndex >= 0) {
        const block = buffer.slice(0, separatorIndex).trim();
        buffer = buffer.slice(separatorIndex + 2);
        if (block) {
          processEventBlock(block);
        }
        separatorIndex = buffer.indexOf("\n\n");
      }
      if (done) {
        break;
      }
    }
    const trailing = buffer.trim();
    if (trailing) {
      processEventBlock(trailing);
    }
    if (!finalResponse) {
      throw new Error("Draft stream finished without a final response.");
    }
    return finalResponse;
  }

  // src/core/settings.ts
  var SETTINGS_STORAGE_KEY = "babel_gold_drafting_settings";
  var DEFAULT_SETTINGS = {
    backendBaseUrl: "https://reviewgen.ovh",
    projectPreset: "ru-gold-2sp-v1"
  };
  function getStorageArea() {
    const chromeApi = globalThis.chrome;
    return chromeApi?.storage?.local ?? null;
  }
  function normalizeSettings(input) {
    const raw = input && typeof input === "object" ? input : {};
    const backendBaseUrl = typeof raw.backendBaseUrl === "string" && raw.backendBaseUrl.trim() ? raw.backendBaseUrl.trim().replace(/\/+$/, "") : DEFAULT_SETTINGS.backendBaseUrl;
    const projectPreset = raw.projectPreset === "ru-gold-2sp-v1" ? raw.projectPreset : DEFAULT_SETTINGS.projectPreset;
    return {
      backendBaseUrl,
      projectPreset
    };
  }
  var settingsStore = createSettingsStore({
    storageKey: SETTINGS_STORAGE_KEY,
    defaults: DEFAULT_SETTINGS,
    normalize: normalizeSettings,
    getStorageArea
  });
  async function loadSettings() {
    return settingsStore.loadSettings();
  }

  // ../../shared/babel-extension-platform/packages/babel-babel-runtime/src/index.mjs
  function normalizeText(element) {
    if (!(element instanceof HTMLElement)) {
      return "";
    }
    const rawText = typeof element.innerText === "string" ? element.innerText : element.textContent || "";
    return rawText.replace(/\s+/g, " ").trim();
  }
  function setEditableValue(element, value) {
    if (!(element instanceof HTMLElement)) {
      return false;
    }
    const nextValue = typeof value === "string" ? value : String(value ?? "");
    const prototype = element instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : element instanceof HTMLInputElement ? HTMLInputElement.prototype : null;
    const setter = prototype ? Object.getOwnPropertyDescriptor(prototype, "value")?.set : null;
    if (typeof setter === "function") {
      setter.call(element, nextValue);
    } else if ("value" in element) {
      element.value = nextValue;
    } else {
      return false;
    }
    try {
      element.dispatchEvent(
        new InputEvent("input", {
          bubbles: true,
          cancelable: false,
          data: null,
          inputType: "insertText"
        })
      );
    } catch {
      element.dispatchEvent(new Event("input", { bubbles: true, cancelable: false }));
    }
    return true;
  }
  function getReactInternalValue(element, prefix) {
    if (!(element instanceof HTMLElement)) {
      return null;
    }
    for (const name of Object.getOwnPropertyNames(element)) {
      if (typeof name === "string" && name.startsWith(prefix)) {
        return element[name];
      }
    }
    return null;
  }
  function getReactFiber(element) {
    return getReactInternalValue(element, "__reactFiber$");
  }

  // src/core/dom.ts
  var TRANSCRIPT_ROW_SELECTOR = "tbody tr";
  var ROW_TEXTAREA_SELECTOR = 'textarea[placeholder^="What was said"]';
  function parseTimeValue(value) {
    const trimmed = value.trim();
    if (!trimmed) {
      return null;
    }
    const match = trimmed.match(/-?\d+(?::\d+)+(?:\.\d+)?/);
    if (!match) {
      return null;
    }
    return match[0].split(":").reduce((total, part) => {
      if (total === null) {
        return null;
      }
      const numeric = Number(part);
      return Number.isFinite(numeric) ? total * 60 + numeric : null;
    }, 0);
  }
  function setControlledTextareaValue(textarea, value) {
    setEditableValue(textarea, value);
    textarea.dispatchEvent(new Event("change", { bubbles: true, cancelable: false }));
  }
  function getTranscriptRowElements(root = document) {
    return Array.from(root.querySelectorAll(TRANSCRIPT_ROW_SELECTOR)).filter(
      (row) => row.querySelector(ROW_TEXTAREA_SELECTOR)
    );
  }

  // src/core/transcript.ts
  function readRowIdentity(row) {
    const startCell = row.children[2];
    const endCell = row.children[3];
    const speakerCell = row.children[1];
    const identity = {
      rowId: null,
      speakerKey: normalizeText(speakerCell),
      startText: normalizeText(startCell),
      endText: normalizeText(endCell)
    };
    const fiber = getReactFiber(row) || getReactFiber(row.querySelector(ROW_TEXTAREA_SELECTOR));
    let current = fiber;
    let depth = 0;
    while (current && depth < 12) {
      const props = current.memoizedProps;
      const annotation = props && typeof props === "object" && "annotation" in props && props.annotation && typeof props.annotation === "object" ? props.annotation : null;
      if (annotation && typeof annotation.id === "string" && annotation.id.trim()) {
        identity.rowId = annotation.id.trim();
        const processedRecordingId = annotation.processedRecordingId != null ? String(annotation.processedRecordingId).trim() : "";
        const trackLabel = typeof annotation.trackLabel === "string" ? annotation.trackLabel.trim() : "";
        if (processedRecordingId || trackLabel) {
          identity.speakerKey = processedRecordingId || trackLabel;
        }
        break;
      }
      current = current.return;
      depth += 1;
    }
    return identity;
  }
  function makeFallbackRowId(identity, rowIndex) {
    return `row:${identity.speakerKey}:${identity.startText}:${identity.endText}:${rowIndex}`;
  }
  function buildJobId(locationLike = window.location) {
    const search = locationLike.search || "";
    const pathname = locationLike.pathname || "";
    const query = new URLSearchParams(search);
    const explicitId = query.get("jobId") || query.get("transcriptionChunkId") || query.get("annotationId") || query.get("id");
    return explicitId ? explicitId.trim() : `${pathname}${search}`;
  }
  function captureTranscriptJob(root = document, locationLike = window.location) {
    const rows = getTranscriptRowElements(root).map((row, index) => {
      const identity = readRowIdentity(row);
      const textarea = row.querySelector(ROW_TEXTAREA_SELECTOR);
      const startCell = row.children[2];
      const endCell = row.children[3];
      return {
        rowId: identity.rowId || makeFallbackRowId(identity, index),
        speakerKey: identity.speakerKey,
        startSeconds: startCell ? parseTimeValue(normalizeText(startCell)) : null,
        endSeconds: endCell ? parseTimeValue(normalizeText(endCell)) : null,
        text: textarea?.value || "",
        index
      };
    });
    return {
      jobId: buildJobId(locationLike),
      rows
    };
  }
  function buildLocatorMap(root = document) {
    const map = /* @__PURE__ */ new Map();
    for (const [index, row] of getTranscriptRowElements(root).entries()) {
      const identity = readRowIdentity(row);
      const textarea = row.querySelector(ROW_TEXTAREA_SELECTOR);
      if (!textarea) {
        continue;
      }
      const key = identity.rowId || makeFallbackRowId(identity, index);
      map.set(key, textarea);
    }
    return map;
  }
  function applyDraftRows(draftRows, root = document) {
    const locatorMap = buildLocatorMap(root);
    let appliedCount = 0;
    const missingRowIds = [];
    for (const row of draftRows) {
      const textarea = locatorMap.get(row.rowId);
      if (!textarea) {
        missingRowIds.push(row.rowId);
        continue;
      }
      setControlledTextareaValue(textarea, row.rewrittenText);
      appliedCount += 1;
    }
    return {
      appliedCount,
      missingRowIds
    };
  }
  function restoreCapturedRows(job, root = document) {
    return applyDraftRows(
      job.rows.map((row) => ({
        rowId: row.rowId,
        rewrittenText: row.text,
        status: "unchanged",
        warnings: []
      })),
      root
    );
  }
  function buildDiffPreviewItems(originalRows, draftRows) {
    const originalById = new Map(originalRows.map((row) => [row.rowId, row]));
    return draftRows.map((draftRow) => {
      const original = originalById.get(draftRow.rowId);
      if (!original) {
        return null;
      }
      const before = original.text;
      const after = draftRow.rewrittenText;
      if (before === after && draftRow.status !== "failed") {
        return null;
      }
      return {
        rowId: draftRow.rowId,
        index: original.index,
        before,
        after,
        status: draftRow.status,
        warnings: [...draftRow.warnings]
      };
    }).filter((item) => item !== null).sort((left, right) => left.index - right.index);
  }

  // src/content/overlay.ts
  var STYLE_ID = "babel-gold-drafting-style";
  var BUTTON_ID = "babel-gold-drafting-magic-button";
  var OVERLAY_ID = "babel-gold-drafting-overlay";
  var TOOLBAR_BUTTON_SELECTOR = 'button[aria-label="Play all tracks"]';
  function createElement(tagName, className, textContent) {
    const element = document.createElement(tagName);
    if (className) {
      element.className = className;
    }
    if (typeof textContent === "string") {
      element.textContent = textContent;
    }
    return element;
  }
  function ensureStyles() {
    if (document.getElementById(STYLE_ID)) {
      return;
    }
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
    :root {
      --bgd-surface: #ffffff;
      --bgd-surface-muted: #faf5ff;
      --bgd-ink: #221b2d;
      --bgd-muted: #7c6995;
      --bgd-line: #eadff7;
      --bgd-accent: #7c3aed;
      --bgd-accent-hover: #6d28d9;
      --bgd-accent-soft: rgba(124, 58, 237, 0.08);
      --bgd-danger: #dc2626;
      --bgd-success: #16a34a;
      --bgd-radius: 10px;
      --bgd-font: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    #${BUTTON_ID} {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0;
      width: 36px;
      height: 36px;
      border: 1px solid #d8b4fe;
      background: #faf5ff;
      color: var(--bgd-accent);
      border-radius: 8px;
      padding: 0;
      font: 600 13px/1 var(--bgd-font);
      cursor: pointer;
      flex: 0 0 auto;
    }

    #${BUTTON_ID}:hover {
      background: #f3e8ff;
      border-color: #c084fc;
    }

    #${BUTTON_ID}[data-state="loading"] {
      cursor: wait;
      opacity: 0.92;
    }

    #${BUTTON_ID}[data-state="done"] {
      background: var(--bgd-success);
      border-color: var(--bgd-success);
    }

    #${BUTTON_ID}[data-state="error"] {
      background: var(--bgd-danger);
      border-color: var(--bgd-danger);
    }

    #${BUTTON_ID} .bgd-icon {
      font-size: 16px;
      line-height: 1;
    }

    #${BUTTON_ID} .bgd-spinner {
      width: 14px;
      height: 14px;
      border-radius: 50%;
      border: 2px solid currentColor;
      border-right-color: transparent;
      display: none;
    }

    #${BUTTON_ID}[data-state="loading"] .bgd-spinner {
      display: inline-block;
      animation: bgd-spin 0.7s linear infinite;
    }

    #${BUTTON_ID}[data-state="loading"] .bgd-icon {
      display: none;
    }

    #${OVERLAY_ID} {
      position: fixed;
      inset: 0;
      z-index: 2147483647;
      font: 13px/1.5 var(--bgd-font);
    }

    #${OVERLAY_ID}[hidden] {
      display: none;
    }

    #${OVERLAY_ID} .bgd-backdrop {
      position: absolute;
      inset: 0;
      background: rgba(15, 10, 25, 0.42);
      backdrop-filter: blur(2px);
    }

    #${OVERLAY_ID} .bgd-shell {
      position: relative;
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 16px;
    }

    #${OVERLAY_ID} .bgd-dialog {
      width: min(980px, calc(100vw - 24px));
      max-height: calc(100vh - 24px);
      overflow: auto;
      background: var(--bgd-surface);
      border: 1px solid var(--bgd-line);
      border-radius: var(--bgd-radius);
      box-shadow: 0 30px 80px rgba(15, 10, 25, 0.28);
      color: var(--bgd-ink);
    }

    #${OVERLAY_ID} .bgd-header {
      padding: 12px 16px;
      border-bottom: 1px solid var(--bgd-line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      position: sticky;
      top: 0;
      background: var(--bgd-surface);
      z-index: 1;
    }

    #${OVERLAY_ID} .bgd-header-title {
      font-weight: 700;
      font-size: 14px;
    }

    #${OVERLAY_ID} .bgd-header-subtitle {
      color: var(--bgd-muted);
      font-size: 12px;
      margin-top: 2px;
    }

    #${OVERLAY_ID} .bgd-close {
      border: 1px solid var(--bgd-line);
      border-radius: 8px;
      padding: 7px 10px;
      background: #fff;
      color: var(--bgd-ink);
      font: inherit;
      cursor: pointer;
    }

    #${OVERLAY_ID} .bgd-main {
      padding: 14px 16px 16px;
      display: grid;
      gap: 12px;
    }

    #${OVERLAY_ID} .bgd-status {
      font-size: 12px;
      color: var(--bgd-muted);
    }

    #${OVERLAY_ID} .bgd-status[data-error="true"] {
      color: var(--bgd-danger);
    }

    #${OVERLAY_ID} .bgd-summary {
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
      font-size: 12px;
      color: var(--bgd-muted);
    }

    #${OVERLAY_ID} .bgd-summary-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }

    #${OVERLAY_ID} .bgd-summary-dot {
      width: 7px;
      height: 7px;
      border-radius: 999px;
      background: var(--bgd-accent);
    }

    #${OVERLAY_ID} .bgd-block {
      border: 1px solid var(--bgd-line);
      border-radius: var(--bgd-radius);
      background: var(--bgd-surface-muted);
      overflow: hidden;
    }

    #${OVERLAY_ID} .bgd-block-header {
      padding: 10px 12px;
      border-bottom: 1px solid var(--bgd-line);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--bgd-muted);
    }

    #${OVERLAY_ID} .bgd-block-body {
      padding: 12px;
    }

    #${OVERLAY_ID} .bgd-empty {
      color: var(--bgd-muted);
      font-size: 12px;
      text-align: center;
      padding: 12px;
      background: #fff;
      border-radius: 8px;
    }

    #${OVERLAY_ID} .bgd-diff-list {
      display: grid;
      gap: 10px;
      max-height: 56vh;
      overflow: auto;
    }

    #${OVERLAY_ID} .bgd-card {
      border: 1px solid var(--bgd-line);
      border-radius: 8px;
      background: #fff;
      padding: 10px;
      display: grid;
      gap: 8px;
    }

    #${OVERLAY_ID} .bgd-card.failed {
      border-color: #fecaca;
      background: #fff5f5;
    }

    #${OVERLAY_ID} .bgd-card-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      flex-wrap: wrap;
    }

    #${OVERLAY_ID} .bgd-card-title {
      font-size: 12px;
      font-weight: 600;
    }

    #${OVERLAY_ID} .bgd-badge {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 11px;
      font-weight: 600;
      background: var(--bgd-accent-soft);
      color: var(--bgd-accent);
    }

    #${OVERLAY_ID} .bgd-diff-view {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      background: #fcfbff;
    }

    #${OVERLAY_ID} .bgd-diff-pane {
      min-width: 0;
    }

    #${OVERLAY_ID} .bgd-diff-label {
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      color: var(--bgd-muted);
      margin-bottom: 4px;
    }

    #${OVERLAY_ID} .bgd-diff-content {
      border: 1px solid var(--bgd-line);
      border-radius: 8px;
      background: #fff;
      padding: 8px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 11px;
      white-space: pre-wrap;
      word-break: break-word;
      min-height: 58px;
    }

    #${OVERLAY_ID} .bgd-warning-list {
      margin: 0;
      padding-left: 18px;
      font-size: 12px;
      color: var(--bgd-muted);
    }

    #${OVERLAY_ID} .bgd-footer {
      padding: 12px 16px 16px;
      border-top: 1px solid var(--bgd-line);
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 8px;
      flex-wrap: wrap;
      background: var(--bgd-surface);
      position: sticky;
      bottom: 0;
    }

    #${OVERLAY_ID} .bgd-button {
      border: 1px solid var(--bgd-line);
      border-radius: 8px;
      padding: 8px 12px;
      background: #fff;
      color: var(--bgd-ink);
      font: inherit;
      font-weight: 600;
      cursor: pointer;
    }

    #${OVERLAY_ID} .bgd-button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    #${OVERLAY_ID} .bgd-button[data-variant="primary"] {
      border-color: transparent;
      background: var(--bgd-accent);
      color: #fff;
    }

    @keyframes bgd-spin {
      to { transform: rotate(360deg); }
    }

    @media (max-width: 720px) {
      #${OVERLAY_ID} .bgd-shell {
        padding: 8px;
      }

      #${OVERLAY_ID} .bgd-dialog {
        width: calc(100vw - 16px);
        max-height: calc(100vh - 16px);
      }

      #${OVERLAY_ID} .bgd-diff-view {
        grid-template-columns: 1fr;
      }
    }
  `;
    document.documentElement.appendChild(style);
  }
  var DraftingOverlayController = class {
    button = null;
    overlay = null;
    dialogEl = null;
    statusEl = null;
    summaryEl = null;
    previewEl = null;
    applyButton = null;
    restoreButton = null;
    closeButton = null;
    state = {
      capturedJob: null,
      draftResponse: null,
      lastApplyResult: null
    };
    streamedRows = [];
    streamedSummary = null;
    streamedCompletedRows = 0;
    streamedTotalRows = 0;
    busy = false;
    observer = null;
    mount() {
      ensureStyles();
      this.ensureButton();
      this.ensureOverlay();
      this.ensureObserver();
      this.render();
    }
    unmount() {
      this.button?.remove();
      this.overlay?.remove();
      this.button = null;
      this.overlay = null;
      this.dialogEl = null;
      this.statusEl = null;
      this.summaryEl = null;
      this.previewEl = null;
      this.applyButton = null;
      this.restoreButton = null;
      this.closeButton = null;
      this.observer?.disconnect();
      this.observer = null;
    }
    ensureButton() {
      const host = this.findToolbarHost();
      if (!host) {
        return;
      }
      if (this.button?.isConnected && this.button.parentElement === host) {
        return;
      }
      this.button?.remove();
      const button = document.createElement("button");
      button.id = BUTTON_ID;
      button.type = "button";
      button.dataset.state = "idle";
      button.setAttribute("aria-label", "Gold Draft");
      button.title = "Gold Draft";
      button.innerHTML = `
      <span class="bgd-icon">\u{1FA84}</span>
      <span class="bgd-spinner"></span>
    `;
      button.addEventListener("click", () => {
        void this.runMagicDraft();
      });
      host.appendChild(button);
      this.button = button;
    }
    findToolbarHost() {
      const anchor = document.querySelector(TOOLBAR_BUTTON_SELECTOR);
      if (!(anchor instanceof HTMLButtonElement)) {
        return null;
      }
      return anchor.parentElement instanceof HTMLElement ? anchor.parentElement : null;
    }
    ensureObserver() {
      if (this.observer || typeof MutationObserver === "undefined") {
        return;
      }
      this.observer = new MutationObserver(() => {
        this.ensureButton();
      });
      const root = document.body || document.documentElement;
      if (root) {
        this.observer.observe(root, { childList: true, subtree: true });
      }
    }
    ensureOverlay() {
      if (this.overlay?.isConnected) {
        return;
      }
      const overlay = createElement("div");
      overlay.id = OVERLAY_ID;
      overlay.hidden = true;
      const backdrop = createElement("div", "bgd-backdrop");
      backdrop.addEventListener("click", () => this.closeDialog());
      const shell = createElement("div", "bgd-shell");
      const dialog = createElement("div", "bgd-dialog");
      dialog.addEventListener("click", (event) => event.stopPropagation());
      const header = createElement("div", "bgd-header");
      const titleWrap = createElement("div");
      titleWrap.append(
        createElement("div", "bgd-header-title", "Gold Draft"),
        createElement("div", "bgd-header-subtitle", "Silver -> Gold draft preview before apply")
      );
      const closeButton = createElement("button", "bgd-close", "Close");
      closeButton.type = "button";
      closeButton.addEventListener("click", () => this.closeDialog());
      header.append(titleWrap, closeButton);
      const main = createElement("div", "bgd-main");
      const statusEl = createElement("div", "bgd-status", "Click the wand to generate a draft.");
      const summaryBlock = createElement("section", "bgd-block");
      summaryBlock.append(createElement("div", "bgd-block-header", "Summary"));
      const summaryBody = createElement("div", "bgd-block-body");
      const summaryEl = createElement("div", "bgd-empty", "No draft generated yet.");
      summaryBody.append(summaryEl);
      summaryBlock.append(summaryBody);
      const previewBlock = createElement("section", "bgd-block");
      previewBlock.append(createElement("div", "bgd-block-header", "Diff Preview"));
      const previewBody = createElement("div", "bgd-block-body");
      const previewEl = createElement("div", "bgd-empty", "The diff will appear here after generation.");
      previewBody.append(previewEl);
      previewBlock.append(previewBody);
      main.append(statusEl, summaryBlock, previewBlock);
      const footer = createElement("div", "bgd-footer");
      const restoreButton = createElement("button", "bgd-button", "Restore Original");
      restoreButton.type = "button";
      restoreButton.addEventListener("click", () => void this.restoreOriginal());
      const applyButton = createElement("button", "bgd-button", "Apply Draft");
      applyButton.type = "button";
      applyButton.dataset.variant = "primary";
      applyButton.addEventListener("click", () => void this.applyDraft());
      footer.append(restoreButton, applyButton);
      dialog.append(header, main, footer);
      shell.append(dialog);
      overlay.append(backdrop, shell);
      document.documentElement.appendChild(overlay);
      this.overlay = overlay;
      this.dialogEl = dialog;
      this.statusEl = statusEl;
      this.summaryEl = summaryEl;
      this.previewEl = previewEl;
      this.applyButton = applyButton;
      this.restoreButton = restoreButton;
      this.closeButton = closeButton;
    }
    openDialog() {
      if (this.overlay) {
        this.overlay.hidden = false;
      }
    }
    closeDialog() {
      if (!this.busy && this.overlay) {
        this.overlay.hidden = true;
      }
    }
    setButtonState(mode, label) {
      if (!(this.button instanceof HTMLButtonElement)) {
        return;
      }
      this.button.dataset.state = mode;
      this.button.disabled = mode === "loading";
      this.button.title = label;
      this.button.setAttribute("aria-label", label);
    }
    setBusy(nextBusy) {
      this.busy = nextBusy;
      if (this.applyButton) {
        this.applyButton.disabled = nextBusy || !this.state.draftResponse;
      }
      if (this.restoreButton) {
        this.restoreButton.disabled = nextBusy || !this.state.capturedJob;
      }
      if (this.closeButton) {
        this.closeButton.disabled = nextBusy;
      }
    }
    setStatus(message, isError = false) {
      if (this.statusEl) {
        this.statusEl.textContent = message;
        this.statusEl.dataset.error = isError ? "true" : "false";
      }
    }
    renderSummary(draftResponse) {
      if (!this.summaryEl) {
        return;
      }
      if (!this.state.capturedJob) {
        this.summaryEl.className = "bgd-empty";
        this.summaryEl.textContent = "No transcript captured yet.";
        return;
      }
      const summary = createElement("div", "bgd-summary");
      summary.append(
        this.createSummaryPill(`Job ${this.state.capturedJob.jobId}`),
        this.createSummaryPill(`${this.state.capturedJob.rows.length} rows captured`)
      );
      if (draftResponse) {
        summary.append(
          this.createSummaryPill(`${draftResponse.summary.rewrittenRows} rewritten`),
          this.createSummaryPill(`${draftResponse.summary.unchangedRows} unchanged`),
          this.createSummaryPill(`${draftResponse.summary.failedRows} failed`)
        );
      } else if (this.streamedTotalRows > 0) {
        summary.append(this.createSummaryPill(`${this.streamedCompletedRows}/${this.streamedTotalRows} rows complete`));
        if (this.streamedSummary) {
          summary.append(
            this.createSummaryPill(`${this.streamedSummary.rewrittenRows} rewritten`),
            this.createSummaryPill(`${this.streamedSummary.unchangedRows} unchanged`),
            this.createSummaryPill(`${this.streamedSummary.failedRows} failed`)
          );
        }
      }
      this.summaryEl.className = "";
      this.summaryEl.replaceChildren(summary);
    }
    createSummaryPill(text) {
      const pill = createElement("div", "bgd-summary-pill");
      pill.append(createElement("span", "bgd-summary-dot"), createElement("span", "", text));
      return pill;
    }
    renderPreview() {
      if (!this.previewEl) {
        return;
      }
      const previousList = this.previewEl.firstElementChild instanceof HTMLDivElement && this.previewEl.firstElementChild.classList.contains("bgd-diff-list") ? this.previewEl.firstElementChild : null;
      const previousListScrollTop = previousList?.scrollTop ?? 0;
      const captured = this.state.capturedJob;
      const draft = this.state.draftResponse;
      if (!captured) {
        this.previewEl.className = "bgd-empty";
        this.previewEl.textContent = "The diff will appear here after generation.";
        return;
      }
      const sourceRows = draft ? draft.draftRows : this.streamedRows;
      if (!sourceRows.length) {
        this.previewEl.className = "bgd-empty";
        this.previewEl.textContent = "Waiting for the first completed row...";
        return;
      }
      const diffItems = buildDiffPreviewItems(captured.rows, sourceRows);
      if (!diffItems.length) {
        this.previewEl.className = "bgd-empty";
        this.previewEl.textContent = draft ? "No row text changed." : "Completed rows have no visible text changes yet.";
        return;
      }
      const list = createElement("div", "bgd-diff-list");
      for (const item of diffItems.slice(0, 40)) {
        const card = createElement("article", `bgd-card ${item.status === "failed" ? "failed" : ""}`);
        const top = createElement("div", "bgd-card-top");
        top.append(
          createElement("div", "bgd-card-title", `Row ${item.index + 1}`),
          createElement("div", "bgd-badge", item.status)
        );
        card.append(top);
        const diffView = createElement("div", "bgd-diff-view");
        const beforePane = createElement("div", "bgd-diff-pane");
        beforePane.append(
          createElement("div", "bgd-diff-label", "Before"),
          this.createDiffContent(item.before)
        );
        const afterPane = createElement("div", "bgd-diff-pane");
        afterPane.append(
          createElement("div", "bgd-diff-label", "After"),
          this.createDiffContent(item.after)
        );
        diffView.append(beforePane, afterPane);
        card.append(diffView);
        if (item.warnings.length) {
          const warnings = createElement("ul", "bgd-warning-list");
          for (const warning of item.warnings) {
            warnings.append(createElement("li", "", warning));
          }
          card.append(warnings);
        }
        list.append(card);
      }
      if (diffItems.length > 40) {
        list.append(createElement("div", "bgd-empty", `Showing first 40 of ${diffItems.length} changed rows.`));
      }
      this.previewEl.className = "";
      this.previewEl.replaceChildren(list);
      list.scrollTop = previousListScrollTop;
    }
    createDiffContent(text) {
      const content = createElement("div", "bgd-diff-content");
      content.textContent = text || "(empty)";
      return content;
    }
    render() {
      const dialogScrollTop = this.dialogEl?.scrollTop ?? 0;
      this.renderSummary(this.state.draftResponse);
      this.renderPreview();
      this.setBusy(this.busy);
      if (this.dialogEl) {
        this.dialogEl.scrollTop = dialogScrollTop;
      }
    }
    async runMagicDraft() {
      this.openDialog();
      this.state = {
        capturedJob: null,
        draftResponse: null,
        lastApplyResult: null
      };
      this.streamedRows = [];
      this.streamedSummary = null;
      this.streamedCompletedRows = 0;
      this.streamedTotalRows = 0;
      this.render();
      try {
        this.setBusy(true);
        this.setButtonState("loading", "Generating...");
        this.setStatus("Capturing transcript...");
        const capturedJob = captureTranscriptJob();
        if (!capturedJob.rows.length) {
          throw new Error("No transcript rows detected on this page.");
        }
        this.state.capturedJob = capturedJob;
        this.streamedTotalRows = capturedJob.rows.length;
        this.render();
        const settings = await loadSettings();
        this.setStatus(`Starting Gold draft stream for ${capturedJob.rows.length} rows...`);
        const draftResponse = await generateDraftStream(settings.backendBaseUrl, {
          projectPreset: settings.projectPreset,
          jobId: capturedJob.jobId,
          rows: capturedJob.rows
        }, {
          onStarted: ({ totalRows }) => {
            this.streamedTotalRows = totalRows;
            this.setStatus(`Streaming Gold draft... 0 / ${totalRows} rows complete.`);
            this.render();
          },
          onRow: ({ row, completedRows, totalRows, summary }) => {
            const existingIndex = this.streamedRows.findIndex((candidate) => candidate.rowId === row.rowId);
            if (existingIndex >= 0) {
              this.streamedRows[existingIndex] = row;
            } else {
              this.streamedRows.push(row);
            }
            this.streamedCompletedRows = completedRows;
            this.streamedTotalRows = totalRows;
            this.streamedSummary = summary;
            this.setStatus(`Streaming Gold draft... ${completedRows} / ${totalRows} rows complete.`);
            this.render();
          },
          onDone: (response) => {
            this.streamedRows = response.draftRows;
            this.streamedSummary = response.summary;
            this.streamedCompletedRows = response.summary.totalRows;
            this.streamedTotalRows = response.summary.totalRows;
          }
        });
        this.state.draftResponse = draftResponse;
        this.setStatus(
          `Draft ready. ${draftResponse.summary.rewrittenRows} rewritten, ${draftResponse.summary.failedRows} failed fallback rows.`
        );
        this.setButtonState("done", "Draft Ready");
        window.setTimeout(() => this.setButtonState("idle", "Gold Draft"), 1600);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        this.setStatus(message, true);
        this.setButtonState("error", "Draft Failed");
        window.setTimeout(() => this.setButtonState("idle", "Gold Draft"), 2200);
      } finally {
        this.setBusy(false);
        this.render();
      }
    }
    async applyDraft() {
      if (!this.state.draftResponse) {
        this.setStatus("No draft available yet.", true);
        return;
      }
      const result = applyDraftRows(this.state.draftResponse.draftRows);
      this.state.lastApplyResult = result;
      const missingNote = result.missingRowIds.length ? ` Missing ${result.missingRowIds.length} rows during apply.` : "";
      this.setStatus(`Applied draft to ${result.appliedCount} rows.${missingNote}`);
      this.setButtonState("done", "Applied");
      this.closeDialog();
      window.setTimeout(() => this.setButtonState("idle", "Gold Draft"), 1600);
      this.render();
    }
    async restoreOriginal() {
      if (!this.state.capturedJob) {
        this.setStatus("No captured snapshot to restore.", true);
        return;
      }
      const result = restoreCapturedRows(this.state.capturedJob);
      this.state.lastApplyResult = result;
      const missingNote = result.missingRowIds.length ? ` Missing ${result.missingRowIds.length} rows during restore.` : "";
      this.setStatus(`Restored ${result.appliedCount} original rows.${missingNote}`);
      this.setButtonState("done", "Restored");
      window.setTimeout(() => this.setButtonState("idle", "Gold Draft"), 1600);
      this.render();
    }
  };

  // src/content/entry.ts
  var controller = null;
  function isTranscriptionRoute() {
    return /^\/transcription(?:\/|$)/.test(window.location.pathname || "");
  }
  function syncController() {
    if (isTranscriptionRoute()) {
      controller ??= new DraftingOverlayController();
      controller.mount();
      return;
    }
    controller?.unmount();
  }
  function patchHistoryMethod(methodName) {
    const original = window.history[methodName];
    if (typeof original !== "function") {
      return;
    }
    window.history[methodName] = function patchedHistoryMethod(...args) {
      const result = original.apply(this, args);
      window.setTimeout(syncController, 0);
      return result;
    };
  }
  function boot() {
    syncController();
    patchHistoryMethod("pushState");
    patchHistoryMethod("replaceState");
    window.addEventListener("popstate", () => window.setTimeout(syncController, 0), true);
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else {
    boot();
  }
})();
//# sourceMappingURL=entry.js.map
