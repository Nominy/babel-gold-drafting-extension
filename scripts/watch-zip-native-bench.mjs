import path from 'node:path';
import { watch } from 'node:fs';
import { open, stat } from 'node:fs/promises';
import { createServer } from 'node:http';

const directory = process.argv[2] && path.resolve(process.argv[2]);
const port = Number(process.argv[3] ?? 53927);
if (!directory || !Number.isInteger(port) || port < 1 || port > 65535) {
  throw new Error('Usage: node scripts/watch-zip-native-bench.mjs PRIVATE_RUN_DIR [PORT]');
}
const filename = path.join(directory, 'events.jsonl');
const chunkBytes = 256 * 1024;
const maxRecordBytes = 1024 * 1024;
const retainedResults = 1000;
const retainedFailures = 100;
const clients = new Set();
const state = {
  directory, source: 'Waiting for events.jsonl', current: null, results: [], failures: [],
  records: 0, resultCount: 0, failureCount: 0, malformedRecords: 0, resets: 0,
  lastActivity: null, sourceError: null, readerWarning: null,
  retention: { results: retainedResults, failures: retainedFailures },
};
let offset = 0;
let identity = null;
let pending = Buffer.alloc(0);
let discardingRecord = false;
let watcher = null;
let watchedDirectory = null;
let timer = null;
let scanning = false;
let requested = false;
let stopped = false;

const pendingClients = new WeakSet();
function publish() {
  const payload = `data: ${JSON.stringify(state)}\n\n`;
  for (const response of clients) {
    // Coalesce updates for a slow client; retain at most one queued snapshot.
    if (response.writableNeedDrain) pendingClients.add(response);
    else response.write(payload);
  }
}

function reset() {
  offset = 0;
  pending = Buffer.alloc(0);
  discardingRecord = false;
  Object.assign(state, {
    current: null, results: [], failures: [], records: 0, resultCount: 0,
    failureCount: 0, malformedRecords: 0, lastActivity: null, readerWarning: null,
    resets: state.resets + 1,
  });
}

// Only small scalar metadata reaches the browser; arrays, audio, transcripts,
// and tensor payloads are never projected into the public snapshot.
function publicMetrics(value, prefix = '', output = {}, depth = 0) {
  if (!value || typeof value !== 'object' || depth > 3) return output;
  if (Array.isArray(value)) {
    if (prefix !== 'numericalChanges') return output;
    for (let index = 0; index < Math.min(value.length, 2); index++) {
      const lane = value[index];
      if (!lane || typeof lane !== 'object') continue;
      for (const key of ['relativeRmse', 'maxAbsError', 'snrDb']) {
        if (typeof lane[key] === 'number' && Number.isFinite(lane[key])) output[`lane${index + 1}.${key}`] = lane[key];
      }
    }
    return output;
  }
  for (const [key, item] of Object.entries(value)) {
    if (Object.keys(output).length >= 60) break;
    if (/audio|transcript|tensor|waveform|samples|payload|base64/i.test(key)) continue;
    const name = prefix ? `${prefix}.${key.slice(0, 80)}` : key.slice(0, 80);
    if (typeof item === 'number' && Number.isFinite(item)) output[name] = item;
    else if (typeof item === 'boolean') output[name] = item;
    else if (typeof item === 'string' && /precision|dtype|batch|error|status|stage|recording|input|device|mode|progress/i.test(key)) {
      output[name] = item.slice(0, 240);
    } else if (item && typeof item === 'object') publicMetrics(item, name, output, depth + 1);
  }
  return output;
}

function consumeRecord(bytes) {
  if (!bytes.toString('utf8').trim()) return;
  let event;
  try {
    event = JSON.parse(bytes.toString('utf8'));
    if (!event || !['status', 'result', 'error'].includes(event.kind) || typeof event.candidate !== 'string') {
      throw new Error('Invalid event schema');
    }
  } catch {
    state.malformedRecords++;
    state.readerWarning = 'Skipped malformed complete JSONL record(s); partial final records are still pending.';
    return;
  }
  const record = {
    time: typeof event.time === 'string' ? event.time.slice(0, 80) : null,
    kind: event.kind, candidate: event.candidate.slice(0, 240),
    message: typeof event.message === 'string' ? event.message.slice(0, 2000) : '',
    metrics: publicMetrics(event.metrics),
  };
  state.records++;
  state.current = record;
  if (record.kind === 'result') {
    state.resultCount++;
    state.results.push(record);
    if (state.results.length > retainedResults) state.results.shift();
  }
  if (record.kind === 'error') {
    state.failureCount++;
    state.failures.push(record);
    if (state.failures.length > retainedFailures) state.failures.shift();
  }
}

function consumeChunk(chunk) {
  const bytes = pending.length ? Buffer.concat([pending, chunk]) : chunk;
  let start = 0;
  for (;;) {
    const end = bytes.indexOf(10, start);
    if (end === -1) break;
    if (!discardingRecord) {
      if (end - start <= maxRecordBytes) consumeRecord(bytes.subarray(start, end));
      else oversizedRecord();
    }
    discardingRecord = false;
    start = end + 1;
  }
  if (discardingRecord) pending = Buffer.alloc(0);
  else if (bytes.length - start > maxRecordBytes) {
    oversizedRecord();
    discardingRecord = true;
    pending = Buffer.alloc(0);
  } else {
    // Copy only the unfinished record, not the entire underlying read buffer.
    pending = Buffer.from(bytes.subarray(start));
  }
}

function oversizedRecord() {
  state.malformedRecords++;
  state.readerWarning = 'Skipped a JSONL record larger than 1 MiB; scalar benchmark events are expected.';
}

function schedule(delay = 100) {
  if (stopped) return;
  if (scanning) { requested = true; return; }
  if (timer) return;
  timer = setTimeout(() => {
    timer = null;
    void refresh();
  }, delay);
}

async function attachWatcher() {
  // Watch the closest existing ancestor while the producer creates its directory.
  // The fallback timer also catches missed/coalesced Windows fs.watch events.
  let root = directory;
  for (;;) {
    try {
      if (!(await stat(root)).isDirectory()) throw new Error(`Not a directory: ${root}`);
      break;
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
      const parent = path.dirname(root);
      if (parent === root) throw error;
      root = parent;
    }
  }
  if (watchedDirectory === root && watcher) return;
  watcher?.close();
  watcher = null;
  watchedDirectory = null;
  watcher = watch(root, (_event, changedName) => {
    if (root !== directory || !changedName || String(changedName) === 'events.jsonl') schedule();
  });
  watchedDirectory = root;
  watcher.on('error', error => {
    state.sourceError = `File watcher: ${error.message}`;
    watcher?.close();
    watcher = null;
    watchedDirectory = null;
    publish();
    schedule();
  });
}

async function refresh() {
  if (stopped) return;
  if (scanning) { requested = true; return; }
  scanning = true;
  let handle;
  let more = false;
  let dirty = false;
  const previousSource = state.source;
  const previousError = state.sourceError;
  try {
    await attachWatcher();
    handle = await open(filename, 'r');
    const info = await handle.stat();
    if (!info.isFile()) throw new Error('events.jsonl is not a regular file');
    const nextIdentity = `${info.dev}:${info.ino}:${info.birthtimeMs}`;
    if ((identity !== null && identity !== nextIdentity) || info.size < offset) {
      reset();
      dirty = true;
    }
    identity = nextIdentity;
    state.source = 'Following events.jsonl';
    state.sourceError = null;
    // At most 1 MiB per turn; resume by byte offset without rereading the log.
    const buffer = info.size > offset ? Buffer.allocUnsafe(chunkBytes) : null;
    for (let count = 0; count < 4 && offset < info.size; count++) {
      const { bytesRead } = await handle.read(buffer, 0, Math.min(chunkBytes, info.size - offset), offset);
      if (!bytesRead) break;
      offset += bytesRead;
      state.lastActivity = info.mtimeMs;
      consumeChunk(buffer.subarray(0, bytesRead));
      dirty = true;
    }
    more = offset < info.size;
  } catch (error) {
    if (error.code === 'ENOENT') {
      state.source = 'Waiting for producer to create events.jsonl';
      state.sourceError = null;
      if (identity !== null) {
        identity = null;
        reset();
        dirty = true;
      }
    } else {
      state.source = 'Unable to read benchmark events';
      state.sourceError = error.message;
    }
  } finally {
    if (handle) {
      try { await handle.close(); }
      catch (error) { state.sourceError = `Closing events.jsonl: ${error.message}`; }
    }
    scanning = false;
    if (!stopped && (dirty || previousSource !== state.source || previousError !== state.sourceError)) publish();
    if (more || requested) {
      requested = false;
      schedule(10);
    }
  }
}

const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ZipEnhancer native benchmark</title>
<style>
:root{color-scheme:light dark;font:15px/1.5 system-ui,sans-serif;background:#10151d;color:#e1e7ef}
*{box-sizing:border-box}body{max-width:1400px;margin:32px auto;padding:0 24px}h1{font-size:27px;margin-bottom:4px}h2{font-size:19px;margin-top:0}p{margin:8px 0}small,.muted{color:#aab8ca}code{overflow-wrap:anywhere}section{background:#18212d;border:1px solid #344052;border-radius:10px;padding:20px;margin:20px 0}.warning{border-left:4px solid #e9bc63;padding:10px 16px;background:#30291e}.error{color:#ffb0b0;white-space:pre-wrap}.good{color:#85e4aa}.slow{color:#ffd088}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}.label{color:#aab8ca;font-size:13px}.value{font-size:20px;overflow-wrap:anywhere}#message{white-space:pre-wrap}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{border-bottom:1px solid #344052;text-align:left;padding:10px;vertical-align:top}th{white-space:nowrap}td:first-child{min-width:180px;overflow-wrap:anywhere}details{min-width:240px}summary{cursor:pointer}dl{display:grid;grid-template-columns:minmax(120px,1fr) minmax(100px,1fr);gap:5px 14px;margin:8px 0}dt,dd{margin:0;overflow-wrap:anywhere}dt{color:#aab8ca}li{margin:12px 0;overflow-wrap:anywhere}footer{margin:24px 0;color:#aab8ca}#connection{font-weight:600}time{font-size:13px;color:#aab8ca}
</style></head><body>
<header><h1>ZipEnhancer · native GPU benchmark</h1><p class="muted">Warm end-to-end target: ≤ 1.000 second for both speaker tracks of the complete minute-long task. No cache hits. Read-only experimental results.</p><p id="connection" role="status">Connecting…</p></header>
<p class="warning">Numeric parity is not perceptual speech-quality proof. Timing alone does not establish acceptable enhancement quality.</p>
<section aria-labelledby="currentTitle"><h2 id="currentTitle">Current activity</h2><div class="cards"><div><div class="label">Candidate</div><div id="candidate" class="value">Not reported</div></div><div><div class="label">Latest event</div><div id="kind" class="value">Waiting</div></div><div><div class="label">Last file activity</div><div id="elapsed" class="value">No events observed</div></div></div><p id="message"></p><p id="source" class="muted"></p><p id="sourceError" class="error" role="alert"></p><p id="readerWarning" class="error"></p><h3>Reported progress / metrics</h3><div id="progress">No progress reported. No ETA is estimated.</div><p class="muted">Only producer-reported values are shown; a result event is not an assumed campaign completion.</p></section>
<section><h2>Completed candidate comparisons</h2><p id="counts" class="muted"></p><div class="scroll"><table><thead><tr><th>Candidate / task</th><th>Warm total (s)</th><th>vs 1 s target</th><th>Network (s)</th><th>DSP (s)</th><th>Precision</th><th>Batch</th><th>Numerical / other metrics</th></tr></thead><tbody id="results"></tbody></table></div><p id="empty">No completed results reported.</p></section>
<section><h2>Failures</h2><p id="failureCount" class="muted"></p><p id="noFailures">No error events reported.</p><ol id="failures"></ol></section>
<footer><div>Source: <code id="directory"></code></div><div id="recordCount"></div><div>No audio, transcripts, or tensors are served. Missing measurements are shown as —.</div></footer>
<script>
const byId = id => document.getElementById(id);
let lastActivity = null;
function node(tag, text, className) { const element = document.createElement(tag); if (text !== undefined) element.textContent = text; if (className) element.className = className; return element; }
function metricList(metrics) {
  const list = node('dl');
  for (const [key, value] of Object.entries(metrics)) { list.append(node('dt', key), node('dd', String(value))); }
  return list;
}
function seconds(value) { return typeof value === 'number' ? value.toFixed(4) : '—'; }
function matching(metrics, pattern) {
  const values = Object.entries(metrics).filter(([key]) => pattern.test(key));
  return values.length ? values.map(([key, value]) => key + ': ' + value).join('; ') : '—';
}
function tick() {
  if (lastActivity === null) { byId('elapsed').textContent = 'No events observed'; return; }
  const elapsed = Math.max(0, Math.floor((Date.now() - lastActivity) / 1000));
  byId('elapsed').textContent = elapsed < 60 ? elapsed + 's ago' : Math.floor(elapsed / 60) + 'm ' + elapsed % 60 + 's ago';
}
function render(state) {
  const current = state.current;
  lastActivity = state.lastActivity;
  tick();
  byId('candidate').textContent = current?.candidate || 'Not reported';
  byId('kind').textContent = current?.kind || 'Waiting';
  byId('message').textContent = current?.message || '';
  byId('source').textContent = state.source + (current?.time ? ' · Event time: ' + current.time : '');
  byId('sourceError').textContent = state.sourceError || '';
  byId('readerWarning').textContent = state.readerWarning || '';
  const metrics = current?.metrics || {};
  byId('progress').replaceChildren(Object.keys(metrics).length ? metricList(metrics) : node('p', 'No progress reported. No ETA is estimated.'));
  const rows = document.createDocumentFragment();
  for (const result of [...state.results].reverse()) {
    const row = node('tr');
    const m = result.metrics;
    const total = m.totalSeconds;
    const validTotal = typeof total === 'number' && total >= 0;
    const candidate = node('td');
    candidate.append(node('strong', result.candidate), node('p', matching(m, /recording|inputName/i), 'muted'));
    if (result.time) candidate.append(node('time', result.time));
    if (result.message) candidate.append(node('p', result.message));
    row.append(candidate, node('td', seconds(total)));
    row.append(node('td', validTotal ? (total <= 1 ? 'Meets target' : 'Over target') + ' (' + total.toFixed(3) + '×)' : 'Not reported', validTotal ? (total <= 1 ? 'good' : 'slow') : 'muted'));
    row.append(node('td', seconds(m.networkSeconds)), node('td', seconds(m.dspSeconds)));
    row.append(node('td', matching(m, /precision|dtype/i)), node('td', matching(m, /batch/i)));
    const errors = matching(m, /error|diff|rmse|mae|snr|parity|mse|numerical/i);
    const detailCell = node('td');
    detailCell.append(node('p', errors));
    if (Object.keys(m).length) {
      const details = node('details');
      details.append(node('summary', 'Reported metrics'), metricList(m));
      detailCell.append(details);
    } else detailCell.textContent = '—';
    row.append(detailCell);
    rows.append(row);
  }
  byId('results').replaceChildren(rows);
  byId('empty').hidden = state.results.length > 0;
  byId('counts').textContent = state.resultCount + ' result events; showing latest ' + state.results.length + ' (retention ' + state.retention.results + '). Each benchmark result covers both full speaker tracks.';
  const failures = document.createDocumentFragment();
  for (const failure of [...state.failures].reverse()) {
    const item = node('li');
    item.append(node('strong', failure.candidate), node('p', failure.message || 'Error event without a message', 'error'));
    if (failure.time) item.append(node('time', failure.time));
    if (Object.keys(failure.metrics).length) item.append(metricList(failure.metrics));
    failures.append(item);
  }
  byId('failures').replaceChildren(failures);
  byId('noFailures').hidden = state.failureCount > 0;
  byId('failureCount').textContent = state.failureCount + ' error events; showing latest ' + state.failures.length + ' (retention ' + state.retention.failures + ').';
  byId('directory').textContent = state.directory;
  byId('recordCount').textContent = state.records + ' complete valid records · ' + state.malformedRecords + ' skipped records · ' + state.resets + ' log resets';
}
const events = new EventSource('/events');
events.onopen = () => { byId('connection').textContent = 'Live connection'; byId('connection').className = 'good'; };
events.onmessage = event => { render(JSON.parse(event.data)); };
events.onerror = () => { byId('connection').textContent = 'Disconnected — retrying; displayed results may be stale'; byId('connection').className = 'error'; };
setInterval(tick, 1000);
</script></body></html>`;

const server = createServer((request, response) => {
  response.setHeader('Cache-Control', 'no-store');
  response.setHeader('X-Content-Type-Options', 'nosniff');
  response.setHeader('Content-Security-Policy', "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'");
  // Binding to loopback is not alone sufficient against DNS rebinding.
  if (request.headers.host !== `127.0.0.1:${port}` && request.headers.host !== `localhost:${port}`) {
    response.writeHead(403).end();
    return;
  }
  if (request.method !== 'GET') { response.writeHead(405, { Allow: 'GET' }).end(); return; }
  if (request.url === '/events') {
    response.writeHead(200, { 'Content-Type': 'text/event-stream; charset=utf-8', Connection: 'keep-alive' });
    response.write(`data: ${JSON.stringify(state)}\n\n`);
    clients.add(response);
    response.on('drain', () => {
      if (pendingClients.has(response)) {
        pendingClients.delete(response);
        response.write(`data: ${JSON.stringify(state)}\n\n`);
      }
    });
    response.on('close', () => clients.delete(response));
    response.on('error', () => clients.delete(response));
  } else if (request.url === '/status') {
    response.setHeader('Content-Type', 'application/json; charset=utf-8');
    response.end(JSON.stringify(state));
  } else if (request.url === '/') {
    response.setHeader('Content-Type', 'text/html; charset=utf-8');
    response.end(html);
  } else response.writeHead(404).end();
});

await refresh();
await new Promise((resolve, reject) => {
  server.once('error', reject);
  server.listen(port, '127.0.0.1', resolve);
});
// Poll metadata only: unchanged logs are never reread, even when very large.
const pollTimer = setInterval(() => schedule(), 1000);
const heartbeatTimer = setInterval(() => {
  for (const response of clients) if (!response.writableNeedDrain) response.write(': keepalive\n\n');
}, 15000);
console.log(`Native benchmark progress: http://127.0.0.1:${port}/`);
for (const signal of ['SIGINT', 'SIGTERM']) process.once(signal, () => {
  stopped = true;
  watcher?.close();
  clearTimeout(timer);
  clearInterval(pollTimer);
  clearInterval(heartbeatTimer);
  for (const response of clients) response.end();
  server.close();
});
