import path from 'node:path';
import { watch } from 'node:fs';
import { readdir, readFile, stat } from 'node:fs/promises';
import { createServer } from 'node:http';

const directory = process.argv[2] && path.resolve(process.argv[2]);
const port = Number(process.argv[3] ?? 53926);
if (!directory || !Number.isInteger(port) || port < 1 || port > 65535) throw new Error('Usage: node scripts/watch-zip-kernel-profile.mjs PRIVATE_CAPTURE_DIR [PORT]');
let waitingForDirectory = false;
try { if (!(await stat(directory)).isDirectory()) throw new Error('Capture path is not a directory'); }
catch (error) { if (error.code !== 'ENOENT') throw error; waitingForDirectory = true; }
const clients = new Set();
let state = { directory, stage: 'Waiting for a checkpoint', total: 0, captured: 0, replayed: 0, sourceRuns: 0, latestArtifact: '', lastActivity: Date.now(), complete: false };
let timer, scanning = false, changed = false;
async function refresh() {
  if (scanning) { changed = true; return; }
  scanning = true;
  try {
    let names;
    try { names = await readdir(directory); }
    catch (error) {
      if (error.code !== 'ENOENT') throw error;
      state.stage = 'Waiting for campaign identity verification and output creation';
      for (const response of clients) response.write(`data: ${JSON.stringify(state)}\n\n`);
      return;
    }
    const manifests = names.filter(n => /^capture-manifest-\d+\.json$/.test(n)).sort((a, b) => Number(b.match(/\d+/)[0]) - Number(a.match(/\d+/)[0]));
    for (const name of manifests) {
      try {
        const value = JSON.parse(await readFile(path.join(directory, name), 'utf8'));
        state = { ...state, stage: value.stage, total: value.configurations.length, captured: value.configurations.filter(c => c.captureComplete).length, sourceRuns: value.audits.length };
        break;
      } catch { /* An artifact is published over several bounded writes; keep the last complete checkpoint. */ }
    }
    state.replayed = 0;
    for (const name of names.filter(n => /^replay-\d+\.json$/.test(n))) {
      try { const r = JSON.parse(await readFile(path.join(directory, name), 'utf8')); if (Array.isArray(r.comparisons)) state.replayed++; } catch { /* In-flight write. */ }
    }
    if (names.includes('report.json')) {
      try { const r = JSON.parse(await readFile(path.join(directory, 'report.json'), 'utf8')); state.complete = true; state.correctness = r.correctness; state.total = r.configurations; state.stage = r.correctness ? 'Complete — all replay outputs bit-exact' : 'Complete — output mismatches found'; } catch { /* In-flight write. */ }
    } else if (state.replayed) state.stage = 'Replaying captured kernels';
    if (names.includes('browser-log.json') && !state.complete) state.stage = 'Run ended before completion — inspect benchmark error';
    const payload = `data: ${JSON.stringify(state)}\n\n`;
    for (const response of clients) response.write(payload);
  } finally { scanning = false; if (changed) { changed = false; schedule(); } }
}
function schedule() { clearTimeout(timer); timer = setTimeout(() => { void refresh().catch(error => console.error(error.message)); }, 500); }
const watchRoot = waitingForDirectory ? path.dirname(directory) : directory;
const watcher = watch(watchRoot, { recursive: waitingForDirectory }, (event, filename) => {
  const relative = filename && String(filename);
  if (waitingForDirectory && relative && relative.split(/[\\/]/)[0] !== path.basename(directory)) return;
  if (relative) state.latestArtifact = path.basename(relative);
  state.lastActivity = Date.now(); schedule();
});
await refresh();
const html = `<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>ZipEnhancer benchmark progress</title><style>body{font:16px system-ui;max-width:780px;margin:48px auto;padding:0 24px;color:#20242b}h1{font-size:25px}progress{width:100%;height:24px}section{margin:28px 0}small,footer{color:#58616d}code{overflow-wrap:anywhere}#stage{font-size:19px;font-weight:600}</style><h1>ZipEnhancer benchmark progress</h1><p id="stage">Connecting…</p><section><label id="captureLabel">Capture</label><progress id="capture" max="1" value="0"></progress><small>Complete, verified configurations at the last checkpoint.</small></section><section><label id="replayLabel">Kernel replay</label><progress id="replay" max="1" value="0"></progress><small>Each configuration: warmups, alternating baseline/optimized GPU timings, bit-exact output comparison.</small></section><p id="runs"></p><p>Latest artifact: <code id="artifact"></code></p><p id="activity"></p><details><summary>Local output directory</summary><code id="directory"></code></details><footer><p>Read-only file watcher. No extra inference, no audio or transcript served. Capture and replay are separate phases; this is not an ETA. A quiet interval does not imply a stalled GPU.</p></footer><script>let latest;const el=id=>document.getElementById(id);new EventSource('/events').onmessage=e=>{latest=JSON.parse(e.data);el('stage').textContent=latest.stage;el('capture').max=el('replay').max=latest.total||1;el('capture').value=latest.captured;el('replay').value=latest.replayed;el('captureLabel').textContent='Captured configurations: '+latest.captured+' / '+latest.total;el('replayLabel').textContent='Replayed configurations: '+latest.replayed+' / '+latest.total;el('runs').textContent='Verified original inference windows: '+latest.sourceRuns;el('artifact').textContent=latest.latestArtifact;el('directory').textContent=latest.directory;};setInterval(()=>{if(latest)el('activity').textContent='Last artifact activity: '+Math.floor((Date.now()-latest.lastActivity)/1000)+' seconds ago';},1000);</script>`;
const server = createServer((request, response) => {
  response.setHeader('Cache-Control', 'no-store');
  if (request.url === '/events') { response.writeHead(200, { 'Content-Type': 'text/event-stream', Connection: 'keep-alive' }); response.write(`data: ${JSON.stringify(state)}\n\n`); clients.add(response); request.on('close', () => clients.delete(response)); }
  else if (request.url === '/status') { response.setHeader('Content-Type', 'application/json'); response.end(JSON.stringify(state)); }
  else if (request.url === '/') { response.setHeader('Content-Type', 'text/html'); response.end(html); }
  else { response.writeHead(404).end(); }
});
await new Promise((resolve, reject) => { server.once('error', reject); server.listen(port, '127.0.0.1', resolve); });
console.log(`Benchmark progress: http://127.0.0.1:${port}/`);
for (const signal of ['SIGINT', 'SIGTERM']) process.once(signal, () => { watcher.close(); clearTimeout(timer); for (const response of clients) response.end(); server.close(); });
