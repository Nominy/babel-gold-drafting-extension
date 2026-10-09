import { mkdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import { chromium } from 'playwright';
import { setTimeout as delay } from 'node:timers/promises';

const options = {};
for (let index = 2; index < process.argv.length; index += 2) {
  const name = process.argv[index], value = process.argv[index + 1];
  if (!['--extension', '--grader', '--profile', '--coordinator', '--browser'].includes(name) || !value) throw new Error(`Invalid worker argument: ${name}`);
  options[name.slice(2)] = value;
}
if (!options.extension || !options.grader || !options.profile || !options.coordinator) throw new Error('Supply --extension BUILT_GOLD --grader BUILT_REVIEW_GRADER --profile DEDICATED_PROFILE --coordinator HTTPS_OR_LOOPBACK_URL [--browser CHROME]');
const endpoint = new URL(options.coordinator);
if (endpoint.username || endpoint.password || endpoint.search || endpoint.hash ||
    !(endpoint.protocol === 'https:' || endpoint.protocol === 'http:' && ['localhost', '127.0.0.1', '[::1]'].includes(endpoint.hostname))) {
  throw new Error('The coordinator must be HTTPS or HTTP loopback, without credentials/query/fragment.');
}
const extension = path.resolve(options.extension), grader = path.resolve(options.grader), profile = path.resolve(options.profile);
const manifest = JSON.parse(await readFile(path.join(extension, 'manifest.json'), 'utf8'));
if (manifest.name !== 'Babel Gold Drafting') throw new Error('The worker requires the built Gold Drafting extension.');
const graderManifest = JSON.parse(await readFile(path.join(grader, 'manifest.json'), 'utf8'));
if (graderManifest.name !== 'Babel Review Grader' || !graderManifest.key || !graderManifest.background?.service_worker) throw new Error('The worker requires the updated Review Grader extension.');
await mkdir(profile, { recursive: true });
const context = await chromium.launchPersistentContext(profile, {
  headless: true, ...(options.browser ? { executablePath: options.browser } : {}),
  ignoreDefaultArgs: ['--disable-extensions'],
  args: [`--disable-extensions-except=${extension},${grader}`, `--load-extension=${extension},${grader}`],
});
let stopping = false;
const stop = () => { stopping = true; };
process.on('SIGINT', stop); process.on('SIGTERM', stop);
context.on('close', stop);
try {
  const isGold = worker => new URL(worker.url()).pathname === `/${manifest.background.service_worker}`;
  const background = context.serviceWorkers().find(isGold) ?? await context.waitForEvent('serviceworker', { predicate: isGold, timeout: 30000 });
  const page = await context.newPage();
  await page.goto(`chrome-extension://${new URL(background.url()).host}/options.html`);
  await page.evaluate(async coordinator => {
    await chrome.storage.local.set({ babel_gold_drafting_settings: {
      mode: 'local', backendBaseUrl: 'https://reviewgen.ovh', openRouterApiKey: '', localModelsEnabled: false,
      volunteerInferenceEnabled: true, l0DontRunLlm: true, l0ReplacementPreviewEnabled: false,
      audioInputEnabled: true, l0CustomBaseUrl: coordinator,
    } });
  }, endpoint.href.replace(/\/+$/, ''));
  let ready = false, previous = '', startupDeadline = Date.now() + 60000;
  while (!stopping) {
    const status = await page.evaluate(() => chrome.runtime.sendMessage({ type: 'babel-l0-volunteer', target: 'background', action: 'status' }));
    if (!status || typeof status.state !== 'string') throw new Error('The GPU worker returned no lifecycle status.');
    if (status.state !== previous) {
      previous = status.state;
      console.log(JSON.stringify({ time: new Date().toISOString(), state: status.state, ...(status.detail ? { detail: status.detail } : {}) }));
    }
    if (!ready && ['connected', 'busy'].includes(status.state)) {
      ready = true;
      console.log('ZipEnhancer swarm GPU worker connected');
    }
    if (!ready && Date.now() > startupDeadline) throw new Error(`GPU worker admission failed: ${status.detail || status.state}`);
    await delay(1000);
  }
  // Shutdown is limited to this dedicated browser/profile, never the user's Chrome.
  await page.evaluate(() => chrome.runtime.sendMessage({ type: 'babel-l0-volunteer', target: 'offscreen', action: 'stop' })).catch(() => {});
} finally {
  process.removeListener('SIGINT', stop); process.removeListener('SIGTERM', stop);
  await context.close();
}
