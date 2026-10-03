// Private stdin/stdout RPC. No HTTP listener and no remote browser-control surface.
import { createInterface } from 'node:readline';
import { mkdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
const root = path.dirname(fileURLToPath(import.meta.url));
const extension = path.join(root, 'extension');
const profile = process.env.BABEL_INFERENCE_PROFILE;
if (!profile) throw new Error('BABEL_INFERENCE_PROFILE must name a dedicated backend browser profile.');
await mkdir(profile, { recursive: true });
const context = await chromium.launchPersistentContext(profile, {
  channel: process.env.BABEL_INFERENCE_BROWSER_CHANNEL || 'chromium', headless: true,
  args: [`--disable-extensions-except=${extension}`, `--load-extension=${extension}`, '--enable-unsafe-webgpu',
    ...(process.platform === 'win32' ? ['--use-angle=d3d11'] : [])]
});
try {
  const worker = context.serviceWorkers()[0] ?? await context.waitForEvent('serviceworker');
  const page = await context.newPage();
  await page.goto(`chrome-extension://${new URL(worker.url()).host}/inference.html`);
  await page.waitForFunction(() => Boolean(globalThis.babelInference));
  const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
  for await (const line of lines) {
    let command;
    try {
      command = JSON.parse(line);
      let result;
      if (command.operation === 'prepare') result = await page.evaluate(async () => {
        await globalThis.babelInference.prepare();
        return { ok: true, release: globalThis.babelInference.release, provider: 'webgpu' };
      });
      else if (command.operation === 'transcribe') {
        // Paths come only from the validated FastAPI upload workspace over this private pipe.
        const audio = {};
        for (const track of command.payload.tracks) audio[track.fieldName] = (await readFile(command.paths[track.fieldName])).toString('base64');
        result = await page.evaluate(({ payload, audio }) => globalThis.babelInference.transcribe(payload, audio), { payload: command.payload, audio });
      } else if (command.operation === 'draft') result = await page.evaluate(({ timing, options }) => globalThis.babelInference.draft(timing, options?.preserveRows), command);
      else throw new Error('Unsupported backend RPC operation.');
      process.stdout.write(`${JSON.stringify({ id: command.id, result })}\n`);
    } catch (error) {
      process.stdout.write(`${JSON.stringify({ id: command?.id, error: String(error) })}\n`);
    }
  }
} finally { await context.close(); }
