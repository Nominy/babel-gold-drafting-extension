// Private stdin/stdout RPC. No HTTP listener and no remote browser-control surface.
import { createInterface } from 'node:readline';
import { mkdir, readFile, open, unlink } from 'node:fs/promises';
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
  executablePath: process.env.BABEL_INFERENCE_BROWSER_EXECUTABLE,
  ignoreDefaultArgs: ['--disable-extensions', '--enable-unsafe-swiftshader'],
  args: [`--disable-extensions-except=${extension}`, `--load-extension=${extension}`, '--enable-unsafe-webgpu',
    ...(process.platform === 'win32' ? ['--use-angle=d3d11'] :
      process.platform === 'linux' ? ['--use-angle=vulkan', '--enable-features=Vulkan', '--disable-vulkan-surface',
        '--disable-gpu-compositing'] : [])]
});
try {
  const worker = context.serviceWorkers()[0] ?? await context.waitForEvent('serviceworker');
  const page = await context.newPage();
  await page.goto(`chrome-extension://${new URL(worker.url()).host}/inference.html`);
  await page.waitForFunction(() => Boolean(globalThis.babelInference));
  let active;
  await page.exposeBinding('babelBackendEvent', async ({ page: source, frame }, id, event) => {
    const request = active;
    if (source !== page || frame !== page.mainFrame() || !request || id !== request.id) throw new Error('Backend event lost request ownership.');
    if (event?.progress) {
      process.stdout.write(`${JSON.stringify({ id, progress: event.progress })}\n`);
      return;
    }
    const output = request.outputs.get(event?.fieldName);
    if (!output || event.offset !== output.offset || typeof event.audio !== 'string' ||
        !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(event.audio)) throw new Error('Invalid backend audio ownership or offset.');
    const bytes = Buffer.from(event.audio, 'base64');
    if (!bytes.length || bytes.length > 512 * 1024 || output.offset + bytes.length > output.expected) throw new Error('Backend audio exceeds its original clock.');
    await output.file.writeFile(bytes);
    output.offset += bytes.length;
  });
  const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
  for await (const line of lines) {
    let command;
    try {
      command = JSON.parse(line);
      if (typeof command.id !== 'string' || !/^[a-f0-9]{32}$/.test(command.id)) throw new Error('Invalid backend RPC request ID.');
      let result;
      if (command.operation === 'prepare') result = await page.evaluate(async () => {
        const enhancement = await globalThis.babelInference.prepare();
        return { ok: true, release: globalThis.babelInference.release, provider: 'webgpu', enhancement };
      });
      else if (command.operation === 'enhance') {
        active = { id: command.id, outputs: new Map() };
        const audio = {};
        try {
          if (command.payload.tracks.length !== 2) throw new Error('Backend enhancement requires a pair.');
          for (const track of command.payload.tracks) {
            if (!['audio:1', 'audio:2'].includes(track.fieldName) || active.outputs.has(track.fieldName) ||
                !Number.isSafeInteger(track.frameCount) || track.frameCount <= 0) throw new Error('Invalid backend track.');
            const destination = command.outputs[track.fieldName];
            const file = await open(destination, 'wx');
            active.outputs.set(track.fieldName, { file, destination, offset: 0, expected: 44 + track.frameCount * 2 });
            audio[track.fieldName] = (await readFile(command.paths[track.fieldName])).toString('base64');
          }
          result = await page.evaluate(({ id, payload, audio }) =>
            globalThis.babelInference.enhance(payload, audio, event => globalThis.babelBackendEvent(id, event)),
          { id: command.id, payload: command.payload, audio });
          if ([...active.outputs.values()].some(output => output.offset !== output.expected)) throw new Error('Backend returned incomplete audio.');
        } catch (error) {
          for (const output of active.outputs.values()) {
            await output.file.close();
            await unlink(output.destination);
          }
          throw error;
        } finally {
          for (const output of active.outputs.values()) await output.file.close();
          active = undefined;
        }
      } else if (command.operation === 'transcribe') {
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
