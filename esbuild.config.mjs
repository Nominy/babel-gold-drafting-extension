import { build, context } from 'esbuild';

const watch = process.argv.includes('--watch');

const shared = {
  bundle: true,
  minify: false,
  sourcemap: true,
  target: 'chrome114',
  format: 'iife',
  logLevel: 'info'
};

const tasks = [
  {
    ...shared,
    entryPoints: ['src/content/entry.ts'],
    outfile: 'dist/content/entry.js'
  },
  {
    ...shared,
    entryPoints: ['src/options/options.ts'],
    outfile: 'dist/options/options.js'
  }
];

if (watch) {
  const contexts = await Promise.all(tasks.map((options) => context(options)));
  await Promise.all(contexts.map((ctx) => ctx.watch()));
  console.log('Watching gold drafting extension bundles...');
} else {
  await Promise.all(tasks.map((options) => build(options)));
}
