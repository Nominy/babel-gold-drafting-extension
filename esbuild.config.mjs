import { buildExtension, defineExtensionBuild } from '@nominy/babel-extension-build';

const watch = process.argv.includes('--watch');

const config = defineExtensionBuild({
  watch,
  sharedOptions: {
    minify: false,
    sourcemap: true,
    target: 'chrome114',
    format: 'iife',
    logLevel: 'info'
  },
  tasks: [
    {
      entryPoints: ['src/content/entry.ts'],
      outfile: 'dist/content/entry.js'
    },
    {
      entryPoints: ['src/content/audio-request-interceptor.ts'],
      outfile: 'dist/content/audio-request-interceptor.js'
    },
    {
      entryPoints: ['src/background/ai-broker.ts'],
      outfile: 'dist/background/ai-broker.js'
    },
    {
      entryPoints: ['src/options/options.ts'],
      outfile: 'dist/options/options.js'
    }
  ],
  watchMessage: 'Watching gold drafting extension bundles...'
});

await buildExtension(config);
