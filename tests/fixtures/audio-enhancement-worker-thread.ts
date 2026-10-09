// Install worker globals before the statically imported production entry executes.
import './audio-enhancement-worker-globals';
import '../../src/workers/audio-enhancement';
