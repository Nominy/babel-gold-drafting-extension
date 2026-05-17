const AUDIO_EXTENSION_PATTERN = /\.(?:aac|aif|aiff|flac|m4a|mp3|oga|ogg|opus|wav|webm)(?:[?#]|$)/i;
const AUDIO_PATH_HINT_PATTERN = /(?:^|[/?#=&._-])(?:audio|audio-recordings|recording|media|waveform|track)(?:[/?#=&._-]|$)/i;

export function isBlobUrl(value: string): boolean {
  return value.trim().toLowerCase().startsWith('blob:');
}

export function isLikelyAudioSource(url: string, mimeType = ''): boolean {
  if (mimeType.trim().toLowerCase().startsWith('audio/')) {
    return true;
  }

  let parsed: URL;
  try {
    parsed = new URL(url, window.location.href);
  } catch {
    return AUDIO_EXTENSION_PATTERN.test(url);
  }

  const pathAndSearch = `${parsed.pathname}${parsed.search}`;
  return AUDIO_EXTENSION_PATTERN.test(pathAndSearch) || AUDIO_PATH_HINT_PATTERN.test(pathAndSearch);
}
