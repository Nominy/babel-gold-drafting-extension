import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { captureAudioTracksForDrafting, installAudioRequestCapture } from '../src/core/audio-cues';
import { AUDIO_RESPONSE_MESSAGE_TYPE, AUDIO_SOURCE_MESSAGE_TYPE, PAGE_TASK_ID_ATTRIBUTE } from '../src/core/audio-intercept-protocol';

function installDom(html: string) {
  const dom = new JSDOM(html, { url: 'https://dashboard.babel.audio/transcription/RU-transcription?jobId=job-42' });
  Object.assign(globalThis, {
    window: dom.window,
    document: dom.window.document,
    HTMLMediaElement: dom.window.HTMLMediaElement,
    Blob: dom.window.Blob,
    fetch: async (url: string) =>
      ({
        ok: true,
        status: 200,
        blob: async () => new dom.window.Blob([`bytes:${url}`], { type: 'audio/webm' })
      })
  });
  return dom;
}

interface FetchCall {
  url: string;
  credentials?: RequestCredentials;
}

function recordFetches(dom: JSDOM): FetchCall[] {
  const calls: FetchCall[] = [];
  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    calls.push({ url, credentials: init?.credentials });
    return {
      ok: true,
      status: 200,
      blob: async () => new dom.window.Blob([`bytes:${url}`], { type: 'audio/wav' })
    } as Response;
  };
  return calls;
}

test('captureAudioTracksForDrafting includes credentials only for same-origin DOM audio sources', async () => {
  const signedS3Url =
    'https://davidai-audio-recordings.s3.us-east-2.amazonaws.com/transcription-chunks/prod/job-42/chunk/speaker-1.wav?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=test%2F20260830%2Fus-east-2%2Fs3%2Faws4_request&X-Amz-Signature=abc123';
  const dom = installDom(`
    <audio src="https://dashboard.babel.audio/api/files/source-a"></audio>
    <audio src="${signedS3Url}"></audio>
  `);
  const fetchCalls = recordFetches(dom);

  await captureAudioTracksForDrafting();

  assert.deepEqual(fetchCalls, [
    { url: 'https://dashboard.babel.audio/api/files/source-a', credentials: 'include' },
    { url: signedS3Url, credentials: 'omit' }
  ]);
});

test('captureAudioTracksForDrafting includes credentials only for same-origin discovered audio sources', async () => {
  const signedS3Url =
    'https://davidai-audio-recordings.s3.us-east-2.amazonaws.com/transcription-chunks/prod/job-42/chunk/speaker-2.wav?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=test%2F20260830%2Fus-east-2%2Fs3%2Faws4_request&X-Amz-Signature=def456';
  const dom = installDom('<main></main>');
  const fetchCalls = recordFetches(dom);
  installAudioRequestCapture();

  for (const source of [
    {
      url: 'https://dashboard.babel.audio/api/files/source-a',
      trackId: 'track-a',
      speakerKey: 'speaker-1'
    },
    {
      url: signedS3Url,
      trackId: 'track-b',
      speakerKey: 'speaker-2'
    }
  ]) {
    window.dispatchEvent(
      new dom.window.MessageEvent('message', {
        source: window,
        data: {
          type: AUDIO_SOURCE_MESSAGE_TYPE,
          ...source,
          mimeType: 'audio/wav',
          discoveredAt: 124
        }
      })
    );
  }

  await captureAudioTracksForDrafting();

  assert.deepEqual(fetchCalls, [
    { url: 'https://dashboard.babel.audio/api/files/source-a', credentials: 'include' },
    { url: signedS3Url, credentials: 'omit' }
  ]);
});

test('captureAudioTracksForDrafting fetches every distinct audio source on the page', async () => {
  installDom(`
    <audio src="https://dashboard.babel.audio/a1.webm"></audio>
    <audio src="https://dashboard.babel.audio/a2.webm"></audio>
    <audio src="https://dashboard.babel.audio/a1.webm"></audio>
  `);

  const tracks = await captureAudioTracksForDrafting();

  assert.equal(tracks.length, 2);
  assert.deepEqual(
    tracks.map((track) => ({ trackId: track.trackId, source: track.source, mimeType: track.blob.type })),
    [
      { trackId: 'audio-1', source: 'https://dashboard.babel.audio/a1.webm', mimeType: 'audio/webm' },
      { trackId: 'audio-2', source: 'https://dashboard.babel.audio/a2.webm', mimeType: 'audio/webm' }
    ]
  );
});

test('captureAudioTracksForDrafting includes audio bytes intercepted from page requests', async () => {
  const dom = installDom('<main></main>');
  installAudioRequestCapture();

  window.dispatchEvent(
    new dom.window.MessageEvent('message', {
      source: window,
      data: {
        type: AUDIO_RESPONSE_MESSAGE_TYPE,
        url: 'https://dashboard.babel.audio/api/recordings/r1/audio',
        mimeType: 'audio/wav',
        trackId: 'track-a',
        speakerKey: 'speaker-1',
        trackLabel: 'Speaker 1',
        source: 'fetch',
        capturedAt: 123,
        bytes: new Uint8Array([1, 2, 3, 4]).buffer
      }
    })
  );

  const tracks = await captureAudioTracksForDrafting();

  assert.equal(tracks.length, 1);
  assert.equal(tracks[0]?.trackId, 'track-a');
  assert.equal(tracks[0]?.source, 'https://dashboard.babel.audio/api/recordings/r1/audio');
  assert.equal(tracks[0]?.speakerKey, 'speaker-1');
  assert.equal(tracks[0]?.trackLabel, 'Speaker 1');
  assert.equal(tracks[0]?.mimeType, 'audio/wav');
  assert.equal(tracks[0]?.blob.size, 4);
});

test('captureAudioTracksForDrafting fetches lane-mapped audio sources discovered in page world', async () => {
  const dom = installDom('<main></main>');
  installAudioRequestCapture();

  window.dispatchEvent(
    new dom.window.MessageEvent('message', {
      source: window,
      data: {
        type: AUDIO_SOURCE_MESSAGE_TYPE,
        url: 'https://dashboard.babel.audio/api/files/source-a',
        mimeType: 'audio/webm',
        trackId: 'track-a',
        speakerKey: 'speaker-1',
        trackLabel: 'Speaker 1',
        discoveredAt: 124
      }
    })
  );

  const tracks = await captureAudioTracksForDrafting();

  assert.equal(tracks.length, 1);
  assert.equal(tracks[0]?.trackId, 'track-a');
  assert.equal(tracks[0]?.speakerKey, 'speaker-1');
  assert.equal(tracks[0]?.trackLabel, 'Speaker 1');
  assert.equal(tracks[0]?.source, 'https://dashboard.babel.audio/api/files/source-a');
  assert.equal(tracks[0]?.blob.type, 'audio/webm');
});

test('captureAudioTracksForDrafting does not retain discovered audio sources across SPA tasks', async () => {
  const dom = installDom('<main></main>');
  installAudioRequestCapture();

  for (const message of [
    {
      url: 'https://dashboard.babel.audio/api/files/task-a-speaker-1',
      trackId: 'task-a-speaker-1',
      speakerKey: 'task-a-speaker-1',
      trackLabel: 'Speaker 1'
    },
    {
      url: 'https://dashboard.babel.audio/api/files/task-a-speaker-2',
      trackId: 'task-a-speaker-2',
      speakerKey: 'task-a-speaker-2',
      trackLabel: 'Speaker 2'
    }
  ]) {
    window.dispatchEvent(
      new dom.window.MessageEvent('message', {
        source: window,
        data: {
          type: AUDIO_SOURCE_MESSAGE_TYPE,
          ...message,
          mimeType: 'audio/webm',
          discoveredAt: 100
        }
      })
    );
  }

  const firstTracks = await captureAudioTracksForDrafting();
  assert.deepEqual(
    firstTracks.map((track) => track.source),
    [
      'https://dashboard.babel.audio/api/files/task-a-speaker-1',
      'https://dashboard.babel.audio/api/files/task-a-speaker-2'
    ]
  );

  for (const message of [
    {
      url: 'https://dashboard.babel.audio/api/files/task-b-speaker-1',
      trackId: 'task-b-speaker-1',
      speakerKey: 'task-b-speaker-1',
      trackLabel: 'Speaker 1'
    },
    {
      url: 'https://dashboard.babel.audio/api/files/task-b-speaker-2',
      trackId: 'task-b-speaker-2',
      speakerKey: 'task-b-speaker-2',
      trackLabel: 'Speaker 2'
    }
  ]) {
    window.dispatchEvent(
      new dom.window.MessageEvent('message', {
        source: window,
        data: {
          type: AUDIO_SOURCE_MESSAGE_TYPE,
          ...message,
          mimeType: 'audio/webm',
          discoveredAt: 200
        }
      })
    );
  }

  const secondTracks = await captureAudioTracksForDrafting();

  assert.deepEqual(
    secondTracks.map((track) => track.source),
    [
      'https://dashboard.babel.audio/api/files/task-b-speaker-1',
      'https://dashboard.babel.audio/api/files/task-b-speaker-2'
    ]
  );
});

test('captureAudioTracksForDrafting ignores stale intercepted audio once current lane sources are rediscovered', async () => {
  const dom = installDom('<main></main>');
  const capturePromise = captureAudioTracksForDrafting();

  dom.window.setTimeout(() => {
    window.dispatchEvent(
      new dom.window.MessageEvent('message', {
        source: window,
        data: {
          type: AUDIO_RESPONSE_MESSAGE_TYPE,
          url: 'https://dashboard.babel.audio/api/files/task-a-speaker-1',
          mimeType: 'audio/webm',
          trackId: 'task-a-speaker-1',
          speakerKey: 'task-a-speaker-1',
          trackLabel: 'Speaker 1',
          source: 'fetch',
          capturedAt: 100,
          bytes: new Uint8Array([1, 1, 1]).buffer
        }
      })
    );

    for (const message of [
      {
        url: 'https://dashboard.babel.audio/api/files/task-b-speaker-1',
        trackId: 'task-b-speaker-1',
        speakerKey: 'task-b-speaker-1',
        trackLabel: 'Speaker 1'
      },
      {
        url: 'https://dashboard.babel.audio/api/files/task-b-speaker-2',
        trackId: 'task-b-speaker-2',
        speakerKey: 'task-b-speaker-2',
        trackLabel: 'Speaker 2'
      }
    ]) {
      window.dispatchEvent(
        new dom.window.MessageEvent('message', {
          source: window,
          data: {
            type: AUDIO_SOURCE_MESSAGE_TYPE,
            ...message,
            mimeType: 'audio/webm',
            discoveredAt: 200
          }
        })
      );
    }
  }, 0);

  const tracks = await capturePromise;

  assert.deepEqual(
    tracks.map((track) => track.source),
    [
      'https://dashboard.babel.audio/api/files/task-b-speaker-1',
      'https://dashboard.babel.audio/api/files/task-b-speaker-2'
    ]
  );
});

test('captureAudioTracksForDrafting drops unmapped captures and keeps one source per speaker lane', async () => {
  const dom = installDom('<main></main>');
  installAudioRequestCapture();

  const messages = [
    {
      url: 'https://clerk.babel.audio/v1/environment?__clerk_api_version=2025-11-10',
      mimeType: 'application/json',
      capturedAt: 1,
      bytes: [1]
    },
    {
      url: 'https://dashboard.babel.audio/api/trpc/transcriptions.getReviewActionDataById?batch=1',
      mimeType: 'application/json',
      capturedAt: 2,
      bytes: [2]
    },
    {
      url: 'https://davidai-audio-recordings.s3.us-east-2.amazonaws.com/transcription-chunks/prod/job/chunk/speaker-2.wav?X-Amz-Signature=test',
      mimeType: 'audio/wav',
      trackId: 'speaker-2',
      speakerKey: 'speaker-2',
      trackLabel: 'Speaker 2',
      capturedAt: 3,
      bytes: [3, 3, 3]
    },
    {
      url: 'blob:https://dashboard.babel.audio/blob-speaker-2',
      mimeType: 'audio/wav',
      trackId: 'speaker-2',
      speakerKey: 'speaker-2',
      trackLabel: 'Speaker 2',
      capturedAt: 4,
      bytes: [4, 4, 4]
    },
    {
      url: 'https://davidai-audio-recordings.s3.us-east-2.amazonaws.com/transcription-chunks/prod/job/chunk/speaker-1.wav?X-Amz-Signature=test',
      mimeType: 'audio/wav',
      trackId: 'speaker-1',
      speakerKey: 'speaker-1',
      trackLabel: 'Speaker 1',
      capturedAt: 5,
      bytes: [5, 5, 5]
    },
    {
      url: 'blob:https://dashboard.babel.audio/blob-speaker-1',
      mimeType: 'audio/wav',
      trackId: 'speaker-1',
      speakerKey: 'speaker-1',
      trackLabel: 'Speaker 1',
      capturedAt: 6,
      bytes: [6, 6, 6]
    }
  ];

  for (const message of messages) {
    window.dispatchEvent(
      new dom.window.MessageEvent('message', {
        source: window,
        data: {
          type: AUDIO_RESPONSE_MESSAGE_TYPE,
          source: 'fetch',
          ...message,
          bytes: new Uint8Array(message.bytes).buffer
        }
      })
    );
  }

  const tracks = await captureAudioTracksForDrafting();

  assert.equal(tracks.length, 2);
  assert.deepEqual(
    tracks.map((track) => ({
      trackId: track.trackId,
      speakerKey: track.speakerKey,
      trackLabel: track.trackLabel,
      source: track.source
    })),
    [
      {
        trackId: 'speaker-2',
        speakerKey: 'speaker-2',
        trackLabel: 'Speaker 2',
        source:
          'https://davidai-audio-recordings.s3.us-east-2.amazonaws.com/transcription-chunks/prod/job/chunk/speaker-2.wav?X-Amz-Signature=test'
      },
      {
        trackId: 'speaker-1',
        speakerKey: 'speaker-1',
        trackLabel: 'Speaker 1',
        source:
          'https://davidai-audio-recordings.s3.us-east-2.amazonaws.com/transcription-chunks/prod/job/chunk/speaker-1.wav?X-Amz-Signature=test'
      }
    ]
  );
});

test('captureAudioTracksForDrafting treats speaker lane as the duplicate key when track ids differ', async () => {
  const dom = installDom('<main></main>');
  installAudioRequestCapture();

  for (const message of [
    {
      url: 'https://dashboard.babel.audio/api/files/speaker-1-source',
      mimeType: 'audio/wav',
      trackId: 'volatile-source-id',
      speakerKey: 'speaker-1',
      trackLabel: 'Speaker 1',
      capturedAt: 1,
      bytes: [1, 1, 1]
    },
    {
      url: 'blob:https://dashboard.babel.audio/speaker-1-copy',
      mimeType: 'audio/wav',
      trackId: 'volatile-blob-id',
      speakerKey: 'speaker-1',
      trackLabel: 'Speaker 1',
      capturedAt: 2,
      bytes: [2, 2, 2]
    }
  ]) {
    window.dispatchEvent(
      new dom.window.MessageEvent('message', {
        source: window,
        data: {
          type: AUDIO_RESPONSE_MESSAGE_TYPE,
          source: 'fetch',
          ...message,
          bytes: new Uint8Array(message.bytes).buffer
        }
      })
    );
  }

  const tracks = await captureAudioTracksForDrafting();

  assert.deepEqual(
    tracks.map((track) => ({
      trackId: track.trackId,
      speakerKey: track.speakerKey,
      trackLabel: track.trackLabel,
      source: track.source
    })),
    [
      {
        trackId: 'volatile-source-id',
        speakerKey: 'speaker-1',
        trackLabel: 'Speaker 1',
        source: 'https://dashboard.babel.audio/api/files/speaker-1-source'
      }
    ]
  );
});

test('captureAudioTracksForDrafting skips DOM fallback audio when lane-mapped tracks already exist', async () => {
  const dom = installDom(`
    <audio src="blob:https://dashboard.babel.audio/blob-speaker-1"></audio>
    <audio src="blob:https://dashboard.babel.audio/blob-speaker-2"></audio>
  `);
  installAudioRequestCapture();
  const fetchedUrls: string[] = [];
  globalThis.fetch = async (input: RequestInfo | URL) => {
    const url = String(input);
    fetchedUrls.push(url);
    return {
      ok: true,
      status: 200,
      blob: async () => new dom.window.Blob([`bytes:${url}`], { type: 'audio/webm' })
    } as Response;
  };

  for (const message of [
    {
      url: 'https://dashboard.babel.audio/audio-speaker-1.webm',
      trackId: 'speaker-1',
      speakerKey: 'speaker-1',
      trackLabel: 'Speaker 1'
    },
    {
      url: 'https://dashboard.babel.audio/audio-speaker-2.webm',
      trackId: 'speaker-2',
      speakerKey: 'speaker-2',
      trackLabel: 'Speaker 2'
    }
  ]) {
    window.dispatchEvent(
      new dom.window.MessageEvent('message', {
        source: window,
        data: {
          type: AUDIO_SOURCE_MESSAGE_TYPE,
          ...message,
          mimeType: 'audio/webm',
          discoveredAt: Date.now()
        }
      })
    );
  }

  const tracks = await captureAudioTracksForDrafting();

  assert.equal(tracks.length, 2);
  assert.deepEqual(
    tracks.map((track) => track.trackId),
    ['speaker-1', 'speaker-2']
  );
  assert.deepEqual(fetchedUrls, [
    'https://dashboard.babel.audio/audio-speaker-1.webm',
    'https://dashboard.babel.audio/audio-speaker-2.webm'
  ]);
});

test('concurrent audio consumers cannot clear each other’s pending capture', async () => {
  const dom = installDom('<main></main>');
  installAudioRequestCapture();
  const flushes: Array<() => void> = [];
  dom.window.setTimeout = ((callback: () => void) => {
    flushes.push(callback);
    return flushes.length;
  }) as typeof dom.window.setTimeout;

  const firstCapture = captureAudioTracksForDrafting();
  const secondCapture = captureAudioTracksForDrafting();
  window.dispatchEvent(new dom.window.MessageEvent('message', {
    source: window,
    data: {
      type: AUDIO_RESPONSE_MESSAGE_TYPE,
      url: 'https://dashboard.babel.audio/audio/speaker-1.wav',
      mimeType: 'audio/wav',
      trackId: 'speaker-1',
      speakerKey: 'speaker-1',
      source: 'fetch',
      capturedAt: 1,
      bytes: new Uint8Array([1, 2, 3, 4]).buffer
    }
  }));

  flushes[0]!();
  const first = await firstCapture;
  flushes[1]!();
  const second = await secondCapture;
  for (const tracks of [first, second]) {
    assert.deepEqual(tracks.map((track) => [track.speakerKey, track.blob.size]), [['speaker-1', 4]]);
  }
});

test('unavailable lane URLs preserve captured lanes and allow a working alternate source', async () => {
  const dom = installDom('<main></main>');
  installAudioRequestCapture();
  const blobUrl = 'blob:https://dashboard.babel.audio/available-lane';
  globalThis.fetch = async (input: RequestInfo | URL) => ({
    ok: String(input) === blobUrl,
    status: String(input) === blobUrl ? 200 : 404,
    blob: async () => new dom.window.Blob(['audio'], { type: 'audio/wav' })
  }) as Response;

  for (const [url, speakerKey] of [
    ['https://dashboard.babel.audio/audio/speaker-2.wav', 'speaker-2'],
    ['https://dashboard.babel.audio/audio/speaker-1.wav', 'speaker-1'],
    [blobUrl, 'speaker-1']
  ]) {
    window.dispatchEvent(new dom.window.MessageEvent('message', {
      source: window,
      data: {
        type: AUDIO_SOURCE_MESSAGE_TYPE,
        url,
        speakerKey,
        discoveredAt: 1
      }
    }));
  }

  const tracks = await captureAudioTracksForDrafting();
  assert.deepEqual(tracks.map((track) => [track.speakerKey, track.source, track.blob.size]), [
    ['speaker-1', blobUrl, 5]
  ]);
});

test('rejected source requests and body reads preserve valid lanes and working alternates', async () => {
  const dom = installDom('<main></main>');
  installAudioRequestCapture();
  const revokedUrl = 'blob:https://dashboard.babel.audio/revoked';
  const unreadableUrl = 'blob:https://dashboard.babel.audio/unreadable';
  const workingUrl = 'blob:https://dashboard.babel.audio/working';
  window.dispatchEvent(new dom.window.MessageEvent('message', {
    source: window,
    data: {
      type: AUDIO_RESPONSE_MESSAGE_TYPE,
      url: 'https://dashboard.babel.audio/audio/speaker-1.wav',
      mimeType: 'audio/wav',
      speakerKey: 'speaker-1',
      source: 'fetch',
      capturedAt: 1,
      bytes: new Uint8Array([1, 2, 3, 4]).buffer
    }
  }));
  for (const [url, speakerKey] of [
    ['https://dashboard.babel.audio/audio/speaker-1.wav', 'speaker-1'],
    [revokedUrl, 'speaker-2'],
    [unreadableUrl, 'speaker-3'],
    [workingUrl, 'speaker-2']
  ]) {
    window.dispatchEvent(new dom.window.MessageEvent('message', {
      source: window,
      data: { type: AUDIO_SOURCE_MESSAGE_TYPE, url, speakerKey, discoveredAt: 1 }
    }));
  }
  globalThis.fetch = async (input: RequestInfo | URL) => {
    if (String(input) === revokedUrl) throw new TypeError('Failed to fetch');
    return {
      ok: true,
      status: 200,
      blob: async () => {
        if (String(input) === unreadableUrl) throw new TypeError('Response body stream failed');
        return new dom.window.Blob(['audio'], { type: 'audio/wav' });
      }
    } as Response;
  };

  const tracks = await captureAudioTracksForDrafting();
  assert.deepEqual(tracks.map((track) => [track.speakerKey, track.blob.size]), [
    ['speaker-1', 4],
    ['speaker-2', 5]
  ]);
});

test('SPA task changes isolate concurrent capture caches and reject stale completions', async () => {
  const dom = installDom('<main></main>');
  // The published identity can lag behind navigation until page-world refresh.
  dom.window.document.documentElement.setAttribute(PAGE_TASK_ID_ATTRIBUTE, 'task-a');
  installAudioRequestCapture();
  const flushes: Array<() => void> = [];
  dom.window.setTimeout = ((callback: () => void) => {
    flushes.push(callback);
    return flushes.length;
  }) as typeof dom.window.setTimeout;
  const taskAUrl = 'https://dashboard.babel.audio/audio/task-a.wav';
  const taskBUrl = 'https://dashboard.babel.audio/audio/task-b.wav';
  const announce = (url: string) => window.dispatchEvent(new dom.window.MessageEvent('message', {
    source: window,
    data: { type: AUDIO_SOURCE_MESSAGE_TYPE, url, speakerKey: 'speaker-1', discoveredAt: 1 }
  }));
  const response = () => ({
    ok: true,
    status: 200,
    blob: async () => new dom.window.Blob(['audio'], { type: 'audio/wav' })
  }) as Response;
  let finishOldFetch!: (response: Response) => void;
  let reachedOldFetch!: () => void;
  const oldFetchStarted = new Promise<void>((resolve) => { reachedOldFetch = resolve; });
  let taskAFetches = 0;
  globalThis.fetch = async (input: RequestInfo | URL) => {
    if (String(input) === taskAUrl && ++taskAFetches === 2) {
      return new Promise<Response>((resolve) => {
        finishOldFetch = resolve;
        reachedOldFetch();
      });
    }
    return response();
  };

  announce(taskAUrl);
  const firstA = captureAudioTracksForDrafting();
  const pendingA = captureAudioTracksForDrafting();
  flushes[0]!();
  assert.deepEqual((await firstA).map((track) => track.source), [taskAUrl]);
  flushes[1]!();
  await oldFetchStarted;

  dom.window.history.replaceState({}, '', '?jobId=task-b');
  announce(taskBUrl);
  const firstB = captureAudioTracksForDrafting();
  const pendingB = captureAudioTracksForDrafting();
  flushes[2]!();
  assert.deepEqual((await firstB).map((track) => track.source), [taskBUrl]);

  const staleCompletion = assert.rejects(pendingA, /Audio capture task changed/);
  finishOldFetch(response());
  await staleCompletion;
  flushes[3]!();
  assert.deepEqual((await pendingB).map((track) => track.source), [taskBUrl]);
  assert.equal(taskAFetches, 2);
});
