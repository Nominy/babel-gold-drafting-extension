import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { captureAudioTracksForDrafting, captureOriginalAudioTracksForEnhancement, installAudioRequestCapture, isAudioSelectionReady } from '../src/core/audio-cues';
import {
  AUDIO_ENHANCEMENT_ACTIVE_REQUEST,
  AUDIO_ENHANCEMENT_ACTIVE_RESPONSE,
  AUDIO_ENHANCEMENT_PROTOCOL_VERSION,
  AUDIO_ENHANCEMENT_STATE_ATTRIBUTE,
  type ActiveAudioRequest,
  type ActiveAudioResponse,
  type AudioEnhancementState
} from '@nominy/babel-babel-runtime';
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

test('search and hash changes on the same task do not invalidate an in-flight capture', async () => {
  const dom = installDom('<main></main>');
  installAudioRequestCapture();
  const laneUrl = 'https://dashboard.babel.audio/audio/speaker-1.wav';
  window.dispatchEvent(new dom.window.MessageEvent('message', {
    source: window,
    data: { type: AUDIO_SOURCE_MESSAGE_TYPE, url: laneUrl, speakerKey: 'speaker-1', discoveredAt: 1 }
  }));
  let releaseFetch!: () => void;
  const fetchStarted = new Promise<void>((resolve) => {
    globalThis.fetch = async () => {
      resolve();
      await new Promise<void>((release) => { releaseFetch = release; });
      return {
        ok: true,
        status: 200,
        blob: async () => new dom.window.Blob(['audio'], { type: 'audio/wav' })
      } as Response;
    };
  });

  const capture = captureAudioTracksForDrafting();
  await fetchStarted;
  // Router noise: same pathname and same explicit job id, new panel state and hash.
  dom.window.history.replaceState({}, '', '?jobId=job-42&panel=timing#segment-3');
  releaseFetch();

  assert.deepEqual((await capture).map((track) => [track.speakerKey, track.source]), [['speaker-1', laneUrl]]);
});

test('pathname changes invalidate an in-flight capture', async () => {
  const dom = installDom('<main></main>');
  installAudioRequestCapture();
  window.dispatchEvent(new dom.window.MessageEvent('message', {
    source: window,
    data: {
      type: AUDIO_SOURCE_MESSAGE_TYPE,
      url: 'https://dashboard.babel.audio/audio/speaker-1.wav',
      speakerKey: 'speaker-1',
      discoveredAt: 1
    }
  }));
  let releaseFetch!: () => void;
  const fetchStarted = new Promise<void>((resolve) => {
    globalThis.fetch = async () => {
      resolve();
      await new Promise<void>((release) => { releaseFetch = release; });
      return {
        ok: true,
        status: 200,
        blob: async () => new dom.window.Blob(['audio'], { type: 'audio/wav' })
      } as Response;
    };
  });

  const capture = captureAudioTracksForDrafting();
  await fetchStarted;
  dom.window.history.replaceState({}, '', '/transcription/EN-transcription?jobId=job-42');
  releaseFetch();

  await assert.rejects(capture, /Audio capture task changed/);
});

test('unavailable lanes are reported with their reason instead of vanishing silently', async () => {
  const dom = installDom('<main></main>');
  installAudioRequestCapture();
  const missingUrl = 'https://dashboard.babel.audio/audio/speaker-2.wav';
  const revokedUrl = 'blob:https://dashboard.babel.audio/revoked';
  for (const [url, speakerKey] of [
    ['https://dashboard.babel.audio/audio/speaker-1.wav', 'speaker-1'],
    [missingUrl, 'speaker-2'],
    [revokedUrl, 'speaker-3']
  ]) {
    window.dispatchEvent(new dom.window.MessageEvent('message', {
      source: window,
      data: { type: AUDIO_SOURCE_MESSAGE_TYPE, url, speakerKey, discoveredAt: 1 }
    }));
  }
  globalThis.fetch = async (input: RequestInfo | URL) => {
    if (String(input) === revokedUrl) throw new TypeError('Failed to fetch');
    return {
      ok: String(input) !== missingUrl,
      status: String(input) === missingUrl ? 403 : 200,
      blob: async () => new dom.window.Blob(['audio'], { type: 'audio/wav' })
    } as Response;
  };
  const warnings: string[] = [];
  const originalWarn = console.warn;
  console.warn = (...args: unknown[]) => warnings.push(args.map(String).join(' '));
  try {
    const tracks = await captureAudioTracksForDrafting();
    assert.deepEqual(tracks.map((track) => track.speakerKey), ['speaker-1']);
  } finally {
    console.warn = originalWarn;
  }
  assert.deepEqual(warnings, [
    `[babel-gold-drafting] Audio lane unavailable (HTTP 403): ${missingUrl}`,
    `[babel-gold-drafting] Audio lane unavailable (Failed to fetch): ${revokedUrl}`
  ]);
});

function installSelectedAudioFixture(strength = 35) {
  const dom = installDom('<audio src="https://dashboard.babel.audio/original.wav"></audio>');
  const fetches = recordFetches(dom);
  const root = dom.window.document.documentElement;
  root.setAttribute(PAGE_TASK_ID_ATTRIBUTE, 'review-action-42');
  const variantKey = strength === 0 ? '' : strength === 100
    ? `zipenhancer:${'a'.repeat(64)}:${'b'.repeat(64)}` : 'c'.repeat(64);
  const state: AudioEnhancementState = {
    version: AUDIO_ENHANCEMENT_PROTOCOL_VERSION,
    taskId: 'review-action-42',
    status: 'ready',
    strength,
    desiredStrength: strength,
    revision: 1,
    variantKey,
    pairedVariantKey: 'b'.repeat(64)
  };
  const publish = (changes: Partial<AudioEnhancementState> = {}) => {
    Object.assign(state, changes);
    root.setAttribute(AUDIO_ENHANCEMENT_STATE_ATTRIBUTE, JSON.stringify(state));
  };
  publish();
  installAudioRequestCapture();
  window.dispatchEvent(new dom.window.MessageEvent('message', {
    source: window,
    data: {
      type: AUDIO_RESPONSE_MESSAGE_TYPE,
      url: 'https://dashboard.babel.audio/original.wav',
      mimeType: 'audio/wav',
      trackId: 'raw-network',
      speakerKey: 'speaker-1',
      source: 'fetch',
      capturedAt: 1,
      bytes: new Uint8Array([1, 2, 3]).buffer
    }
  }));
  const request = Promise.withResolvers<ActiveAudioRequest>();
  const requests: ActiveAudioRequest[] = [];
  dom.window.postMessage = ((data: unknown) => {
    if ((data as ActiveAudioRequest)?.type === AUDIO_ENHANCEMENT_ACTIVE_REQUEST) {
      requests.push(data as ActiveAudioRequest);
      request.resolve(data as ActiveAudioRequest);
    }
  }) as typeof dom.window.postMessage;
  const respond = (requestId: string, changes: Partial<ActiveAudioResponse> = {}, source: Window = window) => {
    const response: ActiveAudioResponse = {
      type: AUDIO_ENHANCEMENT_ACTIVE_RESPONSE,
      version: AUDIO_ENHANCEMENT_PROTOCOL_VERSION,
      requestId,
      taskId: 'review-action-42',
      available: true,
      strength,
      variantKey,
      tracks: [
        { trackId: 'lane-a', speakerKey: 'speaker-1', trackLabel: 'Speaker 1', mimeType: 'audio/wav', bytes: new Uint8Array([91, 92, 93, 94]).buffer },
        { trackId: 'lane-b', speakerKey: 'speaker-2', trackLabel: 'Speaker 2', mimeType: 'audio/wav', bytes: new Uint8Array([81, 82, 83, 84]).buffer }
      ],
      ...changes
    };
    window.dispatchEvent(new dom.window.MessageEvent('message', { source, data: response }));
  };
  return { dom, fetches, root, publish, requests, request: request.promise, respond };
}

function readCapturedBytes(dom: JSDOM, blob: Blob): Promise<number[]> {
  const result = Promise.withResolvers<number[]>();
  const reader = new dom.window.FileReader();
  reader.onload = () => result.resolve(Array.from(new Uint8Array(reader.result as ArrayBuffer)));
  reader.onerror = () => result.reject(reader.error);
  reader.readAsArrayBuffer(blob);
  return result.promise;
}

for (const strength of [0, 35, 100]) {
  test(`drafting consumes committed ${strength} percent WAV lanes instead of network audio`, async (t) => {
    const fixture = installSelectedAudioFixture(strength);
    t.after(() => fixture.dom.window.close());
    const capture = captureAudioTracksForDrafting();
    const request = await fixture.request;
    assert.equal(request.selection, 'active');
    assert.equal(request.taskId, 'review-action-42');
    fixture.respond(request.requestId);
    const tracks = await capture;
    assert.deepEqual(tracks.map((track) => [track.trackId, track.speakerKey, track.mimeType]), [
      ['lane-a', 'speaker-1', 'audio/wav'],
      ['lane-b', 'speaker-2', 'audio/wav']
    ]);
    assert.deepEqual(await Promise.all(tracks.map((track) => readCapturedBytes(fixture.dom, track.blob))), [
      [91, 92, 93, 94], [81, 82, 83, 84]
    ]);
    assert.deepEqual(fixture.fetches, []);
  });
}

test('enhancement input captures exact retained Originals during a pending blend commit', async (t) => {
  const fixture = installSelectedAudioFixture();
  t.after(() => fixture.dom.window.close());
  fixture.publish({ status: 'switching', desiredStrength: 65 });
  const capture = captureOriginalAudioTracksForEnhancement();
  const request = await fixture.request;
  assert.equal(request.selection, 'original');
  fixture.respond(request.requestId, {
    strength: 0,
    variantKey: '',
    tracks: [
      { trackId: 'lane-a', speakerKey: 'speaker-1', trackLabel: 'Speaker 1', mimeType: 'audio/wav', bytes: new Uint8Array([11, 12]).buffer },
      { trackId: 'lane-b', speakerKey: 'speaker-2', trackLabel: 'Speaker 2', mimeType: 'audio/wav', bytes: new Uint8Array([21, 22]).buffer }
    ]
  });
  const tracks = await capture;
  assert.deepEqual(await Promise.all(tracks.map((track) => readCapturedBytes(fixture.dom, track.blob))), [
    [11, 12], [21, 22]
  ]);
  assert.deepEqual(fixture.fetches, []);
});

test('selection readiness distinguishes a pending strength from a terminal failure or another task', (t) => {
  const fixture = installSelectedAudioFixture();
  t.after(() => fixture.dom.window.close());
  const scenarios: Array<{ state: Partial<AudioEnhancementState>; ready: boolean }> = [
    { state: { status: 'ready', strength: 0, desiredStrength: 100, variantKey: '' }, ready: false },
    { state: { status: 'ready', strength: 35, desiredStrength: 65, variantKey: 'c'.repeat(64) }, ready: false },
    { state: { status: 'switching', desiredStrength: 35 }, ready: false },
    { state: { status: 'error', desiredStrength: 65 }, ready: true },
    { state: { status: 'ready', strength: 0, desiredStrength: 0, variantKey: '' }, ready: true },
    { state: { status: 'error', desiredStrength: 100 }, ready: true },
    { state: { status: 'ready', taskId: 'another-review' }, ready: true }
  ];
  for (const { state, ready } of scenarios) {
    fixture.publish(state);
    assert.equal(isAudioSelectionReady(), ready, JSON.stringify(state));
  }
});

for (const strength of [0, 35]) {
  test(`drafting rejects an uncommitted request from ${strength} percent without capturing Original`, async (t) => {
    const fixture = installSelectedAudioFixture(strength);
    t.after(() => fixture.dom.window.close());
    fixture.publish({ desiredStrength: 65 });
    await assert.rejects(captureAudioTracksForDrafting(), /preparing or switching/);
    assert.deepEqual(fixture.requests, []);
    assert.deepEqual(fixture.fetches, []);
  });
}

test('drafting rejects a pending paired commit even when the requested strength is unchanged', async (t) => {
  const fixture = installSelectedAudioFixture();
  t.after(() => fixture.dom.window.close());
  fixture.publish({ status: 'switching' });
  await assert.rejects(captureAudioTracksForDrafting(), /paired source commit/);
  assert.deepEqual(fixture.requests, []);
  assert.deepEqual(fixture.fetches, []);
});

test('a failed strength request retains the committed blend for drafting', async (t) => {
  const fixture = installSelectedAudioFixture();
  t.after(() => fixture.dom.window.close());
  fixture.publish({ status: 'error', desiredStrength: 65 });
  const capture = captureAudioTracksForDrafting();
  const request = await fixture.request;
  fixture.respond(request.requestId);
  const tracks = await capture;
  assert.deepEqual(await readCapturedBytes(fixture.dom, tracks[0]!.blob), [91, 92, 93, 94]);
  assert.deepEqual(fixture.fetches, []);
});

test('unavailable native Original buffers are not replaced with network audio', async (t) => {
  const fixture = installSelectedAudioFixture(0);
  t.after(() => fixture.dom.window.close());
  const capture = captureAudioTracksForDrafting();
  const rejection = assert.rejects(capture, /native audio buffers are not ready/);
  const request = await fixture.request;
  fixture.respond(request.requestId, { available: false, tracks: [] });
  await rejection;
  assert.deepEqual(fixture.fetches, []);
});

for (const change of ['selected-strength', 'selected-original', 'selected-variant', 'pending-strength', 'pending-pair',
  'state-task', 'page-task', 'response-task', 'response-strength', 'response-variant', 'unavailable'] as const) {
  test(`selected blend capture rejects ${change} without substituting original audio`, async (t) => {
    const fixture = installSelectedAudioFixture();
    t.after(() => fixture.dom.window.close());
    const capture = captureAudioTracksForDrafting();
    const rejection = assert.rejects(capture, Error);
    const request = await fixture.request;
    if (change === 'selected-strength') fixture.publish({ strength: 65, desiredStrength: 65 });
    if (change === 'selected-original') fixture.publish({ strength: 0, desiredStrength: 0, variantKey: '' });
    if (change === 'pending-strength') fixture.publish({ desiredStrength: 65 });
    if (change === 'pending-pair') fixture.publish({ status: 'switching' });
    if (change === 'selected-variant') fixture.publish({ variantKey: 'different-model-source' });
    if (change === 'state-task') fixture.publish({ taskId: 'review-action-next' });
    if (change === 'page-task') fixture.root.setAttribute(PAGE_TASK_ID_ATTRIBUTE, 'review-action-next');
    fixture.respond(request.requestId,
      change === 'response-task' ? { taskId: 'review-action-next' } :
      change === 'response-strength' ? { strength: 65 } :
      change === 'response-variant' ? { variantKey: 'different-model-source' } :
      change === 'unavailable' ? { available: false, tracks: [] } : {});
    await rejection;
    assert.deepEqual(fixture.fetches, []);
  });
}

test('selected audio ignores responses from another window and accepts only the owning page', async (t) => {
  const fixture = installSelectedAudioFixture();
  const foreign = new JSDOM('<main></main>');
  t.after(() => { fixture.dom.window.close(); foreign.window.close(); });
  const capture = captureAudioTracksForDrafting();
  const request = await fixture.request;
  fixture.respond(request.requestId, { available: false, tracks: [] }, foreign.window as unknown as Window);
  fixture.respond(request.requestId);
  const tracks = await capture;
  assert.deepEqual(await readCapturedBytes(fixture.dom, tracks[0]!.blob), [91, 92, 93, 94]);
  assert.deepEqual(fixture.fetches, []);
});

test('enhancement input rejects committed blend buffers instead of treating them as Originals', async (t) => {
  const fixture = installSelectedAudioFixture();
  t.after(() => fixture.dom.window.close());
  const capture = captureOriginalAudioTracksForEnhancement();
  const rejection = assert.rejects(capture, Error);
  const request = await fixture.request;
  fixture.respond(request.requestId);
  await rejection;
  assert.deepEqual(fixture.fetches, []);
});
