import {
  PAGE_TASK_ID_ATTRIBUTE,
  PAGE_TASK_ID_REQUEST_MESSAGE_TYPE,
  PAGE_TASK_ID_RESPONSE_MESSAGE_TYPE,
  type PageTaskIdRequestMessage,
  type PageTaskIdResponseMessage
} from '../core/audio-intercept-protocol';

let pendingRefresh: Promise<string> | null = null;
let requestSequence = 0;

function isTaskIdResponse(value: unknown, requestId: string): value is PageTaskIdResponseMessage {
  if (!value || typeof value !== 'object') return false;
  const message = value as Partial<PageTaskIdResponseMessage>;
  return message.type === PAGE_TASK_ID_RESPONSE_MESSAGE_TYPE && message.requestId === requestId;
}

export function readPublishedPageTaskId(documentRef: Document = document): string {
  return documentRef.documentElement?.getAttribute(PAGE_TASK_ID_ATTRIBUTE)?.trim() || '';
}

export function refreshPageTaskIdentity(
  windowRef: Window = window,
  documentRef: Document = document,
  timeoutMs = 500
): Promise<string> {
  if (pendingRefresh) return pendingRefresh;
  const requestId = `${Date.now()}:${requestSequence += 1}`;
  pendingRefresh = new Promise<string>((resolve) => {
    let settled = false;
    const finish = (reviewActionId: string) => {
      if (settled) return;
      settled = true;
      windowRef.removeEventListener('message', onMessage);
      windowRef.clearTimeout(timeoutId);
      resolve(reviewActionId);
    };
    const onMessage = (event: MessageEvent) => {
      if (event.source !== windowRef || !isTaskIdResponse(event.data, requestId)) return;
      const reviewActionId = event.data.reviewActionId.trim();
      if (reviewActionId) {
        documentRef.documentElement?.setAttribute(PAGE_TASK_ID_ATTRIBUTE, reviewActionId);
      }
      finish(reviewActionId);
    };
    const timeoutId = windowRef.setTimeout(() => finish(''), timeoutMs);
    windowRef.addEventListener('message', onMessage);
    windowRef.postMessage({
      type: PAGE_TASK_ID_REQUEST_MESSAGE_TYPE,
      requestId
    } satisfies PageTaskIdRequestMessage, '*');
  }).finally(() => {
    pendingRefresh = null;
  });
  return pendingRefresh;
}
