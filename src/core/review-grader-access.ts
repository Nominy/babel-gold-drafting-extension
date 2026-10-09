import { createReviewGraderAccess } from '@nominy/babel-babel-runtime';
import type { ReviewGraderAccess } from '@nominy/babel-babel-runtime';

let access: ReviewGraderAccess | undefined;

export function reviewGraderAccess(): ReviewGraderAccess {
  return access ??= createReviewGraderAccess();
}

export async function hasReviewGraderAccess(): Promise<boolean> {
  const gate = reviewGraderAccess();
  await gate.start();
  return gate.isAvailable();
}

export async function requireReviewGraderAccess(): Promise<AbortSignal> {
  const signal = await reviewGraderAccess().acquire();
  if (!signal) throw new Error('This operation is unavailable.');
  signal.throwIfAborted();
  return signal;
}
