import { MaiError } from '../core/mai-protocol';
import type { BrokerRedistributionGroup, BrokerRedistributionReview, BrokerRedistributeTextResponse } from '../core/types';

export const MAI_REDISTRIBUTION_MODEL = 'google/gemini-3.8-flash';
export type MaiAuthorizedFetch = (path: 'audio/transcriptions' | 'chat/completions', body: BodyInit, contentType?: string) => Promise<unknown>;
export function parseMaiRedistributionReview(content: string, rowCount: number): BrokerRedistributionReview {
  let parsed: unknown;
  try { parsed = JSON.parse(content.trim().replace(/^```(?:json)?\s*([\s\S]*?)\s*```$/i, '$1').trim()); }
  catch { throw new MaiError('invalid-provider-response', 'Broker redistribution response is not valid JSON.'); }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new MaiError('invalid-provider-response', 'Broker redistribution response is not valid JSON.');
  }
  const review = parsed as Partial<BrokerRedistributionReview>;
  const acceptDraft = review.acceptDraft === true;
  const rawMoves = Array.isArray(review.moves) ? review.moves : [];
  const moves = rawMoves.map((move) => {
    if (!move || typeof move !== 'object') throw new MaiError('invalid-provider-response', 'Broker redistribution move is not an object.');
    const fromIndex = Math.round(Number(move.fromIndex));
    const toIndex = Math.round(Number(move.toIndex));
    const sentenceCount = Math.round(Number(move.sentenceCount));
    if (!Number.isInteger(fromIndex) || !Number.isInteger(toIndex) || Math.abs(fromIndex - toIndex) !== 1 ||
        !Number.isInteger(sentenceCount) || sentenceCount < 1 || fromIndex < 1 || toIndex < 1 || fromIndex > rowCount || toIndex > rowCount) {
      throw new MaiError('invalid-provider-response', 'Broker redistribution move is missing adjacent fromIndex, toIndex, or sentenceCount.');
    }
    return { fromIndex, toIndex, sentenceCount };
  });
  return { acceptDraft, moves, ...(typeof review.notes === 'string' ? { notes: review.notes } : {}) };
}
export async function reviewMaiRedistributions(groups: BrokerRedistributionGroup[], request: MaiAuthorizedFetch): Promise<BrokerRedistributeTextResponse> {
  const system = [
    'You review a deterministic Russian transcript text redistribution draft.',
    'The fullText is the exact text that must remain preserved.',
    'The draftAllocations assign that text to adjacent time segments.',
    'Either accept the draft or return minimal adjacent whole-sentence moves.',
    'Return JSON only with this shape: {"acceptDraft":true,"moves":[],"notes":"..."}.',
    'Each move must be {"fromIndex":1,"toIndex":2,"sentenceCount":1}.',
    'Indexes are one-based row indexes in the segment list, and fromIndex/toIndex must be adjacent.',
    'Do not rewrite, correct, translate, deduplicate, invent, or remove words.'
  ].join('\n');
  const results = [];
  for (const group of groups) {
    try {
      const user = [
        `Group id: ${group.groupId}`, `Speaker key: ${group.speakerKey}`, `Full text: ${group.fullText}`, '', 'Segments:',
        ...group.segments.map((segment) => `- ${segment.id}: index=${segment.index}, time=${segment.startSeconds ?? 'unknown'}-${segment.endSeconds ?? 'unknown'}s, current=${JSON.stringify(segment.text)}`),
        '', 'Draft allocations:', ...group.draftAllocations.map((allocation) => `- ${allocation.segmentId}: ${JSON.stringify(allocation.text)}`)
      ].join('\n');
      const payload = await request('chat/completions', JSON.stringify({
        model: MAI_REDISTRIBUTION_MODEL,
        messages: [{ role: 'system', content: system }, { role: 'user', content: user }],
        provider: { sort: 'latency' }, temperature: 0.1
      }), 'application/json');
      const response = payload as { choices?: Array<{ message?: { content?: unknown } }> } | null;
      const content = response?.choices?.[0]?.message?.content;
      if (typeof content !== 'string') throw new MaiError('invalid-provider-response', 'OpenRouter returned no redistribution review.');
      results.push({ groupId: group.groupId, ok: true as const, review: parseMaiRedistributionReview(content, group.segments.length), model: MAI_REDISTRIBUTION_MODEL });
    } catch (error) {
      results.push({ groupId: group.groupId, ok: false as const, error: error instanceof MaiError ? error.message : 'OpenRouter redistribution request failed.' });
    }
  }
  return { model: MAI_REDISTRIBUTION_MODEL, results };
}
