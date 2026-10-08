import { z } from 'zod';

const stageScores = z.object({ bm25: z.number().nullable().optional(), dense: z.number().nullable().optional(), rrf: z.number().nullable().optional(), rerank: z.number().nullable().optional() });
export const hitSchema = z.object({
  rank: z.number().int().positive(), chunk_id: z.string(), path: z.string().nullable(),
  start_byte: z.number().int().nonnegative().nullable().optional(),
  end_byte: z.number().int().nonnegative().nullable().optional(),
  score: z.number().finite(), scores: stageScores, text: z.string(),
}).refine((hit) => hit.start_byte == null || hit.end_byte == null || hit.end_byte >= hit.start_byte, { message: 'Invalid source byte range.' });
export const retrievalSchema = z.object({
  query: z.string(), k: z.number().int(), route: z.string().nullable(),
  retrieval_mode: z.string(), embedding_provider: z.string(), reranker_loaded: z.boolean(),
  degraded: z.array(z.object({ component: z.string(), error: z.string() })),
  latency_ms: z.record(z.number()), counts: z.record(z.number()), hits: z.array(hitSchema),
  staleness: z.object({ enabled: z.boolean(), applied: z.boolean(), dropped: z.number() }),
});
const probability = z.number().finite().min(0).max(1);
const prediction = z.object({
  match_outcome: z.object({ home: probability, draw: probability, away: probability, conformal_set: z.array(z.string()).optional(), conformal_alpha: probability.optional() }),
  expected_goals: z.object({ home: z.number().finite().nonnegative(), away: z.number().finite().nonnegative() }),
  exact_score: z.object({ top_scorelines: z.array(z.object({ score: z.string(), prob: probability })) }).passthrough().optional(),
  event_sequence: z.record(z.unknown()).optional(),
}).passthrough();
export const soccerSchema = z.object({
  service: z.literal('soccer'), mode: z.enum(['demo', 'live']), status: z.enum(['complete', 'pending_approval']),
  thread_id: z.string(), answer: z.string(), prediction: prediction.nullable().optional(),
  degraded: z.array(z.unknown()), tool_calls: z.array(z.object({ server: z.string(), tool: z.string(), ok: z.boolean().optional(), error: z.string().nullable().optional() }).passthrough()),
  provenance: z.object({ kind: z.string(), model_version: z.string().nullable().optional(), data_backend: z.string().nullable().optional(), notes: z.array(z.string()) }).passthrough(),
  approval_request: z.unknown().optional(),
}).passthrough().superRefine((value, ctx) => {
  if (value.status === 'pending_approval' && value.prediction != null) ctx.addIssue({ code: z.ZodIssueCode.custom, message: 'A paused workflow cannot expose a prediction.' });
  if (value.status === 'complete' && value.prediction == null) ctx.addIssue({ code: z.ZodIssueCode.custom, message: 'A completed analysis must include a prediction.' });
  const p = value.prediction?.match_outcome;
  if (p && Math.abs(p.home + p.draw + p.away - 1) > 0.02) ctx.addIssue({ code: z.ZodIssueCode.custom, message: 'Outcome probabilities do not sum to one.' });
});
export type RetrievalHit = z.infer<typeof hitSchema>;
export type RetrievalResponse = z.infer<typeof retrievalSchema>;
export type SoccerResponse = z.infer<typeof soccerSchema>;
export type Settings = { mode: 'demo' | 'connected'; metaUrl: string; soccerUrl: string; agentUrl: string };
export type SavedItem = { id: string; kind: 'research' | 'soccer'; title: string; subtitle: string; content: string; savedAt: string; provenance: string };
