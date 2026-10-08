import { z } from 'zod';
import { sha256 } from '@noble/hashes/sha256';
import { bytesToHex } from '@noble/hashes/utils';
const fingerprint = (text: string) => bytesToHex(sha256(new TextEncoder().encode(text)));
import { requestJson } from './client';
import { searchDemo } from './demo';

const source = z.object({ source_id: z.string().regex(/^S[1-9]\d*$/), chunk_id: z.string(), path: z.string().nullable(), start_byte: z.number().int().nonnegative().nullable(), end_byte: z.number().int().nonnegative().nullable(), text: z.string(), excerpt_sha256: z.string().regex(/^[0-9a-f]{64}$/), inspected: z.boolean() }).refine(s => s.start_byte == null || s.end_byte == null || s.end_byte >= s.start_byte);
export const agentSchema = z.object({
  schema_version: z.literal(1), service: z.literal('metanavt-agent'), mode: z.enum(['demo', 'live']), model: z.string(), synthetic: z.boolean(),
  status: z.enum(['complete', 'abstained', 'limit_reached', 'blocked']), question: z.string(), answer: z.string(),
  citations: z.array(z.object({ source_id: z.string().regex(/^S[1-9]\d*$/), path: z.string().nullable(), start_byte: z.number().nullable(), end_byte: z.number().nullable(), excerpt_sha256: z.string().regex(/^[0-9a-f]{64}$/) })),
  sources: z.array(source).max(8),
  trace: z.array(z.object({ step: z.number().int().positive(), tool: z.string(), arguments: z.record(z.unknown()), status: z.enum(['ok', 'error']), duration_ms: z.number().nonnegative(), source_ids: z.array(z.string()), error: z.string().nullable().optional() })),
  limits: z.object({ max_model_rounds: z.number().int().positive().max(4), max_tool_calls: z.number().int().positive().max(6), model_rounds: z.number().int().nonnegative(), tool_calls: z.number().int().nonnegative() }),
  citation_check: z.object({ passed: z.boolean(), scope: z.literal('inspected_excerpt_references'), unknown_ids: z.array(z.string()) }),
  stop_reason: z.string(), notice: z.string(),
}).superRefine((r, ctx) => {
  if (new Set(r.sources.map(s => s.source_id)).size !== r.sources.length) ctx.addIssue({ code: 'custom', message: 'Agent source IDs must be unique.' });
  const inspected = new Map(r.sources.filter(s => s.inspected).map(s => [s.source_id, s]));
  for (const c of r.citations) {
    const s = inspected.get(c.source_id);
    if (!s || c.excerpt_sha256 !== s.excerpt_sha256 || c.path !== s.path || c.start_byte !== s.start_byte || c.end_byte !== s.end_byte) ctx.addIssue({ code: 'custom', message: 'Citation does not match an inspected source.' });
  }
  const cited = new Set(r.citations.map(c => c.source_id));
  const references = [...r.answer.matchAll(/\[(S\d+)\]/g)].map(m => m[1]);
  if (references.some(id => !cited.has(id)) || (r.status === 'complete' && !references.length)) ctx.addIssue({ code: 'custom', message: 'Answer contains missing or unknown source references.' });
  if (r.status === 'complete' && (!r.citation_check.passed || !r.citations.length || r.citation_check.unknown_ids.length)) ctx.addIssue({ code: 'custom', message: 'Agent answer has unverified source references.' });
  if (r.limits.tool_calls > r.limits.max_tool_calls || r.limits.model_rounds > r.limits.max_model_rounds) ctx.addIssue({ code: 'custom', message: 'Agent exceeded its execution limits.' });
});
export type AgentResult = z.infer<typeof agentSchema>;
export type AgentSource = AgentResult['sources'][number];

export async function askAgent(base: string, question: string, token: string): Promise<AgentResult> {
  const raw = await requestJson(base, '/agent/ask', { question, mode: 'live' }, token, 90000);
  const parsed = agentSchema.safeParse(raw);
  if (!parsed.success) throw new Error('The research agent returned an incompatible or unverified response.');
  for (const s of parsed.data.sources) {
    const digest = fingerprint(s.text);
    if (digest !== s.excerpt_sha256) throw new Error('The research agent returned an excerpt with a mismatched fingerprint.');
  }
  return parsed.data;
}

export async function runAgentDemo(question: string): Promise<AgentResult> {
  const begin = performance.now();
  const retrieval = searchDemo(question, 3);
  const sources = await Promise.all(retrieval.hits.map(async (hit, index) => ({
    source_id: `S${index + 1}`, chunk_id: hit.chunk_id, path: hit.path,
    start_byte: hit.start_byte ?? null, end_byte: hit.end_byte ?? null, text: hit.text,
    excerpt_sha256: fingerprint(hit.text),
    inspected: true,
  })));
  const citations = sources.map(({ source_id, path, start_byte, end_byte, excerpt_sha256 }) => ({ source_id, path, start_byte, end_byte, excerpt_sha256 }));
  return agentSchema.parse({
    schema_version: 1, service: 'metanavt-agent', mode: 'demo', model: 'scripted-browser-demo', synthetic: true,
    status: sources.length ? 'complete' : 'abstained', question,
    answer: sources.length ? sources.map(s => `${s.path}\n${s.text.trim().slice(0, 700)} [${s.source_id}]`).join('\n\n') : 'No matching research evidence was found. Try a file path or a starter question.',
    citations, sources,
    trace: [{ step: 1, tool: 'search_research', arguments: { query: question, k: 3 }, status: 'ok', duration_ms: performance.now() - begin, source_ids: sources.map(s => s.source_id) }, ...sources.map((s, i) => ({ step: i + 2, tool: 'inspect_source', arguments: { source_id: s.source_id }, status: 'ok', duration_ms: 0, source_ids: [s.source_id] }))],
    limits: { max_model_rounds: 4, max_tool_calls: 6, model_rounds: 0, tool_calls: sources.length + 1 },
    citation_check: { passed: true, scope: 'inspected_excerpt_references', unknown_ids: [] },
    stop_reason: sources.length ? 'scripted_extracts_complete' : 'no_sources',
    notice: 'Scripted offline agent demo: actual local search and excerpt inspection over synthetic files. No language model runs. Source membership is checked; this does not establish that a model claim follows from its source.',
  });
}
