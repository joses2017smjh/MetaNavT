import meta from '../fixtures/meta.json';
import soccer from '../fixtures/soccer.json';
import { retrievalSchema, soccerSchema, type RetrievalResponse, type SoccerResponse } from './contracts';

export const demoFiles = meta.files;
export const starterQueries = ['What is the learning rate for run 47?', 'Open checkpoints/run_047.ckpt.meta.json', 'Which GPU was requested for run 47?'];
export const matchSamples = soccer.samples.map((sample) => ({
  id: sample.id, text: sample.input.text,
  home: sample.id === 'liverpool-city' ? 'Liverpool' : 'Arsenal', away: 'Man City',
  homeShort: sample.id === 'liverpool-city' ? 'LIV' : 'ARS', awayShort: 'MCI',
  approval: sample.id === 'approval-required',
}));
const tokenize = (s: string): string[] => s.toLowerCase().match(/[a-z0-9]+/g) ?? [];
export function searchDemo(query: string, k = 5): RetrievalResponse {
  query = query.trim();
  if (!query || query.length > 2000) throw new Error('Enter a question between 1 and 2,000 characters.');
  const begin = performance.now(), terms = tokenize(query);
  const docs = demoFiles.map((file) => ({ file, tokens: tokenize(`${file.path} ${file.text}`) }));
  const mean = docs.reduce((sum, doc) => sum + doc.tokens.length, 0) / docs.length;
  const ranked = docs.map(({ file, tokens }) => {
    let score = 0;
    for (const term of new Set(terms)) {
      const tf = tokens.filter((token) => token === term).length;
      const df = docs.filter((doc) => doc.tokens.includes(term)).length;
      const idf = Math.log(1 + (docs.length - df + 0.5) / (df + 0.5));
      score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * tokens.length / mean));
    }
    if (query.toLowerCase().includes(file.path.toLowerCase())) score += 10;
    return { file, score };
  }).filter((r) => r.score > 0).sort((a, b) => b.score - a.score).slice(0, k);
  return retrievalSchema.parse({
    query, k, route: query.includes('/') ? 'lexical_path' : 'keyword_search',
    retrieval_mode: 'offline_keyword_demo', embedding_provider: 'none', reranker_loaded: false, degraded: [],
    staleness: { enabled: false, applied: false, dropped: 0 }, counts: { files: docs.length, returned: ranked.length },
    latency_ms: { client_search: performance.now() - begin },
    hits: ranked.map(({ file, score }, i) => ({ rank: i + 1, chunk_id: file.sha256, path: file.path, start_byte: 0, end_byte: new TextEncoder().encode(file.text).length, score, scores: { bm25: score, dense: null, rrf: null, rerank: null }, text: file.text })),
  });
}
export function replaySoccer(sampleId: string): SoccerResponse {
  const sample = soccer.samples.find((sample) => sample.id === sampleId);
  if (!sample) throw new Error('Choose an available fixture match.');
  return soccerSchema.parse(sample.response);
}
export const soccerModel = soccer.model_card;
