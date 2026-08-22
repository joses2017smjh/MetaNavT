# MetaNaviT v2 — Modernization Plan

Bringing a 2025 RAG file organizer up to 2026 practice.

**Target shape:** an agentic retrieval system over a personal file corpus, evaluated on a frozen benchmark, where each retrieval component was ablated and measured.

The eval harness is the first deliverable. Everything after it is a one-change-at-a-time ablation: fixed budget, mean over the gold set, report what lost as well as what won.

---

## Status

| Phase | Item | Status |
|---|---|---|
| 0.1 | Freeze a corpus, hash manifest | done (demo research tree; swap for capstone dir later) |
| 0.2 | Gold set ~100+ Q/A, CRAG taxonomy + staleness | done (generated from registry, not LLM-only) |
| 0.3 | Retrieval scoreboard: Recall@50, nDCG@10, MRR@10 | done |
| 0.3 | End-to-end RAGAS-style metrics | done (heuristic overlap + optional jury columns) |
| 0.4 | p50/p95 latency per stage | done |
| 0.5 | `make bench` / `make bench-compare` | done (`make bench-jury`, `make bench-frontier`) |
| 1.1 | Hybrid BM25 + dense, RRF k=60 | done |
| 1.2 | Cross-encoder rerank top-50 → top-8 | done (overlap teacher in CI; bge row labels fallback if the weights are missing) |
| 1.2b | Real `BAAI/bge-reranker-v2-m3` bench row | code done (`hybrid+bge-rerank` via `make bench-frontier`); **measure on GPU** — CI records `reranker_fallback=overlap` |
| 1.3 | Structure-aware chunking; late vs contextual | done (algorithms + ablation hooks) |
| 1.4 | Embedding sweep | harness flag `embedder=hash\|tfidf\|st:<model>` |
| 1.5 | pgvector halfvec / binary + rescore / HNSW `m`/`ef_search` | done (in-memory NSW analogue + SQL on `VectorStoreManager`) |
| 1.6 | Rule-based query router | done |
| 1.7 | Distilled student reranker | done (`distill_winner` fits the logged teacher) |
| 2.1 | Retrieval loop: route→plan→search→grade→rewrite→verify | done |
| 2.2 | Filesystem MCP tools + stdio server | done |
| 2.3 | HITL: `propose_move` never executes; `apply_plan` needs approval | done |
| 2.4 | RL search policy (Search-R1 / GRPO) | env + reward + masked loss + dummy CPU GRPO step done. **Not GPU-trained.** |
| 3.1 | Deterministic file graph + staleness Tier 1 | done |
| 3.2 | GraphRAG entities + communities | done (heuristic triples; optional LLM) |
| 3.3 | Hierarchical summaries | done (extractive) |
| 3.4 | Graph expansion hops ablation | done (`graph_hops` + `make sweeps`) |
| 3.5 | HippoRAG PPR boost (aggregation / multi-hop only) | done (`app/graph/hipporag.py`) |
| 4.1 | ColPali page-image indexing | done (patch grid + MaxSim; hash embedder in CI) |
| 4.2 | ColBERT-style late interaction | MaxSim implemented; not default path |
| 4.3 | SigLIP/CLIP image embeddings | done (sentence-transformers if local; pixel-hash fallback) |
| 4.4 | Cross-modal RRF fusion | done |
| 5 | Content-hash reindex, citations, fail-loud, compose, tracing spans | done / stubbed |
| 5 | Staleness Tier 2 conflict classification at query time | done (structural fields + negation pairs; optional LLM) |
| 6 | LLM-as-judge, AB/BA, self-consistency, jury, κ gate | done (`app/eval/judge.py`, `app/eval/jury.py`; `make bench-jury`) |
| 7 | RankGPT listwise + honest bge row | done (`app/retrieval/rankgpt.py`) |
| 8 | Deep Research multi-query + scratchpad + token budget | done (`app/agent/deep_research.py`) |
| 9 | HippoRAG PPR | done |
| 10 | Search-R1 / GRPO (CI dummy step) | done (`app/rl/`) |

---

## How to measure

```bash
make freeze-corpus   # rebuilds files + gold + MANIFEST hashes
make bench           # writes bench/results/<git-sha>.json (LLM-free)
make bench-compare   # this run vs bench/results/main.json
make bench-jury      # extra Phase 6 jury columns (heuristic; JUDGE=1 uses Ollama if up)
make bench-frontier  # Phase 7–9 extra configs (bge, RankGPT, multi-query, HippoRAG)
make sweeps          # HNSW, halfvec/binary, hops, GraphRAG
make figures         # doc/figures/*.svg for the README
make test-eval       # unit tests for metrics, router, graph, MCP, …
```

Retrieval metrics run on every `make bench` (no LLM). Jury columns are opt-in. Do not print a jury number on the README unless Cohen’s κ vs gold on simple_factual is ≥ 0.6.

---

## Ablation discipline

One design change at a time. Default bench configs:

1. `dense_only`
2. `bm25_only`
3. `hybrid` (RRF, no rerank, no router)
4. `hybrid+rerank`
5. `hybrid+rerank+router`
6. `hybrid+rerank+router+staleness`

Frontier configs (`make bench-frontier` only):

7. `hybrid+bge-rerank`     (loads `BAAI/bge-reranker-v2-m3` if cached; else overlap, labeled)
8. `hybrid+rankgpt`        (listwise permutation; heuristic listwise in CI)
9. `hybrid+multiquery`     (3 paraphrases, RRF union)
10. `hybrid+hipporag`      (PPR boost on aggregation / multi-hop)

Jury is **not** a retrieval config. `make bench-jury` adds e2e columns to whatever configs you run.

Keep the losers in the table. The commit history of `bench/results/*.json` is the ablation log.

---

## Resume bullets (fill from `make bench`)

First measured row (hash dense + in-memory BM25, 136 questions, 61-file frozen tree):

> Rebuilt retrieval as hybrid BM25 + dense search fused with reciprocal rank fusion, raising Recall@50 from 0.843 (BM25) / 0.915 (dense) to 0.938 on a 136-question frozen benchmark. Query routing lifted exact-path nDCG@10 from 0.596 to 0.938. Deterministic version-clustering raised nDCG@10 from 0.456 to 0.493 with no recall drop. The overlap reranker did not beat RRF (0.452 vs 0.454) and stays in the table until `bge-reranker-v2-m3` is loaded.

After routing + distillation:

> Distilled rerank scores into a lightweight student, recovering __% of reranking's nDCG@10 gain at __x lower p95; query-side routing cut latency on exact-match queries with no recall regression on semantic ones.

After the agent loop + MCP:

> Moved retrieval inside a route → search → grade → rewrite loop with a hard iteration cap, and exposed the corpus as an MCP filesystem server; destructive file moves require human approval.

After staleness:

> Added deterministic version-clustering so retrieval prefers current over superseded sources on conflicting-config questions. Query-time Tier 2 flags disagreeing `learning_rate` / paper claims so the generator cannot silently pick one.

After pgvector / GraphRAG / multimodal:

> Swept HNSW `ef_search` and compared float32 / halfvec / binary+rescore on the same vectors. Layered heuristic GraphRAG communities for corpus-level questions. Indexed pages as patch grids scored with MaxSim and fused with CLIP/SigLIP image hits via RRF.

---

## Remaining GPU work (code is in; numbers are not)

`make bench` is unchanged and LLM-free. What still needs a machine with weights:

1. **Measured `bge-reranker-v2-m3` row.** `make bench-frontier` already emits `hybrid+bge-rerank`. If the weights are not on disk, the payload sets `reranker_fallback=overlap`. Do not claim a cross-encoder gain until `reranker_loaded` is true. Set `BGE_ALLOW_DOWNLOAD=1` only when you intend to pull from Hugging Face.

2. **GPU Search-R1 / GRPO training.** `app/rl/` has the env, the reward, retrieved-token masking, group-relative advantages, and a numpy dummy step for CI. veRL + Qwen2.5-3B is still a training run, not a unit test. PPO remains the fallback if GRPO collapses.

3. **README jury numbers.** `make bench-jury` writes extra columns. Print them on the README only if `jury.kappa_simple_factual.readme_ok` is true (κ ≥ 0.6).

---

## Frontier plan (v3) — research + industry, 2025–2026

Keep the v2 contract: one design change, mean over the 136-question gold set, losers stay in the table. Do **not** replace Recall@50 / nDCG@10 with an LLM score. Judges are extra columns.

### Phase 6 — LLM-as-judge, then a jury (eval)

**Why.** Single-pass RAG eval in industry (RAGAS, DeepEval, LangSmith, Braintrust, Evidently) uses an LLM for faithfulness, context relevance, and answer relevancy. A single judge is noisy: *Reliability without Validity* (2026, ~21 judges) shows high test–retest with severe position bias; TrustJudge (Sep 2025) finds score/transitivity inconsistencies; ConsJudge (ACL Findings 2025) is built specifically for RAG judge inconsistency.

**What we built** (`app/eval/judge.py`, `app/eval/jury.py`):

| step | mechanism | source |
|---|---|---|
| 6.1 | Pointwise LLM judge, `qwen2.5:14b`, T=0: faithfulness / relevancy / groundedness vs retrieved chunks + gold | RAGAS; harness already has heuristic stubs |
| 6.2 | Pairwise: extractive answer vs agent answer, **AB and BA** (swap positions, average) | standard judge protocol; 2026 validation paper |
| 6.3 | **Self-consistency vote:** k=3 samples T=0.3, majority on {correct, partial, wrong} | Wang et al. self-consistency; Evidently “multiple evals then vote” |
| 6.4 | **Jury:** 3 heterogeneous judges (qwen2.5:14b, a smaller local model, optional API). Majority vote. Report kappa vs gold exact-match on the 80 simple-factual slice | *Replacing Judges with Juries* (Verga et al. 2024); ConsJudge dimensions |
| 6.5 | Gate: Cohen’s κ vs gold on simple_factual must be ≥ 0.6 before any judge number appears on the README | 2026 “minimum viable validation” |

Voting is for **evaluation and optional answer selection**, not for replacing BM25. On generation: sample n=3 agent traces, jury picks the one with most gold-path citations (best-of-N + verifier). That is cheaper than GRPO and is what production Deep Research UIs already do.

`make bench` stays LLM-free by default. `make bench JUDGE=1` or `make bench-jury` spends GPU.

### Phase 7 — measured cross-encoder + listwise rerank

**Why.** Overlap teacher lost. BGE-m3 is still the open default; Cohere Rerank 3.5 / Jina v2 are the industry APIs; RankGPT / RankZephyr are listwise LLM rerank (Sun et al.).

**What we built:**

- Load `BAAI/bge-reranker-v2-m3` on the RTX 8000, add config `hybrid+bge-rerank`. Same k=50 → n=8.
- Optional second row: RankGPT-style listwise over top-20 with qwen (latency will lose; report p95).
- Distill the **winner** into the existing student (`app/retrieval/distill.py`) so CI stays cheap.

If bge-m3 also loses to RRF on *this* corpus, keep that number. File search is token-heavy; a cross-encoder is not guaranteed.

### Phase 8 — Deep Research loop (industry shape)

OpenAI / Gemini / Perplexity Deep Research and Zeta Alpha’s agentic RAG write-up (2026): planner → sub-queries → hybrid retrieve → scratchpad → critic → cited report. We already have route / grade / rewrite / MCP. Missing pieces:

- Multi-query expansion (3 paraphrases, RRF the union) — cheap, usually a Recall@50 bump
- Evidence scratchpad with required `(path, byte_range)` before `respond`
- Iteration budget in tokens, not just hops (the 2026 scaling study found a raw file-system agent loses at scale; **BM25 stayed Pareto**. Do not replace hybrid with “just let the LLM ls the tree.”)
- Specialist sub-agents we already have (file, python exec) behind the same MCP interrupt

This is the production path. RL is only if the prompted loop plateaus on multi-hop / aggregation.

### Phase 9 — associative graph memory (not more hops)

Our hop ablation already failed ranking: hops=1 Recall@50 0.938 → 0.986, nDCG@10 0.495 → 0.446. Blind expansion is the wrong next step.

**Research:** HippoRAG 2 (Gutiérrez et al., ICML 2025) — triples + Personalized PageRank; LightRAG (Guo et al. 2025) local/global; Microsoft GraphRAG community reports. We already extract `run:47 —uses_encoder→ dinov2`. Wire PPR over that graph for multi-hop / aggregation only (router already knows those queries). Leave simple_factual at hops=0.

### Phase 10 — Search-R1 / GRPO (last, expensive)

Only after Phases 6–8. Jin et al. 2025: search as environment, `<search>` / `<answer>` tags, **mask retrieved tokens** in the loss, outcome reward (no process reward). GRPO is critic-free but collapses (Lazy Likelihood Displacement); PPO is the stable fallback. veRL stack. Start Qwen2.5-3B, not 14B.

**Reward (this is why the harness exists):**

```
R = 1.0 * exact_match(gold)
  + 0.5 * nDCG@10(retrieved, gold_paths)
  + 0.3 * jury_vote          # Phase 6, only if κ ≥ 0.6
  - 0.1 * n_search_calls     # tax extra searches
  - 1.0 * uncited_or_empty   # fail-loud already in the loop
```

Do not use the overlap teacher as reward. That is how you train a model to quote the archive.

Humanoid / GRPO overlap is a reason to *try* this; it is not a reason to skip Phase 6.

### Out of scope unless the gold set grows

- Swapping the personal corpus for web search (Search-R1’s original env). Our product is a **file tree**.
- Training a 70B judge. Jury of small local models is the 2024–2026 recommendation.
- Replacing BM25. 2026 *Which RAG Paradigm Wins at Scale?* (1k–500k docs): BM25 remained Pareto; graph RAG and file agents were not free wins.

---

## Later / already implemented as hooks

Phases 6–10 are in the tree. Remaining spend is GPU measurement / training, not missing modules.

Original later list that **is** implemented:

| Item | Where |
|---|---|
| 1.5 halfvec / binary / HNSW | `app/retrieval/pgvector_tune.py`, `VectorStoreManager.ensure_hnsw` / `ensure_halfvec` / `ensure_binary` |
| 3.2 GraphRAG | `app/graph/graphrag.py` |
| 4.1 / 4.3 ColPali + CLIP | `app/multimodal/colpali.py`, `app/multimodal/images.py` |
| Phase 5 Tier 2 | `app/graph/conflicts.py`, wired into `extractive_answer` |

---

## Layout

```
app/eval/          harness, metrics, gold, latency, RAGAS-style e2e, jury, SVG figures
app/retrieval/     RRF, BM25, router, hybrid index, distill, RankGPT, HNSW / halfvec / binary
app/chunking/      structure-aware, late, contextual
app/graph/         file graph, version clusters, hierarchy, GraphRAG, HippoRAG PPR, conflicts
app/agent/         retrieval loop, Deep Research, move-plan schema
app/rl/            Search-R1 parse/env, reward, GRPO dummy step
app/mcp/           filesystem tools + stdio MCP server
app/multimodal/    MaxSim, ColPali-style pages, CLIP/SigLIP images, modality RRF
bench/corpus/      frozen files, MANIFEST, registry
bench/gold/        questions.jsonl
bench/results/     one JSON per git sha + sweeps.json
doc/demo.html      live gold-question traces (app palette)
doc/figures/       README SVGs
```
