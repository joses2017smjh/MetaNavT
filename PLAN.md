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
| 0.3 | End-to-end RAGAS-style metrics | done (heuristic judge; LLM judge optional) |
| 0.4 | p50/p95 latency per stage | done |
| 0.5 | `make bench` / `make bench-compare` | done |
| 1.1 | Hybrid BM25 + dense, RRF k=60 | done |
| 1.2 | Cross-encoder rerank top-50 → top-8 | done (overlap teacher in CI; bge-reranker when loaded) |
| 1.3 | Structure-aware chunking; late vs contextual | done (algorithms + ablation hooks) |
| 1.4 | Embedding sweep | harness flag `embedder=hash\|tfidf\|st:<model>` |
| 1.5 | pgvector halfvec / binary + rescore / HNSW `m`/`ef_search` | done (in-memory NSW analogue + SQL on `VectorStoreManager`) |
| 1.6 | Rule-based query router | done |
| 1.7 | Distilled student reranker | done (LightGBM if present, else numpy lstsq / sklearn GBT) |
| 2.1 | Retrieval loop: route→plan→search→grade→rewrite→verify | done |
| 2.2 | Filesystem MCP tools + stdio server | done |
| 2.3 | HITL: `propose_move` never executes; `apply_plan` needs approval | done |
| 2.4 | RL search policy | README future work only |
| 3.1 | Deterministic file graph + staleness Tier 1 | done |
| 3.2 | GraphRAG entities + communities | done (heuristic triples; optional LLM) |
| 3.3 | Hierarchical summaries | done (extractive) |
| 3.4 | Graph expansion hops ablation | done (`graph_hops` + `make sweeps`) |
| 4.1 | ColPali page-image indexing | done (patch grid + MaxSim; hash embedder in CI) |
| 4.2 | ColBERT-style late interaction | MaxSim implemented; not default path |
| 4.3 | SigLIP/CLIP image embeddings | done (sentence-transformers if local; pixel-hash fallback) |
| 4.4 | Cross-modal RRF fusion | done |
| 5 | Content-hash reindex, citations, fail-loud, compose, tracing spans | done / stubbed |
| 5 | Staleness Tier 2 conflict classification at query time | done (structural fields + negation pairs; optional LLM) |

---

## How to measure

```bash
make freeze-corpus   # rebuilds files + gold + MANIFEST hashes
make bench           # writes bench/results/<git-sha>.json
make bench-compare   # this run vs bench/results/main.json
make sweeps          # HNSW, halfvec/binary, hops, GraphRAG
make figures         # doc/figures/*.svg for the README
make test-eval       # unit tests for metrics, router, graph, MCP, …
```

Retrieval metrics run on every `make bench` (no LLM). End-to-end RAGAS-style metrics use an extractive answer + token overlap unless you plug in a judge.

---

## Ablation discipline

One design change at a time. Default bench configs:

1. `dense_only`
2. `bm25_only`
3. `hybrid` (RRF, no rerank, no router)
4. `hybrid+rerank`
5. `hybrid+rerank+router`
6. `hybrid+rerank+router+staleness`

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

## Later / do not build yet

- **2.4** RL search policy (Search-R1 / GRPO). The eval harness *is* the reward; only attempt after there is a specific reason (e.g. humanoid RL overlap). Mention in the README, do not implement.

Everything else on the original later list is in:

| Item | Where |
|---|---|
| 1.5 halfvec / binary / HNSW | `app/retrieval/pgvector_tune.py`, `VectorStoreManager.ensure_hnsw` / `ensure_halfvec` / `ensure_binary` |
| 3.2 GraphRAG | `app/graph/graphrag.py` |
| 4.1 / 4.3 ColPali + CLIP | `app/multimodal/colpali.py`, `app/multimodal/images.py` |
| Phase 5 Tier 2 | `app/graph/conflicts.py`, wired into `extractive_answer` |

---

## Layout

```
app/eval/          harness, metrics, gold, latency, RAGAS-style e2e, sweeps, SVG figures
app/retrieval/     RRF, BM25, router, hybrid index, distill, HNSW / halfvec / binary
app/chunking/      structure-aware, late, contextual
app/graph/         file graph, version clusters, hierarchy, GraphRAG, conflicts
app/agent/         retrieval loop, move-plan schema
app/mcp/           filesystem tools + stdio MCP server
app/multimodal/    MaxSim, ColPali-style pages, CLIP/SigLIP images, modality RRF
bench/corpus/      frozen files, MANIFEST, registry
bench/gold/        questions.jsonl
bench/results/     one JSON per git sha + sweeps.json
doc/demo.html      live gold-question traces (app palette)
doc/figures/       README SVGs
```
