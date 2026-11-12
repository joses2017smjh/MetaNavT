# MetaNaviT v2 — Modernization Plan

Bringing a 2025 RAG file organizer up to 2026 practice.

**Target shape:** an agentic retrieval system over a personal file corpus, evaluated on a frozen benchmark, where each retrieval component was ablated and measured.

The eval harness is the first deliverable. Everything after it is a one-change-at-a-time ablation: fixed budget, mean over the gold set, report what lost as well as what won.

---

## Status

A status says what exists and, for a measured piece, the verdict its row holds: the paired bootstrap interval of nDCG@10 against the named parent is above zero, covers zero or below zero (`tests/eval/test_plan_status.py` checks every verdict below against the results files). The numbers are in the generated tables each row links to: [CPU fixture](README.md#bench-table), [real models](README.md#bench-neural), [prototype arms, CPU](README.md#bench-frontier), [prototype arms, real models](README.md#bench-frontier-neural), [reranker prototypes](README.md#bench-rerank-arms). "Built" without a verdict means no results file scores it.

| Phase | Item | Status |
|---|---|---|
| 0.1 | Freeze a corpus, hash manifest | done (demo research tree; swap for capstone dir later) |
| 0.2 | Gold set (Q/A), CRAG taxonomy + staleness | done: written by `bench/corpus/build.py` from the corpus registry, no LLM; the question count is `n_gold` in the header of `bench/results/main.json` |
| 0.3 | Retrieval scoreboard: Recall@50, nDCG@10, MRR@10 | done, with paired bootstrap intervals ([CPU fixture](README.md#bench-table)) |
| 0.3 | End-to-end RAGAS-style metrics | built: heuristic token-overlap columns, labelled and never an LLM result; the LLM judge is row `6` |
| 0.4 | p50/p95 latency per stage | done |
| 0.5 | `make bench` / `make bench-compare` | done (`make bench-check` in CI, `make bench-jury-v2`, `make bench-frontier`) |
| 1.1 | Hybrid BM25 + dense, RRF (`RRF_K` in `app/retrieval/fuse.py`) | measured: with hash embeddings `hybrid` below zero vs `bm25_only` ([CPU fixture](README.md#bench-table)); with bge-small `hybrid@bge-small` above zero vs `dense_only@bge-small` ([real models](README.md#bench-neural)); served as one Postgres statement ([parity](README.md#bench-parity)) |
| 1.2 | Cross-encoder rerank | measured: `hybrid+bge-rerank@bge-small` above zero vs `hybrid@bge-small` ([real models](README.md#bench-neural)); the CI row is the overlap reranker, a labelled fallback: `hybrid+rerank` covers zero vs `hybrid` ([CPU fixture](README.md#bench-table)). Served: the cross-encoder reranks the fused top `RERANK_DEPTH` (default 20), the rest keep their RRF order, and the top `RERANK_TOP_N` (default 8) are returned; the depth and precision were chosen from the SciFact sweep ([SciFact](README.md#bench-beir)) |
| 1.2b | Real `BAAI/bge-reranker-v2-m3` bench row | measured with the weights loaded (`reranker_loaded` true) on a GPU: the `1.2` row, and on SciFact and FiQA ([SciFact](README.md#bench-beir), [FiQA](README.md#bench-beir-fiqa)); a run without the weights still records `reranker_fallback=overlap` |
| 1.3 | Structure-aware chunking; late vs contextual | evaluated: the structure-aware chunker is not measurably better than a fixed window, `hybrid+fixed-chunks` covers zero vs `hybrid`, `hybrid+fixed-chunks@bge-small` covers zero vs `hybrid@bge-small`; the path-words contextual prefix flips sign, `hybrid+contextual` above zero vs `hybrid` with the hash embedder and `hybrid+contextual@bge-small` below zero vs `hybrid@bge-small` with bge-small (a recorded loss); the LLM-written prefix `hybrid+contextual-llm@bge-small` covers zero vs `hybrid@bge-small`; late chunking (bge-small only) `hybrid+late-chunking@bge-small` covers zero vs `hybrid@bge-small` ([CPU](README.md#bench-frontier), [real models](README.md#bench-frontier-neural)) |
| 1.4 | Embedding sweep | measured: model-free TF-IDF `dense_only@tfidf` above zero vs `dense_only` and `hybrid@tfidf` above zero vs `hybrid` ([CPU](README.md#bench-frontier)); `dense_only@bge-small` above zero vs `bm25_only`, `dense_only@bge-base` above zero vs `bm25_only`, `hybrid@bge-base` above zero vs `dense_only@bge-base` ([real models](README.md#bench-neural)); harness flag `embedder=hash\|tfidf\|st:<model>` |
| 1.5 | pgvector halfvec / binary + rescore / HNSW `m`/`ef_search` | partly: in-memory NSW analogue measured (`make sweeps`, hash vectors); the served statement can read the HNSW index a schema migration builds (`app/database/migrations/vector/`) and sets `ef_search` per statement; in Postgres, `make bench-served-scale` measures that index against the exact scan on the served FiQA table (bge-small chunk vectors) at several `ef_search` values and with the served settings, then a halfvec and a binary + rescore copy against float32 on the same task-local table (`bench/results/pgvector_hnsw.json`, `app/eval/pgvector_hnsw.py`; [generated table](README.md#bench-pgvector-hnsw)); halfvec and binary are not in the served path (the uncalled `ensure_halfvec` / `ensure_binary` helpers were removed with the inline DDL) |
| 1.6 | Rule-based query router | measured: overall `hybrid+rerank+router` covers zero vs `hybrid+rerank` and `hybrid+bge-rerank+router@bge-small` covers zero vs `hybrid+bge-rerank@bge-small`; the generated tables give its delta on the exact-path slice it is aimed at and on every slice where that delta is negative ([CPU fixture](README.md#bench-table), [real models](README.md#bench-neural)); served |
| 1.7 | Distilled student reranker | evaluated, a recorded loss on held-out questions: `hybrid+distilled-student@bge-small` below zero vs `hybrid+bge-rerank@bge-small` and below zero vs `hybrid@bge-small`; `hybrid+distilled-student(lightgbm)@bge-small` below zero vs `hybrid+bge-rerank@bge-small` and covers zero vs `hybrid@bge-small` ([reranker prototypes](README.md#bench-rerank-arms)); nothing serves a student |
| 2.1 | Retrieval loop: route→plan→search→grade→rewrite→verify | built (`app/agent/retrieval_loop.py`); no results file scores its answers |
| 2.2 | Filesystem MCP tools + stdio server | done (`app/mcp/server_sdk.py` on the official SDK, `app/mcp/server.py`) |
| 2.3 | HITL: `propose_move` never executes; `apply_plan` needs approval | done: applying needs an operator-signed, single-use Ed25519 capability; the deterministic approval benchmark is `make bench-trusted-actions` ([results](bench/results/trusted_actions_v2.json)) |
| 2.4 | RL search policy (Search-R1 / GRPO) | built, not evaluated: env + reward + masked loss + a dummy CPU GRPO step; **not GPU-trained** (an open owner decision: evaluate it, or label it a demo) |
| 3.1 | Deterministic file graph + staleness Tier 1 | measured: `hybrid+rerank+router+staleness` above zero vs `hybrid+rerank+router`, `hybrid+bge-rerank+router+staleness@bge-small` above zero vs `hybrid+bge-rerank+router@bge-small` ([CPU fixture](README.md#bench-table), [real models](README.md#bench-neural)); served |
| 3.2 | GraphRAG entities + communities | evaluated, a negative result: `hybrid+rerank+graphrag` covers zero vs `hybrid+rerank`, `hybrid+bge-rerank+graphrag@bge-small` covers zero vs `hybrid+bge-rerank@bge-small`; the community step collapses the fixture into one community (table notes) ([CPU](README.md#bench-frontier), [real models](README.md#bench-frontier-neural)); LLM extraction and modularity clustering not evaluated (an open owner decision: a fairer test, or the negative result as it stands) |
| 3.3 | Hierarchical summaries | built (extractive), not evaluated |
| 3.4 | Graph expansion hops ablation | evaluated, a recorded loss: `hybrid+hops=1` below zero vs `hybrid`, `hybrid+hops=2` below zero vs `hybrid`, `hybrid+hops=1@bge-small` below zero vs `hybrid@bge-small`, `hybrid+hops=2@bge-small` below zero vs `hybrid@bge-small` ([CPU](README.md#bench-frontier), [real models](README.md#bench-frontier-neural)); the default stays at no hops |
| 3.5 | HippoRAG PPR boost (the router's aggregation route only) | evaluated: `hybrid+hipporag` above zero vs `hybrid` with the hash embedder, `hybrid+hipporag@bge-small` covers zero vs `hybrid@bge-small`: no measured gain once the embedder is real ([CPU](README.md#bench-frontier), [real models](README.md#bench-frontier-neural)) |
| 4.1 | ColPali page-image indexing | built (patch grid + MaxSim; hash embedder in CI), not evaluated: the gold set has no image or PDF question |
| 4.2 | ColBERT-style late interaction | built (MaxSim), not evaluated and not on the default path (an open owner decision: evaluate it, or label it a demo) |
| 4.3 | SigLIP/CLIP image embeddings | built (sentence-transformers if local; pixel-hash fallback), not evaluated |
| 4.4 | Cross-modal RRF fusion | built, not evaluated |
| 5 | Content-hash reindex, citations, fail-loud, compose, tracing spans | done: the index step diffs every file's SHA-256 against the ingest manifest and embeds only new or changed files (`app/engine/ingest.py`); citations are byte ranges; empty or uncited answers fail loud; Compose runs the migrations, the index step and the API; spans through `app/observability.py` (OpenTelemetry with the `otel` extra). An open owner decision: the older content hash of `IndexManager` (`indexed_files.content_hash`, `hashes_for`) has no caller on the served path; delete it or keep it |
| 5 | Staleness Tier 2 conflict classification at query time | built (structural fields + negation pairs; optional LLM), not evaluated: no benchmark row |
| 6 | LLM-as-judge, AB/BA, self-consistency, jury, κ gate | measured: `make bench-jury-v2` (`app/eval/llm_eval.py`, `app/eval/jury.py`); which judge passes the kappa gate on the calibration set, and what that validates, is in the [generated table](README.md#bench-jury) |
| 7 | RankGPT listwise + honest bge row | evaluated: listwise reranking by an LLM, `hybrid+listwise(qwen2.5:7b)@bge-small` above zero vs `hybrid@bge-small` and below zero vs `hybrid+bge-rerank@bge-small` ([reranker prototypes](README.md#bench-rerank-arms)); the frontier arm with the token-overlap fallback, `hybrid+rankgpt` covers zero vs `hybrid` and `hybrid+rankgpt@bge-small` below zero vs `hybrid@bge-small` ([CPU](README.md#bench-frontier), [real models](README.md#bench-frontier-neural)); the bge row is row `1.2b` |
| 8 | Deep Research multi-query + scratchpad + token budget | built (`app/agent/deep_research.py`); its multi-query step is evaluated, `hybrid+multiquery` above zero vs `hybrid`, `hybrid+multiquery@bge-small` above zero vs `hybrid@bge-small`, with keyword rules written for this fixture ([CPU](README.md#bench-frontier), [real models](README.md#bench-frontier-neural)); the loop's answers are not evaluated |
| 9 | HippoRAG PPR | evaluated: row `3.5` |
| 10 | Search-R1 / GRPO (CI dummy step) | built, not trained or evaluated: row `2.4` (`app/rl/`) |
| 11 | Research artifacts + code production | built (`app/artifacts/`, MCP HITL), not evaluated: no quality measurement of the generated artifacts (an open owner decision: evaluate it, or label it a demo) |
| 12 | Generated hard-case demos + negative controls | done (`bench/demo/manifest.json`, `app/eval/demo_export.py`) |
| 13 | Spreadsheet aggregation + approval-gated MATLAB visualization | built (`app/artifacts/visualization.py`): a demo of the approval checkpoint on one frozen CSV, not an evaluation |
| 14 | HyDE, Query Decomposition, Corrective RAG | evaluated. HyDE in the first stage: `hybrid+hyde` below zero vs `hybrid`, `dense_only+hyde` below zero vs `dense_only`, `hybrid+hyde+rerank` covers zero vs `hybrid+rerank`; with bge-small `hybrid+hyde@bge-small` below zero vs `hybrid@bge-small`, `dense_only+hyde@bge-small` below zero vs `dense_only@bge-small`, `hybrid+hyde+bge-rerank@bge-small` covers zero vs `hybrid+bge-rerank@bge-small`; with a hypothesis written by an LLM `hybrid+hyde-llm@bge-small` below zero vs `hybrid@bge-small`, `hybrid+hyde-llm+bge-rerank@bge-small` covers zero vs `hybrid+bge-rerank@bge-small`. HyDE after the reranker (the older wiring): `hybrid+rerank+hyde` below zero vs `hybrid+rerank`, `hybrid+bge-rerank+hyde@bge-small` below zero vs `hybrid+bge-rerank@bge-small`. Decomposition: `hybrid+rerank+decompose` covers zero vs `hybrid+rerank`, `hybrid+bge-rerank+decompose@bge-small` covers zero vs `hybrid+bge-rerank@bge-small`. Corrective retrieval: `hybrid+rerank+corrective` covers zero vs `hybrid+rerank`, `hybrid+bge-rerank+corrective@bge-small` covers zero vs `hybrid+bge-rerank@bge-small`; with HyDE after the reranker `hybrid+rerank+hyde+corrective` below zero vs `hybrid+rerank`, `hybrid+bge-rerank+hyde+corrective@bge-small` below zero vs `hybrid+bge-rerank@bge-small` ([CPU](README.md#bench-frontier), [real models](README.md#bench-frontier-neural)) |
| 15 | Adaptive Retrieval, Semantic Cache, Citation Verification | adaptive retrieval evaluated as a no-harm check only (it never skips a gold question): `hybrid+rerank+adaptive` covers zero vs `hybrid+rerank`, `hybrid+bge-rerank+adaptive@bge-small` covers zero vs `hybrid+bge-rerank@bge-small`; the semantic cache, replayed over the gold set as a false-hit test, a recorded loss: `hybrid+rerank+cache` below zero vs `hybrid+rerank`, `hybrid+bge-rerank+cache@bge-small` below zero vs `hybrid+bge-rerank@bge-small` (paraphrase probes in the table notes) ([CPU](README.md#bench-frontier), [real models](README.md#bench-frontier-neural)); citation verification built (`app/agent/citation_verify.py`), no benchmark row |

---

## How to measure

```bash
make freeze-corpus   # rebuilds files + gold + MANIFEST hashes
make bench           # writes bench/results/<git-sha>.json (LLM-free)
make bench-compare   # this run vs bench/results/main.json
make bench-jury      # extra Phase 6 jury columns (heuristic; JUDGE=1 uses Ollama if up)
make bench-frontier  # prototype arms vs their parent rows (CPU stand-ins) -> bench/results/frontier.json
make bench-frontier-neural  # the same arms with bge-small + bge-reranker-v2-m3 -> bench/results/frontier_neural.json
make bench-rerank-arms      # distilled student (and optional listwise LLM) vs the cross-encoder -> bench/results/rerank_arms.json
make sweeps          # HNSW, halfvec/binary, hops, GraphRAG
make figures         # doc/figures/*.svg for the README
make matlab-demo     # approval-gated CSV aggregation → MATLAB .m + PNG
make demos           # generated hard-case traces + figures + GIFs
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

Frontier arms (`make bench-frontier`; each is one switch away from its parent, and the two parents, `hybrid` and `hybrid+rerank`, are computed in the same run):

7. `hybrid+rankgpt`        (parent `hybrid`; listwise permutation, token-overlap fallback unless an LLM is wired in)
8. `hybrid+multiquery`     (parent `hybrid`; heuristic paraphrases, RRF union)
9. `hybrid+hipporag`       (parent `hybrid`; PPR boost gated by the router's own prediction)
10. `hybrid+hops=1`, `hybrid+hops=2` (parent `hybrid`; file-graph neighbour expansion)
11. `hybrid+fixed-chunks`, `hybrid+contextual`, `hybrid+late-chunking` (parent `hybrid`; chunking variants, late chunking in the real-model run only)
12. `hybrid+rerank+hyde`, `hybrid+rerank+decompose`, `hybrid+rerank+corrective`, `hybrid+rerank+hyde+corrective` (parent `hybrid+rerank`)
13. `hybrid+rerank+adaptive` (parent `hybrid+rerank`; retrieve-or-skip gate)
14. `hybrid+rerank+graphrag` (parent `hybrid+rerank`; community-summary routing)

`hybrid+bge-rerank` stays in `FRONTIER_CONFIGS` but is not a frontier arm: it is the reranked parent of the real-model run and a row of `make bench-neural`.

Jury is **not** a retrieval config. `make bench-jury` adds e2e columns to whatever configs you run.

Keep the losers in the table. The commit history of `bench/results/*.json` is the ablation log.

---

## Resume bullets (fill from `make bench`)

First measured row (hash dense + in-memory BM25, 136 questions, 61-file frozen tree):

> Superseded (2026-09-23). This draft led with Recall@50, which on a 61-file corpus mostly measures list length, and credited the router with an exact-path gain (0.596 → 0.938) that compared against plain RRF; the router's own ablation is the reranked hybrid → router row on n = 8. The current numbers, with 95% confidence intervals, are the generated headline in README.md (`make bench-table`, from `bench/results/main.json`); the resume bullets will be written from that file in docs/RESUME_EVIDENCE.md.

After routing + distillation: no bullet. The template that stood here (a student "recovering a share of reranking's gain" and routing "with no recall regression") is dropped: both distilled students are below the cross-encoder on held-out questions (status row `1.7`), and the router's per-category deltas include slices where it is lower (status row `1.6`).

After the agent loop + MCP:

> Moved retrieval inside a route → search → grade → rewrite loop with a hard iteration cap, and exposed the corpus as an MCP filesystem server; destructive file moves require human approval.

After staleness:

> Added deterministic version-clustering so retrieval prefers current over superseded sources on conflicting-config questions. Query-time Tier 2 flags disagreeing `learning_rate` / paper claims so the generator cannot silently pick one.

After pgvector / GraphRAG / multimodal:

> Swept HNSW `ef_search` and compared float32 / halfvec / binary+rescore on the same vectors. Layered heuristic GraphRAG communities for corpus-level questions. Indexed pages as patch grids scored with MaxSim and fused with CLIP/SigLIP image hits via RRF.

---

## Remaining GPU work

`make bench` is unchanged and LLM-free. What needed a machine with weights, and where it stands:

1. **Measured `bge-reranker-v2-m3` row: measured** (status rows `1.2` and `1.2b`; `make bench-neural`, `reranker_loaded` true in `bench/results/neural.json`). A run without the weights still sets `reranker_fallback=overlap`, and no cross-encoder gain is claimed from such a row. Set `BGE_ALLOW_DOWNLOAD=1` only when you intend to pull from Hugging Face.

2. **GPU Search-R1 / GRPO training.** `app/rl/` has the env, the reward, retrieved-token masking, group-relative advantages, and a numpy dummy step for CI. veRL + Qwen2.5-3B is still a training run, not a unit test. PPO remains the fallback if GRPO collapses.

3. **README jury numbers: measured** (status row `6`). `make bench-jury-v2` writes `bench/results/jury_v2.json`; a judge's score is printed on the README only when its kappa on the calibration set reaches the gate (`kappa_gate` in the file; `readme_ok` per judge under `summary.kappa`).

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

**Outcome.** The cross-encoder row is measured and beats plain hybrid (status row `1.2`); the listwise row (status row `7`) and both distilled students (status row `1.7`) are below the cross-encoder, so no student replaces it. The served reranker does not use the depth planned above: it reranks the fused top `RERANK_DEPTH` (default 20), the served default in `app/engine/retriever.py`, chosen from the SciFact depth and precision sweep ([SciFact](README.md#bench-beir)).

### Phase 8 — Deep Research loop (industry shape)

OpenAI / Gemini / Perplexity Deep Research and Zeta Alpha’s agentic RAG write-up (2026): planner → sub-queries → hybrid retrieve → scratchpad → critic → cited report. We already have route / grade / rewrite / MCP. Missing pieces:

- Multi-query expansion (3 paraphrases, RRF the union) — cheap, usually a Recall@50 bump
- Evidence scratchpad with required `(path, byte_range)` before `respond`
- Iteration budget in tokens, not just hops (the 2026 scaling study found a raw file-system agent loses at scale; **BM25 stayed Pareto**. Do not replace hybrid with “just let the LLM ls the tree.”)
- Specialist sub-agents we already have (file, python exec) behind the same MCP interrupt

This is the production path. RL is only if the prompted loop plateaus on multi-hop / aggregation.

### Phase 9 — associative graph memory (not more hops)

Our hop ablation already failed ranking: more Recall@50, lower nDCG@10 (status row `3.4`: both hop settings below zero against plain hybrid). Blind expansion is the wrong next step.

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

### Phase 11 — artifacts and code production (academia + industry)

The 2024 e2b template tool in `app/engine/tools/artifact.py` is still wired to LlamaIndex. The eval/MCP path is `app/artifacts/` and does not auto-write the tree.

**Academic (what a lab actually ships)**

| piece | source | here |
|---|---|---|
| ACM Available / Functional / Reusable badges | ACM Artifact Review | `collect_run_artifact` packs config + sbatch + log + code + paper for a run id |
| Paper2Code plan → analyze → generate | Seo et al., ICLR 2026 | `app/artifacts/paper2code.py` over *this* tree, not a downloaded GitHub repo |
| Execution-grounded + claim-support | SciCode; AutoResearch 2026 | sandbox must pass; hyperparameters in generated code must appear in citations |
| Experiment pack, not a Space | NeurIPS/ICLR reproducibility | yaml + slurm + `.out` + `src/` is the artifact |

**Industry (how code actually ships in 2026)**

| piece | source | here |
|---|---|---|
| Spec card before code | Codex / Cursor / Tessl | `SpecCard` with citations + tests + template |
| SEARCH/REPLACE patches | Aider / OpenHands / Cursor apply | `propose_patch` / `apply_patch` |
| Restricted sandbox, no pip | E2B / Daytona / Modal, CI subset | AST allowlist, timeout, banned `os`/`subprocess` |
| HITL write | same as `propose_move` | `apply_artifact` needs `approved=True`; refuses failed exec |
| Templates | Claude Artifacts / v0, updated | `python-lib`, `pytest`, `jupyter-analysis`, `streamlit`, `research-repro` — not Next.js 13 |

Do **not** replace hybrid retrieve with “let the model ls and rewrite the repo.” Generation consumes the same fused hits. `make bench` stays LLM-free.

MCP additions: `collect_run_artifact`, `propose_artifact`, `apply_artifact`, `propose_patch`, `apply_patch`, `exec_sandboxed`, `inspect_spreadsheet`, `propose_visualization`, `apply_visualization`.

### Phase 12 — hard-case demos as executable evaluations

The original README stories were hand-maintained in four places. The challenge suite now starts with curated gold IDs in `bench/demo/manifest.json`, runs the real offline stack, and writes `doc/demo/traces.json` / `.js`. The interactive HTML and three new GIFs consume the same trace artifact.

Cases cover structural and semantic staleness (`q115`, `q117`), multi-hop graph cases (`q107` and `q108` with typed PPR; `q114`, where the router's gate leaves PPR off), aggregation coverage (`q100`), Deep Research (`q120`), exact-path routing (`q130`), the committed graph-hop loser (`q036` + sweeps), Paper2Code, and sandbox/HITL.

Methodology changes found while building the demos:

- PPR and retrieval scores were on incompatible scales. HippoRAG now rank-fuses base, graph, and typed-path lists and never adds an unseen candidate.
- Markdown emphasis hid `does help` / `does not help` from Tier 2. Claim text is normalized before deterministic conflict matching.
- Paper2Code could cite `draft_v1`. Research evidence now starts with hybrid top-50, expands only through the detected run entity and relevant source terms, applies version filtering, then prioritizes live config → current paper → source.
- Deep Research applies Tier 1 before its evidence scratchpad.

Synthetic artifact/safety cases are labeled outside the 136-question retrieval gold set. The blind-hop regression remains visible rather than being edited out.

### Phase 13 — spreadsheet aggregation to MATLAB, with HITL chart choice

Do not show hidden chain-of-thought. Show an auditable decision trace:

1. profile column names/types, missingness, and a five-row preview;
2. propose `group_by`, numeric metric, and `mean|sum|min|max|count`;
3. return the aggregated rows;
4. recommend a chart and list valid alternatives;
5. wait for the user to approve or override the chart;
6. generate MATLAB-compatible `.m` code and optionally render locally.

The frozen demo asks for average `val_rmse` by `encoder` against the `0.055` baseline. A dot plot is recommended over a bar chart because this is a point-estimate-to-threshold comparison. The generated MATLAB script reads and aggregates the CSV itself, then writes `doc/figures/mean-val-rmse-by-encoder.png`.

`apply_visualization` requires `approved=True`. Paths remain under the configured local root; the execution command is fixed to local MATLAB/Octave rather than model-provided shell text. CSV/TSV is dependency-free; XLSX uses optional `openpyxl`.

### Phase 14 — cutting-edge query processing (HyDE, decomposition, CRAG)

**Why.** State-of-the-art RAG pipelines (2025–2026) interleave query transformation and post-retrieval quality gates to handle complex, ambiguous, and multi-hop queries. Three techniques complement the existing hybrid pipeline:

| step | mechanism | source |
|---|---|---|
| 14.1 | **HyDE** (Hypothetical Document Embedding): embed a generated pseudo-document instead of the raw query, then RRF-fuse with original rankings | Gao et al. 2023 |
| 14.2 | **Query Decomposition**: split compound multi-hop queries into sub-queries, retrieve per sub-query, merge via best-score dedup | Decomposed Prompting (Khot et al. 2023); IRCoT (Trivedi et al. 2023) |
| 14.3 | **Corrective RAG (CRAG)**: classify each retrieved chunk as Correct / Ambiguous / Incorrect, then trigger corrective actions (refine, rewrite, or decompose) | Yan et al. 2024 |

All three are wired into `_retrieve()` in the bench harness with boolean flags. Heuristic-only in CI (no LLM); LLM-backed callables accepted for production use.

### Phase 15 — latency + safety gates (adaptive retrieval, semantic cache, citation verification)

**Why.** Production RAG systems need three things the research pipeline misses: (1) skipping retrieval entirely for queries that don't need the corpus, (2) caching semantically similar queries to avoid redundant compute, and (3) verifying that generated answers cite real evidence.

| step | mechanism | source |
|---|---|---|
| 15.1 | **Adaptive Retrieval**: rule-based classifier decides RETRIEVE / SKIP / UNCERTAIN before the pipeline runs | Self-RAG (Asai et al. 2024); FLARE (Jiang et al. 2023) |
| 15.2 | **Semantic Query Cache**: LRU cache with cosine-similarity lookup (threshold 0.92), TTL eviction, optional JSON persistence | Gptcache (Zilliz 2023); production pattern |
| 15.3 | **Citation Verification**: post-generation gate extracts claims, verifies values and paths against evidence, flags hallucinations, rejects answers below a verification ratio threshold | FActScore (Min et al. 2023); SAFE (Wei et al. 2024) |

Adaptive retrieval is neutral on the gold set (every gold question needs retrieval). Citation verification is a generation-time safety gate, not a retrieval metric. The semantic cache is meant as a latency optimization; replayed over the gold set it serves other questions' lists, a recorded loss (status row `15`).

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
| 1.5 halfvec / binary / HNSW | `app/retrieval/pgvector_tune.py` (in-memory analogue); served: the HNSW index of `app/database/migrations/vector/` and `VectorStoreManager.search_settings`; no halfvec or binary column; measured in Postgres by `app/eval/pgvector_hnsw.py` (`make bench-served-scale`) |
| 3.2 GraphRAG | `app/graph/graphrag.py` |
| 4.1 / 4.3 ColPali + CLIP | `app/multimodal/colpali.py`, `app/multimodal/images.py` |
| Phase 5 Tier 2 | `app/graph/conflicts.py`, wired into `extractive_answer` |

---

## Layout

```
app/eval/          harness, metrics, gold, latency, RAGAS-style e2e, jury, SVG figures
bench/demo/        curated hard-case IDs; no copied trace prose
doc/demo/          generated JSON/JS consumed by HTML and GIFs
app/retrieval/     RRF, BM25, router, hybrid index, distill, RankGPT, HNSW / halfvec / binary, HyDE, adaptive, cache
app/chunking/      structure-aware, late, contextual
app/graph/         file graph, version clusters, hierarchy, GraphRAG, HippoRAG PPR, conflicts
app/agent/         retrieval loop, Deep Research, move-plan schema, decompose, corrective RAG, citation verify
app/artifacts/     ACM bundles, Paper2Code, spec/patch/sandbox
doc/matlab/        generated MATLAB source for checked-in visualization demos
app/rl/            Search-R1 parse/env, reward, GRPO dummy step
app/mcp/           filesystem tools + stdio MCP server
app/multimodal/    MaxSim, ColPali-style pages, CLIP/SigLIP images, modality RRF
bench/corpus/      frozen files, MANIFEST, registry
bench/gold/        questions.jsonl
bench/results/     one JSON per git sha + sweeps.json
doc/demo.html      live gold-question traces (app palette)
doc/figures/       README SVGs
```
