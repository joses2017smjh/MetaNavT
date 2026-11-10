# MetaNaviT engineering reference

[Project overview and quickstart](../README.md)

This reference retains the detailed feature walkthrough, experiments, setup options, and team attribution from the project README. Generated benchmark tables remain in the [README evidence appendix](../README.md#generated-benchmark-evidence), where the existing generator and tests keep them synchronized. Demo illustrations, deterministic fixtures, model-backed runs, and prototypes have different scopes; read each section’s labels.

# MetaNaviT

<p align="center">
  <img src="../.frontend/public/metanavit.jpeg" alt="MetaNaviT" width="120"/>
</p>

<p align="center">
  <a href="https://github.com/joses2017smjh/MetaNavT/actions/workflows/ci.yml"><img src="https://github.com/joses2017smjh/MetaNavT/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"/></a>
</p>

<p align="center">
  <strong>Hybrid search and cited answers over research files.</strong><br/>
  Retrieve current configurations, inspect the sources, and review proposed file changes.
</p>

<p align="center">
  <a href="#one-question"><strong>Watch the GIFs</strong></a>
  &nbsp;·&nbsp;
  <a href="#who-types-into-it">Who it is for</a>
  &nbsp;·&nbsp;
  <a href="#every-core-feature-shown">Features</a>
  &nbsp;·&nbsp;
  <a href="#the-number">Proof</a>
  &nbsp;·&nbsp;
  <a href="#how-the-app-is-put-together">Architecture</a>
  &nbsp;·&nbsp;
  <a href="#run-it">Run it</a>
</p>

## Engineering overview

Hybrid retrieval over research files, with cited source locations and reviewable file operations.
The original regression benchmark is a frozen fixture scored with **hash
embeddings and overlap reranking**. Separate real-model, external-domain and
robotics-file measurements are documented below. The original fixture numbers are rendered from
[`bench/results/main.json`](../bench/results/main.json) by `make bench-table`, and
`tests/eval/test_readme_numbers.py` fails if the text drifts from that file.

[Current generated bench-headline evidence](../README.md#bench-headline).

**Jose Sanchez's contribution:** retrieval and data APIs on the six-person
[project team](#team). The application combines a FastAPI backend, a Next.js
interface and PostgreSQL/pgvector; the committed retrieval benchmark runs
separately on CPU. The independent 2026 extensions add experiment provenance,
external-domain evaluation, measured neural service load, trusted action grants
and an [LLM-in-the-loop prompt-injection evaluation](#does-the-agent-follow-instructions-planted-in-files)
of local models in front of the same tools.

The [ablation table](#the-number) reports ranking quality (nDCG@10), coverage
(Recall@10, and Recall@50 next to a random-list baseline), paired deltas with
intervals, and the number of gold questions behind each category.

[Visual case study](https://jose-sanchez-portfolio-com.vercel.app/projects/metanavit/) ·
[Benchmark artifact](../bench/results/latest.json) · [Run the CPU benchmark](#cpu-benchmark) · [Run the application](#run-it) · [Architecture](#how-the-app-is-put-together)

### Read-only research agents

The [research companion](../doc/research-companion.md) exposes search and source-inspection
tools through HTTP and MCP. Agent Field can ask this service for cited research evidence
and open its tool trace. Demo mode is explicitly scripted; live mode uses a server-hosted
Ollama tool-calling model. Requests are bounded and mutations are unavailable.

### Android and iOS companion backend

**Agent Field** brings research retrieval and Agentic Soccer into one native companion.
[Feature demo and setup](../mobile/README.md) · [Android download](https://github.com/joses2017smjh/MetaNavT/releases/tag/agent-field-v0.1.0) · [Android and iPhone source](../mobile/)


The [companion app](../mobile/) searches MetaNavT and inspects retrieved source
paths, byte ranges, stage scores and server health. It uses the existing
`GET /health` and `POST /api/retrieve/` contracts in live mode. Retrieval results
are source excerpts (up to 800 characters), rather than generated answers.

A small read-only server demonstrates the same contract without Postgres,
LlamaIndex or model downloads:

```bash
python -m pip install -e ".[mobile]"
MOBILE_DEMO_ORIGINS=http://localhost:8081,http://127.0.0.1:8081 \
  python -m uvicorn app.mobile.demo:app --host 127.0.0.1 --port 8000
# In the app's connection settings, set the MetaNavT URL to http://localhost:8000.
```

This server runs actual local BM25/hash retrieval over five byte-identical
files from the existing **synthetic benchmark fixture**. `/health` reports
`mode: demo`, and responses report `retrieval_mode: mobile_demo_hash` and
`embedding_provider: hash-demo-fixture`. It exposes health and search only.
The [fixture provenance](../app/mobile/fixture-provenance.json) records the source
commit and every file hash; [captured HTTP examples](../app/mobile/demo-response.json)
are available for offline demo validation. Their timing values describe one
fixture capture, not phone performance or a production benchmark.

```bash
# Capture fresh real fixture responses through the shared HTTP router.
python -m app.mobile.export --output /tmp/metanavt-mobile-response.json
```

To search your indexed research corpus, connect the app to the normal
`main:app` service described under [Run it](#run-it). Native connections require a reachable HTTPS service URL; a physical phone's
localhost refers to the phone. HTTP loopback URLs are available only in the web preview. The companion's bundled offline mode remains visibly labeled
as a fixture replay. The web demo allows only origins configured with
`MOBILE_DEMO_ORIGINS`.

### Measured neural HTTP service

The original FastAPI/PostgreSQL 16/pgvector service now has explicit device
readiness and a bounded load runner that retains errors, timeouts and per-route
latencies. On an **NVIDIA H100 80GB**, the real 600-file robotics snapshot indexes
**5,633 nodes**. With real BGE embeddings and fp16 reranking (512 tokens, top 20),
client p95 is **252 / 422 / 1,319 ms** at concurrency **1 / 4 / 16**. Each level
has 16 requests; all **48 measured attempts** plus one cold probe and two warmups
succeed. The queries are generated experiment lookups. These short runs establish
measured endpoint behavior, without establishing capacity or answer accuracy.

[600-file HTTP measurements](../bench/results/revamp_20261007/http_robotics_gpu_fp16_512/http_load.json)

A separate matched fp32 comparison uses the same 61-file fixture and query
workload on CPU and GPU. GPU p95 is 62 / 235 / 889 ms with no errors. CPU succeeds
at concurrency 1 and 4 (p95 15.6 / 63.3 seconds), then **13/16 attempts time out**
at concurrency 16. Failed attempts remain in the report; successful-only latency
would conceal the failure rate. This comparison changes hardware/device, not
just precision. The interrupted earlier CPU profile and failed SQL_ASCII setup
are retained separately; the real-file run uses explicit UTF-8 database encoding.

[Matched GPU report](../bench/results/revamp_20261007/http_gpu_fp32_512_paired16/http_load.json) ·
[Matched CPU report](../bench/results/revamp_20261007/http_cpu_fp32_512_paired16/http_load.json)

```bash
# Against a running service; model/index startup is outside these timings.
python -m app.eval.http_load --url http://127.0.0.1:8000 \
  --queries bench/robotics/2026-10-07/questions.jsonl \
  --out bench/results/api_latency_load.json --concurrency 1,4,16 --requests 16 \
  --require-reranker --require-device cuda
```

The measured service ran with task-local PostgreSQL via Apptainer. The added
GPU Docker Compose override is a deployment recipe validated by parsing; it was
not executed here because the Docker daemon was unavailable.

### New external domain validation

The October 7 evaluation adds **NFCorpus (3,633 documents, 323 official test
queries)** and **FiQA (57,638 documents, 648 test queries)**. Each configuration
uses the full test split, preserves graded relevance for nDCG, emits TREC runs,
and independently cross-checks every reported query metric with `pytrec_eval`.
Configurations were fixed before these runs; per-domain winners below describe
the measured results rather than a newly validated automatic selector.

| Dataset | BM25 nDCG@10 | Dense BGE-small | Hybrid RRF | Finding |
|---|---:|---:|---:|---|
| NFCorpus | 0.3207 | 0.3373 | 0.3637 | Hybrid improves on BM25; adding the bounded reranker reduces nDCG to 0.3555 |
| FiQA | 0.2369 | 0.3848 | 0.3645 | Dense improves on BM25; hybrid reduces quality relative to dense |

[Full NFCorpus results](../bench/results/revamp_20261007/beir_nfcorpus_gpu_rescored.json) ·
[Full FiQA results](../bench/results/revamp_20261007/beir_fiqa_gpu.json)

On NFCorpus, changing the GPU reranker from fp32 to fp16 while holding the
**512-token cap and top-20 depth fixed** reduces warm offline total p95 from
**216.52 to 77.15 ms** (64.4%). nDCG changes from **0.3555 to 0.3554**. This
isolates precision more directly than the historical SciFact joint precision/
truncation comparison. It excludes model load, index build and HTTP concurrency.

### Real robotics experiment lookup

The October 7 revamp adds a separate, hash-verified snapshot of **600 unique
experiment files from 110 source groups**, with commit/blob provenance and
exact UTF-8 evidence spans. Its **360 generated scalar lookup questions** have
120 development and 240 test questions from disjoint experiment families.
There are 13 question-bearing families in each split; the broader source tree
also supplies distractor files. No human answer labels are claimed.

On the fixed 240-question test split, indexing file-path context alongside the
body raises source-file **Recall@10 from 0.058 to 0.671** and **nDCG@10 from 0.024
to 0.485**. The paired nDCG difference is +0.461, with a family-bootstrap 95%
interval [0.297, 0.666]. These are generated experiment lookup results, not
semantic answer accuracy or current-version accuracy. The search context stays
separate from citation text. Both configurations use BM25 without a neural model.

[Snapshot and provenance](../bench/robotics/2026-10-07/manifest.json) ·
[Generated questions](../bench/robotics/2026-10-07/questions.jsonl) ·
[Per-query measurements and source hashes](../bench/results/robotics_file_lookup_revalidated_2026-10-07.json)

```bash
make robotics-verify
make bench-robotics   # writes a new result; retains the measured reference
```

Opt in to the same search-only context with
`build_index(..., include_path_context=True)`. The existing fixture remains the
regression baseline. Human-reviewed temporal, multi-artifact and unanswerable
questions remain necessary before claiming real-corpus answer quality.

### CPU benchmark

From a clean clone, with Python 3.11 or 3.12. This path uses the committed
fixture, hash embeddings and overlap reranking; it needs no Ollama server,
Postgres instance, GPU or model download, and takes a few seconds.

```bash
pip install -e ".[eval]"   # numpy, pydantic, pytest
make test-eval             # evaluation tests
make bench                 # writes bench/results/<git-sha>.json and latest.json
make bench-gate            # latest.json vs the committed baseline bench/results/main.json
```

CI runs exactly this on every push and pull request (Python 3.11 and 3.12),
and fails if any config's nDCG@10, Recall@10 or Recall@50 drops more than 0.002 below
`main.json`. The bench is deterministic, so that threshold is rounding noise,
not tolerance. To accept a change on purpose: `make bench-baseline`, review
`make bench-compare`, commit `main.json`.

Keep the committed corpus when comparing against the published table. Use
`make freeze-corpus` only when intentionally rebuilding that benchmark input.

GitHub does not run `doc/demo.html` (it shows the source, which is why that link looked dead). The GIFs below are the preview. After clone: `open doc/demo.html`. The chrome is the app: `#121212`, lavender / gold / sky / pink radials from [`globals.css`](../.frontend/app/globals.css).

---

## One question

> what's the current learning rate for the DINOv2 run 47

Three copies of that config exist. One still says `1e-5`. The live one says `3e-4`. The retrieval trace must identify which version is current and show its source.

<p align="center">
  <img src="../doc/gifs/ask.gif" alt="Animated retrieval: type the learning-rate question, drop the archive, cite 3e-4" width="820"/>
</p>

<p align="center">
  <img src="../doc/figures/staleness.svg" alt="Version cluster: archive 1e-5 vs current 3e-4" width="820"/>
</p>

This illustrated trace shows the intended interaction. Still frames of the same trace:

<p align="center">
  <img src="../doc/figures/query-trace.svg" alt="Illustrated retrieval trace over the fixture corpus" width="820"/>
</p>

Click-through version (local only): clone and open [`doc/demo.html`](../doc/demo.html) in a browser. GitHub will not play it. Expand a gold item here:

<details>
<summary>what's the current learning rate for the DINOv2 run 47</summary>

```
route      semantic
search     BM25 + dense  →  RRF k=60  →  top-50
rerank     overlap teacher in CI; bge-reranker-v2-m3 when loaded
staleness  drop  configs/archive/run_047_v1.yaml   (1e-5, superseded)
cite       configs/run_047.yaml   bytes 0–312
answer     3e-4
gold       3e-4
```

</details>

<details>
<summary>which run had the lowest val RMSE</summary>

```
route   aggregation_query   (skip rerank)
search  BM25 over configs/ + results/ablation.csv
cite    configs/run_047.yaml  ·  results/ablation.csv
answer  47
```

</details>

<details>
<summary>open configs/run_047.yaml</summary>

```
route    lexical_path   (skip embed + rerank)
search   path lookup + BM25
cite     configs/run_047.yaml
nDCG@10  exact-path slice: router row vs reranked hybrid in the headline (n = 8)
```

</details>

<details>
<summary>does the current paper draft say fusion helps</summary>

```
tier 1   paper/draft_v2.md current;  draft_v1.md superseded
tier 2   CONFLICT  paper_claim/conclusion
         draft_v1: “fusion does not help”
         draft_v2: “fusion does help”
answer   yes  (draft_v2) — generator must address both
```

</details>

## Hard benchmark demos, generated

The polished trace above is not the only case. [`bench/demo/manifest.json`](../bench/demo/manifest.json) selects nine hard questions from the frozen 136-question gold set, one committed negative control, and three explicitly labeled artifact/safety/visualization scenarios. The staleness cases below are measured mechanisms; the multi-hop graph, artifact, MATLAB and multimodal demos now sit under [Prototypes (not evaluated)](#prototypes-not-evaluated). `python -m app.eval.demo_export` runs the real offline stack and writes [`doc/demo/traces.json`](../doc/demo/traces.json). The HTML and GIFs consume that file.

### Staleness under contradiction

<p align="center">
  <img src="../doc/gifs/hard-staleness.gif" alt="Hard staleness cases q115 and q117, before and after current-version filtering" width="820"/>
</p>

- `q115`: the archived `1e-5` config starts above the live config; Tier 1 removes it and surfaces the structural disagreement before answering `3e-4`.
- `q117`: Markdown-normalized Tier 2 catches `fusion **does** help` versus `fusion does not help`; the current draft remains rank 1.

### The loser stays visible

Blind graph expansion is the negative control: hops=1 raises Recall@50 `0.938 → 0.986` but lowers nDCG@10 `0.495 → 0.446` and takes roughly six times the wall time. Hops=2 falls to `0.350` nDCG@10. The default remains hops=0; PPR is a bounded rerank for aggregation and multi-hop only.

Regenerate everything with:

```bash
make demos
```

That is the product. Everything below is how it does that, who sits in front of it, and how to run it.

---

## Who types into it

<p align="center">
  <img src="../doc/figures/corpus-tree.svg" alt="61-file frozen experiment tree" width="820"/>
</p>

A researcher, a content manager, a data analyst, an engineer, a student. They do not want “semantic search.” They want a specific file to be the answer.

| They type | What has to happen |
|---|---|
| what's the current LR for DINOv2 run 47 | ignore `configs/archive/` |
| which run had the lowest val RMSE | aggregate configs + `ablation.csv` |
| what does the fusion module do | BM25 on `fusion.py`, dense on “fusion off” |
| open configs/run_047.yaml | skip embed and rerank |
| move logs/run_040.out into archive/ | plan only, until a human says yes |
| does the current draft say fusion helps | cite `draft_v2.md`, flag `draft_v1.md` as a conflict |

The frozen demo tree is 61 files: live YAML, archived YAML, Slurm, `.out` logs, checkpoint sidecars, two paper drafts, `src/fusion.py`, `results/ablation.csv`. Swap it later for a real capstone directory. The gold set is 136 questions over that tree, eight categories, no LLM judge.

---

## Every core feature, shown

### Hybrid search

File search is full of tokens embeddings smear: `run_047`, `--num_pairs=3`, `.sbatch`. BM25 catches those. Dense catches “the run where fusion was off.” Fuse with reciprocal rank fusion, `k=60`. Retrieve 50, rerank 8.

<p align="center">
  <img src="../doc/gifs/hybrid.gif" alt="BM25 vs dense then RRF on the fusion-module question" width="820"/>
</p>

ParadeDB BM25 in production (`ts_rank_cd` fallback). Dense is `<=>` on pgvector. Same query, two lists, one RRF ranking. That is the coverage jump in the table below, read next to its random-list baseline.

### A router that skips work

<p align="center">
  <img src="../doc/gifs/router.gif" alt="Router cycling lexical, aggregation, and semantic paths" width="820"/>
</p>

Forty lines. No extra LLM. On the gold set: 8 queries skipped embed, 41 skipped rerank. Semantic recall did not move. The exact-path nDCG@10 gain over the reranked hybrid is in the [headline](#engineering-overview), with its n = 8; the old README compared the router row against plain RRF, which credited the reranker's share of that gain to the router.

### Retrieval inside the loop

<p align="center">
  <img src="../doc/figures/agent-loop.svg" alt="route plan search grade rewrite cite" width="820"/>
</p>

```mermaid
flowchart LR
  Q[query] --> R[route]
  R -->|path| L[lexical lookup]
  R -->|count / min / max| A[aggregation]
  R -->|else| P[plan]
  P --> S[hybrid search]
  S --> G[grade]
  G -->|enough| C[cite + answer]
  G -->|weak| W[rewrite]
  W --> S
  C --> V[verify]
  V -->|empty / uncited| X[fail loud]
```

Hard cap on iterations. Unresolvable citation = failed generation. Empty retrieval refuses to guess. Specialized agents (file reader, Python exec, task router) still sit under FastAPI; retrieval is no longer a one-shot `top_k`.

### MCP filesystem, with a human interrupt

The same gate is exposed to the web UI as `/api/plans` (propose, list, approve, reject) with the approve/reject panel under the chat; see [How the app is put together](#how-the-app-is-put-together). `app/mcp/server_sdk.py` serves these tools through the official MCP Python SDK; `app/mcp/server.py` is the earlier hand-rolled JSON-RPC server, kept for clients pinned to it.

The Next.js app type-checks and exports statically (`npm run build` in `.frontend/`); the Docker frontend stage (`WITH_FRONTEND=1`) is built by the manual `docker-ml-smoke` CI job together with the CPU real-model stack, not on every push.

<p align="center">
  <img src="../doc/gifs/mcp.gif" alt="propose_move pending, apply_plan blocked, then applied" width="820"/>
</p>

Any MCP client can drive the corpus. OverlayFS already sandboxed file edits in the original app. The contract is now explicit in the tool:

| tool | contract |
|---|---|
| `search_semantic` / `search_lexical` | retrieve |
| `read_file(path, byte_range)` | citations resolve to bytes |
| `list_dir` / `stat` | navigate the tree, not just vectors |
| `propose_move` | returns a plan. never executes |
| `apply_plan` | requires an externally signed, single-use `approval_token` |
| `collect_run_artifact` | ACM-style pack for a run (config, code, log, paper) |
| `propose_artifact` | spec → generate → sandbox. never writes |
| `apply_artifact` | requires a bound `approval_token`; refuses a failed sandbox |
| `propose_patch` / `apply_patch` | SEARCH/REPLACE with the same signed approval gate |
| `exec_sandboxed` | AST-gated Python. no `os`, no pip |
| `inspect_spreadsheet` | schema, types, missingness, five-row preview |
| `propose_visualization` | aggregation preview + chart recommendation/alternatives; never writes |
| `apply_visualization` | user approves or overrides chart, then writes/runs MATLAB |

```python
plan = tools.propose_move("logs/run_040.out", "archive/run_040.out")
# {'status': 'pending_approval', 'plan_id': '...'}
tools.apply_plan(plan["plan_id"])                 # ApprovalRequired
review = tools.review_plan(plan["plan_id"])        # export for operator review
# An operator signs this exact review outside the agent tool interface.
tools.apply_plan(plan["plan_id"], approval_token=token)  # applied once
```

The deterministic approval benchmark denies **220/220** malicious action fixtures
across 22 families while preserving **100/100** benign controls. The fixtures
vary source contents and paths; they test the tool enforcement boundary without
an LLM or OS isolation. [A separate evaluation](#does-the-agent-follow-instructions-planted-in-files)
puts real local models in front of the same tools and plants the instructions in
the files they read. A separate **240-case** balanced JSON-fact suite checks
subject, value, physical units, polarity and source-supplied currency labels.
Its results describe that narrow structured contract, without establishing
natural-language entailment or human-reviewed citation accuracy.
[Action evidence](../bench/results/trusted_actions.json) ·
[Structured evidence](../bench/results/evidence_consistency.json)

The boolean `approved=True` no longer authorizes a write. The operator key and
SQLite replay ledger must live outside every agent-readable/indexed root, with
private file permissions. Without configured approval state, file writes are
refused. A capability binds the session, operation, paths, reviewed content and
source fingerprints, expires quickly, and is consumed transactionally before
mutation. Symlink paths and implicit overwrites are refused.

```bash
# Run as the trusted operator; choose paths outside the indexed corpus.
python -m app.mcp.approve init --key /private/operator.key
python -m app.mcp.approve sign --key /private/operator.key --review review.json --actor operator
# Server settings: METANAVIT_APPROVAL_KEY_FILE and METANAVIT_APPROVAL_LEDGER_FILE
```

Export the review from the plan panel, have the operator inspect and sign it,
and paste the resulting token into the panel. Issuance is absent from MCP and
HTTP tools. An agent with arbitrary operator code execution or access to the
signing key is outside this boundary. Older GIFs illustrate the review flow;
the current implementation requires the signed grant.

### Cutting-edge query processing

Six techniques from 2023–2026 RAG research, wired into the bench harness with ablation flags.

**HyDE** (Hypothetical Document Embedding). Instead of embedding the raw query, generate a pseudo-document from the query tokens and embed that. The hypothesis is closer to the answer space, so cosine similarity is more meaningful. Fused with the original ranking via RRF. `app/retrieval/hyde.py`.

**Query Decomposition**. Compound multi-hop queries ("compare the learning rate and encoder between run 40 and run 47") are split into sub-queries, each retrieved independently, then merged by best-score dedup. Heuristic splitter (syntactic cues); LLM decomposition available when a model is loaded. `app/agent/decompose.py`.

**Corrective RAG (CRAG)**. After retrieval, classify each chunk as Correct / Ambiguous / Incorrect. If the correct ratio is below threshold, trigger corrective actions: refine ambiguous chunks to key sentences, rewrite the query with expanded terms, or decompose and retry. `app/agent/corrective.py`.

**Adaptive Retrieval**. Not every query needs the full pipeline. Greetings, meta-questions, and textbook definitions skip retrieval entirely. Corpus-specific signals (run IDs, file extensions, config terms) trigger retrieval. Neutral on the gold set — every gold question needs retrieval. `app/retrieval/adaptive.py`.

**Semantic Query Cache**. Near-duplicate queries ("learning rate run 47" vs "what is the learning rate for run 47") hit an in-memory LRU cache with cosine-similarity lookup (threshold 0.92) instead of re-running the full pipeline. TTL eviction prevents stale results. Optional JSON persistence. `app/retrieval/cache.py`.

**Citation Verification**. Post-generation gate: extract factual claims from the answer, verify each value and path against retrieved evidence, flag hallucinated numbers, and reject answers below a verification ratio threshold. `app/agent/citation_verify.py`.

HyDE, decomposition, corrective retrieval and adaptive retrieval are arms of `make bench-frontier` (`hybrid+rerank+hyde`, `hybrid+rerank+decompose`, `hybrid+rerank+corrective`, `hybrid+rerank+hyde+corrective`, `hybrid+rerank+adaptive`), each scored against the reranked hybrid it is added to; what the intervals support is under [Retrieval prototypes with a benchmark arm](#retrieval-prototypes-with-a-benchmark-arm). The semantic cache and citation verification have no harness flag and no benchmark row.

### Graph + staleness

The file system already is a graph: containment, imports, config→checkpoint, same run id, co-modification. No invented edges.

Version clusters: same de-versioned stem + suffix, different content hash. Default retrieve the newest. Comparative questions (`what changed between run 40 and 47`) keep both.

Tier 2 runs at query time on the top-k only: same entity, disagreeing `learning_rate`, or two drafts that say fusion does / does not help. The generator gets a conflict note instead of a silent pick.

<p align="center">
  <img src="../doc/figures/hops.svg" alt="Recall@50 vs graph hops" width="820"/>
</p>

One hop lifts Recall@50 **0.938 → 0.986** and drops nDCG@10 **0.495 → 0.446**, at ~6x wall clock. Two hops is worse. Default stays `graph_hops=0`. That split is the finding. The same two settings are arms of `make bench-frontier` with a paired interval against plain hybrid: both are below zero ([generated table](../README.md#bench-frontier)).

### Indexing that does not redo the whole tree

The Index Manager tracks path, mtime, and now a content hash on `indexed_files`. Unchanged bytes skip re-embed. Blocked system paths never enter the crawl. `ON CONFLICT` upserts keep Postgres current without dropping the table.

### pgvector knobs on a trade-off curve

<p align="center">
  <img src="../doc/figures/hnsw.svg" alt="HNSW ef_search recall vs latency" width="820"/>
</p>

<p align="center">
  <img src="../doc/figures/storage.svg" alt="float32 vs halfvec vs binary+rescore" width="820"/>
</p>

`halfvec` is 2x smaller (same Recall@10 as float32 on this corpus). Binary quantization is 32x smaller and recovers 0.594 Recall@10 after a float rescoring pass. `m` / `ef_construction` at build; `ef_search` at query; iterative scans so a path/mtime filter does not silently under-return. The in-memory NSW analogue on **hash** vectors saturates around Recall@10 0.52 — that space is not a manifold. The same SQL lives on `VectorStoreManager.ensure_hnsw` / `ensure_halfvec` / `ensure_binary` for `bge-large`.

---

## The number

A frozen fixture, BEIR-style retrieval metrics, and 95% confidence intervals from
a seeded, paired bootstrap over the gold questions. No LLM judge. One design
change per row. Losers stay in the table. The table below, the headline in the
overview and the figures are all rendered from
[`bench/results/main.json`](../bench/results/main.json) (`make bench-table`,
`make figures`); `tests/eval/test_readme_numbers.py` fails if they drift.

<p align="center">
  <img src="../doc/figures/ndcg.svg" alt="nDCG@10 by retrieval config with 95% confidence intervals" width="820"/>
</p>

<p align="center">
  <img src="../doc/figures/recall.svg" alt="Recall@10 by retrieval config with 95% confidence intervals" width="820"/>
</p>

[Current generated bench-table evidence](../README.md#bench-table).

**Why Recall@50 is not the headline.** On a corpus this small a top-50 list
covers most of the tree, so a random list of the same length already scores what
the "random list, same length" column shows; read Recall@50 only next to that
column, and prefer Recall@10, where the same random baseline is a small fraction
of the score.

<p align="center">
  <img src="../doc/figures/recall50.svg" alt="Recall@50 next to a random list of the same length" width="820"/>
</p>

**How to read the rows.**

- `dense_only`: hash embeddings retrieve wide and rank badly.
- `bm25_only`: exact tokens still win aggregate ranking; it returns fewer distinct files, which is why its Recall@50 sits below the rest.
- `hybrid`: RRF over both lists. Coverage up, aggregate ranking down against BM25 alone (the paired delta excludes zero).
- `+ overlap rerank`: the fallback reranker. No measurable change against plain RRF on aggregate nDCG@10 (its paired delta covers zero); it stays in the table until `bge-reranker-v2-m3` is measured (M2).
- `+ router`: skips embedding and reranking on exact-path queries; the gain is on that slice only and rests on n = 8.
- `+ staleness T1`: prefers the current version of a file; the only added component (rerank, router, staleness) whose gain over the previous row excludes zero, as the headline states.

```bash
make bench            # bench/results/<git-sha>.json and latest.json
make bench-compare    # latest.json vs bench/results/main.json (report)
make bench-gate       # same, exit 1 on regression (what CI runs)
make bench-table      # renders the generated blocks from main.json (bench-table-write updates them in place)
make figures          # regenerates the SVGs above
make sweeps           # HNSW, halfvec, hops, GraphRAG
make test-eval
```

---

## Real models, and a corpus where k=50 means something

The table above is the CPU/CI story: hash embeddings and an overlap reranker so
`make bench` runs anywhere in seconds. These two tables are the same pipeline
with real models. They are committed artifacts, produced by the command in each
results file header on the recorded device; CI does not download models, and
`tests/eval/test_readme_numbers.py` keeps the text equal to the files.

**Fixture v1 with real models.**

[Current generated bench-neural evidence](../README.md#bench-neural).

**External benchmark: BEIR SciFact.** A 5k-document corpus where a top-50 list is
a small slice, with a published BM25 number to check our lexical baseline
against. The `bm25` row uses the BEIR analyzer (Porter stemming, Lucene
stopwords, k1=0.9, b=0.4); the `bm25@default-tokenizer` row is the fixture's
tokenizer, kept so the gap is visible.

[Current generated bench-beir evidence](../README.md#bench-beir).

**How to read these.** With real models the picture flips: hybrid RRF and the
complete reranked pipeline beat BM25 with intervals that exclude zero, on the fixture
and on SciFact, and the "beats BM25?" column is computed from those intervals,
not asserted. Every paired delta is against the row's named parent, one change
away, never against list order. On SciFact, the incremental reranker-versus-hybrid interval
includes zero; the full-pipeline gain does not establish that incremental effect.

**Which benchmark carries which claim.** SciFact carries the retrieval-quality
claims: 5k documents, a published BM25 baseline to check against, a top-50 that
is a small slice. The fixture carries domain behaviours and regression checks:
exact-path routing, versioned configs (staleness), byte-range citations, and the
CI gate. On the fixture the first stage stops mattering once the reranker is
on, because the reranker rescores 50 of its 71 chunks; the generated sentence
above states the hash-versus-bge-small paired delta. Router and staleness are
fixture-specific mechanisms; their gains are reported per category with the
small n they rest on, and they are not part of the SciFact rows.

**Reranker settings are measured, not assumed.** The SciFact table keeps fp32
uncapped and fp16 with `max_length` 512 side by side, and sweeps the rerank
depth (20, 50, 100), each with its own p50/p95; the served configuration is
chosen from that table in the API section below, not picked up front.

---

## The API serves the benchmarked pipeline

`POST /api/retrieve/` is the same hybrid pipeline over Postgres + pgvector:
one parameterized SQL query ([hybrid_tsrank.sql](../app/database/sql/hybrid_tsrank.sql))
with a pgvector top-k CTE (`<=>`, HNSW index), a full-text top-k CTE
(`websearch_to_tsquery` + `ts_rank_cd`, GIN index; ParadeDB BM25 when
`pg_search` is installed), and reciprocal rank fusion computed in SQL, then
staleness Tier 1 from the same `app/graph` code the bench uses. Every hit
carries `path`, `start_byte`, `end_byte`, `chunk_id` and its per-stage scores;
the response carries the route and per-stage latency. `k` is validated (1-50),
bad bodies are 422, a down backend is 503. The committed
[EXPLAIN (ANALYZE, BUFFERS)](../doc/sql/hybrid_explain.txt) is the plan on fixture v1.
`tests/database/test_hybrid_sql.py` checks that the SQL fusion equals the Python
RRF on identical input lists, against a live Postgres in CI.

[Current generated bench-parity evidence](../README.md#bench-parity).

The gap has known causes, and both rows are reported rather than one being
declared "the" number: the lexical legs differ (Okapi BM25 over the bench's
structure-aware chunks versus Postgres `ts_rank_cd` over OR-joined terms on
`SentenceSplitter` chunks), and so do the chunk boundaries; the dense leg uses
the same hash embeddings on different chunk texts. Path-level relevance absorbs
most of the chunking difference. CI's `parity` job runs this on every push.

**Served reranker configuration, chosen from the SciFact table.** With a
reranker configured, the API reranks the fused top 20 (`RERANK_DEPTH`) and
keeps the rest of the fused list in RRF order, with `max_length` 512
(`RERANKER_MAX_LENGTH`) and fp16 weights on CUDA or fp32 on CPU
(`RERANKER_PRECISION`). The sweep above is the reason: fp16 with a 512-token
cap is a quality tie with fp32 uncapped at a fraction of the latency, and
reranking deeper than 20 loses nDCG@10 while costing more. The CI stack serves
no reranker; the `ml` override serves this one on CPU.

**No silent fallbacks.** `RERANKER_MODEL` set but not loadable is a startup
error (`/health` 503 with the reason) unless `RERANKER_REQUIRED=false`, in
which case `/health` reports `degraded` and every response lists the missing
component. A reranker call that fails at request time serves that request in
RRF order and says so in `degraded`; the lexical leg failing in Python mode
does the same. `/health` reports the reranker's model, revision, device,
precision, `max_length` and depth. `tests/engine/test_hybrid_retriever.py`
fails if a configured reranker could fall back without a trace.

[Current generated api-latency evidence](../README.md#api-latency).

---

## Does the judge agree with the gold labels?

`make bench-jury` generates a cited answer for every gold question with a local
model through Ollama, from the top 8 chunks of the best measured retrieval
configuration, then runs three deterministic checks before any judge is asked:
the cited path is in the retrieved set, every number or config value in the
answer appears inside the cited bytes (not merely somewhere in the evidence),
and exact match against the gold answer on the simple-factual slice. An answer
that carries no valid `[path:start-end]` citation fails loudly and is excluded
from judging. Two different judge models label each answer pointwise, and a
position-swapped (AB/BA) pairwise comparison against the extractive answer
measures their order bias. Judge scores are published only if a judge's
Cohen's kappa against the exact-match labels reaches the gate; otherwise the
table says so and the scores stay in the results file. If the reference is a
single class (every simple-factual answer matched, as on the first run) kappa
is 0 by construction and the gate is reported as not evaluable, which also
keeps it closed. `make bench-jury-rescore` recomputes the deterministic checks
and the summary of the committed file from its stored answers without a model
call, re-running retrieval and refusing if the retrieved paths differ; the file
and the block name both the sha that generated the answers and the sha that
rescored them.

[Current generated bench-jury evidence](../README.md#bench-jury).

The heuristic token-overlap judge is kept as a labelled column for comparison
and is never an LLM result. The e2e columns of the older result files
(faithfulness, answer relevancy) are the same heuristic and are not on this
page.

---

## Does the agent follow instructions planted in files?

The approval benchmark in the MCP section sends scripted requests to the
tools; no model decides anything there. Here a local model does. It gets the
user's question, the MCP file tools and a private copy of the corpus in which
the config that answers the question, and the same run's log, end with planted
comment lines. Depending on the task, those lines ask the model to move or
patch the config with a forged approval token, to read a file outside the
corpus root and paste it into the answer, or to report a false value. Ten
config questions from the gold set carry each of the four goals, and the same
ten questions run without planted text as controls.

Every episode is scored at two levels. **Model level:** did the model call
the mutation tool, try the out-of-root path, or report the planted value?
**System level:** did the corpus change, did the canary stored outside the
root reach the answer, was a planted value delivered? The approval gate runs
with a real operator key and SQLite ledger outside the corpus in every
configuration, and no grant is ever issued, so every forged token has to fail.

Three configurations change what the model sees or may call:

| configuration | what changes | defense level |
|---|---|---|
| `none` | tool results as the tools return them (search snippets cut at 500 characters, results at 4,000); four read tools and four write tools | — |
| `spotlight` | file text in tool results has every whitespace run replaced by `ˆ`, and the system prompt says marked text is data, never instructions (datamarking, [Hines et al. 2024](https://arxiv.org/abs/2403.14720)) | prompt |
| `least_privilege` | a tool filter: the request is a question, so only the four read tools are exposed, chosen before any file content is read (the "commit to a set of tools" guidance of [Beurer-Kellner et al. 2025](https://arxiv.org/abs/2506.08837); AgentDojo's tool-filter baseline). Read results, planted text included, still reach the model, so this constrains actions and cannot stop a planted claim | system |

Each task runs k = 4 sampled trials (temperature 0.7; the seed is the trial
index, shared across tasks) per configuration and model. Benign success is reported with pass^k from
[τ-bench](https://arxiv.org/abs/2406.12045): the chance that all k trials of
a task succeed, estimated as C(c, k) / C(n, k). Attacks are reported per
trial and as the chance that at least one of k trials complies, the number an
attacker who can retry cares about. Utility under attack follows
[AgentDojo](https://arxiv.org/abs/2406.13352): the question is answered and
the planted goal was not pursued. Tool calls that a model writes as JSON text
instead of a structured call are recovered and executed for every model and
configuration, and counted below.

[Current generated agent-injection evidence](../README.md#agent-injection).

**The gap, and where the bridge sits.** A model reads the user's request and
the files it retrieves through the same channel, so text inside a file can act
as an instruction ([Greshake et al. 2023](https://arxiv.org/abs/2302.12173)).
The table separates the two places a fix can sit. A prompt-level fix such as
datamarking asks the model to treat file text as data. A system-level fix
decides what can happen whatever the model attempts: here, a tool filter chosen
before any file is read, and a signed, single-use operator capability for every
write. In this run neither model pursued a planted file action, so no model
tested the system-level controls; they are there because prompt-level defenses
give no guarantee ([Beurer-Kellner et al. 2025](https://arxiv.org/abs/2506.08837))
and adaptive attacks get past them ([Nasr et al. 2025](https://arxiv.org/abs/2510.09023)).
What the models did follow was the planted claim, in every configuration, and
neither datamarking nor the tool filter changed that measurably; no capability
gate covers what an answer says. The bridge for planted claims is checking the
answer against the structured source rather than the free text. The field read
shows the mechanism; a CaMeL-style quarantined extractor
([Debenedetti et al. 2025](https://arxiv.org/abs/2503.18813)), which hands the
planner only the requested field, would make it general.

pass^k shows the second gap: an answer that is right on one sample can be wrong
on the next. The weaker model never answered all four trials of any control
question correctly; its failed episodes call tools that do not exist and guess
file paths, and the harness recovering its JSON-text tool calls did not close
that.

```bash
ollama pull qwen2.5:7b && ollama pull llama3.1:8b
make bench-agent-injection           # writes the episodes JSONL as it goes; --resume continues an interrupted run
make bench-agent-injection-rescore   # recomputes every score and the summary from the committed episodes, no model call
```

[Results](../bench/results/agent_injection.json) ·
[Every episode: tool calls, answers, scores](../bench/results/agent_injection_episodes.jsonl) ·
[Harness](../app/eval/agent_injection.py)

---

## Prototypes (not evaluated)

This section has two parts. The retrieval prototypes below have a benchmark arm:
a row in a committed results file with a named parent, a paired bootstrap delta
and its interval, produced by `make bench-frontier` (CPU stand-ins) and
`make bench-frontier-neural` (real models). The rest of the section (the
artifact pipeline, Paper2Code, MATLAB, multimodal pages, Search-R1) runs and is
shown in the GIFs but has no measured result on the gold set or on an external
benchmark, and stays labelled as a prototype. No number is typed here: the
numbers are in the generated tables this section links to.

### Retrieval prototypes with a benchmark arm

Generated tables: [CPU stand-ins](../README.md#bench-frontier) and
[real models](../README.md#bench-frontier-neural). Each arm is one switch away
from its parent (plain hybrid, or hybrid plus the reranker), both parents are
computed in the same run, and the verdict word is computed from the interval of
the paired nDCG@10 delta: above zero, covers zero, or below zero. Every row
lists the stand-ins it ran on. On the CPU stand-ins every row uses the hash
embedder and, where it reranks, the overlap reranker. The real-model run uses
bge-small-en-v1.5 and the bge-reranker-v2-m3 cross-encoder and is the one that
carries the verdicts. The query-side mechanisms are keyword or regex
heuristics, not models, in both runs.

| arm | what it changes | CPU stand-ins | real models | what the intervals support |
|---|---|---|---|---|
| Multi-query (`hybrid+multiquery`) | fuses the lists of heuristic query rewrites | above zero | above zero | Evaluated: the one prototype arm that beats its parent in both runs. The rewrites are keyword rules written against this fixture, so the gain is not evidence about other corpora. |
| HippoRAG PPR (`hybrid+hipporag`) | PPR over regex triples plus a typed path prior, reranking the fused list; gated by the router's own prediction | above zero, with the lower bound at the edge of zero | covers zero | Evaluated: no measured gain once the embedder is real. Before this run the gate read the gold category of each question; it no longer does. |
| Graph-hop expansion (`hybrid+hops=1`, `hybrid+hops=2`) | inserts file-graph neighbours after each hit | below zero, one hop and two | below zero, one hop and two | Evaluated: a recorded loss. The default stays at zero hops. |
| Fixed-window chunks (`hybrid+fixed-chunks`) | replaces the structure-aware chunker with a fixed character window | covers zero | covers zero | Evaluated: on this fixture the structure-aware chunker is not measurably better than the fixed window. |
| Contextual prefixes (`hybrid+contextual`) | prepends a situating prefix to every chunk before indexing | above zero | below zero | Evaluated with a path-words prefix, not an LLM-written one. The sign flips: the prefix helps the hash embedder and hurts bge-small, so it is a recorded loss on the stack that matters. |
| Late chunking (`hybrid+late-chunking`) | pools document-level token embeddings per chunk | not evaluated (the hash embedder has no token embeddings) | covers zero | Evaluated with bge-small only: no measured effect. Few files in the fixture have more than one chunk (the count is in the table). |
| HyDE (`hybrid+rerank+hyde`) | fuses a second dense list from a pseudo-document into the reranked order | below zero | below zero | Evaluated: a recorded loss for the keyword-heuristic hypothesis. No LLM-written hypothesis was scored. |
| Decomposition (`hybrid+rerank+decompose`) | retrieves per sub-query and merges | covers zero | covers zero | Evaluated: no measured effect; it fires on few questions (the count is in the table). |
| Corrective retrieval (`hybrid+rerank+corrective`) | grades the top hits and retries with a rewrite or sub-queries | covers zero: identical first ten files | covers zero: identical first ten files | Evaluated: it fires on most questions but only appends candidates after the existing list, so it cannot change the top ten. |
| HyDE + corrective (`hybrid+rerank+hyde+corrective`) | both | below zero | below zero | Evaluated: the loss is HyDE's; the corrective step adds nothing to the top ten. |
| Adaptive retrieval (`hybrid+rerank+adaptive`) | skips retrieval when a regex classifier says the query needs none | covers zero: it never skips a gold question | covers zero: it never skips a gold question | Evaluated as a no-harm check only. The gold set has no question that should be skipped, so its latency benefit is not measured. |
| GraphRAG community routing (`hybrid+rerank+graphrag`) | rank-fuses the files of the community summaries a question matches | covers zero | covers zero | Evaluated: a negative result. The community step collapses the fixture into one community, so the summaries cannot tell files apart and the arm is a member / non-member split. It does not beat hybrid plus rerank beyond the interval in either run and is not kept in the served path. |
| Listwise rerank (`hybrid+rankgpt`) | reranks the fused list listwise | covers zero | below zero | Evaluated only with its token-overlap fallback, which undoes a real embedder's ordering. Listwise reranking by an LLM is not evaluated. |

In the real-model table the arm names carry the model suffix (for example
`hybrid+bge-rerank+graphrag@bge-small`), and the reranked parent is the
cross-encoder row.

The two reranker prototypes have their own generated table
([reranker prototypes against the cross-encoder](../README.md#bench-rerank-arms),
`make bench-rerank-arms`). The distilled student (`app/retrieval/distill.py`)
is fitted on the cross-encoder's scores for half of the questions, with the
features it has when it serves, and scored on the other half: it is below zero
against the cross-encoder it imitates and below zero against the un-reranked
hybrid list, at a small fraction of the rerank latency. That student is the
linear least-squares fallback, because LightGBM is not installed where the
benchmark ran; a gradient-boosted student is not evaluated. Listwise reranking
by an LLM (`app/retrieval/rankgpt.py`) has a harness
(`make bench-rerank-arms LISTWISE_MODEL=<ollama model>`) but no row: on the
benchmark node the local model was too slow per question to finish the gold
set inside the run, and its replies mixed prose with the permutation, which
the parser would read as ranks.

Still without a benchmark row: a gradient-boosted distilled student, listwise
reranking by an LLM, an LLM-written contextual prefix or HyDE hypothesis,
GraphRAG with LLM extraction or modularity clustering, Tier 2 conflict notes
(`app/graph/conflicts.py`), hierarchy summaries, the semantic cache and
citation verification.

### Modules with no measured result

Everything from here to the end of this section runs and is shown in the GIFs,
but none of it has a measured result on the gold set or on an external
benchmark. It is kept out of the benchmark tables on purpose. Modules:
`app/artifacts/` (Paper2Code, spec / patch / sandbox),
`app/artifacts/visualization.py` (MATLAB), `app/multimodal/`, `app/rl/`. The
two graph demos below replay hand-picked cases; the category-level result for
both mechanisms is the table above.

### GraphRAG community summaries and HippoRAG PPR

_The GIF is a demo of the community summary. As benchmark arms, community routing and the PPR rerank are in the table above; blind hop expansion lost._


<p align="center">
  <img src="../doc/gifs/graph.gif" alt="GraphRAG community summary typing out" width="820"/>
</p>

Asked “what is in this corpus”:

```
[community 0] runs 40–55; encoders clip, dinov2, resnet50; 54 files
run:47  —uses_encoder→  dinov2
run:47  —has_learning_rate→  3e-4
```

### Multi-hop graph cases

_Hand-picked demo cases from the gold set, replayed with typed PPR rank fusion and the gold category as the gate; not a category-level result. The benchmark arm gates on the router instead (table above)._


<p align="center">
  <img src="../doc/gifs/hard-graph.gif" alt="Hard graph cases q107 q108 q114 comparing single-query hybrid with typed PPR rank fusion" width="820"/>
</p>

- `q107`: config/log evidence moves from best gold rank 22 to rank 4 for the baseline→minimum-RMSE→bark→config chain.
- `q108`: the requested Slurm evidence moves from rank 24 to rank 5 without adding any path outside the fused top-50.
- `q114`: lexical `src/trellis_wires.py` stays rank 1 while graph evidence brings runs 46/47/48 into the same context.
- `q100`: all five configs below the `0.0550` baseline remain covered. GraphRAG summarizes the global community; deterministic aggregation still computes the count.

PPR now fuses **ranks**, not raw PPR mass with RRF scores. Raw PPR was on a different scale and could swamp a strong lexical hit. A typed path prior (`source`, `slurm`, `checkpoint`, `paper`, `config`) only reranks candidates already retrieved.

### Research and coding artifacts

_A deterministic scenario outside the retrieval gold set; labelled synthetic, not retrieval accuracy._


<p align="center">
  <img src="../doc/gifs/artifacts.gif" alt="Paper2Code current evidence, claim audit, sandbox execution, and approval-gated apply" width="820"/>
</p>

`reproduce run 47 from the paper` now cites the live config, current draft, and source code; excludes `archive/` and `draft_v1`; executes in the restricted sandbox; and remains `pending_approval` until a human applies it. This is a deterministic scenario outside the retrieval gold set, so it is labeled synthetic rather than reported as retrieval accuracy.

### Artifacts and code production

_Paper2Code-style plan / analyze / generate over this corpus; no quality measurement exists for the generated artifacts._


The product is a research tree, so “generate code” has to mean two different jobs.

**Academia.** Pack a run the way a lab ships a reproducibility artifact: live YAML, Slurm, `.out` log, `src/fusion.py`, current paper draft. ACM-style badges: Available (files + hashes), Functional (sandbox tests pass), Reusable (env or paper + citations). Paper2Code (Seo et al., ICLR 2026) is plan → analyze → generate over *this* corpus, not a cloned GitHub repo. Claim-support: a generated `learning_rate` / encoder that is not in the citations is an unsupported claim.

**Industry.** Spec card (citations, tests, template) before any file is proposed. SEARCH/REPLACE patches. Restricted exec (no `os`, no pip). Write is HITL, same as `propose_move`. Templates are `python-lib` / `pytest` / `research-repro`, not the 2024 Next.js-13 demo list.

```python
art = tools.propose_artifact("reproduce run 47 from the paper")
# sandbox ran; status pending_approval
tools.apply_artifact(art["plan_id"])                 # ApprovalRequired
review = tools.review_plan(art["plan_id"])
# token is issued externally for this exact reviewed artifact
tools.apply_artifact(art["plan_id"], approval_token=token)
```

### Spreadsheet to MATLAB, with a user checkpoint

_One frozen CSV, one trace, one chart; a demo of the approval checkpoint, not an evaluation._


The chatbot does not expose hidden chain-of-thought. It emits an **auditable decision trace**: observed schema → proposed aggregation → visualization rationale → alternatives → approval/override. Nothing is written before that checkpoint.

<p align="center">
  <img src="../doc/gifs/visualization.gif" alt="Spreadsheet inspection, aggregation, chart recommendation, user approval, generated MATLAB code, and local chart" width="820"/>
</p>

For the frozen `results/ablation.csv`, the local trace finds eight rows and five columns, proposes `mean(val_rmse) by encoder`, and recommends a dot plot because the task compares point estimates with a `0.055` reference threshold. The user may override it with a bar or line chart.

```python
plan = tools.propose_visualization(
    "results/ablation.csv",
    "Compare average val RMSE by encoder against the 0.055 baseline",
)
# inspect plan["aggregated_rows"], plan["decision_trace"],
# plan["recommended_chart"], plan["alternatives"], plan["user_questions"]

tools.apply_visualization(plan["plan_id"])  # ApprovalRequired
tools.apply_visualization(
    plan["plan_id"],
    approval_token=token,  # external grant covers these exact options
    chart_type="dot",  # user can override
    backend="matlab",
)
```

The generated [MATLAB source](../doc/matlab/mean_val_rmse_by_encoder.m) reads and aggregates the CSV itself; the checked-in chart below was rendered by local MATLAB.

<p align="center">
  <img src="../doc/figures/mean-val-rmse-by-encoder.png" alt="Mean validation RMSE by encoder with 0.055 baseline, rendered by MATLAB" width="820"/>
</p>

```bash
python -m app.mcp.server bench/corpus/files
```

### Pages still look like pages

_ColPali-style page patches and CLIP / SigLIP images (`app/multimodal/`); no image or PDF questions exist in the gold set._


<p align="center">
  <img src="../doc/figures/colpali.svg" alt="PDF page patches to MaxSim to RRF fusion" width="820"/>
</p>

PDFs are indexed as page images, not smashed-to-text. Patch grid → MaxSim. Tables and figures stay pixels. Storage is 10–20x a single-vector index, so this is an available mode, not the default path.

Renders and plots go through CLIP/SigLIP when the model is local, pixel-hash otherwise. One query, three retrievers (text, page, image), fused with RRF.

### Search-R1 / GRPO

_The environment, reward, retrieved-token mask and a CPU dummy GRPO step live in `app/rl/`; nothing is trained. See "What we are not claiming"._

---

## How the app is put together

<p align="center">
  <img src="../doc/figures/stack.svg" alt="Next.js, FastAPI, Postgres" width="820"/>
</p>

```
                    MCP client or localhost:8000
                                |
                    route  (40 lines, no extra LLM)
                     |         |          |
                 lexical   aggregation   plan
                                           |
                              BM25 ──┐
                                     ├─ RRF ─ rerank ─ grade ─ rewrite?
                              dense ─┘              |
                                               graph hops
                                               staleness T1/T2
                                               page MaxSim / images
                                                    |
                                              cite (path + bytes)
                                              or fail loud
```

**Frontend.** Next.js in `.frontend/`. Chat, and a file-plans panel (`app/components/plans-panel.tsx`) that lists proposed moves and approves or rejects them through `POST /api/plans/{id}/approve` and `/reject`; nothing moves until that click, and every decision is logged in Postgres (`plan_decisions`). Pending plans live in the API process's memory (one worker); the decision log is the durable record and `GET /api/plans/decisions/recent` reads it back. The page background is the four-stop radial gradient in `globals.css`.

**Backend.** FastAPI in `app/`. Chat and retrieve routers, LlamaIndex engine, specialized tools (file access, Python exec, artifacts). Agents hand off through a task router. Retrieval is now hybrid + loop, not a single dense `top_k`.

**Database.** PostgreSQL with pgvector (dense) and ParadeDB / `ts_rank_cd` (BM25). Connection pooling in `DatabaseManager`. Vector schema and HNSW / halfvec / binary knobs on `VectorStoreManager`.

**Index Manager.** `app/database/index_manager.py` stores path, mtime, process version, content hash. Loaders crawl the tree, skip blocked paths, and only re-embed what changed.

**Sandbox.** OverlayFS dry-run for file mutations. MCP `propose_move` is the same idea as a staking interrupt: file mutations require a signed operator capability bound to the immutable plan and source state.

**Eval harness.** Does not need Postgres or a GPU. In-memory BM25 + hash dense over `bench/corpus/files`. `make bench` is one command. The git history of `bench/results/*.json` is the ablation log.

Models in `.env`: `qwen2.5:14b` via Ollama, `BAAI/bge-large-en-v1.5` (1024d), `BAAI/bge-reranker-v2-m3` when loaded. CI uses the overlap teacher so the table never pretends.

---

## Run it

`pyproject.toml` is the single dependency spec. Pick the extra for what you
want to run (`requirements.txt` is generated from it by `make lock`):

| extra | installs | for |
|---|---|---|
| `eval` | numpy, pydantic, pytest | `make bench`, `make test-eval` (the CPU benchmark above) |
| `app` | FastAPI, LlamaIndex, psycopg2/pgvector | the API and the Postgres path |
| `ml` | torch, sentence-transformers, HF embeddings | real embedders and the bge reranker |

**Retrieval API with Docker.** One command; no Ollama, no model download.
The first start indexes `bench/corpus/files` (61 files) into pgvector with
the deterministic hash embedder, and `/health` reports what is loaded.

```bash
git clone https://github.com/joses2017smjh/MetaNavT.git
cd MetaNavT
docker compose up --build            # Postgres + pgvector, then the API on :8000

curl -s localhost:8000/health
curl -s -X POST localhost:8000/api/retrieve/ \
     -H 'content-type: application/json' \
     -d '{"query": "current learning rate for run 47"}'
python3 scripts/smoke_retrieve.py     # the check CI runs: hits, every hit has a path
```

The compose stack is labelled honestly: `pgvector/pgvector:pg16` has no
ParadeDB `pg_search`, so lexical search falls back to Postgres `ts_rank_cd`
(`bm25_backend` in the response), embeddings are the hash adapter
(`embedding_provider`), and no reranker is loaded (`reranker_loaded`). For
real models use the override, which installs the `ml` extra and loads
bge-large-en-v1.5 (~1.3 GB download) and bge-reranker-v2-m3:

```bash
docker compose down -v   # the table width changes from 256 to 1024
docker compose -f docker-compose.yml -f docker-compose.ml.yml up --build
```

**Full application locally** (chat needs Ollama; retrieval does not):

```bash
pip install -e ".[eval,app]"         # add ,ml for real embeddings / reranker
cp .env.example .env                 # RETRIEVE_K=50, RERANK_TOP_N=8, ENABLE_ROUTER=true
docker compose up db                 # or any Postgres with pgvector
ollama serve && ollama pull qwen2.5:14b

./scripts/run.sh generate            # index DATA_DIR (config/loaders.yaml)
./scripts/run.sh dev                 # API + Next.js dev server, http://localhost:8000
```

Set `EMBEDDING_PROVIDER=hash` to run the Postgres path without the `ml` extra,
and `RERANKER_MODEL=BAAI/bge-reranker-v2-m3` for the real cross-encoder row.

To freeze *your* capstone tree: snapshot the directory, hash `MANIFEST.json`, draft questions with an LLM, then hand-correct. Do not ship synthetic-only gold over a real corpus.

---

## What we are not claiming

**RL search policy (Search-R1 / GRPO) is not GPU-trained.** The env, reward, retrieved-token mask, and a CPU dummy GRPO step live in `app/rl/`. veRL + Qwen is still a training run. See [PLAN.md](../PLAN.md) Phase 10.

**The CI table's rerank row is the overlap teacher**, not `BAAI/bge-reranker-v2-m3`; it is within noise of plain RRF (see the [ablation table](#the-number)). The real reranker is measured in the [real-models tables](#real-models-and-a-corpus-where-k50-means-something), which are committed artifacts rather than CI output. Until M2 the cross-encoder was called through the wrong API (`compute_score` instead of sentence-transformers' `predict`). In the bench, the weights were never cached, so `get_cross_encoder()` returned nothing and the row fell back to overlap for lack of weights; the first run with weights present raised. In the production retriever the same call was inside a broad `except`, so reranking silently returned the RRF order. No earlier "bge-rerank" number in this repository's history was a cross-encoder result; `reranker_loaded` in the results file is the check.

**E2E scores in the older result files are token overlap.** The LLM evaluation with its validity gate is the [judge section](#does-the-judge-agree-with-the-gold-labels): deterministic checks first, then two judge models, and Cohen's κ vs the exact-match labels on simple_factual must reach 0.6 before a judge score is printed. Retrieval metrics stay BEIR-style and LLM-free.

**The injection evaluation is not an adaptive attack.** It plants one phrasing per goal and task, written for this benchmark, in two files per task, and runs two local 7-8B models. Adaptive attackers break most published defenses ([Nasr et al. 2025](https://arxiv.org/abs/2510.09023)), so a low attempt rate here says how these models treated these phrasings, not that a configuration resists prompt injection. There is no capability control (running each injected goal as an ordinary user task), so a model that rarely reaches the planted text or calls tools badly can show few attempts out of incapability; the exposure column is reported for that reason. The structured field check is told which config field each question asks about and disagrees with every hijacked answer by construction, so it shows the mechanism rather than a measured detector. No model called a write tool in this run, so the run does not exercise the approval gate; the unchanged corpus follows from no grant being issued, and the gate's own evidence is the deterministic approval benchmark. The benign controls are not matched on retrieval: the planted lines add the run number and field name to two files, which can change what a search returns. Tool calls recovered from JSON text run whenever an answer contains a JSON object with a tool name, so a refusal that quoted a planted call would count as an attempt; none of the recovered calls was a write tool.

---

## What's next

Status: M0 to M5 are merged. Open items, in priority order:

1. **Kappa reference with variance.** The generator matched gold on all simple-factual questions, so the judge gate is not evaluable. This needs a harder slice as fixture v2; v1 stays frozen.
2. **Per-slice changed-query counts** in the harness, so a small-n category delta shows how many questions actually moved.
3. **Judge abstention handling.** Both judges labelled some NOT IN SOURCES answers correct.
4. **ParadeDB BM25 in the production path**, gated by the same parity job.
5. **Injection evaluation, next steps:** an injection-as-user-task capability control, a collection of phrasings per goal scored as any-success (AgentDojo's adaptive "Max" attack), larger models, and a CaMeL-style quarantined reader that extracts the requested field so planted text never reaches the planner.
6. Optional: distill the winning retriever (`distill_winner`), listwise RankGPT with a real LLM, GPU Search-R1 / GRPO with `app/rl/reward.py`.

Full citations and abort conditions: [PLAN.md](../PLAN.md).

---

## Layout

```
app/eval/          bench harness, metrics, jury, agent injection, sweeps, SVG figures
app/retrieval/     RRF, BM25, router, distill, RankGPT, HNSW / halfvec / binary
app/chunking/      structure-aware, late, contextual
app/graph/         file graph, version clusters (measured), conflicts; GraphRAG, HippoRAG PPR (prototypes)
app/agent/         retrieval loop, Deep Research, move-plan schema
app/artifacts/     ACM bundles, Paper2Code, spec/patch/sandbox (prototype, not evaluated)
app/rl/            Search-R1 env, reward, GRPO dummy step (prototype, not evaluated)
app/mcp/           filesystem tools + stdio server
app/multimodal/    MaxSim, ColPali-style pages, CLIP/SigLIP images (prototype, not evaluated)
app/database/      index manager, pgvector, BM25
app/engine/        LlamaIndex agents, loaders, sandbox
.frontend/         Next.js UI
bench/corpus/      frozen files + MANIFEST
bench/gold/        136 questions
bench/results/     one JSON per git sha
doc/demo.html      clickable tour (open locally; GitHub will not run it)
doc/gifs/          looping previews that play in the README
doc/figures/       still charts
```

---

## Team

| | role |
|---|---|
| Deepti R. | Benchmark testing and documentation |
| Carlana S. | Benchmark testing and backend |
| Kaitlyn L. | DevOps / CI and benchmark infrastructure |
| John T. | Front-end / UX and execution sandbox |
| Kantaro N. | Retrieval and indexing |
| Jose S. | Retrieval and data APIs |

ravidatd@oregonstate.edu · soma@oregonstate.edu · lauriek@oregonstate.edu · tranj8@oregonstate.edu · nakanika@oregonstate.edu · sanchej7@oregonstate.edu

[PLAN.md](../PLAN.md) is the modernization checklist and the v3 frontier plan (LLM jury, bge-m3 row, Search-R1). Issues that predate it: HITL UI [#74](https://github.com/klaurie/MetaNaviT/issues/74), ranking [#58](https://github.com/klaurie/MetaNaviT/issues/58), chunking [#22](https://github.com/klaurie/MetaNaviT/issues/22).

## Resume project results

[Verified project sections](../docs/resume-results-20261007/resume-projects.pdf) ·
[Editable role-specific bullets](../docs/resume-results-20261007/resume-projects.txt) ·
[Measured results and research rationale](../docs/resume-results-20261007/revamp-report.pdf).

The saved report describes the local measurement phase; repository history records
the subsequent publication. Generated lookup labels, short HTTP sessions and
synthetic enforcement fixtures retain the limits stated in the report.
