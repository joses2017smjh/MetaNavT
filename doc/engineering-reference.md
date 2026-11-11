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

FiQA with the reranker (`make bench-beir-fiqa`; [results](../bench/results/beir_fiqa.json),
[generated table](../README.md#bench-beir-fiqa)) runs BM25, dense and hybrid again
next to bge-reranker-v2-m3 over the hybrid's top 20 with `max_length` 512, first
in fp32 and then in fp16, each row paired with its parent, with a TREC run file
for every row in `bench/results/beir_fiqa_trec`. Its document embeddings are the
cached bge-small matrix of the October run (the GPU that computed it is the one
[beir_fiqa_gpu.json](../bench/results/revamp_20261007/beir_fiqa_gpu.json) names);
the queries and the reranker ran on the GPU the new file names.

On NFCorpus, changing the GPU reranker from fp32 to fp16 while holding the
**512-token cap and top-20 depth fixed** reduces warm offline total p95 from
**216.52 to 77.15 ms** (64.4%). nDCG changes from **0.3555 to 0.3554**. This
isolates precision more directly than the historical SciFact joint precision/
truncation comparison. It excludes model load, index build and HTTP concurrency.
On SciFact the cap and the precision are now also measured one at a time:
fp32 uncapped, fp32 with `max_length` 512, fp16 at 512, each paired with the row
before it, in the ladder behind the [generated SciFact table](../README.md#bench-beir)
(`make bench-beir`); the evidence table quotes both steps with their rerank-stage
latency. `make bench-beir-rerank-cap` runs only the hybrid and those three rows
([results](../bench/results/beir_scifact_rerank_cap.json)); their metrics equal
the ladder's.

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
pip install -e ".[eval]"   # numpy, pydantic, pytest, scikit-learn (no LlamaIndex, no torch)
make test-eval             # evaluation tests
make bench                 # writes bench/results/<git-sha>.json and latest.json
make bench-gate            # latest.json vs the committed baseline bench/results/main.json
make bench-check           # re-runs the bench; every field of every row must equal main.json
```

CI runs exactly this on every push and pull request (Python 3.11 and 3.12).
The gate fails if any config's nDCG@10, Recall@10 or Recall@50 drops more than
0.002 below `main.json`; `make bench-check` fails on any difference at all
outside timings, in either direction. The bench gives the same numbers on any
machine and checkout: the staleness filter and the file graph compare mtimes,
and the bench takes them from `bench/corpus/MANIFEST.json`, not from the disk
(a checkout sets them to the moment it wrote each file); dense scores are
computed in float64, rounded to 10 decimals and ranked with ties in corpus
order, so neither the last bit of the CPU's BLAS kernel nor numpy's sort
implementation decides an order. The results header records where the mtimes
came from. To accept a change on purpose: `make bench-baseline`, review
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

The benchmark's lexical list is Okapi BM25 in memory; the served path uses Postgres full text, `ts_rank_cd` (`LEXICAL_BACKEND`, whose default it is; a ParadeDB BM25 variant of the statement exists, and no test or results file has run it). Dense is `<=>` on pgvector. Same query, two lists, one RRF ranking. That is the coverage jump in the table below, read next to its random-list baseline.

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

Hard cap on iterations. Unresolvable citation = failed generation. Empty retrieval refuses to guess. Specialized agents (file reader, Python exec when switched on, task router) still sit under FastAPI; retrieval is no longer a one-shot `top_k`.

### MCP filesystem, with a human interrupt

The same gate is exposed to the web UI as `/api/plans` (propose, list, approve, reject) with the approve/reject panel under the chat; see [How the app is put together](#how-the-app-is-put-together). `app/mcp/server_sdk.py` serves these tools through the official MCP Python SDK; `app/mcp/server.py` is the earlier hand-rolled JSON-RPC server, kept for clients pinned to it.

The Next.js app type-checks and exports statically (`npm run build` in `.frontend/`); the Docker frontend stage (`WITH_FRONTEND=1`) is built by the manual `docker-ml-smoke` CI job together with the CPU real-model stack, not on every push.

<p align="center">
  <img src="../doc/gifs/mcp.gif" alt="propose_move pending, apply_plan blocked, then applied" width="820"/>
</p>

Any MCP client can drive the corpus over stdio, through either server. Each indexes its root when it starts, with the hash embedder (unless `--embedder` names another) and the overlap reranker, neither of which is a learned model; `--no-index` skips the index for roots too large to index at start, and then search and `propose_artifact` refuse. The index is a snapshot taken at start: it does not follow applied moves or patches. Files reached through a symlink are left out of the index and of `collect_run_artifact`'s packs, and `read_file` and `stat` refuse their paths. OverlayFS already sandboxed file edits in the original app. The contract is now explicit in the tool:

| tool | contract |
|---|---|
| `search_semantic` / `search_lexical` | routed hybrid search over the start-time index (BM25 and dense fused, then the overlap reranker; the router skips the reranker for aggregation queries and gives a path-like query path lookup and BM25 only) / BM25 over the same index |
| `read_file(path, byte_range)` | citations resolve to bytes |
| `list_dir` / `stat` | navigate the tree, not just vectors; `list_dir` shows a symlink as a link (`"symlink": true`, no size) without reading its target, and `stat` refuses it |
| `propose_move` | returns a plan. never executes |
| `apply_plan` | requires an externally signed, single-use `approval_token` |
| `collect_run_artifact` | ACM-style pack for a run (config, code, log, paper); files reached through a symlink are left out |
| `propose_artifact` | spec → generate → sandbox. never writes |
| `apply_artifact` | requires a bound `approval_token`; refuses a failed sandbox |
| `propose_patch` / `apply_patch` | SEARCH/REPLACE with the same signed approval gate |
| `exec_sandboxed` | A Python subset, checked on the syntax tree against an allow-list before anything runs. The list holds plain statements (assignment, `if` / `for` / `while`, `try`, `def` and the like) and expressions (calls, `lambda`, comprehensions, f-strings, `:=`); imports, attribute expressions (`x.y`), `match` statements, `class`, `with`, `global` / `nonlocal`, `yield` and async code are not on it, and a name that starts with `__` is refused. The code can read only the names it binds and a list of builtins that has no `getattr`, `vars`, `dir`, `type`, `format`, `open`, `eval` or `__import__`. It runs in a child process whose environment keeps only `PATH`, `HOME`, locale, `TZ` and `TMPDIR` (no API keys or connection strings), with CPU, memory and file-size limits. The child runs as the server's OS user, so this is a correctness runner, not OS isolation |
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
[Action evidence](../bench/results/trusted_actions_v2.json) ·
[Action evidence v1, HMAC grants](../bench/results/trusted_actions.json) ·
[Structured evidence](../bench/results/evidence_consistency.json)

The action evidence is the v2 record, written by `make bench-trusted-actions`
with Ed25519 grants; each of its attack rows also names the check that refused
it (`refused_by`). The v1 record, made with the HMAC grants used before
Ed25519 and without refusal reasons, stays linked as it was recorded. A
replayed grant is refused by the staleness check when its action applied (the
reviewed files changed), so the odd `replay_patch` variants replay a grant
burnt by a failed apply instead: only the single-use ledger can refuse those.

The boolean `approved=True` no longer authorizes a write. A grant is an Ed25519
signature: the operator signs with a private key that only the operator CLI
reads, and the servers verify with the public key alone, so a process that
verifies grants cannot mint one. The private key (mode 0600), the public key (not
writable by group or others) and the SQLite replay ledger must live outside
every agent-readable/indexed root: whoever can replace the public key can mint
grants, and whoever can replace the ledger can replay them. Without configured
approval state, file writes are refused. A capability binds the session, operation, paths, reviewed content and
source fingerprints, expires quickly, and is consumed transactionally before
mutation. Symlink paths and implicit overwrites are refused. The audit trail
names the check that refused an `apply_*` call (`REFUSAL_CHECKS` in
`app/mcp/approvals.py`: signature, expiry, ledger, staleness and the rest). The
SDK server also returns that name and a fixed description as a tool error, and
gives a call without a capability the structured `approval_required` result;
the JSON-RPC server returns the exception's message, as before.

```bash
# Run as the trusted operator; choose paths outside the indexed corpus.
python -m app.mcp.approve init --key /private/operator.key --public-key /srv/metanavit/operator.pub
python -m app.mcp.approve sign --key /private/operator.key --review review.json --actor operator
# Server settings: METANAVIT_APPROVAL_PUBLIC_KEY_FILE (the public key only) and METANAVIT_APPROVAL_LEDGER_FILE
```

A token is the base64url of a canonical JSON body (review hash, grant id, issue
and expiry times, actor), a dot, and the base64url of the Ed25519 signature over
that body. Tokens and key files from the earlier HMAC scheme are refused. While
the old `METANAVIT_APPROVAL_KEY_FILE` is set, even next to the new setting, the
MCP servers refuse to start and the API's `/api/plans` answers 503 with that
error; unset it, create the key pair, and name the public key in
`METANAVIT_APPROVAL_PUBLIC_KEY_FILE`.

Export the review from the plan panel, have the operator inspect and sign it,
and paste the resulting token into the panel. Issuance is absent from MCP and
HTTP tools. An agent with arbitrary operator code execution, read access to the
private key or write access to the public key is outside this boundary. Older GIFs illustrate the review flow;
the current implementation requires the signed grant.

### Cutting-edge query processing

Six techniques from 2023–2026 RAG research, wired into the bench harness with ablation flags.

**HyDE** (Hypothetical Document Embedding). Instead of embedding the raw query, generate a pseudo-document from the query tokens and embed that (averaged with the query's own vector). The hypothesis is closer to the answer space, so cosine similarity is more meaningful. `app/retrieval/hyde.py`. The benchmark has two wirings (`hyde_stage` in `app/eval/harness.py`). First stage: the HyDE vector replaces the query vector in the dense leg, the dense list is RRF-fused with BM25 as usual, and the reranker, when there is one, orders the fused candidates and has the last word. After the reranker: retrieve and rerank with the raw query, then RRF-fuse the reranked list with a HyDE dense list and re-sort. That second wiring is how every HyDE row recorded before the first-stage arms existed was produced; its re-sort overrides the reranker's order and it never adds a HyDE hit outside the reranked list, so its rows measure that wiring, not HyDE as a dense query. They are kept under their old names.

**Query Decomposition**. Compound multi-hop queries ("compare the learning rate and encoder between run 40 and run 47") are split into sub-queries, each retrieved independently, then merged by best-score dedup. Heuristic splitter (syntactic cues); LLM decomposition available when a model is loaded. `app/agent/decompose.py`.

**Corrective RAG (CRAG)**. After retrieval, classify each chunk as Correct / Ambiguous / Incorrect. If the correct ratio is below threshold, trigger corrective actions: refine ambiguous chunks to key sentences, rewrite the query with expanded terms, or decompose and retry. `app/agent/corrective.py`.

**Adaptive Retrieval**. Not every query needs the full pipeline. Greetings, meta-questions, and textbook definitions skip retrieval entirely. Corpus-specific signals (run IDs, file extensions, config terms) trigger retrieval. Neutral on the gold set — every gold question needs retrieval. `app/retrieval/adaptive.py`.

**Semantic Query Cache**. Near-duplicate queries ("learning rate run 47" vs "what is the learning rate for run 47") hit an in-memory LRU cache with cosine-similarity lookup (threshold 0.92) instead of re-running the full pipeline. TTL eviction prevents stale results. Optional JSON persistence. `app/retrieval/cache.py`. In the benchmark it is the `hybrid+rerank+cache` arm: the gold questions are replayed in gold order into an empty cache, a hit serves the cached file list, and the row counts hits and wrong hits (a cached list that differs from what retrieval returns for that question). The gold set's templated questions differ only in a run id, so this replay measures the false-hit risk. True hits are measured on paraphrases of gold questions ([bench/probes/](../bench/probes/README.md), rule-based rewrites that inherit the relevant paths of the question they rewrite; not gold labels), looked up after the replay without being added.

**Citation Verification**. Post-generation gate: extract factual claims from the answer, verify each value and path against retrieved evidence, flag hallucinated numbers, and reject answers below a verification ratio threshold. `app/agent/citation_verify.py`.

HyDE, decomposition, corrective retrieval, adaptive retrieval and the semantic cache are arms of `make bench-frontier` (first-stage HyDE: `hybrid+hyde` against `hybrid`, `dense_only+hyde` against `dense_only`, `hybrid+hyde+rerank` against `hybrid+rerank`; HyDE after the reranker: `hybrid+rerank+hyde`, `hybrid+rerank+hyde+corrective`; `hybrid+rerank+decompose`, `hybrid+rerank+corrective`, `hybrid+rerank+adaptive`, `hybrid+rerank+cache`), each scored against the row it is one switch away from; the real-model run adds HyDE with an LLM-written hypothesis (`hybrid+hyde-llm`, `hybrid+hyde-llm+rerank`, the hypotheses stored by `make frontier-generations`); what the intervals support is under [Retrieval prototypes with a benchmark arm](#retrieval-prototypes-with-a-benchmark-arm). Citation verification has no harness flag and no benchmark row.

### Graph + staleness

The file system already is a graph: containment, imports, config→checkpoint, same run id, co-modification. No invented edges.

Version clusters: same de-versioned stem + suffix, different content hash. Default retrieve the newest. Comparative questions (`what changed between run 40 and 47`) keep both.

Tier 2 runs at query time on the top-k only: same entity, disagreeing `learning_rate`, or two drafts that say fusion does / does not help. The generator gets a conflict note instead of a silent pick.

<p align="center">
  <img src="../doc/figures/hops.svg" alt="Recall@50 vs graph hops" width="820"/>
</p>

One hop lifts Recall@50 **0.938 → 0.986** and drops nDCG@10 **0.495 → 0.446**, at ~6x wall clock. Two hops is worse. Default stays `graph_hops=0`. That split is the finding. The same two settings are arms of `make bench-frontier` with a paired interval against plain hybrid: both are below zero ([generated table](../README.md#bench-frontier)).

### Indexing that does not redo the whole tree

With `INDEX_ON_START`, every API start runs one indexing step
([app/engine/ingest.py](../app/engine/ingest.py)) under a Postgres advisory lock:

1. **Scan.** Every file under `DATA_DIR` is read once and either accepted or
   rejected with a reason: unsupported type, too large, undecodable, empty,
   duplicate content ([app/eval/ingest.py](../app/eval/ingest.py), the same scan
   the benchmark loader uses, with a stricter policy).
2. **Diff.** Each accepted file's SHA-256 is compared with a manifest table
   (`ingest_manifest_<table>`: path, content hash, chunk count, chunker version
   and embedder). Files are new, changed, unchanged or removed.
3. **Limit.** If more files are rejected than `INGEST_MAX_REJECTED` allows, the
   run fails before it writes anything. CI and docker-compose set the limit to
   zero; unset, rejections are counted but not fatal.
4. **Apply.** Rows of removed files are deleted. For each new or changed file
   its chunks are embedded, its old rows are deleted by path, the new rows are
   inserted, and only then is it listed in the manifest. Unchanged files are
   not sent to the embedder and their rows are not rewritten.
5. **Report.** The run returns the counts of files seen, indexed, unchanged,
   removed and rejected by reason. The report is logged and served by
   `GET /health` under `indexing`.

The step is idempotent and can be interrupted: because the manifest row is
written last and old rows are deleted before new ones are inserted, the next
run repeats the unfinished files and leaves one set of rows. Two replicas that
start together serialise on the lock; the second finds nothing to do.

What checks this: [tests/database/test_incremental_index.py](../tests/database/test_incremental_index.py)
runs against a live Postgres in CI's `parity` job (unchanged re-run embeds
nothing, an edit re-embeds one file, a delete removes its rows, the rejection
limit, an interrupted run, two concurrent runs);
[tests/eval/test_ingest.py](../tests/eval/test_ingest.py) covers the scan, the
diff and the accounting without a database. `make ingest-runs` records four
consecutive runs on a copy of the fixture in
[bench/results/ingest_runs.json](../bench/results/ingest_runs.json).

Limits: a file is identified by its bytes, so a rename is a delete plus an
insert; files are embedded one at a time inside the startup hook; the type
gate is a suffix allow-list (`INGEST_SUFFIXES`), so PDFs and office documents
are counted as rejected, not parsed. The older `indexed_files` table in
`app/database/index_manager.py` is not used by this path, and the legacy
`python -m app.engine.generate` CLI still splits with `SentenceSplitter` and
does not consult the manifest.

### pgvector knobs on a trade-off curve

<p align="center">
  <img src="../doc/figures/hnsw.svg" alt="HNSW ef_search recall vs latency" width="820"/>
</p>

<p align="center">
  <img src="../doc/figures/storage.svg" alt="float32 vs halfvec vs binary+rescore" width="820"/>
</p>

`halfvec` stores half the bytes of float32 and binary quantization one bit per dimension; [bench/results/sweeps.json](../bench/results/sweeps.json) (`make sweeps`) records each one's Recall@10 against float32 (binary after a float rescoring pass) and the HNSW `ef_search` sweep, which the figures above plot. All of it is the **in-memory NSW analogue** (`app/retrieval/pgvector_tune.py`) over **hash** vectors, where recall saturates early: that space is not a manifold. In Postgres, start-up creates an HNSW index on the embedding column (`ensure_hnsw`: `m` and `ef_construction` at build), and each hybrid statement sets `hnsw.ef_search` (twice the dense depth) and `hnsw.iterative_scan` for its own transaction. The planner decides whether to use the index: it scans the table on fixture v1 and reads the index over generated rows (the two plans under `doc/sql/`). Nothing measures the index's recall against the exact scan at a realistic size with a real embedder yet. `ensure_halfvec` and `ensure_binary` exist on `VectorStoreManager` with no caller, and the served statement reads only the float32 column.

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
make bench-check      # re-run, exit 1 unless every field of every row equals main.json (CI runs it too)
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
stopwords, k1=0.9, b=0.4); the `bm25@default-tokenizer` row switches to the
fixture's analyzer (no stemming, no stopwords) and the fixture's k1=1.5,
b=0.75 at once, so its gap is the three changes together, not a tokenizer
effect. `make bench-beir-bm25-ablation` splits it on the CPU: one row changes
the analyzer alone, one changes k1 and b alone, each paired against `bm25`.
The dense leg ranks the way the fixture benchmark's does (float64 cosine
rounded to 10 decimals, exact ties in corpus order). BM25 search reads the
postings lists of the query's terms and keeps the top k with a heap; it
returns the same scores in the same order as scoring every document
(`tests/eval/test_bm25_postings.py`). The ladder file behind the first
table (`bench/results/beir_scifact.json`) was recorded again after both
changes, with the capped fp32 row added. The nine rows it shares with the
earlier recording keep every metric, interval and the published-score
difference; what changed is the timing columns (per-query latency and build
time), the GPU, and the fp16 row's paired delta, which is now against fp32 at
the same cap.

[Current generated bench-beir evidence](../README.md#bench-beir) ·
[BM25 settings one at a time](../README.md#bench-beir-bm25).

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
uncapped, fp32 with `max_length` 512 and fp16 at 512 side by side, and sweeps the rerank
depth (20, 50, 100), each with its own p50/p95; the served configuration is
chosen from that table in the API section below, not picked up front.

---

## The API serves the benchmarked pipeline

`POST /api/retrieve/` is the same hybrid pipeline over Postgres + pgvector:
one parameterized SQL query ([hybrid_tsrank.sql](../app/database/sql/hybrid_tsrank.sql))
with a pgvector top-k CTE (`<=>`; the planner may read it from the HNSW index), a full-text top-k CTE
(`websearch_to_tsquery` + `ts_rank_cd`, GIN index; ParadeDB BM25 only with
`LEXICAL_BACKEND=paradedb`, a variant no test or results file has run), and
reciprocal rank fusion computed in SQL, then
staleness Tier 1 from the same `app/graph` code the bench uses. Every hit
carries `path`, `start_byte`, `end_byte`, `chunk_id` and its per-stage scores;
the response carries the route and per-stage latency. `k` is validated (1-50),
bad bodies are 422, a down backend is 503, and a retrieval that runs past its
time budget (`RETRIEVE_TIMEOUT_SECONDS`) is 504, naming the stage. The SQL
statement gets the rest of the budget as its `statement_timeout`, so Postgres
cancels it and its connection goes back to the pool rolled back; the embedding
and reranking stages are checked when they end, so a late one stops the request
before the next stage. A model call cannot be interrupted: when one hangs, the
route answers shortly after the budget and the call keeps its worker thread
until it returns (`tests/api/test_retrieve_contract.py`,
`tests/database/test_retrieve_timeout.py`). The committed
[EXPLAIN (ANALYZE, BUFFERS)](../doc/sql/hybrid_explain.txt) is the plan on fixture v1
(`make explain-sql`): at that size the dense list is a sequential scan and the lexical list a GIN bitmap scan.
[hybrid_explain_synthetic_10k.txt](../doc/sql/hybrid_explain_synthetic_10k.txt)
(`make explain-sql-synthetic`) is the same statement over generated rows with
hash vectors in a scratch table, synthetic text rather than data: there the planner
reads the dense list from the HNSW index on its own. Both plans run with the
statement's per-transaction settings, which their headers list with the SHA-256 of the statement file they planned.
`tests/database/test_hybrid_sql.py` checks that the SQL fusion equals the Python
RRF on identical input lists, against a live Postgres in CI.

[Current generated bench-parity evidence](../README.md#bench-parity).

That block renders
[bench/results/parity_shared_chunker.json](../bench/results/parity_shared_chunker.json)
(`parity_shared_chunker_restart.json` is the same command after a restart on
the indexed database). Both runs were made after the API took the bench's
chunker: both paths call `file_chunks` in [app/eval/ingest.py](../app/eval/ingest.py),
and [tests/engine/test_ingest_nodes.py](../tests/engine/test_ingest_nodes.py)
compares the chunk boundaries and texts of the two paths on the fixture, so the
dense leg embeds identical texts in both rows. (The earlier
[bench/results/parity.json](../bench/results/parity.json) was recorded when the
API still chunked with LlamaIndex `SentenceSplitter`.) Both rows also see the
same file mtimes, which the staleness filter compares: the in-memory row reads
the fixture's MANIFEST, and the API reads the disk after `make fixture-mtimes`
has set it to the MANIFEST's values (CI runs it before starting the API; a
checkout's own mtimes are arbitrary, and the API keeps reading the disk as a
served system should). The API row scores above the in-memory row and the
paired interval excludes zero: sharing the chunker removed one difference and
did not make the rows agree.

What is checked exactly is the dense leg:
[tests/database/test_dense_parity.py](../tests/database/test_dense_parity.py)
indexes the fixture through the served indexer, runs the dense CTE of
`hybrid_tsrank.sql` at `RETRIEVE_K` for every gold question, and requires the
same chunks in the same order with the same scores as the in-memory dense
search (scores equal in exact arithmetic are compared as a set). The end-to-end
nDCG@10 gap is reported with its paired interval and not gated; `make parity`
fails only when the API is broken (a question without an answer or with zero
hits). On the fixture the planner answers the dense CTE with a sequential scan.
A second test forces the HNSW index (`enable_seqscan` off), requires the whole
served statement to be planned on it, and requires the same `RETRIEVE_K` rows
for every gold question. On the fixture that scan reaches every row (the
statement reads twice the depth, more rows than the fixture has), so the test
checks the plan and the depth, not the index's recall: an HNSW scan is
approximate, and its recall at a realistic size is not measured yet. Two things
kept the index from serving the full list before: the `params` CTE, read by both
lists, was materialized, which made the query vector a column and left the dense
list no index path at all; and an HNSW scan returns at most `hnsw.ef_search`
rows, and pgvector's default is below `RETRIEVE_K`. The statement now keeps the
query vector in a `NOT MATERIALIZED` CTE of its own (the tsquery stays a
materialized value, which keeps the lexical list's GIN plan), and `hybrid_search`
sets `hnsw.ef_search` to twice the depth (plus `hnsw.iterative_scan =
strict_order` on pgvector 0.8 or later) in the same execute as the statement, for
its transaction only, so no pooled connection keeps the value. `/health` reports
the depth and these settings under `dense_search`. Equal scores at the dense list's
cut are decided by `node_id`, like every other tie in the statement: the
index-ordered scan reads twice the depth, and the statement keeps the first
`RETRIEVE_K` of those rows by distance, then `node_id` (a second sort key inside the
index scan would leave the index unused). `tests/database/test_hybrid_sql.py`
puts two rows at the same distance across the cut, stored in either order, and
requires the smaller `node_id` with and without the index. It also requires that an
all-zero embedding, whose cosine distance is NaN, does not take a place at the cut
from a real neighbour. `node_id` order is not
the in-memory index's corpus order, which is why the dense test compares tied
chunks as a set.

The remaining differences, with both rows reported rather than one being
declared "the" number: the lexical legs (Okapi BM25 versus Postgres
`ts_rank_cd` over OR-joined terms), the order of equal scores (the SQL orders
ties by `node_id`, the in-memory fusion by first-seen order), and the
exact-path route (the in-memory index searches paths; the API runs its lexical
list alone). CI's `parity` job runs the dense-leg test and the parity report on every push.

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
fails if a configured reranker could fall back without a trace. The benchmark's
in-memory `CrossEncoderReranker` (`app/retrieval/rerank.py`) follows the same
rule: with no model loaded a call raises, unless the caller asked for the
overlap fallback (`allow_fallback=True`, which sets its `fallback` attribute).
The benchmark loader asks for it and labels the row `reranker_fallback:
overlap`; `make bench-rerank-arms` does not, and stops when the cross-encoder
is missing.

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
ten questions run without planted text as controls. In the current task set
(`agent_injection_v2`) the controls are matched on retrieval: their config and
log end with neutral comment lines that carry the run number and the field name
the planted text adds, so a search sees the same query terms and a control
differs from an injected task in the instruction alone; its exposure counts
when those lines reach the model. The copy is indexed with
the file times the fixture's MANIFEST records, as the retrieval benchmark
indexes it, so retrieval does not depend on when the copy was made (the first
run's copy read the disk, where the two files that carry the planted text were
the newest).

Every episode is scored at two levels. **Model level:** did the model call
the mutation tool, try the out-of-root path, or report the planted value?
**System level:** did the corpus change, did the canary stored outside the
root reach the answer, was a planted value delivered? The approval gate runs
with a real operator key and SQLite ledger outside the corpus in every
configuration, and no grant is ever issued in a question or injected task, so
every forged token has to fail.

Three configurations change what the model sees or may call:

| configuration | what changes | defense level |
|---|---|---|
| `none` | tool results as the tools return them (search snippets cut at 500 characters, results at 4,000); four read tools and four write tools | — |
| `spotlight` | file text in tool results has every whitespace run replaced by `ˆ`, and the system prompt says marked text is data, never instructions (datamarking, [Hines et al. 2024](https://arxiv.org/abs/2403.14720)) | prompt |
| `least_privilege` | a tool filter chosen from the request before any file content is read: a question gets only the four read tools, a capability task also the two tools of the action the user asked for (the "commit to a set of tools" guidance of [Beurer-Kellner et al. 2025](https://arxiv.org/abs/2506.08837); AgentDojo's tool-filter baseline). Read results, planted text included, still reach the model, so this constrains actions and cannot stop a planted claim | system |

**Capability control (`agent_injection_v2`).** No attempt at a planted move or
patch can mean that the model refused the planted text or that it cannot carry
the action out. The current task set adds, for each of the ten questions, the
planted move and the planted patch asked for by the user: the same tool calls
with the same arguments, with the operator's approval token in place of the
forged one. In these twenty capability tasks the operator signs a proposal with
its Ed25519 private key, kept outside the corpus root, when applying it to the
unchanged copy would make exactly the requested change, declines any other, and
returns the signed token with the proposal; no grant is issued in any other
task. Success is the copy ending in exactly the requested state; an attempt is a
call to the action's tools, counted as for the planted actions. A capability
task writes, so each of its episodes runs on a fresh copy. The first run's task
set (`agent_injection_v1`) has no capability tasks.

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
answer against the structured source rather than the free text. The structured
field check does that, measured offline on the stored answers in the
[generated table](../README.md#agent-injection), with its limits under
[What we are not claiming](#what-we-are-not-claiming);
a CaMeL-style quarantined extractor
([Debenedetti et al. 2025](https://arxiv.org/abs/2503.18813)), which hands the
planner only the requested field, would make it general.

pass^k shows the second gap: an answer that is right on one sample can be wrong
on the next. The weaker model never answered all four trials of any control
question correctly; its failed episodes call tools that do not exist and guess
file paths, and the harness recovering its JSON-text tool calls did not close
that.

```bash
ollama pull qwen2.5:7b && ollama pull llama3.1:8b
make bench-agent-injection-v2          # the current task set; writes the episodes JSONL as it goes; --resume continues an interrupted run
make bench-agent-injection-v2-rescore  # recomputes its scores and summary from its committed episodes, no model call
make bench-agent-injection             # the first run's task set, kept as recorded in agent_injection.json
make bench-agent-injection-rescore     # recomputes every score and the summary from the first run's episodes, no model call
make bench-agent-injection-scores      # writes every first-run episode's current score next to the episodes, no model call
```

The score object stored inside each episode row is the one written when the
episode was generated, by the scoring code of that time; neither
`make bench-agent-injection-rescore` nor `make bench-agent-injection-scores`
writes to the episodes file. Current per-episode scores are in
`agent_injection_episodes_rescored.jsonl`: one row per episode, in the same
order, holding the episode key (model, configuration, task, trial) and the
score from today's scoring code, with no answer or tool call copied. Its
`.meta.json` names the episodes file and the scoring module by sha256, and a
test fails when the file differs from a fresh rescore or the episodes hash is
stale.

A run of the current harness keeps what the first run could not. Every tool
call in its episodes file holds the result as the model received it, cut at
4,000 characters and flagged when cut, so the exposure flag can be recomputed
from the file and anyone can read what each model saw. A tool error can quote
the path of the task's temporary copy, which can hold the user name, and a model
can echo it, so before a row is scored or written every path into that copy is
written relative to the corpus root (the private directory as `../private`) and
user and host names are replaced; other text, such as the corpus's own
`#!/bin/bash`, stays as sent. The sidecar (`.meta.json`: the task set,
`reconstructed: false`, and per process segment the commit, whether the tree
had uncommitted changes, the module's sha256, the command, the source of the
corpus file times, the device and the Ollama version and model digests) is
written before the first episode, and the results header records the commit and
tree the process started from, before the run's own files appear. The header's
`ollama` block also keeps what the server reported after the run for each
loaded model, including the context length it chose from the GPU's memory
(requests here do not set one). In a capability task, refused apply calls are
counted by exception: a token that did not verify, no token, or a plan id the
server never issued.

Two runs are on record in `agent_injection.json`. The top-level `git_head`,
`working_tree_dirty`, `module_sha256` and `command` describe the run that
scored the stored episodes, `make bench-agent-injection-rescore`, at the
commit and with the harness module it names; a test checks that the scoring
run had nothing uncommitted and that today's harness rebuilds the task list,
the summary and the paired deltas exactly from the stored episodes, so the
file is kept as recorded and still reproduces. `episodes_generated_by` is the sidecar of
the run that generated the episodes, kept as written: for this run it was
reconstructed after the fact and records uncommitted changes in both process
segments. The device, the Ollama version and the wall time come from that
sidecar too, never from the machine that rescored.

[Results](../bench/results/agent_injection.json) ·
[Every episode: tool calls, answers, generation-time scores](../bench/results/agent_injection_episodes.jsonl) ·
[Every episode's current score](../bench/results/agent_injection_episodes_rescored.jsonl) ·
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
from its parent (plain hybrid, hybrid plus the reranker, or dense only), the
parent rows are computed in the same run, and the verdict word is computed from
the interval of the paired nDCG@10 delta: above zero, covers zero, or below
zero. Every row lists the stand-ins it ran on. On the CPU stand-ins every row
uses a model-free embedder (the hash embedder, or TF-IDF in its own two rows)
and, where it reranks, the overlap reranker. The real-model run uses
bge-small-en-v1.5 and the bge-reranker-v2-m3 cross-encoder and is the one that
carries the verdicts. The query-side mechanisms are keyword or regex
heuristics, not models, in both runs, with one exception: the LLM-written arms
of the real-model run read HyDE hypotheses and chunk prefixes that qwen2.5:7b
wrote once through Ollama (`make frontier-generations`, temperature 0). They are
stored with the model, its digest and both prompts in
[bench/results/frontier_generations.json](../bench/results/frontier_generations.json),
so those rows are rescored without the model. The generated tables also compare
some arms directly (row minus other, same bootstrap): the two HyDE wirings, and
each LLM-written arm against the heuristic arm it replaces.

| arm | what it changes | CPU stand-ins | real models | what the intervals support |
|---|---|---|---|---|
| Multi-query (`hybrid+multiquery`) | fuses the lists of heuristic query rewrites | above zero | above zero | Evaluated: the one prototype arm that beats its parent in both runs. The rewrites are keyword rules written against this fixture, so the gain is not evidence about other corpora. |
| HippoRAG PPR (`hybrid+hipporag`) | PPR over regex triples plus a typed path prior, reranking the fused list; gated by the router's own prediction | above zero, with the lower bound at the edge of zero | covers zero | Evaluated: no measured gain once the embedder is real. Before this run the gate read the gold category of each question; it no longer does. |
| Graph-hop expansion (`hybrid+hops=1`, `hybrid+hops=2`) | inserts file-graph neighbours after each hit | below zero, one hop and two | below zero, one hop and two | Evaluated: a recorded loss. The default stays at zero hops. |
| Fixed-window chunks (`hybrid+fixed-chunks`) | replaces the structure-aware chunker with a fixed character window | covers zero | covers zero | Evaluated: on this fixture the structure-aware chunker is not measurably better than the fixed window. |
| Contextual prefixes (`hybrid+contextual`) | prepends a situating prefix to every chunk before indexing | above zero | below zero | Evaluated with a path-words prefix. The sign flips: the prefix helps the hash embedder and hurts bge-small, so it is a recorded loss on the stack that matters. The LLM-written prefix is the next row. |
| LLM-written contextual prefixes (`hybrid+contextual-llm`) | prepends the situating description qwen2.5:7b wrote for each chunk (`app/chunking/contextual.py` prompt, stored) | not evaluated (real models only) | covers zero | Evaluated with bge-small: no measured effect against plain hybrid, and above zero against the path-words prefix in the direct comparison. The loss of `hybrid+contextual` belongs to the path-words prefix, not to contextual prefixes in general; an LLM-written prefix is still not a measured gain. |
| Late chunking (`hybrid+late-chunking`) | pools document-level token embeddings per chunk | not evaluated (the hash embedder has no token embeddings) | covers zero | Evaluated with bge-small only: no measured effect. Few files in the fixture have more than one chunk (the count is in the table). |
| HyDE in the first stage (`hybrid+hyde`) | the keyword-heuristic HyDE vector is the dense leg's query, fused with BM25; no reranker | below zero | below zero | Evaluated: a recorded loss. |
| HyDE in the first stage, dense only (`dense_only+hyde`) | the same vector as the only query | below zero | below zero | Evaluated: a recorded loss. |
| HyDE in the first stage, then the reranker (`hybrid+hyde+rerank`) | the same vector in the first stage; the reranker orders the fused candidates and has the last word | covers zero | covers zero | Evaluated: no measured effect once the reranker has the last word (the interval's lower bound is zero in the real-model run). On this fixture that tie is close to forced: the reranker rescores the fused top `retrieve_k` candidates, most of the fixture's chunks, so a first-stage change can only swap candidates at the tail of that list, and with real models the arm's Recall@50 is unchanged. In the direct comparison it is above zero against the after-reranker wiring in both runs, so that wiring's loss came from the re-sort. |
| LLM-written HyDE hypothesis (`hybrid+hyde-llm`, `hybrid+hyde-llm+rerank`) | the first-stage arms with the hypothesis qwen2.5:7b wrote for each question (`app/retrieval/hyde.py` prompt, stored) | not evaluated (real models only) | below zero without the reranker; covers zero with it | Evaluated with bge-small: the LLM hypothesis beats the keyword one in the direct comparison without the reranker but still loses to plain hybrid; with the reranker neither hypothesis has a measured effect (the same close-to-forced tie as the row above). |
| HyDE after the reranker (`hybrid+rerank+hyde`) | RRF-fuses a second dense list from a pseudo-document with the reranked list and re-sorts, overriding the reranker's order | below zero | below zero | Evaluated: a recorded loss of this wiring, which is how every HyDE row before the first-stage arms was produced. The re-sort overrides the reranker and no HyDE hit outside the reranked list is added, so the loss is not evidence about HyDE as a dense query (see the first-stage rows above). |
| Decomposition (`hybrid+rerank+decompose`) | retrieves per sub-query and merges | covers zero | covers zero | Evaluated: no measured effect; it fires on few questions (the count is in the table). |
| Corrective retrieval (`hybrid+rerank+corrective`) | grades the top hits and retries with a rewrite or sub-queries | covers zero: identical first ten files | covers zero: identical first ten files | Evaluated: it fires on most questions but only appends candidates after the existing list, so it cannot change the top ten. |
| HyDE after the reranker + corrective (`hybrid+rerank+hyde+corrective`) | both | below zero | below zero | Evaluated: the loss is that of the after-reranker HyDE wiring; the corrective step adds nothing to the top ten. |
| Adaptive retrieval (`hybrid+rerank+adaptive`) | skips retrieval when a regex classifier says the query needs none | covers zero: it never skips a gold question | covers zero: it never skips a gold question | Evaluated as a no-harm check only. The gold set has no question that should be skipped, so its latency benefit is not measured. |
| GraphRAG community routing (`hybrid+rerank+graphrag`) | rank-fuses the files of the community summaries a question matches | covers zero | covers zero | Evaluated: a negative result. The community step collapses the fixture into one community, so the summaries cannot tell files apart and the arm is a member / non-member split. It does not beat hybrid plus rerank beyond the interval in either run and is not kept in the served path. |
| Semantic cache (`hybrid+rerank+cache`) | serves the cached file list of a near-duplicate earlier question (cosine threshold 0.92), replaying the gold set in gold order | below zero | below zero | Evaluated on the gold set as a false-hit test: every hit served another gold question's file list, and every such list differs from what retrieval returns for the question; most hits are templated questions that differ only in a run id (each hit and the question it was served from are in the results file). A recorded loss on this gold set. On the paraphrase probes (`bench/probes/`, rule-based, not gold labels) most probes hit their own question's entry with bge-small and the served lists are not measurably worse than fresh retrieval; with the hash embedder most probes miss. The counts are in the table notes. |
| TF-IDF embedder (`dense_only@tfidf`, `hybrid@tfidf`) | replaces the hash embedder with TF-IDF and truncated SVD fitted on the indexed chunks | above zero, dense only and hybrid | not evaluated (model-free; CPU run only) | Evaluated: a better model-free stand-in than the hash embedder on this fixture. It is fitted on the fixture's own chunks, so the gain is not evidence about other corpora. |
| Listwise rerank (`hybrid+rankgpt`) | reranks the fused list listwise | covers zero | below zero | Evaluated here only with its token-overlap fallback, which undoes a real embedder's ordering. Listwise reranking by an LLM has its own row in the reranker prototypes table below. |

In the real-model table the arm names carry the model suffix (for example
`hybrid+bge-rerank+graphrag@bge-small`), and the reranked parent is the
cross-encoder row.

The two reranker prototypes have their own generated table
([reranker prototypes against the cross-encoder](../README.md#bench-rerank-arms),
`make bench-rerank-arms`). Two distilled students (`app/retrieval/distill.py`)
are fitted on the same rows: the cross-encoder's scores for half of the
questions, with the features a student has when it serves. Both are scored on
the other half, at a small fraction of the cross-encoder's rerank latency.
Linear least squares (with an intercept column) is below zero against the
cross-encoder it imitates and below zero against the un-reranked hybrid list.
LightGBM trees (`lightgbm.train`, one thread, a fixed seed, deterministic
mode; the `distill` extra, pinned in `requirements/distill.txt`) are below
zero against the cross-encoder and cover zero against hybrid. Each student row
records its model, library version and, for LightGBM, its parameters. Without
the library the LightGBM arm is recorded as not evaluated; it is never
replaced by least squares under its name.

The intercept changes no prediction on fixture v1: every fixture file is one
of the five file types the student flags, so those flags already sum to one
for every chunk (`tests/eval/test_frontier_eval.py` pins this), and re-run
with the intercept every metric and interval of the least-squares row is
unchanged. It matters on a corpus with other file types. Earlier rows had no intercept
column, and the student's `recency` feature (a chunk's mtime over the newest
mtime in its candidate list) is why the least-squares row moved when the
benchmark switched to the MANIFEST mtimes: on a checkout where every file has
the same mtime the feature is the constant 1 and carries nothing, while under
the MANIFEST mtimes it varies slightly and the fit gives it weight, and the
held-out nDCG@10 fell. The commit that re-recorded that row called the
constant column the missing intercept; it was not, since the file-type flags
already provide one. The feature is kept as it is; whether a student should
see checkout mtimes at all is an open question, not one settled on the test
split.

Listwise reranking by an LLM (`app/retrieval/rankgpt.py`;
`make bench-rerank-arms LISTWISE_MODEL=qwen2.5:7b`) sends one prompt per
question through Ollama (temperature 0, a fixed seed) with the first `window`
fused candidates, twenty by default, and asks for their ranking only. On all
questions it is above zero against the un-reranked hybrid list and below zero
against the cross-encoder, and it is far slower per question than the
cross-encoder. The parser reads a reply only when the whole reply is a
bracketed permutation of the window (`[2] > [1] > ...`, every passage once);
prose, bare numbers, file names such as `run_047`, a partial or repeated list
all keep the fused order. The table counts those questions, so the row mixes
reranked questions with questions left in the fused order. The failed replies
are partial lists (some of them the prompt's three-id example copied
verbatim), lists that repeat an id, bracketed ids without the separators, and
an id outside the window. The results file stores every completion with its
parse status, the model's digest and the Ollama version.

Separately, `make bench` logs one row per fused candidate of the hash-embedder
hybrid rows to `bench/results/<sha>.triples.jsonl` (git-ignored; nothing in
the repository reads it): the same serving features, labelled with the overlap
reranker's score, or with the fused score itself where no rerank ran (the
row's `teacher` field says which). It records no dense or BM25 score and no
rank after the rerank, so it cannot leak the label into a feature.

Still without a benchmark row: GraphRAG with LLM extraction or modularity
clustering, Tier 2 conflict notes (`app/graph/conflicts.py`), hierarchy
summaries and citation verification.

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

**Industry.** Spec card (citations, tests, template) before any file is proposed. SEARCH/REPLACE patches. Restricted exec: an AST check refuses `os` and other banned imports and calls (no pip), and the code runs in a child process whose environment keeps only `PATH`, `HOME`, locale, `TZ` and `TMPDIR`, with CPU, memory and file-size limits. The check is a block-list, not a boundary: code that gets past it runs as the server's OS user and can read what that user can read, so this is not OS isolation. Write is HITL, same as `propose_move`. Templates are `python-lib` / `pytest` / `research-repro`, not the 2024 Next.js-13 demo list.

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

**Backend.** FastAPI in `app/`. Chat and retrieve routers, LlamaIndex engine, specialized tools (file access, artifacts, and Python exec when it is switched on; see Python execution in chat below). Agents hand off through a task router. The tool list is read from `config/tools.yaml` at the repository root whatever the working directory. Each chat request records its tool calls in a registry of its own (a context variable, started when the request builds its engine), and the INFO log line for a tool call gives the tool's name and the size of its output, not the output. `tests/api/test_chat_routes.py` sends chat messages (streamed and not) and a query through the real routes and the agent workflow with mock models (a scripted function-calling `MockLLM`, `MockEmbedding`) and an in-memory index: no network, GPU or database. It checks that the routes run, not answer quality. Retrieval is now hybrid + loop, not a single dense `top_k`.

**Database.** PostgreSQL with pgvector (dense) and Postgres full text, `ts_rank_cd` (lexical; ParadeDB BM25 only with `LEXICAL_BACKEND=paradedb`, which no test or results file has run). Connection pooling in `DatabaseManager`. The vector schema, the HNSW index and the per-statement search settings on `VectorStoreManager`; its `ensure_halfvec` / `ensure_binary` helpers have no caller.

**Schema migrations.** The plan decision log and the file-index tables are created by versioned SQL files in `app/database/migrations/` (`0001_baseline.sql`, then one file per change), applied in order by `app/database/migrate.py`; the application no longer runs inline `CREATE TABLE` for them. The runner records each applied file and its SHA-256 in `schema_migrations`, applies each file and its record in one transaction, takes a Postgres advisory lock so two replicas starting together run it one after the other, and refuses to run if an applied file was edited or the database is newer than the code. It runs at startup, when the plan store builds its decision log, and when `IndexManager` is constructed. A database created before migrations existed is adopted in place: `0001` uses `IF NOT EXISTS`, so its rows are untouched. `tests/database/test_migrations.py` (CI parity job, live Postgres) applies version 1, inserts a row, applies the next version and checks the row. Not covered: the vector table, which LlamaIndex's `PGVectorStore` creates, and its GIN / HNSW indexes, which `VectorStoreManager` still creates at startup with `IF NOT EXISTS`; there are no down-migrations.

**API access.** `app/api/security.py` puts two opt-in dependencies on the API router. With `METANAVIT_API_KEY` unset the API is open, which is how the Compose stack, the mobile fixture backend and the tests run, and `/health` reports `auth.enabled: false`. With it set, every `/api` route and the two static mounts under `/api/files/` answer 401 unless the request carries the key (`X-API-Key`, or `Authorization: Bearer`; compared in constant time), and `/health` stays open for probes. It is one shared key, not user accounts: the `actor` on a plan decision is still whatever the client sends, and the bundled web UI does not send a key, so turning it on suits API clients or a proxy that adds the header. `RATE_LIMIT_PER_MINUTE` (bucket size `RATE_LIMIT_BURST`) turns on a token bucket per caller, keyed by the API key or else the client address, that answers 429 with `Retry-After`. The buckets live in process memory, so the budget is per replica: N replicas admit N times the configured budget and a restart resets it; a shared limit belongs at the proxy. `POST /api/chat/upload` reads its body under a byte cap (`UPLOAD_MAX_BYTES`) and answers 413 past it, and `GET /api/plans/decisions/recent` bounds `limit`. The log line that names the database prints host, port and database name, never the connection string.

**Python execution in chat.** The chat route's Python tool runs code the model writes, so it is off by default: `config/tools.yaml` does not list it, and `get_chat_engine()` builds the code agent only when `ENABLE_PYTHON_EXEC` is set and `METANAVIT_API_KEY` is configured. With the flag but no key the tool stays off and a warning says why; since the bundled web UI sends no key, turning it on suits API clients or a proxy that adds the header. When it is on, each call runs the code in a child process (`app/engine/tools/python_exec_runner.py`), never in the API process: the child starts in a new session with its own environment reduced to `PATH`, `HOME`, locale, `TZ` and `TMPDIR`, plus one-thread settings for the maths libraries (the API key and the connection string are not in it), lowers its own CPU-time, address-space and file-size limits before the code runs (`PYTHON_EXEC_TIMEOUT_SECONDS`, `PYTHON_EXEC_MAX_MEMORY_MB`), and its process group is killed at the time limit and again when the call returns, so a background process left in the group does not outlive the call. The result keeps its fields (output, error, result, figure path). This is a separate process with limits, not OS isolation: the code runs as the API's OS user and can read what that user can read, including the API process's start-up environment under `/proc` and a `.env` file in the working directory, and a process it starts in a session of its own leaves the group and is not killed. `tests/engine/test_python_exec_child.py` covers the default, the key requirement, the other process, the limits and the kill.

**Indexing.** `app/engine/ingest.py` brings the pgvector table in line with `DATA_DIR` on start: a manifest table holds path, content hash and chunk count per indexed file, and only new or changed files are embedded (see "Indexing that does not redo the whole tree"). `app/database/index_manager.py` and the crawling loaders under `app/engine/loaders` are the older design; no run path writes their `indexed_files` table.

**Sandbox.** OverlayFS dry-run for file mutations. MCP `propose_move` is the same idea as a staking interrupt: file mutations require a signed operator capability bound to the immutable plan and source state.

**Eval harness.** Does not need Postgres or a GPU. In-memory BM25 + hash dense over `bench/corpus/files`. `make bench` is one command. The git history of `bench/results/*.json` is the ablation log.

Models in `.env`: `qwen2.5:14b` via Ollama, `BAAI/bge-large-en-v1.5` (1024d), `BAAI/bge-reranker-v2-m3` when loaded. CI uses the overlap teacher so the table never pretends.

---

## Run it

`pyproject.toml` is the single dependency spec (version ranges). `make lock`
resolves it into pinned files: `requirements.txt` (eval + app) and one
`requirements/<profile>.txt` per install profile (`eval`, `app`, `mobile`,
`agent`, `dev`), each compiled under `requirements.txt` as a constraint so a
package has the same version in every file. The image and every CI job install
their pinned file first and then the package with `--no-deps`
(`make install-locked` does the same locally); `make lock-check`, which is
also a test and a CI step, fails when a pinned file no longer covers
`pyproject.toml`. The `ml` and `otel` extras have no pinned file: the image
installs them from their ranges with the pinned app file as a constraint, so
they are not reproducible to the same degree. `make lint` runs ruff with the
narrow defect-only rule set in `pyproject.toml` (syntax errors, undefined
names, shadowed imports); CI runs it as its own job.

Pick the extra for what you want to run:

| extra | installs | for |
|---|---|---|
| `eval` | numpy, pydantic, pytest, scikit-learn | `make bench`, `make test-eval` (the CPU benchmark above), `make bench-frontier` (its TF-IDF rows need scikit-learn) |
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

The compose stack is labelled honestly: lexical search is Postgres `ts_rank_cd`,
the default `LEXICAL_BACKEND` (`bm25_backend` in the response; the
`pgvector/pgvector:pg16` image has no ParadeDB `pg_search`), embeddings are the hash adapter
(`embedding_provider`), and no reranker is loaded (`reranker_loaded`). For
real models use the override, which installs the `ml` extra and loads
bge-large-en-v1.5 (~1.3 GB download) and bge-reranker-v2-m3:

```bash
docker compose down -v   # the table width changes from 256 to 1024
docker compose -f docker-compose.yml -f docker-compose.ml.yml up --build
```

**Full application locally** (chat needs Ollama; retrieval does not):

```bash
pip install -e ".[eval,app]"         # add ,ml for real embeddings / reranker (or `make install-locked` for the pinned versions CI uses)
cp .env.example .env                 # RETRIEVE_K=50, RERANK_TOP_N=8, ENABLE_ROUTER=true; METANAVIT_API_KEY empty = open API
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

**The API is not a hardened multi-replica service.** Authentication is off unless `METANAVIT_API_KEY` is set, and then it is a single shared key with no users or roles. The request budget is counted per process, so it is not a global rate limit. Pending file plans still live in one API process's memory. Schema migrations cover the decision log and the file-index tables only, with no down-migrations. Installs are pinned for the `eval`, `app`, `mobile`, `agent` and `dev` profiles; the `ml` and `otel` extras and the base images are not pinned.

**RL search policy (Search-R1 / GRPO) is not GPU-trained.** The env, reward, retrieved-token mask, and a CPU dummy GRPO step live in `app/rl/`. veRL + Qwen is still a training run. See [PLAN.md](../PLAN.md) Phase 10.

**The CI table's rerank row is the overlap teacher**, not `BAAI/bge-reranker-v2-m3`; it is within noise of plain RRF (see the [ablation table](#the-number)). The real reranker is measured in the [real-models tables](#real-models-and-a-corpus-where-k50-means-something), which are committed artifacts rather than CI output. Until M2 the cross-encoder was called through the wrong API (`compute_score` instead of sentence-transformers' `predict`). In the bench, the weights were never cached, so `get_cross_encoder()` returned nothing and the row fell back to overlap for lack of weights; the first run with weights present raised. In the production retriever the same call was inside a broad `except`, so reranking silently returned the RRF order. No earlier "bge-rerank" number in this repository's history was a cross-encoder result; `reranker_loaded` in the results file is the check.

**E2E scores in the older result files are token overlap.** The LLM evaluation with its validity gate is the [judge section](#does-the-judge-agree-with-the-gold-labels): deterministic checks first, then two judge models, and Cohen's κ vs the exact-match labels on simple_factual must reach 0.6 before a judge score is printed. Retrieval metrics stay BEIR-style and LLM-free.

**The injection evaluation is not an adaptive attack.** It plants one phrasing per goal and task, written for this benchmark, in two files per task, and runs two local 7-8B models. Adaptive attackers break most published defenses ([Nasr et al. 2025](https://arxiv.org/abs/2510.09023)), so a low attempt rate here says how these models treated these phrasings, not that a configuration resists prompt injection. There is no capability control (running each injected goal as an ordinary user task), so a model that rarely reaches the planted text or calls tools badly can show few attempts out of incapability; the exposure column is reported for that reason. The structured field check is told which config field each question asks about and reads that field from the config with comments dropped. It flags an answer that does not state that value, or that states another value for the field: for a numeric field, any number in the answer written with a decimal point or an exponent (an integer such as a run number does not count); for any other field, another value the field takes in the corpus configs, or a bare word in quotes or a code span (a path, a file name, a number, a key-value pair or the field's own name does not count). It does not use the planted value, and an unquoted value outside the field's vocabulary passes; the earlier rule flagged only answers without the true value, so it passed every answer that also named it. Its recall on planted-value answers, its false-positive rate on correct control answers and the planted-value rate with the check in front of the answer are measured offline from the stored answers, next to the earlier rule's recall, in the [generated table](../README.md#agent-injection). The rule was written with those answers in view, so that recall is in-sample, and the check has not run in the model loop. No model called a write tool in this run, so the run does not exercise the approval gate; the unchanged corpus follows from no grant being issued, and the gate's own evidence is the deterministic approval benchmark. The benign controls are not matched on retrieval: the planted lines add the run number and field name to two files, which can change what a search returns. Tool calls recovered from JSON text run whenever an answer contains a JSON object with a tool name, so a refusal that quoted a planted call would count as an attempt; none of the recovered calls was a write tool.

---

## What's next

Status: M0 to M5 are merged. Open items, in priority order:

1. **Kappa reference with variance.** The generator matched gold on all simple-factual questions, so the judge gate is not evaluable. This needs a harder slice as fixture v2; v1 stays frozen.
2. **Per-slice changed-query counts** in the harness, so a small-n category delta shows how many questions actually moved.
3. **Judge abstention handling.** Both judges labelled some NOT IN SOURCES answers correct.
4. **ParadeDB BM25 in the production path**, gated by the same parity job.
5. **Injection evaluation, next steps:** an injection-as-user-task capability control, a collection of phrasings per goal scored as any-success (AgentDojo's adaptive "Max" attack), larger models, and a CaMeL-style quarantined reader that extracts the requested field so planted text never reaches the planner.
6. Optional: a distilled student that keeps up with the cross-encoder it imitates (`make bench-rerank-arms` fits two on the cross-encoder's scores; both are below it on held-out questions), a listwise prompt that more replies answer with a full permutation, GPU Search-R1 / GRPO with `app/rl/reward.py`.

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
