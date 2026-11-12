.PHONY: install install-app install-ml install-locked lint lock-check test test-eval test-unit bench bench-compare bench-gate bench-check bench-baseline bench-agent-injection bench-agent-injection-rescore bench-agent-injection-scores bench-agent-injection-v2 bench-agent-injection-v2-rescore bench-jury-rescore bench-jury-heuristic bench-table bench-table-write bench-table-check evidence evidence-check bench-jury bench-frontier bench-neural bench-beir bench-beir-bm25-ablation bench-beir-rerank-cap bench-beir-fiqa parity explain-sql freeze-corpus sweeps figures matlab-demo demo-data gifs demos lock docker-smoke bench-frontier-neural bench-rerank-arms ingest-runs fixture-mtimes probes frontier-generations explain-sql-synthetic migrate migrate-check

PYTHON ?= python3
ROBOTICS_SNAPSHOT ?= bench/robotics/2026-10-07
export PYTHONPATH := $(CURDIR):$(PYTHONPATH)
SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo nogit)

# ---- setup -----------------------------------------------------------------
install:        ## bench + eval tests only: numpy, pydantic, pytest, scikit-learn (no LlamaIndex, no torch)
	$(PYTHON) -m pip install -e ".[eval]"

install-app:    ## + FastAPI, LlamaIndex, Postgres client (what the Docker image installs)
	$(PYTHON) -m pip install -e ".[eval,app]"

install-ml:     ## + torch, sentence-transformers, HuggingFace embeddings
	$(PYTHON) -m pip install -e ".[eval,app,ml]"

install-locked: ## the pinned versions CI and the image use: requirements.txt, then the package with no resolution
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install --no-deps -e .

# requirements.txt (eval + app) is the one resolution. Every requirements/<profile>.txt is
# compiled under it as a constraint (-c), so a package has the same version in every file
# and each CI job / the image installs exactly its own extras, pinned.
#   make lock                  upgrade everything to the newest allowed versions
#   make lock LOCK_UPGRADE=    keep the current pins; only add or drop what pyproject.toml changed
LOCK_UPGRADE ?= --upgrade
PIP_COMPILE = $(PYTHON) -m piptools compile --strip-extras
lock:           ## regenerate requirements.txt and requirements/*.txt from pyproject.toml
	$(PIP_COMPILE) $(LOCK_UPGRADE) --extra eval --extra app -o requirements.txt pyproject.toml
	$(PIP_COMPILE) -c requirements.txt --extra eval -o requirements/eval.txt pyproject.toml
	$(PIP_COMPILE) -c requirements.txt --extra app -o requirements/app.txt pyproject.toml
	$(PIP_COMPILE) -c requirements.txt --extra eval --extra mobile -o requirements/mobile.txt pyproject.toml
	$(PIP_COMPILE) -c requirements.txt --extra eval --extra mobile --extra agent -o requirements/agent.txt pyproject.toml
	$(PIP_COMPILE) -c requirements.txt --extra dev -o requirements/dev.txt pyproject.toml
	$(PIP_COMPILE) -c requirements.txt --extra distill -o requirements/distill.txt pyproject.toml

lock-check:     ## exit 1 if a pinned file no longer satisfies pyproject.toml or disagrees with requirements.txt
	$(PYTHON) scripts/check_lock.py

# ---- lint ------------------------------------------------------------------
lint:           ## ruff, narrow defect-only rule set from pyproject.toml (needs requirements/dev.txt)
	$(PYTHON) -m ruff check .

# ---- tests -----------------------------------------------------------------
test: test-eval test-unit

test-eval:      ## LLM-free eval harness tests (needs .[eval])
	$(PYTHON) -m pytest tests/eval -q --rootdir=$(CURDIR)

test-unit:      ## legacy unit tests + engine + API contract tests (needs .[eval,app], no database)
	$(PYTHON) -m pytest tests/unit_tests tests/engine tests/api tests/mcp -q --rootdir=$(CURDIR)

# ---- bench -----------------------------------------------------------------
bench:          ## frozen fixture v1 -> bench/results/<git-sha>.json and latest.json
	$(PYTHON) -m app.eval.harness

bench-compare:  ## report latest.json vs the committed baseline main.json
	$(PYTHON) -m app.eval.compare

bench-gate:     ## same, but exit 1 if nDCG@10 or Recall@50 drops more than the threshold
	$(PYTHON) -m app.eval.compare --fail-on-regression

bench-check:    ## run the CPU bench into bench/results/check.json; exit 1 unless every field of every row equals main.json (timings excepted)
	$(PYTHON) -m app.eval.harness --out bench/results/check.json
	$(PYTHON) -m app.eval.compare --exact --current bench/results/check.json

bench-baseline: ## promote latest.json to main.json (review `make bench-compare` first)
	cp bench/results/latest.json bench/results/main.json

bench-table:    ## render the README benchmark blocks from bench/results/main.json
	$(PYTHON) -m app.eval.report

bench-table-write: ## replace the generated blocks in README.md and doc/demo.html from main.json
	$(PYTHON) -m app.eval.report --write

bench-table-check: ## exit 1 if a generated block in README.md or doc/demo.html drifted from main.json
	$(PYTHON) -m app.eval.report --check

evidence:       ## render docs/RESUME_EVIDENCE.md from the committed results files
	$(PYTHON) -m app.eval.evidence

evidence-check: ## exit 1 if docs/RESUME_EVIDENCE.md drifted from the results files
	$(PYTHON) -m app.eval.evidence --check

bench-jury:     ## M4: LLM cited answers + deterministic checks + two-model jury with AB/BA swaps (needs Ollama + .[ml])
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.llm_eval --out bench/results/jury.json

bench-jury-rescore: ## recompute the deterministic checks + summary of bench/results/jury.json from its stored answers (no model call)
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.llm_eval --rescore bench/results/jury.json

bench-jury-heuristic: ## the pre-M4 jury columns on the hash table (heuristic judge; not for the README)
	$(PYTHON) -m app.eval.harness --jury

probes:         ## derived probe files outside the frozen fixture (paraphrases of gold questions; not gold labels) -> bench/probes/
	$(PYTHON) -m app.eval.probes

bench-frontier: ## prototype arms vs their parent rows, hash embedder + overlap reranker fallbacks -> bench/results/frontier.json
	$(PYTHON) -m app.eval.harness --frontier --out bench/results/frontier.json

FRONTIER_LLM ?= qwen2.5:7b
frontier-generations: ## LLM-written HyDE hypotheses (gold questions) and chunk prefixes through Ollama, stored -> bench/results/frontier_generations.json
	$(PYTHON) -m app.eval.frontier_llm --model $(FRONTIER_LLM) --out bench/results/frontier_generations.json

bench-frontier-neural: ## the same arms over bge-small + bge-reranker-v2-m3 (needs .[ml]; reads frontier_generations.json, no model call) -> bench/results/frontier_neural.json
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.harness --frontier --neural --out bench/results/frontier_neural.json

LISTWISE_MODEL ?=
bench-rerank-arms: ## distilled students vs the cross-encoder they imitate (needs .[ml]; the LightGBM student needs .[distill], else it is recorded as not evaluated); LISTWISE_MODEL=<ollama model> adds the listwise LLM arm -> bench/results/rerank_arms.json
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.rerank_arms $(if $(LISTWISE_MODEL),--listwise-model $(LISTWISE_MODEL)) --out bench/results/rerank_arms.json

.PHONY: bench-rrf-sweep
bench-rrf-sweep: ## RRF k x dense-vs-BM25 weight chosen on the tune half of the gold questions (bge-small), then scored against k=60 on the held-out half (needs .[ml]; never changes the default) -> bench/results/rrf_sweep.json
	$(PYTHON) -m app.eval.rrf_sweep --out bench/results/rrf_sweep.json

.PHONY: bench-hybrid-drops
bench-hybrid-drops: ## per question, the gold files hybrid@bge-base drops (and gains) at Recall@50 against dense_only@bge-base, with their dense, BM25 and fused ranks (needs .[ml]) -> bench/results/hybrid_drops_bge_base.json
	$(PYTHON) -m app.eval.hybrid_drops --out bench/results/hybrid_drops_bge_base.json

AGENT_MODELS ?= qwen2.5:7b,llama3.1:8b
bench-agent-injection: ## LLM-in-the-loop injection, the first run's task set (agent_injection_v1, kept as recorded): real models choose MCP tool calls over planted text (needs Ollama)
	$(PYTHON) -m app.eval.agent_injection --benchmark agent_injection_v1 --models $(AGENT_MODELS) --k 4 --out bench/results/agent_injection.json --episodes bench/results/agent_injection_episodes.jsonl

bench-agent-injection-rescore: ## rebuild bench/results/agent_injection.json from the committed episodes (no model call)
	$(PYTHON) -m app.eval.agent_injection --summarize-only --models $(AGENT_MODELS) --k 4 --out bench/results/agent_injection.json --episodes bench/results/agent_injection_episodes.jsonl

bench-agent-injection-scores: ## every committed episode's score from the current scoring code -> bench/results/agent_injection_episodes_rescored.jsonl (no model call; the episodes file is not changed)
	$(PYTHON) -m app.eval.agent_injection --write-scores bench/results/agent_injection_episodes_rescored.jsonl --k 4 --episodes bench/results/agent_injection_episodes.jsonl

.PHONY: bench-companion-live bench-companion-live-rescore
COMPANION_MODEL ?= qwen2.5:7b
COMPANION_LOG_DIR ?=
bench-companion-live: ## research companion, live mode: the Ollama model at OLLAMA_BASE_URL over two served APIs (default stack, databases created on PG_TEST_DSN), every gold question + ten planted-claim questions -> bench/results/companion_live.json; sets the live-run fields of doc/research-agent-validation.json
	$(PYTHON) -m app.eval.companion_live --model $(COMPANION_MODEL) --out bench/results/companion_live.json --validation doc/research-agent-validation.json $(if $(COMPANION_LOG_DIR),--log-dir $(COMPANION_LOG_DIR))

bench-companion-live-rescore: ## recompute the scores and summary of bench/results/companion_live.json from its stored responses (no model call)
	$(PYTHON) -m app.eval.companion_live --rescore bench/results/companion_live.json --validation doc/research-agent-validation.json

.PHONY: chat-route-smoke
CHAT_SMOKE_MODEL ?= qwen2.5:7b
CHAT_SMOKE_LOG_DIR ?=
chat-route-smoke: ## POST /api/chat and /api/chat/request with the Ollama model at OLLAMA_BASE_URL on the default stack (hash embedder, code execution off, database created on PG_TEST_DSN), the first gold question of each category -> bench/results/chat_route_smoke.json
	$(PYTHON) -m app.eval.chat_smoke --model $(CHAT_SMOKE_MODEL) --out bench/results/chat_route_smoke.json $(if $(CHAT_SMOKE_LOG_DIR),--log-dir $(CHAT_SMOKE_LOG_DIR))

bench-agent-injection-v2: ## LLM-in-the-loop injection, the current task set (agent_injection_v2, with the capability control) -> bench/results/agent_injection_v2.json and its episodes (needs Ollama)
	$(PYTHON) -m app.eval.agent_injection --benchmark agent_injection_v2 --models $(AGENT_MODELS) --k 4 --out bench/results/agent_injection_v2.json --episodes bench/results/agent_injection_v2_episodes.jsonl

bench-agent-injection-v2-rescore: ## rebuild the current run's scores and summary from its committed episodes -> bench/results/agent_injection_v2_rescore.json, not tracked, to compare with agent_injection_v2.json (no model call)
	$(PYTHON) -m app.eval.agent_injection --summarize-only --models $(AGENT_MODELS) --k 4 --out bench/results/agent_injection_v2_rescore.json --episodes bench/results/agent_injection_v2_episodes.jsonl

.PHONY: bench-trusted-actions
TRUSTED_ACTIONS_OUT ?= bench/results/trusted_actions_v2.json
bench-trusted-actions: ## approval-gate regression on temporary fixtures (no LLM); each row names the refusing check -> $(TRUSTED_ACTIONS_OUT)
	$(PYTHON) -m app.eval.trusted_actions --output $(TRUSTED_ACTIONS_OUT)

freeze-corpus:
	$(PYTHON) bench/corpus/build.py

fixture-mtimes: ## set bench/corpus/files mtimes to the MANIFEST's (contents untouched; refuses on drift): the served indexer reads disk mtimes
	$(PYTHON) -m app.eval.corpus --apply-mtimes

sweeps:
	$(PYTHON) -m app.eval.sweeps

# ---- neural rows (needs .[ml]; downloads bge-small / bge-base / bge-reranker-v2-m3 once) ----
bench-neural:   ## fixture v1 with real embedders + the real reranker -> bench/results/neural.json
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.harness --neural --out bench/results/neural.json

.PHONY: bench-rerank-depth
bench-rerank-depth: ## fixture v1, bge-small + bge-reranker-v2-m3: every fused candidate -> depth 20 -> max_length 512 -> fp16 (the served settings) -> + router and staleness -> staleness before the rerank, each vs its parent (needs .[ml] and CUDA) -> bench/results/rerank_depth.json
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.harness --rerank-depth --out bench/results/rerank_depth.json

# The BEIR archives are downloaded once into BEIR_DATA_DIR (gitignored by default; sha256-pinned in app/eval/beir.py).
BEIR_DATA_DIR ?= bench/external
bench-beir:     ## BEIR SciFact: BM25 / dense / hybrid / + reranker -> bench/results/beir_scifact.json
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.beir --data-dir $(BEIR_DATA_DIR) --out bench/results/beir_scifact.json

bench-beir-bm25-ablation: ## BEIR SciFact BM25 (CPU, no model): the analyzer alone, k1/b alone and both, each vs bm25 -> bench/results/beir_scifact_bm25_ablation.json
	$(PYTHON) -m app.eval.beir --ladder bm25-ablation --device cpu --data-dir $(BEIR_DATA_DIR) --out bench/results/beir_scifact_bm25_ablation.json

BEIR_RERANK_CAP_ROWS = hybrid@bge-small,hybrid+bge-rerank@bge-small,hybrid+bge-rerank@bge-small/fp32-512/top20,hybrid+bge-rerank@bge-small/fp16-512/top20
bench-beir-rerank-cap: ## BEIR SciFact reranker at depth 20: fp32 uncapped -> fp32 max_length 512 -> fp16 512, each vs its parent -> bench/results/beir_scifact_rerank_cap.json
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.beir --data-dir $(BEIR_DATA_DIR) --only $(BEIR_RERANK_CAP_ROWS) --out bench/results/beir_scifact_rerank_cap.json

BEIR_FIQA_ROWS = bm25,dense@bge-small,hybrid@bge-small,hybrid+bge-rerank@bge-small/fp32-512/top20,hybrid+bge-rerank@bge-small/fp16-512/top20
bench-beir-fiqa: ## BEIR FiQA: BM25 / dense / hybrid / + bge-reranker at depth 20 (fp32, then fp16; max_length 512), TREC runs -> bench/results/beir_fiqa.json
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.beir --dataset fiqa --data-dir $(BEIR_DATA_DIR) --only $(BEIR_FIQA_ROWS) --trec-dir bench/results/beir_fiqa_trec --out bench/results/beir_fiqa.json

.PHONY: robotics-verify bench-robotics bench-robotics-chunking bench-robotics-multiquery
robotics-verify: ## verify real experiment source hashes, evidence spans and family split; prints the files that differ from their tracked blob
	$(PYTHON) -m app.eval.robotics verify --snapshot $(ROBOTICS_SNAPSHOT)

bench-robotics: robotics-verify ## generated lookup benchmark; preserves the measured reference
	$(PYTHON) -m app.eval.robotics evaluate --snapshot $(ROBOTICS_SNAPSHOT) --output bench/results/robotics_file_lookup_latest.json

bench-robotics-chunking: robotics-verify ## structure-aware vs fixed-window chunks on the robotics snapshot, dev and test, paired family bootstrap (CPU, hash embedder) -> bench/results/robotics_chunking.json
	$(PYTHON) -m app.eval.robotics_checks chunking --snapshot $(ROBOTICS_SNAPSHOT) --out bench/results/robotics_chunking.json

bench-robotics-multiquery: robotics-verify ## hybrid vs hybrid+multiquery (keyword-heuristic rewrites) on the robotics snapshot, dev and test -> bench/results/robotics_multiquery.json
	$(PYTHON) -m app.eval.robotics_checks multiquery --snapshot $(ROBOTICS_SNAPSHOT) --out bench/results/robotics_multiquery.json

# ---- demos / figures -------------------------------------------------------
figures:
	$(PYTHON) -m app.eval.figures

matlab-demo:
	$(PYTHON) -m app.eval.matlab_demo

demo-data:
	$(PYTHON) -m app.eval.demo_export

gifs: demo-data
	$(PYTHON) -m app.eval.gifs

demos: matlab-demo demo-data figures gifs

# ---- parity (API over Postgres vs the in-memory bench) ----------------------
API_URL ?= http://localhost:8000
PARITY_OUT ?= bench/results/parity.json
# The in-memory row's named configuration (app/eval/parity.py IN_MEMORY_CONFIGS): CI's hash one by default; the
# model-backed one is in_memory:hybrid+bge-rerank+router+staleness@bge-small/fp16-512/top20/staleness-first.
PARITY_CONFIG ?= in_memory:hybrid+router+staleness

parity:         ## replay the gold set through a running API; report the nDCG@10 gap with its paired CI against the in-memory row PARITY_CONFIG (fails if the API is broken; refuses a fixture that differs from its MANIFEST)
	$(PYTHON) -m app.eval.harness --backend api --url $(API_URL) --out $(PARITY_OUT) --parity-config '$(PARITY_CONFIG)'

migrate:        ## the deploy step: apply pending schema migrations (PG_CONNECTION_STRING; vector table from PGVECTOR_TABLE, EMBEDDING_DIM, LEXICAL_BACKEND). Until it has run, the API answers 503 and retries.
	$(PYTHON) -m app.database.migrate

migrate-check:  ## list pending schema migrations, exit 1 if any (applies nothing)
	$(PYTHON) -m app.database.migrate --check

ingest-runs:    ## first start, unchanged restart, one edit, one delete on a copy of the fixture (PG_TEST_DSN) -> bench/results/ingest_runs.json
	$(PYTHON) -m app.engine.ingest_runs

explain-sql:    ## EXPLAIN (ANALYZE, BUFFERS) of the hybrid query against PG_TEST_DSN (fixture v1 already indexed) -> doc/sql/hybrid_explain.txt
	$(PYTHON) scripts/explain_hybrid.py

explain-sql-synthetic: ## the same over 10k generated hash-vector rows in a scratch table on PG_TEST_DSN -> doc/sql/hybrid_explain_synthetic_10k.txt
	$(PYTHON) scripts/explain_hybrid.py --synthetic-rows 10000 --out doc/sql/hybrid_explain_synthetic_10k.txt

# ---- docker ----------------------------------------------------------------
docker-smoke:   ## build the compose stack, wait for /health, POST one query, tear down
	docker compose up --build -d --wait --wait-timeout 300
	$(PYTHON) scripts/smoke_retrieve.py
	docker compose down -v
