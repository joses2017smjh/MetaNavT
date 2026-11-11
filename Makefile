.PHONY: install install-app install-ml install-locked lint lock-check test test-eval test-unit bench bench-compare bench-gate bench-check bench-baseline bench-agent-injection bench-agent-injection-rescore bench-agent-injection-scores bench-jury-rescore bench-jury-heuristic bench-table bench-table-write bench-table-check evidence evidence-check bench-jury bench-frontier bench-neural bench-beir parity explain-sql freeze-corpus sweeps figures matlab-demo demo-data gifs demos lock docker-smoke bench-frontier-neural bench-rerank-arms ingest-runs fixture-mtimes

PYTHON ?= python3
ROBOTICS_SNAPSHOT ?= bench/robotics/2026-10-07
export PYTHONPATH := $(CURDIR):$(PYTHONPATH)
SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo nogit)

# ---- setup -----------------------------------------------------------------
install:        ## bench + eval tests only: numpy, pydantic, pytest (no LlamaIndex, no torch)
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

bench-frontier: ## prototype arms vs their parent rows, hash embedder + overlap reranker fallbacks -> bench/results/frontier.json
	$(PYTHON) -m app.eval.harness --frontier --out bench/results/frontier.json

bench-frontier-neural: ## the same arms over bge-small + bge-reranker-v2-m3 (needs .[ml]) -> bench/results/frontier_neural.json
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.harness --frontier --neural --out bench/results/frontier_neural.json

LISTWISE_MODEL ?=
bench-rerank-arms: ## distilled student vs the cross-encoder it imitates (needs .[ml]); LISTWISE_MODEL=<ollama model> adds the listwise LLM arm -> bench/results/rerank_arms.json
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.rerank_arms $(if $(LISTWISE_MODEL),--listwise-model $(LISTWISE_MODEL)) --out bench/results/rerank_arms.json

AGENT_MODELS ?= qwen2.5:7b,llama3.1:8b
bench-agent-injection: ## LLM-in-the-loop injection: real models choose MCP tool calls over planted text (needs Ollama)
	$(PYTHON) -m app.eval.agent_injection --models $(AGENT_MODELS) --k 4 --out bench/results/agent_injection.json --episodes bench/results/agent_injection_episodes.jsonl

bench-agent-injection-rescore: ## rebuild bench/results/agent_injection.json from the committed episodes (no model call)
	$(PYTHON) -m app.eval.agent_injection --summarize-only --models $(AGENT_MODELS) --k 4 --out bench/results/agent_injection.json --episodes bench/results/agent_injection_episodes.jsonl

bench-agent-injection-scores: ## every committed episode's score from the current scoring code -> bench/results/agent_injection_episodes_rescored.jsonl (no model call; the episodes file is not changed)
	$(PYTHON) -m app.eval.agent_injection --write-scores bench/results/agent_injection_episodes_rescored.jsonl --k 4 --episodes bench/results/agent_injection_episodes.jsonl

freeze-corpus:
	$(PYTHON) bench/corpus/build.py

fixture-mtimes: ## set bench/corpus/files mtimes to the MANIFEST's (contents untouched; refuses on drift): the served indexer reads disk mtimes
	$(PYTHON) -m app.eval.corpus --apply-mtimes

sweeps:
	$(PYTHON) -m app.eval.sweeps

# ---- neural rows (needs .[ml]; downloads bge-small / bge-base / bge-reranker-v2-m3 once) ----
bench-neural:   ## fixture v1 with real embedders + the real reranker -> bench/results/neural.json
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.harness --neural --out bench/results/neural.json

bench-beir:     ## BEIR SciFact: BM25 / dense / hybrid / + reranker -> bench/results/beir_scifact.json
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.beir --out bench/results/beir_scifact.json

.PHONY: robotics-verify bench-robotics
robotics-verify: ## verify real experiment source hashes, evidence spans and family split
	$(PYTHON) -m app.eval.robotics verify --snapshot $(ROBOTICS_SNAPSHOT)

bench-robotics: robotics-verify ## generated lookup benchmark; preserves the measured reference
	$(PYTHON) -m app.eval.robotics evaluate --snapshot $(ROBOTICS_SNAPSHOT) --output bench/results/robotics_file_lookup_latest.json

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

parity:         ## replay the gold set through a running API; report the nDCG@10 gap with its paired CI (fails only if the API is broken)
	$(PYTHON) -m app.eval.harness --backend api --url $(API_URL) --out $(PARITY_OUT)

ingest-runs:    ## first start, unchanged restart, one edit, one delete on a copy of the fixture (PG_TEST_DSN) -> bench/results/ingest_runs.json
	$(PYTHON) -m app.engine.ingest_runs

explain-sql:    ## EXPLAIN (ANALYZE, BUFFERS) of the hybrid query against PG_TEST_DSN -> doc/sql/
	$(PYTHON) scripts/explain_hybrid.py

# ---- docker ----------------------------------------------------------------
docker-smoke:   ## build the compose stack, wait for /health, POST one query, tear down
	docker compose up --build -d --wait --wait-timeout 300
	$(PYTHON) scripts/smoke_retrieve.py
	docker compose down -v
