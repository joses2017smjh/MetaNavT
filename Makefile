.PHONY: install install-app install-ml install-locked lint lock-check test test-eval test-unit bench bench-compare bench-gate bench-check bench-baseline bench-agent-injection bench-agent-injection-rescore bench-agent-injection-scores bench-agent-injection-v2 bench-agent-injection-v2-rescore bench-agent-injection-v3 bench-agent-injection-v3-shard bench-agent-injection-v3-merge bench-agent-injection-v3-rescore bench-agent-injection-adaptive bench-agent-injection-adaptive-tune bench-agent-injection-adaptive-shard bench-agent-injection-adaptive-merge bench-agent-injection-adaptive-rescore bench-jury-rescore bench-jury-heuristic bench-table bench-table-write bench-table-check evidence evidence-check bench-jury bench-frontier bench-neural bench-beir bench-beir-bm25-ablation bench-beir-rerank-cap bench-beir-fiqa parity explain-sql freeze-corpus sweeps figures matlab-demo demo-data gifs demos lock docker-smoke bench-frontier-neural bench-rerank-arms ingest-runs fixture-mtimes probes frontier-generations explain-sql-synthetic migrate migrate-check bench-jury-v2 bench-jury-v2-rescore ingest-scale index bench-served-scale

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
# requirements/ml.txt is resolved with PyTorch's CPU wheel index next to PyPI, so torch is
# pinned as its CPU build (+cpu) with no CUDA packages; pip-compile writes the index line
# into the file, so `pip install -r requirements/ml.txt` finds the wheel.
# The image installs app.txt, ml.txt and otel.txt with --no-deps, so those three are compiled
# with --allow-unsafe and pin setuptools where it is needed (app.txt, ml.txt): llama-index-core
# needs a newer one than the base image ships, which pip used to fetch unpinned while it
# resolved app.txt.
#   make lock                  upgrade everything to the newest allowed versions
#   make lock LOCK_UPGRADE=    keep the current pins; only add or drop what pyproject.toml changed
LOCK_UPGRADE ?= --upgrade
PIP_COMPILE = $(PYTHON) -m piptools compile --strip-extras
TORCH_CPU_INDEX = https://download.pytorch.org/whl/cpu
lock:           ## regenerate requirements.txt and requirements/*.txt from pyproject.toml
	$(PIP_COMPILE) $(LOCK_UPGRADE) --extra eval --extra app -o requirements.txt pyproject.toml
	$(PIP_COMPILE) -c requirements.txt --extra eval -o requirements/eval.txt pyproject.toml
	$(PIP_COMPILE) --allow-unsafe -c requirements.txt --extra app -o requirements/app.txt pyproject.toml
	$(PIP_COMPILE) -c requirements.txt --extra eval --extra mobile -o requirements/mobile.txt pyproject.toml
	$(PIP_COMPILE) -c requirements.txt --extra eval --extra mobile --extra agent -o requirements/agent.txt pyproject.toml
	$(PIP_COMPILE) -c requirements.txt --extra dev -o requirements/dev.txt pyproject.toml
	$(PIP_COMPILE) -c requirements.txt --extra distill -o requirements/distill.txt pyproject.toml
	$(PIP_COMPILE) --allow-unsafe -c requirements.txt --extra ml --extra-index-url $(TORCH_CPU_INDEX) -o requirements/ml.txt pyproject.toml
	$(PIP_COMPILE) --allow-unsafe -c requirements.txt --extra otel -o requirements/otel.txt pyproject.toml

lock-check:     ## exit 1 if a pinned file no longer satisfies pyproject.toml or disagrees with requirements.txt
	$(PYTHON) scripts/check_lock.py

# ---- lint ------------------------------------------------------------------
lint:           ## ruff, narrow rule set from pyproject.toml: defects and unused imports (needs requirements/dev.txt)
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

bench-gate:     ## same, but exit 1 if nDCG@10, Recall@10 or Recall@50 drops more than the threshold
	$(PYTHON) -m app.eval.compare --fail-on-regression

bench-check:    ## run the CPU bench into bench/results/check.json; exit 1 unless every field of every row equals main.json (timings excepted)
	$(PYTHON) -m app.eval.harness --out bench/results/check.json
	$(PYTHON) -m app.eval.compare --exact --current bench/results/check.json

bench-baseline: ## promote latest.json and its per-question sidecar to main.json (review `make bench-compare` first)
	cp bench/results/latest.json bench/results/main.json
	cp bench/results/latest.per_query.json bench/results/main.per_query.json

.PHONY: bench-intervals
RESULTS ?= bench/results/main.json
bench-intervals: ## recompute every interval, paired delta and mean of a harness results file (RESULTS) from its per-question sidecar; exit 1 unless all equal the file
	$(PYTHON) -m app.eval.per_question $(RESULTS)

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

.PHONY: doc-numbers-check
doc-numbers-check: ## exit 1 if a number in the prose of doc/engineering-reference.md or README.md (outside generated blocks) differs from the file it cites, or is neither cited nor classified
	$(PYTHON) -m app.eval.prose_numbers

bench-jury:     ## M4: LLM cited answers + deterministic checks + two-model jury with AB/BA swaps (needs Ollama + .[ml]); rewrites bench/results/jury.json, the M4 record, with today's protocol
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.llm_eval --out bench/results/jury.json

bench-jury-rescore: ## recompute the deterministic checks + summary of bench/results/jury.json from its stored answers (no model call)
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.llm_eval --rescore bench/results/jury.json

bench-jury-v2:  ## M29: qwen2.5:7b answers + a calibration set from qwen2.5:1.5b (retrieved and planted contexts) as the kappa gate's reference, judges qwen2.5:7b + llama3.1:8b (needs Ollama + .[ml]) -> bench/results/jury_v2.json
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.llm_eval --out bench/results/jury_v2.json --generator qwen2.5:7b --judges qwen2.5:7b,llama3.1:8b --calibration-generator qwen2.5:1.5b

bench-jury-v2-rescore: ## recompute the deterministic checks + summary of bench/results/jury_v2.json, calibration set included, from its stored answers (no model call)
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.llm_eval --rescore bench/results/jury_v2.json

# ---- M42: current open models as generators and judges, in two stages (app/eval/judge_stored.py) ----
# Stage 1, one run per generator: answers and deterministic checks, no judge. Stage 2, one run per judge: the judge is
# gated on the calibration answers stored in jury_v2.json (never regenerated), then labels jury_v2's stored answers
# and each generator's. Both need an Ollama server at OLLAMA_BASE_URL that has the models; stage 1 also needs .[ml].
# Nothing here rewrites jury.json or jury_v2.json.
.PHONY: bench-jury-v3-answers bench-jury-v3-answers-rescore bench-jury-v3-judge bench-jury-v3-check bench-judge-stored
# GENERATOR: the Ollama model that answers. THINK=off sends `think: false` with every generation call, for a model
# whose reasoning does not end inside the context (file suffix _think-off). ANSWERS_LIMIT=N answers the first N
# questions only (file suffix _firstN): a record of a configuration that fails, never judged.
GENERATOR ?= qwen3.5:27b
THINK ?= default
ANSWERS_LIMIT ?=
jury_v3_answers_out = bench/results/jury_v3_answers_$(subst :,-,$(GENERATOR))$(if $(filter off,$(THINK)),_think-off)$(if $(ANSWERS_LIMIT),_first$(ANSWERS_LIMIT)).json
jury_v3_answers_all = $(sort $(wildcard bench/results/jury_v3_answers_*.json))
jury_v3_answers_full = $(filter-out $(wildcard bench/results/jury_v3_answers_*_first*.json),$(jury_v3_answers_all))
jury_v3_judge = bench/results/jury_v3_judge_$(subst :,-,$(1)).json
bench-jury-v3-answers: ## M42 stage 1: GENERATOR's cited answers + deterministic checks on fixture v1, no judge (needs Ollama + .[ml]; THINK=off, ANSWERS_LIMIT=N) -> bench/results/jury_v3_answers_<generator>[_think-off][_firstN].json
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.llm_eval --out $(jury_v3_answers_out) --generator $(GENERATOR) --judges none $(if $(filter off,$(THINK)),--think off) $(if $(ANSWERS_LIMIT),--limit $(ANSWERS_LIMIT))

bench-jury-v3-answers-rescore: ## recompute the deterministic checks + summary of every jury_v3 answers file present from its stored answers (no LLM call; needs .[ml])
	for f in $(jury_v3_answers_all); do BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.llm_eval --rescore $$f || exit 1; done

# JUDGE: the Ollama model that judges. ANSWERS: comma-separated files (answer-evaluation files or JSONL of stored
# answers: id, question, gold, answer, cited_paths). JUDGED: the file written. JUDGE_FLAGS=--resume continues one.
JUDGE ?= gpt-oss:120b
ANSWERS ?=
JUDGED ?= $(basename $(firstword $(subst $(comma), ,$(ANSWERS))))_judged_by_$(subst :,-,$(JUDGE)).json
JUDGE_FLAGS ?=
comma := ,
empty :=
space := $(empty) $(empty)
bench-judge-stored: ## judge stored answers with a judge validated in the same run: JUDGE is gated on jury_v2.json's stored calibration answers, then labels ANSWERS (needs Ollama) -> JUDGED
	@test -n "$(ANSWERS)" || { echo "set ANSWERS to the stored-answers file(s), comma-separated (JSONL: id, question, gold, answer, cited_paths)"; exit 2; }
	$(PYTHON) -m app.eval.judge_stored --judge $(JUDGE) --answers $(ANSWERS) --out $(JUDGED) $(JUDGE_FLAGS)

bench-jury-v3-judge: ## M42 stage 2: JUDGE over jury_v2.json's stored answers and every full jury_v3 answers file present (needs Ollama) -> bench/results/jury_v3_judge_<judge>.json
	$(PYTHON) -m app.eval.judge_stored --judge $(JUDGE) --answers $(subst $(space),$(comma),$(strip bench/results/jury_v2.json $(jury_v3_answers_full))) --out $(call jury_v3_judge,$(JUDGE)) $(JUDGE_FLAGS)

bench-jury-v3-check: ## recompute the gate and every summary of each jury_v3 judge file present from its stored verdicts and its inputs' hashes (no model call, no retrieval); exit 1 if anything differs
	for f in $(sort $(wildcard bench/results/jury_v3_judge_*.json)); do $(PYTHON) -m app.eval.judge_stored --check $$f || exit 1; done

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

.PHONY: bench-file-depth
bench-file-depth: ## M37: BM25, dense-only and hybrid rows under the file-depth cut (every list keeps chunks until it holds retrieve_k files) per real embedder, each paired with its chunk-cut parent recomputed in the run (needs .[ml]; adds rows, changes none) -> bench/results/file_depth.json
	$(PYTHON) -m app.eval.file_depth --out bench/results/file_depth.json

.PHONY: bench-rrf-sweep-hash
bench-rrf-sweep-hash: ## the same RRF sweep with the hash embedder on the CPU, recorded by the owner's decision (record, do not adopt): its held-out half was seen before this run, so not blind; the default never changes -> bench/results/rrf_sweep_hash.json
	$(PYTHON) -m app.eval.rrf_sweep --embedder hash --device cpu --out bench/results/rrf_sweep_hash.json

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

bench-agent-injection-v2: ## LLM-in-the-loop injection, the second run's task set (agent_injection_v2, with the capability control) -> bench/results/agent_injection_v2.json and its episodes (needs Ollama)
	$(PYTHON) -m app.eval.agent_injection --benchmark agent_injection_v2 --models $(AGENT_MODELS) --k 4 --out bench/results/agent_injection_v2.json --episodes bench/results/agent_injection_v2_episodes.jsonl

bench-agent-injection-v2-rescore: ## rebuild the second run's scores and summary from its committed episodes -> bench/results/agent_injection_v2_rescore.json, not tracked, to compare with agent_injection_v2.json (no model call)
	$(PYTHON) -m app.eval.agent_injection --summarize-only --models $(AGENT_MODELS) --k 4 --out bench/results/agent_injection_v2_rescore.json --episodes bench/results/agent_injection_v2_episodes.jsonl

# agent_injection_v3: every goal in seven phrasings (any-success) and the answer check in the loop as a fourth configuration.
# One process runs it all; the committed run was split into shards (one GPU job each) and merged.
SHARD_MODELS ?= $(AGENT_MODELS)
bench-agent-injection-v3: ## LLM-in-the-loop injection, phrasings scored as any-success and the in-loop answer check (agent_injection_v3), in one process -> bench/results/agent_injection_v3.json and its episodes (needs Ollama)
	$(PYTHON) -m app.eval.agent_injection --benchmark agent_injection_v3 --models $(AGENT_MODELS) --k 4 --out bench/results/agent_injection_v3.json --episodes bench/results/agent_injection_v3_episodes.jsonl

bench-agent-injection-v3-shard: ## one shard of agent_injection_v3, SHARD=name SHARD_MODELS=... SHARD_QUESTIONS=q011,... (SHARD_FLAGS=--resume continues one) -> bench/results/agent_injection_v3_shards/, not tracked (needs Ollama)
	$(PYTHON) -m app.eval.agent_injection --benchmark agent_injection_v3 --models $(SHARD_MODELS) $(if $(SHARD_QUESTIONS),--questions $(SHARD_QUESTIONS)) --k 4 --out bench/results/agent_injection_v3_shards/$(SHARD).json --episodes bench/results/agent_injection_v3_shards/$(SHARD)_episodes.jsonl $(SHARD_FLAGS)

bench-agent-injection-v3-merge: ## merge the shards in bench/results/agent_injection_v3_shards/ into agent_injection_v3_episodes.jsonl in a fixed order and score it -> bench/results/agent_injection_v3.json (no model call)
	$(PYTHON) -m app.eval.agent_injection --merge $(sort $(wildcard bench/results/agent_injection_v3_shards/*_episodes.jsonl)) --models $(AGENT_MODELS) --k 4 --out bench/results/agent_injection_v3.json --episodes bench/results/agent_injection_v3_episodes.jsonl

bench-agent-injection-v3-rescore: ## rebuild agent_injection_v3's scores and summary from its committed episodes -> bench/results/agent_injection_v3_rescore.json, not tracked, to compare with agent_injection_v3.json (no model call)
	$(PYTHON) -m app.eval.agent_injection --summarize-only --models $(AGENT_MODELS) --k 4 --out bench/results/agent_injection_v3_rescore.json --episodes bench/results/agent_injection_v3_episodes.jsonl

# agent_injection_adaptive (D20): an attacker model rewrites the planted text round by round against each configuration,
# from the scorer's outcome on its earlier rounds. Its instructions are tuned on the training questions (the tune target,
# one trial, not tracked); the held-out questions are the scored run, in one process or as shards merged like v3's.
TUNE ?= tune
ADAPTIVE_MODELS ?= qwen2.5:7b
bench-agent-injection-adaptive: ## the adaptive attacker on the held-out questions (agent_injection_adaptive), in one process -> bench/results/agent_injection_adaptive.json and its episodes (needs Ollama)
	$(PYTHON) -m app.eval.agent_injection_adaptive --split held_out --models $(ADAPTIVE_MODELS) --out bench/results/agent_injection_adaptive.json --episodes bench/results/agent_injection_adaptive_episodes.jsonl

bench-agent-injection-adaptive-tune: ## the attacker on the training questions, one trial, TUNE=name SHARD_MODELS=... (SHARD_FLAGS=--resume continues one) -> bench/results/agent_injection_adaptive_tune/, not tracked (needs Ollama)
	$(PYTHON) -m app.eval.agent_injection_adaptive --split train --k 1 --models $(SHARD_MODELS) --out bench/results/agent_injection_adaptive_tune/$(TUNE).json --episodes bench/results/agent_injection_adaptive_tune/$(TUNE)_episodes.jsonl $(SHARD_FLAGS)

bench-agent-injection-adaptive-shard: ## one shard of the held-out run, SHARD=name SHARD_MODELS=... SHARD_QUESTIONS=q021,... (SHARD_FLAGS=--resume continues one) -> bench/results/agent_injection_adaptive_shards/, not tracked (needs Ollama)
	$(PYTHON) -m app.eval.agent_injection_adaptive --split held_out --models $(SHARD_MODELS) $(if $(SHARD_QUESTIONS),--questions $(SHARD_QUESTIONS)) --out bench/results/agent_injection_adaptive_shards/$(SHARD).json --episodes bench/results/agent_injection_adaptive_shards/$(SHARD)_episodes.jsonl $(SHARD_FLAGS)

bench-agent-injection-adaptive-merge: ## merge the shards in bench/results/agent_injection_adaptive_shards/ in a fixed order and score them, compared with v3's phrasings -> bench/results/agent_injection_adaptive.json (no model call)
	$(PYTHON) -m app.eval.agent_injection_adaptive --merge $(sort $(wildcard bench/results/agent_injection_adaptive_shards/*_episodes.jsonl)) --models $(ADAPTIVE_MODELS) --out bench/results/agent_injection_adaptive.json --episodes bench/results/agent_injection_adaptive_episodes.jsonl

bench-agent-injection-adaptive-rescore: ## rebuild the adaptive run's scores, summary and comparison from its committed episodes -> bench/results/agent_injection_adaptive_rescore.json, not tracked (no model call)
	$(PYTHON) -m app.eval.agent_injection_adaptive --summarize-only --models $(ADAPTIVE_MODELS) --out bench/results/agent_injection_adaptive_rescore.json --episodes bench/results/agent_injection_adaptive_episodes.jsonl

.PHONY: bench-trusted-actions
# ---- the search policy (M44): app/rl, trained once ----------------------------
# SEARCH_POLICY_WORK is a directory OUTSIDE the repository: the training corpora, the run directories (checkpoints,
# curves) and the arm files live there. Only search-policy-eval writes under bench/results.
.PHONY: search-policy-corpora search-policy-train search-policy-eval search-policy-rescore
SEARCH_POLICY_WORK ?=
SEARCH_POLICY_CORPUS_SEEDS ?= 1-32
SEARCH_POLICY_ALGO ?= grpo
SEARCH_POLICY_SEED ?= 0
SEARCH_POLICY_SEEDS ?= 0 1 2
SEARCH_POLICY_STEPS ?= 300
SEARCH_POLICY_ARGS ?=
SEARCH_POLICY_JOBS ?=
search-policy-corpora: ## M44: the search policy's training corpora, seeded variants written by the fixture's builder (never fixture v1), with the leakage report -> $(SEARCH_POLICY_WORK)/corpora
	$(PYTHON) -m app.rl.corpora --seeds $(SEARCH_POLICY_CORPUS_SEEDS) --out $(SEARCH_POLICY_WORK)/corpora

search-policy-train: ## M44: one training run of the search policy on a GPU (needs .[ml,rl]): SEARCH_POLICY_SEED, SEARCH_POLICY_ALGO=grpo|ppo, SEARCH_POLICY_ARGS=--resume continues one -> $(SEARCH_POLICY_WORK)/runs/<algo>-seed<seed>
	$(PYTHON) -m app.rl.train --out $(SEARCH_POLICY_WORK)/runs/$(SEARCH_POLICY_ALGO)-seed$(SEARCH_POLICY_SEED) --corpora $(SEARCH_POLICY_WORK)/corpora --corpus-seeds $(SEARCH_POLICY_CORPUS_SEEDS) --algo $(SEARCH_POLICY_ALGO) --seed $(SEARCH_POLICY_SEED) --steps $(SEARCH_POLICY_STEPS) $(SEARCH_POLICY_ARGS)

search-policy-eval: ## M44: the held-out arms on a GPU (untrained, single-shot RAG, each seed of SEARCH_POLICY_SEEDS), then the results file from them and the training runs (needs .[ml,rl]) -> bench/results/search_policy.json and its two sidecars
	$(PYTHON) -m app.rl.evaluate --out-dir $(SEARCH_POLICY_WORK)/arms $(foreach s,$(SEARCH_POLICY_SEEDS),--adapter $(SEARCH_POLICY_ALGO)/seed=$(s)=$(SEARCH_POLICY_WORK)/runs/$(SEARCH_POLICY_ALGO)-seed$(s)/best) $(SEARCH_POLICY_ARGS)
	$(PYTHON) -m app.rl.results --arms $(SEARCH_POLICY_WORK)/arms $(foreach s,$(SEARCH_POLICY_SEEDS),--run $(SEARCH_POLICY_ALGO)/seed=$(s)=$(SEARCH_POLICY_WORK)/runs/$(SEARCH_POLICY_ALGO)-seed$(s)) $(if $(SEARCH_POLICY_JOBS),--jobs $(SEARCH_POLICY_JOBS))

search-policy-rescore: ## recompute bench/results/search_policy.json from its sidecars on a CPU (every episode scored again, every interval redrawn; no model, no torch); exit 1 unless every value equals the file
	$(PYTHON) -m app.rl.results --check

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

# ---- M41: current open embedders and rerankers beside the committed bge rows (app/eval/models_2026.py) ----
# Every model must load and pass its load-and-score check, or the run stops and writes nothing; the bge parents are
# recomputed in the same run, and whether they equal the committed rows is recorded whatever the answer.
.PHONY: models-2026-check bench-models-2026 bench-models-2026-harrier bench-models-2026-cpu
models-2026-check: ## load each new model once, read its dtype back and run its load-and-score check (needs .[ml]; MODELS=tag,... for a subset) -> stdout
	$(PYTHON) -m app.eval.models_2026 --check $(if $(MODELS),--models $(MODELS))

bench-models-2026: ## fixture v1: Qwen3-Embedding 0.6B/8B (dense, hybrid) and mxbai-rerank-base/large-v2, Qwen3-Reranker 0.6B/8B over hybrid@bge-small, each vs its bge parent recomputed in the run (needs .[ml] and a 48 GB GPU) -> bench/results/models_2026.json
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.harness --models-2026 main --out bench/results/models_2026.json

bench-models-2026-harrier: ## fixture v1: harrier-oss-v1-27b (dense, hybrid) vs bge-small, recomputed in the run (needs .[ml] and an 80 GB GPU) -> bench/results/models_2026_harrier.json
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.harness --models-2026 harrier --out bench/results/models_2026_harrier.json

bench-models-2026-cpu: ## fixture v1 on the CPU: the CPU-servable models (Qwen3-Embedding-0.6B, mxbai-rerank-base-v2, Qwen3-Reranker-0.6B) at the app's default CPU settings (depth 20, max_length 512, fp32) vs bge, for their latency (needs .[ml]) -> bench/results/models_2026_cpu.json
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.harness --models-2026 cpu --device cpu --out bench/results/models_2026_cpu.json

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

# M41: the current open models on BEIR beside the committed bge rows, recomputed in the run (bm25, dense@bge-small,
# hybrid@bge-small and the bge reranker at top 20 / fp32 / max_length 512): every model must load and pass its
# load-and-score check; whether the parents equal the committed file's rows (and, for FiQA, its TREC runs) is recorded.
.PHONY: bench-beir-models-2026 bench-beir-fiqa-models-2026 bench-beir-models-2026-harrier bench-beir-fiqa-models-2026-harrier
bench-beir-models-2026: ## BEIR SciFact: Qwen3-Embedding 0.6B/8B (dense, hybrid), mxbai-rerank-base/large-v2 and Qwen3-Reranker 0.6B/8B over hybrid@bge-small, each vs its bge parent (needs .[ml] and a 48 GB GPU) -> bench/results/beir_scifact_models_2026.json
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.beir --ladder models-2026 --data-dir $(BEIR_DATA_DIR) --out bench/results/beir_scifact_models_2026.json

bench-beir-fiqa-models-2026: ## BEIR FiQA: the same rows (needs .[ml] and a 48 GB GPU) -> bench/results/beir_fiqa_models_2026.json
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.beir --ladder models-2026 --dataset fiqa --data-dir $(BEIR_DATA_DIR) --out bench/results/beir_fiqa_models_2026.json

bench-beir-models-2026-harrier: ## BEIR SciFact: harrier-oss-v1-27b (dense, hybrid) vs bge-small (needs .[ml] and an 80 GB GPU) -> bench/results/beir_scifact_models_2026_harrier.json
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.beir --ladder models-2026-harrier --data-dir $(BEIR_DATA_DIR) --out bench/results/beir_scifact_models_2026_harrier.json

bench-beir-fiqa-models-2026-harrier: ## BEIR FiQA: harrier-oss-v1-27b (dense, hybrid) vs bge-small (needs .[ml] and an 80 GB GPU) -> bench/results/beir_fiqa_models_2026_harrier.json
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.beir --ladder models-2026-harrier --dataset fiqa --data-dir $(BEIR_DATA_DIR) --out bench/results/beir_fiqa_models_2026_harrier.json

# ---- M43: small rerankers for the single host, and their label-free distillation (app/eval/small_rerankers.py) ----
# The bar, the students and their training recipe were declared in that module before any row was measured. The
# synthetic queries never contain a benchmark query: generation reads documents only, and the filter step, the only one
# that reads test queries, removes their near-duplicates. DISTILL_WORK is where the uncommitted working files go.
DISTILL_WORK ?= bench/external/distill
.PHONY: small-rerankers-check distill-queries distill-queries-filter distill-teacher distill-teacher-merge
small-rerankers-check: ## load each small reranker at the served shape (top 20, max_length 512, fp32), read its dtype back and run the load-and-score check (needs .[ml]; MODELS=tag,... for a subset; CHECKPOINTS=<dir> for the students) -> stdout
	$(PYTHON) -m app.eval.small_rerankers --check $(if $(MODELS),--models $(MODELS)) $(if $(CHECKPOINTS),--checkpoint-dir $(CHECKPOINTS))

distill-queries: ## synthetic queries: one prompt per sampled document (every fixture chunk, seeded samples of SciFact and FiQA) to gemma4:12b through Ollama (OLLAMA_BASE_URL); reads documents only; resumes -> $(DISTILL_WORK)/completions.jsonl
	$(PYTHON) -m app.eval.distill_queries generate --data-dir $(BEIR_DATA_DIR) --work $(DISTILL_WORK)

distill-queries-filter: ## parse the completions, drop repeats and every near-duplicate of a fixture, SciFact or FiQA test query, split by source document (needs .[ml] for bge-small) -> bench/distill/queries.jsonl.gz, completions.jsonl.gz, queries.meta.json
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.distill_queries filter --data-dir $(BEIR_DATA_DIR) --work $(DISTILL_WORK)

SHARD ?= 0/1
distill-teacher: ## the teacher (mxbai-rerank-large-v2, fp32, max_length 512) scores each synthetic query's fused top 20 of hybrid@bge-small plus its source document; SHARD=i/n takes every n-th query; TEACHER_BATCH=n for a GPU under 16 GB; resumes (needs .[ml] and a GPU) -> $(DISTILL_WORK)/teacher.shard-i-of-n.jsonl
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.distill_teacher score --data-dir $(BEIR_DATA_DIR) --work $(DISTILL_WORK) --shard $(SHARD) $(if $(TEACHER_BATCH),--batch-size $(TEACHER_BATCH))

distill-teacher-merge: ## check that the shards cover every query once, at one clean commit -> bench/distill/teacher_scores.jsonl.gz, teacher_scores.meta.json
	$(PYTHON) -m app.eval.distill_teacher merge --work $(DISTILL_WORK)

# The rows (app/eval/small_rerankers.py): every reranker over the fused top 20 of hybrid@bge-small at max_length 512 in
# fp32, paired with the unreranked list (the bar), with bge-reranker-v2-m3 and with the teacher, all in one run; the
# students' rows appear once bench/distill/students.json is committed and need CHECKPOINTS=<dir>. Each run writes its
# per-query scores beside the results file; small-rerankers-rescore recomputes every interval from those, with no model.
.PHONY: bench-small-rerankers bench-beir-small-rerankers bench-beir-fiqa-small-rerankers small-rerankers-rescore small-rerankers-bar
bench-small-rerankers: ## fixture v1: the small rerankers on the held-out half (the bar), the tune half and all 136 questions (needs .[ml] and a GPU; CHECKPOINTS=<dir> for the students) -> bench/results/small_rerankers.json
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.small_rerankers --out bench/results/small_rerankers.json $(if $(MODELS),--models $(MODELS)) $(if $(CHECKPOINTS),--checkpoint-dir $(CHECKPOINTS))

bench-beir-small-rerankers: ## BEIR SciFact: the same rows (needs .[ml] and a GPU; CHECKPOINTS=<dir> for the students) -> bench/results/beir_scifact_small_rerankers.json
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.beir --ladder small-rerankers --data-dir $(BEIR_DATA_DIR) $(if $(CHECKPOINTS),--checkpoint-dir $(CHECKPOINTS)) --out bench/results/beir_scifact_small_rerankers.json

bench-beir-fiqa-small-rerankers: ## BEIR FiQA: the same rows (needs .[ml] and a GPU; CHECKPOINTS=<dir> for the students) -> bench/results/beir_fiqa_small_rerankers.json
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.beir --ladder small-rerankers --dataset fiqa --data-dir $(BEIR_DATA_DIR) $(if $(CHECKPOINTS),--checkpoint-dir $(CHECKPOINTS)) --out bench/results/beir_fiqa_small_rerankers.json

small-rerankers-rescore: ## recompute every interval and paired delta of the three small-rerankers results files from their per-query sidecars (CPU, no model, no dataset); exit 1 unless all equal the files
	$(PYTHON) -m app.eval.small_rerankers --rescore

small-rerankers-bar: ## print the quality bar row by row from the three committed files -> stdout
	$(PYTHON) -m app.eval.small_rerankers --bar

.PHONY: distill-train distill-manifest
distill-train: ## train one student by the declared recipe: STUDENT=student-minilm-l6|student-minilm-l12, CHECKPOINTS=<dir>; TRAIN_MICRO=n accumulates over chunks of n lists for a GPU under 16 GB (needs .[ml] and a GPU) -> $(CHECKPOINTS)/$(STUDENT)/ (not committed)
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.distill_train train --student $(STUDENT) --data-dir $(BEIR_DATA_DIR) --checkpoint-dir $(CHECKPOINTS) $(if $(TRAIN_MICRO),--micro-lists $(TRAIN_MICRO))

distill-manifest: ## the students' weights sha256, recipe, loss curve and reference logits, from their checkpoints (CHECKPOINTS=<dir>) -> bench/distill/students.json
	$(PYTHON) -m app.eval.distill_train manifest --checkpoint-dir $(CHECKPOINTS)

# ---- M51 Phase 2: the two students retrained by M43's RECIPE with only max_length 256 (attempt 2; declared in
# app/eval/small_rerankers_shapes.py PHASE2 before any M51 row ran). Reads M43's synthetic queries and teacher scores
# only: no test query, gold question or label. The checkpoints are not committed; the manifest is.
.PHONY: distill-train-ml256 distill-manifest-ml256
distill-train-ml256: ## M51 Phase 2: retrain one student at max_length 256 as attempt 2 under the tag <student>-ml256 (STUDENT=student-minilm-l6|student-minilm-l12, CHECKPOINTS=<dir>; TRAIN_MICRO=n; TRAIN_MAX_STEPS=n is a smoke run the manifest refuses; needs .[ml] and a GPU) -> $(CHECKPOINTS)/$(STUDENT)-ml256/ (not committed)
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.distill_train train --student $(STUDENT) --max-length 256 --attempt 2 --tag $(STUDENT)-ml256 --data-dir $(BEIR_DATA_DIR) --checkpoint-dir $(CHECKPOINTS) $(if $(TRAIN_MICRO),--micro-lists $(TRAIN_MICRO)) $(if $(TRAIN_MAX_STEPS),--max-steps $(TRAIN_MAX_STEPS))

distill-manifest-ml256: ## M51 Phase 2: the retrained students' weights sha256, recipe, loss curve, training device and reference logits, from their checkpoints (CHECKPOINTS=<dir>; refuses a smoke run, a dirty tree or another attempt) -> bench/distill/students_ml256.json
	$(PYTHON) -m app.eval.distill_train manifest --checkpoint-dir $(CHECKPOINTS) --students student-minilm-l6-ml256,student-minilm-l12-ml256 --attempt 2 --out bench/distill/students_ml256.json

# ---- M51: the small rerankers at cheaper shapes (app/eval/small_rerankers_shapes.py) ----
# The grid (6 models x depth 5/10/20 x max_length 256/512), the budget bands, the families and the answer levels were
# declared before any row at a new shape was measured; these two targets are CPU-only and read committed files.
.PHONY: small-rerankers-shapes-mde small-rerankers-shapes-exposure
small-rerankers-shapes-mde: ## the design-time Bonferroni half-widths (F1 k=6, F2 k=16) of the paired nDCG@10 delta, from M43's committed sidecars (CPU, seconds) -> stdout
	$(PYTHON) -m app.eval.small_rerankers_shapes --mde

small-rerankers-shapes-exposure: ## which BEIR SciFact / FiQA test queries have a relevant document among the students' synthetic source documents (CPU; needs BEIR_DATA_DIR) -> bench/results/small_rerankers_shapes_exposure.json
	$(PYTHON) -m app.eval.small_rerankers_shapes --exposure --data-dir $(BEIR_DATA_DIR)

# The rows (GPU): every model at every shape over hybrid@bge-small, the parent recomputed, the six 20/512 rows checked
# against M43's committed rows (the anchors). OUT= writes elsewhere (the replicate: *_shapes_rep.json); ROWS=anchors
# runs only the parent and the six 20/512 rows on BEIR (the smoke). CHECKPOINTS=<dir> holds the students' checkpoints.
.PHONY: small-rerankers-shapes-check small-rerankers-shapes-smoke bench-small-rerankers-shapes bench-beir-small-rerankers-shapes bench-beir-fiqa-small-rerankers-shapes
small-rerankers-shapes-check: ## load the six models at max_length 256 and 512 (fp32), check each pin, read loaded_max_length and the dtype back and run the load-and-score check (needs .[ml]; CHECKPOINTS=<dir>) -> stdout
	$(PYTHON) -m app.eval.small_rerankers_shapes --check $(if $(MODELS),--models $(MODELS)) $(if $(CHECKPOINTS),--checkpoint-dir $(CHECKPOINTS))

small-rerankers-shapes-smoke: ## fixture v1: the parent and the six 20/512 rows against M43's file (three scopes), every other shape on the tune half only with the depth and truncation invariants pass/fail; writes nothing (needs .[ml] and a GPU; CHECKPOINTS=<dir>) -> stdout
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.small_rerankers_shapes --smoke $(if $(CHECKPOINTS),--checkpoint-dir $(CHECKPOINTS))

bench-small-rerankers-shapes: ## fixture v1 at every shape: held-out half (the bar), tune half, all 136 questions (needs .[ml] and a GPU; CHECKPOINTS=<dir>; OUT=<file>) -> bench/results/small_rerankers_shapes.json
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.small_rerankers_shapes --out $(or $(OUT),bench/results/small_rerankers_shapes.json) $(if $(MODELS),--models $(MODELS)) $(if $(CHECKPOINTS),--checkpoint-dir $(CHECKPOINTS))

bench-beir-small-rerankers-shapes: ## BEIR SciFact at every shape (needs .[ml] and a GPU; CHECKPOINTS=<dir>; OUT=<file>; ROWS=anchors) -> bench/results/beir_scifact_small_rerankers_shapes.json
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.beir --ladder small-rerankers-shapes --rows $(or $(ROWS),all) --data-dir $(BEIR_DATA_DIR) $(if $(CHECKPOINTS),--checkpoint-dir $(CHECKPOINTS)) --out $(or $(OUT),bench/results/beir_scifact_small_rerankers_shapes.json)

bench-beir-fiqa-small-rerankers-shapes: ## BEIR FiQA at every shape (needs .[ml] and a GPU; CHECKPOINTS=<dir>; OUT=<file>; ROWS=anchors) -> bench/results/beir_fiqa_small_rerankers_shapes.json
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.beir --ladder small-rerankers-shapes --rows $(or $(ROWS),all) --dataset fiqa --data-dir $(BEIR_DATA_DIR) $(if $(CHECKPOINTS),--checkpoint-dir $(CHECKPOINTS)) --out $(or $(OUT),bench/results/beir_fiqa_small_rerankers_shapes.json)

# From the committed files (CPU, no model): the anchors, the frontier (quality x indicative cost per cell and the
# answer per family), every number again from the sidecars, the bar, and the replicate against the result.
.PHONY: small-rerankers-shapes-anchors small-rerankers-shapes-frontier small-rerankers-shapes-rescore small-rerankers-shapes-bar small-rerankers-shapes-replicate
small-rerankers-shapes-anchors: ## the parent and the six 20/512 rows of each shapes file against M43's committed rows (FILES=<results files>; default the three committed) -> stdout, exit 1 on a metric difference
	$(PYTHON) -m app.eval.small_rerankers_shapes --anchors $(FILES)

small-rerankers-shapes-frontier: ## per cell its band, indicative cost and deltas (95% and the family's Bonferroni level), and the F1 / F2 answers, from the committed files (CPU) -> bench/results/small_rerankers_shapes_frontier.json
	$(PYTHON) -m app.eval.small_rerankers_shapes --frontier

small-rerankers-shapes-rescore: ## recompute every interval, per-row total and invariant of the three shapes files from their sidecars, and the frontier from them (CPU, no model, no dataset); exit 1 unless all equal
	$(PYTHON) -m app.eval.small_rerankers_shapes --rescore

small-rerankers-shapes-bar: ## print the 36 cells and both family answers from the frontier file -> stdout
	$(PYTHON) -m app.eval.small_rerankers_shapes --bar

small-rerankers-shapes-replicate: ## the replicate's files (*_shapes_rep.json, the other device) against the result's, row by row (CPU) -> bench/results/small_rerankers_shapes_replicate.json
	$(PYTHON) -m app.eval.small_rerankers_shapes --replicate

# M51 Phase 2 rows: the two students retrained at 256 (attempt 2) at depth 5/10/20 x max_length 256, beside the
# attempt-1 students and their inits at the same shapes (the anchors: they must equal the committed shapes rows).
# CHECKPOINTS=<attempt-1 dir>, ML256_CHECKPOINTS=<attempt-2 dir>; the CPU targets read the committed files only.
.PHONY: small-rerankers-ml256-check bench-small-rerankers-ml256 bench-beir-small-rerankers-ml256 bench-beir-fiqa-small-rerankers-ml256 small-rerankers-ml256-anchors small-rerankers-ml256-frontier small-rerankers-ml256-bar
small-rerankers-ml256-check: ## Phase 2: load the attempt-2 students, the attempt-1 students and their inits at max_length 256, check each pin and run the load-and-score check (needs .[ml]; CHECKPOINTS=<dir>, ML256_CHECKPOINTS=<dir>) -> stdout
	$(PYTHON) -m app.eval.small_rerankers_shapes --check --phase2 $(if $(CHECKPOINTS),--checkpoint-dir $(CHECKPOINTS)) $(if $(ML256_CHECKPOINTS),--ml256-checkpoint-dir $(ML256_CHECKPOINTS))

bench-small-rerankers-ml256: ## Phase 2 on fixture v1: held-out half, tune half, all 136 questions (needs .[ml] and a GPU; CHECKPOINTS=<dir>, ML256_CHECKPOINTS=<dir>; OUT=<file>) -> bench/results/small_rerankers_ml256.json
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.small_rerankers_shapes --phase2 --out $(or $(OUT),bench/results/small_rerankers_ml256.json) $(if $(CHECKPOINTS),--checkpoint-dir $(CHECKPOINTS)) $(if $(ML256_CHECKPOINTS),--ml256-checkpoint-dir $(ML256_CHECKPOINTS))

bench-beir-small-rerankers-ml256: ## Phase 2 on BEIR SciFact (needs .[ml] and a GPU; CHECKPOINTS=<dir>, ML256_CHECKPOINTS=<dir>; OUT=<file>) -> bench/results/beir_scifact_small_rerankers_ml256.json
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.beir --ladder small-rerankers-ml256 --data-dir $(BEIR_DATA_DIR) $(if $(CHECKPOINTS),--checkpoint-dir $(CHECKPOINTS)) $(if $(ML256_CHECKPOINTS),--ml256-checkpoint-dir $(ML256_CHECKPOINTS)) --out $(or $(OUT),bench/results/beir_scifact_small_rerankers_ml256.json)

bench-beir-fiqa-small-rerankers-ml256: ## Phase 2 on BEIR FiQA (needs .[ml] and a GPU; CHECKPOINTS=<dir>, ML256_CHECKPOINTS=<dir>; OUT=<file>) -> bench/results/beir_fiqa_small_rerankers_ml256.json
	BGE_ALLOW_DOWNLOAD=1 $(PYTHON) -m app.eval.beir --ladder small-rerankers-ml256 --dataset fiqa --data-dir $(BEIR_DATA_DIR) $(if $(CHECKPOINTS),--checkpoint-dir $(CHECKPOINTS)) $(if $(ML256_CHECKPOINTS),--ml256-checkpoint-dir $(ML256_CHECKPOINTS)) --out $(or $(OUT),bench/results/beir_fiqa_small_rerankers_ml256.json)

small-rerankers-ml256-anchors: ## Phase 2: the parent, the attempt-1 students and their inits at 5/10/20 x 256 of each Phase-2 file against M51's committed shapes rows (FILES=<results files>; default the three committed) -> stdout, exit 1 on a metric difference
	$(PYTHON) -m app.eval.small_rerankers_shapes --phase2 --anchors $(FILES)

small-rerankers-ml256-frontier: ## Phase 2: per attempt-2 cell its cost (the attempt-1 row's), deltas vs no reranker (95% and F3's Bonferroni level), vs the attempt-1 student and vs the init, and the F3 answer (CPU) -> bench/results/small_rerankers_ml256_frontier.json
	$(PYTHON) -m app.eval.small_rerankers_shapes --frontier --phase2

small-rerankers-ml256-bar: ## Phase 2: print the six attempt-2 cells and the F3 answer from its frontier file -> stdout
	$(PYTHON) -m app.eval.small_rerankers_shapes --bar --phase2

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
# Untracked by default; a committed parity file is written by naming it (PARITY_OUT=bench/results/parity_shared_chunker.json).
PARITY_OUT ?= bench/results/parity_latest.json
# The in-memory row's named configuration (app/eval/parity.py IN_MEMORY_CONFIGS): CI's hash one by default; the
# model-backed one is in_memory:hybrid+bge-rerank+router+staleness@bge-small/fp16-512/top20/staleness-first.
PARITY_CONFIG ?= in_memory:hybrid+router+staleness

parity:         ## replay the gold set through a running API; report the nDCG@10 gap with its paired CI against the in-memory row PARITY_CONFIG (fails if the API is broken; refuses a fixture that differs from its MANIFEST)
	$(PYTHON) -m app.eval.harness --backend api --url $(API_URL) --out $(PARITY_OUT) --parity-config '$(PARITY_CONFIG)'

migrate:        ## the deploy step: apply pending schema migrations (PG_CONNECTION_STRING; vector table from PGVECTOR_TABLE, EMBEDDING_DIM, LEXICAL_BACKEND). Until it has run, the API answers 503 and retries.
	$(PYTHON) -m app.database.migrate

migrate-check:  ## list pending schema migrations, exit 1 if any (applies nothing)
	$(PYTHON) -m app.database.migrate --check

index:          ## the index step, after migrate and before the API: DATA_DIR into pgvector, only new and changed files embedded (PG_CONNECTION_STRING, DATA_DIR, the embedder settings); prints the report, exit 1 on failure
	$(PYTHON) -m app.engine.ingest

ingest-runs:    ## first start, unchanged restart, one edit, one delete on a copy of the fixture (PG_TEST_DSN) -> bench/results/ingest_runs.json
	$(PYTHON) -m app.engine.ingest_runs

INGEST_SCALE_LABEL ?=
ingest-scale:   ## indexing and start-up time at 1x, 10x and 100x copies of the fixture, hash embedder (PG_TEST_DSN); INGEST_SCALE_LABEL=before|after names the rows it writes, other rows are kept -> bench/results/ingest_scale.json
	$(PYTHON) -m app.engine.ingest_scale --label $(INGEST_SCALE_LABEL)

.PHONY: two-replicas
TWO_REPLICAS_OUT ?= bench/results/two_replicas.json
two-replicas:   ## two API processes on one Postgres (PG_TEST_DSN; a database of its own, dropped after): one index step, both serve retrieve, a plan proposed on one approved on the other, approve without a valid capability 403, a burnt capability refused on the other -> $(TWO_REPLICAS_OUT)
	$(PYTHON) -m app.eval.two_replicas --out $(TWO_REPLICAS_OUT)

explain-sql:    ## EXPLAIN (ANALYZE, BUFFERS) of the hybrid query against PG_TEST_DSN (fixture v1 already indexed) -> doc/sql/hybrid_explain.txt
	$(PYTHON) scripts/explain_hybrid.py

explain-sql-synthetic: ## the same over 10k generated hash-vector rows in a scratch table on PG_TEST_DSN -> doc/sql/hybrid_explain_synthetic_10k.txt
	$(PYTHON) scripts/explain_hybrid.py --synthetic-rows 10000 --out doc/sql/hybrid_explain_synthetic_10k.txt

# ---- served path at scale (GPU node + Apptainer; app/eval/served_scale.py) ----
# FiQA written one file per document and taken through the deploy order into a task-local PostgreSQL 16 +
# pgvector (Apptainer image PGVECTOR_SIF) with bge-small: the migrations, a timed build (index_directory with the
# index step's arguments), the index step itself (python -m app.engine.ingest, which must find nothing to embed),
# the API; scored through POST /api/retrieve/ and paired with the in-memory BEIR rows in the same job. SERVED_SCALE_SCRATCH: node-local disk for the
# corpus files and the database; SERVED_SCALE_STATE: where each stage's results and the database snapshot
# go (shared storage lets a resumed run, SERVED_SCALE_ARGS=--resume, continue on another node).
# BEIR_DATA_DIR (above) holds FiQA and the BEIR embedding cache.
PGVECTOR_SIF ?=
SERVED_SCALE_SCRATCH ?=
SERVED_SCALE_STATE ?=
SERVED_SCALE_ARGS ?=
bench-served-scale: ## FiQA through the served indexer, Postgres and the API vs the in-memory rows (GPU; set PGVECTOR_SIF, SERVED_SCALE_SCRATCH) -> bench/results/served_scale_fiqa.json
	@test -n "$(PGVECTOR_SIF)" || { echo "set PGVECTOR_SIF to an Apptainer image with PostgreSQL 16 and pgvector"; exit 2; }
	@test -n "$(SERVED_SCALE_SCRATCH)" || { echo "set SERVED_SCALE_SCRATCH to a node-local directory"; exit 2; }
	$(PYTHON) -m app.eval.served_scale --data-dir $(BEIR_DATA_DIR) --container $(PGVECTOR_SIF) --scratch $(SERVED_SCALE_SCRATCH) $(if $(SERVED_SCALE_STATE),--state $(SERVED_SCALE_STATE)) $(SERVED_SCALE_ARGS)

# ---- docker ----------------------------------------------------------------
docker-smoke:   ## build the compose stack, wait for /health, POST one query, tear down
	docker compose up --build -d --wait --wait-timeout 300
	$(PYTHON) scripts/smoke_retrieve.py
	docker compose down -v

# ---- delivery --------------------------------------------------------------
# CI's image job pushes the smoked api image to ghcr.io on a push to main. Merges into main wait for
# the checks branch protection requires (a GitHub setting); scripts/required_checks.py derives them
# from .github/workflows: every job that reports on every pull request into main.
.PHONY: required-checks required-checks-check required-checks-apply
required-checks: ## print the checks main's branch protection requires, derived from .github/workflows
	$(PYTHON) scripts/required_checks.py

required-checks-check: ## exit 1 unless main's protection on GitHub requires exactly those checks and auto-merge is on (gh)
	$(PYTHON) scripts/required_checks.py --check

required-checks-apply: ## make main's protection require exactly those checks, and turn on auto-merge (gh; repository admin)
	$(PYTHON) scripts/required_checks.py --apply
