.PHONY: install install-app install-ml test test-eval test-unit bench bench-compare bench-gate bench-baseline bench-table bench-table-write bench-table-check bench-jury bench-frontier freeze-corpus sweeps figures matlab-demo demo-data gifs demos lock docker-smoke

PYTHON ?= python3
export PYTHONPATH := $(CURDIR):$(PYTHONPATH)
SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo nogit)

# ---- setup -----------------------------------------------------------------
install:        ## bench + eval tests only: numpy, pydantic, pytest (no LlamaIndex, no torch)
	$(PYTHON) -m pip install -e ".[eval]"

install-app:    ## + FastAPI, LlamaIndex, Postgres client (what the Docker image installs)
	$(PYTHON) -m pip install -e ".[eval,app]"

install-ml:     ## + torch, sentence-transformers, HuggingFace embeddings
	$(PYTHON) -m pip install -e ".[eval,app,ml]"

lock:           ## regenerate requirements.txt from pyproject.toml (eval + app extras)
	$(PYTHON) -m piptools compile --upgrade --extra eval --extra app --strip-extras -o requirements.txt pyproject.toml

# ---- tests -----------------------------------------------------------------
test: test-eval test-unit

test-eval:      ## LLM-free eval harness tests (needs .[eval])
	$(PYTHON) -m pytest tests/eval -q --rootdir=$(CURDIR)

test-unit:      ## legacy unit tests + engine + API contract tests (needs .[eval,app], no database)
	$(PYTHON) -m pytest tests/unit_tests tests/engine tests/api -q --rootdir=$(CURDIR)

# ---- bench -----------------------------------------------------------------
bench:          ## frozen fixture v1 -> bench/results/<git-sha>.json and latest.json
	$(PYTHON) -m app.eval.harness

bench-compare:  ## report latest.json vs the committed baseline main.json
	$(PYTHON) -m app.eval.compare

bench-gate:     ## same, but exit 1 if nDCG@10 or Recall@50 drops more than the threshold
	$(PYTHON) -m app.eval.compare --fail-on-regression

bench-baseline: ## promote latest.json to main.json (review `make bench-compare` first)
	cp bench/results/latest.json bench/results/main.json

bench-table:    ## render the README benchmark blocks from bench/results/main.json
	$(PYTHON) -m app.eval.report

bench-table-write: ## replace the generated blocks in README.md and doc/demo.html from main.json
	$(PYTHON) -m app.eval.report --write

bench-table-check: ## exit 1 if a generated block in README.md or doc/demo.html drifted from main.json
	$(PYTHON) -m app.eval.report --check

bench-jury:
	$(PYTHON) -m app.eval.harness --jury

bench-frontier:
	$(PYTHON) -m app.eval.harness --frontier

freeze-corpus:
	$(PYTHON) bench/corpus/build.py

sweeps:
	$(PYTHON) -m app.eval.sweeps

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

# ---- docker ----------------------------------------------------------------
docker-smoke:   ## build the compose stack, wait for /health, POST one query, tear down
	docker compose up --build -d --wait --wait-timeout 300
	$(PYTHON) scripts/smoke_retrieve.py
	docker compose down -v
