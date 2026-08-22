.PHONY: bench bench-compare freeze-corpus test-eval sweeps figures gifs

PYTHON ?= python3
export PYTHONPATH := $(CURDIR):$(CURDIR)/.vendor:$(PYTHONPATH)
SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo nogit)

bench:
	$(PYTHON) -m app.eval.harness

bench-compare:
	$(PYTHON) -m app.eval.compare

freeze-corpus:
	$(PYTHON) bench/corpus/build.py

test-eval:
	$(PYTHON) -m pytest tests/eval -q --rootdir=$(CURDIR)

sweeps:
	$(PYTHON) -m app.eval.sweeps

figures:
	$(PYTHON) -m app.eval.figures

gifs:
	$(PYTHON) -m app.eval.gifs


