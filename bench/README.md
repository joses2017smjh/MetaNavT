# Frozen benchmark

Phase 0 of the v2 plan. Do not edit `files/` by hand.

```bash
make freeze-corpus   # rebuild from bench/corpus/registry.py
make bench           # retrieval + e2e metrics → bench/results/<sha>.json
make bench-compare
make sweeps          # HNSW, halfvec/binary, graph hops, GraphRAG
make figures         # doc/figures/*.svg used by the README
```

## Gold taxonomy

simple factual · conditional · comparative · aggregation · multi-hop · staleness · exact_path · semantic

Each line in `gold/questions.jsonl` has `answer` and `relevant_paths` so retrieval is measurable without an LLM.

Staleness questions must retrieve the **current** config/draft, not `configs/archive/*` or `paper/draft_v1.md`.

## Swapping in a real corpus

1. Point a new snapshot directory (your capstone experiment tree).
2. Hash it into `MANIFEST.json`.
3. Draft questions with an LLM, then **hand-correct**. Do not ship synthetic-only gold over a real corpus.


## Gold taxonomy

simple factual · conditional · comparative · aggregation · multi-hop · staleness · exact_path · semantic

Each line in `gold/questions.jsonl` has `answer` and `relevant_paths` so retrieval is measurable without an LLM.

Staleness questions must retrieve the **current** config/draft, not `configs/archive/*` or `paper/draft_v1.md`.

## Swapping in a real corpus

1. Point a new snapshot directory (your capstone experiment tree).
2. Hash it into `MANIFEST.json`.
3. Draft questions with an LLM, then **hand-correct**. Do not ship synthetic-only gold over a real corpus.
