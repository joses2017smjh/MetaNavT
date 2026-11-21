"""Current open embedders and rerankers (M41), measured beside the committed bge rows.

    python -m app.eval.models_2026 --check [--models qwen3-emb-0.6b,mxbai-rerank-base-v2] [--device cpu|cuda]

The models, by the tag their rows carry (the fixture rows: `make bench-models-2026`, app/eval/harness.py; the BEIR
rows: `make bench-beir-models-2026` and `make bench-beir-fiqa-models-2026`, app/eval/beir.py):

  embedders  qwen3-emb-0.6b  Qwen/Qwen3-Embedding-0.6B (float32: the one small enough to serve on a CPU)
             qwen3-emb-8b    Qwen/Qwen3-Embedding-8B (bfloat16)
             harrier-27b     microsoft/harrier-oss-v1-27b (bfloat16, about 54 GB: an 80 GB GPU, its own run and files)
  rerankers  mxbai-rerank-base-v2, mxbai-rerank-large-v2, qwen3-rerank-0.6b (float32, as the bge-reranker-v2-m3
             parent row), qwen3-rerank-8b (bfloat16)

How each is called is in the models' own files and app/retrieval: an embedder's query prompt is the one its
config_sentence_transformers.json names (QUERY_PROMPT_NAMES), documents get none, and its dtype is named
(MODEL_DTYPES); the rerankers are sentence-transformers CrossEncoders scored with .predict(pairs), the causal-LM ones
through the chat template and the yes/no (Qwen3-Reranker) or 1/0 (mxbai-rerank-v2) logit difference their repositories
configure, Qwen3-Reranker with its default "query" instruction.

A model that loads but cannot score is the failure a benchmark must not average into a row, so every row that uses one
of these models first runs load_and_score() on the instance it loaded: a question from the Qwen3 model cards with one
passage that answers it and one that does not, and for a reranker the answering passage again with text appended past
the token cap (the chat template's scoring position must survive truncation). The answer must outscore the other
passage, every score must be finite; otherwise the run stops (exit 1), and nothing is written. The check is recorded
in the row (`load_and_score`). It shows that a model can rank, not how well.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

EMBEDDERS = {  # tag: the embedder name the harness and the BEIR runner take
    "qwen3-emb-0.6b": "st:Qwen/Qwen3-Embedding-0.6B",
    "qwen3-emb-8b": "st:Qwen/Qwen3-Embedding-8B",
    "harrier-27b": "st:microsoft/harrier-oss-v1-27b",
}
RERANKERS = {  # tag: (model, precision)
    "mxbai-rerank-base-v2": ("mixedbread-ai/mxbai-rerank-base-v2", "fp32"),
    "mxbai-rerank-large-v2": ("mixedbread-ai/mxbai-rerank-large-v2", "fp32"),
    "qwen3-rerank-0.6b": ("Qwen/Qwen3-Reranker-0.6B", "fp32"),
    "qwen3-rerank-8b": ("Qwen/Qwen3-Reranker-8B", "bf16"),
}
CPU_SERVABLE = ("qwen3-emb-0.6b", "mxbai-rerank-base-v2", "qwen3-rerank-0.6b")  # small enough to serve on a CPU
HARRIER = "harrier-27b"  # needs an 80 GB GPU: its rows are a separate run and file
CHECK_MAX_LENGTH = 512  # the token cap the reranker check loads with, so its third pair runs past it

# The check's texts: the example of the Qwen3 model cards (Qwen/Qwen3-Embedding-8B, Qwen/Qwen3-Reranker-8B).
CHECK_QUERY = "What is the capital of China?"
CHECK_ANSWER = "The capital of China is Beijing."
CHECK_OTHER = ("Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is "
               "responsible for the movement of planets around the sun.")
# The answer with the other passage appended forty times: about 1,200 tokens, past a 512-token cap (the BEIR rows, the
# CPU rows, CHECK_MAX_LENGTH), where truncation must keep the chat template's scoring position.
CHECK_LONG = CHECK_ANSWER + " " + " ".join([CHECK_OTHER] * 40)


class ModelCheckFailed(RuntimeError):
    """A model loaded but did not rank the check's answer first, or returned a score that is not a number."""


def check_embedder(embedder) -> dict[str, Any]:
    """The query (through the query side, with the model's prompt) against the answering and the other passage
    (documents, no prompt): cosines, finite, the answer first. Raises ModelCheckFailed otherwise."""
    from app.retrieval.embedders import encode_queries

    q = np.asarray(encode_queries(embedder, [CHECK_QUERY]), dtype=np.float64)[0]
    docs = np.asarray(embedder.encode([CHECK_ANSWER, CHECK_OTHER]), dtype=np.float64)
    if not (np.isfinite(q).all() and np.isfinite(docs).all()):
        raise ModelCheckFailed(f"{getattr(embedder, 'name', embedder)}: an embedding is not finite")
    cos = [float(d @ q / max(np.linalg.norm(d) * np.linalg.norm(q), 1e-12)) for d in docs]
    out = {"query": CHECK_QUERY, "cosine_answer": round(cos[0], 6), "cosine_other": round(cos[1], 6), "passed": cos[0] > cos[1]}
    if not out["passed"]:
        raise ModelCheckFailed(f"{getattr(embedder, 'name', embedder)}: the answer does not outscore the other passage ({out})")
    return out


def check_reranker(model, name: str = "") -> dict[str, Any]:
    """The three pairs through the scorer every reranked row uses (app.retrieval.rerank.cross_encoder_scores): the
    answer, the other passage, and the answer with text appended (CHECK_LONG). Raises ModelCheckFailed unless every
    score is finite and both answering pairs outscore the other passage."""
    from app.retrieval.rerank import cross_encoder_scores

    scores = cross_encoder_scores(model, [(CHECK_QUERY, CHECK_ANSWER), (CHECK_QUERY, CHECK_OTHER), (CHECK_QUERY, CHECK_LONG)])
    out = {"query": CHECK_QUERY, "score_answer": scores[0], "score_other": scores[1], "score_answer_long": scores[2]}
    if not all(math.isfinite(s) for s in scores):
        raise ModelCheckFailed(f"{name}: a score is not finite ({out})")
    out["passed"] = scores[0] > scores[1] and scores[2] > scores[1]
    if not out["passed"]:
        raise ModelCheckFailed(f"{name}: an answering pair does not outscore the other passage ({out})")
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in out.items()}


def header(which: str) -> dict[str, Any]:
    """The results header's `models_2026` block: the set run, how each of its models is called, and the check."""
    from app.retrieval.embedders import ENCODE_BATCH_SIZES, MODEL_DTYPES, QUERY_PROMPT_NAMES

    tags = {"main": [t for t in [*EMBEDDERS, *RERANKERS] if t != HARRIER], "harrier": [HARRIER], "cpu": list(CPU_SERVABLE)}[which]
    embedders = {}
    for tag in [t for t in tags if t in EMBEDDERS]:
        model = EMBEDDERS[tag].split(":", 1)[1]
        embedders[tag] = {"model": model, "query_prompt_name": QUERY_PROMPT_NAMES.get(model), "document_prompt": None,
                          "dtype_requested": MODEL_DTYPES.get(model), "batch_size": ENCODE_BATCH_SIZES.get(model)}
    rerankers = {tag: {"model": RERANKERS[tag][0], "precision": RERANKERS[tag][1]} for tag in tags if tag in RERANKERS}
    return {"set": which, "embedders": embedders, "rerankers": rerankers,
            "load_and_score": ("every row that loads a model first ranks, with that instance, the answer to "
                               f"{CHECK_QUERY!r} against a passage about gravity (and, for a reranker, the answer with "
                               "that passage appended forty times); the answer must win and every score be finite, or "
                               "the run stops: it shows that a model can rank, not how well"),
            "served_defaults_changed": False}


def reproduces(root: Path, rel: str, results: list[dict], names: list[str], *, extra=None) -> dict[str, Any]:
    """Whether the rows named `names`, recomputed in this run, equal the same-named rows of the committed file `rel`:
    every field of each row's retrieval block (rounded means; by category on the fixture). `extra(row, committed)`
    adds per-row checks (the BEIR runner compares the ranking's sha256 with a committed TREC run). Recorded whatever
    the answer; a row that does not reproduce is reported, not an error."""
    path = root / rel
    if not path.exists():
        return {"file": rel, "found": False, "all": False}
    blob = json.loads(path.read_text())
    committed = {r["config"]: r for r in blob.get("results", [])}
    mine = {r["config"]: r for r in results}
    rows: dict[str, Any] = {}
    for name in names:
        if name not in mine:
            continue
        if name not in committed:
            rows[name] = {"in_committed_file": False, "equal": False}
            continue
        same = mine[name].get("retrieval") == committed[name].get("retrieval")
        entry: dict[str, Any] = {"equal": same}
        if not same:
            entry["differs"] = {k: [committed[name]["retrieval"].get(k), v] for k, v in mine[name]["retrieval"].items()
                                if committed[name]["retrieval"].get(k) != v}
        if extra is not None:
            entry.update(extra(mine[name], committed[name]))
        rows[name] = entry
    return {"file": rel, "git_sha": blob.get("git_sha"), "device": (blob.get("device") or {}).get("device_name"),
            "compared": "every field of each row's retrieval block (rounded means; by category on the fixture), "
                        "[committed, this run] where they differ",
            "rows": rows, "all": bool(rows) and all(e.get("equal") and e.get("ranking_equal", True) for e in rows.values())}


def _parse(argv: list[str] | None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="load every model named and run its load-and-score check")
    p.add_argument("--models", default=",".join([*EMBEDDERS, *RERANKERS]), help="comma-separated tags (default: every model)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Load each model, read its dtype back and run its check: one JSON line per model; exit 1 if any fails."""
    args = _parse(argv)
    if not args.check:
        print("nothing to do: pass --check", file=sys.stderr)
        return 2
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from app.eval.provenance import model_device, model_provenance
    from app.retrieval.embedders import SentenceTransformerEmbedder
    from app.retrieval.rerank import cross_encoder_provenance, get_cross_encoder

    failed = 0
    for tag in [t.strip() for t in args.models.split(",") if t.strip()]:
        line: dict[str, Any] = {"tag": tag}
        try:
            if tag in EMBEDDERS:
                name = EMBEDDERS[tag]
                line.update(model_provenance(name, "embedder"))
                e = SentenceTransformerEmbedder(name.split(":", 1)[1])
                line.update(device=model_device(e._model), dim=e.dim, **e.provenance(), check=check_embedder(e))
                del e
            elif tag in RERANKERS:
                model_name, precision = RERANKERS[tag]
                line.update(model_provenance(model_name, "reranker"), precision=precision, max_length=CHECK_MAX_LENGTH)
                m = get_cross_encoder(model_name, precision=precision, max_length=CHECK_MAX_LENGTH,
                                      device=None if args.device == "auto" else args.device, raise_errors=True)
                line.update(device=model_device(m), **cross_encoder_provenance(m), check=check_reranker(m, model_name))
                del m
            else:
                raise KeyError(f"unknown tag {tag!r}; known: {sorted([*EMBEDDERS, *RERANKERS])}")
            line["ok"] = True
        except Exception as e:  # noqa: BLE001 - reported per model, then the exit status says it failed
            line.update(ok=False, error=f"{type(e).__name__}: {e}")
            failed += 1
        free_gpu()
        print(json.dumps(line), flush=True)
    return 1 if failed else 0


def free_gpu() -> None:
    """Collect what the last model left behind and return its cached CUDA blocks, before the next model loads."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
