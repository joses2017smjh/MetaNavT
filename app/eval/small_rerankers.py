"""Small rerankers for a small server (M43): what is measured, and what was declared before any of it ran.

    python -m app.eval.small_rerankers --check [--models minilm-l6,student-minilm-l6] [--device cpu|cuda]
                                       [--checkpoint-dir <dir of the students' checkpoints>]
    python -m app.eval.small_rerankers --out bench/results/small_rerankers.json [--models ...] [--checkpoint-dir ...]
    python -m app.eval.small_rerankers --rescore          # no model: every interval again from the per-query sidecars
    python -m app.eval.small_rerankers --bar              # the quality bar, row by row, from the three results files

The live demo serves hybrid retrieval without a reranker: bge-reranker-v2-m3 does not fit its host
(bench/results/api_memory_cpu.json). This module names the rerankers measured as replacements, by the tag their rows
carry, all over the fused top SHAPE["rerank_top"] of hybrid@bge-small at max_length SHAPE["max_length"] in fp32 (the
single host's RERANK_DEPTH, RERANKER_MAX_LENGTH and RERANKER_PRECISION):

  candidates  six small cross-encoders as published (CANDIDATES), scored zero-shot
  students    STUDENTS: a candidate fine-tuned on the teacher's scores for synthetic queries (label-free
              distillation: app/eval/distill_queries.py, distill_teacher.py, distill_train.py)
  references  bge-reranker-v2-m3, the app's default reranker, and the teacher, mxbai-rerank-large-v2, both recomputed
              in the same run at the same shape

Declared before any row was measured (the commit that adds this file precedes every results file), and pinned by
tests/eval/test_small_rerankers.py so that a later change has to be made on purpose:

  BAR      what "passes on quality" means, for every reranked row alike
  RECIPE   how a student is trained and how its checkpoint is chosen (no gold label, no test query)
  STUDENTS which candidates are trained

The serving bar (added p95 latency and peak memory on the host itself) is not decided by anything here.

--out runs fixture v1 (make bench-small-rerankers): hybrid@bge-small's fused top 50 chunks per question, the first 20
reranked by each model and the tail kept, as InMemoryHybridIndex.retrieve does with rerank_depth 20; rows for the
held-out half (the bar), the tune half (no choice was made on it) and all 136 questions, each reranked row paired with
the unreranked list, with the app's default reranker, with the teacher and, for a student, with the candidate it
started from. Every model passes M41's load-and-score check on the loaded instance first (app/eval/models_2026.py), a
student also the reference logits its manifest records; nothing falls back. The BEIR rows are the BEIR runner's
`small-rerankers` ladder (make bench-beir-small-rerankers, make bench-beir-fiqa-small-rerankers). All three write
their per-query scores beside the results file, and --rescore recomputes every interval and paired delta from those
on a CPU, with no model and no dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

SHAPE = {"rerank_top": 20, "max_length": 512, "precision": "fp32"}
SUFFIX = f"/{SHAPE['precision']}-{SHAPE['max_length']}/top{SHAPE['rerank_top']}"
PARENT = "hybrid@bge-small"  # the no-reranker row every reranked row is paired with
BGE_ROW = "hybrid+bge-rerank@bge-small" + SUFFIX  # the app's default reranker at the served shape (rerank_depth.json, both BEIR files)
BGE_RERANKER = "BAAI/bge-reranker-v2-m3"

# Hugging Face facts looked up on the API on 2026-11-22 (licence, parameters from the safetensors header, the commit of
# main), before any run; each row records the revision it loaded and refuses another (check_revision). Only permissive
# licences.
PERMISSIVE_LICENCES = ("apache-2.0", "mit")
CANDIDATES: dict[str, dict[str, Any]] = {
    "minilm-l6": {"model": "cross-encoder/ms-marco-MiniLM-L6-v2", "licence": "apache-2.0", "parameters": 22_714_113,
                  "revision": "233902d25c440f23af6f7d6e94d2946bac0bee0a"},
    "minilm-l12": {"model": "cross-encoder/ms-marco-MiniLM-L12-v2", "licence": "apache-2.0", "parameters": 33_360_897,
                   "revision": "7b0235231ca2674cb8ca8f022859a6eba2b1c968"},
    "mxbai-xsmall-v1": {"model": "mixedbread-ai/mxbai-rerank-xsmall-v1", "licence": "apache-2.0", "parameters": 70_830_337,
                        "revision": "b5c6e9da73abc3711f593f705371cdbe9e0fe422"},
    "mxbai-base-v1": {"model": "mixedbread-ai/mxbai-rerank-base-v1", "licence": "apache-2.0", "parameters": 184_422_913,
                      "revision": "800f24c113213a187e65bde9db00c15a2bb12738"},
    "gte-modernbert-base": {"model": "Alibaba-NLP/gte-reranker-modernbert-base", "licence": "apache-2.0", "parameters": 149_605_633,
                            "revision": "f7481e6055501a30fb19d090657df9ec1f79ab2c"},
    "bge-reranker-base": {"model": "BAAI/bge-reranker-base", "licence": "mit", "parameters": 278_044_931,
                          "revision": "2cfc18c9415c912f9d8155881c133215df768a70"},
}
# The teacher: the one reranker M41 measured above bge-reranker-v2-m3 on fixture v1, SciFact and FiQA alike.
TEACHER_TAG, TEACHER_MODEL = "mxbai-rerank-large-v2", "mixedbread-ai/mxbai-rerank-large-v2"
# The students, by the candidate each starts from: the two smallest, the ones most likely to meet the serving bar.
STUDENTS = {"student-minilm-l6": "minilm-l6", "student-minilm-l12": "minilm-l12"}

BAR = {
    "name": "December's quality bar",
    "passes": ("the row's paired nDCG@10 delta against hybrid@bge-small has a 95% interval whose lower end is above zero on "
               "fixture v1's held-out questions AND on BEIR SciFact AND on BEIR FiQA"),
    "applies_to": "every reranked row: the candidates, the students, bge-reranker-v2-m3 and the teacher",
    "interval": "percentile bootstrap over the queries, paired, 2000 resamples, seed 0 (app/eval/stats.py)",
    "held_out": ("fixture v1: the test half of app.eval.rerank_arms.split_questions (alternate inside each category, gold "
                 "order), as make bench-rrf-sweep uses it; no choice is made on the other (tune) half; BEIR: the test split"),
    "also_reported": "each row's delta against bge-reranker-v2-m3 and against the teacher; a student's against the candidate it started from",
    "not_decided_here": ("the serving bar: added p95 latency and peak memory at the served depth, measured on the host "
                         "itself; a CPU timing from another machine is indicative only"),
}

# How a student is trained (app/eval/distill_train.py reads these; nothing else sets them). Frozen before the zero-shot
# rows were measured, because those numbers arrive before training does. A student retrained after any held-out or BEIR
# number was seen is recorded as a later attempt (ATTEMPT), with what changed, and the earlier attempt's rows stay.
ATTEMPT = 1
RECIPE: dict[str, Any] = {
    "teacher": TEACHER_MODEL,
    "targets": "the teacher's raw logits (fp32, max_length 512), one per candidate",
    "list": ("the fused top 20 of hybrid@bge-small for a synthetic query (the candidates a served reranker sees), plus the "
             "query's source document when it is not among them"),
    "loss": ("MarginMSE over every pair of candidates of one query's list: the mean over i < j of "
             "((s_i - s_j) - (t_i - t_j))^2, s the student's logit and t the teacher's"),
    "max_length": 512,
    "lists_per_step": 8,
    "epochs": 4,
    "steps_per_epoch": "ceil(training lists / lists_per_step)",
    "optimizer": "AdamW, betas (0.9, 0.999)",
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "schedule": "linear warm-up over the first 10% of the steps, then linear decay to zero",
    "warmup_fraction": 0.1,
    "max_grad_norm": 1.0,
    "precision": "float32 weights and float32 arithmetic (no mixed precision: the targets are logit differences)",
    "seed": 0,
    "corpus_mixing": ("each list of a step comes from a corpus drawn with probability proportional to the square root of "
                      "that corpus's number of training lists; inside a corpus lists are drawn without replacement and "
                      "reshuffled when exhausted"),
    "validation": ("the synthetic queries of 5% of each corpus's source documents (at least 4 documents), held out by "
                   "document before training; never a gold question, a tune-half question or a BEIR query"),
    "selection": ("the epoch end with the lowest validation loss: the same MarginMSE, averaged per corpus and then over "
                  "the three corpora; ties go to the earlier epoch"),
}


class CheckpointError(RuntimeError):
    """A student's checkpoint is missing, or its weights are not the ones its manifest names."""


def row_name(tag: str) -> str:
    """The results row of a reranker tag at the served shape, over hybrid@bge-small."""
    return BGE_ROW if tag == "bge-rerank" else f"hybrid+{tag}@bge-small{SUFFIX}"


def reranker_tags() -> list[str]:
    """Every reranked row's tag, in table order: the app's default, the teacher, the candidates, the students."""
    return ["bge-rerank", TEACHER_TAG, *CANDIDATES, *STUDENTS]


def model_of(tag: str) -> str:
    """The Hugging Face id behind a tag (a student has none: its weights are a local checkpoint)."""
    if tag == "bge-rerank":
        return BGE_RERANKER
    if tag == TEACHER_TAG:
        return TEACHER_MODEL
    return CANDIDATES[tag]["model"]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


STUDENT_MANIFEST = "bench/distill/students.json"  # committed: each student's weights sha256, recipe and training run
WEIGHTS_FILE = "model.safetensors"


def student_checkpoint(tag: str, checkpoint_dir: Path | str | None, manifest: dict | None = None) -> tuple[Path, dict]:
    """(the directory of student `tag`'s checkpoint, its manifest entry), after checking that the weights file there has
    the sha256 the committed manifest records. The checkpoints are not in the repository: `checkpoint_dir` (or
    $SMALL_RERANKER_CHECKPOINTS) says where they are, and a results row records the sha256, never the path."""
    if tag not in STUDENTS:
        raise KeyError(f"unknown student {tag!r}; known: {sorted(STUDENTS)}")
    base = checkpoint_dir or os.environ.get("SMALL_RERANKER_CHECKPOINTS")
    if not base:
        raise CheckpointError(f"{tag}: no checkpoint directory (pass --checkpoint-dir or set SMALL_RERANKER_CHECKPOINTS)")
    if manifest is None:
        path = Path(__file__).resolve().parents[2] / STUDENT_MANIFEST
        if not path.exists():
            raise CheckpointError(f"{STUDENT_MANIFEST} is missing: no student has been trained and recorded")
        manifest = json.loads(path.read_text())
    entry = (manifest.get("students") or {}).get(tag)
    if not entry:
        raise CheckpointError(f"{tag} is not in {STUDENT_MANIFEST}")
    directory = Path(base) / entry["directory"]
    weights = directory / WEIGHTS_FILE
    if not weights.exists():
        raise CheckpointError(f"{tag}: {entry['directory']}/{WEIGHTS_FILE} is not in the checkpoint directory")
    found = sha256_of(weights)
    if found != entry["weights_sha256"]:
        raise CheckpointError(f"{tag}: {WEIGHTS_FILE} has sha256 {found}, the manifest records {entry['weights_sha256']}")
    return directory, entry


def reranker_source(tag: str, checkpoint_dir: Path | str | None = None) -> tuple[str, dict[str, Any]]:
    """(what get_cross_encoder loads for `tag`, what a row records about where it came from): a hub id with its pinned
    revision, or a student's verified local checkpoint (named by its sha256 and its init, never by its path)."""
    if tag in STUDENTS:
        directory, entry = student_checkpoint(tag, checkpoint_dir)
        return str(directory), {"model": tag, "student_of": CANDIDATES[STUDENTS[tag]]["model"], "weights_sha256": entry["weights_sha256"],
                                "attempt": entry.get("attempt"), "manifest": STUDENT_MANIFEST, "revision": None, "neural": True}
    model = model_of(tag)
    pinned = CANDIDATES.get(tag, {}).get("revision")
    return model, {"model": model, "pinned_revision": pinned}


def check_revision(tag: str, revision: str | None) -> None:
    """A candidate's cached snapshot must be the pinned commit: a moved main branch is another model."""
    pinned = CANDIDATES.get(tag, {}).get("revision")
    if pinned and revision != pinned:
        raise RuntimeError(f"{tag}: the cached revision is {revision}, the pinned one is {pinned}")


def tag_of(name: str) -> str:
    """The tag of a reranker named by its tag or by its Hugging Face id."""
    if name in reranker_tags():
        return name
    for tag in ("bge-rerank", TEACHER_TAG, *CANDIDATES):
        if model_of(tag) == name:
            return tag
    raise KeyError(f"{name!r} is not one of the small-rerankers models: {reranker_tags()}")


def resolver(checkpoint_dir: Path | str | None = None, root: Path | None = None):
    """name (a tag or a hub id) -> (what get_cross_encoder loads, what the row records, a check of the loaded model or
    None): the hook the BEIR runner's engine and the fixture run share. A candidate whose cached revision is not the
    pinned one raises; a student resolves to its verified checkpoint and, once loaded, must give the reference logits
    its manifest records (app.eval.distill_train.check_reference)."""
    root = root or Path(__file__).resolve().parents[2]

    def resolve(name: str):
        tag = tag_of(name)
        source, record = reranker_source(tag, checkpoint_dir)
        if tag in STUDENTS:
            _, entry = student_checkpoint(tag, checkpoint_dir)

            def after_load(model):
                from app.eval.distill_train import check_reference

                return {"reference_check": check_reference(model, entry, root)}

            return source, record, after_load
        from app.eval.provenance import hf_revision

        check_revision(tag, hf_revision(source))
        return source, ({"pinned_revision": record["pinned_revision"]} if record.get("pinned_revision") else {}), None

    return resolve


def header(row_names: list[str] | None = None) -> dict[str, Any]:
    """The results header's `small_rerankers` block: the shape, the bar and what was declared, as the run had them."""
    return {"shape": SHAPE, "parent": PARENT, "bar": BAR,
            "references": {"bge-rerank": BGE_RERANKER, TEACHER_TAG: TEACHER_MODEL},
            "candidates": CANDIDATES, "students": {tag: {"init": init, "attempt": ATTEMPT} for tag, init in STUDENTS.items()},
            "rows": list(row_names) if row_names is not None else None,
            "load_and_score": "every model first ranks the check's answer above an unrelated passage with the loaded instance, or the run stops (app/eval/models_2026.py)",
            "served_defaults_changed": False}


# ---------------------------------------------------------------- fixture v1 (make bench-small-rerankers)

RESULTS = {"fixture": "bench/results/small_rerankers.json", "scifact": "bench/results/beir_scifact_small_rerankers.json",
           "fiqa": "bench/results/beir_fiqa_small_rerankers.json"}
SCOPES = ("held_out", "tune", "all_questions")
SIDECAR_SCHEMA = "small-rerankers-per-question-v1"
# The committed rows the fixture run's recomputed parents stand for, on all 136 questions.
FIXTURE_REPRODUCE = {"bench/results/neural.json": (PARENT,), "bench/results/rerank_depth.json": (BGE_ROW,)}


def comparisons(rows: list[str]) -> list[tuple[str, str]]:
    """(row, other) for every paired delta of a run: each reranked row against the unreranked parent (the bar), the
    app's default reranker and the teacher, and a student against the candidate it started from."""
    teacher = row_name(TEACHER_TAG)
    out = [(n, ref) for n in rows for ref in (PARENT, BGE_ROW, teacher) if n not in (PARENT, ref) and ref in rows]
    out += [(row_name(s), row_name(init)) for s, init in STUDENTS.items() if row_name(s) in rows and row_name(init) in rows]
    return out


def scope_rows(scores: dict[str, dict[str, list[float]]], rows: list[str], subset: list[int], pairs: list[tuple[str, str]],
               metrics: tuple[str, ...]) -> tuple[list[dict[str, Any]], dict]:
    """One scope's rows from per-question scores: each row's mean and interval per metric over the questions `subset`
    (indices in gold order), and its paired deltas; with the bootstrap settings. The run and --rescore both come here,
    so a results file and its sidecar cannot disagree about how an interval is made."""
    from app.eval.stats import DEFAULT_N_BOOT, DEFAULT_SEED, mean_ci, paired_delta_ci, resample_indices, settings

    idx = resample_indices(len(subset), n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED)
    column = lambda name, m: [scores[name][m][i] for i in subset]  # noqa: E731
    out = []
    for name in rows:
        ci = {}
        for m in metrics:
            mean, lo, hi = mean_ci(column(name, m), idx)
            ci[m] = {"mean": round(mean, 4), "lo": round(lo, 4), "hi": round(hi, 4)}
        deltas = []
        for a, other in pairs:
            if a != name:
                continue
            block: dict[str, Any] = {"reference": other}
            for m in metrics:
                d, lo, hi = paired_delta_ci(column(name, m), column(other, m), idx)
                block[m] = {"delta": round(d, 4), "lo": round(lo, 4), "hi": round(hi, 4)}
            deltas.append(block)
        out.append({"config": name, "n": len(subset), "ci": ci, **({"deltas": deltas} if deltas else {})})
    return out, settings(len(subset))


def _ms(samples: list[float]) -> dict[str, float]:
    import numpy as np

    return {"p50_ms": round(float(np.percentile(samples, 50)), 3), "p95_ms": round(float(np.percentile(samples, 95)), 3), "n": len(samples)}


def run_fixture(root: Path | None = None, *, tags: list[str] | None = None, checkpoint_dir: Path | str | None = None, device: str = "auto",
                retrieve_k: int = 50, load=None, embedder: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """(the results blob, its per-question sidecar) for fixture v1. `load` (tests) replaces the model loader:
    tag -> (model, record); `embedder` replaces bge-small."""
    import time
    from datetime import datetime, timezone

    from app.eval.corpus import load_manifest, verify_manifest
    from app.eval.gold import load_gold
    from app.eval.harness import BGE_SMALL, CI_METRICS
    from app.eval.index_loader import build_index, mtime_provenance
    from app.eval.metrics import aggregate_retrieval
    from app.eval.models_2026 import check_embedder, check_reranker, free_gpu, reproduces
    from app.eval.provenance import command_line, device_info, git_sha, git_state, model_device, model_provenance, run_provenance, sha256_file
    from app.eval.rerank_arms import _paths, split_questions
    from app.retrieval.rerank import CrossEncoderReranker, cross_encoder_provenance, get_cross_encoder

    root = root or Path(__file__).resolve().parents[2]
    started = git_state(root)  # before anything is written (run_provenance)
    files_root = root / "bench" / "corpus" / "files"
    gold_path = root / "bench" / "gold" / "questions.jsonl"
    gold = load_gold(gold_path)
    manifest = load_manifest(root / "bench" / "corpus" / "MANIFEST.json")
    mismatches = verify_manifest(files_root, manifest)
    if mismatches:
        raise RuntimeError("corpus drift:\n" + "\n".join(mismatches))
    n_files = manifest["n_files"]
    embedder = embedder or BGE_SMALL
    index = build_index(files_root, embedder_name=embedder, retrieve_k=retrieve_k, enable_router=False, enable_rerank=False, mtimes="manifest")
    embedder_check = check_embedder(index.embedder) if embedder.startswith("st:") else None
    fused = [[(h.chunk, h.score) for h in index.retrieve(q.question, k=retrieve_k).hits] for q in gold]  # fused order, RRF score
    orders: dict[str, list[list[str]]] = {PARENT: [_paths(pairs) for pairs in fused]}
    tags = list(tags) if tags is not None else [t for t in reranker_tags() if t not in STUDENTS or (root / STUDENT_MANIFEST).exists()]
    resolve = resolver(checkpoint_dir, root)
    top = SHAPE["rerank_top"]
    rerankers: dict[str, dict[str, Any]] = {}
    for tag in tags:
        if load is not None:
            model, record = load(tag)
        else:
            source, record, after_load = resolve(tag)
            model = get_cross_encoder(source, precision=SHAPE["precision"], max_length=SHAPE["max_length"],
                                      device=None if device == "auto" else device, raise_errors=True)
            record = {**model_provenance(model_of(tag) if tag not in STUDENTS else tag, "reranker"), **record,
                      **(after_load(model) if after_load else {})}
        info = {**record, "loaded": True, "device": model_device(model), "precision": SHAPE["precision"], "max_length": SHAPE["max_length"],
                "depth": top, **cross_encoder_provenance(model), "load_and_score": check_reranker(model, tag)}
        rerank = CrossEncoderReranker(model=model, model_name=tag, precision=SHAPE["precision"], max_length=SHAPE["max_length"])
        if gold:
            rerank(gold[0].question, fused[0][:top])  # warm-up: CUDA kernels; scoring is deterministic
        name, latency = row_name(tag), []
        orders[name] = []
        for q, pairs in zip(gold, fused):
            t0 = time.perf_counter()
            ranked = list(rerank(q.question, pairs[:top])) + pairs[top:]  # the tail keeps the fused order
            latency.append((time.perf_counter() - t0) * 1000.0)
            orders[name].append(_paths(ranked))
        info["rerank_latency"] = {**_ms(latency), "note": "per question on this run's device, after one warm-up question: not the server"}
        info["n_top10_differs_from_parent"] = sum(1 for a, b in zip(orders[name], orders[PARENT]) if a[:10] != b[:10])
        rerankers[tag] = info
        del model, rerank
        free_gpu()

    def rows_of(ids):
        return [{"id": gold[i].id, "category": gold[i].category, "relevant": gold[i].relevant_ids()} for i in ids]

    every = list(range(len(gold)))
    tune_ids, held_ids = split_questions(gold)
    scored = {name: aggregate_retrieval([{**r, "retrieved": lists[i]} for i, r in zip(every, rows_of(every))], n_files=n_files) for name, lists in orders.items()}
    scores = {name: {m: [float(q[m]) for q in s.per_query] for m in CI_METRICS} for name, s in scored.items()}
    names = list(orders)
    pairs = comparisons(names)
    blob: dict[str, Any] = {
        "git_sha": git_sha(root), **run_provenance(root=root, started=started), "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command_line(), "device": device_info(device), "corpus_sha256": manifest["aggregate_sha256"], "gold_sha256": sha256_file(gold_path),
        "n_gold": len(gold), "n_files": n_files, "mtimes": mtime_provenance(files_root, "manifest", base=root), "n_chunks": len(index.chunks),
        "small_rerankers": header(names),
        "first_stage": {"config": PARENT, "retrieve_k": retrieve_k, "reranked": f"the fused top {top} chunks; the rest keep the fused order",
                        "embedder": {**model_provenance(embedder, "embedder"), "device": model_device(getattr(index.embedder, "_model", None)),
                                     **(index.embedder.provenance() if hasattr(index.embedder, "provenance") else {}),
                                     **({"load_and_score": embedder_check} if embedder_check else {})}},
        "rerankers": rerankers,
        "split": {"rule": "alternate train / test inside each category, gold order (app.eval.rerank_arms.split_questions)", "tune": "train",
                  "held_out": "test", "tune_ids": [gold[i].id for i in tune_ids], "held_out_ids": [gold[i].id for i in held_ids],
                  "use": "the bar is read on the held-out half; no choice was made on the tune half; no gold question is a training query"},
    }
    for scope, ids in zip(SCOPES, (held_ids, tune_ids, every)):
        rows, boot = scope_rows(scores, names, ids, pairs, CI_METRICS)
        for row in rows:
            subset = aggregate_retrieval([{**r, "retrieved": orders[row["config"]][i]} for i, r in zip(ids, rows_of(ids))], n_files=n_files)
            row["retrieval"] = subset.as_dict()
        blob[scope] = {"bootstrap": boot, "results": rows}
    blob["results"] = blob["all_questions"]["results"]  # the rows a committed file's are compared with, by the same name
    blob["reproduces"] = [reproduces(root, rel, blob["results"], [n for n in rows if n in orders]) for rel, rows in FIXTURE_REPRODUCE.items()
                          if any(n in orders for n in rows)]
    blob["reproduces_all"] = all(block["all"] for block in blob["reproduces"])
    blob["not_reproduced_anywhere"] = (f"{row_name(TEACHER_TAG)}: no committed row has the teacher at this shape on the fixture "
                                       "(models_2026.json reranks every fused candidate at the model's own token cap)")
    del blob["results"]
    sidecar = {"schema": SIDECAR_SCHEMA,
               "about": "per-question scores of the results file with the same name (.json), in gold order and unrounded: "
                        "`make small-rerankers-rescore` recomputes every interval and paired delta of its three scopes from them",
               "git_sha": blob["git_sha"], "timestamp": blob["timestamp"], "metrics": list(CI_METRICS), "ids": [q.id for q in gold],
               "categories": [q.category for q in gold], "split": {"tune": tune_ids, "held_out": held_ids}, "scores": scores}
    return blob, sidecar


# ---------------------------------------------------------------- every interval again, with no model (make small-rerankers-rescore)


def rescore_fixture(blob: dict, sidecar: dict) -> list[str]:
    """Every interval, paired delta and mean of the fixture file's three scopes that its sidecar does not reproduce."""
    from app.eval.compare import _leaf_differences

    out: list[str] = []
    for key in ("git_sha", "timestamp"):
        if sidecar.get(key) != blob.get(key):
            return [f"{key}: results file {blob.get(key)!r}, sidecar {sidecar.get(key)!r} (not the same run)"]
    n = len(sidecar["ids"])
    subsets = {"held_out": sidecar["split"]["held_out"], "tune": sidecar["split"]["tune"], "all_questions": list(range(n))}
    metrics = tuple(sidecar["metrics"])
    for scope in SCOPES:
        stored = blob[scope]["results"]
        names = [r["config"] for r in stored]
        pairs = [(r["config"], d["reference"]) for r in stored for d in r.get("deltas", [])]
        again, boot = scope_rows(sidecar["scores"], names, subsets[scope], pairs, metrics)
        _leaf_differences(f"{scope}.bootstrap", blob[scope]["bootstrap"], boot, out)
        for a, b in zip(stored, again):
            for key in ("n", "ci", "deltas"):
                _leaf_differences(f"{scope}.{a['config']}.{key}", a.get(key), b.get(key), out)
            for m in metrics:
                column = [sidecar["scores"][a["config"]][m][i] for i in subsets[scope]]
                mean = round(sum(column) / len(column), 4)
                if a["retrieval"].get(m) != mean:
                    out.append(f"{scope}.{a['config']}.retrieval.{m}: results file {a['retrieval'].get(m)!r}, mean of the sidecar {mean!r}")
    return out


def rescore_beir(blob: dict, sidecar: dict) -> list[str]:
    """Every interval, parent delta, comparison and mean of a BEIR small-rerankers file that its sidecar does not
    reproduce (the BEIR runner's own attach_confidence over the sidecar's scores)."""
    from app.eval import beir
    from app.eval.compare import _leaf_differences

    out: list[str] = []
    for key in ("git_sha", "timestamp"):
        if sidecar.get(key) != blob.get(key):
            return [f"{key}: results file {blob.get(key)!r}, sidecar {sidecar.get(key)!r} (not the same run)"]
    metrics = sidecar["metrics"]
    rows = []
    for stored in blob["results"]:
        columns = sidecar["scores"][stored["config"]]
        rows.append({"config": stored["config"], "parent": stored.get("parent"), "retrieval": stored["retrieval"],
                     "_per_query": [dict(zip(metrics, values), id=qid) for qid, *values in zip(sidecar["ids"], *(columns[m] for m in metrics))]})
    boot = blob["bootstrap"]
    again = beir.attach_confidence(rows, len(sidecar["ids"]), seed=boot["seed"], n_boot=boot["n_boot"], published=None,
                                   comparisons=[(c["row"], c["other"]) for c in boot.get("comparisons") or []])
    _leaf_differences("bootstrap.comparisons", boot.get("comparisons"), again.get("comparisons"), out)
    for stored, row in zip(blob["results"], rows):
        for key in ("ci", "delta_vs_parent"):
            _leaf_differences(f"{stored['config']}.{key}", stored.get(key), row.get(key), out)
        for m in metrics:
            column = sidecar["scores"][stored["config"]][m]
            mean = round(sum(column) / len(column), 4)
            if stored["retrieval"].get(m) != mean:
                out.append(f"{stored['config']}.retrieval.{m}: results file {stored['retrieval'].get(m)!r}, mean of the sidecar {mean!r}")
    return out


def rescore(root: Path | None = None) -> tuple[int, list[str]]:
    """(files checked, differences) over the committed small-rerankers results files that exist."""
    from app.eval import per_question

    root = root or Path(__file__).resolve().parents[2]
    checked, problems = 0, []
    for dataset, rel in RESULTS.items():
        path = root / rel
        if not path.exists():
            continue
        side = per_question.sidecar_path(path)
        if not side.exists():
            problems.append(f"{rel}: no sidecar {side.name}")
            continue
        blob, sidecar = json.loads(path.read_text()), json.loads(side.read_text())
        diffs = rescore_fixture(blob, sidecar) if dataset == "fixture" else rescore_beir(blob, sidecar)
        problems += [f"{rel}: {d}" for d in diffs]
        checked += 1
    return checked, problems


# ---------------------------------------------------------------- the bar, from the three files


def _delta_to(row: dict, reference: str, bootstrap: dict | None = None) -> dict | None:
    """A row's paired nDCG@10 delta against `reference`, wherever the file keeps it: the fixture rows' deltas, a BEIR
    row's delta_vs_parent, or the run's comparisons."""
    for block in row.get("deltas") or []:
        if block["reference"] == reference:
            return block["ndcg@10"]
    parent = row.get("delta_vs_parent")
    if parent and parent.get("reference") == reference:
        return parent["ndcg@10"]
    for c in (bootstrap or {}).get("comparisons") or []:
        if c["row"] == row["config"] and c["other"] == reference:
            return c["ndcg@10"]
    return None


def bar_table(root: Path | None = None) -> dict[str, Any]:
    """The quality bar read off the three committed files: per reranked row and dataset, nDCG@10 and its paired deltas
    against the unreranked parent, the app's default reranker, the teacher and (a student) its init; whether the parent
    delta's interval is above zero there; and whether the row passes, which needs all three datasets. A dataset whose
    file is missing, or a row a file does not have, leaves the verdict open (None), never passed."""
    root = root or Path(__file__).resolve().parents[2]
    blobs = {d: json.loads((root / rel).read_text()) for d, rel in RESULTS.items() if (root / rel).exists()}
    teacher = row_name(TEACHER_TAG)
    table: dict[str, Any] = {"datasets": {}, "rows": []}
    scopes = {}
    for dataset, blob in blobs.items():
        scope = blob["held_out"] if dataset == "fixture" else blob
        scopes[dataset] = ({r["config"]: r for r in scope["results"]}, scope.get("bootstrap"))
        parent = scopes[dataset][0].get(PARENT)
        table["datasets"][dataset] = {"file": RESULTS[dataset], "git_sha": blob.get("git_sha"), "n": blob["held_out"]["bootstrap"]["n_queries"] if dataset == "fixture" else blob["n_queries"],
                                      "device": (blob.get("device") or {}).get("device_name"),
                                      "parent_ndcg@10": (parent or {}).get("ci", {}).get("ndcg@10")}
    for tag in reranker_tags():
        name = row_name(tag)
        entry: dict[str, Any] = {"tag": tag, "config": name, "datasets": {}}
        above = []
        for dataset in RESULTS:
            rows, boot = scopes.get(dataset, ({}, None))
            row = rows.get(name)
            if row is None:
                above.append(None)
                continue
            vs_parent = _delta_to(row, PARENT, boot)
            cell = {"ndcg@10": row["ci"]["ndcg@10"], "vs_parent": vs_parent, "vs_bge": _delta_to(row, BGE_ROW, boot), "vs_teacher": _delta_to(row, teacher, boot),
                    "above_zero": bool(vs_parent and vs_parent["lo"] > 0)}
            if tag in STUDENTS:
                cell["vs_init"] = _delta_to(row, row_name(STUDENTS[tag]), boot)
            entry["datasets"][dataset] = cell
            above.append(cell["above_zero"])
        entry["passes"] = False if any(a is False for a in above) else (True if all(above) else None)
        if entry["datasets"]:
            table["rows"].append(entry)
    return table


def _parse(argv: list[str] | None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="load every model named and run its load-and-score check")
    p.add_argument("--out", type=Path, default=None, help="run fixture v1 and write this results file with its per-question sidecar")
    p.add_argument("--rescore", action="store_true", help="recompute every interval of the committed small-rerankers files from their sidecars (no model)")
    p.add_argument("--bar", action="store_true", help="print the quality bar, row by row, from the committed files")
    p.add_argument("--models", default=None, help="comma-separated tags (default: the references and the candidates; the students too once trained)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--checkpoint-dir", type=Path, default=None, help="where the students' checkpoints are (default: $SMALL_RERANKER_CHECKPOINTS)")
    return p.parse_args(argv)


def _fmt(block: dict | None) -> str:
    return f"{block['delta']:+.4f} [{block['lo']:+.4f}, {block['hi']:+.4f}]" if block else "-"


def main(argv: list[str] | None = None) -> int:
    """--check: load each reranker at the served shape, read its dtype back and run M41's load-and-score check, one JSON
    line per model, exit 1 if any fails. --out: the fixture benchmark. --rescore, --bar: from the committed files."""
    args = _parse(argv)
    root = Path(__file__).resolve().parents[2]
    if args.rescore:
        checked, problems = rescore(root)
        for line in problems[:60]:
            print(f"  - {line}")
        print(f"small-rerankers intervals: {'OK' if not problems else f'{len(problems)} differences'}, {checked} results files recomputed from their per-query sidecars")
        return 1 if problems or not checked else 0
    if args.bar:
        table = bar_table(root)
        for row in table["rows"]:
            cells = "; ".join(f"{d} {_fmt(c['vs_parent'])}" for d, c in row["datasets"].items())
            print(f"{'PASS' if row['passes'] else ('open' if row['passes'] is None else 'no  ')}  {row['tag']:22s} {cells}")
        return 0
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.out is not None:
        from app.eval import per_question

        tags = [t.strip() for t in args.models.split(",") if t.strip()] if args.models else None
        blob, sidecar = run_fixture(root, tags=tags, checkpoint_dir=args.checkpoint_dir, device=args.device)
        out = args.out if args.out.is_absolute() else root / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(blob, indent=2) + "\n")
        per_question.write(per_question.sidecar_path(out), sidecar)
        for row in blob["held_out"]["results"]:
            print("held_out", row["config"], row["ci"]["ndcg@10"], [(d["reference"], d["ndcg@10"]) for d in row.get("deltas", [])])
        print("reproduces", json.dumps(blob["reproduces"]))
        print(f"wrote {out}")
        return 0
    if not args.check:
        print("nothing to do: pass --check, --out, --rescore or --bar", file=sys.stderr)
        return 2
    from app.eval.models_2026 import check_reranker, free_gpu
    from app.eval.provenance import hf_revision, model_device
    from app.retrieval.rerank import cross_encoder_provenance, get_cross_encoder

    manifest = Path(__file__).resolve().parents[2] / STUDENT_MANIFEST
    default = ["bge-rerank", TEACHER_TAG, *CANDIDATES, *(STUDENTS if manifest.exists() else [])]
    failed = 0
    for tag in [t.strip() for t in (args.models.split(",") if args.models else default) if t.strip()]:
        line: dict[str, Any] = {"tag": tag}
        try:
            source, where = reranker_source(tag, args.checkpoint_dir)
            line.update(where, precision=SHAPE["precision"], max_length=SHAPE["max_length"])
            if tag not in STUDENTS:
                line["revision"] = hf_revision(source)
                check_revision(tag, line["revision"])
            m = get_cross_encoder(source, precision=SHAPE["precision"], max_length=SHAPE["max_length"],
                                  device=None if args.device == "auto" else args.device, raise_errors=True)
            line.update(device=model_device(m), **cross_encoder_provenance(m), check=check_reranker(m, tag))
            del m
            line["ok"] = True
        except Exception as e:  # noqa: BLE001 - reported per model, then the exit status says it failed
            from app.eval.provenance import scrub

            line.update(ok=False, error=scrub(f"{type(e).__name__}: {e}"))
            failed += 1
        free_gpu()
        print(json.dumps(line), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
