"""The small embedder adapted to the three test corpora on LLM-written synthetic queries (M48): what is measured, and
what was declared before any of it ran.

The live demo serves BAAI/bge-small-en-v1.5 as published (deploy/single-host/compose.yml: EMBEDDING_MODEL). M43 left
the material of GPL-style unsupervised domain adaptation in bench/distill/: 10,622 synthetic queries written by
gemma4:12b from documents of the fixture, SciFact and FiQA, each with the fused top 20 of hybrid@bge-small (plus its
source document when absent) scored by the cross-encoder mxbai-rerank-large-v2. This module names the embedders
fine-tuned on it (ARMS), how they are trained (RECIPE), and what "the adapted embedder is better" means (BAR).

Declared before any run (the commit that adds this file precedes every M48 job and every results file), and pinned by
tests/eval/test_adapted_embedders.py so that a later change has to be made on purpose:

  BAR      one pre-declared row, hybrid@bge-small-marginmse, against hybrid@bge-small on three datasets
  RECIPE   how every arm is trained and how its checkpoint is chosen (no gold label, no test query)
  ARMS     which embedders are trained, from what, on what, with which seed; table order is README order

A gain costs nothing at query time (the architecture does not change), but serving it needs a re-index and the
owner's decision; nothing here changes a served default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

ATTEMPT = 1  # a retrain after any held-out or BEIR number was seen is attempt 2; only attempt 1 is read against the bar

# The models an arm starts from. Hugging Face facts: the revision each cached snapshot's refs/main named on 2026-11-25
# (equal to the brief's pins); the loader refuses any other cached revision. bge-base's licence and revision are
# confirmed on the Hugging Face API before that optional arm is used, and recorded with it.
INITS: dict[str, dict[str, Any]] = {
    "bge-small": {"model": "BAAI/bge-small-en-v1.5", "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a", "licence": "mit",
                  "embedder": "st:BAAI/bge-small-en-v1.5"},
    "bge-base": {"model": "BAAI/bge-base-en-v1.5", "revision": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a", "licence": "mit",
                 "embedder": "st:BAAI/bge-base-en-v1.5", "optional": True},
}
# How both inits encode, as every committed bge row does: CLS pooling, Normalize, max_seq_length 512, do_lower_case,
# float32 weights, no prompt on either side (app.retrieval.embedders.QUERY_PROMPT_NAMES has no entry for them).
ENCODING = {"pooling": "cls", "normalize": True, "max_seq_length": 512, "do_lower_case": True, "dtype": "float32",
            "query_prompt": None, "document_prompt": None}

CORPORA = ("fixture", "scifact", "fiqa")
# The training sets an arm reads, by the corpora of its lists. The scale s is fixed per (init, training set).
TRAINING_SETS = {"all": CORPORA, "beir": ("scifact", "fiqa")}
SCALE_KEYS = (("bge-small", "all"), ("bge-small", "beir"), ("bge-base", "all"))  # the three entries of SCALE_FILE

BAR_ARM = "bge-small-marginmse"
# tag: init, loss, training set (all three corpora or SciFact + FiQA), seed, role. Every arm other than the bar row is
# exploratory and can never become "the fine-tuned model" after the fact.
ARMS: dict[str, dict[str, Any]] = {
    "bge-small-marginmse": {"init": "bge-small", "loss": "marginmse", "training_set": "all", "seed": 0, "role": "bar",
                            "about": "the bar row: bge-small fine-tuned by the MarginMSE recipe on all three corpora's training lists"},
    "bge-small-marginmse-nofixture": {"init": "bge-small", "loss": "marginmse", "training_set": "beir", "seed": 0, "role": "control",
                                      "about": ("control, fixture training data vs none: SciFact + FiQA only (9,396 lists; its own s); "
                                                "its fixture rows are zero-shot transfer. It differs from the bar row in s, in the number of "
                                                "lists (1,175 vs 1,261 steps per epoch) and in a two-corpus mix: confounded by all three")},
    "bge-small-marginmse-s1": {"init": "bge-small", "loss": "marginmse", "training_set": "all", "seed": 1, "role": "seed",
                               "about": "exploratory: the bar recipe with seed 1 (seed spread)"},
    "bge-small-marginmse-s2": {"init": "bge-small", "loss": "marginmse", "training_set": "all", "seed": 2, "role": "seed",
                               "about": "exploratory: the bar recipe with seed 2 (seed spread)"},
    "bge-small-mnrl": {"init": "bge-small", "loss": "mnrl", "training_set": "all", "seed": 0, "role": "loss",
                       "about": ("exploratory, the loss not chosen: a hand-written MultipleNegativesRankingLoss with doc-id masking, "
                                 "on the lists where the teacher ranks the source first")},
    "bge-base-marginmse": {"init": "bge-base", "loss": "marginmse", "training_set": "all", "seed": 0, "role": "base",
                           "about": "optional, exploratory: bge-base by the same recipe with its own s; its own eval files"},
}
BGE_SMALL_ARMS = tuple(t for t, a in ARMS.items() if a["init"] == "bge-small")
BASE_ARM = "bge-base-marginmse"
SEED_ARMS = ("bge-small-marginmse", "bge-small-marginmse-s1", "bge-small-marginmse-s2")  # the bar recipe's seeds 0, 1, 2
CONTROL_ARM = "bge-small-marginmse-nofixture"
MNRL_ARM = "bge-small-mnrl"

BAR: dict[str, Any] = {
    "name": "M48's bar: the adapted small embedder",
    "row": f"hybrid@{BAR_ARM}",
    "parent": "hybrid@bge-small",
    "metric": "ndcg@10",
    "passes": (f"the paired nDCG@10 delta of hybrid@{BAR_ARM} against hybrid@bge-small has a 95% interval whose lower end "
               "is strictly above zero (the stored 4-decimal lower end > 0; a stored 0.0 is not above zero) on fixture v1's "
               "held-out half (68 questions) AND on BEIR SciFact test (300) AND on BEIR FiQA test (648)"),
    "datasets": {"fixture": 68, "scifact": 300, "fiqa": 648},
    "one_seed": ("the verdict is read on seed 0 alone, because what would be served is one checkpoint; seeds 1 and 2 are "
                 "reported beside it"),
    "parent_rule": ("hybrid@bge-small recomputed in the same run, on the same device, from the same code path; on BEIR its "
                    "document matrix is computed in that run, never read from the shared cache"),
    "hybrid": ("BM25 plus the embedder fused by RRF, k = 60, equal weights, exactly as the committed hybrid@bge-small rows "
               "build it; fixture: the harness settings (retrieve_k 50, router, reranker and staleness off, mtimes from "
               "the manifest); BEIR: the runner's hybrid@bge-small config with the embedder swapped"),
    "held_out": ("fixture v1: the test half of app.eval.rerank_arms.split_questions; no choice of any kind is made on the "
                 "tune half, whose rows and the all-136 rows are reported beside it; BEIR: the test split"),
    "read_from": ("bench/results/adapted_embedders.json (scope held_out)", "bench/results/beir_scifact_adapted_embedders.json",
                  "bench/results/beir_fiqa_adapted_embedders.json"),
    "which_run": ("the first complete eval run per dataset at the C3 (manifest) commit; a re-run for any reason goes to "
                  "<name>_rerun<k>.json and never replaces it; a defect found later is recorded beside the first run's verdict"),
    "interval": ("percentile bootstrap over queries, paired, 2000 resamples, seed 0 (app/eval/stats.py); one index matrix "
                 "per dataset and scope, shared by every row"),
    "outcomes": {
        "passes": "above zero on all three",
        "does not pass": "any measured leg not above zero; a leg whose upper end is below zero is also a recorded loss",
        "open": "a leg is not measured and no measured leg fails; an open bar is not passed",
        "not run": "the bar row was not trained by the cutoff; no other arm is read in its place",
    },
    "multiple_testing": ("one pre-declared conjunctive test (intersection-union): three two-sided 95% intervals whose lower "
                         "ends must all be above zero give an overall one-sided level of at most 2.5%; every other delta "
                         "is descriptive and labelled exploratory"),
    "secondary": ("dense@bge-small-marginmse (fixture dense_only@) against the base dense row on the same three datasets",
                  "if its arm runs, the same bar form for hybrid@bge-base-marginmse against hybrid@bge-base",
                  "the mean delta over the bar recipe's seeds that ran (0 to 2), beside the seed-0 verdict"),
    "llm_judge": "no metric uses an LLM judge; the teacher's logits are training targets and the validation loss only",
}
OUTCOMES = ("passes", "does not pass", "open", "not run")

# How every arm is trained (app/eval/embedder_train.py reads these; nothing else sets them). M43's optimisation recipe
# (app.eval.small_rerankers.RECIPE) unchanged; the student is a bi-encoder scored by a fixed-scale cosine.
RECIPE: dict[str, Any] = {
    "data": ("bench/distill/queries.jsonl.gz and teacher_scores.jsonl.gz as committed by M43, never regenerated; the "
             "trainer refuses to run if either file's sha256 differs from its meta; every training id must be in "
             "queries.jsonl.gz with split train"),
    "train_lists": {"fixture": 691, "scifact": 4267, "fiqa": 5129},
    "validation_lists": {"fixture": 42, "scifact": 223, "fiqa": 270},
    "teacher": "mixedbread-ai/mxbai-rerank-large-v2 (not re-run): its raw logits, fp32, max_length 512, one per candidate",
    "list": ("the fused top 20 of hybrid@bge-small for a synthetic query, plus its source document when it is not among "
             "them, in the committed order"),
    "student_score": ("s * cos(E(q), E(d)): E is the SentenceTransformer forward (Transformer, CLS pooling, Normalize) on "
                      "the model's own preprocess() output, the path encode() takes, so training and serving cut a long "
                      "passage at the same token; each unique text is tokenized once; queries and documents bare (no prompt)"),
    "scale": ("s is fixed per init and training set before training, by a label-free rule: s = std(t_i - t_j) / std(c_i - "
              "c_j), numpy std with ddof 0 in float64, pooled over every pair i < j (the list's committed order) of every "
              "training list of that training set, t the teacher's logit and c the init model's cosine; recorded with 1/s "
              "and the epoch-0 validation loss in bench/distill/embedder_scale.json"),
    "loss": ("MarginMSE over every pair of one list: the mean over i < j of ((s_i - s_j) - (t_i - t_j))^2, s the student's "
             "score and t the teacher's logit (M43's definition); the first step is checked, torch against numpy"),
    "lists_per_step": 8,
    "epochs": 4,
    "steps_per_epoch": "ceil(training lists / lists_per_step)",
    "optimizer": "AdamW, betas (0.9, 0.999), no weight decay on biases and LayerNorm weights",
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "schedule": "linear warm-up over the first 10% of the steps, then linear decay to zero",
    "warmup_fraction": 0.1,
    "max_grad_norm": 1.0,
    "corpus_mixing": ("each list of a step comes from a corpus drawn with probability proportional to the square root of "
                      "that corpus's number of training lists; inside a corpus lists are drawn without replacement and "
                      "reshuffled when exhausted"),
    "precision": ("float32 throughout: torch.backends.cuda.matmul.allow_tf32 False and float32 matmul precision 'highest'; "
                  "both, with torch.backends.cudnn.allow_tf32, recorded in training.json and every results header"),
    "memory": "micro_lists (M43) on a GPU under 40 GB, recorded per resume segment: a memory setting, never a hyper-parameter",
    "validation": ("the synthetic queries of the held-out source documents (M43's split field): never a gold question, a "
                   "tune-half question or a BEIR query"),
    "selection": ("the epoch end (1 to 4) with the lowest macro validation MarginMSE (each corpus's mean list loss, then the "
                  "mean over the arm's corpora); ties go to the earlier epoch; whether the chosen epoch is the last is "
                  "recorded; the source document's mean rank inside its validation list is recorded per epoch as a "
                  "diagnostic and is not used"),
    "seeds": "0 for every arm; 1 and 2 for the bar recipe only, as an exploratory spread",
    "prompts": None,
    "mnrl": {
        "formula": ("cross-entropy over 20 * cos(q, every in-batch candidate), query to document (sentence-transformers 6.1 "
                    "MultipleNegativesRankingLoss defaults), hand-written so that it can mask"),
        "scale": 20.0,
        "lists": "only the lists where the teacher ranks the source document first; the source is the positive",
        "hard_negatives": 7,
        "hard_negative_rule": ("the teacher's bottom half of the list (the 10 lowest-scored candidates of a 20- or 21-candidate "
                               "list), the first 7 of those in the list's order"),
        "queries_per_batch": 32,
        "batches": ("all 32 queries from one corpus, the corpus drawn with probability proportional to the square root of "
                    "its eligible lists; no source document twice in a batch: a list whose source is already in the batch "
                    "is deferred to that corpus's next batch, never dropped; deferrals counted per epoch"),
        "mask": ("for each anchor, every in-batch candidate whose doc id equals the anchor's own positive doc id scores "
                 "-inf, except the anchor's own positive column; masked entries counted per epoch"),
        "steps_per_epoch": "ceil(eligible training lists / queries_per_batch)",
        "selection": ("the epoch end with the lowest masked loss on the validation lists where the teacher ranks the source "
                      "first, in batches built once with seed 0 (each corpus's mean, then the mean over corpora)"),
        "hardware": "a card of 40 GB or more, no micro-batching; resumed only onto such a card",
    },
}

# Leakage risk 2: near-twins of the test queries among the training and validation queries (bench/distill/embedder_twins.json)
TWINS = {
    "thresholds": (0.80, 0.85),  # bge-small cosine: twinned / not twinned subgroups of every bar delta
    "fixture_reading_threshold": 0.85,
    "assert_below": 0.90,  # M43's filter dropped every synthetic query at bge-small cosine >= 0.90 to a test query
    "assert_tolerance": 0.001,  # float drift across devices: the assertion fails at 0.901 or above
    "min_subgroup": 10,  # a subgroup under 10 queries gets no interval
    "query_sets": {"all": "the training AND validation queries of all three corpora (10,622)",
                   "beir": "the training AND validation queries of SciFact and FiQA (9,889)"},
    "embedder": "BAAI/bge-small-en-v1.5 as published, unit embeddings of the bare queries",
    "jaccard": "token-set Jaccard after app.eval.distill_queries.normalise, M43's rule",
}
FIXTURE_READINGS = ("adaptation beyond templated twins", "not separable from templated twins")

CUTOFF = "2026-12-06"  # an arm not trained by then is "not run" in these files and never evaluated in them later
CPU_BAND = (0.90, 1.10)  # "unchanged" CPU cost: the adapted/base timing ratio's interval inside this band

# Files (paths relative to the repository root).
SCALE_FILE = "bench/distill/embedder_scale.json"
TWINS_FILE = "bench/distill/embedder_twins.json"
MANIFEST_FILE = "bench/distill/embedders.json"
RESULTS = {"fixture": "bench/results/adapted_embedders.json", "scifact": "bench/results/beir_scifact_adapted_embedders.json",
           "fiqa": "bench/results/beir_fiqa_adapted_embedders.json"}
BASE_RESULTS = {"fixture": "bench/results/adapted_embedders_base.json", "scifact": "bench/results/beir_scifact_adapted_embedders_base.json",
                "fiqa": "bench/results/beir_fiqa_adapted_embedders_base.json"}
CPU_FILE = "bench/results/adapted_embedders_cpu.json"
CPU_RERUN_FILE = "bench/results/adapted_embedders_cpu_rerun.json"
TREC_DIRS = {"scifact": "bench/results/beir_scifact_adapted_embedders_trec", "fiqa": "bench/results/beir_fiqa_adapted_embedders_trec"}
SIDECAR_SCHEMA = "adapted-embedders-per-query-v1"
SCOPES = ("held_out", "tune", "all_questions")


def declarations_sha256() -> str:
    """The sha256 of what an arm is trained by (RECIPE, ARMS, INITS, ENCODING, TRAINING_SETS) as canonical JSON: a later
    arm (bge-base) must be trained by the same declarations as the bge-small arms, whatever else this file gains."""
    blob = {"RECIPE": RECIPE, "ARMS": ARMS, "INITS": INITS, "ENCODING": ENCODING, "TRAINING_SETS": TRAINING_SETS}
    return hashlib.sha256(json.dumps(blob, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def embedder_name(tag: str) -> str:
    """The embedder name an adapted arm's rows load (index_loader.build_index, the BEIR engine): ft:<tag>."""
    if tag not in ARMS:
        raise KeyError(f"unknown arm {tag!r}; known: {list(ARMS)}")
    return f"ft:{tag}"


def row_names(tag: str, dataset: str) -> tuple[str, str]:
    """(dense row, hybrid row) of an embedder tag (an arm, bge-small or bge-base): the fixture's names are
    dense_only@<tag> / hybrid@<tag>, BEIR's dense@<tag> / hybrid@<tag>."""
    dense = "dense_only" if dataset == "fixture" else "dense"
    return f"{dense}@{tag}", f"hybrid@{tag}"


# ---------------------------------------------------------------- loading an arm: ft:<tag> (index_loader.build_index, the BEIR engine)

CHECKPOINT_ENV = "ADAPTED_EMBEDDER_CHECKPOINTS"  # where the checkpoints are; --checkpoint-dir sets it
WEIGHTS_FILE = "model.safetensors"
LOADED: dict[str, dict[str, Any]] = {}  # embedder name -> what its rows record (the last load in this process)


class CheckpointError(RuntimeError):
    """An arm's checkpoint is missing, or its weights are not the ones the manifest names."""


def _repo() -> Path:
    return Path(__file__).resolve().parents[2]


def read_manifest(root: Path | None = None) -> dict[str, Any] | None:
    path = (root or _repo()) / MANIFEST_FILE
    return json.loads(path.read_text()) if path.exists() else None


def trained_arms(root: Path | None = None, manifest: dict | None = None) -> list[str]:
    """The arms the committed manifest holds, in table order (an arm not there was not trained)."""
    manifest = manifest if manifest is not None else read_manifest(root)
    return [t for t in ARMS if t in ((manifest or {}).get("arms") or {})]


def arm_checkpoint(tag: str, checkpoint_dir: Path | str | None = None, manifest: dict | None = None) -> tuple[Path, dict]:
    """(the directory of arm `tag`'s checkpoint, its manifest entry), after checking that the weights file there has the
    sha256 the committed manifest records. The checkpoints are not in the repository: `checkpoint_dir` (or
    $ADAPTED_EMBEDDER_CHECKPOINTS) says where they are, and a results row records the sha256, never the path."""
    if tag not in ARMS:
        raise KeyError(f"unknown arm {tag!r}; known: {list(ARMS)}")
    base = checkpoint_dir or os.environ.get(CHECKPOINT_ENV)
    if not base:
        raise CheckpointError(f"{tag}: no checkpoint directory (pass --checkpoint-dir or set {CHECKPOINT_ENV})")
    manifest = manifest if manifest is not None else read_manifest()
    if manifest is None:
        raise CheckpointError(f"{MANIFEST_FILE} is missing: no arm has been trained and recorded")
    entry = (manifest.get("arms") or {}).get(tag)
    if not entry:
        raise CheckpointError(f"{tag} is not in {MANIFEST_FILE}")
    directory = Path(base) / entry["directory"]
    weights = directory / WEIGHTS_FILE
    if not weights.exists():
        raise CheckpointError(f"{tag}: {entry['directory']}/{WEIGHTS_FILE} is not in the checkpoint directory")
    from app.eval.embedder_train import sha256_of

    found = sha256_of(weights)
    if found != entry["weights_sha256"]:
        raise CheckpointError(f"{tag}: {WEIGHTS_FILE} has sha256 {found}, the manifest records {entry['weights_sha256']}")
    return directory, entry


def load_checkpoint_embedder(directory: Path | str, tag: str, device: str | None = None):
    """A saved arm as the app's embedder (app.retrieval.embedders.SentenceTransformerEmbedder over the local directory),
    named ft:<tag> so that no row carries the path; it must encode as ENCODING declares."""
    from app.eval.embedder_train import check_encoding
    from app.retrieval.embedders import SentenceTransformerEmbedder

    emb = SentenceTransformerEmbedder(str(directory))
    if device is not None:
        emb._model.to(device)
    emb.model_name, emb.name = tag, f"ft:{tag}"
    check_encoding(emb._model)
    return emb


def resolve_embedder(name: str, checkpoint_dir: Path | str | None = None, root: Path | None = None, device: str | None = None):
    """(the embedder, what its rows record) for `name`, or None for a name this module does not own: ft:<tag> is an
    arm's verified checkpoint (weights sha256, then the reference embeddings its manifest records); st:<model> of an
    init is that model at its pinned revision (any other cached revision raises)."""
    if name.startswith("ft:"):
        tag = name.split(":", 1)[1]
        directory, entry = arm_checkpoint(tag, checkpoint_dir)
        from app.eval.embedder_train import check_reference

        emb = load_checkpoint_embedder(directory, tag, device=device)
        record = {"model": tag, "init": INITS[ARMS[tag]["init"]]["model"], "init_revision": INITS[ARMS[tag]["init"]]["revision"],
                  "weights_sha256": entry["weights_sha256"], "attempt": entry.get("attempt"), "manifest": MANIFEST_FILE, "revision": None,
                  "neural": True, "reference_check": check_reference(emb, entry, root or _repo())}
    else:
        model = name.split(":", 1)[1] if name.startswith("st:") else name
        init = next((k for k, v in INITS.items() if v["model"] == model), None)
        if init is None:
            return None
        from app.retrieval.embedders import SentenceTransformerEmbedder

        emb = SentenceTransformerEmbedder(model, revision=INITS[init]["revision"])
        if device is not None:
            emb._model.to(device)
        record = {"pinned_revision": INITS[init]["revision"]}
    LOADED[name] = record
    return emb, record


# ---------------------------------------------------------------- the twin flags (make embedder-twins, m48-prep, C2)


def training_query_sets(queries: Sequence[dict]) -> dict[str, list[dict]]:
    """TWINS["query_sets"]: every training AND validation query of the arms' two training sets."""
    return {name: [q for q in queries if q["corpus"] in corpora] for name, corpora in TRAINING_SETS.items()}


def twin_flags(tests: Sequence[dict[str, str]], sets: dict[str, list[dict]], embed) -> list[dict[str, Any]]:
    """Per test query, per query set: the highest base bge-small cosine to one of its queries and the highest token
    Jaccard (M43's normalisation), each with the nearest id. `embed` returns unit rows."""
    import numpy as np

    from app.eval.distill_queries import jaccard, normalise

    test_vecs = np.asarray(embed([t["text"] for t in tests]), dtype=np.float64)
    test_tokens = [set(normalise(t["text"]).split()) for t in tests]
    out = [{"set": t["set"], "id": t["id"]} for t in tests]
    for name, rows in sets.items():
        vecs = np.asarray(embed([q["query"] for q in rows]), dtype=np.float64)
        cos = test_vecs @ vecs.T
        tokens = [set(normalise(q["query"]).split()) for q in rows]
        for i, t in enumerate(test_tokens):
            c = int(np.argmax(cos[i]))
            jac = [jaccard(t, s) for s in tokens]
            j = int(np.argmax(jac))
            out[i][name] = {"cosine": round(float(cos[i, c]), 4), "cosine_nearest": rows[c]["id"], "jaccard": round(float(jac[j]), 4),
                            "jaccard_nearest": rows[j]["id"]}
    return out


def twin_assertion(flags: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Leakage risk 2 at 0.90: zero test queries at or above TWINS["assert_below"] + the tolerance, per dataset and set."""
    limit = TWINS["assert_below"] + TWINS["assert_tolerance"]
    worst: dict[str, dict[str, float]] = {}
    for f in flags:
        for name in TRAINING_SETS:
            worst.setdefault(f["set"], {})[name] = max(worst.get(f["set"], {}).get(name, -1.0), f[name]["cosine"])
    failing = {d: {n: v for n, v in by.items() if v >= limit} for d, by in worst.items()}
    failing = {d: by for d, by in failing.items() if by}
    return {"threshold": TWINS["assert_below"], "tolerance": TWINS["assert_tolerance"], "fails_at_or_above": round(limit, 4),
            "max_cosine": worst, "failing": failing, "passed": not failing}


class TwinAssertionFailed(RuntimeError):
    """A test query sits at bge-small cosine 0.90 or more (beyond the tolerance) to a training or validation query."""


def subgroup_sizes(flags: Sequence[dict[str, Any]], halves: dict[str, str] | None = None) -> dict[str, Any]:
    """Twinned test queries per dataset, query set and threshold (the fixture also per held-out / tune half)."""
    out: dict[str, Any] = {}
    for dataset in sorted({f["set"] for f in flags}):
        mine = [f for f in flags if f["set"] == dataset]
        block: dict[str, Any] = {"n": len(mine)}
        for name in TRAINING_SETS:
            block[name] = {f"{t:.2f}": sum(1 for f in mine if f[name]["cosine"] >= t) for t in TWINS["thresholds"]}
            if dataset == "fixture" and halves:
                for half in ("held_out", "tune"):
                    sub = [f for f in mine if halves.get(f["id"]) == half]
                    block[f"{name}_{half}"] = {"n": len(sub), **{f"{t:.2f}": sum(1 for f in sub if f[name]["cosine"] >= t) for t in TWINS["thresholds"]}}
        out[dataset] = block
    return out


def run_twins(root: Path, data_dir: Path, device: str = "auto") -> dict[str, Any]:
    """bench/distill/embedder_twins.json: the one step outside the eval that reads test-query text (M43's
    benchmark_queries: text only, never qrels' relevance, gold answers or a score); its output is never read by the
    trainer or the scale step. Raises TwinAssertionFailed (nothing written) when the 0.90 assertion fails."""
    from datetime import datetime, timezone

    from app.eval.distill_queries import QUERIES_FILE, benchmark_queries, read_jsonl
    from app.eval.embedder_train import sha256_of, tf32_flags
    from app.eval.gold import load_gold
    from app.eval.provenance import command_line, device_info, git_sha, git_state, model_device, run_provenance
    from app.eval.rerank_arms import split_questions
    from app.retrieval.embedders import SentenceTransformerEmbedder

    started = git_state(root)
    queries = read_jsonl(root / "bench" / "distill" / QUERIES_FILE)
    sets = training_query_sets(queries)
    tests = benchmark_queries(root, data_dir)
    gold = load_gold(root / "bench" / "gold" / "questions.jsonl")
    tune, held = split_questions(gold)
    halves = {gold[i].id: "tune" for i in tune} | {gold[i].id: "held_out" for i in held}
    emb = SentenceTransformerEmbedder(INITS["bge-small"]["model"], revision=INITS["bge-small"]["revision"])
    flags = twin_flags(tests, sets, emb.encode)
    for f in flags:
        if f["set"] == "fixture":
            f["half"] = halves[f["id"]]
    check = twin_assertion(flags)
    if not check["passed"]:
        raise TwinAssertionFailed(f"test queries at bge-small cosine >= {check['fails_at_or_above']} to a training query: {check['failing']}")
    import torch  # noqa: F401  (tf32_flags)

    return {"git_sha": git_sha(root), **run_provenance(root=root, started=started), "timestamp": datetime.now(timezone.utc).isoformat(),
            "command": command_line(), "device": device_info(device), "embedder_device": model_device(emb._model), "tf32": tf32_flags(),
            "embedder": {"model": INITS["bge-small"]["model"], "revision": INITS["bge-small"]["revision"], "note": TWINS["embedder"]},
            "rule": {k: (list(v) if isinstance(v, tuple) else v) for k, v in TWINS.items()},
            "query_sets": {name: {"n": len(rows), "corpora": list(TRAINING_SETS[name]), "about": TWINS["query_sets"][name]} for name, rows in sets.items()},
            "queries_sha256": sha256_of(root / "bench" / "distill" / QUERIES_FILE),
            "test_queries": {d: sum(1 for t in tests if t["set"] == d) for d in sorted({t["set"] for t in tests})},
            "reads": "the test queries' text only (app.eval.distill_queries.benchmark_queries); no qrels relevance, gold answer or score",
            "mnrl_note": "bge-small-mnrl is flagged against the all-three set, a superset of the queries it trained on: its flags are an upper bound",
            "assertion": check, "subgroups": subgroup_sizes(flags, halves), "flags": flags}


def read_twins(root: Path | None = None) -> tuple[dict[str, Any], dict[str, dict[str, dict]]]:
    """(the committed twin file, {dataset: {test id: flags}}), after re-checking its 0.90 assertion on the committed
    values; raises TwinAssertionFailed when it fails (the eval then writes no row)."""
    root = root or _repo()
    blob = json.loads((root / TWINS_FILE).read_text())
    check = twin_assertion(blob["flags"])
    if not check["passed"]:
        raise TwinAssertionFailed(f"{TWINS_FILE}: the 0.90 assertion fails on the committed flags: {check['failing']}")
    by: dict[str, dict[str, dict]] = {}
    for f in blob["flags"]:
        by.setdefault(f["set"], {})[f["id"]] = f
    return blob, by


# ---------------------------------------------------------------- deltas, subgroups, the seed mean (the run and --rescore share these)


def arm_of(row: str) -> str | None:
    """The arm a row name belongs to (None for a base row)."""
    tag = row.split("@", 1)[1] if "@" in row else None
    return tag if tag in ARMS else None


def base_tag(tag: str) -> str:
    return "bge-base" if ARMS[tag]["init"] == "bge-base" else "bge-small"


def comparisons(rows: Sequence[str], dataset: str) -> list[tuple[str, str]]:
    """(row, other) for every paired delta of a run, as declared: each adapted row against its base counterpart (dense
    with dense, hybrid with hybrid; the bar is hybrid@bge-small-marginmse's), hybrid@<tag> against dense@<tag>, the
    control against the bar row, each seed against seed 0."""
    names = set(rows)
    out: list[tuple[str, str]] = []
    for tag in ARMS:
        dense, hybrid = row_names(tag, dataset)
        bdense, bhybrid = row_names(base_tag(tag), dataset)
        for a, b in ((dense, bdense), (hybrid, bhybrid), (hybrid, dense)):
            if a in names and b in names:
                out.append((a, b))
    for tag in (CONTROL_ARM, *SEED_ARMS[1:]):
        for a, b in zip(row_names(tag, dataset), row_names(BAR_ARM, dataset)):
            if a in names and b in names:
                out.append((a, b))
    return out


def base_pairs(pairs: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    """The pairs that are an adapted row against its base counterpart (these get the subgroup splits)."""
    return [(a, b) for a, b in pairs if arm_of(a) and not arm_of(b)]


def _round(block: tuple[float, float, float]) -> dict[str, float]:
    return {"delta": round(block[0], 4), "lo": round(block[1], 4), "hi": round(block[2], 4)}


def subgroup_blocks(scores: dict[str, dict[str, list[float]]], pairs: Sequence[tuple[str, str]], subset: Sequence[int],
                    splits: dict[str, dict[str, list[bool]]], metric: str = "ndcg@10") -> list[dict[str, Any]]:
    """Each (row, other) delta split by each flag: per group its size and mean delta, and an interval from a fresh
    index matrix of the group's size (seed 0) when it has TWINS["min_subgroup"] queries or more. `splits` maps a split
    name to {"flag source": [bool per query]} chosen per pair (an arm's twin set) by the caller's key: the split's
    flags are splits[name][set], set = the row's arm's training set, or "any" for flags every arm shares."""
    import numpy as np

    from app.eval.stats import DEFAULT_N_BOOT, DEFAULT_SEED, paired_delta_ci, resample_indices

    out = []
    for name, by_set in splits.items():
        for row, other in pairs:
            tag = arm_of(row)
            flags = by_set.get(ARMS[tag]["training_set"]) if tag else None
            flags = flags if flags is not None else by_set.get("any")
            if flags is None:
                continue
            for value, group in ((True, "flagged"), (False, "not flagged")):
                ids = [i for i in subset if flags[i] is value]
                block: dict[str, Any] = {"split": name, "row": row, "other": other, "group": group, "n": len(ids)}
                if ids:
                    a = [scores[row][metric][i] for i in ids]
                    b = [scores[other][metric][i] for i in ids]
                    block["delta"] = round(float(np.mean(np.asarray(a) - np.asarray(b))), 4)
                    if len(ids) >= TWINS["min_subgroup"]:
                        d = paired_delta_ci(a, b, resample_indices(len(ids), n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED))
                        block[metric] = _round(d)
                    else:
                        block["interval"] = f"none: fewer than {TWINS['min_subgroup']} queries"
                out.append(block)
    return out


def twin_splits(twins: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, list[bool]]]:
    """The declared twin splits from per-query max cosines ({set: [cosine per query]}): flagged = twinned (cosine >= t)."""
    return {f"twin_cosine_{t:.2f}": {name: [c >= t for c in cos] for name, cos in twins.items()} for t in TWINS["thresholds"]}


def seed_mean(scores: dict[str, dict[str, list[float]]], subset: Sequence[int], dataset: str, metric: str = "ndcg@10") -> dict[str, Any] | None:
    """M44's form, exploratory: per query the mean of the bar recipe's seeds that ran, minus the parent, bootstrapped
    with the scope's own index matrix (seed 0); with one seed it is that seed's delta."""
    import numpy as np

    from app.eval.stats import DEFAULT_N_BOOT, DEFAULT_SEED, paired_delta_ci, resample_indices

    seeds = [t for t in SEED_ARMS if row_names(t, dataset)[1] in scores]
    parent = row_names("bge-small", dataset)[1]
    if not seeds or parent not in scores:
        return None
    mean = np.mean([[scores[row_names(t, dataset)[1]][metric][i] for i in subset] for t in seeds], axis=0)
    d = paired_delta_ci(mean, [scores[parent][metric][i] for i in subset], resample_indices(len(subset), n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED))
    return {"seeds": seeds, "n_seeds": len(seeds), "row": "the per-query mean of " + ", ".join(row_names(t, dataset)[1] for t in seeds),
            "other": parent, metric: _round(d), "label": "exploratory (M44's form); the bar is read on seed 0 alone"}


# ---------------------------------------------------------------- the results header (every file)


def m43_provenance(root: Path | None = None) -> dict[str, Any]:
    """What the training data are, copied from M43's metas: the files' sha256, the teacher and the query writer, and the
    near-duplicate counts per corpus (dropped, and at bge-small cosine 0.80 / 0.85 / 0.90 / 0.95)."""
    root = root or _repo()
    q = json.loads((root / "bench" / "distill" / "queries.meta.json").read_text())
    t = json.loads((root / "bench" / "distill" / "teacher_scores.meta.json").read_text())
    near = {c: {"checked": b["checked"], "dropped": b["near_duplicates_dropped"], "cosine_at_least": b["at_other_thresholds"]["cosine_at_least"]}
            for c, b in q["near_duplicates"]["by_corpus"].items()}
    llm = q["generation"].get("llm") or {}
    return {"queries": q["files"]["queries.jsonl.gz"], "teacher_scores": t["file"],
            "teacher": {k: t["teacher"].get(k) for k in ("model", "revision", "precision", "max_length", "scores")},
            "query_writer": {k: llm.get(k) for k in ("model", "revision", "ollama_version", "options", "request")},
            "prompt_sha256": (q["generation"].get("prompt") or {}).get("sha256"), "near_duplicates": near,
            "near_duplicate_rule": "dropped if normalised-equal, token Jaccard >= 0.7 or bge-small cosine >= 0.90 against any of the 1,084 test queries (M43)"}


def _lists_and_steps(training_set: str) -> tuple[int, int]:
    """(training lists, steps an epoch) of a training set, from RECIPE's committed list counts."""
    n = sum(RECIPE["train_lists"][c] for c in TRAINING_SETS[training_set])
    return n, -(-n // RECIPE["lists_per_step"])


def variance_note(seeds_ran: Sequence[str], base: bool = False) -> str:
    """The "not variance-robust" line, from the bar recipe's seed arms that ran (never a hard-coded seed count)."""
    if base:
        return "not variance-robust: one run (bge-base, seed 0)"
    seeds = [ARMS[t]["seed"] for t in SEED_ARMS if t in seeds_ran]
    if len(seeds) > 1:
        return (f"not variance-robust: one run per arm, except the bar recipe, which ran with {len(seeds)} seeds "
                f"({', '.join(str(s) for s in seeds)}); the bar is read on seed 0 alone")
    return "not variance-robust: one run per arm, the bar recipe included (seed 0 only); the bar is read on that one run"


def not_shown(seeds_ran: Sequence[str], base: bool = False) -> list[str]:
    """What a results file does not show (the brief's list, in its order); the seed line is derived from the arms that ran."""
    bar_lists, bar_steps = _lists_and_steps(ARMS[BAR_ARM]["training_set"])
    control_lists, control_steps = _lists_and_steps(ARMS[CONTROL_ARM]["training_set"])
    return [
        "not the live host's quality: the host embeds queries with LlamaIndex's bge instruction (compose sets no EMBEDDING_QUERY_INSTRUCTION); every row here embeds bare queries, and nothing here is deployed",
        "not the host's latency: HPC timings are indicative (adapted_embedders_cpu.json); the structural identity is the argument",
        "not transfer to an unseen corpus: the bar row read the documents of all three test corpora in training (unsupervised domain adaptation, as GPL); only the control's fixture rows measure transfer, on one small templated corpus",
        "not re-measurable from a clone: every interval re-scores from the per-query sidecar on a CPU, and nDCG recomputes from the committed TREC runs where they exist, but the checkpoints are not committed",
        ("not card-independent on FiQA: rankings are specific to the card they ran on; M41 found that bge-small's own FiQA rankings "
         "differ between cards (near-tied documents swapping adjacent places below the top ten, with no metric change); how far an "
         "adapted model's rankings move across cards is not measured"),
        (f"not a clean 'what the fixture data added': the control differs from the bar row in its scale s, in its number of training "
         f"lists ({control_lists:,} vs {bar_lists:,}; {control_steps:,} vs {bar_steps:,} steps an epoch) and in its two-corpus "
         f"square-root-of-size mix; the control-vs-bar delta is confounded by all three"),
        "not label-free in the strict sense: the teacher's and bge-small's own training data are unknown and may include these benchmarks' train splits",
        "not GPL itself: gemma4:12b queries, an mxbai-rerank-large-v2 teacher, a cosine model with a fixed scale and about 5k steps",
        variance_note(seeds_ran, base),
        "no reranked rows: the served shape has no reranker",
        "no comparison with Qwen3 or harrier except across runs (their rows are in other files, from other runs), so no paired delta",
    ]


def selection_record(entry: dict | None) -> dict[str, Any] | None:
    """How an arm's checkpoint was chosen, copied from its manifest entry for the results header: the chosen epoch, whether
    it is the last one (then the validation curve had not flattened) and whether any epoch beat epoch 0 (the untrained
    init). The loss named is the arm's own validation loss (MarginMSE, or the masked MNRL loss for bge-small-mnrl)."""
    if not entry:
        return None
    epochs = entry.get("epochs") or []
    chosen, last, beats = entry.get("chosen_epoch"), entry.get("chosen_is_last"), entry.get("beats_epoch_0")
    words = [f"epoch {chosen} of {len(epochs)} kept (the lowest validation loss on the held-out documents' synthetic queries)"
             if chosen and epochs else "the chosen epoch is not recorded in the manifest"]
    if last:
        words.append("the chosen epoch is the last one: the validation curve had not flattened")
    if beats is False:
        words.append("no epoch beat epoch 0's validation loss (the untrained init); the chosen epoch is still evaluated")
    elif beats:
        words.append("the chosen epoch beats epoch 0's validation loss (the untrained init)")
    return {"chosen_epoch": chosen, "epochs": len(epochs), "chosen_is_last": last, "beats_epoch_0": beats,
            "validation_before_training": (entry.get("validation_before_training") or {}).get("macro"),
            "validation_curve": [e.get("validation_macro") for e in epochs], "statement": "; ".join(words)}


def manifest_entry_sha256(entry: dict | None) -> str | None:
    """The sha256 of one manifest entry as canonical JSON: a later commit adds bge-base's entry to the manifest without
    changing the others, so a results file is tied to its arms' entries, not to the whole file's sha256."""
    return hashlib.sha256(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()).hexdigest() if entry else None


def arm_status(root: Path | None = None, base: bool = False) -> dict[str, str]:
    """Per arm of the run: "trained" (in the committed manifest) or "not run" (not trained by the cutoff)."""
    trained = set(trained_arms(root))
    arms = [BASE_ARM] if base else list(BGE_SMALL_ARMS)
    return {t: ("trained" if t in trained else "not run") for t in arms}


def header(rows: Sequence[str], *, root: Path | None = None, base: bool = False, flags: dict | None = None, device: str | None = None) -> dict[str, Any]:
    """The results header's `adapted_embedders` block: the bar and its arms, the M43 provenance, the twin file, the
    precision flags and device, and what the file does not show."""
    root = root or _repo()
    entries = (read_manifest(root) or {}).get("arms") or {}
    twins = root / TWINS_FILE
    from app.eval.embedder_train import sha256_of

    status = arm_status(root, base)
    ran = [t for t, s in status.items() if s == "trained"]
    return {"bar": BAR, "arms": {t: {**ARMS[t], "status": s, "weights_sha256": (entries.get(t) or {}).get("weights_sha256"),
                                     "selection": selection_record(entries.get(t)) if s == "trained" else None}
                                for t, s in status.items()},
            "bar_outcome_rule": ("this file holds one leg of the bar (bar_leg, written after its rows); the bar's outcome word over the "
                                 "three files (passes / does not pass / open / not run) is bar_table's reading of the three declared files "
                                 "(make adapted-embedders-bar, the README block); one leg not above zero is enough for 'does not pass'"),
            "attempt": ATTEMPT, "recipe": RECIPE, "inits": INITS, "encoding": ENCODING, "declarations_sha256": declarations_sha256(),
            "rows": list(rows), "m43": m43_provenance(root),
            "twins_file": {"file": TWINS_FILE, "sha256": sha256_of(twins) if twins.exists() else None},
            "manifest": {"file": MANIFEST_FILE, "sha256": sha256_of(root / MANIFEST_FILE) if (root / MANIFEST_FILE).exists() else None,
                         "arms_sha256": {t: manifest_entry_sha256(entries.get(t)) for t in ran}},
            "tf32": flags, "device": device,
            "tf32_note": ("torch.backends.cudnn.allow_tf32 is recorded as torch sets it; it governs cuDNN convolutions only, and this "
                          "encoder has none, so it does not touch these numbers (TF32 also exists only on Ampere and newer cards)"),
            "llm_judge": "no metric uses an LLM judge",
            "mnrl_twins": "bge-small-mnrl's twin flags are against the all-three query set, a superset of what it trained on: an upper bound",
            "mnrl_seen_documents": ("bge-small-mnrl's BEIR seen-document flags are computed from every training list, a superset of the "
                                    "lists it trained on (those where the teacher ranks the source first): an upper bound"),
            "load_and_score": "every embedder first ranks the check's answer above an unrelated passage with the loaded instance, or the run stops (app/eval/models_2026.py); an adapted one also reproduces its manifest's reference embeddings",
            "exploratory": "every delta other than the bar row's hybrid delta on the three datasets is descriptive and exploratory",
            "served_defaults_changed": False, "not_shown": not_shown(ran, base)}


# ---------------------------------------------------------------- fixture v1 (make bench-adapted-embedders)


FIXTURE_REPRODUCE = {"bench/results/neural.json": ("bm25_only", "dense_only@bge-small", "hybrid@bge-small", "dense_only@bge-base", "hybrid@bge-base")}
HELD_OUT_REPRODUCE = ("bench/results/small_rerankers.json", "held_out", ("hybrid@bge-small",))


def fixture_configs(arms: Sequence[str], base: bool = False) -> list:
    """The base rows of neural.json (bge-base's too for the bge-base file), then dense_only@<tag> and hybrid@<tag> per
    arm: the same BenchConfig as dense_only@bge-small / hybrid@bge-small with the embedder swapped for ft:<tag>."""
    from dataclasses import replace

    from app.eval.harness import NEURAL_CONFIGS

    by = {c.name: c for c in NEURAL_CONFIGS}
    names = ["bm25_only", "dense_only@bge-small", "hybrid@bge-small"] + (["dense_only@bge-base", "hybrid@bge-base"] if base else [])
    configs = [replace(by[n]) for n in names]
    for tag in arms:
        dense, hybrid = row_names(tag, "fixture")
        bdense, bhybrid = row_names(base_tag(tag), "fixture")
        configs.append(replace(by["dense_only@bge-small"], name=dense, embedder=embedder_name(tag), parent=bdense))
        configs.append(replace(by["hybrid@bge-small"], name=hybrid, embedder=embedder_name(tag), parent=bhybrid))
    return configs


def _fixture_twins(twins_by: dict[str, dict[str, dict]], ids: Sequence[str]) -> dict[str, dict[str, list[float]]]:
    out = {}
    for name in TRAINING_SETS:
        out[name] = {"cosine": [twins_by[i][name]["cosine"] for i in ids], "jaccard": [twins_by[i][name]["jaccard"] for i in ids]}
    return out


def _scope_blocks(scores, names, subset, pairs, metrics, twins_cos) -> dict[str, Any]:
    from app.eval.small_rerankers import scope_rows

    rows, boot = scope_rows(scores, names, subset, pairs, metrics)
    return {"bootstrap": boot, "results": rows,
            "subgroups": subgroup_blocks(scores, base_pairs(pairs), subset, twin_splits(twins_cos)),
            "seed_mean": seed_mean(scores, subset, "fixture")}


def run_fixture(root: Path | None = None, *, arms: Sequence[str] | None = None, checkpoint_dir: Path | str | None = None, device: str = "auto",
                base: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    """(the results blob, its per-question sidecar) for fixture v1: every row through harness.run_config, the code path
    of neural.json; three scopes; the twin splits of the held-out half; the parents compared with the committed rows."""
    from datetime import datetime, timezone

    from app.eval.corpus import load_manifest, verify_manifest
    from app.eval.embedder_train import tf32_flags
    from app.eval.gold import load_gold
    from app.eval.harness import CI_METRICS, run_config
    from app.eval.index_loader import mtime_provenance
    from app.eval.metrics import aggregate_retrieval
    from app.eval.models_2026 import free_gpu, reproduces
    from app.eval.provenance import command_line, device_info, git_sha, git_state, hf_revision, run_provenance, scrub, sha256_file
    from app.eval.rerank_arms import split_questions

    root = root or _repo()
    started = git_state(root)
    files_root = root / "bench" / "corpus" / "files"
    gold_path = root / "bench" / "gold" / "questions.jsonl"
    gold = load_gold(gold_path)
    manifest = load_manifest(root / "bench" / "corpus" / "MANIFEST.json")
    mismatches = verify_manifest(files_root, manifest)
    if mismatches:
        raise RuntimeError("corpus drift:\n" + "\n".join(mismatches))
    twins_blob, twins_by = read_twins(root)  # the 0.90 assertion again, on the committed flags; raises before any row
    if checkpoint_dir is not None:
        os.environ[CHECKPOINT_ENV] = str(checkpoint_dir)
    for init in (["bge-small", "bge-base"] if base else ["bge-small"]):  # the base rows' snapshots are the pinned ones
        found = hf_revision(INITS[init]["model"])
        if found != INITS[init]["revision"]:
            raise RuntimeError(f"{INITS[init]['model']}: the cached revision is {found}, the pinned one is {INITS[init]['revision']}")
    arms = list(arms) if arms is not None else [t for t in trained_arms(root) if (t == BASE_ARM) == base]
    configs = fixture_configs(arms, base=base)
    n_files = manifest["n_files"]
    scores: dict[str, dict[str, list[float]]] = {}
    orders: dict[str, list[list[str]]] = {}
    rows_meta: dict[str, dict[str, Any]] = {}
    for cfg in configs:
        row = run_config(cfg, gold, files_root, n_files=n_files, keep_retrieved=True, mtimes="manifest", check_models=True)
        scores[cfg.name] = {m: [float(q[m]) for q in row["_per_query_scores"]] for m in CI_METRICS}
        orders[cfg.name] = row["_retrieved"]
        rows_meta[cfg.name] = {"settings": row["settings"], "load_and_score": row.get("load_and_score"), "n_chunks": row["n_chunks"],
                               "latency": row["latency"], "embedder": LOADED.get(cfg.embedder)}
        free_gpu()
    names = [c.name for c in configs]
    pairs = comparisons(names, "fixture")
    tune_ids, held_ids = split_questions(gold)
    every = list(range(len(gold)))
    twins = _fixture_twins(twins_by["fixture"], [q.id for q in gold])
    twins_cos = {name: t["cosine"] for name, t in twins.items()}

    def rows_of(ids):
        return [{"id": gold[i].id, "category": gold[i].category, "relevant": gold[i].relevant_ids()} for i in ids]

    import torch  # noqa: F401  (tf32_flags)

    flags = tf32_flags()
    dev = device_info(device)
    blob: dict[str, Any] = {
        "git_sha": git_sha(root), **run_provenance(root=root, started=started), "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command_line(), "device": dev, "corpus_sha256": manifest["aggregate_sha256"], "gold_sha256": sha256_file(gold_path),
        "n_gold": len(gold), "n_files": n_files, "mtimes": mtime_provenance(files_root, "manifest", base=root),
        "adapted_embedders": header(names, root=root, base=base, flags=flags, device=dev.get("device_name")),
        "rows": rows_meta,
        "split": {"rule": "alternate train / test inside each category, gold order (app.eval.rerank_arms.split_questions)", "tune": "train",
                  "held_out": "test", "tune_ids": [gold[i].id for i in tune_ids], "held_out_ids": [gold[i].id for i in held_ids],
                  "use": "the bar is read on the held-out half; no choice was made on the tune half; no gold question is a training query"},
        "seen_documents": "every fixture file is a training source (67 of 71 chunks) or appears in training lists (the 4 validation sources): no split",
    }
    for scope, ids in zip(SCOPES, (held_ids, tune_ids, every)):
        block = _scope_blocks(scores, names, ids, pairs, CI_METRICS, twins_cos)
        for row in block["results"]:
            row["retrieval"] = aggregate_retrieval([{**r, "retrieved": orders[row["config"]][i]} for i, r in zip(ids, rows_of(ids))], n_files=n_files).as_dict()
        if scope != "held_out":
            block.pop("subgroups")  # the twin splits are read on the held-out half
        blob[scope] = block
    blob["adapted_embedders"]["bar_leg"] = bar_leg(blob, "fixture")
    committed =[{"config": r["config"], "retrieval": r["retrieval"]} for r in blob["all_questions"]["results"]]
    held = [{"config": r["config"], "retrieval": r["retrieval"]} for r in blob["held_out"]["results"]]
    blob["reproduces"] = [reproduces(root, rel, committed, [n for n in rows if n in scores]) for rel, rows in FIXTURE_REPRODUCE.items()]
    rel, scope, rows = HELD_OUT_REPRODUCE
    blob["reproduces"].append(reproduces(root, rel, held, [n for n in rows if n in scores], scope=scope))
    blob["reproduces_all"] = all(b["all"] for b in blob["reproduces"])
    sidecar = {"schema": SIDECAR_SCHEMA, "dataset": "fixture",
               "about": ("per-question scores of the results file with the same name (.json), in gold order and unrounded, with each "
                         "question's twin cosine and Jaccard per training-query set: `make adapted-embedders-rescore` recomputes every "
                         "interval, paired delta, subgroup and seed mean of its three scopes from them"),
               "git_sha": blob["git_sha"], "timestamp": blob["timestamp"], "metrics": list(CI_METRICS), "ids": [q.id for q in gold],
               "categories": [q.category for q in gold], "split": {"tune": tune_ids, "held_out": held_ids}, "scores": scores, "twins": twins}
    return scrub(blob), sidecar


# ---------------------------------------------------------------- BEIR (the runner's adapted-embedders ladder)


def seen_flags(root: Path, dataset: str, qrels: dict[str, dict[str, int]]) -> dict[str, dict[str, bool]]:
    """Leakage risk 3, per BEIR test query, from the qrels and the committed lists (computed by the eval before any
    adapted row is scored): a relevant document was a training source / appeared in a training list. Every arm trains
    on all of SciFact's and FiQA's training lists (MNRL on a subset), so the flags are the same for every arm."""
    from app.eval.distill_queries import QUERIES_FILE, read_jsonl
    from app.eval.distill_teacher import SCORES_FILE

    queries = [q for q in read_jsonl(root / "bench" / "distill" / QUERIES_FILE) if q["corpus"] == dataset and q["split"] == "train"]
    train_ids = {q["id"] for q in queries}
    sources = {q["doc_id"] for q in queries}
    listed = {c for r in read_jsonl(root / "bench" / "distill" / SCORES_FILE) if r["id"] in train_ids for c in r["candidates"]}
    return {qid: {"relevant_is_training_source": any(d in sources for d in rel), "relevant_in_training_list": any(d in listed for d in rel)}
            for qid, rel in qrels.items()}


SEEN_SPLITS = ("relevant_is_training_source", "relevant_in_training_list")


TREC_ROWS = ("dense@bge-small", "hybrid@bge-small", f"dense@{BAR_ARM}", f"hybrid@{BAR_ARM}")


def beir_twins(twins_by: dict[str, dict[str, dict]], dataset: str, ids: Sequence[str]) -> dict[str, dict[str, list[float]]]:
    missing = [i for i in ids if i not in twins_by.get(dataset, {})]
    if missing:
        raise RuntimeError(f"{TWINS_FILE} has no flags for {len(missing)} {dataset} test queries (first {missing[:3]})")
    return _fixture_twins(twins_by[dataset], ids)


def finish_beir(blob: dict[str, Any], per_query: dict[str, list[dict]], seen: dict[str, dict[str, bool]], root: Path | None = None) -> tuple[dict, dict]:
    """The BEIR results blob with its subgroups and seed mean, and its sidecar (schema adapted-embedders-per-query-v1):
    every row's per-query scores, the twin cosine and Jaccard per training-query set, both seen-document flags."""
    from app.eval.beir import PER_QUERY_METRICS
    from app.eval.provenance import scrub

    root = root or _repo()
    dataset = blob["dataset"].split("/")[-1]
    _, twins_by = read_twins(root)
    first = next(iter(per_query.values()))
    ids = [r["id"] for r in first]
    scores = {name: {m: [float(r[m]) for r in rows] for m in PER_QUERY_METRICS} for name, rows in per_query.items()}
    twins = beir_twins(twins_by, dataset, ids)
    seen_cols = {k: [bool(seen[i][k]) for i in ids] for k in SEEN_SPLITS}
    blob["adapted_embedders"].update(beir_extras(scores, ids, blob["bootstrap"], twins, seen_cols, dataset))
    from app.eval.embedder_train import tf32_flags

    blob["adapted_embedders"].update(tf32=tf32_flags(), device=(blob.get("device") or {}).get("device_name"))
    blob["adapted_embedders"]["bar_leg"] = bar_leg(blob, dataset)
    sidecar = {"schema": SIDECAR_SCHEMA, "dataset": blob["dataset"], "split": blob["split"],
               "about": ("per-query scores of the results file with the same name (.json), in the run's query order and unrounded, with "
                         "each query's twin cosine and Jaccard per training-query set and its two seen-document flags: "
                         "`make adapted-embedders-rescore` recomputes every interval, delta, subgroup and seed mean from them"),
               "git_sha": blob.get("git_sha"), "timestamp": blob.get("timestamp"), "metrics": list(PER_QUERY_METRICS), "ids": ids,
               "scores": scores, "twins": twins, "seen": seen_cols}
    return scrub(blob), sidecar


def beir_extras(scores, ids, boot, twins, seen_cols, dataset) -> dict[str, Any]:
    """The subgroup splits (twins at 0.80 / 0.85 per arm's query set; the two seen-document flags, the same for every arm)
    and the seed mean, over every query of the run."""
    subset = list(range(len(ids)))
    pairs = [(c["row"], c["other"]) for c in boot.get("comparisons") or []]
    parents = [(a, b) for a, b in base_pairs(pairs)]
    for name in scores:  # the parent deltas are the runner's delta_vs_parent, not in comparisons
        tag = arm_of(name)
        if tag:
            for a, b in zip(row_names(tag, "beir"), row_names(base_tag(tag), "beir")):
                if a == name and b in scores and (a, b) not in parents:
                    parents.append((a, b))
    splits = twin_splits({n: t["cosine"] for n, t in twins.items()})
    splits.update({k: {"any": v} for k, v in seen_cols.items()})
    return {"subgroups": subgroup_blocks(scores, parents, subset, splits), "seed_mean": seed_mean(scores, subset, "beir"),
            "subgroup_note": "descriptive only (BEIR): twins at bge-small cosine 0.80 / 0.85 to the arm's training and validation queries; seen-document flags from qrels and the committed lists"}


# ---------------------------------------------------------------- every interval again, with no model (make adapted-embedders-rescore)


def rescore_fixture(blob: dict, sidecar: dict) -> list[str]:
    from app.eval.compare import _leaf_differences

    out: list[str] = []
    for key in ("git_sha", "timestamp"):
        if sidecar.get(key) != blob.get(key):
            return [f"{key}: results file {blob.get(key)!r}, sidecar {sidecar.get(key)!r} (not the same run)"]
    n = len(sidecar["ids"])
    subsets = {"held_out": sidecar["split"]["held_out"], "tune": sidecar["split"]["tune"], "all_questions": list(range(n))}
    metrics = tuple(sidecar["metrics"])
    twins_cos = {name: t["cosine"] for name, t in sidecar["twins"].items()}
    held_again = None
    for scope in SCOPES:
        stored = blob[scope]["results"]
        names = [r["config"] for r in stored]
        pairs = [(r["config"], d["reference"]) for r in stored for d in r.get("deltas", [])]
        again = _scope_blocks(sidecar["scores"], names, subsets[scope], pairs, metrics, twins_cos)
        if scope == "held_out":
            held_again = again
        _leaf_differences(f"{scope}.bootstrap", blob[scope]["bootstrap"], again["bootstrap"], out)
        for a, b in zip(stored, again["results"]):
            for key in ("n", "ci", "deltas"):
                _leaf_differences(f"{scope}.{a['config']}.{key}", a.get(key), b.get(key), out)
            for m in metrics:
                column = [sidecar["scores"][a["config"]][m][i] for i in subsets[scope]]
                mean = round(sum(column) / len(column), 4)
                if a["retrieval"].get(m) != mean:
                    out.append(f"{scope}.{a['config']}.retrieval.{m}: results file {a['retrieval'].get(m)!r}, mean of the sidecar {mean!r}")
        _leaf_differences(f"{scope}.seed_mean", blob[scope].get("seed_mean"), again["seed_mean"], out)
        if "subgroups" in blob[scope]:
            _leaf_differences(f"{scope}.subgroups", blob[scope]["subgroups"], again["subgroups"], out)
    if "adapted_embedders" in blob:  # the header's leg of the bar, from the recomputed held-out rows
        leg = bar_leg({"adapted_embedders": blob["adapted_embedders"], "held_out": held_again}, "fixture")
        _leaf_differences("adapted_embedders.bar_leg", blob["adapted_embedders"].get("bar_leg"), leg, out)
    return out


def rescore_beir(blob: dict, sidecar: dict) -> list[str]:
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
    again = beir.attach_confidence(rows, len(sidecar["ids"]), seed=boot["seed"], n_boot=boot["n_boot"], published=blob.get("published"),
                                   comparisons=[(c["row"], c["other"]) for c in boot.get("comparisons") or []])
    _leaf_differences("bootstrap.comparisons", boot.get("comparisons"), again.get("comparisons"), out)
    for stored, row in zip(blob["results"], rows):
        for key in ("ci", "delta_vs_parent", "delta_vs_bm25"):
            _leaf_differences(f"{stored['config']}.{key}", stored.get(key), row.get(key), out)
        for m in metrics:
            column = sidecar["scores"][stored["config"]][m]
            mean = round(sum(column) / len(column), 4)
            if stored["retrieval"].get(m) != mean:
                out.append(f"{stored['config']}.retrieval.{m}: results file {stored['retrieval'].get(m)!r}, mean of the sidecar {mean!r}")
    dataset = blob["dataset"].split("/")[-1]
    extras = beir_extras(sidecar["scores"], sidecar["ids"], boot, sidecar["twins"], sidecar["seen"], dataset)
    for key in ("subgroups", "seed_mean"):
        _leaf_differences(key, blob["adapted_embedders"].get(key), extras[key], out)
    if "arms" in blob["adapted_embedders"]:  # the header's leg of the bar, from the recomputed rows
        leg = bar_leg({"adapted_embedders": blob["adapted_embedders"], "results": rows, "bootstrap": again, "split": blob.get("split")}, dataset)
        _leaf_differences("adapted_embedders.bar_leg", blob["adapted_embedders"].get("bar_leg"), leg, out)
    return out


def rescore_file(blob: dict, sidecar: dict) -> list[str]:
    """Route a sidecar of this schema to its dataset's recomputation (per_question._other_schema calls this)."""
    return rescore_fixture(blob, sidecar) if sidecar.get("dataset") == "fixture" else rescore_beir(blob, sidecar)


def result_files(root: Path | None = None) -> list[str]:
    root = root or _repo()
    rels = [*RESULTS.values(), *BASE_RESULTS.values(), CPU_RERUN_FILE]
    reruns = sorted(p.relative_to(root).as_posix() for rel in rels for p in (root / rel).parent.glob(Path(rel).stem + "_rerun*.json")
                    if not p.name.endswith(".per_query.json"))
    return [r for r in [*rels, *reruns] if (root / r).exists()]


def rescore(root: Path | None = None) -> tuple[int, list[str]]:
    """(files checked, differences) over every committed adapted-embedders results file with rows."""
    from app.eval import per_question

    root = root or _repo()
    checked, problems = 0, []
    for rel in result_files(root):
        path = root / rel
        side = per_question.sidecar_path(path)
        if not side.exists():
            problems.append(f"{rel}: no sidecar {side.name}")
            continue
        problems += [f"{rel}: {d}" for d in rescore_file(json.loads(path.read_text()), json.loads(side.read_text()))]
        checked += 1
    return checked, problems


# ---------------------------------------------------------------- the bar, from the three files


def _delta(rows: dict[str, dict], name: str, reference: str, boot: dict | None) -> dict | None:
    from app.eval.small_rerankers import _delta_to

    row = rows.get(name)
    return _delta_to(row, reference, boot) if row else None


def _scope(blob: dict, dataset: str) -> tuple[dict[str, dict], dict | None, dict]:
    """(rows by name, bootstrap, the block holding subgroups / seed mean) of the scope the bar reads."""
    if dataset == "fixture":
        s = blob["held_out"]
        return {r["config"]: r for r in s["results"]}, s.get("bootstrap"), s
    return {r["config"]: r for r in blob["results"]}, blob.get("bootstrap"), blob.get("adapted_embedders") or {}


def outcome(legs: Sequence[dict | None], trained: bool = True) -> str:
    """BAR["outcomes"] from the bar row's three legs (each a delta block, None when not measured)."""
    if not trained:
        return "not run"
    if any(leg is not None and not leg["lo"] > 0 for leg in legs):
        return "does not pass"
    if all(leg is not None and leg["lo"] > 0 for leg in legs) and len(legs) == len(CORPORA):
        return "passes"
    return "open"


def bar_leg(blob: dict, dataset: str) -> dict[str, Any]:
    """One results file's leg of the bar, written into its header after its rows exist (and recomputed by the rescore):
    the bar row's paired nDCG@10 delta against its parent, whether its stored lower end is above zero, and its
    report.verdict word. One file cannot give the bar's outcome over the three files (bar_table reads that); what it can
    say is that a leg not above zero makes the bar "does not pass" whatever the other two show. In a bge-base file the
    leg is bge-base's same bar form, exploratory."""
    from app.eval.report import verdict

    arms = (blob.get("adapted_embedders") or {}).get("arms") or {}
    base = BASE_ARM in arms
    arm = BASE_ARM if base else BAR_ARM
    row, parent = row_names(arm, dataset)[1], row_names(base_tag(arm), dataset)[1]
    rows, boot, _ = _scope(blob, dataset)
    block = _delta(rows, row, parent, boot)
    leg: dict[str, Any] = {"row": row, "parent": parent, "dataset": dataset, "scope": "held_out" if dataset == "fixture" else blob.get("split", "test"),
                           "metric": "ndcg@10"}
    if base:
        leg["label"] = "exploratory: bge-base's same bar form against hybrid@bge-base, never the bar"
    if block is None:
        trained = (arms.get(arm) or {}).get("status") == "trained"
        leg.update(measured=False, outcome_from_this_leg="not measured in this file" if trained else "not run")
        return leg
    above = block["lo"] > 0
    leg.update(measured=True, delta=block["delta"], lo=block["lo"], hi=block["hi"], lo_above_zero=above, verdict=verdict(block),
               outcome_from_this_leg=("not decided by this leg alone: the bar passes only if the other two legs are above zero too" if above
                                      else "does not pass: this leg's stored lower end is not above zero"
                                      + ("; its upper end is below zero, a recorded loss" if block["hi"] < 0 else "")))
    return leg


NOT_MEASURED = "not measured"  # the fixture reading when the file holds no held-out twin split of the bar row


def fixture_reading(block: dict | None, arm: str = BAR_ARM) -> dict[str, Any]:
    """The declared reading of the fixture leg: "adaptation beyond templated twins" only when the not-twinned held-out
    subgroup at cosine 0.85 has at least 10 questions and its bar-delta interval is above zero; "not measured" when
    there is no such split to read (no fixture file, or no bar row in it)."""
    t = TWINS["fixture_reading_threshold"]
    groups = [g for g in (block or {}).get("subgroups") or [] if g["split"] == f"twin_cosine_{t:.2f}" and g["row"] == f"hybrid@{arm}"
              and g["other"] == "hybrid@bge-small"]
    if not groups:
        return {"reading": NOT_MEASURED, "threshold": t, "not_twinned": None, "twinned": None}
    clean = next((g for g in groups if g["group"] == "not flagged"), None)
    twinned = next((g for g in groups if g["group"] == "flagged"), None)
    ok = bool(clean and clean["n"] >= TWINS["min_subgroup"] and clean.get("ndcg@10") and clean["ndcg@10"]["lo"] > 0)
    return {"reading": FIXTURE_READINGS[0] if ok else FIXTURE_READINGS[1], "threshold": t, "not_twinned": clean, "twinned": twinned}


def file_problems(blob: dict, root: Path | None = None, base: bool = False) -> tuple[list[str], list[str]]:
    """(why a results file is refused, what is flagged beside it) before bar_table reads it: the file must be a run of
    the committed checkpoints. Refused, so its rows are not read and its legs are not measured: no header with its arms;
    an arm whose status (ran / not run) differs from the committed manifest's; an arm whose weights sha256 or manifest
    entry differs from the committed one. Flagged, read and shown beside its rows: written from a working tree with
    uncommitted changes (the first run is never replaced; a defect is recorded beside its verdict). Entries are compared
    one by one, not the manifest's sha256, because bge-base's entry is added to the manifest in a later commit."""
    root = root or _repo()
    ext = blob.get("adapted_embedders")
    if not isinstance(ext, dict) or not isinstance(ext.get("arms"), dict):
        return ["no adapted_embedders header with its arms: not a file of this eval"], []
    entries = (read_manifest(root) or {}).get("arms") or {}
    hashes = (ext.get("manifest") or {}).get("arms_sha256") or {}
    refused = []
    for tag, committed in arm_status(root, base).items():
        got = ext["arms"].get(tag) or {}
        ran = got.get("status") == "trained"
        if ran != (committed == "trained"):
            refused.append(f"{tag}: {'trained' if ran else 'not run'} in the file, {committed} in the committed {MANIFEST_FILE}")
        elif ran and got.get("weights_sha256") != entries[tag]["weights_sha256"]:
            refused.append(f"{tag}: the file's weights sha256 {str(got.get('weights_sha256'))[:16]} is not the committed manifest's "
                           f"{entries[tag]['weights_sha256'][:16]}")
        elif ran and hashes.get(tag) != manifest_entry_sha256(entries[tag]):
            refused.append(f"{tag}: the manifest entry the file ran with is not the committed one")
    flags = ["written from a working tree with uncommitted changes (working_tree_dirty true)"] if blob.get("working_tree_dirty") else []
    return refused, flags


def bar_table(root: Path | None = None, base: bool = False) -> dict[str, Any]:
    """The bar read off the committed files: per arm and dataset its dense and hybrid deltas against the base rows and
    the verdict word; the bar row's outcome (BAR["outcomes"]); the seed line; the fixture twin reading; the control. A
    file that is not a run of the committed checkpoints is refused (file_problems): it is listed with the reasons, and
    its rows are not read."""
    from app.eval.report import verdict

    root = root or _repo()
    files = BASE_RESULTS if base else RESULTS
    blobs = {d: json.loads((root / rel).read_text()) for d, rel in files.items() if (root / rel).exists()}
    status = arm_status(root, base)
    bar_arm = BASE_ARM if base else BAR_ARM
    table: dict[str, Any] = {"base": base, "bar_arm": bar_arm, "datasets": {}, "arms": [], "status": status, "refused": {}}
    scopes = {}
    for dataset, blob in blobs.items():
        refused, flags = file_problems(blob, root, base)
        if refused:
            table["refused"][dataset] = {"file": files[dataset], "git_sha": blob.get("git_sha"), "reasons": refused, "flags": flags}
            continue
        rows, boot, extra = _scope(blob, dataset)
        scopes[dataset] = (rows, boot, extra)
        table["datasets"][dataset] = {"file": files[dataset], "git_sha": blob.get("git_sha"), "device": (blob.get("device") or {}).get("device_name"),
                                      "n": (boot or {}).get("n_queries"), "dirty": blob.get("working_tree_dirty"), "flags": flags,
                                      "parent": {row_names(base_tag(bar_arm), dataset)[k]: (rows.get(row_names(base_tag(bar_arm), dataset)[k]) or {}).get("ci", {}).get("ndcg@10") for k in (0, 1)}}
    for tag in [bar_arm] if base else list(BGE_SMALL_ARMS):
        entry: dict[str, Any] = {"tag": tag, "role": ARMS[tag]["role"], "status": status.get(tag), "datasets": {}}
        for dataset, (rows, boot, extra) in scopes.items():
            dense, hybrid = row_names(tag, dataset)
            bdense, bhybrid = row_names(base_tag(tag), dataset)
            if hybrid not in rows and dense not in rows:
                continue
            cell = {"dense": (rows.get(dense) or {}).get("ci", {}).get("ndcg@10"), "hybrid": (rows.get(hybrid) or {}).get("ci", {}).get("ndcg@10"),
                    "dense_vs_base": _delta(rows, dense, bdense, boot), "hybrid_vs_base": _delta(rows, hybrid, bhybrid, boot),
                    "hybrid_vs_dense": _delta(rows, hybrid, dense, boot)}
            if tag != BAR_ARM and not base and (tag == CONTROL_ARM or tag in SEED_ARMS):
                cell["hybrid_vs_bar"] = _delta(rows, hybrid, row_names(BAR_ARM, dataset)[1], boot)
                cell["dense_vs_bar"] = _delta(rows, dense, row_names(BAR_ARM, dataset)[0], boot)
            for key in ("dense_vs_base", "hybrid_vs_base"):
                cell[key + "_verdict"] = verdict(cell[key]) if cell[key] else None
            entry["datasets"][dataset] = cell
        legs = [(entry["datasets"].get(d) or {}).get("hybrid_vs_base") for d in CORPORA]
        entry["form_holds"] = outcome(legs, trained=status.get(tag) == "trained")
        table["arms"].append(entry)
    bar = next(a for a in table["arms"] if a["tag"] == bar_arm)
    legs = [(bar["datasets"].get(d) or {}).get("hybrid_vs_base") for d in CORPORA]
    table["outcome"] = outcome(legs, trained=status.get(bar_arm) == "trained")
    table["losses"] = [d for d, leg in zip(CORPORA, legs) if leg is not None and leg["hi"] < 0]
    table["legs"] = dict(zip(CORPORA, legs))
    if not base:
        ran = [a for a in table["arms"] if a["tag"] in SEED_ARMS and a["status"] == "trained"]
        table["seeds"] = {"n": len(ran), "k": sum(1 for a in ran if a["form_holds"] == "passes"), "ran": [a["tag"] for a in ran],
                          "mean": {d: (extra or {}).get("seed_mean") for d, (_, _, extra) in scopes.items()}}
        table["fixture_reading"] = fixture_reading(scopes.get("fixture", ({}, None, {}))[2])
        table["fixture_reading_080"] = fixture_reading_at(scopes.get("fixture", ({}, None, {}))[2], 0.80)
    return table


def interval_count(blob: dict, metric: str = "ndcg@10") -> int:
    """How many paired intervals of `metric` a results file holds (every scope of the fixture file; every row's deltas,
    the declared comparisons, the subgroups with an interval and the seed mean of a BEIR file): the N of "N intervals
    in this file, unadjusted", next to a delta selected because its interval excludes zero."""
    def blocks(scope: dict) -> int:  # the subgroups with an interval, and the seed mean
        return sum(1 for g in scope.get("subgroups") or [] if g.get(metric)) + (1 if (scope.get("seed_mean") or {}).get(metric) else 0)

    if any(s in blob for s in SCOPES):
        return sum(sum(1 for r in blob[s]["results"] for d in r.get("deltas") or [] if d.get(metric)) + blocks(blob[s]) for s in SCOPES if s in blob)
    rows = sum(1 for r in blob.get("results") or [] for key in ("delta_vs_parent", "delta_vs_bm25") if (r.get(key) or {}).get(metric))
    comps = sum(1 for c in (blob.get("bootstrap") or {}).get("comparisons") or [] if c.get(metric))
    return rows + comps + blocks(blob.get("adapted_embedders") or {})

def fixture_reading_at(block: dict | None, threshold: float, arm: str = BAR_ARM) -> dict[str, Any]:
    """The twin split at another threshold, reported beside the reading (0.80)."""
    groups = [g for g in (block or {}).get("subgroups") or [] if g["split"] == f"twin_cosine_{threshold:.2f}" and g["row"] == f"hybrid@{arm}"
              and g["other"] == "hybrid@bge-small"]
    return {"threshold": threshold, "not_twinned": next((g for g in groups if g["group"] == "not flagged"), None),
            "twinned": next((g for g in groups if g["group"] == "flagged"), None)}


def file_role(out: Path | str, root: Path | None = None) -> str:
    """Whether a results file is on a path the bar reads (BAR["read_from"]), stated in its header when it is written."""
    root = root or _repo()
    try:
        rel = Path(out).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        rel = None
    if rel in RESULTS.values():
        return ("a path the bar reads (BAR['read_from']): read against the bar only when written by the first complete eval run of its "
                "dataset at the C3 (manifest) commit, and only as a run of the committed checkpoints (file_problems)")
    if rel in BASE_RESULTS.values():
        return "bge-base's own file: exploratory, never read against the bar"
    return "not a path the bar reads (a re-run, the CPU re-run or a rehearsal): never read against the bar"


def seeds_line(table: dict[str, Any]) -> str:
    """'the bar's form holds for k of n seeds' (n = the seeds that ran), or 'one run (seed 0)'."""
    s = table.get("seeds") or {}
    if s.get("n", 0) <= 1:
        return "one run (seed 0)"
    return f"the bar's form holds for {s['k']} of {s['n']} seeds"


# ---------------------------------------------------------------- command line


def _parse(argv: list[str] | None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="load every embedder named and run its load-and-score check")
    p.add_argument("--out", type=Path, default=None, help="run fixture v1 and write this results file with its per-question sidecar")
    p.add_argument("--rescore", action="store_true", help="recompute every interval of the committed files from their sidecars (no model)")
    p.add_argument("--bar", action="store_true", help="print the bar, arm by arm, from the committed files")
    p.add_argument("--twins", action="store_true", help="write bench/distill/embedder_twins.json (reads the test queries' text)")
    p.add_argument("--base", action="store_true", help="the bge-base files (--out, --bar)")
    p.add_argument("--arms", default=None, help="comma-separated arm tags (default: every arm the manifest holds)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--checkpoint-dir", type=Path, default=None, help=f"where the checkpoints are (default: ${CHECKPOINT_ENV})")
    p.add_argument("--data-dir", type=Path, default=_repo() / "bench" / "external", help="--twins: where the BEIR datasets are")
    return p.parse_args(argv)


def _fmt(block: dict | None) -> str:
    return f"{block['delta']:+.4f} [{block['lo']:+.4f}, {block['hi']:+.4f}]" if block else "-"


def main(argv: list[str] | None = None) -> int:
    from app.eval.provenance import scrub

    args = _parse(argv)
    root = _repo()
    if args.rescore:
        checked, problems = rescore(root)
        for line in problems[:60]:
            print(f"  - {line}")
        print(f"adapted-embedders intervals: {'OK' if not problems else f'{len(problems)} differences'}, {checked} results files recomputed from their per-query sidecars")
        return 1 if problems else 0
    if args.bar:
        table = bar_table(root, base=args.base)
        for r in table["refused"].values():
            print(f"REFUSED {r['file']} (not read): " + "; ".join(r["reasons"] + r["flags"]))
        for meta in table["datasets"].values():
            for flag in meta.get("flags") or []:
                print(f"flagged {meta['file']}: {flag}")
        for arm in table["arms"]:
            cells = "; ".join(f"{d} {_fmt(c['hybrid_vs_base'])}" for d, c in arm["datasets"].items())
            print(f"{arm['form_holds']:14s} {arm['tag']:32s} {cells}")
        print(f"bar ({table['bar_arm']}): {table['outcome']}" + (f"; {seeds_line(table)}; fixture: {table['fixture_reading']['reading']}" if not args.base else ""))
        return 0
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.checkpoint_dir is not None:
        os.environ[CHECKPOINT_ENV] = str(args.checkpoint_dir)
    arms = [t.strip() for t in args.arms.split(",") if t.strip()] if args.arms else None
    if args.twins:
        blob = run_twins(root, args.data_dir, args.device)
        (root / TWINS_FILE).write_text(json.dumps(scrub(blob), indent=1) + "\n")
        print(json.dumps({"assertion": blob["assertion"], "subgroups": blob["subgroups"]}))
        print(f"wrote {TWINS_FILE}")
        return 0
    if args.out is not None:
        from app.eval import per_question

        blob, sidecar = run_fixture(root, arms=arms, checkpoint_dir=args.checkpoint_dir, device=args.device, base=args.base)
        out = args.out if args.out.is_absolute() else root / args.out
        blob["adapted_embedders"]["file_role"] = file_role(out, root)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(blob, indent=2) + "\n")
        per_question.write(per_question.sidecar_path(out), sidecar)
        for row in blob["held_out"]["results"]:
            print("held_out", row["config"], row["ci"]["ndcg@10"], [(d["reference"], d["ndcg@10"]) for d in row.get("deltas", [])])
        print("reproduces", json.dumps([{k: b.get(k) for k in ("file", "all")} for b in blob["reproduces"]]))
        print(f"wrote {out}")
        return 0
    if not args.check:
        print("nothing to do: pass --check, --out, --rescore, --bar or --twins", file=sys.stderr)
        return 2
    from app.eval.models_2026 import check_embedder, free_gpu
    from app.eval.provenance import model_device

    names = [INITS["bge-small"]["embedder"], *([INITS["bge-base"]["embedder"]] if args.base else [])]
    names += [embedder_name(t) for t in (arms if arms is not None else trained_arms(root))]
    failed = 0
    for name in names:
        line: dict[str, Any] = {"embedder": name}
        try:
            emb, record = resolve_embedder(name, args.checkpoint_dir, root)
            line.update(record, device=model_device(emb._model), dim=emb.dim, **emb.provenance(), check=check_embedder(emb))
            line["ok"] = True
            del emb
        except Exception as e:  # noqa: BLE001 - reported per model, then the exit status says it failed
            line.update(ok=False, error=scrub(f"{type(e).__name__}: {e}"))
            failed += 1
        free_gpu()
        print(json.dumps(line), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
