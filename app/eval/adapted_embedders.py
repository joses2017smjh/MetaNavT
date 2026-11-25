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

from typing import Any

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
