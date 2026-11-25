"""Small rerankers at cheaper serving shapes (M51): what is measured, and what was declared before any of it ran.

    python -m app.eval.small_rerankers_shapes --mde                       # the design-time half-widths, from M43's sidecars
    python -m app.eval.small_rerankers_shapes --exposure [--data-dir bench/external]
                                              # which BEIR test queries have a relevant document the students read

M43 (app/eval/small_rerankers.py) measured six small cross-encoders and two distilled students at one shape, the single
host's: the fused top 20 of hybrid@bge-small at max_length 512 in fp32. At that shape every model's API-process peak on
the indicative HPC CPU is over the api container's memory limit (bench/results/small_rerankers_cpu.json), and only
shallower depths or a 256-token cap come under it. M51 asks one question of those cheaper shapes: is there ANY shape
that beats no reranker on all three datasets (fixture v1's held-out half, BEIR SciFact, BEIR FiQA) and is under the
indicative HPC-CPU budget?

Declared before any row at a new shape was measured (the commit that adds this file precedes every M51 results file),
and pinned by tests/eval/test_small_rerankers_shapes.py so that a later change has to be made on purpose:

  GRID            six models x depth {5, 10, 20} x max_length {256, 512}, fp32: 36 reranked rows
  BANDS           under-indicative-budget / borderline / over-budget, from the committed CPU file and api_memory_mib
                  alone (quality-blind); band_of() is the rule, BAND_MEMBERSHIP its result, pinned
  FAMILIES        F1 = under-indicative-budget (k = 6), F2 = F1 + borderline (k = 16), F3 = Phase 2's three cells (k = 3),
                  each with its Bonferroni level 1 - 0.05/k on its own 20,000-resample index matrix (ADJUSTED)
  ANSWER_LEVELS   the four answers per family, in the order they are evaluated (family_answer)
  BAR             what passes, what answers, what is only descriptive
  REPLICATE       the second-device replicate (never a result row)
  PHASE2          the students retrained at max_length 256 (attempt 2; its own family F3; never merged into F1 or F2)

Nothing here sets a served default: RERANK_DEPTH, RERANKER_MAX_LENGTH, compose and Terraform are unchanged. A band is an
indicative budget on one HPC CPU, never a statement that a shape fits the host (scripts/time_reranker_on_host.py is the
owner's serving bar, and it stays unrun).
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path
from typing import Any

from app.eval import small_rerankers as sr

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------- the grid

MODELS = ("minilm-l6", "minilm-l12", "student-minilm-l6", "student-minilm-l12", "gte-modernbert-base", "bge-rerank")
DEPTHS = (5, 10, 20)
MAX_LENGTHS = (256, 512)
PRECISION = "fp32"
ANCHOR = (20, 512)  # M43's served shape: these six rows keep M43's names and must reproduce M43's committed rows
PARENT = sr.PARENT  # hybrid@bge-small: BM25 + bge-small-en-v1.5, RRF k = 60, equal weights; recomputed in the same run
METRIC = "ndcg@10"
# A model's six rows run next to each other, grouped by cap, so each (model, cap) is loaded once and released after.
GRID: list[tuple[str, int, int]] = [(tag, d, m) for tag in MODELS for m in MAX_LENGTHS for d in DEPTHS]
# bge-reranker-v2-m3: M43's CANDIDATES does not pin it; M51 pins the cached refs/main and refuses another snapshot.
BGE_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
PINNED = {**{t: sr.CANDIDATES[t]["revision"] for t in MODELS if t in sr.CANDIDATES}, "bge-rerank": BGE_REVISION}
STUDENT_WEIGHTS_SHA256 = {"student-minilm-l6": "7c3ae64c2c9002ef2ab0a28f27e025afcf9f407c6446214ea471297d603ccb0c",
                          "student-minilm-l12": "4c33938060e2b9d32f6e9b37bd910fc1c36f23e95077f04ca2379ddb1aac51fe"}
DATASETS = ("fixture", "scifact", "fiqa")
N_QUERIES = {"fixture": 68, "scifact": 300, "fiqa": 648}  # fixture: the held-out half of rerank_arms.split_questions


def row_name(tag: str, depth: int, max_length: int) -> str:
    """The results row of reranker `tag` over hybrid@bge-small at (depth, max_length), fp32, in M43's suffix style; the
    20/512 rows are exactly M43's names."""
    return f"hybrid+{tag}@bge-small/{PRECISION}-{max_length}/top{depth}"


def cells() -> list[tuple[str, int, int]]:
    """Every (tag, depth, max_length) of the grid, in run order."""
    return list(GRID)


# ---------------------------------------------------------------- budget bands (quality-blind, from committed files only)

CPU_FILE = "bench/results/small_rerankers_cpu.json"  # M43: Xeon Silver 4114, 2 torch threads, torch 2.5.1+cu121
LIMIT_FILE = "infra/modules/single-host/variables.tf"
LIMIT_VARIABLE = "api_memory_mib"
BAND_NAMES = ("under-indicative-budget", "borderline", "over-budget")
BAND_RULES = {
    "inputs": ("p95 = shapes[depth = d].p95_ms and peak = that shape's memory_mib.VmHWM of the small_rerankers_cpu.json "
               "entry with this tag and max_length = m, joined on (tag, max_length, shapes[].depth), never on the entry's "
               "config string (it says fp32-512/top20 even on the 256 entries); L = api_memory_mib's default in "
               "infra/modules/single-host/variables.tf"),
    "under-indicative-budget": "p95 <= 1000 ms AND peak <= 0.9 L (at least 10% memory headroom)",
    "borderline": "not under-indicative-budget, p95 <= 2500 ms AND peak <= L",
    "over-budget": "everything else",
    "p95_ms_under": 1000.0, "p95_ms_borderline": 2500.0, "headroom": 0.9,
    "what_a_band_is": ("an indicative budget on one HPC CPU (Xeon Silver 4114, 2 torch threads, the torch 2.5.1+cu121 "
                       "wheel), NOT a host budget: the t3.small host may be more than twice as slow, and no band says a "
                       "shape fits the host"),
}
# The result of band_of over the committed CPU file at the declarations commit (both reviewers recomputed it).
BAND_MEMBERSHIP = {
    "under-indicative-budget": [("minilm-l6", 5, 256), ("minilm-l6", 10, 256), ("minilm-l12", 5, 256),
                                ("student-minilm-l6", 5, 256), ("student-minilm-l6", 10, 256), ("student-minilm-l12", 5, 256)],
    "borderline": [("minilm-l6", 20, 256), ("minilm-l6", 5, 512), ("minilm-l6", 10, 512), ("minilm-l12", 10, 256),
                   ("minilm-l12", 5, 512), ("student-minilm-l6", 20, 256), ("student-minilm-l6", 5, 512),
                   ("student-minilm-l6", 10, 512), ("student-minilm-l12", 10, 256), ("student-minilm-l12", 5, 512)],
}


def memory_limit_mib(root: Path | None = None) -> int:
    """L: the default of api_memory_mib, read from the Terraform variables (never typed here)."""
    text = ((root or ROOT) / LIMIT_FILE).read_text()
    block = re.search(rf'variable\s+"{LIMIT_VARIABLE}"\s*\{{(.*?)\n\}}', text, re.S)
    if not block:
        raise ValueError(f"{LIMIT_FILE} has no variable {LIMIT_VARIABLE!r}")
    value = re.search(r"^\s*default\s*=\s*(\d+)\s*$", block.group(1), re.M)
    if not value:
        raise ValueError(f"{LIMIT_VARIABLE} in {LIMIT_FILE} has no numeric default")
    return int(value.group(1))


def load_cpu(root: Path | None = None) -> dict:
    return json.loads(((root or ROOT) / CPU_FILE).read_text())


def cpu_cost(tag: str, depth: int, max_length: int, cpu_blob: dict) -> dict[str, Any]:
    """The indicative CPU cost of a cell: its p50/p95 and its peak (VmHWM), joined on (tag, max_length, depth)."""
    entries = [e for e in cpu_blob["results"] if e["tag"] == tag and e["max_length"] == max_length]
    if len(entries) != 1:
        raise KeyError(f"{CPU_FILE}: {len(entries)} entries for tag {tag!r} at max_length {max_length}")
    shapes = [s for s in entries[0]["shapes"] if s["depth"] == depth]
    if len(shapes) != 1:
        raise KeyError(f"{CPU_FILE}: {len(shapes)} shapes at depth {depth} for {tag!r} at max_length {max_length}")
    s = shapes[0]
    return {"p50_ms": s["p50_ms"], "p95_ms": s["p95_ms"], "peak_mib": s["memory_mib"]["VmHWM"], "n": s["n"]}


def band_of(tag: str, depth: int, max_length: int, cpu_blob: dict, limit_mib: int) -> str:
    """The cell's band, from its indicative p95 and peak alone (no quality number is read)."""
    cost = cpu_cost(tag, depth, max_length, cpu_blob)
    p95, peak = cost["p95_ms"], cost["peak_mib"]
    if p95 <= BAND_RULES["p95_ms_under"] and peak <= BAND_RULES["headroom"] * limit_mib:
        return "under-indicative-budget"
    if p95 <= BAND_RULES["p95_ms_borderline"] and peak <= limit_mib:
        return "borderline"
    return "over-budget"


def bands(root: Path | None = None, cpu_blob: dict | None = None, limit_mib: int | None = None) -> dict[str, list[tuple[str, int, int]]]:
    """{band: [cells in grid order]} computed from the committed CPU file and the Terraform limit."""
    cpu_blob = cpu_blob or load_cpu(root)
    limit_mib = limit_mib if limit_mib is not None else memory_limit_mib(root)
    out: dict[str, list[tuple[str, int, int]]] = {b: [] for b in BAND_NAMES}
    for cell in GRID:
        out[band_of(*cell, cpu_blob, limit_mib)].append(cell)
    return out


# ---------------------------------------------------------------- the verdict per cell, the families and the answer

VERDICT = {
    "interval": ("the paired nDCG@10 delta against hybrid@bge-small per dataset: a percentile bootstrap over queries, 2000 "
                 "resamples, seed 0, 95% (app/eval/stats.py defaults, np.percentile linear interpolation; one index matrix "
                 "per file and scope, shared by every row)"),
    "rule": "on the stored 4-decimal bounds, as M43's bar_table: win if lo > 0, loss if hi < 0, tie otherwise",
    "passes_on_quality": "a cell wins on all three datasets: fixture v1's held-out half, BEIR SciFact and BEIR FiQA (an intersection: no adjustment across datasets)",
}
ADJUSTED = {"n_boot": 20_000, "seed": 0, "rng": "numpy.random.RandomState.randint (app.eval.stats.resample_indices)",
            "level": "1 - 0.05/k", "percentiles": "100 * (0.05/k) / 2 and 100 * (1 - (0.05/k) / 2), np.percentile linear interpolation",
            "matrix": "its own index matrix per dataset (fixture: drawn over the 68 held-out queries)",
            "rounding": "4 decimals; read with the same rule (lo > 0 on the stored value)"}
FAMILIES: dict[str, dict[str, Any]] = {
    "F1": {"bands": ("under-indicative-budget",), "k": 6, "headline": True,
           "about": "the headline: the shapes under the indicative HPC-CPU budget"},
    "F2": {"bands": ("under-indicative-budget", "borderline"), "k": 16, "headline": False,
           "about": "reported beside F1 and never modifying it: an F2 YES with an F1 non-YES says only that a borderline cell qualifies"},
    "F3": {"cells": (("student-minilm-l6-ml256", 5, 256), ("student-minilm-l6-ml256", 10, 256), ("student-minilm-l12-ml256", 5, 256)),
           "k": 3, "headline": False,
           "about": ("Phase 2 only (attempt 2): the retrained students' cells whose attempt-1 twin is under-indicative-budget; "
                     "its own answer, never merged into F1 or F2")},
}
FAMILY_CONTROL = "each family controls its own family-wise error at 5%; the families are not jointly controlled"
ANSWER_LEVELS = (
    ("YES", "at least one cell of the family has its adjusted interval above zero on all three datasets; the cells are named"),
    ("PASSES PER CELL, NOT AFTER ADJUSTMENT", "no cell is YES, and at least one cell passes at 95% on all three; those cells are named, and the answer is not yes"),
    ("NO", "every cell of the family has a 95% interval below zero (hi < 0) on at least one dataset, so every cell shows a measured loss somewhere"),
    ("NOT SHOWN at this n", "none of the above; this is not evidence that the cheap shapes do not help"),
)


def adjusted_alpha(k: int) -> float:
    return 0.05 / k


def adjusted_percentiles(k: int) -> tuple[float, float]:
    a = adjusted_alpha(k)
    return 100 * a / 2, 100 * (1 - a / 2)


def family_cells(family: str, root: Path | None = None, membership: dict | None = None) -> list[tuple[str, int, int]]:
    spec = FAMILIES[family]
    if "cells" in spec:
        return list(spec["cells"])
    membership = membership or bands(root)
    return [c for b in spec["bands"] for c in membership[b]]


def verdict(block: dict | None) -> str | None:
    """win / loss / tie of a stored delta block {'delta', 'lo', 'hi'} (None when the row is missing)."""
    if not block:
        return None
    return "win" if block["lo"] > 0 else ("loss" if block["hi"] < 0 else "tie")


def family_answer(cell_results: dict[str, dict[str, dict[str, dict]]]) -> dict[str, Any]:
    """The family's answer from its cells: cell row name -> dataset -> {"ci95": {delta, lo, hi}, "adjusted": {...}}.
    Evaluated in ANSWER_LEVELS order, the first that holds. A cell missing a dataset can be neither YES nor a pass,
    nor a measured loss on that dataset."""
    def every(row: dict, key: str, test) -> bool:
        return all(row.get(d, {}).get(key) is not None and test(row[d][key]) for d in DATASETS)

    yes = [n for n, row in cell_results.items() if every(row, "adjusted", lambda b: b["lo"] > 0)]
    if yes:
        return {"answer": ANSWER_LEVELS[0][0], "cells": yes}
    passes = [n for n, row in cell_results.items() if every(row, "ci95", lambda b: b["lo"] > 0)]
    if passes:
        return {"answer": ANSWER_LEVELS[1][0], "cells": passes}
    losing = {n: [d for d in DATASETS if (row.get(d) or {}).get("ci95") and row[d]["ci95"]["hi"] < 0] for n, row in cell_results.items()}
    if cell_results and all(losing.values()):
        return {"answer": ANSWER_LEVELS[2][0], "cells": sorted(cell_results), "losses": losing}
    return {"answer": ANSWER_LEVELS[3][0], "cells": []}


BAR = {
    "name": "M51: the cheaper shapes",
    "question": ("is there ANY shape that beats no reranker on all three datasets (fixture v1's held-out half, BEIR SciFact, "
                 "BEIR FiQA) and is under the indicative HPC-CPU budget?"),
    "headline": "the F1 answer; F2's answer is reported beside it and never modifies it; an over-budget cell never answers the question",
    "passes": VERDICT["passes_on_quality"],
    "verdict": VERDICT["rule"],
    "families": FAMILY_CONTROL,
    "against_the_served_shape": ("descriptive only (no adjustment; it does not enter the answer): each cell's paired delta "
                                 "against the same tag's 20/512 row at 95%; above zero the cheaper shape is better, below "
                                 "zero a measured quality cost, covering zero no measured difference, which is NOT equivalence"),
    "also_reported": ("each student against its init at the same shape; per row the reranked pairs the cap truncated and "
                      "loaded_max_length read back; the GPU rerank latency (this run's device, not the server; recorded at "
                      "run time, not re-scored)"),
    "sensitivity": ("descriptive, from the sidecars on a CPU, never changing an answer: per F1 cell and dataset a paired "
                    "sign-flip permutation p-value for the nDCG@10 delta (20,000 sign draws, seed 0); per student cell in "
                    "F1 or F2 its SciFact and FiQA deltas re-read on the test queries with no relevant document among the "
                    "students' synthetic source documents (the exposure file)"),
    "not_decided_here": "the serving bar: added p95 latency and peak memory on the host itself (scripts/time_reranker_on_host.py), unrun",
    "frozen": ("the grid, models, bands, families, adjusted levels, answer levels and their order, the replicate arm and "
               "Phase 2, at the declarations commit, before any job that scores a test query; a later change is a second "
               "reading recorded beside the first, never a replacement"),
}
SENSITIVITY = {"permutation": {"draws": 20_000, "seed": 0, "statistic": "mean paired nDCG@10 difference",
                               "p_value": "two-sided, (1 + #{|mean of sign-flipped differences| >= |observed|}) / (1 + draws)"},
               "exposure_excluded": "the SciFact and FiQA test queries with no relevant document among the students' source documents"}
DEVICE = {"result": "RTX 2080 Ti (Slurm -p eecs --constraint=rtx2080), the device of every row of M43's three files",
          "fallback": ("only after a recorded 24 h wait: all three datasets together on -p gpu --constraint=rtx8000; A40 only "
                       "if both are blocked for a further 24 h, again all three, recorded"),
          "other_device": "a row computed on any device other than the result's is a labelled replicate, never the result"}
REPLICATE = {"configs": "the parent and all 36 rows, on all three datasets, at commit C",
             "device": "the other device class of {RTX 2080 Ti, RTX 8000} than the result's",
             "reports": ("per row and dataset: whether every retrieval field, ci and delta equals the result's; the count of "
                         "per-query metric values that differ; the count of per-query ranking hashes that differ"),
             "role": "tests whether the 30 non-anchor cells per dataset are device-robust; never replaces a result row; outside F1 and F2"}
PHASE2 = {
    "attempt": 2,
    "label": "attempt 2, retrained at 256 after M43's attempt-1 numbers were seen",
    "students": {"student-minilm-l6-ml256": "student-minilm-l6", "student-minilm-l12-ml256": "student-minilm-l12"},
    "recipe_delta": {"max_length": 256},
    "recipe": ("M43's RECIPE with only max_length changed to 256: the same queries, teacher logits (scored at 512), lists, "
               "loss, schedule, epochs, selection rule and seed 0; M43's RECIPE constant is not edited"),
    "manifest": "bench/distill/students_ml256.json",
    "checkpoints": "<checkpoint root>/<tag>/ with SHA256SUMS (outside the repository)",
    "shapes": ((5, 256), (10, 256), (20, 256)),
    "family": "F3",
    "cost_source": "cost of the same-architecture attempt-1 row: the attempt-1 student's CPU row (same architecture, tokenizer and cap)",
    "gpu_hours_cap": 7, "deadline": "2026-12-12",
    "not_gating": "Phase 2 runs after the headline commit; if it does not finish, the report says so and the core stands",
    "device_note": "attempt 1 trained on A40s; a Phase-2 student trained on another device is a second stated difference, recorded in the manifest",
}
GPU_HOURS_CAP = {"core": 8, "phase2": 7}

# ---------------------------------------------------------------- design-time half-widths (make small-rerankers-shapes-mde)

MDE_TAGS = ("minilm-l6", "minilm-l12", "student-minilm-l6", "student-minilm-l12")
# The Bonferroni half-width (hi - lo) / 2 of the paired nDCG@10 delta against hybrid@bge-small, from M43's committed
# sidecars at 20/512, at each family's level, 20,000 resamples, seed 0: what effect the adjustment can detect.
DESIGN_HALF_WIDTHS = {
    "fixture": {"n": 68, "F1": {"minilm-l6": 0.0769, "minilm-l12": 0.0793, "student-minilm-l6": 0.0775, "student-minilm-l12": 0.0726},
                "F2": {"minilm-l6": 0.0874, "minilm-l12": 0.0887, "student-minilm-l6": 0.0880, "student-minilm-l12": 0.0815}},
    "scifact": {"n": 300, "F1": {"minilm-l6": 0.0358, "minilm-l12": 0.0356, "student-minilm-l6": 0.0350, "student-minilm-l12": 0.0353},
                "F2": {"minilm-l6": 0.0405, "minilm-l12": 0.0401, "student-minilm-l6": 0.0392, "student-minilm-l12": 0.0399}},
    "fiqa": {"n": 648, "F1": {"minilm-l6": 0.0202, "minilm-l12": 0.0205, "student-minilm-l6": 0.0204, "student-minilm-l12": 0.0194},
             "F2": {"minilm-l6": 0.0226, "minilm-l12": 0.0229, "student-minilm-l6": 0.0228, "student-minilm-l12": 0.0218}},
}


def _m43_columns(root: Path, dataset: str) -> tuple[dict[str, dict[str, list[float]]], list[int]]:
    """(M43's per-query scores of one dataset, the query indices its bar reads: the held-out half on the fixture)."""
    from app.eval import per_question

    side = json.loads(per_question.sidecar_path(root / sr.RESULTS[dataset]).read_text())
    subset = side["split"]["held_out"] if dataset == "fixture" else list(range(len(side["ids"])))
    return side["scores"], subset


def adjusted_delta(row: list[float], parent: list[float], k: int, idx=None) -> dict[str, float]:
    """The paired delta of `row` against `parent` at the family level 1 - 0.05/k on ADJUSTED's index matrix (or `idx`),
    rounded to 4 decimals; the call the frontier makes too."""
    from app.eval.stats import paired_delta_ci, resample_indices

    if idx is None:
        idx = resample_indices(len(row), n_boot=ADJUSTED["n_boot"], seed=ADJUSTED["seed"])
    d, lo, hi = paired_delta_ci(row, parent, idx, alpha=adjusted_alpha(k))
    return {"delta": round(d, 4), "lo": round(lo, 4), "hi": round(hi, 4)}


def design_half_widths(root: Path | None = None) -> dict[str, Any]:
    """DESIGN_HALF_WIDTHS recomputed from M43's committed sidecars, with the 20/512 deltas they come from."""
    from app.eval.stats import paired_delta_ci, resample_indices

    root = root or ROOT
    out: dict[str, Any] = {}
    for dataset in DATASETS:
        scores, subset = _m43_columns(root, dataset)
        idx = resample_indices(len(subset), n_boot=ADJUSTED["n_boot"], seed=ADJUSTED["seed"])
        parent = [scores[PARENT][METRIC][i] for i in subset]
        entry: dict[str, Any] = {"n": len(subset), "F1": {}, "F2": {}, "delta_20_512": {}}
        for tag in MDE_TAGS:
            row = [scores[sr.row_name(tag)][METRIC][i] for i in subset]
            for family in ("F1", "F2"):
                d, lo, hi = paired_delta_ci(row, parent, idx, alpha=adjusted_alpha(FAMILIES[family]["k"]))
                entry[family][tag] = round((hi - lo) / 2, 4)  # from the unrounded bounds
                entry.setdefault(f"{family}_interval", {})[tag] = {"delta": round(d, 4), "lo": round(lo, 4), "hi": round(hi, 4)}
            entry["delta_20_512"][tag] = round(sum(row) / len(row) - sum(parent) / len(parent), 4)
        out[dataset] = entry
    return out


def mde_table(widths: dict[str, Any]) -> str:
    lines = ["Design-time Bonferroni half-widths of the paired nDCG@10 delta vs hybrid@bge-small (M43's sidecars at 20/512;",
             f"{ADJUSTED['n_boot']} resamples, seed {ADJUSTED['seed']}; tags {', '.join(MDE_TAGS)})", "",
             "| dataset (n) | F1 level (k = 6) half-width | F2 level (k = 16) half-width | students' 20/512 deltas |", "|---|---|---|---|"]
    for dataset, e in widths.items():
        f1, f2 = e["F1"].values(), e["F2"].values()
        students = ", ".join(f"{e['delta_20_512'][t]:+.4f}" for t in ("student-minilm-l6", "student-minilm-l12"))
        lines.append(f"| {dataset} ({e['n']}) | {min(f1):.4f} to {max(f1):.4f} | {min(f2):.4f} to {max(f2):.4f} | {students} |")
    s = widths["scifact"]["F1_interval"]["student-minilm-l6"]
    lines += ["", f"student-minilm-l6 on SciFact at the F1 level: {s['delta']:+.4f} [{s['lo']:+.4f}, {s['hi']:+.4f}]",
              "matches the pinned DESIGN_HALF_WIDTHS: " + str(all(widths[d][f] == DESIGN_HALF_WIDTHS[d][f] for d in DATASETS for f in ("F1", "F2")))]
    return "\n".join(lines)


# ---------------------------------------------------------------- the students' exposure (make small-rerankers-shapes-exposure)

EXPOSURE_FILE = "bench/results/small_rerankers_shapes_exposure.json"
DISTILL_QUERIES = "bench/distill/queries.jsonl.gz"
EXPOSURE_RULE = ("a test query is exposed when at least one of its relevant documents (BEIR test qrels, grade > 0; on the "
                 "fixture, its relevant files) is a source document of one of the students' synthetic training queries "
                 f"({DISTILL_QUERIES}, every split); the students never saw a test query or a label")


def source_documents(root: Path | None = None) -> dict[str, set[str]]:
    """{corpus: the document ids the students' synthetic queries were written from}."""
    out: dict[str, set[str]] = {}
    with gzip.open((root or ROOT) / DISTILL_QUERIES, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                out.setdefault(row["corpus"], set()).add(str(row["doc_id"]))
    return out


def exposure(root: Path | None = None, data_dir: Path | None = None) -> dict[str, Any]:
    """The exposure file's body: per dataset, the test queries with a relevant source document, counted and listed."""
    from app.eval import beir
    from app.eval.gold import load_gold
    from app.eval.provenance import sha256_file
    from app.eval.rerank_arms import split_questions

    root = root or ROOT
    data_dir = Path(data_dir or root / "bench" / "external")
    sources = source_documents(root)
    out: dict[str, Any] = {}
    gold = load_gold(root / "bench" / "gold" / "questions.jsonl")
    _, held = split_questions(gold)
    files = {doc.split("::", 1)[0] for doc in sources.get("fixture", ())}
    exposed = {q.id for q in gold if any(p in files for p in q.relevant_ids())}
    out["fixture"] = {"n_source_documents": len(sources.get("fixture", ())), "source_unit": "fixture chunks (their files: " + str(len(files)) + ")",
                      "n_questions": len(gold), "n_exposed_questions": len(exposed),
                      "n_held_out": len(held), "n_held_out_exposed": sum(1 for i in held if gold[i].id in exposed),
                      "exposed_ids": "every gold question" if len(exposed) == len(gold) else [q.id for q in gold if q.id in exposed]}
    for dataset in ("scifact", "fiqa"):
        zip_path = data_dir / f"{dataset}.zip"
        zip_sha256 = beir.sha256_of(zip_path) if zip_path.exists() else None
        if zip_sha256 != beir.DATASETS[dataset]["sha256"]:
            raise RuntimeError(f"{dataset}.zip sha256 {zip_sha256} is not the pinned {beir.DATASETS[dataset]['sha256']}")
        dataset_root = data_dir / dataset
        _, queries, qrels = beir.load_beir(dataset_root, "test")
        src = sources.get(dataset, set())
        exposed_q = [q for q in qrels if any(d in src for d in qrels[q])]  # in the qrels' order
        relevant = {d for docs in qrels.values() for d in docs}
        out[dataset] = {"zip_sha256": zip_sha256,
                        "files_sha256": {str(p.relative_to(dataset_root)): sha256_file(p) for p in
                                         (dataset_root / "corpus.jsonl", dataset_root / "queries.jsonl", dataset_root / "qrels" / "test.tsv")},
                        "n_source_documents": len(src), "n_test_queries": len(qrels), "n_exposed_queries": len(exposed_q),
                        "n_unexposed_queries": len(qrels) - len(exposed_q),
                        "relevant_documents": {"unique": len(relevant), "unique_in_source": len(relevant & src),
                                               "pairs": sum(len(v) for v in qrels.values()),
                                               "pairs_in_source": sum(1 for v in qrels.values() for d in v if d in src)},
                        "exposed_query_ids": exposed_q}
    return out


def write_exposure(root: Path | None = None, data_dir: Path | None = None, out: Path | None = None) -> dict[str, Any]:
    from datetime import datetime, timezone

    from app.eval.provenance import command_line, git_sha, git_state, run_provenance, sha256_file

    root = root or ROOT
    started = git_state(root)
    body = exposure(root, data_dir)
    blob = {"git_sha": git_sha(root), **run_provenance(root=root, started=started), "timestamp": datetime.now(timezone.utc).isoformat(),
            "command": command_line(), "label": "M51: which test queries the distilled students' synthetic training documents touch (CPU, no model)",
            "rule": EXPOSURE_RULE, "queries_file": DISTILL_QUERIES, "queries_sha256": sha256_file(root / DISTILL_QUERIES),
            "datasets": body,
            "does_not_show": ["whether a student memorised anything: it saw documents of all three test corpora, never a test query or a label",
                              "near-duplicate synthetic queries of fixture gold questions (M43's near-twin count is in bench/distill/queries.meta.json)",
                              "the publishers' or the teacher's training data, which is not audited for any model"]}
    path = out or root / EXPOSURE_FILE
    path.write_text(json.dumps(blob, indent=2) + "\n")
    return blob


# ---------------------------------------------------------------- CLI


def _parse(argv: list[str] | None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mde", action="store_true", help="print the design-time half-widths from M43's committed sidecars (CPU, seconds)")
    p.add_argument("--exposure", action="store_true", help=f"write {EXPOSURE_FILE} (CPU; needs the BEIR data directory)")
    p.add_argument("--data-dir", type=Path, default=ROOT / "bench" / "external")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if args.mde:
        print(mde_table(design_half_widths(ROOT)))
        return 0
    if args.exposure:
        blob = write_exposure(ROOT, args.data_dir)
        for dataset, e in blob["datasets"].items():
            if dataset == "fixture":
                print(f"fixture: {e['n_exposed_questions']} of {e['n_questions']} gold questions ({e['n_held_out_exposed']} of {e['n_held_out']} held out) "
                      f"have a relevant file among the {e['n_source_documents']} source chunks")
            else:
                r = e["relevant_documents"]
                print(f"{dataset}: {e['n_exposed_queries']} of {e['n_test_queries']} test queries have a relevant source document "
                      f"({r['unique_in_source']} of {r['unique']} relevant documents; {e['n_source_documents']} source documents)")
        print(f"wrote {EXPOSURE_FILE}")
        return 0
    print("nothing to do: pass --mde or --exposure", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
