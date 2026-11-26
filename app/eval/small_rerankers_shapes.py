"""Small rerankers at cheaper serving shapes (M51): what is measured, and what was declared before any of it ran.

    python -m app.eval.small_rerankers_shapes --mde                       # the design-time half-widths, from M43's sidecars
    python -m app.eval.small_rerankers_shapes --exposure [--data-dir bench/external]
                                              # which BEIR test queries have a relevant document the students read
    python -m app.eval.small_rerankers_shapes --check [--checkpoint-dir <dir>]   # every model at both caps (GPU or CPU)
    python -m app.eval.small_rerankers_shapes --smoke [--checkpoint-dir <dir>]   # anchors + tune-half invariants; writes nothing
    python -m app.eval.small_rerankers_shapes --out bench/results/small_rerankers_shapes.json [--checkpoint-dir <dir>]
    python -m app.eval.small_rerankers_shapes --anchors [files]   # the parent and six 20/512 rows against M43's rows
    python -m app.eval.small_rerankers_shapes --frontier | --rescore | --bar | --replicate     # CPU, committed files only
    python -m app.eval.small_rerankers_shapes --phase2 --out bench/results/small_rerankers_ml256.json \
        --checkpoint-dir <attempt-1 dir> --ml256-checkpoint-dir <attempt-2 dir>   # Phase 2 rows (also --anchors, --frontier, --bar)

The BEIR rows are the BEIR runner's `small-rerankers-shapes` ladder (app/eval/beir.py; --rows anchors for the smoke), and
Phase 2's its `small-rerankers-ml256` ladder.

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
import hashlib
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


# ---------------------------------------------------------------- how a row is made (fixture v1 here; BEIR: app/eval/beir.py's ladder)

RESULTS = {"fixture": "bench/results/small_rerankers_shapes.json", "scifact": "bench/results/beir_scifact_small_rerankers_shapes.json",
           "fiqa": "bench/results/beir_fiqa_small_rerankers_shapes.json"}
FRONTIER_FILE = "bench/results/small_rerankers_shapes_frontier.json"
REPLICATE_FILE = "bench/results/small_rerankers_shapes_replicate.json"
REPLICATE_SUFFIX = "_rep"
N_ANCHOR_ROWS = 7  # the parent and the six 20/512 rows: an anchor comparing fewer rows fails
RUN_TIME_ONLY = ("the GPU rerank latency (this run's device, not the server) and the per-query rank hashes themselves (the full "
                 "rankings are not committed): recorded at run time, not re-scored; every aggregate and invariant derived from "
                 "the hashes is re-scored")


def replicate_path(rel: str) -> str:
    """bench/results/x.json -> bench/results/x_rep.json (the replicate's file of the same dataset)."""
    return rel[: -len(".json")] + REPLICATE_SUFFIX + ".json"


def anchor_rows() -> list[str]:
    return [PARENT, *(row_name(tag, *ANCHOR) for tag in MODELS)]


# Phase 2 (attempt 2): the students retrained at max_length 256, run beside the attempt-1 students and their inits at
# the same shapes, which must reproduce M51's committed shapes files (the anchors).
ML256_RESULTS = {"fixture": "bench/results/small_rerankers_ml256.json", "scifact": "bench/results/beir_scifact_small_rerankers_ml256.json",
                 "fiqa": "bench/results/beir_fiqa_small_rerankers_ml256.json"}
ML256_FRONTIER_FILE = "bench/results/small_rerankers_ml256_frontier.json"
ML256_ANCHOR_TAGS = ("minilm-l6", "minilm-l12", "student-minilm-l6", "student-minilm-l12")
N_ML256_ANCHOR_ROWS = 1 + len(ML256_ANCHOR_TAGS) * len(PHASE2["shapes"])  # the parent and 12 rows
ML256_ROWS_NOTE = ("the parent, the attempt-1 students and their inits at depth 5/10/20 x max_length 256 (the anchors: they must "
                   "equal M51's committed shapes files), and the two attempt-2 students at the same shapes")


def ml256_rows() -> list[tuple[str, str, int, int]]:
    """(row name, tag, depth, max_length) of a Phase-2 run, a model's rows next to each other: the anchors' tags, then
    the attempt-2 students."""
    return [(row_name(tag, d, m), tag, d, m) for tag in (*ML256_ANCHOR_TAGS, *PHASE2["students"]) for d, m in PHASE2["shapes"]]


def ml256_anchor_rows() -> list[str]:
    return [PARENT, *(name for name, tag, _, _ in ml256_rows() if tag in ML256_ANCHOR_TAGS)]


def ml256_checkpoint(tag: str, checkpoint_dir: Path | str | None, root: Path | None = None, manifest: dict | None = None) -> tuple[Path, dict]:
    """(the directory of Phase-2 student `tag`, its students_ml256.json entry), after checking the entry is attempt 2
    trained at max_length 256 and the weights there have the sha256 the committed manifest records. The checkpoints
    are not in the repository: `checkpoint_dir` (or $SMALL_RERANKER_ML256_CHECKPOINTS) says where they are."""
    import os

    if tag not in PHASE2["students"]:
        raise KeyError(f"unknown Phase-2 student {tag!r}; known: {sorted(PHASE2['students'])}")
    base = checkpoint_dir or os.environ.get("SMALL_RERANKER_ML256_CHECKPOINTS")
    if not base:
        raise sr.CheckpointError(f"{tag}: no checkpoint directory (pass --ml256-checkpoint-dir or set SMALL_RERANKER_ML256_CHECKPOINTS)")
    if manifest is None:
        path = (root or ROOT) / PHASE2["manifest"]
        if not path.exists():
            raise sr.CheckpointError(f"{PHASE2['manifest']} is missing: no Phase-2 student has been trained and recorded")
        manifest = json.loads(path.read_text())
    entry = (manifest.get("students") or {}).get(tag)
    if not entry:
        raise sr.CheckpointError(f"{tag} is not in {PHASE2['manifest']}")
    if entry.get("attempt") != PHASE2["attempt"] or (entry.get("recipe") or {}).get("max_length") != PHASE2["recipe_delta"]["max_length"]:
        raise sr.CheckpointError(f"{tag}: the manifest records attempt {entry.get('attempt')} at max_length "
                                 f"{(entry.get('recipe') or {}).get('max_length')}, not Phase 2's")
    weights = Path(base) / entry["directory"] / sr.WEIGHTS_FILE
    if not weights.exists():
        raise sr.CheckpointError(f"{tag}: {entry['directory']}/{sr.WEIGHTS_FILE} is not in the checkpoint directory")
    found = sr.sha256_of(weights)
    if found != entry["weights_sha256"]:
        raise sr.CheckpointError(f"{tag}: {sr.WEIGHTS_FILE} has sha256 {found}, the manifest records {entry['weights_sha256']}")
    return weights.parent, entry


def loader(resolve, device: str = "auto"):
    """(tag, max_length) -> (model, record) for rerank_rows: what its own loader does, for every tag including Phase 2's
    (whose provenance is named by its tag, as an attempt-1 student's is)."""
    def load(tag: str, cap: int):
        from app.eval.provenance import model_provenance
        from app.retrieval.rerank import get_cross_encoder

        source, record, after_load = resolve(tag)
        model = get_cross_encoder(source, precision=PRECISION, max_length=cap, device=None if device == "auto" else device, raise_errors=True)
        named = tag if tag in sr.STUDENTS or tag in PHASE2["students"] else sr.model_of(tag)
        return model, {**model_provenance(named, "reranker"), **record, **(after_load(model) if after_load else {})}

    return load


def parse_row(name: str) -> tuple[str, int, int] | None:
    """(tag, depth, max_length) of a reranked row's name; None for the parent or another name."""
    m = re.fullmatch(r"hybrid\+(.+)@bge-small/fp32-(\d+)/top(\d+)", name)
    return (m.group(1), int(m.group(3)), int(m.group(2))) if m else None


def comparisons(names: list[str], *, with_parent: bool = True) -> list[tuple[str, str]]:
    """(row, other) for every paired delta: each reranked row against the unreranked parent (the bar; on BEIR that is
    the row's delta_vs_parent, so with_parent=False), against the same model's 20/512 row (the cost of cheapening;
    descriptive), and a student against its init at the same shape. A Phase-2 student (attempt 2) is compared with the
    attempt-1 student and with the published init, both at the same shape."""
    present = set(names)
    out = []
    for name in names:
        cell = parse_row(name)
        if cell is None:
            continue
        tag, depth, cap = cell
        if with_parent and PARENT in present:
            out.append((name, PARENT))
        anchor = row_name(tag, *ANCHOR)
        if name != anchor and anchor in present:
            out.append((name, anchor))
        for init in (sr.STUDENTS.get(tag), PHASE2["students"].get(tag), sr.STUDENTS.get(PHASE2["students"].get(tag, ""))):
            if init and row_name(init, depth, cap) in present:
                out.append((name, row_name(init, depth, cap)))
    return out


def rank_hashes(ids: list[str]) -> dict[str, str]:
    """Name-independent hashes of one ranked list: the ordered top 10, the sorted top 10 and the ordered full list
    (12-hex-digit sha256 prefixes of the ids joined by newlines). Two rows ranking a query alike hash alike."""
    h = lambda xs: hashlib.sha256("\n".join(xs).encode("utf-8")).hexdigest()[:12]  # noqa: E731
    return {"top10": h(ids[:10]), "top10_set": h(sorted(ids[:10])), "ranking": h(ids)}


def hash_columns(lists: list[list[str]]) -> dict[str, list[str]]:
    cols: dict[str, list[str]] = {"top10": [], "top10_set": [], "ranking": []}
    for ids in lists:
        for k, v in rank_hashes(ids).items():
            cols[k].append(v)
    return cols


def ranking_digest(rank_hash: dict[str, list[str]]) -> str:
    """The sha256 of a row's per-query `ranking` hashes joined in sidecar id order: equal for rows that rank alike."""
    return hashlib.sha256("\n".join(rank_hash["ranking"]).encode("utf-8")).hexdigest()


def pair_token_lengths(model, query: str, passages: list[str]) -> list[int]:
    """Each (query, passage) pair's token count by the loaded cross-encoder's own tokenizer, special tokens on,
    truncation and padding OFF (gte-modernbert's tokenizer.json would otherwise pad every pair to 8,000 tokens)."""
    if not passages:
        return []
    enc = model.tokenizer([query] * len(passages), list(passages), add_special_tokens=True, truncation=False, padding=False)
    return [len(ids) for ids in enc["input_ids"]]


def derived(extra: dict[str, Any], parent_extra: dict[str, Any] | None) -> dict[str, Any]:
    """What a row records from its per-query extras (and --rescore re-derives): ranking_digest, how many queries' top
    10 differ from the parent's (from the top10 hashes) and the truncation totals."""
    out: dict[str, Any] = {"ranking_digest": ranking_digest(extra["rank_hash"])}
    if parent_extra is not None:
        out["n_top10_differs_from_parent"] = sum(1 for a, b in zip(extra["rank_hash"]["top10"], parent_extra["rank_hash"]["top10"]) if a != b)
    counts = extra.get("truncated_pairs")
    if counts is not None and any(c is not None for c in counts):
        out["truncated_pairs"] = {"total": sum(counts), "queries_with_any": sum(1 for c in counts if c)}
    return out


def resolver(checkpoint_dir: Path | str | None = None, root: Path | None = None, ml256_dir: Path | str | None = None):
    """M43's resolver (pinned revisions; a student's verified checkpoint and reference logits), plus M51's pin of
    bge-reranker-v2-m3: its cached snapshot must be BGE_REVISION, or the run stops. Phase 2's tags resolve to their
    verified attempt-2 checkpoints (ml256_checkpoint) in `ml256_dir`, and must give their manifest's reference logits."""
    inner = sr.resolver(checkpoint_dir, root)

    def resolve(name: str):
        if name in PHASE2["students"]:
            directory, entry = ml256_checkpoint(name, ml256_dir, root)

            def after_load(model):
                from app.eval.distill_train import check_reference

                return {"reference_check": check_reference(model, entry, root or ROOT)}

            return str(directory), {"model": name, "student_of": sr.model_of(sr.STUDENTS[PHASE2["students"][name]]),
                                    "attempt_1_twin": PHASE2["students"][name], "weights_sha256": entry["weights_sha256"],
                                    "attempt": entry.get("attempt"), "manifest": PHASE2["manifest"], "revision": None, "neural": True}, after_load
        source, record, after_load = inner(name)
        if sr.tag_of(name) == "bge-rerank":
            from app.eval.provenance import hf_revision

            found = hf_revision(source)
            if found != BGE_REVISION:
                raise RuntimeError(f"bge-rerank: the cached revision is {found}, the pinned one is {BGE_REVISION}")
            record = {**record, "pinned_revision": BGE_REVISION}
        return source, record, after_load

    return resolve


def header(row_names: list[str] | None = None, rows: str = "all", phase2: bool = False) -> dict[str, Any]:
    """The results header's `small_rerankers` block (the key M43's sidecar routing reads), with a `shapes` block in
    place of M43's single `shape` and M51's bar in place of M43's; a Phase-2 run adds its `phase2` block."""
    extra = {"phase2": {**PHASE2, "rows": ML256_ROWS_NOTE, "anchor_files": RESULTS}} if phase2 else {}
    return {"milestone": "M51", "parent": PARENT, **extra,
            "shapes": {"depths": list(DEPTHS), "max_lengths": list(MAX_LENGTHS), "precision": PRECISION, "cells": len(GRID),
                       "anchor": {"rerank_top": ANCHOR[0], "max_length": ANCHOR[1], "rows": "M43's committed rows, which this run must reproduce"},
                       "rows_filter": rows},
            "bar": BAR, "verdict": VERDICT, "families": {k: {kk: vv for kk, vv in v.items()} for k, v in FAMILIES.items()},
            "family_control": FAMILY_CONTROL, "answer_levels": [a for a, _ in ANSWER_LEVELS], "adjusted": ADJUSTED, "bands": BAND_RULES,
            "models": {tag: ({"model": sr.model_of(tag), "pinned_revision": PINNED[tag]} if tag in PINNED else
                             {"init": sr.STUDENTS[tag], "attempt": 1, "weights_sha256": STUDENT_WEIGHTS_SHA256[tag]}) for tag in MODELS},
            "device_rule": DEVICE, "rows": list(row_names) if row_names is not None else None, "run_time_only": RUN_TIME_ONLY,
            "load_and_score": ("every (model, cap) first ranks the check's answer above an unrelated passage with the loaded instance, "
                               "and its loaded_max_length reads back as the cap, or the run stops (app/eval/models_2026.py)"),
            "served_defaults_changed": False}


def _first_stage(root: Path, embedder: str | None, retrieve_k: int):
    """Fixture v1's first stage exactly as M43's run_fixture builds it: (gold, manifest, index, fused lists, embedder
    check, embedder name)."""
    from app.eval.corpus import load_manifest, verify_manifest
    from app.eval.gold import load_gold
    from app.eval.harness import BGE_SMALL
    from app.eval.index_loader import build_index
    from app.eval.models_2026 import check_embedder

    files_root = root / "bench" / "corpus" / "files"
    gold = load_gold(root / "bench" / "gold" / "questions.jsonl")
    manifest = load_manifest(root / "bench" / "corpus" / "MANIFEST.json")
    mismatches = verify_manifest(files_root, manifest)
    if mismatches:
        raise RuntimeError("corpus drift:\n" + "\n".join(mismatches))
    embedder = embedder or BGE_SMALL
    index = build_index(files_root, embedder_name=embedder, retrieve_k=retrieve_k, enable_router=False, enable_rerank=False, mtimes="manifest")
    embedder_check = check_embedder(index.embedder) if embedder.startswith("st:") else None
    fused = [[(h.chunk, h.score) for h in index.retrieve(q.question, k=retrieve_k).hits] for q in gold]  # fused order, RRF score
    return gold, manifest, index, fused, embedder_check, embedder


def grid_rows(tags: list[str] | None = None, *, anchors_only: bool = False, skip_anchors: bool = False) -> list[tuple[str, str, int, int]]:
    """(row name, tag, depth, max_length) for rerank_rows, in grid order (a model's rows next to each other, by cap)."""
    tags = list(tags) if tags is not None else list(MODELS)
    out = []
    for tag, d, m in GRID:
        if tag not in tags or (anchors_only and (d, m) != ANCHOR) or (skip_anchors and (d, m) == ANCHOR):
            continue
        out.append((row_name(tag, d, m), tag, d, m))
    return out


def run_fixture(root: Path | None = None, *, tags: list[str] | None = None, checkpoint_dir: Path | str | None = None, device: str = "auto",
                retrieve_k: int = 50, load=None, embedder: str | None = None, count_tokens=None, phase2: bool = False,
                ml256_dir: Path | str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """(the results blob, its per-question sidecar) for fixture v1 at every shape of the grid. `load` (tests) replaces
    the model loader: (tag, max_length) -> (model, record); `count_tokens` the tokenizer count (default: the model's
    own, pair_token_lengths); `embedder` replaces bge-small. phase2: Phase 2's rows (ml256_rows), anchored on M51's
    committed shapes file; the attempt-2 checkpoints are in `ml256_dir`."""
    from datetime import datetime, timezone

    from app.eval.harness import CI_METRICS
    from app.eval.index_loader import mtime_provenance
    from app.eval.metrics import aggregate_retrieval
    from app.eval.models_2026 import reproduces
    from app.eval.provenance import command_line, device_info, git_sha, git_state, model_device, model_provenance, run_provenance, sha256_file
    from app.eval.rerank_arms import _paths, split_questions

    root = root or ROOT
    started = git_state(root)
    gold, manifest, index, fused, embedder_check, embedder = _first_stage(root, embedder, retrieve_k)
    files_root = root / "bench" / "corpus" / "files"
    n_files = manifest["n_files"]
    orders: dict[str, list[list[str]]] = {PARENT: [_paths(pairs) for pairs in fused]}
    rows = ml256_rows() if phase2 else grid_rows(tags)
    resolve = resolver(checkpoint_dir, root, ml256_dir)
    ranked, infos, loads = sr.rerank_rows(gold, fused, rows, resolve=resolve, load=load or (loader(resolve, device) if phase2 else None), device=device,
                                          parent=orders[PARENT], check_cap=True, count_tokens=count_tokens or pair_token_lengths)
    orders.update(ranked)
    extra: dict[str, dict[str, Any]] = {PARENT: {"truncated_pairs": None, "rank_hash": hash_columns(orders[PARENT])}}
    row_info: dict[str, dict[str, Any]] = {}
    for name, tag, depth, cap in rows:
        info = infos[name]
        extra[name] = {"truncated_pairs": info.pop("_truncated_pairs"), "rank_hash": hash_columns(orders[name])}
        got = derived(extra[name], extra[PARENT])
        if got["n_top10_differs_from_parent"] != info["n_top10_differs_from_parent"]:
            raise RuntimeError(f"{name}: the top-10 hashes and the lists disagree about which questions moved")
        row_info[name] = {"tag": tag, "depth": depth, "max_length": cap, "load": f"{tag}|{cap}", "rerank_latency": info["rerank_latency"], **got}

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
        "command": command_line(), "device": device_info(device), "corpus_sha256": manifest["aggregate_sha256"], "gold_sha256": sha256_file(root / "bench" / "gold" / "questions.jsonl"),
        "n_gold": len(gold), "n_files": n_files, "mtimes": mtime_provenance(files_root, "manifest", base=root), "n_chunks": len(index.chunks),
        "small_rerankers": header(names, phase2=phase2),
        "first_stage": {"config": PARENT, "retrieve_k": retrieve_k, "reranked": "the fused top d chunks of each row; the rest keep the fused order",
                        "embedder": {**model_provenance(embedder, "embedder"), "device": model_device(getattr(index.embedder, "_model", None)),
                                     **(index.embedder.provenance() if hasattr(index.embedder, "provenance") else {}),
                                     **({"load_and_score": embedder_check} if embedder_check else {})}},
        "loads": loads, "rerankers": row_info,
        "split": {"rule": "alternate train / test inside each category, gold order (app.eval.rerank_arms.split_questions)", "tune": "train",
                  "held_out": "test", "tune_ids": [gold[i].id for i in tune_ids], "held_out_ids": [gold[i].id for i in held_ids],
                  "use": "the bar is read on the held-out half; no choice was made on the tune half; no gold question is a training query"},
    }
    for scope, ids in zip(sr.SCOPES, (held_ids, tune_ids, every)):
        scope_rows, boot = sr.scope_rows(scores, names, ids, pairs, CI_METRICS)
        for row in scope_rows:
            subset = aggregate_retrieval([{**r, "retrieved": orders[row["config"]][i]} for i, r in zip(ids, rows_of(ids))], n_files=n_files)
            row["retrieval"] = subset.as_dict()
        blob[scope] = {"bootstrap": boot, "results": scope_rows}
    blob["results"] = blob["all_questions"]["results"]
    blob["reproduces"] = [reproduces(root, rel, blob["results"], [n for n in want if n in orders]) for rel, want in sr.FIXTURE_REPRODUCE.items()
                          if any(n in orders for n in want)]
    del blob["results"]
    blob["anchors"] = (fixture_anchors(root, blob, RESULTS["fixture"], ml256_anchor_rows(), N_ML256_ANCHOR_ROWS) if phase2 else fixture_anchors(root, blob))
    blob["does_not_show"] = DOES_NOT_SHOW["fixture"] + (DOES_NOT_SHOW["phase2"] if phase2 else [])
    sidecar = {"schema": sr.SIDECAR_SCHEMA,
               "about": "per-question scores of the results file with the same name (.json), in gold order and unrounded: "
                        "`make small-rerankers-shapes-rescore` recomputes every interval and paired delta of its three scopes from them, "
                        "and every per-row total from per_query_extra",
               "git_sha": blob["git_sha"], "timestamp": blob["timestamp"], "metrics": list(CI_METRICS), "ids": [q.id for q in gold],
               "categories": [q.category for q in gold], "split": {"tune": tune_ids, "held_out": held_ids}, "scores": scores,
               "per_query_extra": extra}
    return blob, sidecar


def _delta_block(row: dict, reference: str) -> dict | None:
    for block in row.get("deltas") or []:
        if block["reference"] == reference:
            return block
    return None


def fixture_anchors(root: Path, blob: dict, committed_rel: str | None = None, names: list[str] | None = None, n_rows: int = N_ANCHOR_ROWS) -> dict[str, Any]:
    """The parent and the six 20/512 rows of this run against M43's committed fixture file, scope by scope (its rows
    are under each scope: it has no top-level results): retrieval and ci; each 20/512 row's delta block against the
    parent; each student's against its init at 20/512. All three scopes must compare N_ANCHOR_ROWS rows. Phase 2 passes
    M51's committed shapes file, its 13 anchor rows (the parent, the attempt-1 students and their inits at 5/10/20 x
    256) and n_rows = 13; a student's init delta is then at the row's own shape."""
    committed_rel = committed_rel or sr.RESULTS["fixture"]
    committed = json.loads((root / committed_rel).read_text())
    out: dict[str, Any] = {"file": committed_rel, "git_sha": committed.get("git_sha"), "device": (committed.get("device") or {}).get("device_name"),
                           "compared": "retrieval, ci, the delta against hybrid@bge-small and a student's delta against its init at the row's shape, per scope",
                           "scopes": {}}
    for scope in sr.SCOPES:
        mine = {r["config"]: r for r in blob[scope]["results"]}
        theirs = {r["config"]: r for r in committed[scope]["results"]}
        rows: dict[str, Any] = {}
        for name in (names or anchor_rows()):
            if name not in mine:
                continue
            if name not in theirs:
                rows[name] = {"in_committed_file": False, "equal": False}
                continue
            entry: dict[str, Any] = {"retrieval_equal": mine[name]["retrieval"] == theirs[name]["retrieval"], "ci_equal": mine[name]["ci"] == theirs[name]["ci"]}
            refs = [] if name == PARENT else [PARENT]
            cell = parse_row(name)
            if cell and cell[0] in sr.STUDENTS:
                refs.append(row_name(sr.STUDENTS[cell[0]], cell[1], cell[2]))
            for ref in refs:
                entry[f"delta_vs {ref} equal"] = _delta_block(mine[name], ref) == _delta_block(theirs[name], ref) and _delta_block(mine[name], ref) is not None
            entry["equal"] = all(v for k, v in entry.items() if k.endswith("equal"))
            rows[name] = entry
        out["scopes"][scope] = {"rows": rows, "n_compared": len(rows), "all": len(rows) == n_rows and all(e["equal"] for e in rows.values())}
    out["all"] = all(s["all"] for s in out["scopes"].values())
    return out


def beir_anchors(blob: dict, n_rows: int = N_ANCHOR_ROWS) -> dict[str, Any]:
    """The BEIR run's reproduction of M43's committed file (Phase 2: of M51's): n_rows rows, each with retrieval, ci and
    the parent delta equal (a metric difference stops the commit); ranking_equal reported, never stopped on."""
    blocks = blob.get("reproduces") or []
    block = blocks[0] if isinstance(blocks, list) and blocks else (blocks or {})
    rows = block.get("rows") or {}
    metric = {n: bool(e.get("equal") and e.get("ci_equal", False) and (n == PARENT or e.get("delta_vs_parent_equal", False))) for n, e in rows.items()}
    return {"file": block.get("file"), "n_compared": len(rows), "metrics_equal": len(rows) == n_rows and all(metric.values()),
            "metric_differences": sorted(n for n, ok in metric.items() if not ok),
            "ranking_differences": sorted(n for n, e in rows.items() if e.get("ranking_equal") is False)}


def annotate_beir(blob: dict, extra: dict[str, dict] | None, n_anchor_rows: int = N_ANCHOR_ROWS) -> dict:
    """Per row of a BEIR shapes file: ranking_digest, n_top10_differs_from_parent and the truncation totals (from the
    per-query extras the sidecar keeps); and the file's anchors and what it does not show."""
    extra = extra or {}
    for row in blob["results"]:
        if row["config"] in extra:
            row.update(derived(extra[row["config"]], extra.get(PARENT) if row["config"] != PARENT else None))
    blob["anchors"] = beir_anchors(blob, n_anchor_rows)
    blob["does_not_show"] = DOES_NOT_SHOW["beir"] + (DOES_NOT_SHOW["phase2"] if n_anchor_rows == N_ML256_ANCHOR_ROWS else [])
    return blob


DOES_NOT_SHOW = {
    "fixture": ["host timing or memory: the serving bar, unrun (scripts/time_reranker_on_host.py)",
                "the cap: no fixture (question, chunk) pair exceeds 256 tokens, so on the fixture every cap-256 row measures depth only",
                "students' quality on an unseen corpus: they were trained on synthetic queries over every fixture chunk (never a gold question or label)",
                "equivalence: a delta against 20/512 that covers zero is no measured difference, not the same quality",
                "answer quality: no LLM answer is generated or judged", RUN_TIME_ONLY],
    "phase2": ["the attempt-2 students are one training run each (seed 0), retrained after M43's attempt-1 numbers were seen; "
               "their cost is the same-architecture attempt-1 row's (not timed again)",
               "the training corpora: like attempt 1, they read every fixture chunk and 1,500 SciFact and 1,800 FiQA documents "
               "(never a test query or label)"],
    "beir": ["host timing or memory: the serving bar, unrun (scripts/time_reranker_on_host.py)",
             "students' quality on an unseen corpus: they read 1,500 SciFact and 1,800 FiQA documents (never a test query or label; see the exposure file)",
             "the publishers' training data overlap with BEIR test sets: not audited for any model",
             "equivalence: a delta against 20/512 that covers zero is no measured difference, not the same quality",
             "answer quality: no LLM answer is generated or judged", RUN_TIME_ONLY],
}


# ---------------------------------------------------------------- the smoke (code commit B; computes no new number)


def smoke(root: Path | None = None, *, checkpoint_dir: Path | str | None = None, device: str = "auto", load=None, embedder: str | None = None,
          count_tokens=None) -> dict[str, Any]:
    """Before any number at a new shape exists: the parent and the six 20/512 rows on every question (M43 committed
    those numbers) against M43's file, scope by scope; then every other shape on the fixture's TUNE half only (no
    held-out question is reranked at a new shape), with the depth and truncation invariants reported pass/fail and
    no metric computed. Nothing is written."""
    from app.eval.harness import CI_METRICS
    from app.eval.metrics import aggregate_retrieval
    from app.eval.rerank_arms import _paths, split_questions

    root = root or ROOT
    gold, manifest, _, fused, _, _ = _first_stage(root, embedder, 50)
    tune, held = split_questions(gold)
    parent = [_paths(pairs) for pairs in fused]
    count = count_tokens or pair_token_lengths
    resolve = resolver(checkpoint_dir, root)
    a_rows = grid_rows(anchors_only=True)
    a_orders, a_infos, a_loads = sr.rerank_rows(gold, fused, a_rows, resolve=resolve, load=load, device=device, parent=parent, check_cap=True, count_tokens=count)
    orders = {PARENT: parent, **a_orders}
    every = list(range(len(gold)))
    rows_of = lambda ids: [{"id": gold[i].id, "category": gold[i].category, "relevant": gold[i].relevant_ids()} for i in ids]  # noqa: E731
    scored = {n: aggregate_retrieval([{**r, "retrieved": lists[i]} for i, r in zip(every, rows_of(every))], n_files=manifest["n_files"]) for n, lists in orders.items()}
    scores = {n: {m: [float(q[m]) for q in s.per_query] for m in CI_METRICS} for n, s in scored.items()}
    blob: dict[str, Any] = {}
    names = list(orders)
    for scope, ids in zip(sr.SCOPES, (held, tune, every)):
        scope_rows, boot = sr.scope_rows(scores, names, ids, comparisons(names), CI_METRICS)
        for row in scope_rows:
            row["retrieval"] = aggregate_retrieval([{**r, "retrieved": orders[row["config"]][i]} for i, r in zip(ids, rows_of(ids))], n_files=manifest["n_files"]).as_dict()
        blob[scope] = {"bootstrap": boot, "results": scope_rows}
    anchors = fixture_anchors(root, blob)
    g_t, f_t, p_t = [gold[i] for i in tune], [fused[i] for i in tune], [parent[i] for i in tune]
    t_orders, t_infos, t_loads = sr.rerank_rows(g_t, f_t, grid_rows(skip_anchors=True), resolve=resolve, load=load, device=device, parent=p_t,
                                                check_cap=True, count_tokens=count)
    extra = {PARENT: {"truncated_pairs": None, "rank_hash": hash_columns(p_t)}}
    for name, _, _, _ in a_rows:
        extra[name] = {"truncated_pairs": [a_infos[name]["_truncated_pairs"][i] for i in tune], "rank_hash": hash_columns([a_orders[name][i] for i in tune])}
    for name, info in t_infos.items():
        extra[name] = {"truncated_pairs": info["_truncated_pairs"], "rank_hash": hash_columns(t_orders[name])}
    problems = invariant_problems("fixture", extra)
    same_files = [n for n, lists in {**{k: [v[i] for i in tune] for k, v in a_orders.items()}, **t_orders}.items()
                  if any(sorted(a) != sorted(b) for a, b in zip(lists, p_t))]
    problems += [f"{n}: a tune-half list does not hold exactly the parent's files" for n in same_files]
    loads = {**a_loads, **t_loads}
    return {"loads": {k: {"loaded_max_length": v.get("loaded_max_length"), "dtype": v.get("dtype"), "device": v.get("device"),
                          "load_and_score": (v.get("load_and_score") or {}).get("passed")} for k, v in loads.items()},
            "anchors": anchors, "invariants": {"questions": f"the tune half ({len(tune)} questions) at every non-anchor shape; the anchors' tune-half lists",
                                               "rows": len(extra) - 1, "passed": not problems, "problems": problems[:20]}}


# ---------------------------------------------------------------- invariants and the re-score (CPU, no model)


def invariant_problems(dataset: str, extra: dict[str, dict], scores: dict[str, dict[str, list[float]]] | None = None) -> list[str]:
    """The depth and truncation invariants, per query, from the per-query extras (and the scores, when given):
    - at depth <= 10 a reranker only permutes the parent's top 10: every query's top10_set hash (and recall@10) is the parent's;
    - at every depth the whole list holds the parent's documents: fixture recall@50, BEIR recall@100 equal the parent's;
    - fixture: no reranked pair is longer than 256 tokens, and every (d, 256) row ranks every question as its (d, 512) row;
    - BEIR: on every query where the (d, 256) row cut no pair, it ranks as the (d, 512) row (identical inputs)."""
    out: list[str] = []
    parent = extra.get(PARENT)
    whole = "recall@50" if dataset == "fixture" else "recall@100"
    for name, e in extra.items():
        cell = parse_row(name)
        if cell is None:
            continue
        tag, depth, cap = cell
        if parent is not None and depth <= 10:
            bad = sum(1 for a, b in zip(e["rank_hash"]["top10_set"], parent["rank_hash"]["top10_set"]) if a != b)
            if bad:
                out.append(f"{name}: {bad} queries whose top-10 set differs from the parent's at depth {depth}")
        if scores is not None and PARENT in scores and name in scores:
            for metric in (["recall@10"] if depth <= 10 else []) + [whole]:
                bad = sum(1 for a, b in zip(scores[name][metric], scores[PARENT][metric]) if a != b)
                if bad:
                    out.append(f"{name}: {bad} queries whose {metric} differs from the parent's")
        if dataset == "fixture" and any(e.get("truncated_pairs") or []):
            out.append(f"{name}: {sum(e['truncated_pairs'])} reranked fixture pairs longer than {cap} tokens")
        if cap == 256:
            twin = extra.get(row_name(tag, depth, 512))
            if twin is None:
                continue
            counts = e.get("truncated_pairs") or [0] * len(e["rank_hash"]["ranking"])
            bad = sum(1 for a, b, c in zip(e["rank_hash"]["ranking"], twin["rank_hash"]["ranking"], counts) if (dataset == "fixture" or not c) and a != b)
            if bad:
                out.append(f"{name}: {bad} queries {'' if dataset == 'fixture' else 'with no pair cut '}ranked differently from {row_name(tag, depth, 512)}")
    return out


def extra_differences(dataset: str, blob: dict, sidecar: dict) -> list[str]:
    """Every per-row total of a shapes file (ranking_digest, n_top10_differs_from_parent, truncation) that its
    sidecar's per-query extras do not re-derive."""
    extra = sidecar.get("per_query_extra")
    if not extra:
        return ["no per_query_extra in the sidecar"]
    out = []
    stored = blob["rerankers"] if dataset == "fixture" else {r["config"]: r for r in blob["results"]}
    for name, row in stored.items():
        if name not in extra:
            out.append(f"{name}: no per-query extras")
            continue
        again = derived(extra[name], extra.get(PARENT) if name != PARENT else None)
        for key, value in again.items():
            if row.get(key) != value:
                out.append(f"{name}.{key}: results file {row.get(key)!r}, from the sidecar {value!r}")
    return out


def _load_pair(root: Path, rel: str) -> tuple[dict, dict] | None:
    from app.eval import per_question

    path = root / rel
    if not path.exists():
        return None
    return json.loads(path.read_text()), json.loads(per_question.sidecar_path(path).read_text())


PROVENANCE_KEYS = ("timestamp", "git_sha", "git_head", "working_tree_dirty", "command", "source_sha256")


def rescore(root: Path | None = None, files: dict[str, str] | None = None, *, with_frontier: bool = True) -> tuple[int, list[str]]:
    """(files checked, differences): every interval of the shapes files from their sidecars (M43's rescore_fixture /
    rescore_beir), every per-row total from the per-query extras, the invariants, and the frontier recomputed and
    compared field by field except its run-provenance header."""
    from app.eval.compare import _leaf_differences

    root = root or ROOT
    checked, problems = 0, []
    for dataset, rel in (files or RESULTS).items():
        pair = _load_pair(root, rel)
        if pair is None:
            continue
        blob, side = pair
        diffs = sr.rescore_fixture(blob, side) if dataset == "fixture" else sr.rescore_beir(blob, side)
        diffs += extra_differences(dataset, blob, side)
        diffs += invariant_problems(dataset, side.get("per_query_extra") or {}, side["scores"])
        problems += [f"{rel}: {d}" for d in diffs]
        checked += 1
    if with_frontier and (root / FRONTIER_FILE).exists():
        stored = json.loads((root / FRONTIER_FILE).read_text())
        again = frontier(root)
        out: list[str] = []
        _leaf_differences("frontier", {k: v for k, v in stored.items() if k not in PROVENANCE_KEYS}, {k: v for k, v in again.items() if k not in PROVENANCE_KEYS}, out)
        problems += [f"{FRONTIER_FILE}: {d}" for d in out]
        checked += 1
    if files is None and with_frontier:  # Phase 2's files (anchored on the shapes files) and its frontier, once committed
        n, more = rescore(root, ML256_RESULTS, with_frontier=False)
        checked, problems = checked + n, problems + more
        if (root / ML256_FRONTIER_FILE).exists():
            stored = json.loads((root / ML256_FRONTIER_FILE).read_text())
            again = ml256_frontier(root)
            out = []
            _leaf_differences("ml256_frontier", {k: v for k, v in stored.items() if k not in PROVENANCE_KEYS}, {k: v for k, v in again.items() if k not in PROVENANCE_KEYS}, out)
            problems += [f"{ML256_FRONTIER_FILE}: {d}" for d in out]
            checked += 1
    return checked, problems


# ---------------------------------------------------------------- the frontier: cost x quality per cell, and the answers


def sign_flip_p(row: list[float], parent: list[float], draws: int = 20_000, seed: int = 0) -> float:
    """Two-sided paired sign-flip permutation p-value of the mean difference (descriptive)."""
    import numpy as np

    d = np.asarray(row, dtype=float) - np.asarray(parent, dtype=float)
    observed = abs(d.mean())
    signs = np.random.RandomState(seed).randint(0, 2, size=(draws, d.shape[0])) * 2 - 1
    flipped = np.abs((signs * d).mean(axis=1))
    return round(float((1 + np.sum(flipped >= observed - 1e-12)) / (1 + draws)), 5)


def _file_delta(dataset: str, blob: dict, name: str, reference: str) -> dict | None:
    """A row's stored nDCG@10 delta against `reference` (fixture: its held-out scope)."""
    if dataset == "fixture":
        row = {r["config"]: r for r in blob["held_out"]["results"]}.get(name)
        return (_delta_block(row, reference) or {}).get(METRIC) if row else None
    row = {r["config"]: r for r in blob["results"]}.get(name)
    return sr._delta_to(row, reference, blob.get("bootstrap")) if row else None


def frontier(root: Path | None = None, files: dict[str, str] | None = None) -> dict[str, Any]:
    """The frontier file's body: per cell its band and indicative cost, per dataset its delta against hybrid@bge-small
    (95%, verdict), its adjusted interval in each family it belongs to, its delta against 20/512 and (a student) against
    its init, its per-row totals and the sensitivity checks; the F1 and F2 answers; the anchors; what it does not show."""
    from datetime import datetime, timezone

    from app.eval.provenance import command_line, git_sha, git_state, run_provenance, sha256_file
    from app.eval.stats import DEFAULT_N_BOOT, DEFAULT_SEED, paired_delta_ci, resample_indices

    root = root or ROOT
    files = files or RESULTS
    started = git_state(root)
    pairs = {d: _load_pair(root, rel) for d, rel in files.items()}
    missing = [files[d] for d, p in pairs.items() if p is None]
    if missing:
        raise FileNotFoundError(f"the frontier needs all three shapes files; missing: {missing}")
    cpu, limit = load_cpu(root), memory_limit_mib(root)
    membership = bands(root, cpu, limit)
    exposure_blob = json.loads((root / EXPOSURE_FILE).read_text())
    families = {f: family_cells(f, root, membership) for f in ("F1", "F2")}
    columns, idx = {}, {}
    for dataset, (blob, side) in pairs.items():
        subset = side["split"]["held_out"] if dataset == "fixture" else list(range(len(side["ids"])))
        columns[dataset] = (side, subset)
        idx[dataset] = resample_indices(len(subset), n_boot=ADJUSTED["n_boot"], seed=ADJUSTED["seed"])
    col = lambda dataset, name: [columns[dataset][0]["scores"][name][METRIC][i] for i in columns[dataset][1]]  # noqa: E731
    cells_out: list[dict[str, Any]] = []
    by_family: dict[str, dict[str, dict]] = {"F1": {}, "F2": {}}
    for tag, depth, cap in GRID:
        name = row_name(tag, depth, cap)
        band = band_of(tag, depth, cap, cpu, limit)
        cell: dict[str, Any] = {"config": name, "tag": tag, "depth": depth, "max_length": cap, "band": band,
                                "indicative_cpu": cpu_cost(tag, depth, cap, cpu),
                                "families": [f for f in ("F1", "F2") if (tag, depth, cap) in families[f]], "datasets": {}}
        for dataset, (blob, side) in pairs.items():
            vs = _file_delta(dataset, blob, name, PARENT)
            entry: dict[str, Any] = {"vs_parent": vs, "verdict": verdict(vs)}
            if cell["families"]:
                row, parent = col(dataset, name), col(dataset, PARENT)
                entry["adjusted"] = {f: adjusted_delta(row, parent, FAMILIES[f]["k"], idx[dataset]) for f in cell["families"]}
                if "F1" in cell["families"]:
                    entry["sign_flip_p"] = sign_flip_p(row, parent, SENSITIVITY["permutation"]["draws"], SENSITIVITY["permutation"]["seed"])
                if tag in sr.STUDENTS and dataset != "fixture":
                    exposed = set(exposure_blob["datasets"][dataset]["exposed_query_ids"])
                    keep = [i for i, q in enumerate(side["ids"]) if q not in exposed]
                    r2 = [side["scores"][name][METRIC][i] for i in keep]
                    p2 = [side["scores"][PARENT][METRIC][i] for i in keep]
                    d, lo, hi = paired_delta_ci(r2, p2, resample_indices(len(keep), n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED))
                    entry["exposure_excluded"] = {"n": len(keep), "delta": round(d, 4), "lo": round(lo, 4), "hi": round(hi, 4)}
            if (depth, cap) != ANCHOR:
                entry["vs_20_512"] = _file_delta(dataset, blob, name, row_name(tag, *ANCHOR))
            if tag in sr.STUDENTS:
                entry["vs_init"] = _file_delta(dataset, blob, name, row_name(sr.STUDENTS[tag], depth, cap))
            stored = blob["rerankers"].get(name, {}) if dataset == "fixture" else {r["config"]: r for r in blob["results"]}.get(name, {})
            entry.update({k: stored.get(k) for k in ("truncated_pairs", "n_top10_differs_from_parent", "ranking_digest")})
            if cap == 256:
                mine, twin = side["per_query_extra"][name], side["per_query_extra"][row_name(tag, depth, 512)]
                entry["n_queries_ranked_unlike_512"] = sum(1 for a, b in zip(mine["rank_hash"]["ranking"], twin["rank_hash"]["ranking"]) if a != b)
            cell["datasets"][dataset] = entry
            for f in cell["families"]:
                by_family[f].setdefault(name, {})[dataset] = {"ci95": vs, "adjusted": entry["adjusted"][f]}
        cell["passes_at_95"] = all(cell["datasets"][d]["verdict"] == "win" for d in DATASETS)
        cells_out.append(cell)
    answers = {f: {**family_answer(by_family[f]), "k": FAMILIES[f]["k"], "percentiles": [round(x, 4) for x in adjusted_percentiles(FAMILIES[f]["k"])]}
               for f in ("F1", "F2")}
    anchor_diffs = {}
    for dataset, (blob, side) in pairs.items():
        m43 = _load_pair(root, sr.RESULTS[dataset])
        anchor_diffs[dataset] = sum(1 for n in anchor_rows() for m in side["metrics"] for a, b in zip(side["scores"][n][m], m43[1]["scores"][n][m]) if a != b)
    inputs = [*files.values(), *(_sidecar_rel(r) for r in files.values()), CPU_FILE, LIMIT_FILE, EXPOSURE_FILE, *(sr.RESULTS[d] for d in DATASETS)]
    return {"git_sha": git_sha(root), **run_provenance(root=root, started=started), "timestamp": datetime.now(timezone.utc).isoformat(),
            "command": command_line(), "label": "M51: quality x indicative CPU cost per cell, and the answer per family (CPU, from committed files only)",
            "inputs_sha256": {rel: sha256_file(root / rel) for rel in inputs},
            "devices": {d: (b.get("device") or {}).get("device_name") for d, (b, _) in pairs.items()},
            "limit_mib": limit, "bands": {b: [row_name(*c) for c in cs] for b, cs in membership.items()},
            "settings": {"ci95": "the stored 95% intervals of the three files (2000 resamples, seed 0)",
                         "adjusted": {**ADJUSTED, "families": {f: {"k": FAMILIES[f]["k"], "alpha": adjusted_alpha(FAMILIES[f]["k"]),
                                                                   "percentiles": [round(x, 4) for x in adjusted_percentiles(FAMILIES[f]["k"])]} for f in ("F1", "F2")}},
                         "sensitivity": SENSITIVITY, "exposure_excluded_interval": "95%, 2000 resamples, seed 0, over the unexposed queries"},
            "answers": answers, "headline": {"family": "F1", "answer": answers["F1"]["answer"], "cells": answers["F1"]["cells"]},
            "family_control": FAMILY_CONTROL, "cells": cells_out,
            "anchors": {d: (b.get("anchors") or {}) for d, (b, _) in pairs.items()},
            "anchor_per_query_differences_vs_m43": anchor_diffs,
            "host_commands": host_commands([c for c in cells_out if "F1" in c["families"] or (answers["F2"]["answer"] == "YES" and "F2" in c["families"])]),
            "does_not_show": DOES_NOT_SHOW_FRONTIER}


def _sidecar_rel(rel: str) -> str:
    return rel[: -len(".json")] + ".per_query.json"


def host_commands(cells: list[dict]) -> list[str]:
    """The owner's host timing, printed and never run: a student is timed by its init's hub id (same cost)."""
    out = []
    for c in cells:
        tag = sr.STUDENTS.get(c["tag"], c["tag"])
        out.append(f"python scripts/time_reranker_on_host.py --model {sr.model_of(tag)} --depths {c['depth']} --max-length {c['max_length']} --stop-api")
    return sorted(set(out), key=out.index)


DOES_NOT_SHOW_FRONTIER = [
    "host timing and memory (the serving bar, unrun): the t3.small's 2 vCPUs are two hyperthreads of one core and it runs in standard credit mode, so it can be well over twice as slow as the HPC CPU; a band is an indicative HPC-CPU budget, never a statement that a shape fits the host",
    "resident memory on the host: the CPU file ran the torch 2.5.1+cu121 wheel; the image installs the CPU wheel",
    "concurrency: one request at a time",
    "typical latency: the CPU timings cut every passage at the cap (600-word passages), the worst case per shape",
    "the cap on the fixture: no fixture pair exceeds 256 tokens; only SciFact and FiQA say anything about the cap",
    "absence of an effect: NOT SHOWN is not evidence that the cheap shapes do not help",
    "students at 256 as trained: both attempt-1 students were trained at max_length 512 (Phase 2 measures this)",
    "equivalence: a delta against 20/512 that covers zero does not show the cheap shape is as good",
    "the other M43 candidates (mxbai-xsmall-v1, mxbai-base-v1, bge-reranker-base): over 1,024 MiB at every shape",
    "quantised or ONNX serving, fp16 on CPU, or another first stage; the real corpus; the final context (RERANK_TOP_N); answer quality",
    "variance across training seeds: one inference run, one training run per student",
    "a student YES on an unseen corpus: the students read documents of all three test corpora (never a query or a label)",
    "a fresh search: M51's question follows M43's failed 20/512 search over 8 small rows",
    "the publishers' (and the teacher's) training data: not audited for overlap with the BEIR test sets",
]


def write_frontier(root: Path | None = None) -> dict:
    root = root or ROOT
    blob = frontier(root)
    (root / FRONTIER_FILE).write_text(json.dumps(blob, indent=2) + "\n")
    return blob


def ml256_frontier(root: Path | None = None) -> dict[str, Any]:
    """Phase 2's frontier: per attempt-2 cell (the two students retrained at 256, at depth 5/10/20 x 256) its cost (the
    same-architecture attempt-1 row's: same architecture, tokenizer and cap), its band from that cost, per dataset its
    delta against hybrid@bge-small (95%, verdict) and, in F3, at the Bonferroni level k = 3 with the sign-flip p-value;
    its delta against the attempt-1 student and against the published init at the same shape; the F3 answer."""
    from datetime import datetime, timezone

    from app.eval.provenance import command_line, git_sha, git_state, run_provenance, sha256_file
    from app.eval.stats import DEFAULT_N_BOOT, DEFAULT_SEED, paired_delta_ci, resample_indices

    root = root or ROOT
    started = git_state(root)
    pairs = {d: _load_pair(root, rel) for d, rel in ML256_RESULTS.items()}
    missing = [ML256_RESULTS[d] for d, p in pairs.items() if p is None]
    if missing:
        raise FileNotFoundError(f"the Phase-2 frontier needs all three Phase-2 files; missing: {missing}")
    cpu, limit = load_cpu(root), memory_limit_mib(root)
    exposure_blob = json.loads((root / EXPOSURE_FILE).read_text())
    family = set(family_cells("F3"))
    k = FAMILIES["F3"]["k"]
    columns, idx = {}, {}
    for dataset, (blob, side) in pairs.items():
        subset = side["split"]["held_out"] if dataset == "fixture" else list(range(len(side["ids"])))
        columns[dataset] = (side, subset)
        idx[dataset] = resample_indices(len(subset), n_boot=ADJUSTED["n_boot"], seed=ADJUSTED["seed"])
    col = lambda dataset, name: [columns[dataset][0]["scores"][name][METRIC][i] for i in columns[dataset][1]]  # noqa: E731
    cells_out: list[dict[str, Any]] = []
    by_family: dict[str, dict] = {}
    for tag, twin in PHASE2["students"].items():
        for depth, cap in PHASE2["shapes"]:
            name = row_name(tag, depth, cap)
            cell: dict[str, Any] = {"config": name, "tag": tag, "attempt": PHASE2["attempt"], "attempt_1_twin": row_name(twin, depth, cap),
                                    "depth": depth, "max_length": cap, "band": band_of(twin, depth, cap, cpu, limit),
                                    "indicative_cpu": cpu_cost(twin, depth, cap, cpu), "cost_source": PHASE2["cost_source"],
                                    "families": ["F3"] if (tag, depth, cap) in family else [], "datasets": {}}
            for dataset, (blob, side) in pairs.items():
                vs = _file_delta(dataset, blob, name, PARENT)
                entry: dict[str, Any] = {"vs_parent": vs, "verdict": verdict(vs)}
                if cell["families"]:
                    row, parent = col(dataset, name), col(dataset, PARENT)
                    entry["adjusted"] = {"F3": adjusted_delta(row, parent, k, idx[dataset])}
                    entry["sign_flip_p"] = sign_flip_p(row, parent, SENSITIVITY["permutation"]["draws"], SENSITIVITY["permutation"]["seed"])
                    if dataset != "fixture":
                        exposed = set(exposure_blob["datasets"][dataset]["exposed_query_ids"])
                        keep = [i for i, q in enumerate(side["ids"]) if q not in exposed]
                        r2 = [side["scores"][name][METRIC][i] for i in keep]
                        p2 = [side["scores"][PARENT][METRIC][i] for i in keep]
                        d, lo, hi = paired_delta_ci(r2, p2, resample_indices(len(keep), n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED))
                        entry["exposure_excluded"] = {"n": len(keep), "delta": round(d, 4), "lo": round(lo, 4), "hi": round(hi, 4)}
                    by_family.setdefault(name, {})[dataset] = {"ci95": vs, "adjusted": entry["adjusted"]["F3"]}
                entry["vs_attempt_1"] = _file_delta(dataset, blob, name, row_name(twin, depth, cap))
                entry["vs_init"] = _file_delta(dataset, blob, name, row_name(sr.STUDENTS[twin], depth, cap))
                stored = blob["rerankers"].get(name, {}) if dataset == "fixture" else {r["config"]: r for r in blob["results"]}.get(name, {})
                entry.update({key: stored.get(key) for key in ("truncated_pairs", "n_top10_differs_from_parent", "ranking_digest")})
                cell["datasets"][dataset] = entry
            cell["passes_at_95"] = all(cell["datasets"][d]["verdict"] == "win" for d in DATASETS)
            cells_out.append(cell)
    answer = {**family_answer(by_family), "k": k, "percentiles": [round(x, 4) for x in adjusted_percentiles(k)]}
    anchor_diffs = {}
    for dataset, (blob, side) in pairs.items():
        m51 = _load_pair(root, RESULTS[dataset])
        anchor_diffs[dataset] = sum(1 for n in ml256_anchor_rows() for m in side["metrics"] for a, b in zip(side["scores"][n][m], m51[1]["scores"][n][m]) if a != b)
    inputs = [*ML256_RESULTS.values(), *(_sidecar_rel(r) for r in ML256_RESULTS.values()), CPU_FILE, LIMIT_FILE, EXPOSURE_FILE, PHASE2["manifest"],
              *(RESULTS[d] for d in DATASETS)]
    return {"git_sha": git_sha(root), **run_provenance(root=root, started=started), "timestamp": datetime.now(timezone.utc).isoformat(),
            "command": command_line(), "label": f"M51 Phase 2 ({PHASE2['label']}): quality x indicative CPU cost per cell, and the F3 answer (CPU, from committed files only)",
            "inputs_sha256": {rel: sha256_file(root / rel) for rel in inputs},
            "devices": {d: (b.get("device") or {}).get("device_name") for d, (b, _) in pairs.items()},
            "limit_mib": limit, "family": {"F3": {**{key: v for key, v in FAMILIES["F3"].items() if key != "cells"},
                                                  "cells": [row_name(*c) for c in FAMILIES["F3"]["cells"]], "alpha": adjusted_alpha(k)}},
            "settings": {"ci95": "the stored 95% intervals of the three files (2000 resamples, seed 0)", "adjusted": ADJUSTED,
                         "sensitivity": SENSITIVITY, "exposure_excluded_interval": "95%, 2000 resamples, seed 0, over the unexposed queries"},
            "answers": {"F3": answer}, "family_control": "F3 is its own family, never merged into F1 or F2; it does not modify the headline",
            "cells": cells_out, "anchors": {d: (b.get("anchors") or {}) for d, (b, _) in pairs.items()},
            "anchor_per_query_differences_vs_m51": anchor_diffs,
            "does_not_show": DOES_NOT_SHOW_FRONTIER[:6] + DOES_NOT_SHOW["phase2"] + DOES_NOT_SHOW_FRONTIER[7:]}


def write_ml256_frontier(root: Path | None = None) -> dict:
    root = root or ROOT
    blob = ml256_frontier(root)
    (root / ML256_FRONTIER_FILE).write_text(json.dumps(blob, indent=2) + "\n")
    return blob


def _fmt(block: dict | None) -> str:
    return f"{block['delta']:+.4f} [{block['lo']:+.4f}, {block['hi']:+.4f}]" if block else "-"


def bar_lines(blob: dict) -> list[str]:
    out = []
    for c in blob["cells"]:
        per = "; ".join(f"{d} {_fmt(e['vs_parent'])} {e['verdict']}" + (" adj " + ", ".join(f"{f} {_fmt(a)}" for f, a in e["adjusted"].items()) if e.get("adjusted") else "")
                        for d, e in c["datasets"].items())
        out.append(f"{'PASS' if c['passes_at_95'] else 'no  '} {c['band']:24s} {c['config']:58s} {per}")
    for f, a in blob["answers"].items():
        out.append(f"{f} (k = {a['k']}): {a['answer']}" + (f": {', '.join(a['cells'])}" if a["cells"] and a["answer"] != "NO" else ""))
    return out


# ---------------------------------------------------------------- the replicate on the other device (make small-rerankers-shapes-replicate)


def _scope_fields(blob: dict, scope: str, name: str) -> dict | None:
    row = {r["config"]: r for r in blob[scope]["results"]}.get(name)
    return {k: row.get(k) for k in ("retrieval", "ci", "deltas")} if row else None


_ABSENT_A, _ABSENT_B = object(), object()  # a value one side lacks never equals the other side's (or its own absence)


def _column_differences(ids_a: list, col_a: list, ids_b: list, col_b: list) -> int:
    """How many per-query values of two columns differ, matched by query id: a query on one side only, a column shorter
    than its ids, or a value past the end of its ids counts as a difference (a zip would count them as equal)."""
    a = {q: col_a[i] if i < len(col_a) else _ABSENT_A for i, q in enumerate(ids_a)}
    b = {q: col_b[i] if i < len(col_b) else _ABSENT_B for i, q in enumerate(ids_b)}
    extra = max(0, len(col_a) - len(ids_a)) + max(0, len(col_b) - len(ids_b))
    return extra + sum(1 for q in [*a, *(q for q in b if q not in a)] if a.get(q, _ABSENT_A) != b.get(q, _ABSENT_B))


def _score_column(sidecar: dict, name: str, metric: str) -> list:
    return (sidecar["scores"].get(name) or {}).get(metric) or []


def _rank_column(sidecar: dict, name: str) -> list:
    return (((sidecar.get("per_query_extra") or {}).get(name) or {}).get("rank_hash") or {}).get("ranking") or []


def replicate(root: Path | None = None) -> dict[str, Any]:
    """The replicate's files against the result's, per dataset and row: whether every retrieval field, ci and delta is
    equal; how many per-query metric values differ; how many per-query ranking hashes differ. Values are matched by
    query id, and a row, metric or query that one side lacks counts as differing, never as equal."""
    from datetime import datetime, timezone

    from app.eval.provenance import command_line, git_sha, git_state, run_provenance, sha256_file

    root = root or ROOT
    started = git_state(root)
    out: dict[str, Any] = {}
    for dataset, rel in RESULTS.items():
        a, b = _load_pair(root, rel), _load_pair(root, replicate_path(rel))
        if a is None or b is None:
            out[dataset] = {"compared": False, "reason": f"{rel if a is None else replicate_path(rel)} is missing"}
            continue
        (ra, sa), (rb, sb) = a, b
        rows = {}
        ids_a, ids_b = sa.get("ids") or [], sb.get("ids") or []
        for name in [*sa["scores"], *(n for n in sb["scores"] if n not in sa["scores"])]:
            if dataset == "fixture":
                equal = all(_scope_fields(ra, scope, name) == _scope_fields(rb, scope, name) for scope in sr.SCOPES)
            else:
                x = {r["config"]: r for r in ra["results"]}.get(name, {})
                y = {r["config"]: r for r in rb["results"]}.get(name, {})
                equal = all(x.get(k) == y.get(k) for k in ("retrieval", "ci", "delta_vs_parent"))
                equal = equal and [c for c in ra["bootstrap"].get("comparisons") or [] if c["row"] == name] == [c for c in rb["bootstrap"].get("comparisons") or [] if c["row"] == name]
            equal = equal and name in sa["scores"] and name in sb["scores"]
            values = sum(_column_differences(ids_a, _score_column(sa, name, m), ids_b, _score_column(sb, name, m))
                         for m in [*sa["metrics"], *(m for m in sb["metrics"] if m not in sa["metrics"])])
            hashes = _column_differences(ids_a, _rank_column(sa, name), ids_b, _rank_column(sb, name))
            rows[name] = {"fields_equal": equal, "per_query_metric_values_differ": values, "ranking_hashes_differ": hashes}
        out[dataset] = {"compared": True, "result_device": (ra.get("device") or {}).get("device_name"), "replicate_device": (rb.get("device") or {}).get("device_name"),
                        "result_git_sha": ra.get("git_sha"), "replicate_git_sha": rb.get("git_sha"),
                        "rows_fields_equal": sum(1 for r in rows.values() if r["fields_equal"]), "rows": rows}
    inputs = [p for rel in RESULTS.values() for p in (rel, _sidecar_rel(rel), replicate_path(rel), _sidecar_rel(replicate_path(rel))) if (root / p).exists()]
    return {"git_sha": git_sha(root), **run_provenance(root=root, started=started), "timestamp": datetime.now(timezone.utc).isoformat(), "command": command_line(),
            "label": "M51 replicate: the same commit on the other device class, compared with the result; never a result row",
            "inputs_sha256": {rel: sha256_file(root / rel) for rel in inputs}, "datasets": out,
            "does_not_show": ["which device is right: a difference says the ranking is device-sensitive at near-ties, not which run is correct",
                              "anything about the host: both devices are GPUs"]}


# ---------------------------------------------------------------- CLI


def _parse(argv: list[str] | None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mde", action="store_true", help="print the design-time half-widths from M43's committed sidecars (CPU, seconds)")
    p.add_argument("--exposure", action="store_true", help=f"write {EXPOSURE_FILE} (CPU; needs the BEIR data directory)")
    p.add_argument("--data-dir", type=Path, default=ROOT / "bench" / "external")
    p.add_argument("--check", action="store_true", help="load every model at both caps, read the cap and dtype back, run the load-and-score check")
    p.add_argument("--out", type=Path, default=None, help="run fixture v1 at every shape and write this results file with its sidecar")
    p.add_argument("--smoke", action="store_true", help="the anchors on fixture v1 and the tune-half invariants at every shape; writes nothing")
    p.add_argument("--anchors", nargs="*", type=Path, default=None, help="check the anchors of these shapes results files (default: the committed three)")
    p.add_argument("--frontier", action="store_true", help=f"write {FRONTIER_FILE} from the three committed files (CPU)")
    p.add_argument("--rescore", action="store_true", help="recompute every interval, per-row total, invariant and the frontier from the sidecars (CPU)")
    p.add_argument("--bar", action="store_true", help="print the 36 cells and the family answers from the frontier file")
    p.add_argument("--replicate", action="store_true", help=f"compare the replicate's files with the result's -> {REPLICATE_FILE}")
    p.add_argument("--models", default=None, help="comma-separated tags (default: all six)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--checkpoint-dir", type=Path, default=None, help="where the students' checkpoints are (default: $SMALL_RERANKER_CHECKPOINTS)")
    p.add_argument("--phase2", action="store_true", help="Phase 2 (attempt 2): --check, --out, --anchors, --frontier and --bar act on its rows and files")
    p.add_argument("--ml256-checkpoint-dir", type=Path, default=None, help="Phase 2: where the attempt-2 checkpoints are (default: $SMALL_RERANKER_ML256_CHECKPOINTS)")
    return p.parse_args(argv)


def check_models(tags: list[str], *, checkpoint_dir: Path | str | None = None, device: str = "auto", caps: tuple[int, ...] = MAX_LENGTHS,
                 ml256_dir: Path | str | None = None) -> list[dict[str, Any]]:
    """make small-rerankers-shapes-check: each model at both caps, loaded as serving loads it, its pin (a candidate's
    cached revision, bge's BGE_REVISION, a student's sha256 and reference logits) checked, its loaded_max_length and
    dtype read back, and M41's load-and-score check run. One record per (model, cap); nothing falls back. Phase 2:
    its rows' tags at 256 only, the attempt-2 students resolved from `ml256_dir`."""
    from app.eval.models_2026 import check_reranker, free_gpu
    from app.eval.provenance import model_device, scrub
    from app.retrieval.rerank import cross_encoder_provenance, get_cross_encoder

    resolve = resolver(checkpoint_dir, None, ml256_dir)
    out = []
    for tag in tags:
        for cap in caps:
            line: dict[str, Any] = {"tag": tag, "max_length": cap, "precision": PRECISION}
            try:
                source, record, after_load = resolve(tag)
                line.update({k: v for k, v in record.items() if k in ("pinned_revision", "weights_sha256", "attempt")})
                m = get_cross_encoder(source, precision=PRECISION, max_length=cap, device=None if device == "auto" else device, raise_errors=True)
                prov = cross_encoder_provenance(m)
                line.update(device=model_device(m), dtype=prov["dtype"], loaded_max_length=prov["loaded_max_length"], default_prompt_name=prov["default_prompt_name"])
                if prov["loaded_max_length"] != cap:
                    raise RuntimeError(f"loaded_max_length reads back {prov['loaded_max_length']!r}, not {cap}")
                line["check"] = check_reranker(m, tag)
                if after_load:
                    line.update(after_load(m))
                del m
                line["ok"] = True
            except Exception as e:  # noqa: BLE001 - reported per (model, cap), then the exit status says it failed
                line.update(ok=False, error=scrub(f"{type(e).__name__}: {e}"))
            free_gpu()
            out.append(line)
    return out


def anchor_report(paths: list[Path], root: Path | None = None, phase2: bool = False) -> tuple[bool, list[str]]:
    """Before the results commit: each shapes file's anchors (N_ANCHOR_ROWS rows; a metric difference fails), and how
    many of its anchor rows' per-query values differ from M43's sidecars (expected 0). Phase 2: its 13 anchor rows
    against M51's committed shapes files and sidecars."""
    from app.eval import per_question

    root = root or ROOT
    ok, lines = True, []
    for path in paths:
        blob = json.loads(Path(path).read_text())
        side = json.loads(per_question.sidecar_path(Path(path)).read_text())
        dataset = "fixture" if "held_out" in blob else blob["dataset"].split("/")[-1]
        a = blob.get("anchors") or {}
        ref = _load_pair(root, (RESULTS if phase2 else sr.RESULTS)[dataset])[1]
        names = ml256_anchor_rows() if phase2 else anchor_rows()
        values = sum(1 for n in names if n in side["scores"] for m in side["metrics"] for x, y in zip(side["scores"][n][m], ref["scores"][n][m]) if x != y)
        if dataset == "fixture":
            good = bool(a.get("all"))
            detail = {s: (v["n_compared"], v["all"]) for s, v in a.get("scopes", {}).items()}
        else:
            good = bool(a.get("metrics_equal"))
            detail = {"n_compared": a.get("n_compared"), "metric_differences": a.get("metric_differences"), "ranking_differences": a.get("ranking_differences")}
        ok = ok and good and values == 0
        lines.append(f"{dataset}: anchors {'OK' if good else 'DIFFER'} {json.dumps(detail)}; anchor per-query values differing from {'M51' if phase2 else 'M43'}'s sidecar: {values}; "
                     f"device {(blob.get('device') or {}).get('device_name')}; git_sha {blob.get('git_sha')} dirty {blob.get('working_tree_dirty')}")
    return ok, lines


def main(argv: list[str] | None = None) -> int:
    import os

    args = _parse(argv)
    root = ROOT
    if args.mde:
        print(mde_table(design_half_widths(root)))
        return 0
    if args.exposure:
        blob = write_exposure(root, args.data_dir)
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
    if args.rescore:
        checked, problems = rescore(root)
        for line in problems[:60]:
            print(f"  - {line}")
        print(f"small-rerankers-shapes: {'OK' if not problems else f'{len(problems)} differences'}, {checked} files recomputed from the per-query sidecars")
        return 1 if problems or not checked else 0
    if args.frontier:
        blob = write_ml256_frontier(root) if args.phase2 else write_frontier(root)
        print("\n".join(bar_lines(blob)))
        print(f"wrote {ML256_FRONTIER_FILE if args.phase2 else FRONTIER_FILE}")
        return 0
    if args.bar:
        path = root / (ML256_FRONTIER_FILE if args.phase2 else FRONTIER_FILE)
        if not path.exists():
            print(f"{path.relative_to(root)} is missing: run make small-rerankers-{'ml256' if args.phase2 else 'shapes'}-frontier", file=sys.stderr)
            return 1
        print("\n".join(bar_lines(json.loads(path.read_text()))))
        return 0
    if args.replicate:
        blob = replicate(root)
        (root / REPLICATE_FILE).write_text(json.dumps(blob, indent=2) + "\n")
        for dataset, e in blob["datasets"].items():
            print(f"{dataset}: " + (f"{e['rows_fields_equal']} of {len(e['rows'])} rows equal in every field ({e['result_device']} vs {e['replicate_device']}); "
                                    f"per-query values differing {sum(r['per_query_metric_values_differ'] for r in e['rows'].values())}, "
                                    f"ranking hashes differing {sum(r['ranking_hashes_differ'] for r in e['rows'].values())}" if e["compared"] else e["reason"]))
        print(f"wrote {REPLICATE_FILE}")
        return 0
    if args.anchors is not None:
        ok, lines = anchor_report(args.anchors or [root / rel for rel in (ML256_RESULTS if args.phase2 else RESULTS).values()], root, phase2=args.phase2)
        print("\n".join(lines))
        return 0 if ok else 1
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    tags = [t.strip() for t in args.models.split(",") if t.strip()] if args.models else list(MODELS)
    if args.smoke:
        result = smoke(root, checkpoint_dir=args.checkpoint_dir, device=args.device)
        print(json.dumps(result, indent=1))
        good = result["anchors"]["all"] and result["invariants"]["passed"] and all(v["load_and_score"] for v in result["loads"].values())
        print(f"smoke: anchors {'OK' if result['anchors']['all'] else 'DIFFER'}; invariants {'PASS' if result['invariants']['passed'] else 'FAIL'}")
        return 0 if good else 1
    if args.out is not None:
        from app.eval import per_question

        blob, sidecar = run_fixture(root, tags=tags, checkpoint_dir=args.checkpoint_dir, device=args.device, phase2=args.phase2, ml256_dir=args.ml256_checkpoint_dir)
        out = args.out if args.out.is_absolute() else root / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(blob, indent=2) + "\n")
        per_question.write(per_question.sidecar_path(out), sidecar)
        print("anchors", json.dumps({s: (v["n_compared"], v["all"]) for s, v in blob["anchors"]["scopes"].items()}))
        print("reproduces", json.dumps(blob["reproduces"]))
        print(f"wrote {out}")
        return 0
    if args.check:
        if args.phase2:
            lines = check_models([*ML256_ANCHOR_TAGS, *PHASE2["students"]], checkpoint_dir=args.checkpoint_dir, device=args.device,
                                 caps=(PHASE2["recipe_delta"]["max_length"],), ml256_dir=args.ml256_checkpoint_dir)
        else:
            lines = check_models(tags, checkpoint_dir=args.checkpoint_dir, device=args.device)
        for line in lines:
            print(json.dumps(line), flush=True)
        return 0 if all(line["ok"] for line in lines) else 1
    print("nothing to do: pass --mde, --exposure, --check, --smoke, --out, --anchors, --frontier, --rescore, --bar or --replicate", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
