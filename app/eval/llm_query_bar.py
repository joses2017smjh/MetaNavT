"""M46, part (c): the bootstrap wrapper, the verdict word, `passes`, the replication label, the placement decision and
its sentences, and the re-score of every interval from the per-query sidecars.

    python -m app.eval.llm_query_bar --rescore   # make llm-query-arms-rescore: CPU, no model, no dataset
    python -m app.eval.llm_query_bar --bar       # make llm-query-arms-bar: the bar, the decision, from the committed files

Byte-identical from the pre-probe commit D0 to the declaration commit D1, frozen after it (llm_query_spec.py says why).
One definition of the verdict: from the RAW bounds of the paired nDCG@10 interval, stored in the comparison block as
`verdict`, and every reader (the README block, the evidence rows, --bar, the PLAN reader) uses the stored word.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from app.eval import llm_query_spec as spec

PRIMARY = "ndcg@10"
SCHEMA = "llm-query-arms-v1"
SIDECAR_SCHEMA = "llm-query-arms-per-query-v1"
RESULTS = {"fixture": "bench/results/llm_query_arms.json", "scifact": "bench/results/beir_scifact_llm_query_arms.json",
           "fiqa": "bench/results/beir_fiqa_llm_query_arms.json"}
STRETCH_RESULTS = {d: rel.replace(".json", "_122b.json") for d, rel in RESULTS.items()}
BAR_SCOPE = {"fixture": "held_out", "scifact": "test", "fiqa": "test"}
PERMUTATIONS = {"n": 10000, "seed": 0, "method": "paired sign-flip of the per-query differences, one sign matrix per scope "
                "(numpy RandomState(seed).randint(0, 2)), two-sided, p = (1 + #|mean| >= |observed|) / (1 + n)"}
MDE_FACTOR = 2.8  # indicative minimum detectable delta: 2.8 x SD of the per-query P1 - P0 differences / sqrt(n)


# ---------------------------------------------------------------- one interval, computed once, stored twice


def verdict_word(lo_raw: float, hi_raw: float) -> str:
    """'above zero' when lo_raw > 0, 'below zero' when hi_raw < 0, otherwise 'covers zero' (a bound exactly 0.0 too)."""
    if lo_raw > 0:
        return "above zero"
    if hi_raw < 0:
        return "below zero"
    return "covers zero"


def mirrored(word: str | None) -> str | None:
    """The word of Y minus X from the stored word of X minus Y (lo(Y-X) = -hi(X-Y) and hi(Y-X) = -lo(X-Y))."""
    return {"above zero": "below zero", "below zero": "above zero"}.get(word, word) if word else word


def fmt_bound(raw: float, rounded: float | None = None) -> str:
    """A bound as printed: 4 decimals, except a bound that rounds to +/-0.0000 is printed as its raw value to two
    significant digits (+4.0e-05), so a printed interval never disagrees with its word."""
    rounded = round(raw, 4) if rounded is None else rounded
    if rounded == 0 and raw != 0:
        return f"{raw:+.1e}"
    return f"{rounded:+.4f}"


def fmt_block(block: dict | None, metric: str = PRIMARY) -> str:
    """'+0.0123 [+0.0010, +0.0300]' from a stored comparison block (raw bounds where a rounded bound is 0)."""
    if not block or metric not in block:
        return "-"
    b = block[metric]
    return f"{b['delta']:+.4f} [{fmt_bound(b['lo_raw'], b['lo'])}, {fmt_bound(b['hi_raw'], b['hi'])}]"


def permutation_signs(n: int, n_perm: int = PERMUTATIONS["n"], seed: int = PERMUTATIONS["seed"]) -> np.ndarray:
    """(n_perm, n) matrix of +1/-1, one per scope (like the bootstrap's one index matrix)."""
    return np.random.RandomState(seed).randint(0, 2, size=(n_perm, n)).astype(np.float64) * 2.0 - 1.0


def sign_flip_p(diff: Sequence[float], signs: np.ndarray) -> float:
    """Two-sided paired sign-flip permutation p-value of the mean difference."""
    d = np.asarray(diff, dtype=np.float64)
    observed = abs(float(d.mean()))
    means = np.abs(signs @ d) / len(d)
    return float((1 + int(np.sum(means >= observed - 1e-15))) / (1 + signs.shape[0]))


def bh_qvalues(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg q-values, in the input order (descriptive: no decision reads them)."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = len(p)
    if not m:
        return []
    order = np.argsort(p, kind="mergesort")
    ranked = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return [float(x) for x in out]


def paired(a: dict[str, Sequence[float]], b: dict[str, Sequence[float]], idx: np.ndarray, metrics: Sequence[str],
           signs: np.ndarray | None = None) -> dict[str, Any]:
    """The comparison block of a minus b over one scope: per metric the delta and its 95% interval rounded to 4
    decimals and raw; the verdict word (nDCG@10, raw bounds); the number of queries whose nDCG@10 differs; and, given a
    sign matrix, the permutation p-value of nDCG@10 (descriptive)."""
    from app.eval.stats import paired_delta_ci

    block: dict[str, Any] = {}
    for m in metrics:
        d, lo, hi = paired_delta_ci(a[m], b[m], idx)
        block[m] = {"delta": round(d, 4), "lo": round(lo, 4), "hi": round(hi, 4), "delta_raw": d, "lo_raw": lo, "hi_raw": hi}
    block["verdict"] = verdict_word(block[PRIMARY]["lo_raw"], block[PRIMARY]["hi_raw"])
    block["n_changed"] = int(sum(1 for x, y in zip(a[PRIMARY], b[PRIMARY]) if x != y))
    if signs is not None:
        block["permutation_p"] = sign_flip_p(np.asarray(a[PRIMARY], dtype=np.float64) - np.asarray(b[PRIMARY], dtype=np.float64), signs)
    return block


def row_ci(values: dict[str, Sequence[float]], idx: np.ndarray, metrics: Sequence[str]) -> tuple[dict, dict]:
    """(ci, retrieval): per metric the mean and its interval rounded to 4 decimals, and the rounded means."""
    from app.eval.stats import mean_ci

    ci, means = {}, {}
    for m in metrics:
        mean, lo, hi = mean_ci(values[m], idx)
        ci[m] = {"mean": round(mean, 4), "lo": round(lo, 4), "hi": round(hi, 4)}
        means[m] = round(float(np.mean(np.asarray(values[m], dtype=np.float64))), 4)
    return ci, means


def scope_blocks(scores: dict[str, dict[str, list[float]]], subset: Sequence[int], rows: Sequence[str],
                 pairs: Sequence[tuple[str, str, str]], metrics: Sequence[str]) -> dict[str, Any]:
    """One scope of a results file from per-query scores: the bootstrap settings, each row's ci and means, and each
    comparison (row, other, kind) as a paired block. The scoring run and --rescore both come here."""
    from app.eval.stats import DEFAULT_N_BOOT, DEFAULT_SEED, resample_indices, settings

    n = len(subset)
    idx = resample_indices(n, n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED)
    signs = permutation_signs(n)
    col = {name: {m: [scores[name][m][i] for i in subset] for m in metrics} for name in rows}
    out_rows = []
    for name in rows:
        ci, means = row_ci(col[name], idx, metrics)
        out_rows.append({"config": name, "n": n, "ci": ci, "retrieval": means})
    comparisons = []
    for a, b, kind in pairs:
        comparisons.append({"row": a, "other": b, "kind": kind, **paired(col[a], col[b], idx, metrics, signs)})
    return {"n": n, "bootstrap": {**settings(n), "permutation": PERMUTATIONS}, "rows": out_rows, "comparisons": comparisons}


# ---------------------------------------------------------------- reading the stored words


def find_block(scope: dict, row: str, other: str) -> dict | None:
    for c in scope.get("comparisons") or []:
        if c["row"] == row and c["other"] == other:
            return c
    return None


def stored_word(block: dict | None) -> str | None:
    return block.get("verdict") if block else None


def cell(blobs: dict[str, dict], dataset: str, row: str, other: str) -> dict[str, Any]:
    """The bar cell of (row - other) on one dataset: {"word": stored verdict | None (missing) | "withheld", "block"}.
    A dataset whose parents did not reproduce gives "withheld"; a missing file or row gives None."""
    blob = blobs.get(dataset)
    if blob is None:
        return {"word": None, "block": None, "missing": "no results file"}
    if (blob.get("reproduction") or {}).get("status") == "failed":
        return {"word": "withheld", "block": None}
    scope = (blob.get("scopes") or {}).get(BAR_SCOPE[dataset]) or {}
    block = find_block(scope, row, other)
    if block is None:
        return {"word": None, "block": None, "missing": "no row"}
    return {"word": stored_word(block), "block": block}


def passes(words: dict[str, str | None]) -> str:
    """'passes', 'does not pass', 'open' (a cell missing) or 'withheld' (a dataset's parents did not reproduce), from
    the stored words of the three cells: above zero on SciFact AND on FiQA AND not below zero on the fixture."""
    if any(words.get(d) == "withheld" for d in spec.DATASETS):
        return "withheld"
    if any(words.get(d) is None for d in spec.DATASETS):
        return "open"
    ok = words["scifact"] == "above zero" and words["fiqa"] == "above zero" and words["fixture"] != "below zero"
    return "passes" if ok else "does not pass"


def all_three(words: dict[str, str | None]) -> str:
    """M43's stricter reading, above zero on all three (a column, never a decision)."""
    state = passes(words)
    if state in ("open", "withheld"):
        return state
    return "yes" if all(words[d] == "above zero" for d in spec.DATASETS) else "no"


def any_below(words: dict[str, str | None]) -> bool:
    return any(words.get(d) == "below zero" for d in spec.DATASETS)


def replication_label(qwen_words: dict[str, str | None], gemma_words: dict[str, str | None]) -> str:
    """The first row of spec.REPLICATION that matches (qwen3.5:27b's cells, gemma4:26b's cells at the same arm)."""
    q = passes(qwen_words)
    if q in ("open", "withheld"):
        return q
    if any(gemma_words.get(d) is None for d in spec.DATASETS):
        return "not run"
    g = passes(gemma_words)
    if g == "withheld":  # the parents are shared, so qwen would be withheld too; kept for completeness
        return "withheld"
    if q == "passes":
        if g == "passes":
            return "replicated"
        return "contradicted" if any_below(gemma_words) else "not replicated"
    if g == "passes":
        return "contradicted" if any_below(qwen_words) else "gemma only"
    return "neither passes"


# ---------------------------------------------------------------- the placement decision


def _arm_words(blobs: dict[str, dict], technique: str, placement: str, tag: str) -> tuple[dict[str, str | None], dict[str, dict]]:
    row = spec.arm_name(technique, placement, tag)
    parent = spec.parent_name(technique, placement)
    cells = {d: cell(blobs, d, row, parent) for d in spec.DATASETS}
    return {d: c["word"] for d, c in cells.items()}, cells


def _direct_words(blobs: dict[str, dict], x: str, y: str) -> dict[str, str | None]:
    """The words of X minus Y per dataset: the stored (X, Y) block, else the mirrored (Y, X) block."""
    out = {}
    for d in spec.DATASETS:
        c = cell(blobs, d, x, y)
        if c["word"] is None:
            c2 = cell(blobs, d, y, x)
            out[d] = mirrored(c2["word"])
        else:
            out[d] = c["word"]
    return out


def _separates(words: dict[str, str | None]) -> bool:
    return words.get("scifact") == "above zero" and words.get("fiqa") == "above zero" and words.get("fixture") not in ("below zero", None)


DATASET_NAMES = {"fixture": "the fixture", "scifact": "SciFact", "fiqa": "FiQA"}


def _names(datasets: Sequence[str]) -> str:
    names = [DATASET_NAMES[d] for d in datasets]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _mid(name: str) -> str:
    """A technique's name inside a sentence: an acronym stays, otherwise the first letter is lowercased."""
    return name if name[:2].isupper() else name[0].lower() + name[1:]


def technique_placements(technique: str) -> list[str]:
    return [p for t, p, _, _ in spec.ARMS if t == technique]


def decide(blobs: dict[str, dict], technique: str) -> dict[str, Any]:
    """The one sentence of `technique` (rules 0-4 of spec.DECISION, qwen3.5:27b), with the gemma4:26b replication label
    of each arm it names, and the arms it names (their cost is printed beside it by the caller)."""
    name = spec.TECHNIQUE_NAMES[technique]
    q, g = spec.MAIN_MODEL, spec.REPLICATION_MODEL
    placements = technique_placements(technique)
    words = {p: _arm_words(blobs, technique, p, q)[0] for p in placements}
    gwords = {p: _arm_words(blobs, technique, p, g)[0] for p in placements}
    states = {p: passes(words[p]) for p in placements}
    labels = {p: replication_label(words[p], gwords[p]) for p in placements}
    phrase = {p: ("without a reranker" if p == "none" else spec.PLACEMENT_PHRASES[p]) for p in placements}
    withheld = sorted({d for p in placements for d in spec.DATASETS if words[p].get(d) == "withheld"}, key=spec.DATASETS.index)
    missing = sorted({d for p in placements for d in spec.DATASETS if words[p].get(d) is None}, key=spec.DATASETS.index)
    out: dict[str, Any] = {"technique": technique, "states": states, "replication": labels}
    if withheld:
        out.update(rule=0, sentence=f"{name}: withheld ({_names(withheld)}: the parents did not reproduce).", named=[])
        return out
    if missing:
        out.update(rule=0, sentence=f"{name}: open ({_names(missing)} not measured).", named=[])
        return out
    reranked = [p for p in placements if p != "none"]
    R = [p for p in reranked if states[p] == "passes"]
    N = "none" in placements and states["none"] == "passes"
    clause = (" " + spec.DECISION["clause_when_n_holds_in_1_and_2"]) if N else ""
    if len(R) == 1:
        x = R[0]
        named = [x] + (["none"] if N else [])
        sentence, rule = f"{name} passes the declared bar {phrase[x]}.{clause}", 1
    elif len(R) == 2:
        x, y = R
        named = [x, y] + (["none"] if N else [])
        xy = _direct_words(blobs, spec.arm_name(technique, x, q), spec.arm_name(technique, y, q))
        yx = {d: mirrored(w) for d, w in xy.items()}
        if _separates(xy):
            tail = f"in their direct comparison {phrase[x]} is above {phrase[y]}"
        elif _separates(yx):
            tail = f"in their direct comparison {phrase[y]} is above {phrase[x]}"
        else:
            tail = "their direct comparison does not separate them"
        sentence, rule = f"{name} passes the declared bar {phrase[x]} and {phrase[y]}; {tail}.{clause}", 2
        out["direct"] = xy
    elif N:
        named = ["none"]
        sentence, rule = f"{name} passes against P0 without a reranker; with the reranker it is not shown to help.", 3
        out["vs_P1"] = {d: (cell(blobs, d, spec.arm_name(technique, "none", q), spec.P1)["block"] or {}) for d in spec.DATASETS}
    else:
        named = []
        sentence, rule = f"No placement of {_mid(name)} passes the declared bar on these datasets.", 4
    if rule == 4:
        listed = [p for p in placements if labels[p] != "neither passes"]
        rep = ("; ".join(f"{phrase[p]}: {labels[p]}" for p in listed) if listed else "neither model passes any arm")
    else:
        rep = "; ".join(f"{phrase[p]}: {labels[p]}" for p in named)
    out.update(rule=rule, named=named, sentence=f"{sentence} gemma4:26b replication: {rep}.")
    return out


def bar_rows(blobs: dict[str, dict], tag: str) -> list[dict[str, Any]]:
    """Per arm of model `tag`: the three cells (stored words and blocks), passes, all three, and (both bar models) the
    replication label."""
    rows = []
    for technique, placement, _, _ in spec.ARMS:
        words, cells = _arm_words(blobs, technique, placement, tag)
        entry = {"technique": technique, "placement": placement, "config": spec.arm_name(technique, placement, tag),
                 "parent": spec.parent_name(technique, placement), "words": words, "cells": cells, "passes": passes(words),
                 "all_three": all_three(words)}
        if tag in spec.BAR_MODELS:
            qw = _arm_words(blobs, technique, placement, spec.MAIN_MODEL)[0]
            gw = _arm_words(blobs, technique, placement, spec.REPLICATION_MODEL)[0]
            entry["replication"] = replication_label(qw, gw)
        rows.append(entry)
    return rows


def load_blobs(root: Path, results: dict[str, str] | None = None) -> dict[str, dict]:
    results = results or RESULTS
    return {d: json.loads((root / rel).read_text()) for d, rel in results.items() if (root / rel).exists()}


def decisions(blobs: dict[str, dict]) -> list[dict[str, Any]]:
    return [decide(blobs, t) for t in spec.TECHNIQUES]


def n_intervals(blobs: dict[str, dict]) -> dict[str, int]:
    """The milestone's paired nDCG@10 intervals by kind, summed over the files' own counts."""
    total: dict[str, int] = {}
    for blob in blobs.values():
        for kind, n in (blob.get("n_intervals") or {}).items():
            total[kind] = total.get(kind, 0) + int(n)
    total["total"] = sum(v for k, v in total.items() if k != "total")
    return total


def minimum_detectable(diffs: Sequence[float]) -> float:
    """MDE_FACTOR x the SD (ddof 1) of per-query differences / sqrt(n): a scale for reading 'covers zero'."""
    d = np.asarray(diffs, dtype=np.float64)
    return float(MDE_FACTOR * d.std(ddof=1) / math.sqrt(len(d)))


# ---------------------------------------------------------------- every interval again, from the sidecars


def _differences(where: str, stored: Any, again: Any, out: list[str]) -> None:
    if isinstance(stored, dict) and isinstance(again, dict):
        for k in sorted(set(stored) | set(again)):
            _differences(f"{where}.{k}", stored.get(k), again.get(k), out)
    elif isinstance(stored, list) and isinstance(again, list) and len(stored) == len(again):
        for i, (s, a) in enumerate(zip(stored, again)):
            _differences(f"{where}[{i}]", s, a, out)
    elif stored != again:
        out.append(f"{where}: results file {stored!r}, recomputed {again!r}")


def rescore_blob(blob: dict, sidecar: dict) -> list[str]:
    """Every row ci, mean, comparison block (rounded and raw bounds, word, changed count, permutation p) and the
    bootstrap settings of every scope of one results file that its sidecar does not reproduce."""
    out: list[str] = []
    for key in ("git_sha", "timestamp"):
        if sidecar.get(key) != blob.get(key):
            return [f"{key}: results file {blob.get(key)!r}, sidecar {sidecar.get(key)!r} (not the same run)"]
    metrics = tuple(sidecar["metrics"])
    for scope_name, scope in (blob.get("scopes") or {}).items():
        subset = sidecar["scopes"][scope_name]
        rows = [r["config"] for r in scope["rows"]]
        pairs = [(c["row"], c["other"], c["kind"]) for c in scope["comparisons"]]
        again = scope_blocks(sidecar["scores"], subset, rows, pairs, metrics)
        _differences(f"{scope_name}.bootstrap", scope["bootstrap"], again["bootstrap"], out)
        _differences(f"{scope_name}.n", scope["n"], again["n"], out)
        for s_row, a_row in zip(scope["rows"], again["rows"]):
            for key in ("config", "n", "ci", "retrieval"):
                _differences(f"{scope_name}.{s_row['config']}.{key}", s_row.get(key), a_row.get(key), out)
        for s_c, a_c in zip(scope["comparisons"], again["comparisons"]):
            _differences(f"{scope_name}.{s_c['row']} - {s_c['other']}", {k: v for k, v in s_c.items()},
                         {k: v for k, v in a_c.items()}, out)
    counted = count_intervals(blob)
    _differences("n_intervals", blob.get("n_intervals"), counted, out)
    return out


def count_intervals(blob: dict) -> dict[str, int]:
    """A results file's paired nDCG@10 intervals, by kind of comparison, over all its scopes."""
    counts: dict[str, int] = {}
    for scope in (blob.get("scopes") or {}).values():
        for c in scope.get("comparisons") or []:
            counts[c["kind"]] = counts.get(c["kind"], 0) + 1
    return dict(sorted(counts.items()))


def rescore(root: Path | None = None, *, results: dict[str, str] | None = None) -> tuple[int, list[str]]:
    """(files checked, differences) over the committed M46 results files that exist (the stretch's too): every interval
    from the sidecars, every LLM, cost and fired block of an arm row from bench/llm_query, and the decision sentences
    from the recomputed words against those from the stored words."""
    from app.eval import per_question

    root = root or Path(__file__).resolve().parents[2]
    files = dict(results) if results else {**RESULTS, **{f"{d}-122b": rel for d, rel in STRETCH_RESULTS.items()}}
    checked, problems = 0, []
    recomputed: dict[str, dict] = {}
    for dataset, rel in files.items():
        path = root / rel
        if not path.exists():
            continue
        side = per_question.sidecar_path(path)
        if not side.exists():
            problems.append(f"{rel}: no sidecar {side.name}")
            continue
        blob, sidecar = json.loads(path.read_text()), json.loads(side.read_text())
        problems += [f"{rel}: {d}" for d in rescore_blob(blob, sidecar)]
        problems += [f"{rel}: {d}" for d in rescore_llm_blocks(root, blob)]
        again = json.loads(json.dumps(blob))
        for scope_name, scope in again.get("scopes", {}).items():
            subset = sidecar["scopes"][scope_name]
            fresh = scope_blocks(sidecar["scores"], subset, [r["config"] for r in scope["rows"]],
                                 [(c["row"], c["other"], c["kind"]) for c in scope["comparisons"]], tuple(sidecar["metrics"]))
            scope["comparisons"] = fresh["comparisons"]
        recomputed[dataset] = again
        checked += 1
    stored = {d: b for d, b in load_blobs(root, results or RESULTS).items()}
    if stored:
        for a, b in zip(decisions(stored), decisions({d: recomputed[d] for d in stored if d in recomputed})):
            if a["sentence"] != b["sentence"]:
                problems.append(f"decision {a['technique']}: stored words give {a['sentence']!r}, recomputed {b['sentence']!r}")
    return checked, problems


def rescore_llm_blocks(root: Path, blob: dict) -> list[str]:
    """The llm, cost and fired-by-cache blocks of every arm row, recomputed from bench/llm_query (no model)."""
    from app.eval.llm_query_arms import llm_blocks_from_cache

    out: list[str] = []
    stored = {r["config"]: r for r in blob.get("rows") or []}
    try:
        again = llm_blocks_from_cache(root, blob)
    except FileNotFoundError as e:
        return [f"bench/llm_query: {e}"]
    for name, blocks in again.items():
        row = stored.get(name)
        if row is None:
            out.append(f"{name}: in the cache's arms but not in the results file")
            continue
        for key, value in blocks.items():
            _differences(f"{name}.{key}", row.get(key), value, out)
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rescore", action="store_true", help="recompute every interval of the committed M46 files from their sidecars (no model)")
    p.add_argument("--bar", action="store_true", help="print the declared bar and the placement decision from the committed files")
    args = p.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if args.rescore:
        checked, problems = rescore(root)
        for line in problems[:80]:
            print(f"  - {line}")
        print(f"llm-query-arms intervals: {'OK' if not problems else f'{len(problems)} differences'}, {checked} results files "
              "recomputed from their per-query sidecars and bench/llm_query")
        return 1 if problems or not checked else 0
    if args.bar:
        from app.eval.llm_query_gen import print_bar  # the printing is plumbing; every word and sentence comes from here

        return print_bar(root)
    p.error("pass --rescore or --bar")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
