"""Compare a bench result against the committed baseline; optionally fail on regression.

    python -m app.eval.compare                       # report latest.json vs main.json
    python -m app.eval.compare --fail-on-regression  # exit 1 if a gated metric drops

Gate: for every config in the baseline, each metric in GATED_METRICS (nDCG@10,
Recall@10, Recall@50) must not be lower than the baseline by more than
--threshold. The default threshold is 0.002: the bench is deterministic (hash
embeddings, overlap reranker, fixed corpus) and reproduced to four decimals on
Python 3.11 / numpy 1.26 and 3.12 / numpy 2.5, so a drop of 0.002 is a real
change, not rounding noise. A gated metric missing from the baseline (older
result files) is skipped with a note. Improvements never fail the gate; promote
them with `make bench-baseline`.

Only the retrieval metrics are compared. Timestamps, latencies and the heuristic
e2e block differ from run to run and are ignored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPORTED_METRICS = ("ndcg@10", "recall@10", "recall@50", "mrr@10")
GATED_METRICS = ("ndcg@10", "recall@10", "recall@50")
DEFAULT_THRESHOLD = 0.002


def load_result(path: Path) -> dict:
    return json.loads(path.read_text())


def _by_config(blob: dict) -> dict[str, dict]:
    return {row["config"]: row for row in blob.get("results", [])}


def compare(current: dict, baseline: dict, metrics=REPORTED_METRICS) -> str:
    """Markdown table of baseline vs current for every config in either file."""
    cur_map = _by_config(current)
    base_map = _by_config(baseline)
    names = list(dict.fromkeys([*base_map, *cur_map]))
    lines = [
        f"current sha: {current.get('git_sha')}  baseline sha: {baseline.get('git_sha')}",
        "",
        "| config | metric | baseline | current | delta |",
        "|---|---|---:|---:|---:|",
    ]
    for name in names:
        if name not in cur_map or name not in base_map:
            lines.append(f"| {name} | — | missing | missing | — |")
            continue
        for metric in metrics:
            b = base_map[name]["retrieval"].get(metric)
            c = cur_map[name]["retrieval"].get(metric)
            if b is None or c is None:
                lines.append(f"| {name} | {metric} | {'—' if b is None else f'{b:.3f}'} | {'—' if c is None else f'{c:.3f}'} | — |")
                continue
            delta = c - b
            sign = "+" if delta >= 0 else ""
            lines.append(f"| {name} | {metric} | {b:.3f} | {c:.3f} | {sign}{delta:.3f} |")
    return "\n".join(lines)


def regressions(
    current: dict,
    baseline: dict,
    threshold: float = DEFAULT_THRESHOLD,
    metrics=GATED_METRICS,
    notes: list[str] | None = None,
) -> list[str]:
    """Human-readable reasons the current result fails the gate (empty = pass).

    Every config in the baseline must exist in the current result, and each gated
    metric must not drop by more than `threshold`. Configs that only exist in the
    current result are new rows and are not gated. A gated metric the baseline
    does not have is skipped and mentioned in `notes`.
    """
    cur_map = _by_config(current)
    base_map = _by_config(baseline)
    problems: list[str] = []
    for name, base_row in base_map.items():
        cur_row = cur_map.get(name)
        if cur_row is None:
            problems.append(f"{name}: present in baseline but missing from current result")
            continue
        for metric in metrics:
            b = base_row["retrieval"].get(metric)
            c = cur_row["retrieval"].get(metric)
            if b is None:
                if notes is not None:
                    notes.append(f"{name}: {metric} not in baseline; not gated (run `make bench-baseline`)")
                continue
            if c is None:
                problems.append(f"{name}: {metric} missing from current result")
                continue
            if b - c > threshold + 1e-9:  # metrics are rounded to 4 dp; absorb float error at the boundary
                problems.append(
                    f"{name}: {metric} dropped {b:.4f} -> {c:.4f} (delta {c - b:+.4f}, threshold {threshold})"
                )
    return problems


def default_paths(root: Path) -> tuple[Path, Path]:
    results = root / "bench" / "results"
    return results / "latest.json", results / "main.json"


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    default_current, default_baseline = default_paths(root)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--current", type=Path, default=default_current, help="result to check (default: bench/results/latest.json)")
    p.add_argument("--baseline", type=Path, default=default_baseline, help="committed baseline (default: bench/results/main.json)")
    p.add_argument("--fail-on-regression", action="store_true", help="exit 1 when a gated metric drops beyond --threshold")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help=f"largest tolerated drop per gated metric (default {DEFAULT_THRESHOLD})")
    args = p.parse_args(argv)

    if not args.current.exists():
        print(f"no current result at {args.current} — run `make bench` first")
        return 2
    if not args.baseline.exists():
        print(f"no baseline at {args.baseline} — run `make bench-baseline` and commit it")
        return 2 if args.fail_on_regression else 0

    current = load_result(args.current)
    baseline = load_result(args.baseline)
    print(compare(current, baseline))
    notes: list[str] = []
    problems = regressions(current, baseline, threshold=args.threshold, notes=notes)
    for line in notes:
        print(f"note: {line}")
    if not problems:
        print(f"\ngate: OK ({', '.join(GATED_METRICS)} within {args.threshold} of baseline for {len(_by_config(baseline))} configs)")
        return 0
    print("\ngate: REGRESSION" if args.fail_on_regression else "\ngate: regression (report only)")
    for line in problems:
        print(f"  - {line}")
    return 1 if args.fail_on_regression else 0


if __name__ == "__main__":
    raise SystemExit(main())
