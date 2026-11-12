"""Per-question scores next to a harness results file, and every interval recomputed from them.

    python -m app.eval.per_question bench/results/main.json [more results files]   # make bench-intervals

A results file stores means, percentile intervals and paired deltas; the per-question scores they come from used
to be dropped after the bootstrap, so an interval could only be checked by re-running the benchmark. The harness
now writes them into a sidecar beside each results file it writes (`main.json` -> `main.per_query.json`, found by
that name: `make bench-baseline` copies latest.json's sidecar with it):

  ids, categories   the gold questions, in gold order
  metrics           the metrics with intervals (harness.CI_METRICS when the file was written)
  scores            config -> metric -> one score per question, unrounded (JSON keeps every float exactly)
  git_sha, timestamp  copied from the results header, so a sidecar from another run is refused

Nothing else: no command, path, host or user, and no retrieved lists. recompute() runs the harness's own
bootstrap (attach_confidence, the seed and resample count in the file's header) over these scores and
differences() compares every interval, paired delta, per-category delta, comparison and mean with the file:
empty when the file is reproduced exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCHEMA = "per-question-scores-v1"
SUFFIX = ".per_query.json"
INTERVAL_KEYS = ("ci", "delta_vs_bm25_only", "delta_vs_previous", "delta_vs_parent_by_category", "delta_vs_parent_by_category_all")
ABOUT = ("per-question scores of the results file with the same name (.json), in gold order and unrounded: "
         "`make bench-intervals` recomputes every interval and paired delta in that file from them")


def sidecar_path(results: Path) -> Path:
    """bench/results/main.json -> bench/results/main.per_query.json."""
    results = Path(results)
    return results.with_name(results.name[: -len(".json")] + SUFFIX if results.name.endswith(".json") else results.name + SUFFIX)


def build(blob: dict, scores_by_config: dict[str, list[dict[str, float]]], ids: list[str], categories: list[str],
          metrics: tuple[str, ...]) -> dict[str, Any]:
    """The sidecar of results header `blob`: the per-question scores of every row, column by column."""
    return {
        "schema": SCHEMA,
        "about": ABOUT,
        "git_sha": blob.get("git_sha"),
        "timestamp": blob.get("timestamp"),
        "metrics": list(metrics),
        "ids": list(ids),
        "categories": list(categories),
        "scores": {row["config"]: {m: [float(q[m]) for q in scores_by_config[row["config"]]] for m in metrics}
                   for row in blob["results"] if scores_by_config.get(row["config"])},
    }


def _dump(value: Any, depth: int = 0) -> str:
    """JSON with every list of scalars on one line (one line per score column), objects indented."""
    if isinstance(value, dict) and value:
        pad = "  " * (depth + 1)
        return "{\n" + ",\n".join(f"{pad}{json.dumps(k)}: {_dump(v, depth + 1)}" for k, v in value.items()) + "\n" + "  " * depth + "}"
    return json.dumps(value)


def write(path: Path, sidecar: dict) -> Path:
    path = Path(path)
    path.write_text(_dump(sidecar) + "\n")
    return path


def recompute(blob: dict, sidecar: dict) -> tuple[list[dict], dict]:
    """The results file's rows reduced to their names and parents, with every interval block recomputed from the
    sidecar's scores by the harness's own attach_confidence; and the bootstrap settings it returns."""
    from app.eval.harness import attach_confidence

    metrics = tuple(sidecar["metrics"])
    scores = {name: [dict(zip(metrics, values)) for values in zip(*(columns[m] for m in metrics))]
              for name, columns in sidecar["scores"].items()}
    rows = [{"config": r["config"], "settings": {"parent": (r.get("settings") or {}).get("parent")}} for r in blob["results"]]
    boot = blob.get("bootstrap") or {}
    settings = attach_confidence(
        rows, scores, n_queries=len(sidecar["ids"]), seed=boot["seed"], n_boot=boot["n_boot"], categories=sidecar["categories"],
        comparisons=[(c["row"], c["other"]) for c in boot.get("comparisons") or []],
        all_categories=any("delta_vs_parent_by_category_all" in r for r in blob["results"]), metrics=metrics)
    return rows, settings


def differences(blob: dict, sidecar: dict) -> list[str]:
    """Every interval, paired delta, comparison and mean of `blob` that the sidecar does not reproduce (empty: all do)."""
    from app.eval.compare import _leaf_differences

    out: list[str] = []
    if sidecar.get("schema") != SCHEMA:
        return [f"sidecar schema {sidecar.get('schema')!r}, expected {SCHEMA!r}"]
    for key in ("git_sha", "timestamp"):
        if sidecar.get(key) != blob.get(key):
            out.append(f"{key}: results file {blob.get(key)!r}, sidecar {sidecar.get(key)!r} (not the same run)")
    if len(sidecar["ids"]) != blob.get("n_gold", len(sidecar["ids"])) or len(sidecar["categories"]) != len(sidecar["ids"]):
        out.append(f"{len(sidecar['ids'])} ids and {len(sidecar['categories'])} categories for n_gold {blob.get('n_gold')}")
    missing = [r["config"] for r in blob["results"] if r["config"] not in sidecar["scores"]]
    if missing:
        out.append(f"no scores for rows {missing}")
    if out:
        return out
    rows, settings = recompute(blob, sidecar)
    for stored, again in zip(blob["results"], rows):
        name = stored["config"]
        for key in INTERVAL_KEYS:
            if key in stored or key in again:
                _leaf_differences(f"{name}.{key}", stored.get(key), again.get(key), out)
        for metric in sidecar["metrics"]:
            column = sidecar["scores"][name][metric]
            mean = round(sum(column) / len(column), 4)
            if stored["retrieval"].get(metric) != mean:
                out.append(f"{name}.retrieval.{metric}: results file {stored['retrieval'].get(metric)!r}, mean of the sidecar {mean!r}")
    _leaf_differences("bootstrap.comparisons", (blob.get("bootstrap") or {}).get("comparisons"), settings.get("comparisons"), out)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", nargs="+", type=Path, help="harness results files (their sidecars are found by name)")
    args = p.parse_args(argv)
    failed = 0
    for path in args.results:
        side = sidecar_path(path)
        if not side.exists():
            print(f"{path}: no sidecar {side.name} (written by the harness since B32; re-record the file to get one)")
            failed += 1
            continue
        blob, sidecar = json.loads(path.read_text()), json.loads(side.read_text())
        problems = differences(blob, sidecar)
        if problems:
            failed += 1
            print(f"intervals: {len(problems)} differences between {path.name} and {side.name}:")
            for line in problems[:50]:
                print(f"  - {line}")
        else:
            print(f"intervals: OK, every interval, paired delta and mean of the {len(blob['results'])} rows of {path.name} "
                  f"recomputed exactly from {side.name} ({len(sidecar['ids'])} questions, {', '.join(sidecar['metrics'])})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
