"""Compare a bench result JSON against another (default: main / previous latest)."""

from __future__ import annotations

import json
from pathlib import Path


METRICS = ("recall@50", "ndcg@10", "mrr@10")


def load_result(path: Path) -> dict:
    return json.loads(path.read_text())


def _by_config(blob: dict) -> dict[str, dict]:
    return {row["config"]: row for row in blob.get("results", [])}


def compare(current: dict, baseline: dict) -> str:
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
        for metric in METRICS:
            b = base_map[name]["retrieval"][metric]
            c = cur_map[name]["retrieval"][metric]
            delta = c - b
            sign = "+" if delta >= 0 else ""
            lines.append(f"| {name} | {metric} | {b:.3f} | {c:.3f} | {sign}{delta:.3f} |")
    return "\n".join(lines)


def default_paths(root: Path) -> tuple[Path, Path | None]:
    results = root / "bench" / "results"
    current = results / "latest.json"
    main = results / "main.json"
    baseline = main if main.exists() else None
    if baseline is None:
        jsons = sorted(results.glob("*.json"))
        jsons = [p for p in jsons if p.name not in {"latest.json", "main.json"}]
        if len(jsons) >= 2:
            baseline = jsons[-2]
    return current, baseline


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    current_path, baseline_path = default_paths(root)
    if not current_path.exists():
        raise SystemExit("no bench/results/latest.json — run `make bench` first")
    current = load_result(current_path)
    if baseline_path is None or not baseline_path.exists():
        print("no baseline result yet; copying latest -> main.json")
        main_path = root / "bench" / "results" / "main.json"
        main_path.write_text(current_path.read_text())
        print(f"seeded {main_path}")
        return
    print(compare(current, load_result(baseline_path)))


if __name__ == "__main__":
    main()
