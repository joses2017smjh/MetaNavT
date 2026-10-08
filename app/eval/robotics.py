"""Frozen real-experiment file lookup benchmark, separate from synthetic gold.

Questions are mechanically generated from scalar JSON fields. Scores measure
source-file retrieval, never answer entailment or independently annotated QA.
The split and bootstrap unit is an experiment directory, not an episode row.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import random
import re
import socket
import subprocess
import time

import numpy as np

from app.eval.metrics import score_query
from app.retrieval.bm25 import BM25Index
from app.retrieval.context import path_context

SCHEMA = "robotics-file-lookup-v1"
SUFFIXES = {".json", ".csv", ".yaml", ".yml", ".md", ".txt"}
FIELDS = {
    "success", "success_rate", "clean_success", "completion_s", "fallen",
    "falls", "fall_rate", "goal_reached", "path_length_m", "mapped_fraction_end",
    "wall_contact_steps", "verdict", "complete", "passed", "completed_episodes",
    "total_mass_urdf_kg", "total_mass_mjcf_kg", "agrees", "episodes", "steps",
    "elapsed_s", "wall_seconds", "turns", "replans", "map_updates", "status",
    "num_envs", "variant", "sensor_mode", "initial_heading_rad", "outcome",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def confined(root: Path, relative: str) -> Path:
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"unsafe relative path: {relative}")
    target = (root / p).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes snapshot: {relative}")
    return target


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def scalar_questions(relative: str, data: bytes, family: str) -> list[dict]:
    """Read only top-level scalar fields with an exact UTF-8 evidence span."""
    if not relative.endswith(".json"):
        return []
    try:
        obj = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        return []
    if not isinstance(obj, dict):
        return []
    out = []
    for key in sorted(FIELDS & obj.keys()):
        value = obj[key]
        if value is None or isinstance(value, (dict, list)):
            continue
        try:
            token = json.dumps(value, ensure_ascii=False, allow_nan=False)
        except ValueError:
            continue
        if len(token) > 160:
            continue
        # JSON may encode the same string using escapes; parse the matched token.
        pattern = re.escape(json.dumps(key).encode()) + rb'\s*:\s*("(?:[^"\\]|\\.)*"|true|false|-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)'
        for match in re.finditer(pattern, data):
            try:
                if json.loads(match.group(1)) != value:
                    continue
            except ValueError:
                continue
            prefix = data[:match.start()]
            # A nested occurrence could share this key: require top-level depth.
            # The JSON decoder below locates object keys without trusting regex nesting.
            if not _is_top_level(prefix):
                continue
            label = re.sub(r"[_/.-]+", " ", relative.removeprefix("results/").removesuffix(".json"))
            out.append({
                "id": sha256((relative + ":" + key).encode())[:20],
                "family": family, "question": f"For experiment {label}, what is the {key.replace('_', ' ')}?",
                "field": key, "answer": value, "relevant_paths": [relative],
                "evidence": {"path": relative, "start_byte": match.start(1), "end_byte": match.end(1), "sha256": sha256(data)},
                "label_origin": "generated_from_top_level_json_scalar",
                "human_reviewed": False,
            })
            break
    return out


def _is_top_level(prefix: bytes) -> bool:
    depth = 0
    quoted = escaped = False
    for char in prefix:
        if quoted:
            if escaped:
                escaped = False
            elif char == 92:
                escaped = True
            elif char == 34:
                quoted = False
        elif char == 34:
            quoted = True
        elif char in (123, 91):
            depth += 1
        elif char in (125, 93):
            depth -= 1
    return depth == 1 and not quoted


def freeze(source: Path, output: Path, *, n_files: int = 600, dev_n: int = 120, test_n: int = 240) -> dict:
    if output.exists():
        raise ValueError("snapshot destination already exists; choose a new destination")
    source = source.resolve()
    tracked = _git(source, "ls-files", "-s", "--", "results", "docs").splitlines()
    families: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    rejected = defaultdict(int)
    for row in tracked:
        meta, relative = row.split("\t", 1)
        path = confined(source, relative)
        if not path.is_file() or path.suffix not in SUFFIXES or not 0 < path.stat().st_size <= 65536:
            rejected["type_or_size"] += 1
            continue
        data = path.read_bytes()
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            rejected["not_utf8"] += 1
            continue
        digest = sha256(data)
        if digest in seen:
            rejected["duplicate_content"] += 1
            continue
        # Do not export likely secrets from source artifacts.
        if re.search(rb'(?i)(api[_-]?key|access[_-]?token|password)\s*[=:]\s*["\']?[A-Za-z0-9_-]{12,}', data):
            rejected["possible_secret"] += 1
            continue
        seen.add(digest)
        family = "/".join(Path(relative).parts[:2])
        git_blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        families[family].append({"path": relative, "sha256": digest, "size_bytes": len(data), "family": family,
                                 "source_git_blob": meta.split()[1], "matches_tracked_blob": git_blob == meta.split()[1]})
    # Round robin prevents a large campaign from consuming the file budget.
    selected = []
    depth = 0
    while len(selected) < n_files:
        next_rows = [families[f][depth] for f in sorted(families) if len(families[f]) > depth]
        if not next_rows:
            raise ValueError(f"only {len(selected)} unique eligible files; cannot reach {n_files}")
        selected.extend(next_rows[:n_files - len(selected)])
        depth += 1
    candidates: dict[str, list[dict]] = defaultdict(list)
    for row in selected:
        candidates[row["family"]].extend(scalar_questions(row["path"], confined(source, row["path"]).read_bytes(), row["family"]))
    # Assign whole experiment families before selecting questions. No query tuning.
    dev_families = {f for f in candidates if int(sha256(f.encode())[:8], 16) % 3 == 0}
    gold = []
    for split, limit in (("dev", dev_n), ("test", test_n)):
        groups = {f: sorted(qs, key=lambda q: q["id"]) for f, qs in candidates.items()
                  if (f in dev_families) == (split == "dev") and qs}
        rows = []
        i = 0
        while len(rows) < limit:
            wave = [groups[f][i] for f in sorted(groups) if len(groups[f]) > i]
            if not wave:
                raise ValueError(f"not enough {split} questions: {len(rows)} < {limit}")
            rows.extend(wave[:limit - len(rows)])
            i += 1
        for q in rows:
            q["split"] = split
        gold.extend(rows)
    output.mkdir(parents=True)
    corpus = output / "corpus"
    for row in selected:
        target = confined(corpus, row["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(confined(source, row["path"]).read_bytes())
    aggregate = sha256(json.dumps([(r["path"], r["sha256"]) for r in sorted(selected, key=lambda r: r["path"])], separators=(",", ":")).encode())
    manifest = {"schema": SCHEMA, "source_remote": _git(source, "remote", "get-url", "origin"),
                "source_commit": _git(source, "rev-parse", "HEAD"), "aggregate_sha256": aggregate,
                "n_files": len(selected), "n_families": len({r["family"] for r in selected}),
                "selection": "unique UTF-8 tracked experiment artifacts, <=64KiB, family round-robin",
                "rejected": dict(rejected), "files": sorted(selected, key=lambda r: r["path"]),
                "labels": {"generated": len(gold), "human_reviewed": 0, "dev": dev_n, "test": test_n},
                "limitations": ["generated field lookup labels, not human answer accuracy", "no real-robot deployment", "snapshot does not establish newest-version semantics"]}
    question_bytes = "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in gold).encode()
    manifest["questions_sha256"] = sha256(question_bytes)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "questions.jsonl").write_bytes(question_bytes)
    verify(output)
    return manifest


def verify(snapshot: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((snapshot / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported manifest schema")
    corpus = snapshot / "corpus"
    paths = [r["path"] for r in manifest["files"]]
    if len(set(paths)) != len(paths) or len(paths) != manifest["n_files"]:
        raise ValueError("invalid file inventory")
    disk = {str(p.relative_to(corpus)) for p in corpus.rglob("*") if p.is_file()}
    if disk != set(paths):
        raise ValueError("corpus file inventory mismatch")
    for row in manifest["files"]:
        data = confined(corpus, row["path"]).read_bytes()
        if sha256(data) != row["sha256"] or len(data) != row["size_bytes"]:
            raise ValueError(f"corpus changed: {row['path']}")
    aggregate = sha256(json.dumps([(r["path"], r["sha256"]) for r in sorted(manifest["files"], key=lambda r: r["path"])], separators=(",", ":")).encode())
    if aggregate != manifest["aggregate_sha256"]:
        raise ValueError("aggregate hash mismatch")
    question_bytes = (snapshot / "questions.jsonl").read_bytes()
    if sha256(question_bytes) != manifest["questions_sha256"]:
        raise ValueError("questions changed since freeze")
    questions = [json.loads(line) for line in question_bytes.decode().splitlines() if line]
    family_splits: dict[str, str] = {}
    ids = set()
    for q in questions:
        if q["id"] in ids or q["split"] not in {"dev", "test"} or q["label_origin"] != "generated_from_top_level_json_scalar" or q["human_reviewed"]:
            raise ValueError("invalid generated question metadata")
        ids.add(q["id"])
        if family_splits.setdefault(q["family"], q["split"]) != q["split"]:
            raise ValueError("experiment family leaks across splits")
        evidence = q["evidence"]
        if q["relevant_paths"] != [evidence["path"]] or evidence["path"] not in paths:
            raise ValueError("question relevance not in snapshot")
        row = next(r for r in manifest["files"] if r["path"] == evidence["path"])
        if q["family"] != row["family"]:
            raise ValueError("question family does not match file")
        data = confined(corpus, evidence["path"]).read_bytes()
        a, b = evidence["start_byte"], evidence["end_byte"]
        if not 0 <= a < b <= len(data) or sha256(data) != evidence["sha256"] or json.loads(data[a:b]) != q["answer"]:
            raise ValueError("answer evidence mismatch")
    if len(questions) != manifest["labels"]["generated"]:
        raise ValueError("question count mismatch")
    for split in ("dev", "test"):
        if sum(q["split"] == split for q in questions) != manifest["labels"][split]:
            raise ValueError("split count mismatch")
    return manifest, questions


def group_bootstrap(rows: list[dict], field: str, *, paired_field: str | None = None, n_boot: int = 2000) -> dict:
    """Sample whole families with replacement; report the query-weighted mean."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        grouped[r["family"]].append(r[field] - r[paired_field] if paired_field else r[field])
    groups = list(grouped.values())
    if not groups:
        raise ValueError("bootstrap requires rows")
    rng = random.Random(20261007)
    means = []
    for _ in range(n_boot):
        values = [v for _ in groups for v in rng.choice(groups)]
        means.append(sum(values) / len(values))
    point = sum(v for g in groups for v in g) / sum(map(len, groups))
    low, high = np.quantile(means, [.025, .975])
    return {"mean": point, "ci95": [float(low), float(high)], "n_families": len(groups), "resampling_unit": "experiment_family"}


def evaluate(snapshot: Path, output: Path) -> dict:
    manifest, questions = verify(snapshot)
    manifest_fingerprint = sha256((snapshot / "manifest.json").read_bytes())
    app_root = Path(__file__).resolve().parents[1]
    code_paths = ("eval/robotics.py", "eval/metrics.py", "retrieval/bm25.py", "retrieval/context.py")
    code_sha256 = {p: sha256((app_root / p).read_bytes()) for p in code_paths}
    paths = [r["path"] for r in manifest["files"]]
    texts = [confined(snapshot / "corpus", p).read_text() for p in paths]
    configs = {"bm25_body": texts, "bm25_path_context": [path_context(p) + "\n" + t for p, t in zip(paths, texts)]}
    rows = [{"id": q["id"], "family": q["family"], "split": q["split"], "label_origin": q["label_origin"], "configs": {}} for q in questions]
    builds = {}
    for name, documents in configs.items():
        start = time.perf_counter()
        index = BM25Index().fit(paths, documents)
        builds[name] = (time.perf_counter() - start) * 1000
        for q, row in zip(questions, rows):
            start = time.perf_counter()
            retrieved = [p for p, _ in index.search(q["question"], k=50)]
            latency = (time.perf_counter() - start) * 1000
            row["configs"][name] = {**score_query(retrieved, q["relevant_paths"], n_files=len(paths)), "retrieved": retrieved, "latency_ms": latency}
    summary = {}
    for split in ("dev", "test"):
        subset = [r for r in rows if r["split"] == split]
        summary[split] = {}
        for name in configs:
            scored = [{"family": r["family"], **r["configs"][name]} for r in subset]
            summary[split][name] = {m: group_bootstrap(scored, m) for m in ("ndcg@10", "mrr@10", "recall@5", "recall@10", "recall@50")}
            summary[split][name]["warm_search_ms"] = {"p50": float(np.quantile([r["latency_ms"] for r in scored], .5)), "p95": float(np.quantile([r["latency_ms"] for r in scored], .95))}
            summary[split][name]["random_recall@10"] = float(np.mean([r["random_recall@10"] for r in scored]))
        paired = [{"family": r["family"], "context": r["configs"]["bm25_path_context"]["ndcg@10"], "body": r["configs"]["bm25_body"]["ndcg@10"]} for r in subset]
        summary[split]["paired_delta_ndcg@10"] = group_bootstrap(paired, "context", paired_field="body")
    # Refuse a mixed run if either the data or executing source changed.
    verify(snapshot)
    if sha256((snapshot / "manifest.json").read_bytes()) != manifest_fingerprint:
        raise ValueError("manifest changed during evaluation")
    if code_sha256 != {p: sha256((app_root / p).read_bytes()) for p in code_paths}:
        raise ValueError("evaluator source changed during evaluation")
    result = {"schema": SCHEMA, "snapshot_sha256": manifest["aggregate_sha256"], "source_commit": manifest["source_commit"],
              "questions_sha256": manifest["questions_sha256"], "manifest_sha256": manifest_fingerprint, "code_sha256": code_sha256,
              "n_files": manifest["n_files"], "n_questions": len(questions), "n_families": manifest["n_families"],
              "labels": manifest["labels"], "runtime": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(), "host": socket.gethostname(), "device": "CPU", "backend": "BM25, no neural model"},
              "build_ms": builds, "summary": summary, "per_query": rows,
              "interpretation": "Generated source-file lookup, not human-reviewed answer correctness; warm offline search excludes HTTP and model inference.",
              "configuration_policy": "two fixed configurations declared before test evaluation; no parameter search"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("freeze")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--files", type=int, default=600)
    run = sub.add_parser("evaluate")
    run.add_argument("--snapshot", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args.source, args.output, n_files=args.files)
        print(json.dumps({k: result[k] for k in ("n_files", "n_families", "aggregate_sha256", "labels")}))
    elif args.command == "verify":
        m, q = verify(args.snapshot)
        print(json.dumps({"verified": True, "n_files": m["n_files"], "n_questions": len(q)}))
    else:
        result = evaluate(args.snapshot, args.output)
        print(json.dumps({"n_files": result["n_files"], "n_questions": result["n_questions"], "test": result["summary"]["test"]}, indent=2))


if __name__ == "__main__":
    main()
