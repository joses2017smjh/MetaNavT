"""M46: the memorisation and grader audit, from the cached LLM outputs (make llm-query-audit; CPU, after generation).

    python -m app.eval.llm_query_audit --data-dir <beir> --out bench/results/llm_query_audit.json

Every share here is descriptive; nothing decides anything. It reads the merged caches and parents.json.gz of
bench/llm_query, the fixture's gold questions and files, and (a named qrels reader, like scoring) the BEIR corpora and
test qrels, and the M46 results sidecars for the subgroup delta of (c).

  (a) fixture answers   per model and kind, the share of query-only outputs (HyDE, multi-query, decomposition, rewrite)
                        that contain the gold answer (judge.contains_bounded on lowercased text, the 80 simple_factual
                        questions), beside the same share for the question text itself
  (b) fixture paths     per model and kind, the share of query-only outputs that contain a gold relevant path or its
                        basename where the question text does not; beside it, the same share for a control: the basename
                        of a non-gold file in the gold file's directory (the first in sorted order)
  (c) BEIR verbatim     for each HyDE output, the longest run of word tokens it shares verbatim with any relevant
                        document, and with P0's top non-relevant document; the share with a run of SPAN tokens or more;
                        and the HyDE no-rerank delta (nDCG@10, mean) on the queries without such a run: a subgroup chosen
                        by a variable that reads the qrels, so biased and descriptive only
  (d) grader vs qrels   per model and dataset, the precision and recall of "yes" against the relevant units (fixture: a
                        chunk whose file is a gold path); unusable grades are counted and left out. SciFact's qrels are
                        sparse, so an unjudged document is not known to be irrelevant
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from app.eval import llm_query_spec as spec

ROOT = Path(__file__).resolve().parents[2]
SPAN = 13
OUT = "bench/results/llm_query_audit.json"
_TOKEN = re.compile(r"\w+")
DOES_NOT_SHOW = (
    "Pretraining exposure to BEIR or to this repository cannot be excluded: the audit bounds verbatim copying (c) and "
    "mentions of gold answers and paths (a, b) only.",
    "Whether bench/gold/questions.jsonl (in the repository since 2026-08-22) is in any model's training data is unknown.",
    "The subgroup delta of (c) is chosen by a variable that reads the qrels: it is biased and descriptive, not an estimate.",
    "The grader's precision and recall (d) are against qrels that are sparse on SciFact; they explain the corrective rows "
    "and are not a judge metric.",
)


def tokens(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def longest_common_run(a: Sequence[str], b: Sequence[str]) -> int:
    """The length of the longest run of tokens that appears verbatim in both sequences."""
    if not a or not b:
        return 0
    best, prev = 0, [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            if x == y:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def _share(hits: int, n: int) -> dict[str, Any]:
    return {"n": n, "hits": hits, "share": round(hits / n, 4) if n else None}


def _entries(root: Path, dataset: str, tag: str) -> list[dict]:
    from app.eval.llm_query_arms import cache_file, load_cache

    path = root / cache_file(dataset, tag)
    return load_cache(path) if path.exists() else []


def fixture_audit(root: Path, models: Sequence[str]) -> dict[str, Any]:
    from app.eval.gold import load_gold
    from app.eval.judge import contains_bounded

    gold = {q.id: q for q in load_gold(root / "bench" / "gold" / "questions.jsonl")}
    files = sorted(p.relative_to(root / "bench" / "corpus" / "files").as_posix() for p in (root / "bench" / "corpus" / "files").rglob("*") if p.is_file())
    by_dir: dict[str, list[str]] = {}
    for f in files:
        by_dir.setdefault(f.rsplit("/", 1)[0] if "/" in f else "", []).append(f)

    def control(paths: Sequence[str]) -> str | None:
        first = sorted(paths)[0]
        d = first.rsplit("/", 1)[0] if "/" in first else ""
        others = [f for f in by_dir.get(d, []) if f not in paths]
        return others[0].rsplit("/", 1)[-1] if others else None

    simple = [q for q in gold.values() if q.category == "simple_factual"]
    out: dict[str, Any] = {"answers": {"rule": "judge.contains_bounded on lowercased text; the simple_factual questions",
                                       "question_text": _share(sum(contains_bounded(q.question.lower(), q.answer.lower()) for q in simple), len(simple)),
                                       "models": {}},
                           "paths": {"rule": "a gold relevant path or its basename in the output where the question text has neither; control: the "
                                             "basename of the first other file (sorted) in the first gold file's directory, likewise",
                                     "models": {}}}
    for tag in models:
        entries = [e for e in _entries(root, "fixture", tag) if e["kind"] in spec.QUERY_KINDS and not e.get("error")]
        if not entries:
            continue
        ans, paths = {}, {}
        for kind in spec.QUERY_KINDS:
            mine = [e for e in entries if e["kind"] == kind]
            sf = [e for e in mine if gold[e["qid"]].category == "simple_factual"]
            ans[kind] = _share(sum(contains_bounded((e.get("reply") or "").lower(), gold[e["qid"]].answer.lower()) for e in sf), len(sf))
            hit = ctl = n_ctl = 0
            for e in mine:
                q, text = gold[e["qid"]], (e.get("reply") or "").lower()
                names = {p.lower() for p in q.relevant_paths} | {p.rsplit("/", 1)[-1].lower() for p in q.relevant_paths}
                qtext = q.question.lower()
                hit += int(any(n in text and n not in qtext for n in names))
                c = control(q.relevant_paths) if q.relevant_paths else None
                if c:
                    n_ctl += 1
                    ctl += int(c.lower() in text and c.lower() not in qtext)
            paths[kind] = {"gold": _share(hit, len(mine)), "control": _share(ctl, n_ctl)}
        out["answers"]["models"][tag] = ans
        out["paths"]["models"][tag] = paths
    return out


def beir_audit(root: Path, data_dir: Path, dataset: str, models: Sequence[str], parents: dict) -> dict[str, Any]:
    from app.eval import beir
    from app.eval.llm_query_arms import read_call

    corpus, _, qrels = beir.load_beir(Path(data_dir) / dataset, "test")  # a named qrels reader (audit)
    queries = {q["id"]: q for q in parents[dataset]["queries"]}
    side_path = root / f"bench/results/beir_{dataset}_llm_query_arms.per_query.json"
    side = json.loads(side_path.read_text()) if side_path.exists() else None
    out: dict[str, Any] = {"span": SPAN, "models": {}}
    for tag in models:
        entries = _entries(root, dataset, tag)
        if not entries:
            continue
        hyde = {e["qid"]: e for e in entries if e["kind"] == "hyde"}
        rel_runs, nonrel_runs = {}, {}
        for qid, e in hyde.items():
            text, _, _ = read_call(e, "hyde", queries[qid]["text"])
            t = tokens(text or "")
            rel = qrels.get(qid, {})
            rel_runs[qid] = max((longest_common_run(t, tokens(corpus[d])) for d in rel if d in corpus), default=0)
            top_nonrel = next((d for d in queries[qid]["p0_top"] if d not in rel), None)
            nonrel_runs[qid] = longest_common_run(t, tokens(corpus[top_nonrel])) if top_nonrel else 0
        n = len(hyde)
        block: dict[str, Any] = {"hyde_outputs": n,
                                 "relevant": {**_share(sum(1 for v in rel_runs.values() if v >= SPAN), n), "max_run": max(rel_runs.values(), default=0)},
                                 "p0_top_nonrelevant": {**_share(sum(1 for v in nonrel_runs.values() if v >= SPAN), n),
                                                        "max_run": max(nonrel_runs.values(), default=0)}}
        arm = spec.arm_name("hyde", "none", tag)
        if side and arm in side["scores"]:
            ids = side["ids"]
            d = [a - b for a, b in zip(side["scores"][arm]["ndcg@10"], side["scores"][spec.P0]["ndcg@10"])]
            keep = [i for i, q in enumerate(ids) if rel_runs.get(q, 0) < SPAN]
            block["hyde_delta_vs_P0"] = {"all": {"n": len(d), "mean": round(sum(d) / len(d), 4)},
                                         "without_a_run": {"n": len(keep), "mean": round(sum(d[i] for i in keep) / len(keep), 4) if keep else None},
                                         "label": "subgroup selected by a relevance-correlated variable (it reads qrels): biased, descriptive only"}
        out["models"][tag] = block
    return out


def grader_audit(root: Path, data_dir: Path, models: Sequence[str], parents: dict) -> dict[str, Any]:
    from app.eval.gold import load_gold
    from app.eval.llm_query_arms import read_call

    out: dict[str, Any] = {}
    gold = {q.id: q for q in load_gold(root / "bench" / "gold" / "questions.jsonl")}
    for dataset in spec.DATASETS:
        if dataset not in parents:
            continue
        if dataset == "fixture":
            relevant = lambda qid, unit: unit.split("::", 1)[0] in gold[qid].relevant_paths  # noqa: E731
        else:
            from app.eval import beir

            _, _, qrels = beir.load_beir(Path(data_dir) / dataset, "test")
            relevant = lambda qid, unit, q=qrels: unit in q.get(qid, {})  # noqa: E731
        texts = {q["id"]: q["text"] for q in parents[dataset]["queries"]}
        for tag in models:
            grades = [e for e in _entries(root, dataset, tag) if e["kind"] == "grade"]
            if not grades:
                continue
            tp = fp = fn = tn = unusable = 0
            for e in grades:
                value, status, _ = read_call(e, "grade", texts[e["qid"]])
                if value not in ("yes", "no"):
                    unusable += 1
                    continue
                r = relevant(e["qid"], e["unit"])
                tp += int(value == "yes" and r)
                fp += int(value == "yes" and not r)
                fn += int(value == "no" and r)
                tn += int(value == "no" and not r)
            out.setdefault(dataset, {})[tag] = {"graded": len(grades), "unusable": unusable, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                                                "precision_of_yes": round(tp / (tp + fp), 4) if tp + fp else None,
                                                "recall_of_yes": round(tp / (tp + fn), 4) if tp + fn else None}
    return out


def audit(root: Path, data_dir: Path, models: Sequence[str]) -> dict[str, Any]:
    from app.eval.llm_query_arms import load_parents
    from app.eval.provenance import command_line, git_sha, run_provenance

    parents = load_parents(root)
    blob: dict[str, Any] = {"git_sha": git_sha(root), **run_provenance(root=root), "timestamp": datetime.now(timezone.utc).isoformat(),
                            "command": command_line(), "models": list(models), "about": __doc__.split("\n\n", 2)[1].strip()}
    blob["fixture"] = fixture_audit(root, models)
    blob["beir_verbatim"] = {d: beir_audit(root, data_dir, d, models, parents) for d in ("scifact", "fiqa") if d in parents}
    blob["grader_vs_qrels"] = grader_audit(root, data_dir, models, parents)
    blob["does_not_show"] = list(DOES_NOT_SHOW)
    return blob


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--models", default=",".join(spec.BAR_MODELS))
    p.add_argument("--out", type=Path, default=ROOT / OUT)
    args = p.parse_args(argv)
    blob = audit(ROOT, args.data_dir, [m.strip() for m in args.models.split(",") if m.strip()])
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.write_text(json.dumps(blob, indent=1) + "\n")
    print(json.dumps({k: blob[k] for k in ("fixture", "beir_verbatim", "grader_vs_qrels")}, indent=1)[:4000])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
