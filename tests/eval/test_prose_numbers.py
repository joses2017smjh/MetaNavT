"""D36: numbers in the prose of the engineering reference are checked against the files they cite
(app/eval/prose_numbers.py). The synthetic documents below pin the rules; the last tests run the real reference."""

import json
import re
import shutil
from pathlib import Path

from app.eval import prose_numbers as pn

ROOT = Path(__file__).resolve().parents[2]


def _root(tmp_path: Path) -> Path:
    (tmp_path / "bench").mkdir()
    (tmp_path / "bench" / "r.json").write_text(json.dumps({
        "n_docs": 5633,
        "gain": 0.46116,
        "drop": -0.0204,
        "level": 0.95,
        "runs": [{"concurrency": 1, "p95_ms": 251.686}, {"concurrency": 16, "p95_ms": 15557.068}],
        "rows": [{"config": "a/b", "ndcg": 0.3555}, {"config": "c", "ndcg": 0.3554}],
        "ids": ["q1", "q2", "q3"],
        "name": "text",
    }))
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "x.py").write_text('RRF_K = 60\nMAX = 20_000\nos.getenv("RERANK_DEPTH", "20")\n')
    return tmp_path


def _problems(text: str, root: Path, phrases=()) -> list[str]:
    return [p.message for p in pn.check(text, root, not_results=tuple(phrases))]


def test_a_cited_number_matches_at_the_precision_the_prose_prints(tmp_path):
    root = _root(tmp_path)
    text = ("It indexes **5,633 nodes**; p95 is 252 / 15.6 seconds at concurrency 1 and 16<!-- num: bench/r.json#/n_docs; "
            "#/runs/[concurrency=1]/p95_ms; #/runs/[concurrency=16]/p95_ms|/1000; #/runs/0/concurrency; #/runs/1/concurrency -->.\n"
            "The paired difference is +0.461<!-- num: bench/r.json#/gain --> and -0.020<!-- num: bench/r.json#/drop -->, "
            "95%<!-- num: bench/r.json#/level|*100 --> intervals, 3<!-- num: bench/r.json#/ids|len --> ids.\n"
            "Rows `0.3555` and 0.3554<!-- num: bench/r.json#/rows/[config=a~1b]/ndcg; #/rows/[config=c]/ndcg -->; "
            "k = 60, 20,000 and 20<!-- num: app/x.py:\"RRF_K = {}\"; app/x.py:\"MAX = {}\"; app/x.py:\"RERANK_DEPTH\\\", \\\"{}\\\"\" -->.\n")
    assert _problems(text, root) == []


def test_a_drifted_number_in_the_prose_fails(tmp_path):
    root = _root(tmp_path)
    for wrong in ("5,634", "5,633.5", "5,632"):
        problems = _problems(f"It indexes {wrong} nodes<!-- num: bench/r.json#/n_docs -->.\n", root)
        assert len(problems) == 1 and "differs from its source" in problems[0], (wrong, problems)
    assert _problems("drop +0.020<!-- num: bench/r.json#/drop -->\n", root)  # the sign is part of the number


def test_a_drifted_value_in_the_cited_file_fails(tmp_path):
    root = _root(tmp_path)
    text = "It indexes 5,633 nodes<!-- num: bench/r.json#/n_docs -->.\n"
    assert _problems(text, root) == []
    blob = json.loads((root / "bench" / "r.json").read_text())
    blob["n_docs"] = 5634
    (root / "bench" / "r.json").write_text(json.dumps(blob))
    assert any("differs from its source" in p for p in _problems(text, root))


def test_a_citation_that_does_not_resolve_fails(tmp_path):
    root = _root(tmp_path)
    for spec in ("bench/r.json#/missing", "bench/missing.json#/n_docs", "bench/r.json#/rows/[config=nope]/ndcg",
                 "bench/r.json#/rows/7/ndcg", "bench/r.json#/name", "bench/r.json#/runs", "#/n_docs",
                 'app/x.py:"NOPE = {}"', "../outside.json#/x", "bench/r.json#/n_docs|len"):
        problems = _problems(f"It indexes 5,633 nodes<!-- num: {spec} -->.\n", root)
        assert problems and "does not resolve" in problems[0], (spec, problems)
    assert any("cannot read a spec" in p for p in _problems("5,633<!-- num: bench/r.json n_docs -->\n", root))


def test_every_number_is_cited_or_classified(tmp_path):
    root = _root(tmp_path)
    uncited = _problems("The gate held 97 of the questions.\n", root)
    assert len(uncited) == 1 and "neither cited" in uncited[0]
    assert _problems("Run 47 is current.\n", root, [("Run 47", "a corpus id")]) == []
    stale = _problems("Nothing here.\n", root, [("Run 47", "a corpus id")])
    assert stale and "no longer classifies" in stale[0]
    # a citation claims only the numbers before it on its line, after the previous citation
    assert any("neither cited" in p for p in _problems("12 and 5,633<!-- num: bench/r.json#/n_docs -->\n", root))
    assert any("only 1 precede" in p for p in _problems("5,633<!-- num: bench/r.json#/n_docs; #/n_docs -->\n", root))


def test_a_citation_at_the_start_of_a_line_fails(tmp_path):
    root = _root(tmp_path)
    problems = _problems("It indexes 5,633 nodes\n<!-- num: bench/r.json#/n_docs -->\n", root)
    assert any("same line" in p for p in problems) and any("neither cited" in p for p in problems)


def test_rules_for_names_markers_status_codes_and_code(tmp_path):
    root = _root(tmp_path)
    names = ("nDCG@10 and Recall@50 with fp16 on qwen2.5:7b and llama3.1:8b, an H100 80GB, k1=0.9, hops=1, run_047, "
             "v2, M17, PR #74, `k=60`, `1e-5` and `--num_pairs=3`.\n")
    assert _problems(names, root) == []
    assert _problems("1. **Scan.** First.\n2. Then.\n", root) == []
    assert _problems("It answers 503 with the text, then 200; a bad body is 422 (415, 413, 409).\n", root) == []
    # a status-like number that is a measurement is not a status code
    for text in ("p95 422 ms.\n", "p95 is 252 / 422 / 1,319 ms.\n", "It took ~1.3 GB.\n", "at 6x the wall time.\n",
                 "denies 220/220 cases.\n", "the gold `0.938 → 0.986` row.\n"):
        assert any("neither cited" in p for p in _problems(text, root)), text
    fenced = "```\nretrieve 50, rerank 8\n```\n<!-- bench-table:start (generated) -->\n| a | 0.493 |\n<!-- bench-table:end -->\n"
    assert _problems(fenced, root) == []


def test_matches_reads_the_precision_off_the_prose():
    assert pn.matches("0.938", 0.9382) and not pn.matches("0.93", 0.9382) and pn.matches("0.9", 0.9382)
    assert pn.matches("1,319", 1318.985) and pn.matches("+0.461", 0.46116) and not pn.matches("+0.461", -0.46116)
    assert pn.matches("0.000", -0.0001) and pn.matches("20,000", 20000.0) and not pn.matches("20", 20.4 + 0.2)


# ------------------------------------------------------------------ the engineering reference


def _reference() -> str:
    return (ROOT / pn.DOC).read_text(encoding="utf-8")


def test_the_engineering_reference_numbers_match_their_sources():
    """Every number in the prose of doc/engineering-reference.md is cited and equals its source, or is on the reviewed
    list of numbers that are not results. Fenced code is not checked; README.md and PLAN.md prose are not covered here."""
    problems = pn.check(_reference())
    assert problems == [], "\n".join(map(str, problems)) + "\n\nfix the prose or its citation (app/eval/prose_numbers.py)"


def test_a_drifted_number_in_the_reference_fails():
    text = _reference()
    match = re.search(r"\*\*(\d[\d,]*) nodes\*\*", text)
    assert match, "the reference no longer states the robotics node count this test edits"
    drifted = text.replace(match.group(0), f"**{int(match.group(1).replace(',', '')) + 1:,} nodes**", 1)
    assert any("differs from its source" in p.message for p in pn.check(drifted))


def _cited_files(text: str) -> set[str]:
    files = set()
    for _, line in pn.prose_lines(text):
        for comment in pn.CITATION.finditer(line):
            files |= {spec.get("file") or spec.get("src") for spec in pn.parse_specs(comment.group("body"))} - {None}
    return files


def test_a_drifted_value_in_a_cited_results_file_fails(tmp_path):
    """A results file the reference cites, changed in a copy of the cited files: the check fails on those numbers."""
    rel = "bench/results/trusted_actions_v2.json"
    text = _reference()
    cited = _cited_files(text)
    assert rel in cited and "bench/results/main.json" in cited and "app/engine/retriever.py" in cited
    for name in cited:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / name, tmp_path / name)
    assert pn.check(text, tmp_path) == []
    blob = json.loads((ROOT / rel).read_text())
    blob["attack_cases"] += 1
    (tmp_path / rel).write_text(json.dumps(blob))
    problems = [p.message for p in pn.check(text, tmp_path)]
    assert problems and all("differs from its source" in p and "attack_cases" in p for p in problems)


def test_the_readme_prose_numbers_match_their_sources():
    """The README's prose outside its generated blocks (the evidence table, the fixture size) is checked the same way."""
    problems = pn.check((ROOT / pn.README).read_text(encoding="utf-8"), doc=pn.README)
    assert problems == [], "\n".join(map(str, problems))
    assert set(pn.NOT_RESULTS) == {pn.DOC, pn.README}


def test_a_drifted_number_in_the_readme_fails():
    text = (ROOT / pn.README).read_text(encoding="utf-8")
    assert "**5,633" not in text and "5,633 nodes" in text, "the README no longer states the node count this test edits"
    problems = pn.check(text.replace("5,633 nodes", "5,634 nodes", 1), doc=pn.README)
    assert problems and all(p.doc == pn.README for p in problems) and any("differs from its source" in p.message for p in problems)


def test_every_cited_file_is_tracked_by_git():
    """A citation into an ignored or untracked file passes here and fails on a fresh clone (CI)."""
    import subprocess

    if not (ROOT / ".git").exists():
        import pytest

        pytest.skip("not a git checkout")
    cited = sorted(_cited_files(_reference()) | _cited_files((ROOT / pn.README).read_text(encoding="utf-8")))
    tracked = subprocess.run(["git", "-c", "pack.threads=1", "-C", str(ROOT), "ls-files", "--", *cited],
                             capture_output=True, text=True, check=True).stdout.split()
    assert sorted(tracked) == cited, sorted(set(cited) - set(tracked))
