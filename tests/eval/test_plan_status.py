"""B27: the repository PLAN.md's status column says what the results files and the served code say.

A status cell states a measured verdict as "`<row>` <verdict> vs `<reference>`" (more than one reference may follow
the same row: "`<row>` below zero vs `<a>` and covers zero vs `<b>`"), with the verdict word read off the paired
bootstrap interval of nDCG@10 the way the generated tables read it (app.eval.report.verdict); for the search policy
(search_policy.json, M44) the verdict is that of exact match, the metric of the bar declared before it was trained.
Every such claim must equal the verdict of that row against that reference in one of the results files below, and
every evaluated prototype arm of frontier.json, frontier_neural.json and rerank_arms.json, and the search policy's
headline row against each of its references, must have its claim, so a recorded loser cannot drop out of the plan. The item and status columns type no number: the numbers are in the generated tables
the cells link to, except a served default written as "`SETTING` (default N)", which must equal the default in the
served code. This test reads the code's defaults as text: CI's eval job does not install llama_index, which
app/engine/retriever.py imports.
"""

import json
import re
from pathlib import Path

import pytest

from app.eval import late_interaction as li
from app.eval.report import verdict

ROOT = Path(__file__).resolve().parents[2]
VERDICT_FILES = ("bench/results/main.json", "bench/results/neural.json", "bench/results/frontier.json",
                 "bench/results/frontier_neural.json", "bench/results/rerank_arms.json")
SEARCH_POLICY_FILE = "bench/results/search_policy.json"  # M44: its verdicts are of exact match (`em`), not nDCG@10


def late_interaction_files(root: Path = ROOT) -> tuple[str, ...]:
    """M49: the deciding attempt's late-interaction files (the three `primary` files hold the plan-flagged claims; the
    `extra` files their sensitivity and stretch entries), with dataset-qualified names (`fixture:`, `scifact:`, `fiqa:`)
    so a row name in three files keeps one verdict per (row, reference)."""
    attempt = li.deciding_attempt(root)
    return tuple(rel for d in li.DATASETS for part in li.PARTS if (root / (rel := li.results_file(d, part, attempt))).exists())


LATE_INTERACTION_FILES = late_interaction_files()
CLAIM = re.compile(r"`([^`]+)` ((?:(?:above zero|covers zero|below zero) vs `[^`]+`(?:,? and |, )?)+)")
SUB = re.compile(r"(above zero|covers zero|below zero) vs `([^`]+)`")
DEFAULT = re.compile(r"`([A-Z][A-Z0-9_]+)` \(default (\d+)\)")
NUMBER = re.compile(r"(?<![\w.@#/+\-:=~])\d+(?:[.,]\d+)*(?![\w])")


def status_rows(text: str) -> dict[str, tuple[str, str]]:
    """The '## Status' table: phase -> (item, status). Several rows may share a phase (two '5' rows): the cells are joined."""
    section = text.split("\n## Status", 1)[1].split("\n---", 1)[0]
    rows: dict[str, tuple[str, str]] = {}
    for line in section.splitlines():
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line)[1:-1]]
        if len(cells) != 3 or cells[0] in ("Phase", "") or set(cells[0]) <= {"-"}:
            continue
        item, status = rows.get(cells[0], ("", ""))
        rows[cells[0]] = ((item + " " + cells[1]).strip(), (status + " " + cells[2]).strip())
    return rows


def plan_claims(text: str) -> list[tuple[str, str, str, str]]:
    """(phase, row, verdict, reference) for every verdict claim in the status column."""
    out = []
    for phase, (_, status) in status_rows(text).items():
        for match in CLAIM.finditer(status):
            out += [(phase, match.group(1), v, ref) for v, ref in SUB.findall(match.group(2))]
    return out


def recorded_verdicts(root: Path = ROOT) -> dict[tuple[str, str], set[str]]:
    """(row, reference) -> the verdicts the results files hold for it (one, unless two files disagree)."""
    out: dict[tuple[str, str], set[str]] = {}
    for rel in VERDICT_FILES:
        blob = json.loads((root / rel).read_text())
        if "test_split" in blob:  # rerank_arms.json: each arm against the cross-encoder and against plain hybrid
            for scope in ("test_split", "all_questions"):
                for row in blob[scope]["results"]:
                    for block in row.get("deltas") or []:
                        out.setdefault((row["config"], block["reference"]), set()).add(verdict(block.get("ndcg@10")))
            continue
        for row in blob["results"]:
            dp = row.get("delta_vs_previous") or {}
            if dp.get("reference"):
                out.setdefault((row["config"], dp["reference"]), set()).add(verdict(dp.get("ndcg@10")))
    for v in search_policy_verdicts(root):
        out.setdefault((v["config"], v["reference"]), set()).add(v["interval"])
    for v in late_interaction_verdicts(root):
        out.setdefault((v["config"], v["reference"]), set()).add(v["interval"])
    return out


def late_interaction_verdicts(root: Path = ROOT) -> list[dict]:
    """M49: every `verdicts` entry of the deciding attempt's late-interaction files (nDCG@10 in the bar's scope)."""
    return [v for rel in late_interaction_files(root) for v in json.loads((root / rel).read_text())["verdicts"]]


def search_policy_verdicts(root: Path = ROOT) -> list[dict]:
    """The search policy's verdicts in exact match: its headline row (the seed mean) against each reference."""
    return [v for v in json.loads((root / SEARCH_POLICY_FILE).read_text())["verdicts"] if v["metric"] == "em"]


def evaluated_arms(root: Path = ROOT) -> set[tuple[str, str]]:
    """(arm, reference) pairs the plan must state: every frontier arm against its parent, every rerank arm against
    each reference it was scored against."""
    pairs = set()
    for rel in ("bench/results/frontier.json", "bench/results/frontier_neural.json"):
        blob = json.loads((root / rel).read_text())
        by = {row["config"]: row for row in blob["results"]}
        for arm in (blob.get("frontier") or {}).get("arms") or []:
            if arm in by:
                pairs.add((arm, by[arm]["delta_vs_previous"]["reference"]))
    blob = json.loads((root / "bench/results/rerank_arms.json").read_text())
    parents = {blob["candidates"]["config"]} | {row["config"] for scope in ("test_split", "all_questions")
                                                 for row in blob[scope]["results"] if not row.get("deltas")}
    parents.add("hybrid+bge-rerank@bge-small")  # the cross-encoder the arms are measured against
    for scope in ("test_split", "all_questions"):
        for row in blob[scope]["results"]:
            if row["config"] not in parents:
                pairs |= {(row["config"], block["reference"]) for block in row.get("deltas") or []}
    pairs |= {(v["config"], v["reference"]) for v in search_policy_verdicts(root)}
    pairs |= {(v["config"], v["reference"]) for v in late_interaction_verdicts(root) if v["plan"]}  # M49: PLAN_FLAGS, all eighteen
    return pairs


def served_default(name: str, root: Path = ROOT) -> int:
    """The default the served code gives a setting: os.getenv("NAME", "N") or os.getenv("NAME", os.getenv("OTHER", "N"))."""
    src = (root / "app/engine/retriever.py").read_text()
    match = re.search(rf'os\.getenv\("{name}", (?:os\.getenv\("\w+", )?"(\d+)"', src)
    assert match, f"no default for {name} in app/engine/retriever.py"
    return int(match.group(1))


def test_every_verdict_in_the_status_column_is_the_one_the_results_files_hold():
    text = (ROOT / "PLAN.md").read_text()
    recorded = recorded_verdicts()
    claims = plan_claims(text)
    assert claims, "the status column states no verdict"
    wrong = [(phase, row, said, ref, sorted(recorded.get((row, ref), set()) or {"no such row and reference"}))
             for phase, row, said, ref in claims if recorded.get((row, ref)) != {said}]
    assert wrong == [], "PLAN.md status claims that differ from the results files (phase, row, said, reference, recorded):\n" + "\n".join(map(str, wrong))


def test_every_evaluated_prototype_arm_has_its_verdict_in_the_status_column():
    stated = {(row, ref) for _, row, _, ref in plan_claims((ROOT / "PLAN.md").read_text())}
    missing = sorted(evaluated_arms() - stated)
    assert missing == [], "evaluated arms whose verdict PLAN.md's status column does not state:\n" + "\n".join(map(str, missing))


def test_status_and_item_columns_type_no_number_and_state_the_served_defaults():
    rows = status_rows((ROOT / "PLAN.md").read_text())
    defaults = {name: served_default(name) for name in ("RETRIEVE_K", "RERANK_DEPTH", "RERANK_TOP_N")}
    typed = []
    for phase, (item, status) in rows.items():
        for cell in (item, status):
            for name, value in DEFAULT.findall(cell):
                assert name in defaults and int(value) == defaults[name], (phase, name, value, defaults)
            # a checked default, inline code, a link target and the names "Tier 1" / "Tier 2" are not typed numbers
            bare = re.sub(r"`[^`]*`|\]\([^)]*\)|\bTier \d\b", " ", DEFAULT.sub(" ", cell))
            typed += [(phase, n) for n in NUMBER.findall(bare)]
    assert typed == [], f"numbers typed in PLAN.md's item or status column (phase, number): {typed}"
    status_1_2 = rows["1.2"][1]
    assert "`RERANK_DEPTH` (default" in status_1_2 and "`RERANK_TOP_N` (default" in status_1_2
    for name, value in DEFAULT.findall((ROOT / "PLAN.md").read_text()):  # a served default anywhere in the plan
        assert int(value) == defaults[name], (name, value, defaults)


def test_no_status_row_describes_the_served_reranker_as_a_top_50_rerank():
    rows = status_rows((ROOT / "PLAN.md").read_text())
    assert not [phase for phase, (item, status) in rows.items() if "top-50" in item or "top-50" in status]


def test_claim_parser_reads_lists_and_reports_a_drifted_verdict(tmp_path):
    """The parser takes several references after one row, and a claim whose verdict the files do not hold is found."""
    text = ("x\n## Status\n\n| Phase | Item | Status |\n|---|---|---|\n"
            "| 9.9 | thing | `a` below zero vs `b` and covers zero vs `c`; `d` above zero vs `e` |\n\n---\n")
    assert plan_claims(text) == [("9.9", "a", "below zero", "b"), ("9.9", "a", "covers zero", "c"), ("9.9", "d", "above zero", "e")]
    real = (ROOT / "PLAN.md").read_text()
    phase, row, said, ref = plan_claims(real)[0]
    other = next(v for v in ("above zero", "covers zero", "below zero") if v != said)
    drifted = real.replace(f"`{row}` {said} vs `{ref}`", f"`{row}` {other} vs `{ref}`", 1)
    recorded = recorded_verdicts()
    assert any(recorded.get((r, f)) != {v} for _, r, v, f in plan_claims(drifted))


def test_each_late_interaction_verdict_word_is_the_one_its_stored_interval_gives():
    """M44's check trusts a stored word; here the word is recomputed from the entry's own stored lo and hi."""
    entries = late_interaction_verdicts()
    if not entries:
        pytest.skip("no late-interaction results file yet")
    assert [v for v in entries if v["interval"] != verdict({"lo": v["lo"], "hi": v["hi"]})] == []
    flagged = {(v["config"], v["reference"]) for v in entries if v["plan"]}
    measured = {t for t in li.CANDIDATES if any(f"@{t}" in c or f"+{t}@" in c for c, _ in flagged)}
    assert flagged == {tuple(f) for f in li.PLAN_FLAGS if any(f"@{t}" in f[0] or f"+{t}@" in f[0] for t in measured) and
                       (ROOT / li.results_file(f[0].split(":")[0], "primary", li.deciding_attempt())).exists()}


def test_row_4_2_states_the_outcome_the_bar_computes_from_the_committed_files():
    """PLAN.md row 4.2's top-level outcome is the one late_interaction.bar_table reads off the deciding attempt's primary
    files: 'measured: ... passes', 'measured, not adopted' or 'labelled a demo', with every OPEN clause, every named model
    not measured, and every sensitivity or stretch row above zero on all three datasets."""
    if not late_interaction_files():
        pytest.skip("no late-interaction results file yet")
    status = status_rows((ROOT / "PLAN.md").read_text())["4.2"][1]
    out = li.bar_table(ROOT)["outcome"]
    assert status.startswith(out["word"]), (status[:80], out["word"])
    for clause in out["passing"] + out["open"] + out["outside_bar_all_three"]:
        assert clause in status, clause
    for tag in out["not_measured"]:
        assert f"`{tag}` not measured" in status
