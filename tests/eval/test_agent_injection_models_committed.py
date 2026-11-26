"""M50's committed files: the probe file now, and once Tier 1 is merged, the results file, its episodes and sidecar, and
the comparison file (the brief's committed-file tests, leakage checks 1, 4, 10 and 11).

BAR is read out of the commit each record names (`git show <head>:app/eval/agent_injection_models.py`, then
ast.literal_eval), so a result generated under another bar, or a bar edited after the first commit that held it, fails
here. The results-file tests skip until `make bench-agent-injection-v3-2026-merge` has written the file; the probe
file is committed (before any shard), so its tests run now. Every other number is checked against the episodes the
files store, never typed here.
"""

import ast
import json
import re
import shutil
import subprocess
from datetime import datetime

import pytest

from app.eval import agent_injection as ai
from app.eval import agent_injection_models as m50
from app.eval import agent_injection_probe as probe_module
from app.eval.provenance import scrub_names

ROOT = m50.ROOT
PROBE = ROOT / m50.PROBE
RESULTS = ROOT / m50.RESULTS
EPISODES = ROOT / m50.EPISODES
COMPARE = ROOT / m50.COMPARE
MODULE = "app/eval/agent_injection_models.py"
MACHINE_ROOTS = ("/nfs/", "/home/", "/scratch/")


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def _commit_in_history(head):
    """Skip outside a git checkout or in a shallow clone that lacks `head` (CI fetches the full history for the eval
    job); in a full clone, `head` must be an ancestor of HEAD."""
    if shutil.which("git") is None or _git("rev-parse", "--is-inside-work-tree").stdout.strip() != "true":
        pytest.skip("not a git checkout: the commit a record names cannot be read")
    if _git("cat-file", "-e", f"{head}^{{commit}}").returncode != 0:
        if _git("rev-parse", "--is-shallow-repository").stdout.strip() == "true":
            pytest.skip(f"shallow clone without commit {head[:7]}: fetch the full history (git fetch --unshallow)")
        pytest.fail(f"commit {head[:7]} is not in the history")
    assert _git("merge-base", "--is-ancestor", head, "HEAD").returncode == 0, f"{head[:7]} is not an ancestor of HEAD"


def bar_at(head):
    """BAR as the module holds it at commit `head` (None when that commit's module holds no BAR)."""
    _commit_in_history(head)
    shown = _git("show", f"{head}:{MODULE}")
    if shown.returncode != 0:
        return None
    for node in ast.parse(shown.stdout).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "BAR":
            return ast.literal_eval(node.value)
    return None


def _module_commits():
    """(sha, committer time) of every commit that changed the module, oldest first."""
    lines = _git("log", "--reverse", "--format=%H %ct", "--", MODULE).stdout.splitlines()
    return [(sha, int(ct)) for sha, ct in (line.split() for line in lines if line.strip())]


def _utc(stamp):
    return datetime.fromisoformat(stamp).timestamp()


def _probe():
    return json.loads(PROBE.read_text())


def _private(text):
    """What in `text` would name the machine: a user or host name, or a shared or home filesystem path."""
    return [x for x in MACHINE_ROOTS if x in text] + (["a user or host name"] if scrub_names(text) != text else [])


def test_bar_was_committed_before_the_first_m50_model_call_and_never_edited_after():
    # leakage check 1: the first commit whose module holds BAR precedes every M50 model call (the probe's included),
    # and every later commit that changed the module holds the same BAR
    commits = _module_commits()
    holding = [(sha, ct) for sha, ct in commits if bar_at(sha) is not None]
    assert holding and all(bar_at(sha) == m50.BAR for sha, _ in holding)
    starts = [entry["started_utc"] for entry in _probe()["models"].values()]
    if RESULTS.exists():
        starts += [segment["started_utc"] for segment in json.loads(ai.sidecar(EPISODES).read_text())["segments"]]
    assert starts and holding[0][1] < min(_utc(s) for s in starts)


def test_the_probe_file_copies_bar_and_every_model_was_probed_at_one_clean_commit_holding_it():
    blob = _probe()
    assert blob["bar"] == m50.BAR
    assert {entry["git_head"] for entry in blob["models"].values()} == {blob["git_head"]}
    assert all(entry["working_tree_dirty"] is False for entry in blob["models"].values())
    assert bar_at(blob["git_head"]) == m50.BAR
    tier1 = m50.BAR["models"]["tier1"]
    assert list(blob["models"]) == [m for m in tier1 if m in blob["models"]]  # BAR's order
    assert blob["probe_questions"] == m50.BAR["data"]["probe_questions"]
    assert all(c["question"] in blob["probe_questions"] for e in blob["models"].values() for c in e["calls"])


def test_the_probe_decisions_are_bars_tables_applied_to_the_probes_own_records():
    # the reasoning case, the workers, the shard split, the parser-addition trigger and the request block every shard
    # must record, recomputed from the calls and batches the file stores
    for model, entry in _probe()["models"].items():
        stored = dict(entry["decisions"])
        request = stored.pop("request")
        assert probe_module.decisions(entry["calls"], entry["batches"]) == stored, model
        assert m50.request_for(entry) == request, model


def test_the_probe_file_names_no_test_question_outside_its_copy_of_bar():
    # D1: BAR's data block lists the ten test questions, so the copy under `bar` holds them; nothing else does
    rest = json.dumps({k: v for k, v in _probe().items() if k != "bar"})
    assert not [gold_id for gold_id, *_ in ai.USER_TASKS if gold_id in rest]


def test_the_probe_file_holds_no_user_or_host_name_or_machine_path():
    assert _private(PROBE.read_text()) == []


# ---- Tier 1's merged file (skipped until the merge has written it) ---------------------------------------------------

@pytest.fixture(scope="module")
def m50_run():
    """The M50 results file, its stored episodes and tasks, its sidecar, and today's scores of those episodes, loaded
    and rescored once."""
    if not RESULTS.exists():
        pytest.skip(f"{m50.RESULTS} is not written yet (make bench-agent-injection-v3-2026-merge after the shards)")
    blob = json.loads(RESULTS.read_text())
    tasks, rows = ai.stored_episodes(EPISODES, k=4)
    return blob, tasks, rows, json.loads(ai.sidecar(EPISODES).read_text()), ai.rescore(tasks, rows)


def _compare():
    if not COMPARE.exists():
        pytest.skip(f"{m50.COMPARE} is not written yet (make bench-agent-injection-v3-2026-compare)")
    return json.loads(COMPARE.read_text())


def test_every_m50_segment_ran_at_the_one_generation_commit_holding_bar_and_each_shard_chains(m50_run):
    # one generation SHA per results file (the merge scores at it too); each shard's segments chain (a resume starts
    # from the rows without an error); every (model, configuration, task, trial) once, seed = trial, in MERGE_ORDER
    blob, tasks, rows, meta, _ = m50_run
    assert blob["bar"] == m50.BAR and blob["working_tree_dirty"] is False and blob["benchmark"] == ai.BENCHMARK_V3
    assert blob["episodes_generated_by"] == meta and meta["merged"]["order"] == ai.MERGE_ORDER
    segments = meta["segments"]
    assert {s["git_head"] for s in segments} == {blob["git_head"]} and bar_at(blob["git_head"]) == m50.BAR
    assert all(s["working_tree_dirty"] is False and "--bar m50" in s["command"] and "--no-summary" in s["command"]
               for s in segments)
    chains = {}
    for segment in segments:
        chains.setdefault(segment["shard"], []).append(segment)
    assert [f["file"] for f in meta["merged"]["shards"]] == list(chains)
    for shard in meta["merged"]["shards"]:
        chain = chains[shard["file"]]
        assert chain[0]["episodes_before"] == 0 and chain[-1]["episodes_after"] == shard["n_episodes"]
        assert all(b["episodes_before"] == a["episodes_after"] - a["errors_after"] for a, b in zip(chain, chain[1:]))
    keys = [ai._episode_key(r) for r in rows]
    models = blob["config"]["models"]
    defenses = ai.BENCHMARK_DEFENSES[ai.BENCHMARK_V3]
    assert models == [m for m in m50.BAR["models"]["tier1"] if m in models]
    assert sorted(keys) == sorted((m, d, t.task_id, trial) for m in models for t in tasks for d in defenses
                                  for trial in range(4))
    assert len(set(keys)) == len(keys) and all(r["seed"] == r["trial"] for r in rows)
    rank = {t.task_id: i for i, t in enumerate(tasks)}
    assert keys == [ai._episode_key(r) for r in sorted(rows, key=lambda r: (
        models.index(r["model"]), rank[r["task_id"]], defenses.index(r["defense"]), r["trial"]))]


def test_every_segment_request_is_bars_rules_applied_to_the_committed_probe(m50_run):
    # leakage check 1: the request block each segment recorded is the one BAR's rules give from the probe file, and the
    # results file's config holds exactly that block per model
    blob, _, _, meta, _ = m50_run
    probe = _probe()
    expected = {m: m50.request_for(probe["models"][m]) for m in blob["config"]["models"]}
    assert all(s["request"] == expected[m] for s in meta["segments"] for m in s["models"])
    assert blob["config"]["request"] == expected
    assert blob["config"]["think"] == {m: block["think"] for m, block in expected.items()}


def test_the_m50_file_rebuilds_from_its_stored_episodes_and_ran_the_committed_tasks(m50_run):
    # the scoring code is frozen: today's harness rebuilds every score, the summary and the paired deltas exactly, and
    # the carriers and planted text are the committed v3 file's (task sha256s)
    blob, tasks, rows, _, rescored = m50_run
    assert json.loads(json.dumps(ai.task_rows(tasks, ai.BENCHMARK_V3))) == blob["tasks"]
    v3_tasks = json.loads((ROOT / m50.RESULTS_V3).read_text())["tasks"]
    assert [t["injection_sha256"] for t in blob["tasks"]] == [t["injection_sha256"] for t in v3_tasks]
    assert [r["score"] for r in rescored] == [r["score"] for r in rows]
    assert ai.summarize(tasks, rescored, k=4) == blob["summary"]
    assert ai.paired_deltas(tasks, rescored, defenses=ai.BENCHMARK_DEFENSES[ai.BENCHMARK_V3]) == blob["paired_deltas"]
    assert blob["n_episodes"] == len(rows) and blob["n_episode_errors"] == sum(bool(r.get("error")) for r in rows)


def test_every_model_with_an_interval_has_all_its_episodes_and_no_error_row(m50_run):
    # the error rule: a verdict needs every key and 0 error rows; a model the outcome table stopped holds no interval
    _, _, rows, _, _ = m50_run
    compare = _compare()
    for model, state in compare["states"].items():
        mine = [r for r in rows if r["model"] == model] if model in m50.BAR["models"]["tier1"] else None
        entries = [v for v in compare["verdicts"] if v["config"].endswith(f"@{model}")]
        if any(v["interval"] is not None for v in entries):
            assert state["state"] is None and state["error_rows"] == 0 and state["episodes"] == state["expected"]
            if mine is not None:
                assert len(mine) == state["expected"] and not any(r.get("error") for r in mine)
        else:  # no interval at all: the outcome table stopped the model at cannot run or incomplete
            assert state["state"] is not None


def test_the_described_counts_equal_what_the_episodes_record(m50_run):
    # leakage checks 4 and 5: reasoning calls, calls stopped at the cap and recovered tool calls per model and
    # configuration are the sums of the episodes' own keys; marker counts are of the texts each episode stores
    _, _, rows, _, _ = m50_run
    described = _compare()["described"]
    for model in m50.BAR["models"]["tier1"]:
        for defense, block in described.get(model, {}).items():
            mine = [r for r in rows if r["model"] == model and r["defense"] == defense]
            got = block["counts"]
            assert got["episodes"] == len(mine)
            assert got["thinking_calls"] == sum(r["thinking_calls"] for r in mine)
            assert got["length_stops"] == sum(r["length_stops"] for r in mine)
            assert got["tool_calls_recovered_from_text"] == sum(bool(c.get("recovered_from_text"))
                                                                for r in mine for c in r["tool_calls"])
            texts = [t for r in mine for t in m50.stored_texts(r)]
            assert got["texts_with_reasoning_markers"] == sum(
                any(x in t for x in m50.BAR["reasoning"]["markers"]) for t in texts)
            assert got["texts_with_unparsed_call_forms"] == sum(
                any(x in t for x in m50.BAR["tool_calls"]["unparsed_markers"]) for t in texts)


def test_the_m50_files_hold_no_user_or_host_name_and_machine_paths_only_where_a_model_wrote_them(m50_run):
    # leakage check 11: no user or host name anywhere, and no shared or home path in the results file, the sidecar and
    # the comparison file; in the episodes, such a path may appear only as text a model wrote in that episode (an
    # answer, a draft, a tool call's arguments or recovered text), or where a tool result echoes it
    from app.eval.provenance import _ABS_PATH_RE
    _, _, rows, _, _ = m50_run
    for path in (RESULTS, ai.sidecar(EPISODES), COMPARE):
        if path.exists():
            assert _private(path.read_text()) == [], path.name
    text = EPISODES.read_text()
    assert scrub_names(text) == text
    stray = []
    for row in rows:
        strings = list(_strings(row))
        written = "\n".join(s for where, s in strings if MODEL_WRITTEN.match(where))
        stray += [(*ai._episode_key(row), where, m.group(0)) for where, s in strings if any(x in s for x in MACHINE_ROOTS)
                  for m in _ABS_PATH_RE.finditer(s) if any(x in m.group(0) for x in MACHINE_ROOTS)
                  and m.group(0) not in written]
    assert stray == []


MODEL_WRITTEN = re.compile(r"^(answer|answer_check\.drafts\[\d+\]\.answer|tool_calls\[\d+\]\.(arguments(\..*)?|recovered_text))$")


def _strings(value, path=""):
    """(where, text) for every string in an episode row, `where` as answer_check.drafts[0].answer."""
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(item, f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _strings(item, f"{path}[{i}]")
