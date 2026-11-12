"""bench/results/two_replicas.json (make two-replicas, app/eval/two_replicas.py): two API processes on one Postgres.

Reads the committed file only (no database): every observation the run records held, and the file names no path,
user, host, connection string, review or plan id.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "bench" / "results" / "two_replicas.json"
OBSERVATIONS = ["index_once", "both_serve_retrieve", "plan_shared", "refused_without_a_valid_capability",
                "approved_on_the_other", "replay_after_apply", "rejection_shared", "burnt_grant_refused_elsewhere",
                "decision_log_shared"]


def test_the_committed_two_replica_run_held_every_observation():
    data = json.loads(RESULTS.read_text())
    assert data["schema"] == "two-replicas-v1" and data["replicas"] == 2 and data["worktree_clean"] is True
    assert data["command"] == "python -m app.eval.two_replicas --out bench/results/two_replicas.json"
    assert [o["name"] for o in data["observations"]] == OBSERVATIONS
    assert all(o["passed"] is True and o["checks"] and all(o["checks"].values()) for o in data["observations"])
    assert data["passed"] is True and data["index_step"]["runs"] == 1 and data["index_step"]["exit"] == 0
    by_name = {o["name"]: o for o in data["observations"]}
    assert set(by_name["refused_without_a_valid_capability"]["statuses"].values()) == {403}
    assert by_name["burnt_grant_refused_elsewhere"]["refused_on_a_by"] == "ledger"
    assert data["environment"]["METANAVIT_APPROVAL_LEDGER"] == "postgres" and data["environment"]["INDEX_ON_START"] == "false"


def test_the_committed_two_replica_run_names_no_path_user_host_connection_string_or_plan():
    from app.eval.provenance import host_names, user_names

    text = RESULTS.read_text()
    assert "postgresql://" not in text and "dbname=" not in text
    assert not re.search(r'"(?:/|~/)[^"<]', text), "an absolute path"  # sanitized paths read <path>/name
    assert not [name for name in user_names() | host_names() if name in text]
    assert '"review"' not in text and not re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", text)
