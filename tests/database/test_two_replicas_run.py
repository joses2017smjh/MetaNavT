"""C56: the two-replica run (app/eval/two_replicas.py, make two-replicas) against a live Postgres: two uvicorn
processes on one database, one index step, plans, decisions and grants shared. The committed results file comes from
the make target; this runs the same code into a temporary directory so it cannot rot unnoticed. Needs PG_TEST_DSN
(a server where the run may create a database); skipped otherwise.
"""

import json
import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("PG_TEST_DSN"), reason="PG_TEST_DSN not set")


def test_two_api_processes_on_one_database_share_plans_decisions_and_grants(tmp_path):
    from app.eval.two_replicas import results, run

    data = results(run(os.environ["PG_TEST_DSN"], tmp_path))
    assert data["passed"] is True, json.dumps(data["observations"], indent=1)
    names = [o["name"] for o in data["observations"]]
    assert names == ["index_once", "both_serve_retrieve", "plan_shared", "refused_without_a_valid_capability",
                     "approved_on_the_other", "replay_after_apply", "rejection_shared", "burnt_grant_refused_elsewhere",
                     "decision_log_shared"]
    refused = next(o for o in data["observations"] if o["name"] == "refused_without_a_valid_capability")
    assert set(refused["statuses"].values()) == {403}
    text = json.dumps(data)
    assert str(tmp_path) not in text and os.environ["PG_TEST_DSN"] not in text
