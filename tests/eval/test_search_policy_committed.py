"""M44: the committed search-policy results file is what its sidecars compute, and it says what the bar declared.

bench/results/search_policy.json was assembled from the held-out arm files and the training runs on the HPC; its
sidecars keep every episode and every training curve. Here, on a CPU with no model: every episode is scored again,
every interval redrawn, the checkpoint each seed selected and the collapse rule re-read, the training corpora
rebuilt from their seeds (app.rl.results.check, `make search-policy-rescore`), and the file's declared texts are the
ones in the code.
"""

import json
from pathlib import Path

from app.rl import results as R

ROOT = Path(__file__).resolve().parents[2]


def _blob() -> dict:
    return json.loads((ROOT / R.RESULTS).read_text())


def test_the_committed_results_file_is_recomputed_exactly_from_its_sidecars():
    assert R.check(ROOT / R.RESULTS) == []


def test_every_arm_holds_every_held_out_question_and_the_verdict_is_the_declared_one():
    blob = _blob()
    rollouts = json.loads((ROOT / R.ROLLOUTS).read_text())
    held_out = blob["split"]["held_out_ids"]
    assert blob["split"]["n_held_out"] == len(held_out) == 68 and not set(held_out) & set(blob["split"]["tune_ids"])
    assert all(sorted(arm["rows"]) == sorted(held_out) for arm in rollouts["arms"].values())
    assert {R.UNTRAINED, R.SINGLE_SHOT} <= set(rollouts["arms"]) and any(a["kind"] == "trained" for a in rollouts["arms"].values())
    assert blob["bar"] == R.BAR and blob["reward"] == R.REWARD and blob["does_not_show"] == R.DOES_NOT_SHOW
    primary = [v for v in blob["verdicts"] if v["primary"]]
    assert len(primary) == 1 and primary[0]["metric"] == "em" and primary[0]["reference"] == R.UNTRAINED
    n_trained = sum(1 for a in rollouts["arms"].values() if a["kind"] == "trained")
    assert primary[0]["n_seeds"] == n_trained  # with several seeds the verdict is the seed mean's, never one seed's
    rows = {row["config"]: row for row in blob["results"]}
    assert primary[0]["config"] in rows and all(name in rows for name in rollouts["arms"])  # every arm keeps its row


def test_no_training_question_has_a_fixture_question_text_and_every_arm_ran_on_one_gpu_type():
    blob = _blob()
    leak = blob["leakage"]
    assert leak["n_kept_with_a_fixture_question_text"] == 0 and leak["variant_run_ids_also_in_fixture_v1"] == []
    assert blob["training_corpora"]["runs_whose_recorded_corpora_differ"] == []
    assert blob["stored_scores_differ_from_rescored"] == []
    assert blob["gpu"] and blob["precision"] and blob["device"]["device_name"] == blob["gpu"]
    assert all(not at["working_tree_dirty"] for at in blob["episodes_generated_at"].values())
