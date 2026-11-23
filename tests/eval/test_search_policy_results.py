"""M44: the search policy's results file is a function of its sidecars, and --check recomputes it without a model.

The arms here are scripted policies over the fixture with the hash embedder (no model, no torch): the tests check
the arithmetic of the results file, the declared rules (checkpoint selection, collapse, three seeds as one verdict)
and that a changed sidecar is reported. The committed results file is checked in test_search_policy_committed.py.
"""

import json
import shutil
from pathlib import Path

import pytest

from app.eval.gold import load_gold
from app.eval.rerank_arms import split_questions
from app.eval.stats import DEFAULT_N_BOOT, DEFAULT_SEED, paired_delta_ci, resample_indices
from app.rl import results as R
from app.rl.episode import corpus_env, greedy_episodes

ROOT = Path(__file__).resolve().parents[2]


class Answers:
    """Searches once with the run's name, then answers `value [first file shown]` for the questions it knows."""

    def __init__(self, known: dict[str, str]):
        self.known = known

    def begin(self, questions, max_searches):
        return [{"q": q, "shown": [], "n": 0} for q in questions]

    def extend(self, state, text):
        state["shown"].append(text)

    def generate(self, states, *, max_new_tokens, sample):
        out = []
        for state in states:
            state["n"] += 1
            if state["n"] == 1 and not state["shown"]:
                out.append(f"<search>{state['q']}</search>")
            else:
                path = state["shown"][-1].split("\n")[2].split(" [")[0] if state["shown"] else "none"
                out.append(f"<answer>{self.known.get(state['q'], 'unknown')} [{path}]</answer>")
        return out


@pytest.fixture(scope="module")
def world():
    gold = load_gold(ROOT / "bench" / "gold" / "questions.jsonl")
    _, test_ids = split_questions(gold)
    held = [gold[i] for i in test_ids]
    env = corpus_env(ROOT / "bench" / "corpus" / "files")
    simple = [q for q in held if q.category == "simple_factual"]

    def arm(kind, known, forced=False, adapter=None):
        rows = greedy_episodes(Answers({q.question: q.answer for q in known}), held, env, forced_first_search=forced)
        return {"kind": kind, "adapter": adapter, "max_searches": 1 if forced else 3, "prompt_sha256": "x", "seconds": 1.0,
                "command": "python -m app.rl.evaluate", "timestamp": "t", "rows": rows}

    def sha(tag):
        return {"sha256": {"adapter_config.json": tag, "adapter_model.safetensors": tag}}

    arms = {
        R.SINGLE_SHOT: arm("single_shot", simple[:10], forced=True),
        R.UNTRAINED: arm("untrained", simple[:8]),
        "policy-grpo/seed=0": arm("trained", simple[:30], adapter={"name": "grpo/seed=0", **sha("a"), "selected": {"step": 100}}),
        "policy-grpo/seed=1": arm("trained", simple[:20], adapter={"name": "grpo/seed=1", **sha("b"), "selected": {"step": 50}}),
        "policy-grpo/seed=2": arm("trained", simple[5:25], adapter={"name": "grpo/seed=2", **sha("c"), "selected": {"step": 100}}),
    }

    def run(tune_rewards, rewards, answered=1.0, loss=0.1):
        curve = [{"step": i, "loss": loss, "reward": r, "em": 0.5, "answered": answered, "rollout_s": 2.0, "update_s": 1.0} for i, r in enumerate(rewards)]
        tune = [{"step": s, "reward": r, "em": 0.1, "n_search": 1.0, "uncited_or_empty": 0.0, "seconds": 3.0, "per_question": {}} for s, r in tune_rewards]
        return {"config": {"algo": "grpo", "corpus_seeds": [1, 2], "seed": 0}, "curve": curve, "tune": tune, "adapter": sha("a")}

    training = {"runs": {
        "policy-grpo/seed=0": run([(0, 0.9), (50, 0.5), (100, 0.7)], [0.0] * 20 + [0.5] * 20),
        "policy-grpo/seed=1": run([(0, 0.2), (50, 0.6), (100, 0.6)], [0.5] * 20 + [0.45] * 20),
        "policy-grpo/seed=2": run([(0, 0.2), (50, 0.1), (100, 0.3)], [0.5] * 20 + [0.3] * 20),
    }}
    return {"gold": gold, "held": held, "rollouts": {"arms": arms}, "training": training}


def test_rows_deltas_and_the_seed_mean_are_what_the_stored_episodes_score(world):
    out = R.compute(world["rollouts"], world["training"], world["gold"])
    held, ids = world["held"], [q.id for q in world["held"]]
    assert out["split"]["held_out_ids"] == ids and out["split"]["n_held_out"] == 68 and out["stored_scores_differ_from_rescored"] == []
    assert out["split"]["held_out_gold_forms"]["single_value"] + out["split"]["held_out_gold_forms"]["composite"] == 68
    assert out["split"]["held_out_gold_forms"]["single_value"] == sum(1 for q in held if R.gold_form(q.answer) == "single_value") == 52
    assert [r["config"] for r in out["results"]] == [R.SINGLE_SHOT, R.UNTRAINED, "policy-grpo", "policy-grpo/seed=0", "policy-grpo/seed=1", "policy-grpo/seed=2"]
    by = {r["config"]: r for r in out["results"]}
    assert by[R.UNTRAINED]["ci"]["em"]["mean"] == round(8 / 68, 4) and by["policy-grpo/seed=0"]["ci"]["em"]["mean"] == round(30 / 68, 4)
    assert by["policy-grpo"]["kind"] == "trained, mean of seeds" and by["policy-grpo"]["seeds"] == ["policy-grpo/seed=0", "policy-grpo/seed=1", "policy-grpo/seed=2"]
    assert by["policy-grpo"]["ci"]["em"]["mean"] == round((30 + 20 + 20) / 3 / 68, 4)  # a question's trained score is the mean over the seeds
    assert "stop" not in by["policy-grpo"] and by["policy-grpo/seed=0"]["stop"] == {"answer": 68}
    assert by[R.SINGLE_SHOT]["deltas"] == [] and [d["reference"] for d in by[R.UNTRAINED]["deltas"]] == [R.SINGLE_SHOT]
    assert [d["reference"] for d in by["policy-grpo/seed=1"]["deltas"]] == [R.UNTRAINED, R.SINGLE_SHOT]
    # the paired delta is the bootstrap of the per-question differences, on the one index matrix every row shares
    idx = resample_indices(68, n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED)
    scores = {name: R.rescore(arm["rows"], {q.id: q for q in world["gold"]}) for name, arm in world["rollouts"]["arms"].items()}
    mean = R.seed_mean([scores[f"policy-grpo/seed={s}"] for s in range(3)], ids)
    assert {round(v["em"], 6) for v in mean.values()} == {0.0, round(1 / 3, 6), round(2 / 3, 6), 1.0}
    delta, lo, hi = paired_delta_ci([mean[q]["em"] for q in ids], [scores[R.UNTRAINED][q]["em"] for q in ids], idx)
    block = by["policy-grpo"]["deltas"][0]
    assert block["em"] == {"delta": round(delta, 4), "lo": round(lo, 4), "hi": round(hi, 4)} and block["n_questions_changed"]["em"]["down"] == 5  # the five questions seed 2 lost
    assert by["policy-grpo"]["by_category"]["simple_factual"]["n"] == 40 and by["policy-grpo"]["by_gold_form"]["composite"]["em"] == 0.0
    assert by[R.SINGLE_SHOT]["means"]["n_search"] == 1.0 and by[R.UNTRAINED]["means"]["uncited_or_empty"] <= 1.0


def test_the_verdict_is_read_from_the_seed_mean_row_and_names_the_primary_comparison(world):
    out = R.compute(world["rollouts"], world["training"], world["gold"])
    primary = [v for v in out["verdicts"] if v["primary"]]
    assert len(primary) == 1 and primary[0]["config"] == "policy-grpo" and primary[0]["reference"] == R.UNTRAINED and primary[0]["metric"] == "em"
    assert primary[0]["n_seeds"] == 3 and primary[0]["interval"] == "above zero" and primary[0]["outcome"] == "win" and primary[0]["lo"] > 0
    assert {(v["reference"], v["metric"]) for v in out["verdicts"]} == {(r, m) for r in (R.UNTRAINED, R.SINGLE_SHOT) for m in ("em", "reward", "em_bounded")}
    # one trained seed: the verdict is that seed's, and there is no mean row
    one = {"arms": {k: v for k, v in world["rollouts"]["arms"].items() if not k.endswith(("seed=1", "seed=2"))}}
    single = R.compute(one, world["training"], world["gold"])
    assert [r["config"] for r in single["results"]] == [R.SINGLE_SHOT, R.UNTRAINED, "policy-grpo/seed=0"]
    assert [v["config"] for v in single["verdicts"] if v["primary"]] == ["policy-grpo/seed=0"]
    # a tie and a loss are words too
    assert R.verdict({"lo": -0.1, "hi": 0.2}) == "covers zero" and R.verdict({"lo": -0.3, "hi": -0.1}) == "below zero"


def test_checkpoint_selection_and_the_collapse_rule_are_read_from_the_training_file(world):
    runs = {r["name"]: r for r in R.compute(world["rollouts"], world["training"], world["gold"])["training_runs"]}
    assert runs["policy-grpo/seed=0"]["selected_step"] == 100  # step 0 scored highest and is not eligible
    assert runs["policy-grpo/seed=1"]["selected_step"] == 50  # a tie goes to the earliest step
    assert runs["policy-grpo/seed=0"]["tune_reward_untrained"] == 0.9 and runs["policy-grpo/seed=0"]["tune_reward_selected"] == 0.7
    assert runs["policy-grpo/seed=0"]["collapse"]["collapsed"] is False and runs["policy-grpo/seed=1"]["collapse"]["collapsed"] is False  # -0.05
    assert runs["policy-grpo/seed=2"]["collapse"]["collapsed"] is True and "more than 0.1 below" in runs["policy-grpo/seed=2"]["collapse"]["reasons"][0]
    assert runs["policy-grpo/seed=0"]["seconds"] == {"rollouts": 80.0, "updates": 40.0, "tune_scoring": 9.0}
    assert R.selected_step([{"step": 0, "reward": 1.0}]) is None
    flat = [{"loss": 0.1, "reward": 0.5, "em": 0.1, "answered": 0.4}] * 40
    assert R.collapse(flat)["reasons"] == ["fewer than 0.5 of the last 20 steps' episodes hold an answer"]
    assert R.collapse([{"loss": float("nan"), "reward": 0.5, "em": 0.1, "answered": 1.0}] * 5)["reasons"] == ["the loss is not finite"]


def test_the_leakage_report_is_rebuilt_from_the_corpus_seeds_and_names_the_untrained_templates(world):
    leakage = R.compute(world["rollouts"], world["training"], world["gold"])["leakage"]
    assert leakage["seeds"] == [1, 2] and leakage["n_kept_with_a_fixture_question_text"] == 0 and leakage["variant_run_ids_also_in_fixture_v1"] == []
    held = {q.id for q in world["held"]}
    listed = leakage["held_out_questions_whose_template_is_not_trained"]
    assert set(listed) <= held and "q110" in listed and "q134" in listed and "q002" not in listed
    assert leakage["n_held_out_questions_whose_template_is_not_trained"] == len(listed)


def test_check_recomputes_the_file_from_its_sidecars_and_reports_a_changed_episode(tmp_path, world):
    (tmp_path / "bench" / "results").mkdir(parents=True)
    shutil.copytree(ROOT / "bench" / "gold", tmp_path / "bench" / "gold")
    rollouts, training = json.loads(json.dumps(world["rollouts"])), json.loads(json.dumps(world["training"]))
    for name, arm in rollouts["arms"].items():
        if name in training["runs"]:
            arm["adapter"]["selected"] = {"step": R.selected_step(training["runs"][name]["tune"])}
    blob = {"gold_sha256": R.sha256_file(tmp_path / "bench" / "gold" / "questions.jsonl"), "sidecars": {"rollouts": R.ROLLOUTS, "training": R.TRAINING},
            **json.loads(json.dumps(R.compute(rollouts, training, world["gold"])))}
    path = tmp_path / R.RESULTS
    path.write_text(json.dumps(blob))
    (tmp_path / R.ROLLOUTS).write_text(json.dumps(rollouts))
    (tmp_path / R.TRAINING).write_text(json.dumps(training))
    assert R.check(path, root=tmp_path) == []
    # an answer edited in the sidecar: its stored score no longer matches, and the rows move
    qid = world["held"][0].id
    edited = json.loads(json.dumps(rollouts))
    edited["arms"][R.UNTRAINED]["rows"][qid]["episode"]["answer"] = "something else"
    (tmp_path / R.ROLLOUTS).write_text(json.dumps(edited))
    problems = R.check(path, root=tmp_path)
    assert "results: differs from the recomputation" in problems and any(p.startswith("stored episode scores differ") and qid in p for p in problems)
    (tmp_path / R.ROLLOUTS).write_text(json.dumps(rollouts))
    # a number typed into the results file
    typed = json.loads(json.dumps(blob))
    typed["results"][2]["ci"]["em"]["mean"] += 0.01
    path.write_text(json.dumps(typed))
    assert R.check(path, root=tmp_path) == ["results: differs from the recomputation"]
    # the adapter that was scored must be the checkpoint the tune half selects
    path.write_text(json.dumps(blob))
    wrong = json.loads(json.dumps(rollouts))
    wrong["arms"]["policy-grpo/seed=1"]["adapter"]["selected"] = {"step": 100}
    (tmp_path / R.ROLLOUTS).write_text(json.dumps(wrong))
    assert R.check(path, root=tmp_path) == ["policy-grpo/seed=1: the scored adapter is step 100, the tune half selects step 50"]


def test_gold_forms_and_the_declared_texts():
    assert R.gold_form("1e-04") == R.gold_form("configs/run_047.yaml") == R.gold_form("yes") == "single_value"
    assert R.gold_form("46, 47, 48") == R.gold_form("dinov2; 3e-4") == R.gold_form("file src/fusion.py") == R.gold_form("baseline val RMSE") == "composite"
    assert "jury" in R.REWARD["jury_term"] and "not in this reward" in R.REWARD["jury_term"]
    assert any("Synthetic corpora" in line for line in R.DOES_NOT_SHOW) and any("Templated questions" in line for line in R.DOES_NOT_SHOW)
    assert any("One small model" in line for line in R.DOES_NOT_SHOW)
    assert "exact match" in R.BAR["primary"] and "mean over the seeds" in R.BAR["seeds"] and "step 0 is not eligible" in R.BAR["checkpoint"]


def _blob(world) -> dict:
    """A results blob as assemble() writes it, with the header fields the README block reads."""
    computed = json.loads(json.dumps(R.compute(world["rollouts"], world["training"], world["gold"])))
    return {"git_sha": "abc1234", "working_tree_dirty": False, "model": {"name": "Qwen/Qwen2.5-1.5B-Instruct", "revision": "989aa7980e4c0000"},
            "licence": {"model": "apache-2.0"}, "gpu": "NVIDIA A40", "precision": "bf16", "device": {"torch": "2.5.1+cu121"},
            "libraries": {"transformers": "5.17.0", "peft": "0.22.0"}, "information_chunks_per_search": 4, "max_new_tokens_per_piece": 48,
            "environment": {"index": "the in-memory hybrid index", "max_searches": {R.SINGLE_SHOT: 1, R.UNTRAINED: 3}},
            "episodes_generated_at": {R.UNTRAINED: {"git_sha": "abc1234", "working_tree_dirty": False}, R.SINGLE_SHOT: {"git_sha": "abc1234", "working_tree_dirty": False}},
            "bar": R.BAR, "reward": R.REWARD, "algorithm": {"label": "a policy gradient with GRPO's group-normalised advantage", "not": "nothing is clipped"},
            "does_not_show": R.DOES_NOT_SHOW, "sidecars": {"rollouts": R.ROLLOUTS, "training": R.TRAINING}, **computed}


def test_the_readme_block_states_the_verdict_the_guard_columns_and_what_is_not_shown(world):
    from app.eval.report import SEARCH_POLICY_FILE, search_policy_table

    assert SEARCH_POLICY_FILE == R.RESULTS
    missing = search_policy_table(None)
    assert missing.startswith(f"_{R.RESULTS} is missing") and "make search-policy-eval" in missing
    blob = _blob(world)
    text = search_policy_table(blob)
    rows = [line for line in text.splitlines() if line.startswith("| `")]
    assert [line.split("`")[1] for line in rows] == [r["config"] for r in blob["results"]]  # every arm has its row, the losers too
    primary = next(v for v in blob["verdicts"] if v["primary"])
    assert f"{primary['delta']:+.4f} [{primary['lo']:+.4f}, {primary['hi']:+.4f}], **a win**" in text and "the mean of 3 training seeds" in text
    untrained = next(r for r in blob["results"] if r["config"] == R.UNTRAINED)
    assert f"{untrained['ci']['em']['mean']:.4f} [" in rows[1] and f"{untrained['ci']['em_bounded']['mean']:.4f} [" in rows[1]
    assert "bounded match (guard)" in text and "answer length (chars)" in text and "single values" in text
    assert "collapsed by the declared rule (the mean reward of the last 20 steps is more than 0.1 below the first 20)" in text
    assert "not collapsed by the declared rule" in text and "at the selected step 50" in text
    assert "Synthetic corpora" in text and "Templated questions" in text and "One small model" in text and "not in this reward" in text
    assert "make search-policy-rescore" in text and f"[{R.ROLLOUTS}]({R.ROLLOUTS})" in text
    # a tie is printed as a tie
    tie = json.loads(json.dumps(blob))
    for v in tie["verdicts"]:
        if v["primary"]:
            v.update({"delta": 0.0147, "lo": -0.0441, "hi": 0.0735, "interval": "covers zero", "outcome": "tie"})
    assert "+0.0147 [-0.0441, +0.0735], **a tie** (the interval is covers zero)" in search_policy_table(tie)


def test_training_corpora_are_rebuilt_from_their_seeds_and_a_run_on_other_corpora_is_named(world):
    out = R.compute(world["rollouts"], world["training"], world["gold"])["training_corpora"]
    assert out["seeds"] == [1, 2] and [c["seed"] for c in out["rebuilt_from_seeds"]] == [1, 2] and out["runs_whose_recorded_corpora_differ"] == []
    assert all(len(c["corpus_sha256"]) == 64 and c["n_gold"] == 136 and c["n_files"] == 61 for c in out["rebuilt_from_seeds"])
    training = json.loads(json.dumps(world["training"]))
    rebuilt = out["rebuilt_from_seeds"]
    training["runs"]["policy-grpo/seed=0"]["config"]["corpora"] = [{"seed": 1, "corpus_sha256": rebuilt[0]["corpus_sha256"], "gold_sha256": rebuilt[0]["gold_sha256"]},
                                                                   {"seed": 2, "corpus_sha256": "0" * 64, "gold_sha256": rebuilt[1]["gold_sha256"]}]
    differ = R.compute(world["rollouts"], training, world["gold"])["training_corpora"]["runs_whose_recorded_corpora_differ"]
    assert differ == ["policy-grpo/seed=0/corpus 2"]


def test_assemble_takes_arms_from_two_nodes_with_the_same_gpu_and_refuses_another_gpu(tmp_path, world):
    """The base-model arms are scored before training and the trained ones after, often on another node: the node's
    CPU and kernel may differ, the GPU, the software and everything about the questions may not."""
    device = {"requested": "cuda", "python": "3.11.9", "platform": "Linux-a", "cpu_count": 64, "cpu_model": "AMD EPYC 7543",
              "torch": "2.5.1+cu121", "cuda_available": True, "device": "cuda", "device_name": "NVIDIA A40", "threads": 4, "sentence_transformers": "6.1.0"}
    common = {"model": {"name": "m", "revision": "r"}, "embedder": {"name": "e"}, "libraries": {"peft": "0.22.0"}, "gpu": "NVIDIA A40",
              "precision": "bf16", "corpus_sha256": "c", "gold_sha256": "g", "n_files": 61, "n_chunks": 100, "decoding": "greedy",
              "max_new_tokens_per_piece": 48, "information_chunks_per_search": 4, "split": "held_out", "question_ids": [q.id for q in world["held"]],
              "command": "python -m app.rl.evaluate", "timestamp": "t", "git_sha": "abc1234", "git_head": "abc1234", "working_tree_dirty": False}
    arms_dir = tmp_path / "arms"
    arms_dir.mkdir()
    for name, node in ((R.UNTRAINED, {}), (R.SINGLE_SHOT, {"cpu_count": 128, "cpu_model": "AMD EPYC 7713", "platform": "Linux-b"})):
        arm = world["rollouts"]["arms"][name]
        blob = {**common, "arm": name, "kind": arm["kind"], "adapter": None, "max_searches": arm["max_searches"], "prompt_sha256": "x",
                "seconds": 1.0, "device": {**device, **node}, "rows": arm["rows"]}
        (arms_dir / f"{name}.json").write_text(json.dumps(blob))
    results, rollouts, _ = R.assemble(arms_dir, {}, None)
    assert results["device"] == {k: device[k] for k in R.SHARED_DEVICE_KEYS}
    assert rollouts["arms"][R.SINGLE_SHOT]["device"]["cpu_count"] == 128 and rollouts["arms"][R.UNTRAINED]["device"]["cpu_count"] == 64
    other = json.loads((arms_dir / f"{R.SINGLE_SHOT}.json").read_text())
    other["device"]["device_name"] = "Quadro RTX 8000"
    (arms_dir / f"{R.SINGLE_SHOT}.json").write_text(json.dumps(other))
    with pytest.raises(SystemExit, match="device differs from the other arms"):
        R.assemble(arms_dir, {}, None)
