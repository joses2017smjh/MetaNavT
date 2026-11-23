"""M43 (app/eval/distill_train.py): the parts of a student's training that need no torch: the loss as declared, the
schedule, how lists are mixed across corpora, which epoch is kept, and what the committed manifest refuses.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from app.eval import distill_train as tr
from app.eval.small_rerankers import RECIPE, STUDENT_MANIFEST, WEIGHTS_FILE

ROOT = Path(__file__).resolve().parents[2]


def test_margin_mse_is_the_mean_squared_margin_error_over_every_pair():
    s, t = [3.0, 1.0, 0.0], [2.0, 2.0, -1.0]
    pairs = [((s[i] - s[j]) - (t[i] - t[j])) ** 2 for i in range(3) for j in range(i + 1, 3)]  # (2-0)^2, (3-3)^2, (1-3)^2
    assert tr.margin_mse(s, t) == pytest.approx(sum(pairs) / 3) == pytest.approx(8 / 3)
    assert tr.margin_mse([x + 7.5 for x in s], t) == pytest.approx(tr.margin_mse(s, t))  # only differences are compared
    assert tr.margin_mse(t, t) == 0.0 and tr.margin_mse([1.0], [5.0]) == 0.0
    d = np.array(s) - np.array(t)
    assert tr.margin_mse(s, t) == pytest.approx(2 * 3 / 2 * d.var())  # = 2n/(n-1) times the variance of s - t
    assert "MarginMSE over every pair" in RECIPE["loss"]


def test_the_learning_rate_warms_up_then_decays_to_zero():
    total = 100
    factors = [tr.lr_factor(step, total, 0.1) for step in range(total)]
    assert factors[0] == pytest.approx(0.1) and factors[9] == pytest.approx(1.0) and factors[10] == pytest.approx(1.0)
    assert all(a > b for a, b in zip(factors[10:], factors[11:]))
    assert factors[-1] == pytest.approx(1 / 90) and tr.lr_factor(total, total, 0.1) == 0.0


def test_corpora_are_mixed_by_the_square_root_of_their_size_without_replacement_inside_a_corpus():
    sizes = {"fixture": 100, "scifact": 400, "fiqa": 900}
    sampler = tr.MixedSampler(sizes, seed=0)
    draws = [sampler.draw() for _ in range(6000)]
    share = {c: sum(1 for d in draws if d[0] == c) / len(draws) for c in sizes}
    assert share["fixture"] == pytest.approx(10 / 60, abs=0.02) and share["scifact"] == pytest.approx(20 / 60, abs=0.02)
    assert share["fiqa"] == pytest.approx(30 / 60, abs=0.02)
    fixture = [i for c, i in draws if c == "fixture"]
    assert sorted(fixture[:100]) == list(range(100))  # one full pass before any list repeats
    assert sorted(fixture[100:200]) == list(range(100)) and fixture[:100] != fixture[100:200]  # reshuffled, then again
    again = tr.MixedSampler(sizes, seed=0)
    assert [again.draw() for _ in range(50)] == draws[:50]
    assert tr.MixedSampler({"fixture": 5, "empty": 0}, seed=0).draw()[0] == "fixture"


def test_lists_join_queries_scores_and_texts_and_refuse_a_gap():
    queries = [{"id": "fixture:a:q1", "corpus": "fixture", "doc_id": "a", "query": "one", "split": "train"},
               {"id": "scifact:9:s1", "corpus": "scifact", "doc_id": "9", "query": "two", "split": "validation"}]
    scores = [{"id": "scifact:9:s1", "corpus": "scifact", "candidates": ["9", "4"], "scores": [3.5, -2.0]},
              {"id": "fixture:a:q1", "corpus": "fixture", "candidates": ["b", "a"], "scores": [0.5, 4.0]}]
    texts = {"fixture": {"a": "text a", "b": "text b"}, "scifact": {"9": "nine", "4": "four"}}
    lists = tr.build_lists(queries, scores, texts)
    assert lists["train"] == {"fixture": [{"id": "fixture:a:q1", "query": "one", "candidates": ["b", "a"], "texts": ["text b", "text a"], "scores": [0.5, 4.0]}]}
    assert lists["validation"]["scifact"][0]["texts"] == ["nine", "four"]
    with pytest.raises(KeyError, match="no teacher scores"):
        tr.build_lists(queries, scores[:1], texts)
    scores[0]["scores"] = [3.5]
    with pytest.raises(ValueError, match="2 candidates, 1 scores"):
        tr.build_lists(queries, scores, texts)


def test_the_epoch_with_the_lowest_validation_loss_is_kept_and_a_tie_goes_to_the_earlier_one():
    assert tr.choose_epoch([0.9, 0.5, 0.7, 0.6]) == 2
    assert tr.choose_epoch([0.5, 0.4, 0.4, 0.4]) == 2
    assert tr.choose_epoch([0.3]) == 1
    assert tr.macro_validation({"fixture": 1.0, "scifact": 2.0, "fiqa": 6.0}) == pytest.approx(3.0)  # each corpus counts once
    assert "averaged per corpus and then over the three corpora" in RECIPE["selection"]


def _trained(tmp_path, tag, **changes):
    directory = tmp_path / tag
    directory.mkdir(parents=True)
    (directory / WEIGHTS_FILE).write_bytes(tag.encode())
    record = {"student": tag, "attempt": 1, "smoke_run_max_steps": None, "working_tree_dirty": False, "chosen_epoch": 2,
              "epochs": [{"epoch": 1, "validation_macro": 0.5}, {"epoch": 2, "validation_macro": 0.4}],
              "files": {WEIGHTS_FILE: {"sha256": hashlib.sha256(tag.encode()).hexdigest(), "bytes": len(tag)}}, "reference_logits": [], **changes}
    (directory / tr.TRAINING_FILE).write_text(json.dumps(record))
    return directory


def test_the_manifest_refuses_a_smoke_run_a_dirty_tree_and_changed_weights(tmp_path):
    for tag in ("student-minilm-l6", "student-minilm-l12"):
        _trained(tmp_path / "ok", tag)
    manifest = tr.build_manifest(tmp_path / "ok")
    entry = manifest["students"]["student-minilm-l6"]
    assert entry["directory"] == "student-minilm-l6" and entry["weights_bytes"] == len("student-minilm-l6")
    assert entry["weights_sha256"] == hashlib.sha256(b"student-minilm-l6").hexdigest() and entry["chosen_epoch"] == 2
    assert "not in the repository" in manifest["about"]
    with pytest.raises(SystemExit, match="no training.json"):
        tr.build_manifest(tmp_path / "missing")
    _trained(tmp_path / "smoke", "student-minilm-l6", smoke_run_max_steps=3)
    with pytest.raises(SystemExit, match="smoke run"):
        tr.build_manifest(tmp_path / "smoke", ["student-minilm-l6"])
    _trained(tmp_path / "dirty", "student-minilm-l6", working_tree_dirty=True)
    with pytest.raises(SystemExit, match="uncommitted changes"):
        tr.build_manifest(tmp_path / "dirty", ["student-minilm-l6"])
    directory = _trained(tmp_path / "moved", "student-minilm-l6")
    (directory / WEIGHTS_FILE).write_bytes(b"other weights")
    with pytest.raises(SystemExit, match="its training recorded"):
        tr.build_manifest(tmp_path / "moved", ["student-minilm-l6"])


def test_the_manifest_command_writes_the_committed_file(tmp_path):
    for tag in ("student-minilm-l6", "student-minilm-l12"):
        _trained(tmp_path / "ckpt", tag)
    out = tmp_path / "students.json"
    assert tr.main(["manifest", "--checkpoint-dir", str(tmp_path / "ckpt"), "--out", str(out)]) == 0
    assert set(json.loads(out.read_text())["students"]) == {"student-minilm-l6", "student-minilm-l12"}
    assert STUDENT_MANIFEST == "bench/distill/students.json"
    with pytest.raises(SystemExit, match="needs --student"):
        tr.main(["train", "--checkpoint-dir", str(tmp_path / "ckpt")])


def test_the_training_module_sets_no_hyper_parameter_of_its_own():
    source = (ROOT / "app" / "eval" / "distill_train.py").read_text()
    for key in ("learning_rate", "weight_decay", "epochs", "lists_per_step", "warmup_fraction", "max_grad_norm", "max_length", "seed"):
        assert f'RECIPE["{key}"]' in source, key  # read from the declaration, where a test pins it
