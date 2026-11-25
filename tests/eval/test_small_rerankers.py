"""M43 (app/eval/small_rerankers.py): what was declared before any small reranker was measured, and how a student's
checkpoint is found and verified.

No torch here. The declarations are pinned value by value: the zero-shot rows are measured before the students are
trained, so a recipe, a bar or a student list that could change quietly afterwards would not be a declaration.
"""

import hashlib
import json
import re
from pathlib import Path

import pytest

from app.eval import small_rerankers as sr

ROOT = Path(__file__).resolve().parents[2]


def test_the_served_shape_is_the_single_hosts():
    compose = (ROOT / "deploy" / "single-host" / "compose.yml").read_text()
    served = {key: re.search(rf'^\s*{key}: "?([^"\s]+)"?', compose, re.M).group(1) for key in ("RERANK_DEPTH", "RERANKER_MAX_LENGTH", "RERANKER_PRECISION")}
    assert sr.SHAPE == {"rerank_top": int(served["RERANK_DEPTH"]), "max_length": int(served["RERANKER_MAX_LENGTH"]), "precision": served["RERANKER_PRECISION"]}
    assert sr.SUFFIX == "/fp32-512/top20"
    assert sr.BGE_ROW == "hybrid+bge-rerank@bge-small/fp32-512/top20"  # the row rerank_depth.json and both BEIR files already have
    assert sr.row_name("bge-rerank") == sr.BGE_ROW and sr.row_name("minilm-l6") == "hybrid+minilm-l6@bge-small/fp32-512/top20"


def test_the_candidates_are_small_permissive_and_pinned():
    assert list(sr.CANDIDATES) == ["minilm-l6", "minilm-l12", "mxbai-xsmall-v1", "mxbai-base-v1", "gte-modernbert-base", "bge-reranker-base"]
    for tag, c in sr.CANDIDATES.items():
        assert c["licence"] in sr.PERMISSIVE_LICENCES, tag
        assert re.fullmatch(r"[0-9a-f]{40}", c["revision"]), tag
        assert 20_000_000 <= c["parameters"] <= 300_000_000, tag
    assert sr.CANDIDATES["minilm-l6"]["model"] == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert sr.reranker_tags() == ["bge-rerank", "mxbai-rerank-large-v2", *sr.CANDIDATES, "student-minilm-l6", "student-minilm-l12"]
    assert sr.model_of("bge-rerank") == "BAAI/bge-reranker-v2-m3" and sr.model_of(sr.TEACHER_TAG) == "mixedbread-ai/mxbai-rerank-large-v2"
    sr.check_revision("minilm-l6", sr.CANDIDATES["minilm-l6"]["revision"])
    with pytest.raises(RuntimeError, match="pinned"):
        sr.check_revision("minilm-l6", "0" * 40)  # main moved: another model
    sr.check_revision("bge-rerank", None)  # the references are pinned by the committed rows they reproduce, not here


def test_the_bar_is_declared_for_every_reranked_row():
    assert "lower end is above zero" in sr.BAR["passes"]
    for needle in ("held-out", "SciFact", "FiQA"):
        assert needle in sr.BAR["passes"]
    assert "bge-reranker-v2-m3 and the teacher" in sr.BAR["applies_to"]  # not only the new models
    assert "split_questions" in sr.BAR["held_out"] and "no choice is made on the other (tune) half" in sr.BAR["held_out"]
    assert "indicative only" in sr.BAR["not_decided_here"]


def test_the_students_and_their_recipe_are_frozen():
    assert sr.STUDENTS == {"student-minilm-l6": "minilm-l6", "student-minilm-l12": "minilm-l12"}
    assert sr.ATTEMPT == 1
    numbers = {k: v for k, v in sr.RECIPE.items() if not isinstance(v, str)}
    assert numbers == {"max_length": 512, "lists_per_step": 8, "epochs": 4, "learning_rate": 2e-5, "weight_decay": 0.01,
                       "warmup_fraction": 0.1, "max_grad_norm": 1.0, "seed": 0}
    assert sr.RECIPE["teacher"] == sr.TEACHER_MODEL
    assert "MarginMSE over every pair" in sr.RECIPE["loss"]
    assert "square root" in sr.RECIPE["corpus_mixing"]
    # a checkpoint is chosen without a label: on held-out synthetic queries, never on a gold or BEIR query
    assert "lowest validation loss" in sr.RECIPE["selection"]
    assert "never a gold question, a tune-half question or a BEIR query" in sr.RECIPE["validation"]


def _checkpoint(tmp_path, tag="student-minilm-l6", content=b"weights"):
    directory = tmp_path / "ckpt" / tag
    directory.mkdir(parents=True)
    (directory / sr.WEIGHTS_FILE).write_bytes(content)
    manifest = {"students": {tag: {"directory": tag, "weights_sha256": hashlib.sha256(content).hexdigest(), "attempt": 1}}}
    return tmp_path / "ckpt", manifest


def test_a_students_checkpoint_is_verified_against_the_manifest(tmp_path):
    base, manifest = _checkpoint(tmp_path)
    directory, entry = sr.student_checkpoint("student-minilm-l6", base, manifest)
    assert directory == base / "student-minilm-l6" and entry["attempt"] == 1
    (directory / sr.WEIGHTS_FILE).write_bytes(b"other weights")
    with pytest.raises(sr.CheckpointError, match="the manifest records"):
        sr.student_checkpoint("student-minilm-l6", base, manifest)
    with pytest.raises(sr.CheckpointError, match="is not in the checkpoint directory"):
        sr.student_checkpoint("student-minilm-l6", tmp_path / "elsewhere", manifest)
    with pytest.raises(sr.CheckpointError, match="is not in"):
        sr.student_checkpoint("student-minilm-l12", base, manifest)
    with pytest.raises(KeyError):
        sr.student_checkpoint("student-of-nothing", base, manifest)


def test_no_checkpoint_directory_is_an_error_not_a_fallback(tmp_path, monkeypatch):
    _, manifest = _checkpoint(tmp_path)
    monkeypatch.delenv("SMALL_RERANKER_CHECKPOINTS", raising=False)
    with pytest.raises(sr.CheckpointError, match="no checkpoint directory"):
        sr.student_checkpoint("student-minilm-l6", None, manifest)


def test_a_row_records_a_students_sha256_and_never_its_path(tmp_path, monkeypatch):
    base, manifest = _checkpoint(tmp_path)
    monkeypatch.setattr(sr, "student_checkpoint", lambda tag, d, m=None: (Path(base) / tag, manifest["students"][tag]))
    source, record = sr.reranker_source("student-minilm-l6", base)
    assert source == str(base / "student-minilm-l6")
    assert record["model"] == "student-minilm-l6" and record["student_of"] == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert record["weights_sha256"] == manifest["students"]["student-minilm-l6"]["weights_sha256"]
    assert str(tmp_path) not in json.dumps(record)
    source, record = sr.reranker_source("mxbai-xsmall-v1")
    assert source == "mixedbread-ai/mxbai-rerank-xsmall-v1" and record["pinned_revision"] == sr.CANDIDATES["mxbai-xsmall-v1"]["revision"]


def test_the_cli_needs_a_step():
    assert sr.main([]) == 2
