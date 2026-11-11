"""Gold targets are file paths: a question whose only targets are chunk ids is refused, not scored as zero.

Every scorer (the harness, the parity run, the rerank arms) ranks file paths. A question naming only
relevant_chunk_ids used to be compared with those paths and scored zero even when its chunk came back first.
"""

import json
from pathlib import Path

import pytest

from app.eval.gold import GoldQuestion, load_gold
from app.eval.harness import BenchConfig, run_config
from app.eval.index_loader import load_chunks

BM25 = BenchConfig(name="bm25_only", mode="bm25", enable_rerank=False, enable_router=False, staleness_tier1=False,
                   e2e=False, log_triples=False)


def _corpus(root: Path) -> list:
    (root / "configs").mkdir(parents=True)
    (root / "configs" / "run_a.yaml").write_text("encoder: dinov2\nlearning_rate: 0.0003\n")
    (root / "notes.md").write_text("# Notes\n\nThe fusion module concatenates per-view features.\n")
    return load_chunks(root)


def _question(**targets) -> GoldQuestion:
    return GoldQuestion(id="q2", question="which encoder does run a use", category="simple_factual", answer="dinov2", **targets)


def test_load_gold_refuses_a_question_whose_only_targets_are_chunk_ids(tmp_path):
    chunks = _corpus(tmp_path / "files")
    run_a = next(c for c in chunks if c.path == "configs/run_a.yaml")
    gold = tmp_path / "questions.jsonl"
    rows = [
        _question(relevant_paths=["notes.md"]).to_dict() | {"id": "q1"},
        _question(relevant_chunk_ids=[run_a.chunk_id]).to_dict(),
    ]
    gold.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(ValueError, match="questions.jsonl:2 q2 names only relevant_chunk_ids"):
        load_gold(gold)


def test_a_chunk_id_question_built_in_code_is_refused_by_the_scorer(tmp_path):
    """The chunk is ranked first, so scoring it on paths would have reported nDCG@10 0 for a perfect list."""
    chunks = _corpus(tmp_path)
    run_a = next(c for c in chunks if c.path == "configs/run_a.yaml")
    by_path = run_config(BM25, [_question(relevant_paths=[run_a.path])], tmp_path, n_files=2)
    assert by_path["retrieval"]["ndcg@10"] == 1.0  # the right file, and its only chunk, is ranked first

    question = _question(relevant_chunk_ids=[run_a.chunk_id])
    with pytest.raises(ValueError, match="names only relevant_chunk_ids"):
        question.relevant_ids()
    with pytest.raises(ValueError, match="names only relevant_chunk_ids"):
        run_config(BM25, [question], tmp_path, n_files=2)


def test_paths_are_the_targets_when_both_are_given(tmp_path):
    question = _question(relevant_paths=["configs/run_a.yaml"], relevant_chunk_ids=["configs/run_a.yaml::000000000000"])
    gold = tmp_path / "questions.jsonl"
    gold.write_text(json.dumps(question.to_dict()) + "\n")
    (loaded,) = load_gold(gold)
    assert loaded.relevant_ids() == ["configs/run_a.yaml"]
    assert _question().relevant_ids() == []  # no target at all is not a chunk-id question
