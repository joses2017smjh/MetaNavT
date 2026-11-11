"""B17: per question, the gold files hybrid drops (or gains) against dense alone, with the ranks that explain it."""

import json
import sys
from pathlib import Path

from app.eval import hybrid_drops
from app.eval.gold import GoldQuestion

ROOT = Path(__file__).resolve().parents[2]
PATHS = {"b1": "x.md", "b2": "y.md", "b3": "z.md", "d1": "x.md", "d2": "w.md", "g1": "gold.md"}


def _q(*relevant):
    return GoldQuestion(id="q", question="?", category="semantic", answer="", relevant_paths=list(relevant))


def test_a_gold_file_only_dense_ranks_low_falls_below_the_fused_cut():
    # k = 60, retrieve_k = 3. Fused, stable by score: b1 1/61, d1 1/61, b2 1/62, d2 1/62, b3 1/63, g1 1/63; kept b1, d1, b2.
    entry = hybrid_drops.diff_question(_q("gold.md", "z.md"), ["b1", "b2", "b3"], ["d1", "d2", "g1"], PATHS, retrieve_k=3)
    assert entry["recall@50"] == {"dense_only": 0.5, "hybrid": 0.0}
    assert entry["cut"] == {"rank": 3, "rrf": round(1 / 62, 8), "n_fused": 6}
    assert entry["gained"] == []
    (drop,) = entry["dropped"]  # z.md is in neither list, so it is not a drop
    assert drop == {"path": "gold.md", "dense": {"file_rank": 3, "chunk_rank": 3}, "bm25": {"file_rank": None, "chunk_rank": None},
                    "fused": {"chunk_rank": 6, "rrf": round(1 / 63, 8), "kept": False, "tied_with_cut": False}}


def test_a_gold_file_only_bm25_ranks_is_a_gain():
    entry = hybrid_drops.diff_question(_q("y.md"), ["b1", "b2", "b3"], ["d1", "d2", "g1"], PATHS, retrieve_k=3)
    assert entry["recall@50"] == {"dense_only": 0.0, "hybrid": 1.0} and entry["dropped"] == []
    assert entry["gained"] == [{"path": "y.md", "dense": {"file_rank": None, "chunk_rank": None}, "bm25": {"file_rank": 2, "chunk_rank": 2},
                                "fused": {"chunk_rank": 3, "rrf": round(1 / 62, 8), "kept": True, "tied_with_cut": True}}]


def test_a_drop_decided_by_a_tie_at_the_cut_says_so():
    # b1 1/61, d1 1/61, b2 1/62, g1 1/62: the cut (rank 3) is b2, and g1 has its score but was seen later.
    entry = hybrid_drops.diff_question(_q("gold.md"), ["b1", "b2"], ["d1", "g1"], PATHS, retrieve_k=3)
    (drop,) = entry["dropped"]
    assert drop["fused"] == {"chunk_rank": 4, "rrf": round(1 / 62, 8), "kept": False, "tied_with_cut": True}


def test_an_unaffected_question_is_not_listed():
    assert hybrid_drops.diff_question(_q("x.md"), ["b1", "b2", "b3"], ["d1", "d2", "g1"], PATHS, retrieve_k=3) is None


def test_hash_run_reproduces_main_json_and_accounts_for_the_whole_delta(tmp_path, monkeypatch):
    """A CPU dry run: with the hash embedder the lists are main.json's dense_only and hybrid rows exactly, and the listed
    questions add up to the row's Recall@50 delta."""
    out = tmp_path / "drops.json"
    monkeypatch.setattr(sys, "argv", [str(ROOT / "app" / "eval" / "hybrid_drops.py"), "--embedder", "hash", "--out", str(out)])
    assert hybrid_drops.main() == 0
    text = out.read_text()
    blob = json.loads(text)
    assert blob["reproduces"]["all"] is True and set(blob["reproduces"]["rows"]) == {"bm25_only", "dense_only", "hybrid"}
    s = blob["summary"]
    assert s["every_question_that_moves_is_listed"] is True
    listed = sum(q["recall@50"]["hybrid"] - q["recall@50"]["dense_only"] for q in blob["questions"]) / s["n_questions"]
    assert abs(listed - s["recall@50"]["delta"]) < 1e-3  # per-question values and the delta are rounded to 4 decimals
    assert s["n_questions_affected"] == len(blob["questions"])
    assert s["n_dropped"] == sum(len(q["dropped"]) for q in blob["questions"])
    assert s["n_gained"] == sum(len(q["gained"]) for q in blob["questions"])
    for q in blob["questions"]:
        assert q["dropped"] or q["gained"]
        for d in q["dropped"]:
            assert d["dense"]["chunk_rank"] is not None and d["fused"]["chunk_rank"] > blob["retrieve_k"] and not d["fused"]["kept"]
        for g in q["gained"]:
            assert g["dense"] == {"file_rank": None, "chunk_rank": None} and g["fused"]["kept"]
    assert str(ROOT) not in text and str(tmp_path) not in text
    assert "python -m app.eval.hybrid_drops --embedder hash --out <path>/drops.json" in blob["command"]
