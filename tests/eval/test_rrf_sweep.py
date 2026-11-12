"""B12: the RRF sweep rebuilds the hybrid index's fusion from its two legs, picks one setting on the tune half by a fixed
rule, and scores it against k = 60 on the held-out half without ever changing the default."""

import json
import sys
from pathlib import Path

import pytest

from app.eval import rrf_sweep
from app.eval.gold import load_gold
from app.eval.index_loader import build_index
from app.eval.metrics import aggregate_retrieval
from app.eval.report import verdict
from app.eval.rerank_arms import split_questions
from app.retrieval.fuse import RRF_K

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def hash_index():
    return build_index(ROOT / "bench" / "corpus" / "files", embedder_name="hash", retrieve_k=50, enable_router=False,
                       enable_rerank=False, mtimes="manifest")


@pytest.fixture(scope="module")
def gold():
    return load_gold(ROOT / "bench" / "gold" / "questions.jsonl")


@pytest.mark.parametrize("k", [RRF_K, 1, 10, 200])
def test_fusion_rebuilt_from_the_two_legs_is_what_the_index_retrieves(hash_index, gold, k):
    """Every gold question: the sweep's fused files at (k, equal weights) equal InMemoryHybridIndex.retrieve's with rrf_k = k."""
    legs = rrf_sweep.leg_lists(hash_index, [q.question for q in gold], 50)
    rebuilt = rrf_sweep.hybrid_files(hash_index, legs, k=k, dense_weight=1.0, retrieve_k=50)
    saved = hash_index.rrf_k
    try:
        hash_index.rrf_k = k
        served = [rrf_sweep.files_of([h.chunk.chunk_id for h in hash_index.retrieve(q.question).hits], hash_index) for q in gold]
    finally:
        hash_index.rrf_k = saved
    assert rebuilt == served


def test_the_dense_weight_moves_the_fused_order(hash_index, gold):
    legs = rrf_sweep.leg_lists(hash_index, [q.question for q in gold], 50)
    equal = rrf_sweep.hybrid_files(hash_index, legs, k=RRF_K, dense_weight=1.0, retrieve_k=50)
    for w in (0.25, 4.0):
        assert rrf_sweep.hybrid_files(hash_index, legs, k=RRF_K, dense_weight=w, retrieve_k=50) != equal


def test_choose_takes_the_best_tune_mean_and_breaks_ties_toward_the_default():
    choose = rrf_sweep.choose
    assert choose([(60, 1.0, 0.5), (10, 2.0, 0.6), (100, 1.0, 0.55)]) == (10, 2.0)
    assert choose([(60, 1.0, 0.6), (10, 2.0, 0.6)]) == (60, 1.0)  # the default wins a tie it is in
    assert choose([(200, 1.0, 0.6), (60, 2.0, 0.6)]) == (200, 1.0)  # equal weights first, then k
    assert choose([(30, 1.0, 0.6), (100, 1.0, 0.6)]) == (100, 1.0)  # |ln(100/60)| < |ln(30/60)|
    assert choose([(60, 0.5, 0.6), (60, 2.0, 0.6)]) == (60, 0.5)  # same distance both ways: the smaller weight
    assert choose([(60, 1.0, 0.6), (10, 1.0, 0.6 + 1e-12)]) == (60, 1.0)  # means equal to 10 decimals tie


def test_grid_has_the_default():
    assert rrf_sweep.DEFAULT_SETTING == (RRF_K, 1.0) == (60, 1.0)
    assert RRF_K in rrf_sweep.RRF_KS and 1.0 in rrf_sweep.DENSE_WEIGHTS
    assert rrf_sweep.RRF_KS == (1, 10, 30, 60, 100, 200)


def test_hash_run_reproduces_main_json_and_records_the_decision(tmp_path, gold, monkeypatch):
    """A CPU dry run: with the hash embedder the default setting's lists are the committed bm25_only, dense_only and hybrid
    rows of main.json exactly, the split is the students' split, and the default is never changed."""
    out = tmp_path / "rrf_sweep.json"
    monkeypatch.setattr(sys, "argv", [str(ROOT / "app" / "eval" / "rrf_sweep.py"), "--embedder", "hash", "--out", str(out)])
    assert rrf_sweep.main() == 0
    text = out.read_text()
    blob = json.loads(text)
    assert blob["reproduces"]["all"] is True and set(blob["reproduces"]["rows"]) == {"bm25_only", "dense_only", "hybrid"}
    tune_ids, test_ids = split_questions(gold)
    assert blob["split"]["tune_ids"] == [gold[i].id for i in tune_ids] and blob["split"]["held_out_ids"] == [gold[i].id for i in test_ids]
    points = blob["tune"]["points"]
    assert len(points) == len(rrf_sweep.RRF_KS) * len(rrf_sweep.DENSE_WEIGHTS) and blob["tune"]["n"] == len(tune_ids)
    default = next(p for p in points if (p["k"], p["dense_weight"]) == (60, 1.0))
    assert default["ndcg@10_vs_default"] == {"delta": 0.0, "lo": 0.0, "hi": 0.0} and default["n_top10_differs_from_default"] == 0
    best = max(p["ndcg@10"] for p in points)
    decision = blob["decision"]
    chosen = next(p for p in points if (p["k"], p["dense_weight"]) == (decision["chosen"]["k"], decision["chosen"]["weights"]["dense"]))
    assert chosen["ndcg@10"] == best and decision["default_changed"] is False
    held = {r["config"]: r for r in blob["held_out"]["results"]}
    assert all(r["n"] == len(test_ids) for r in held.values())
    if decision["chosen_is_default"]:
        assert decision["held_out_ndcg@10"] is None and list(held) == [blob["tune"]["default"]]
    else:
        block = held[blob["tune"]["chosen"]]["deltas"][0]
        assert block["reference"] == blob["tune"]["default"] and decision["held_out_ndcg@10"] == block["ndcg@10"]
        assert decision["owner_decision_needed"] == (block["ndcg@10"]["lo"] > 0)
    assert str(ROOT) not in text and str(tmp_path) not in text
    assert "python -m app.eval.rrf_sweep --embedder hash --out <path>/rrf_sweep.json" in blob["command"]


def test_the_dense_weight_multiplies_the_dense_list_only():
    """The one tuned weight is the dense list's: [BM25, dense] fused with weights (1, w)."""
    assert rrf_sweep.fused_ranking(["b"], ["d"], k=60, dense_weight=2.0) == [("d", 2.0 / 61), ("b", 1.0 / 61)]
    assert rrf_sweep.fused_ranking(["b"], ["d"], k=60, dense_weight=0.5) == [("b", 1.0 / 61), ("d", 0.5 / 61)]
    assert rrf_sweep.fused_ranking(["x", "b"], ["d", "x"], k=10, dense_weight=4.0) == [("x", 1.0 / 11 + 4.0 / 12), ("d", 4.0 / 11),
                                                                                        ("b", 1.0 / 12)]


def test_reproduces_is_false_when_one_block_differs_from_its_row(tmp_path):
    """`reproduces` compares each leg's block with the committed row it stands for, so one wrong list makes it false."""
    rows = {r["config"]: r["retrieval"] for r in json.loads((ROOT / "bench" / "results" / "main.json").read_text())["results"]}
    same = {"bm25": rows["bm25_only"], "dense": rows["dense_only"], "hybrid": rows["hybrid"]}
    assert rrf_sweep.reproduces(ROOT, "hash", same)["all"] is True
    wrong = rrf_sweep.reproduces(ROOT, "hash", {**same, "dense": rows["hybrid"]})
    assert wrong["rows"] == {"bm25_only": True, "dense_only": False, "hybrid": True} and wrong["all"] is False
    assert rrf_sweep.reproduces(ROOT, "st:an/embedder-without-rows", same) is None
    assert rrf_sweep.reproduces(tmp_path, "hash", same) == {"file": "bench/results/main.json", "found": False, "all": False}


@pytest.fixture(scope="module")
def hash_sweep():
    return rrf_sweep.run(ROOT, embedder="hash")


def test_the_grid_is_scored_on_the_tune_half_and_the_held_out_rows_on_the_other_half(hash_index, gold, hash_sweep):
    """The default setting's tune-half point and held-out row equal an independent scoring of index.retrieve() over the
    train and the test ids of split_questions; the two halves score differently, so a swapped half fails."""
    tune_ids, test_ids = split_questions(gold)
    files = [rrf_sweep.files_of([h.chunk.chunk_id for h in hash_index.retrieve(q.question).hits], hash_index) for q in gold]

    def ndcg(ids):
        return aggregate_retrieval([{"id": gold[i].id, "retrieved": files[i], "relevant": gold[i].relevant_ids(), "category": gold[i].category}
                                    for i in ids], n_files=hash_sweep["n_files"]).ndcg_10

    assert hash_index.rrf_k == RRF_K and abs(ndcg(tune_ids) - ndcg(test_ids)) > 0.01
    default = next(p for p in hash_sweep["tune"]["points"] if (p["k"], p["dense_weight"]) == (60, 1.0))
    assert default["ndcg@10"] == round(ndcg(tune_ids), 4)
    held = {r["config"]: r for r in hash_sweep["held_out"]["results"]}
    assert abs(held[hash_sweep["tune"]["default"]]["ci"]["ndcg@10"]["mean"] - ndcg(test_ids)) < 1e-4
    if not hash_sweep["decision"]["chosen_is_default"]:
        chosen = hash_sweep["decision"]["chosen"]
        legs = rrf_sweep.leg_lists(hash_index, [q.question for q in gold], 50)
        lists = rrf_sweep.hybrid_files(hash_index, legs, k=chosen["k"], dense_weight=chosen["weights"]["dense"], retrieve_k=50)
        held_out = rrf_sweep.scored(gold, lists, test_ids, hash_sweep["n_files"]).ndcg_10
        assert abs(held[hash_sweep["tune"]["chosen"]]["ci"]["ndcg@10"]["mean"] - held_out) < 1e-4


@pytest.mark.parametrize("block, needed", [
    ({"delta": 0.01, "lo": -0.02, "hi": 0.04}, False),  # covers zero (the kind of interval the recorded run has)
    ({"delta": 0.05, "lo": 0.01, "hi": 0.09}, True),  # above zero: in favour of the chosen setting
    ({"delta": -0.05, "lo": -0.09, "hi": -0.01}, False),  # below zero: in favour of the default
])
def test_only_a_held_out_interval_above_zero_asks_the_owner(monkeypatch, block, needed):
    """The owner decision is read off the lower bound of the held-out nDCG@10 interval (chosen minus default), and the
    command never changes the default either way. The tune-half choice is fixed to a non-default setting here."""
    real = rrf_sweep.score_rows

    def with_interval(orders, gold, ids, latency, n_files, deltas):
        rows, boot = real(orders, gold, ids, latency, n_files, deltas)
        for row in rows:
            for d in row.get("deltas", []):
                d["ndcg@10"] = dict(block)
        return rows, boot

    monkeypatch.setattr(rrf_sweep, "choose", lambda points: (200, 0.5))
    monkeypatch.setattr(rrf_sweep, "score_rows", with_interval)
    decision = rrf_sweep.run(ROOT, embedder="hash")["decision"]
    assert decision["chosen"] == {"k": 200, "weights": {"bm25": 1.0, "dense": 0.5}} and decision["held_out_ndcg@10"] == block
    assert decision["excludes_zero_in_favour_of_chosen"] is needed and decision["owner_decision_needed"] is needed
    assert decision["verdict"] == f"held-out nDCG@10, chosen minus default: {verdict(block)}" and decision["default_changed"] is False


def test_the_engineering_reference_states_the_recorded_outcome():
    """The typed outcome sentence of the engineering reference's RRF paragraph agrees with the file it cites (a
    re-recording with another outcome must rewrite it)."""
    d = json.loads((ROOT / "bench" / "results" / "rrf_sweep.json").read_text())["decision"]
    text = (ROOT / "doc" / "engineering-reference.md").read_text()
    para = next(p for p in text.split("\n\n") if p.startswith("`k=60` with equal list weights"))
    assert ("On the recorded run (`bench/results/rrf_sweep.json`) the chosen setting's held-out interval covers zero, so the "
            "default stays") in para
    assert d["chosen_is_default"] is False and verdict(d["held_out_ndcg@10"]) == "covers zero"
    assert d["owner_decision_needed"] is False and d["default_changed"] is False
