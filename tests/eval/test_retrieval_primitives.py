import random

import pytest

from app.retrieval.fuse import reciprocal_rank_fusion, rrf_score_map
from app.retrieval.bm25 import BM25Index, tokenize
from app.retrieval.router import QueryRouter, RouteType


def test_rrf_prefers_items_high_in_both_lists():
    fused = reciprocal_rank_fusion(
        [
            ["shared", "only_a"],
            ["shared", "only_b"],
        ]
    )
    assert fused[0][0] == "shared"
    assert fused[0][1] > fused[1][1]


def test_rrf_k60_first_rank_score():
    scores = rrf_score_map([["a"]], k=60)
    assert abs(scores["a"] - 1.0 / 61) < 1e-9


def _rrf_before_list_weights(ranked_lists, k=60):
    """rrf_score_map as it was before it took list weights (34146bd), kept to pin the default fusion."""
    fused = {}
    for ranked in ranked_lists:
        for rank, key in enumerate(ranked, start=1):
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
    return fused


def test_rrf_weight_one_reproduces_the_unweighted_fusion_exactly():
    """B12: weight 1.0 on every list is the unweighted fusion to the last bit and in the same key order (the hybrid
    index sorts the map stably, so the insertion order breaks score ties), for every k of the sweep grid."""
    rng = random.Random(0)
    ids = [f"chunk{i}" for i in range(71)]
    for k in (1, 10, 30, 60, 100, 200):
        for _ in range(25):
            lists = [rng.sample(ids, rng.randint(0, 50)) for _ in range(rng.choice((1, 2, 3)))]
            before = list(_rrf_before_list_weights(lists, k=k).items())
            assert list(rrf_score_map(lists, k=k).items()) == before
            assert list(rrf_score_map(lists, k=k, weights=[1.0] * len(lists)).items()) == before


def test_rrf_weights_scale_each_lists_contribution():
    scores = rrf_score_map([["a", "b"], ["b", "c"]], k=60, weights=[2.0, 0.5])
    assert scores == {"a": 2.0 / 61, "b": 2.0 / 62 + 0.5 / 61, "c": 0.5 / 62}
    # Only the ratio of the weights changes the order: scaling both leaves it as it was.
    lists = [["a", "b", "c"], ["c", "d", "a"]]
    one, two = rrf_score_map(lists, k=10, weights=[1.0, 3.0]), rrf_score_map(lists, k=10, weights=[2.0, 6.0])
    assert sorted(one, key=one.get) == sorted(two, key=two.get)


@pytest.mark.parametrize("weights", [[1.0], [1.0, 1.0, 1.0], [1.0, 0.0], [1.0, -0.5], [1.0, float("nan")], [float("inf"), 1.0]])
def test_rrf_weights_are_one_positive_finite_number_per_list(weights):
    with pytest.raises(ValueError):
        rrf_score_map([["a"], ["b"]], k=60, weights=weights)


def test_bm25_ranks_exact_term_higher():
    idx = BM25Index().fit(
        ["d1", "d2", "d3"],
        [
            "learning_rate: 3e-4 encoder dinov2 fusion true",
            "the quick brown fox",
            "learning_rate: 1e-4 encoder resnet50",
        ],
    )
    hits = idx.search("learning_rate dinov2", k=3)
    assert hits[0][0] == "d1"
    assert hits[0][1] > hits[1][1]


def test_tokenize_keeps_flags_and_paths():
    # A flag keeps its dashes and a path stays one token; "=" ends the flag. Bare "num_pairs" is not a token.
    assert tokenize("see --num_pairs=3 in configs/run_047.yaml") == ["see", "--num_pairs", "3", "in", "configs/run_047.yaml"]


def test_router_path_aggregation_semantic():
    r = QueryRouter()
    path = r.route("open configs/run_047.yaml")
    assert path.route == RouteType.LEXICAL_PATH
    assert path.skip_embed() and path.skip_rerank()

    agg = r.route("how many runs used 3 stereo pairs")
    assert agg.route == RouteType.AGGREGATION
    assert agg.skip_rerank()

    sem = r.route("what does the fusion module do")
    assert sem.route == RouteType.SEMANTIC
    assert not sem.skip_embed()

    art = r.route("reproduce run 47 from the paper")
    assert art.route == RouteType.RESEARCH_ARTIFACT
    assert not art.skip_embed()

    code = r.route("write a test for the fusion module")
    assert code.route == RouteType.CODE_PRODUCTION
