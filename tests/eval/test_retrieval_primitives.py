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
    toks = tokenize("see --num_pairs=3 in configs/run_047.yaml")
    assert "--num_pairs" in toks or "num_pairs" in toks
    assert any("run_047" in t for t in toks)


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
