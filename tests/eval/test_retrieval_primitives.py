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


def _route_lines() -> dict[str, str]:
    """The router module docstring's "Routes:" list, route name -> its description (continuation lines joined)."""
    import app.retrieval.router as router_module

    doc = router_module.__doc__ or ""
    lines, out, name = doc.split("Routes", 1)[1].splitlines()[1:], {}, None
    for line in lines:
        head = line.strip().split(" ", 1)[0]
        if head in {r.value for r in RouteType}:
            name, out[head] = head, line.split("—", 1)[1].strip()
        elif name and line.startswith("   ") and line.strip():
            out[name] += " " + line.strip()
        elif not line.strip():
            name = None
    return out


def test_router_docstring_names_every_route_and_what_it_skips():
    """B16: the module docstring lists every route and says what the code skips on it (RouteDecision.skip_embed /
    skip_rerank, which the in-memory index and the served retriever read): "no query embedding" exactly on the routes
    that skip the embedding, "no reranker" exactly on the routes that skip the reranker, "the full pipeline" on the
    others. It used to call the aggregation route a "metadata path", which no code has."""
    from app.retrieval.router import RouteDecision

    described = _route_lines()
    assert set(described) == {r.value for r in RouteType}
    for route in RouteType:
        text, decision = described[route.value], RouteDecision(route, "doc")
        assert ("no query embedding" in text) == decision.skip_embed(), (route, text)
        assert ("no reranker" in text) == decision.skip_rerank(), (route, text)
        assert ("the full pipeline" in text) == (not decision.skip_embed() and not decision.skip_rerank()), (route, text)
        assert "metadata" not in text


def test_hash_embedder_docstring_names_the_ngram_sizes_it_hashes():
    """B16: a reference encoder built from the n-gram sizes the HashEmbedder docstring names reproduces the class's
    vectors (the docstring said 3-grams while the loop hashes 3- and 4-grams)."""
    import hashlib
    import re

    import numpy as np

    from app.retrieval.embedders import HashEmbedder

    match = re.search(r"character ([\d\- and]+)-grams", HashEmbedder.__doc__ or "")
    assert match, HashEmbedder.__doc__
    sizes = [int(n) for n in re.findall(r"\d+", match.group(1))]

    def reference(text: str, dim: int) -> np.ndarray:
        vec = np.zeros(dim, dtype=np.float64)
        blob = text.lower()
        for n in sizes:
            for j in range(len(blob) - n + 1):
                digest = hashlib.blake2b(blob[j : j + n].encode("utf-8"), digest_size=8).digest()
                vec[int.from_bytes(digest[:4], "little") % dim] += 1.0 if digest[4] & 1 else -1.0
        return vec / max(np.linalg.norm(vec), 1e-12)

    texts = ["abcd", "learning_rate: 3e-4", "configs/run_047.yaml"]
    emb = HashEmbedder(dim=64)
    for text, row in zip(texts, emb.encode(texts)):
        assert np.allclose(row, reference(text, 64), atol=1e-6), (sizes, text)


def test_in_memory_index_takes_its_rrf_default_from_fuse():
    """B16: InMemoryHybridIndex's rrf_k defaults to fuse.RRF_K (the constant the served retriever reads), not a second
    literal; a value test cannot tell the two apart while both are 60, so this reads the signature's source."""
    import ast
    import inspect

    import app.retrieval.hybrid as hybrid

    tree = ast.parse(inspect.getsource(hybrid))
    init = next(node for cls in tree.body if isinstance(cls, ast.ClassDef) and cls.name == "InMemoryHybridIndex"
                for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    names = [a.arg for a in init.args.args]
    default = init.args.defaults[names.index("rrf_k") - (len(names) - len(init.args.defaults))]
    assert isinstance(default, ast.Name) and default.id == "RRF_K", ast.dump(default)
    assert any(isinstance(node, ast.ImportFrom) and node.module == "app.retrieval.fuse" and "RRF_K" in [a.name for a in node.names]
               for node in tree.body)
