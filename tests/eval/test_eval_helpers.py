from app.eval.gold import GoldQuestion, load_gold, write_gold
from app.eval.latency import StageTimer, percentile
from app.eval.ragas_metrics import faithfulness, context_precision, context_recall
from app.eval.reindex import files_to_reindex
from app.retrieval.distill import DistilledReranker, RerankTriple
from app.retrieval.hybrid import Chunk
from app.retrieval.rerank import extract_features, feature_vector


def test_percentile_and_timer():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    timer = StageTimer()
    with timer.stage("embed"):
        pass
    summary = timer.summary()
    assert summary["embed"]["n"] == 1
    assert "p95_ms" in summary["embed"]


def test_gold_roundtrip(tmp_path):
    q = GoldQuestion(
        id="q1",
        question="lr?",
        category="simple_factual",
        answer="3e-4",
        relevant_paths=["configs/run_047.yaml"],
    )
    path = tmp_path / "q.jsonl"
    write_gold(path, [q])
    loaded = load_gold(path)
    assert loaded[0].relevant_ids() == ["configs/run_047.yaml"]


def test_ragas_heuristics():
    assert faithfulness("fusion concatenates features", ["the fusion module concatenates features"]) > 0.3
    assert context_precision(["a"], ["a", "b"], k=10) == 0.5
    assert context_recall(["a", "b"], ["a"], k=50) == 0.5


def test_reindex_only_changed_hashes():
    prev = {"a": "1", "b": "2"}
    cur = {"a": "1", "b": "3", "c": "4"}
    assert files_to_reindex(cur, prev) == ["b", "c"]
    assert files_to_reindex(prev, prev) == []


def test_distilled_reranker_learns_teacher_order():
    triples = []
    for i in range(20):
        feats = {
            "dense_cosine": 0.1 * (i % 3),
            "bm25_score": float(i),
            "rrf_score": 0.05 * i,
            "rrf_rank": float(20 - i),
            "jaccard": 0.1,
            "exact_overlap": 0.2,
            "path_depth": 2.0,
            "recency": 1.0,
            "is_yaml": 1.0,
            "is_log": 0.0,
            "is_code": 0.0,
            "is_csv": 0.0,
            "is_md": 0.0,
        }
        triples.append(
            RerankTriple("q", f"c{i}", "configs/x.yaml", ce_score=float(i), features=feats)
        )
    student = DistilledReranker().fit(triples)
    chunks = [
        Chunk("low", "configs/a.yaml", "x", 0, 1),
        Chunk("high", "configs/b.yaml", "y", 0, 1),
    ]
    pairs = [(chunks[0], 0.1), (chunks[1], 0.9)]
    ranked = student("q", pairs)
    assert ranked[0][0].chunk_id in {"low", "high"}
    vec = feature_vector(extract_features("q", chunks[0], bm25=1.0))
    assert len(vec) == 13
