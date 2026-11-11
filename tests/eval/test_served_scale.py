"""The served-path scale runner (app/eval/served_scale.py) without a database, a model or an API.

The stages that need Postgres are in tests/database/test_served_scale.py; the GPU run itself is
`make bench-served-scale` (results in bench/results/served_scale_fiqa.json).
"""

import json
import sys
import types
from pathlib import Path

import pytest

from app.eval import served_scale

ROOT = Path(__file__).resolve().parents[2]


def tiny_beir(tmp_path: Path) -> Path:
    """A BEIR-format directory <data>/fiqa with four documents and two judged queries (no zip: nothing to pin)."""
    root = tmp_path / "data" / "fiqa"
    (root / "qrels").mkdir(parents=True, exist_ok=True)
    docs = {"1": "alpha beta", "2": "", "3": "gamma", "4": "gamma", "5": "delta epsilon"}
    (root / "corpus.jsonl").write_text("".join(json.dumps({"_id": d, "title": "", "text": t}) + "\n" for d, t in docs.items()))
    (root / "queries.jsonl").write_text("".join(json.dumps({"_id": q, "text": t}) + "\n" for q, t in {"q2": "gamma?", "q1": "alpha?", "q9": "unjudged"}.items()))
    (root / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\t1\t1\nq2\t3\t1\nq2\t2\t1\n")
    return root.parent


def test_select_keeps_the_split_or_a_labelled_slice():
    corpus = {str(i): f"doc {i}" for i in range(10)}
    queries = {"b": "x", "a": "y", "c": "z"}
    qrels = {"a": {"7": 1}, "b": {"3": 1, "missing": 1}, "c": {"9": 1}}
    c, q, r, note = served_scale.select(corpus, queries, qrels)
    assert list(q) == ["a", "b", "c"] and c == corpus and note == "full split"
    c, q, r, note = served_scale.select(corpus, queries, qrels, max_queries=2, max_docs=4)
    assert list(q) == ["a", "b"] and set(r) == {"a", "b"}
    assert sorted(c) == ["0", "1", "3", "7"]  # the judged documents, then the first others by id
    assert "first 2 queries" in note and "slice of 4 documents" in note


@pytest.mark.parametrize("plan_file, dense_hnsw, table, hnsw", [
    ("doc/sql/hybrid_explain_synthetic_10k.txt", True, "data_explain_synthetic", "explain_synthetic_embedding_hnsw"),
    ("doc/sql/hybrid_explain.txt", False, "data_llamaindex_embedding", "llamaindex_embedding_embedding_hnsw"),
])
def test_plan_facts_read_the_committed_plans(plan_file, dense_hnsw, table, hnsw):
    plan = (ROOT / plan_file).read_text()
    facts = served_scale.plan_facts(plan, table=table, hnsw_index=hnsw, text_index=f"{table}_text_tsv_gin")
    assert facts["dense_uses_hnsw"] is dense_hnsw
    assert facts["lexical_uses_gin"] is True
    assert facts["final_join_scan"] == "Seq Scan"  # the join back to the table by node_id has no index to use
    assert facts["execution_ms"] > 0
    assert {s["alias"] for s in facts["scans"] if s["relation"] == table} >= {"t", "t_1", "t_2"}


def test_elide_vectors_keeps_the_length_of_each_literal():
    plan = "Order By: (embedding <=> '[0.1,-2e-05,3]'::vector)\nOrder By: (e <=> '[1,2]'::halfvec)"
    assert served_scale.elide_vectors(plan) == "Order By: (embedding <=> '[3 values]'::vector)\nOrder By: (e <=> '[2 values]'::halfvec)"


def test_not_indexed_names_judged_documents_and_their_queries():
    corpus = {"1": "a", "2": "", "3": "x", "4": "x"}
    out = served_scale.not_indexed(corpus, {"q1": {"1": 1}, "q2": {"2": 1, "3": 1}}, {"1.txt", "3.txt"})
    assert out == {"n": 2, "ids": ["2", "4"], "judged": ["2"], "queries": ["q2"]}


def test_differences_are_filled_from_the_run_records():
    lines = served_scale.differences(rows=117, chunker="app/chunking (structure-aware)", rejected={"empty": 2, "duplicate_content": 1, "too_large": 0},
                                     not_indexed={"judged": ["9"]}, routes={"semantic": 5, "lexical_path": 1}, served_depth=50,
                                     search_settings={"hnsw.ef_search": 100}, in_memory_settings={"k": 100, "rrf_k": 60, "k1": 0.9, "b": 0.4, "reranker": "R", "precision": "fp16", "max_length": 512, "rerank_top": 20},
                                     reranker={"model": "R", "precision": "fp16", "max_length": 512, "depth": 20})
    text = "\n".join(lines)
    assert "117 chunks" in text and "512 characters with 50 overlap" in text  # the window comes from fixed_size_chunks' defaults
    assert "rejected 3 documents" in text and "'too_large'" not in text and "['9']" in text
    assert "{'semantic': 5, 'lexical_path': 1}" in text and "100 + 100" in text and "50 + 50" in text
    assert "Represent this question for searching relevant passages: " in text


def test_trec_outputs_cross_check_catches_a_metric_mismatch(tmp_path, monkeypatch):
    rankings = {"api:hybrid": {"q1": ["d1", "d2"], "q2": []}}
    per_query = {"api:hybrid": [{"id": "q1", "ndcg@10": 1.0, "recall@10": 1.0, "recall@50": 1.0}, {"id": "q2", "ndcg@10": 0.0, "recall@10": 0.0, "recall@50": 0.0}]}
    qrels = {"q1": {"d1": 1}, "q2": {"d3": 1}}
    monkeypatch.setitem(sys.modules, "pytrec_eval", None)  # not installed
    out = served_scale.trec_outputs(tmp_path, rankings, qrels, per_query)
    assert out["cross_check"]["status"] == "not_run"
    assert (tmp_path / "api-hybrid.run").read_text() == "q1 Q0 d1 1 2 api-hybrid\nq1 Q0 d2 2 1 api-hybrid\n"

    class Evaluator:
        def __init__(self, qrels, measures):
            pass

        def evaluate(self, run):
            assert set(run) == {"q1"}  # an empty ranking is not in a TREC run; it scores zero on both sides
            return {"q1": {"ndcg_cut_10": 1.0, "recall_10": 1.0, "recall_50": self.recall_50}}

    monkeypatch.setitem(sys.modules, "pytrec_eval", types.SimpleNamespace(RelevanceEvaluator=Evaluator))
    Evaluator.recall_50 = 1.0
    assert served_scale.trec_outputs(tmp_path, rankings, qrels, per_query)["cross_check"]["status"] == "passed"
    Evaluator.recall_50 = 0.5
    with pytest.raises(RuntimeError, match="recall@50"):
        served_scale.trec_outputs(tmp_path, rankings, qrels, per_query)


def args_for(tmp_path, *extra):
    return ["--data-dir", str(tiny_beir(tmp_path)), "--container", str(tmp_path / "pg.sif"), "--scratch", str(tmp_path / "scratch"), *extra]


def test_files_stage_writes_the_corpus_records_it_and_is_skipped_on_resume(tmp_path, capsys):
    assert served_scale.main(args_for(tmp_path, "--stages", "files", "--embedder", "hash", "--device", "cpu")) == 3  # incomplete: no results file
    corpus = tmp_path / "scratch" / "corpus"
    assert sorted(p.name for p in corpus.iterdir()) == ["1.txt", "2.txt", "3.txt", "4.txt", "5.txt"]
    recorded = json.loads((tmp_path / "scratch" / "state" / "files.json").read_text())
    assert recorded["n_files"] == 5 and recorded["stage"] == "files"
    assert "not recorded yet: index" in capsys.readouterr().out
    assert served_scale.main(args_for(tmp_path, "--stages", "files", "--resume", "--embedder", "hash", "--device", "cpu")) == 3
    assert "files: recorded, skipped" in capsys.readouterr().out


def test_files_stage_refuses_a_directory_holding_other_files(tmp_path):
    corpus = tmp_path / "scratch" / "corpus"
    corpus.mkdir(parents=True)
    (corpus / "stray.txt").write_text("not a document of this run")
    with pytest.raises(RuntimeError, match="not documents of this run"):
        served_scale.main(args_for(tmp_path, "--stages", "files", "--embedder", "hash", "--device", "cpu"))
    assert (corpus / "stray.txt").exists()  # nothing deleted


def test_unknown_stages_are_refused(tmp_path):
    with pytest.raises(SystemExit):
        served_scale.main(args_for(tmp_path, "--stages", "files,nonsense"))


def test_make_target_writes_the_committed_results_file():
    makefile = (ROOT / "Makefile").read_text()
    assert "bench-served-scale:" in makefile and "-m app.eval.served_scale" in makefile
    assert served_scale.RESULTS_FILE == "bench/results/served_scale_fiqa.json"
