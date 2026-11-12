"""The served-path scale runner (app/eval/served_scale.py) without a database, a model or an API.

The stages that need Postgres are in tests/database/test_served_scale.py; the GPU run itself is
`make bench-served-scale` (results in bench/results/served_scale_fiqa.json).
"""

import json
import os
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


def differences_for(query_form: dict) -> str:
    return "\n".join(served_scale.differences(
        rows=117, chunker="app/chunking (structure-aware)", rejected={"empty": 2, "duplicate_content": 1, "too_large": 0},
        not_indexed={"judged": ["9"]}, routes={"semantic": 5, "lexical_path": 1}, served_depth=50,
        search_settings={"hnsw.ef_search": 100}, in_memory_settings={"k": 100, "rrf_k": 60, "k1": 0.9, "b": 0.4, "reranker": "R", "precision": "fp16", "max_length": 512, "rerank_top": 20},
        reranker={"model": "R", "precision": "fp16", "max_length": 512, "depth": 20}, query_form=query_form))


def test_differences_are_filled_from_the_run_records():
    text = differences_for({"served": {"query": "Prompt words: ", "text": "", "source": "Model._model.prompts"}})
    assert "117 chunks" in text and "512 characters with 50 overlap" in text  # the window comes from fixed_size_chunks' defaults
    assert "rejected 3 documents" in text and "'too_large'" not in text and "['9']" in text
    assert "{'semantic': 5, 'lexical_path': 1}" in text and "100 + 100" in text and "50 + 50" in text
    # the query prompt is the one the run read from the served model, not a string typed into the runner
    assert "'Prompt words: '" in text and "Represent this" not in text and served_scale.SAME_FORM_SUFFIX in text
    assert "neither side puts anything before a query" in differences_for({"served": {"query": "", "text": "", "source": "HashEmbedding"}})


class FakeSentenceTransformer:
    def __init__(self, prompts):
        self.prompts = prompts


class FakeServedEmbedding:
    def __init__(self, prompts):
        self._model = FakeSentenceTransformer(prompts)


def test_served_prompts_are_read_from_the_model_object():
    got = served_scale.served_prompts(FakeServedEmbedding({"query": "Q: ", "text": ""}))
    assert (got["query"], got["text"]) == ("Q: ", "") and "FakeServedEmbedding._model.prompts" in got["source"]
    none = served_scale.served_prompts(object())  # the hash embedding: no sentence-transformers model, so no prompt
    assert (none["query"], none["text"]) == ("", "") and "no sentence-transformers prompts" in none["source"]


def test_query_vector_check_compares_directions():
    import numpy as np

    rng = np.random.default_rng(0)
    served, bare = rng.normal(size=(4, 8)).astype(np.float32), rng.normal(size=(4, 8)).astype(np.float32)
    out = served_scale.query_vector_check(served, served * 3.0, bare)  # the same directions at another length
    assert out["n_queries"] == 4
    assert out["with_served_prompt"]["max_abs_difference"] < 1e-6 and out["with_served_prompt"]["min_cosine"] > 0.999999
    assert out["bare_query"]["min_cosine"] < 0.99 and out["bare_query"]["max_abs_difference"] > 0.01


def test_served_rows_pair_with_the_in_memory_row_of_the_same_query_form():
    bare = "hybrid@bge-small"
    twin = served_scale.same_form_name(bare)
    assert twin == bare + served_scale.SAME_FORM_SUFFIX
    assert served_scale.paired_rows(bare, {bare: [], twin: []}) == (twin, bare)
    assert served_scale.paired_rows(bare, {bare: []}) == (bare, None)  # no prompt on either side (the hash dry run)


def test_committed_reproduction_counts_identical_rankings(tmp_path):
    run = tmp_path / "bench" / "runs" / "h.run"
    run.parent.mkdir(parents=True)
    run.write_text("q1 Q0 d1 1 2 h\nq1 Q0 d2 2 1 h\nq2 Q0 d3 1 1 h\n")
    committed = {"corpus_sha256": "abc", "n_queries": 2, "git_sha": "2583576", "device": {"device_name": "GPU X"},
                 "results": [{"config": "hybrid@bge-small", "retrieval": {"ndcg@10": 0.25}, "trec": {"run": "bench/runs/h.run", "run_sha256": served_scale.file_sha256(run)}}]}
    mine = {"hybrid@bge-small": [{"id": "q1", "ranking": ["d1", "d2"], "ndcg@10": 0.5}, {"id": "q2", "ranking": ["d4"], "ndcg@10": 0.0}],
            served_scale.same_form_name("hybrid@bge-small"): [{"id": "q1", "ranking": [], "ndcg@10": 0.0}, {"id": "q2", "ranking": [], "ndcg@10": 0.0}]}
    out = served_scale.committed_reproduction(mine, committed, tmp_path, corpus_sha256="abc", query_ids=["q1", "q2"])
    assert out["status"] == "compared" and out["git_sha"] == "2583576" and out["device"] == "GPU X"
    assert out["rows"] == {"hybrid@bge-small": {"committed_ndcg@10": 0.25, "this_job_ndcg@10": 0.25, "ndcg@10_equal": True, "identical_rankings": 1,
                                                "identical_top10": 1, "n": 2, "run_file": "bench/runs/h.run", "run_sha256_ok": True}}  # the twin has no committed row
    run.write_text("q1 Q0 d1 1 2 h\n")  # a run file that no longer matches its recorded digest is reported as such
    assert out["rows"]["hybrid@bge-small"]["run_sha256_ok"] is True
    assert served_scale.committed_reproduction(mine, committed, tmp_path, corpus_sha256="abc", query_ids=["q1", "q2"])["rows"]["hybrid@bge-small"]["run_sha256_ok"] is False
    assert served_scale.committed_reproduction(mine, committed, tmp_path, corpus_sha256="other", query_ids=["q1", "q2"])["status"] == "not_comparable"
    assert served_scale.committed_reproduction(mine, None, tmp_path, corpus_sha256="abc", query_ids=["q1", "q2"])["status"] == "not_run"


@pytest.fixture
def restore_environ():
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def test_inmemory_stage_adds_a_same_query_form_row_when_the_served_embedder_has_a_prompt(tmp_path, monkeypatch, restore_environ):
    from llama_index.core.settings import Settings

    monkeypatch.setattr(Settings, "_embed_model", None)  # restored after the test; the stage sets the served (hash) embedder
    bare = sorted(paired for _, paired in served_scale.PASSES.values())
    assert served_scale.main(args_for(tmp_path, "--stages", "files,inmemory", "--embedder", "hash", "--device", "cpu")) == 3
    blob = json.loads((tmp_path / "scratch" / "state" / "inmemory.json").read_text())
    assert sorted(blob["rows"]) == bare and blob["query_form"]["served"]["query"] == ""  # the hash embedding has no prompt
    assert blob["query_vectors_check"]["status"] == "not_run"
    monkeypatch.setattr(served_scale, "served_prompts", lambda model: {"query": "Q: ", "text": "", "source": "a test"})
    assert served_scale.main(args_for(tmp_path, "--stages", "inmemory", "--embedder", "hash", "--device", "cpu")) == 3
    blob = json.loads((tmp_path / "scratch" / "state" / "inmemory.json").read_text())
    assert sorted(blob["rows"]) == sorted(bare + [served_scale.same_form_name(b) for b in bare])
    for name in bare:
        assert blob["rows"][name]["settings"]["query_instruction"] is None
        assert blob["rows"][served_scale.same_form_name(name)]["settings"]["query_instruction"] == "Q: "
        assert blob["query_form"]["rows"][served_scale.same_form_name(name)] == {"dense_leg": "Q: <query>", "lexical_leg": "<query>", "reranker": "<query>"}
        assert [r["id"] for r in blob["rows"][name]["rows"]] == ["q1", "q2"]  # every judged query, sorted by id


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
    # each stage names the machine it ran on (a resumed run can finish on another node), without a host name
    assert set(recorded["machine"]) == {"device", "device_name", "gpu_memory_gb", "cpu_model", "allocation"} and recorded["machine"]["device"] == "cpu"
    assert "not recorded yet: index" in capsys.readouterr().out
    assert served_scale.main(args_for(tmp_path, "--stages", "files", "--resume", "--embedder", "hash", "--device", "cpu")) == 3
    assert "files: recorded, skipped" in capsys.readouterr().out


def test_stage_machines_say_whether_every_stage_ran_on_one_machine():
    a, b = {"device_name": "GPU A", "cpu_model": "C"}, {"device_name": "GPU B", "cpu_model": "C"}
    assert served_scale.stage_machines({"index": {"machine": a}, "replay_hybrid": {"machine": dict(a)}})["one_machine"] is True
    split = served_scale.stage_machines({"index": {"machine": a}, "replay_hybrid": {"machine": b}})
    assert split["one_machine"] is False and split["stage_machines"] == {"index": a, "replay_hybrid": b}


def test_files_stage_refuses_a_directory_holding_other_files(tmp_path):
    corpus = tmp_path / "scratch" / "corpus"
    corpus.mkdir(parents=True)
    (corpus / "stray.txt").write_text("not a document of this run")
    with pytest.raises(RuntimeError, match="not documents of this run"):
        served_scale.main(args_for(tmp_path, "--stages", "files", "--embedder", "hash", "--device", "cpu"))
    assert (corpus / "stray.txt").exists()  # nothing deleted


def test_the_served_results_file_reads_only_its_own_stages():
    asked = []

    class Recorder:
        queries = {"q1": "x"}

        def done(self, stage):
            asked.append(stage)
            return None

    with pytest.raises(RuntimeError, match="stages not recorded yet"):
        served_scale.assemble(Recorder())
    # a pgvector stage that failed (hnsw, variants) must not hold back served_scale_fiqa.json
    assert asked == list(served_scale.SERVED_STAGES)


def test_unknown_stages_are_refused(tmp_path):
    with pytest.raises(SystemExit):
        served_scale.main(args_for(tmp_path, "--stages", "files,nonsense"))


def test_make_target_writes_the_committed_results_file():
    makefile = (ROOT / "Makefile").read_text()
    assert "bench-served-scale:" in makefile and "-m app.eval.served_scale" in makefile
    assert served_scale.RESULTS_FILE == "bench/results/served_scale_fiqa.json"


def test_the_results_files_are_not_ignored_by_git():
    import subprocess

    # bench/results/*.json is ignored except for a list of committed artifacts; these two must be on it
    for rel in (served_scale.RESULTS_FILE, served_scale.HNSW_FILE, served_scale.TREC_DIR + "/api-hybrid.run"):
        ignored = subprocess.run(["git", "check-ignore", "-q", rel], cwd=ROOT).returncode
        assert ignored == 1, f"{rel} is ignored by .gitignore"
