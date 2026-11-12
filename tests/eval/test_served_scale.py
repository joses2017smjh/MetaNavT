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


@pytest.mark.parametrize("plan_file, uses_indexes, table, hnsw", [
    # the index names the schema migrations give (app/database/migrations/vector/0005 and 0007)
    ("doc/sql/hybrid_explain_synthetic_10k.txt", True, "data_explain_synthetic", "explain_synthetic_embedding_hnsw_m16_efc64"),
    ("doc/sql/hybrid_explain.txt", False, "data_llamaindex_embedding", "llamaindex_embedding_embedding_hnsw_m16_efc64"),  # 71 rows: both legs scan the table
])
def test_plan_facts_read_the_committed_plans(plan_file, uses_indexes, table, hnsw):
    plan = (ROOT / plan_file).read_text()
    facts = served_scale.plan_facts(plan, table=table, hnsw_index=hnsw, text_index=f"{table}_text_tsv_english_gin")
    assert facts["dense_uses_hnsw"] is uses_indexes
    assert facts["lexical_uses_gin"] is uses_indexes
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

    from app.eval.report import SERVED_SCALE_PRE_M20_FILE

    # bench/results/*.json is ignored except for a list of committed artifacts; these must be on it (the earlier run too,
    # whose start-up the README block renders). --no-index: git check-ignore never reports a tracked file, and the files
    # are tracked, so without it this could not fail.
    for rel in (served_scale.RESULTS_FILE, served_scale.HNSW_FILE, served_scale.TREC_DIR + "/api-hybrid.run", SERVED_SCALE_PRE_M20_FILE):
        ignored = subprocess.run(["git", "check-ignore", "-q", "--no-index", rel], cwd=ROOT).returncode
        assert ignored == 1, f"{rel} is ignored by .gitignore"


def test_the_plan_block_reads_the_stored_plans_with_the_catalog_names():
    """The facts come from the plan text and the index names the index stage read from pg_indexes, whatever names
    the explain stage recorded (a stage file once carried the pre-migration names, and every fact came out False)."""
    plan = (ROOT / "doc/sql/hybrid_explain_synthetic_10k.txt").read_text()
    stage = {"queries": [{"id": "q1", "facts": {"dense_uses_hnsw": False}, "plan": plan.splitlines()}], "settings": {"hnsw.ef_search": 100},
             "names": {"table": "data_explain_synthetic", "hnsw_index": "explain_synthetic_embedding_hnsw", "text_index": "data_explain_synthetic_text_tsv_gin"}}
    catalog = {"hnsw": "explain_synthetic_embedding_hnsw_m16_efc64", "text_index": "data_explain_synthetic_text_tsv_english_gin"}
    block = served_scale.plan_block(stage, catalog, table="data_explain_synthetic")
    facts = block["queries"][0]["facts"]
    assert facts["dense_uses_hnsw"] is True and facts["lexical_uses_gin"] is True and facts["final_join_scan"] == "Seq Scan"
    assert block["names"] == {"table": "data_explain_synthetic", "hnsw_index": catalog["hnsw"], "text_index": catalog["text_index"]}
    assert block["settings"] == {"hnsw.ef_search": 100} and block["queries"][0]["plan"] == plan.splitlines()


def test_git_dirty_ignores_the_files_the_run_writes(tmp_path):
    import subprocess

    def git(*args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    trec = tmp_path / served_scale.TREC_DIR
    trec.mkdir(parents=True)
    (trec / "api-hybrid.run").write_text("old\n")
    (tmp_path / served_scale.RESULTS_FILE).write_text("{}\n")
    (tmp_path / "code.py").write_text("x = 1\n")
    git("add", "-A")
    git("commit", "-q", "-m", "init")
    assert served_scale.git_dirty(tmp_path) is False
    (trec / "api-hybrid.run").write_text("new\n")  # the run rewrites its own committed outputs
    (tmp_path / served_scale.RESULTS_FILE).write_text('{"new": 1}\n')
    assert served_scale.git_dirty(tmp_path) is False
    (tmp_path / "code.py").write_text("x = 2\n")
    assert served_scale.git_dirty(tmp_path) is True


BUILD_REPORT = {"files_indexed": 3, "files_rejected": 2, "chunks_embedded": 7, "n_nodes": 7, "pipeline": "app.chunking/auto/v2|hash-64"}


def test_the_deploy_orders_index_step_must_accept_the_build(tmp_path, monkeypatch):
    """After the timed build, `python -m app.engine.ingest` (the step a deployment runs before the API) runs over the same
    files: it must find nothing to embed under the same pipeline, or the API would serve an index that step would not have written."""
    from app.eval import service

    answer = {}
    monkeypatch.setattr(service, "index_step", lambda env, **kwargs: (dict(answer), 12.3456))
    answer.update(chunks_embedded=0, files_indexed=0, files_unchanged=3, files_rejected=2, n_nodes=7, pipeline=BUILD_REPORT["pipeline"], data_dir="/somewhere")
    out = served_scale.index_step_unchanged({}, BUILD_REPORT, log_path=tmp_path / "index-step.log")
    assert out["seconds"] == 12.346 and out["command"] == "python -m app.engine.ingest"
    assert out["report"]["files_unchanged"] == 3 and out["report"]["chunks_embedded"] == 0
    accepted = dict(answer)
    # each field on its own: chunks embedded, a file indexed anew, another pipeline, a file not found unchanged, another
    # rejection count, other rows
    for change in ({"chunks_embedded": 2}, {"files_indexed": 1}, {"pipeline": "app.chunking/auto/v2|hash-32"}, {"files_unchanged": 2}, {"files_rejected": 1}, {"n_nodes": 6}):
        answer.clear()
        answer.update(accepted, **change)
        with pytest.raises(RuntimeError, match="index step"):
            served_scale.index_step_unchanged({}, BUILD_REPORT, log_path=tmp_path / "index-step.log")


def test_the_build_passes_the_arguments_the_index_step_passes(tmp_path, monkeypatch):
    """The timed build calls index_directory as app/engine/ingest.index_data_dir does (the call of the index step and of a
    start with INDEX_ON_START): the same directory, embedder label and rejection limit; only embed_fn is wrapped by a timer."""
    from llama_index.core.settings import Settings

    from app.engine import ingest
    from app.engine.embeddings import HashEmbedding

    calls = []
    monkeypatch.setattr(ingest, "index_directory", lambda vsm, data_dir, **kwargs: calls.append((data_dir, kwargs)) or {})
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INGEST_MAX_REJECTED", "7")
    model = HashEmbedding(8)
    monkeypatch.setattr(Settings, "_embed_model", model)
    ingest.index_data_dir(object())
    served_scale.build_index(object(), tmp_path, model, embed_fn=lambda texts: [])
    (step_dir, step), (build_dir, build) = calls
    assert step_dir == build_dir == str(tmp_path)

    def same(kwargs):
        return {k: v for k, v in kwargs.items() if k != "embed_fn" and v is not None}

    assert same(build) == same(step) == {"embedder": "hash-8", "max_rejected": 7}


def test_start_up_health_must_list_what_the_build_wrote():
    build = {"rows": 7, "report": BUILD_REPORT}
    health = {"n_nodes": 7, "indexing": {"indexed": False, "manifest_at_start": {"files": 3, "chunks": 7, "pipelines": [BUILD_REPORT["pipeline"]]}}}
    served_scale.check_startup_health(health, build)
    for bad in ({"n_nodes": 6}, {"indexing": {"indexed": False, "reason": "INDEX_ON_START not set"}},  # an API from before M20 reports no manifest
                {"indexing": {"manifest_at_start": {"files": 3, "chunks": 7, "pipelines": ["app.chunking/auto/v2|other"]}}},
                {"indexing": {"manifest_at_start": {"files": 2, "chunks": 7, "pipelines": [BUILD_REPORT["pipeline"]]}}},
                {"indexing": {"manifest_at_start": {"files": 3, "chunks": 6, "pipelines": [BUILD_REPORT["pipeline"]]}}}):
        with pytest.raises(RuntimeError, match="does not match the index stage"):
            served_scale.check_startup_health({**health, **bad}, build)


def test_the_load_note_says_whether_the_api_had_just_served_the_replay(tmp_path, monkeypatch):
    """The note was typed ("the service had just served the replay"); a resumed run that starts the API for the load stage
    would have carried it too. It now comes from what the running API served in this process."""
    from app.eval import beir_api, http_load, service

    class Api:
        url = "http://127.0.0.1:1"

        def __init__(self, env, **kwargs):
            pass

        def start(self):
            return self

        def wait_ready(self, timeout):
            return {"n_nodes": 1, "reranker": {"loaded": False}}, 0.25

        def stop(self):
            pass

    async def fake_load(url, queries, **kwargs):
        return {"runs": []}

    monkeypatch.setattr(service, "ApiServer", Api)
    monkeypatch.setattr(http_load, "run_load", fake_load)
    monkeypatch.setattr(beir_api, "replay", lambda post, queries, qrels, **kwargs: [])
    def fake_postgres(self):
        self.pg = types.SimpleNamespace(env=lambda: {}, stop=lambda: None)
        return self.pg

    monkeypatch.setattr(served_scale.Run, "postgres", fake_postgres)
    run = served_scale.Run(served_scale.parse_args(args_for(tmp_path, "--embedder", "hash", "--device", "cpu")))
    assert run.stage_load()["report"]["workload_note"].endswith("; the API was started for this stage")  # a resumed run
    run.stop_api()
    run.stage_replay("api:hybrid+rerank")  # starts the API, as the replay stage of a full run does
    assert run.stage_load()["report"]["workload_note"].endswith("; the API had served replay_rerank in this process, without a restart")
    run.close()


def test_the_results_file_hashes_the_modules_that_produce_its_numbers():
    """source_sha256 names the code behind the file: the index step, the start-up reads and their text-read counter among it."""
    for rel in ("app/engine/ingest.py", "app/engine/bootstrap.py", "app/engine/ingest_scale.py", "app/eval/service.py", "app/database/vector_store_manager.py"):
        assert rel in served_scale.SOURCES and (ROOT / rel).exists()
