"""The database stages of the served-path scale run (app/eval/served_scale.py) on a tiny corpus.

Needs a live Postgres with pgvector (PG_TEST_DSN); skipped otherwise. The corpus is written the way
the run writes FiQA (one <doc_id>.txt per document, beir_api.write_corpus), indexed through the served
indexer with the hash embedder into a temporary table the schema migrations created (as the deploy step
does), and the stages' helpers are run against it: the build, the indexes read from the catalog and the
sizes, the unchanged re-run, the documents the scanner rejected, what start-up reads, the probe and the
plan facts.
"""

import os
import types
import uuid

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("PG_TEST_DSN"), reason="PG_TEST_DSN not set")

DIM = 64
LONG = " ".join(f"word{i} returns dividends" for i in range(120))  # well over one 512-character window
CORPUS = {"1": "index funds and expense ratios", "2": "", "3": "roth ira conversion rules", "4": "roth ira conversion rules", "5": LONG}
QRELS = {"q1": {"1": 1}, "q2": {"2": 1, "3": 1}, "q3": {"5": 2}}


@pytest.fixture(scope="module")
def indexed(tmp_path_factory):
    dsn = os.environ["PG_TEST_DSN"]
    saved = {k: os.environ.get(k) for k in ("PG_CONNECTION_STRING", "PSYCOPG2_CONNECTION_STRING", "PGVECTOR_TABLE", "EMBEDDING_DIM")}
    os.environ.update(PG_CONNECTION_STRING=dsn, PSYCOPG2_CONNECTION_STRING=dsn, PGVECTOR_TABLE="scaletest_" + uuid.uuid4().hex[:8], EMBEDDING_DIM=str(DIM))
    from app.database.migrate import drop_vector_table, migrate_all
    from app.database.vector_store_manager import VectorStoreManager
    from app.eval import served_scale
    from app.eval.beir_api import write_corpus
    from app.retrieval.embedders import HashEmbedder

    corpus_dir = tmp_path_factory.mktemp("scale") / "corpus"
    write_corpus(CORPUS, corpus_dir)
    vsm = VectorStoreManager(conn_string=dsn)
    with vsm.get_connection() as conn:
        migrate_all(conn, vsm.vector_table)  # the deploy step: the table, the manifest, the GIN and HNSW indexes, on the empty table
    emb = HashEmbedder(DIM)
    blob = served_scale.index_corpus(vsm, corpus_dir, lambda texts: emb.encode(texts).tolist(), "hash-64")
    yield vsm, blob, emb
    with vsm.get_connection() as conn:
        drop_vector_table(conn, vsm.vector_table)
    vsm.close()
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_build_counts_rejections_and_reads_the_migrated_indexes(indexed):
    vsm, blob, _ = indexed
    report = blob["report"]
    assert report["files_seen"] == 5 and report["files_indexed"] == 3
    assert report["rejected_by_reason"]["empty"] == 1 and report["rejected_by_reason"]["duplicate_content"] == 1
    assert blob["rows"] == report["chunks_embedded"] > 3  # the long document is split
    assert blob["embedding"]["texts"] == report["chunks_embedded"] and blob["embedding"]["calls"] == 3  # one embed call per file
    ddl, names = blob["indexes_ddl"], blob["index_names"]
    assert "USING hnsw (embedding vector_cosine_ops)" in ddl[names["hnsw"]] and "USING gin (text_tsv)" in ddl[names["text_index"]]
    assert names["hnsw"].endswith("_embedding_hnsw_m16_efc64")  # the migration's name, read from the catalog
    assert blob["analyze_seconds"] >= 0 and "per insert" in blob["index_maintenance"]
    sizes = blob["sizes"]
    assert sizes["rows"] == blob["rows"] and sizes["total_bytes"] >= sizes["table_bytes_with_toast"] + sizes["indexes_bytes"] - 1
    assert {i["method"] for i in sizes["indexes"].values()} >= {"hnsw", "gin", "btree"}
    assert sizes["avg_column_bytes"]["embedding"] > DIM * 4  # float4 values plus the varlena header
    rerun = blob["unchanged_rerun"]
    assert rerun["files_unchanged"] == 3 and rerun["chunks_embedded"] == 0 and rerun["embed_calls"] == 0


def test_rejected_documents_are_named_with_the_queries_that_judge_them(indexed):
    from app.eval import served_scale

    vsm, _, _ = indexed
    missing = served_scale.not_indexed(CORPUS, QRELS, set(vsm.ingest_manifest()))
    assert missing == {"n": 2, "ids": ["2", "4"], "judged": ["2"], "queries": ["q2"]}  # 4.txt repeats 3.txt, which is scanned first


def test_start_up_reads_every_chunk_and_finds_no_version_cluster(indexed):
    from app.eval import served_scale

    vsm, blob, _ = indexed
    parts = served_scale.startup_components(vsm)
    assert parts["rows_read"] == blob["rows"] and parts["clusters"] == 0
    assert parts["text_bytes_read"] > len(LONG)  # overlapping windows read the long document more than once


def test_probe_and_plan_facts_on_the_served_statement(indexed):
    from app.eval import served_scale

    vsm, blob, emb = indexed
    queries = [("q1", "index funds"), ("q3", "returns dividends")]
    result = served_scale.probe(vsm, queries, emb.encode([t for _, t in queries]), k=5)
    assert [r["id"] for r in result["queries"]] == ["q1", "q3"]
    assert all(r["hybrid_rows"] > 0 and r["lexical_rows"] > 0 for r in result["queries"])
    plan = vsm.explain_hybrid("returns dividends", emb.encode(["returns dividends"])[0].tolist(), k=5)
    facts = served_scale.plan_facts(plan, table=vsm.data_table, hnsw_index=blob["index_names"]["hnsw"], text_index=blob["index_names"]["text_index"])
    assert facts["execution_ms"] > 0 and facts["final_join_scan"] is not None
    assert any(s["alias"] == "t_1" for s in facts["scans"])  # the dense CTE's scan, whichever the planner chose
    assert "values]'::vector" in served_scale.elide_vectors(plan)


def test_index_stage_refuses_a_table_that_already_has_rows(indexed, tmp_path, monkeypatch):
    from app.eval import served_scale, service

    vsm, _, _ = indexed
    monkeypatch.setattr(service, "migrate", lambda env, **kwargs: 0.0)  # the fixture's migrate_all was the deploy step

    class Stub(served_scale.Run):
        def __init__(self):
            self.settings, self.scratch = {}, tmp_path

        def embed_model(self):
            return object()

        def postgres(self):
            return types.SimpleNamespace(env=lambda: {})

        def vsm(self):
            return vsm

    with pytest.raises(RuntimeError, match="already holds rows"):
        Stub().stage_index()
