"""README blocks for the served-path scale run (app/eval/report.py): bench-served-scale and bench-pgvector-hnsw.

The blobs below have the shape app/eval/served_scale.assemble and assemble_hnsw write; the numbers are made up.
"""

import copy

from app.eval import report
from app.eval.served_scale import SAME_FORM_SUFFIX

RERANK = "hybrid+bge-rerank@bge-small/fp16-512/top20"
HYBRID = "hybrid@bge-small"
MACHINE = {"device": "cuda", "device_name": "GPU X", "gpu_memory_gb": 44.4, "cpu_model": "CPU Y", "allocation": {"cpus": "8", "memory_mb": "65536"}}


def ci(mean):
    return {"mean": mean, "lo": round(mean - 0.02, 4), "hi": round(mean + 0.02, 4)}


def delta(d):
    return {"delta": d, "lo": round(d - 0.01, 4), "hi": round(d + 0.01, 4), "covers_zero": d - 0.01 <= 0 <= d + 0.01}


def summary(ndcg):
    return {"n_queries": 4, "retrieval": {"ndcg@10": ndcg, "recall@10": 0.5, "recall@50": 0.7, "mrr@10": 0.4}, "ci": {"ndcg@10": ci(ndcg), "recall@10": ci(0.5), "recall@50": ci(0.7)}}


def served_blob():
    served = {}
    for row, bare, ndcg in (("api:hybrid+rerank", RERANK, 0.3123), ("api:hybrid", HYBRID, 0.2456)):
        served[row] = {**summary(ndcg), "requests": {"failed": 1}, "latency": {"e2e": {"n": 3, "p50_ms": 900.5, "p95_ms": 1500.25}},
                       "paired_with": bare + SAME_FORM_SUFFIX, "bare_query_row": bare,
                       "delta_vs_in_memory": {m: delta(-0.0512) for m in ("ndcg@10", "recall@10", "recall@50")},
                       "delta_vs_bare_query_row": {m: delta(-0.0634) for m in ("ndcg@10", "recall@10", "recall@50")},
                       "documents_per_response": {"mean": 31.5, "queries_under_50": 4}, "routes": {"semantic": 3, "lexical_path": 1}}
    in_memory = {name: {**summary(0.3645), "latency": {"total": {"n": 4, "p50_ms": 60.1, "p95_ms": 90.2}}} for name in (RERANK, RERANK + SAME_FORM_SUFFIX, HYBRID, HYBRID + SAME_FORM_SUFFIX)}
    return {
        "schema": "served-scale-v2", "n_docs": 57638, "n_queries": 4, "query_selection": "full split", "embedder_note": None, "git_sha": "abc1234",
        "stage_machines": {"index": MACHINE, "inmemory": dict(MACHINE)},
        "build": {"rows": 117000, "server": {"postgres": "PostgreSQL 16.15", "pgvector": "0.8.6", "settings": {"shared_buffers": "128MB", "maintenance_work_mem": "64MB", "work_mem": "4MB"}},
                  "report": {"files_rejected": 40, "rejected_by_reason": {"empty": 38, "duplicate_content": 2, "too_large": 0}, "files_indexed": 57598, "chunks_embedded": 117000},
                  "migrate_seconds": 12.25, "index_directory_seconds": 1200.5, "embedding": {"seconds": 300.25}, "files_per_second": 47.98, "analyze_seconds": 2.5,
                  "index_names": {"hnsw": "h_idx", "text_index": "g_idx"}, "indexes_ddl": {"h_idx": "CREATE INDEX h_idx ... WITH (m='16', ef_construction='64')"},
                  "sizes": {"table_bytes_with_toast": 500 * 2**20, "indexes": {"h_idx": {"bytes": 230 * 2**20}, "g_idx": {"bytes": 60 * 2**20}, "pkey": {"bytes": 3 * 2**20}}},
                  "unchanged_rerun": {"seconds": 45.5}},
        "not_indexed": {"judged": ["117276"]},
        "startup": {"api": {"seconds_to_health_ok": 61.5}, "components": {"fetch_all_chunks_seconds": 3.25, "build_clusters_seconds": 6.5}},
        "query_form": {"served": {"query": "Prompt words: ", "text": "", "source": "S"}, "rows": {RERANK + SAME_FORM_SUFFIX: {"dense_leg": "Prompt words: <query>"}, HYBRID + SAME_FORM_SUFFIX: {"dense_leg": "Prompt words: <query>"},
                                                                                         RERANK: {"dense_leg": "<query>"}, HYBRID: {"dense_leg": "<query>"}},
                       "vectors_check": {"status": "compared", "n_queries": 16, "with_served_prompt": {"min_cosine": 1.0, "max_abs_difference": 1e-07}, "bare_query": {"mean_cosine": 0.93}},
                       "in_memory_prompt_effect": {HYBRID: {"row": HYBRID + SAME_FORM_SUFFIX, "delta": {"ndcg@10": delta(0.0071)}}}},
        "served": served, "in_memory": in_memory,
        "in_memory_reproduction": {"status": "compared", "file": "bench/results/beir_fiqa.json", "git_sha": "2583576", "device": "GPU Z",
                                   "rows": {HYBRID: {"this_job_ndcg@10": 0.3645, "committed_ndcg@10": 0.3645, "identical_rankings": 640, "identical_top10": 647, "n": 648}}},
        "plan": {"settings": {"hnsw.ef_search": 100}, "queries": [{"facts": {"dense_uses_hnsw": True, "lexical_uses_gin": True, "final_join_scan": "Seq Scan", "execution_ms": 812.5}}] * 3},
        "load": {"k": 8, "workload": {"requests_per_level": 64}, "workload_note": "the first 64 test queries by id",
                 "runs": [{"concurrency": c, "attempted": 64, "failed": f, "error_counts": e, "e2e_successful": {"n": 64 - f, "p50_ms": 1000.5 * c, "p95_ms": 2000.5 * c}, "successful_requests_per_second": 1.25}
                          for c, f, e in ((1, 0, {}), (4, 0, {}), (16, 6, {"http_503": 6}))]},
    }


def test_served_block_shows_every_row_with_its_pair_and_labels_the_fallback():
    text = report.served_scale_table(served_blob())
    assert "57638 documents" in text and "117000 chunks" in text and "rejected 40 documents: empty 38, duplicate_content 2" in text
    assert "judged relevant among them: 1" in text and "GPU X (44.4 GB) with CPU Y, 8 CPUs allocated" in text and "results sha abc1234" in text
    assert "Postgres ts_rank_cd on the served side (a fallback, not BM25)" in text and "| api:hybrid+rerank | served | ts_rank_cd (fallback) | prompt + query | 0.3123 [0.2923, 0.3323]" in text
    for name in (RERANK, RERANK + SAME_FORM_SUFFIX, HYBRID, HYBRID + SAME_FORM_SUFFIX):  # every in-memory row stays in the table
        assert f"| {name} | in memory | Okapi BM25 |" in text
    assert f"| {HYBRID} | in memory | Okapi BM25 | query |" in text and f"| {HYBRID + SAME_FORM_SUFFIX} | in memory | Okapi BM25 | prompt + query |" in text
    assert "'Prompt words: '" in text and "1 of 4 | 900.5 / 1500.25 (HTTP)" in text
    assert f"api:hybrid minus {HYBRID + SAME_FORM_SUFFIX} (the same query form), paired over 4 queries: nDCG@10 -0.0512 [-0.0612, -0.0412] (the interval excludes zero)" in text
    assert f"Against {HYBRID} (bare query): nDCG@10 -0.0634" in text and "The served prompt alone, in memory" in text and "(the interval covers zero)" in text
    assert "minimum cosine 1.0" in text and "the same ranking for 640 of 648 queries (the same first 10 for 647)" in text
    assert "Recall@50: a served ranking holds the documents of at most 50 returned chunks (mean 31.5" in text
    assert "the deploy step (`python -m app.database.migrate`) 12.25 s on the empty database, creating the table, the HNSW index (m 16, ef_construction 64)" in text
    assert "index_directory 1200.5 s for 57598 files and 117000 chunks with both indexes maintained per insert" in text
    assert "the table 500.0 MiB" in text and "the HNSW index 230.0 MiB, the GIN index 60.0 MiB, 1 other indexes 3.0 MiB" in text
    assert "61.5 s from launch to /health ok" in text and "the dense CTE read the HNSW index in 3 of 3" in text
    assert "| 16 | 64 | 6 | http_503 6 | 16008.0 / 32008.0 | 1.25 |" in text


def test_served_block_refuses_a_slice_or_a_dry_run_and_names_split_machines():
    slice_blob = served_blob()
    slice_blob["query_selection"] = "first 40 queries by id; slice of 4000 documents"
    assert "not the benchmark" in report.served_scale_table(slice_blob) and "0.3123" not in report.served_scale_table(slice_blob)
    dry = served_blob()
    dry["embedder_note"] = "hash embedder, no reranker: a dry run of the machinery, not a measurement"
    assert "not the benchmark" in report.served_scale_table(dry)
    assert "is missing; run `make bench-served-scale`" in report.served_scale_table(None)
    moved = served_blob()
    moved["stage_machines"]["inmemory"] = {**MACHINE, "device_name": "GPU W"}
    assert "stages on 2 different machines" in report.served_scale_table(moved)


def hnsw_blob():
    row = {"recall@10": ci(0.97), "recall@50": ci(0.95), "queries_with_full_recall@50": 300, "rows_returned": {"mean": 50.0, "min": 50, "queries_under_k": 0},
           "latency": {"n": 4, "p50_ms": 1.5, "p95_ms": 2.5}}
    names = {"float32": "h_idx", "halfvec": "h_half", "binary": "h_bit"}
    sizes = {"table_bytes_with_toast": 500 * 2**20, "indexes": {"h_idx": {"bytes": 230 * 2**20}, "h_half": {"bytes": 120 * 2**20}, "h_bit": {"bytes": 20 * 2**20}}}
    return {
        "schema": "pgvector-hnsw-v2", "query_selection": "full split", "git_sha": "abc1234", "k": 50, "stage_machines": {"index": MACHINE, "hnsw": MACHINE},
        "table": {"rows": 117000, "dim": 384, "embedder": "BAAI/bge-small-en-v1.5"}, "queries": {"n": 648},
        "index": {"m": 16, "ef_construction": 64, "reindex_seconds": 400.75, "bytes": 230 * 2**20}, "server": {"postgres": "PostgreSQL 16.15", "pgvector": "0.8.6"},
        "recall_definition": "R", "latency_definition": "L",
        "rows": [{"config": "exact", "plan_node": "Seq Scan", "recall@10": {"mean": 1.0}, "recall@50": {"mean": 1.0}, "latency": {"n": 4, "p50_ms": 150.5, "p95_ms": 170.5}},
                 {**row, "config": "hnsw ef_search=40", "ef_search": 40, "plan_node": "Index Scan using h_idx", "rows_returned": {"mean": 40.0, "min": 40, "queries_under_k": 648}},
                 {**row, "config": "served dense CTE", "settings": {"hnsw.ef_search": 100}, "plan_node": "Index Scan using h_idx"}],
        "variants": {"indexes": names, "sizes_before": {"table_bytes_with_toast": 500 * 2**20, "heap_bytes": 200 * 2**20, "rows": 117000},
                     "sizes_after": {**sizes, "table_bytes_with_toast": 900 * 2**20, "heap_bytes": 600 * 2**20, "rows": 117000,
                                     "avg_column_bytes": {"embedding": 1540.0, "embedding_half": 775.0, "embedding_bit": 53.0}},
                     "build_seconds": {"reindex_float32": 300.5, "reindex_halfvec": 200.5, "reindex_binary": 50.5}, "exact_unchanged": True,
                     "rows": [{**row, "variant": "float32", "ef_search": 100, "plan_node": "Index Scan using h_idx"}, {**row, "variant": "halfvec", "ef_search": 100, "plan_node": "Index Scan using h_half"},
                              {**row, "variant": "binary+rescore", "candidates": 400, "ef_search": 400, "plan_node": "Index Scan using h_bit"}]},
    }


def test_pgvector_block_shows_the_sweep_and_the_variants_on_one_table():
    text = report.pgvector_hnsw_table(hnsw_blob())
    assert "117000 chunk vectors, 384 dimensions (BAAI/bge-small-en-v1.5)" in text and "the 648 test queries" in text
    assert "the served index (m 16, ef_construction 64, 230.0 MiB) as a deployment builds it" in text
    # the REINDEX ran in the variants step, on the table the two copies had rewritten, not on the table the served rows read
    assert "a REINDEX of it took 400.75 s, in the variants step below, after the two copies had been added and the table vacuumed" in text
    assert "on the final table" not in text
    assert "| exact | — | — | 1.0000 | 1.0000 | — | 150.5 / 170.5 | Seq Scan |" in text
    assert "| hnsw ef_search=40 | 40 | 40.0 | 0.9700 [0.9500, 0.9900] | 0.9500 [0.9300, 0.9700] | 300 | 1.5 / 2.5 | Index Scan using h_idx |" in text
    # a scan that returns fewer than k rows cannot reach Recall@k = 1: the block says so, from the rows each scan returned
    assert "hnsw ef_search=40 returned 40.0 rows per query on average, so its Recall@50 is at most 0.8000" in text
    assert "served dense CTE returned" not in text
    assert "| served dense CTE | 100 | 50.0 |" in text and "GPU X (44.4 GB) with CPU Y" in text
    assert "float32 230.0 MiB, halfvec 120.0 MiB, binary 20.0 MiB" in text
    # the growth is mostly the row versions the two UPDATEs left, not the copies' own bytes (775 + 53 bytes x 117000 rows)
    assert "they average 775.0 and 53.0 bytes per row, about 92.4 MiB over 117000 rows" in text
    assert "grew from 500.0 MiB to 900.0 MiB (the heap from 200.0 MiB to 600.0 MiB)" in text and "most of that growth is not the copies' own size" in text
    assert "Both copies stay next to the float32 column, so the table grew" not in text
    assert "| binary+rescore | 400 | 400 | 50.0 |" in text and "| halfvec | 100 | — | 50.0 |" in text and "reads only the float32 column" in text
    assert "no schema migration adds either copy" in text and "ensure_halfvec" not in text


def test_pgvector_block_refuses_a_hash_dry_run_or_a_slice():
    dry = copy.deepcopy(hnsw_blob())
    dry["table"]["embedder"] = "hash-384"
    assert "not the benchmark" in report.pgvector_hnsw_table(dry) and "| exact |" not in report.pgvector_hnsw_table(dry)
    part = hnsw_blob()
    part["query_selection"] = "first 60 queries by id"
    assert "not the benchmark" in report.pgvector_hnsw_table(part)
    assert "is missing" in report.pgvector_hnsw_table(None)


def write_results(root, served=None, hnsw=None):
    import json

    results = root / "bench" / "results"
    results.mkdir(parents=True, exist_ok=True)
    if served is not None:
        (results / "served_scale_fiqa.json").write_text(json.dumps(served))
    if hnsw is not None:
        (results / "pgvector_hnsw.json").write_text(json.dumps(hnsw))


def test_evidence_rows_come_from_the_full_run_files(tmp_path):
    from app.eval import evidence

    write_results(tmp_path, served_blob(), hnsw_blob())
    rows = evidence._served_scale_claims(tmp_path)
    claims = {r["claim"]: r for r in rows}
    head = next(r for r in rows if r["claim"].startswith("BEIR FiQA (57638 documents) through the served API"))
    assert (head["value"], head["ci"], head["n"]) == ("0.3123", "[0.2923, 0.3323]", 4) and "ts_rank_cd (a fallback, not BM25)" in head["not_shown"]
    assert head["source"] == "bench/results/served_scale_fiqa.json @ abc1234" and head["command"] == "make bench-served-scale"
    pair = claims[f"FiQA: served `api:hybrid` minus in-memory `{HYBRID + SAME_FORM_SUFFIX}` (same query form, same job), paired delta in nDCG@10"]
    assert (pair["value"], pair["ci"]) == ("-0.0512", "[-0.0612, -0.0412]") and pair["not_shown"].startswith("the interval excludes zero")
    build = next(r for r in rows if r["claim"].startswith("FiQA served index build"))
    assert build["value"] == "12.25 + 1200.5" and build["n"] == 57598 and "maintained per insert" in build["claim"]
    load = next(r for r in rows if r["claim"].startswith("FiQA served API under load"))
    assert "concurrency 16" in load["claim"] and load["value"] == "16008.0 / 32008.0" and load["not_shown"].startswith("6 of 64 requests failed (http_503 6)")
    recall = next(r for r in rows if r["claim"].startswith("pgvector HNSW on the served FiQA table (117000 chunk vectors)"))
    assert (recall["value"], recall["ci"], recall["n"]) == ("0.9500", "[0.9300, 0.9700]", 648)
    half = next(r for r in rows if r["claim"].startswith("halfvec vs float32 HNSW on the same table at ef_search 100"))
    assert half["value"] == "0.9500 vs 0.9500 (120.0 vs 230.0)"


def test_evidence_has_no_row_from_a_slice_or_a_dry_run(tmp_path):
    from app.eval import evidence

    served, hnsw = served_blob(), hnsw_blob()
    served["query_selection"] = "first 40 queries by id"
    hnsw["table"]["embedder"] = "hash-384"
    write_results(tmp_path, served, hnsw)
    assert evidence._served_scale_claims(tmp_path) == []


def test_a_file_measured_before_the_schema_migrations_is_not_rendered(tmp_path):
    """served-scale-v1 / pgvector-hnsw-v1 files describe the served path before the vector table and its indexes became
    schema migrations (the lexical leg computed to_tsvector per row, start-up built the indexes)."""
    from app.eval import evidence

    old_served, old_hnsw = served_blob(), hnsw_blob()
    old_served["schema"], old_hnsw["schema"] = "served-scale-v1", "pgvector-hnsw-v1"
    for text in (report.served_scale_table(old_served), report.pgvector_hnsw_table(old_hnsw)):
        assert "before the served schema moved to migrations" in text and "not the benchmark" in text and "0.3123" not in text
    write_results(tmp_path, old_served, old_hnsw)
    assert evidence._served_scale_claims(tmp_path) == []


def test_a_reassembled_file_names_where_its_stages_were_measured():
    blob, hnsw = served_blob(), hnsw_blob()
    assert "results sha abc1234" in report.served_scale_table(blob)
    for b in (blob, hnsw):
        b["stage_git_sha"] = {"index": "def5678", "replay_hybrid": "def5678"}
    assert "stages measured at def5678, file assembled at abc1234" in report.served_scale_table(blob)
    assert "stages measured at def5678, file assembled at abc1234" in report.pgvector_hnsw_table(hnsw)
