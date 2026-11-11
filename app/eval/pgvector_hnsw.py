"""pgvector's HNSW index against the exact scan, on the served table at FiQA size with bge-small (C03),
and its half-precision and binary variants on the same table (C04).

Stages of the served-path scale run (app/eval/served_scale.py, `make bench-served-scale`): they run on the
table its index stage built (FiQA's chunks, the served embedder's vectors, the HNSW index the API's start-up
creates with its m and ef_construction) and use the served query embedding of every FiQA test query.
Results: bench/results/pgvector_hnsw.json.

  exact     ORDER BY embedding <=> q, node_id LIMIT k with index scans off: a sequential scan over every row
  hnsw      ORDER BY embedding <=> q LIMIT k with sequential scans off, at several hnsw.ef_search values and
            hnsw.iterative_scan off: the index alone, which returns at most ef_search rows
  served    the dense CTE of app/database/sql/hybrid_tsrank.sql with the settings hybrid_search sends
            (VectorStoreManager.settings_sql: ef_search twice the depth, iterative scan, read 2k keep k),
            planned freely: what the served statement reads

variants   (last: it rewrites the table) VectorStoreManager.ensure_halfvec and ensure_binary add a halfvec
           and a bit copy of the embedding column, each with an HNSW index; VACUUM (ANALYZE) then clears the
           old row versions the rewrite left in every index, and the three HNSW indexes are rebuilt (REINDEX,
           timed) so they are built the same way on the same rows. Then, on that one table: the exact scan
           again (it must find the same rows), float32 and halfvec at the same ef_search values, and binary
           candidates read from the bit index (hnsw.ef_search set to the candidate depth, so the index does
           not cut the list short) and rescored by float32 cosine. The served statement reads none of them.

Every configuration's plan is read once (EXPLAIN, same settings) and must use the scan it is named for, so an
"hnsw" row can never be the exact scan in disguise. Recall@k: a returned row counts when its float32 distance
is at most the exact k-th distance plus TIE_EPS, so rows tied at the cut count either way; Recall@10 is read
from the first 10 rows of the same k = 50 list. Latency: wall time of each statement (execute and fetch) on
one connection in this process, after an untimed pass of the exact scan and of the widest index scan; p50 and
p95 over the queries. Intervals: the percentile bootstrap over queries of app/eval/stats.
"""

from __future__ import annotations

import re
import time
from typing import Iterable, Sequence

from app.eval.latency import percentile
from app.eval.stats import mean_ci

K = 50
EF_SEARCH_VALUES = (40, 64, 100, 200, 400, 1000)  # 40 is pgvector's default; 1000 its maximum
TIE_EPS = 1e-6
_SCAN = re.compile(r"(Parallel Seq Scan|Seq Scan|Index Scan|Index Only Scan|Bitmap Heap Scan)(?: using (\S+))? on (\S+)")


def literal(vector: Iterable[float]) -> str:
    return "[" + ",".join(f"{float(x):.8g}" for x in vector) + "]"


def exact_sql(table: str) -> str:
    return f"SELECT node_id, embedding <=> %(qvec)s::vector AS distance FROM {table} ORDER BY embedding <=> %(qvec)s::vector, node_id LIMIT %(k)s"


def float32_sql(table: str) -> str:
    """One ordering key, so pgvector's HNSW index can answer it."""
    return f"SELECT node_id, embedding <=> %(qvec)s::vector AS distance FROM {table} ORDER BY embedding <=> %(qvec)s::vector LIMIT %(k)s"


def halfvec_sql(table: str) -> str:
    """Ordered by the halfvec column's distance (its HNSW index), reporting each row's float32 distance."""
    return f"SELECT node_id, embedding <=> %(qvec)s::vector AS distance FROM {table} ORDER BY embedding_half <=> %(qvec)s::halfvec LIMIT %(k)s"


def binary_rescore_sql(table: str) -> str:
    """%(candidates)s rows by Hamming distance from the bit index, then the k nearest of them by float32 cosine."""
    return (f"SELECT node_id, embedding <=> %(qvec)s::vector AS distance FROM (SELECT node_id, embedding FROM {table} "
            f"ORDER BY embedding_bit <~> binary_quantize(%(qvec)s::vector) LIMIT %(candidates)s) c "
            f"ORDER BY embedding <=> %(qvec)s::vector, node_id LIMIT %(k)s")


def served_dense_sql(vsm) -> str:
    """The served statement's dense CTE, verbatim (as tests/database/test_dense_parity.py reads it), returning distances."""
    sql = vsm.hybrid_sql("ts_rank_cd")
    head, sep, _ = sql.partition(",\nlexical AS (")
    if not sep:
        raise RuntimeError("hybrid_tsrank.sql no longer has a lexical CTE after the dense CTE")
    return head + "\nSELECT d.node_id, 1 - d.dense_score AS distance FROM dense d ORDER BY d.dense_rank;"


def scans(plan: str) -> list[dict]:
    return [{"node": m.group(1), "index": m.group(2), "relation": m.group(3)} for m in _SCAN.finditer(plan)]


def check_plan(plan: str, *, expect_index: str | None, table: str) -> str:
    """The table's scan node; raises unless it is a scan of `expect_index` (or, with None, a sequential scan)."""
    on_table = [s for s in scans(plan) if s["relation"] == table]
    if not on_table:
        raise RuntimeError(f"no scan of {table} in the plan:\n{plan}")
    node = on_table[0]
    if expect_index is None:
        if "Seq Scan" not in node["node"]:
            raise RuntimeError(f"expected a sequential scan of {table}, the plan has {node}:\n{plan}")
        return node["node"]
    if node["node"] != "Index Scan" or node["index"] != expect_index:
        raise RuntimeError(f"expected an index scan using {expect_index}, the plan has {node}:\n{plan}")
    return f"Index Scan using {expect_index}"


def recall_at(returned: Sequence[tuple[str, float]], exact: Sequence[tuple[str, float]], k: int, eps: float = TIE_EPS) -> float:
    """Share of the k exact neighbours a returned list holds, rows at the exact k-th distance counting as neighbours."""
    if len(exact) < k:
        raise ValueError(f"the exact list has {len(exact)} rows, fewer than k = {k}")
    cut = float(exact[k - 1][1]) + eps
    return sum(1 for _, distance in returned[:k] if distance is not None and float(distance) <= cut) / k


def _ci(values: Sequence[float], idx) -> dict:
    mean, lo, hi = mean_ci(values, idx)
    return {"mean": round(mean, 4), "lo": round(lo, 4), "hi": round(hi, 4)}


def latency(values: Sequence[float]) -> dict:
    v = list(values)
    return {"n": len(v), "p50_ms": round(percentile(v, 50), 3), "p95_ms": round(percentile(v, 95), 3), "mean_ms": round(sum(v) / len(v), 3)} if v else {"n": 0}


def summarize(name: str, returned: Sequence[Sequence[tuple[str, float]]], exact: Sequence[Sequence[tuple[str, float]]], times_ms: Sequence[float], idx, **settings) -> dict:
    r10 = [recall_at(r, e, 10) for r, e in zip(returned, exact)]
    r50 = [recall_at(r, e, K) for r, e in zip(returned, exact)]
    counts = [len(r) for r in returned]
    return {
        "config": name, **settings,
        "recall@10": _ci(r10, idx), "recall@50": _ci(r50, idx),
        "queries_with_full_recall@50": sum(1 for x in r50 if x >= 1.0),
        "rows_returned": {"mean": round(sum(counts) / len(counts), 2), "min": min(counts), "queries_under_k": sum(1 for c in counts if c < K)},
        "latency": latency(times_ms),
    }


class Session:
    """One dedicated connection (autocommit) to the served table, with session settings for a configuration."""

    def __init__(self, vsm):
        import psycopg2

        self.vsm = vsm
        self.table = f'"{vsm.schema_name}"."{vsm.data_table}"'
        self.conn = psycopg2.connect(vsm.conn_string)
        self.conn.autocommit = True

    def configure(self, settings: dict) -> None:
        """RESET everything this module sets, then SET the configuration's values for the session."""
        with self.conn.cursor() as cur:
            for name in ("hnsw.ef_search", "hnsw.iterative_scan", "enable_seqscan", "enable_indexscan", "enable_bitmapscan"):
                cur.execute(f"RESET {name};")
            for name, value in settings.items():
                cur.execute("SELECT set_config(%s, %s, false);", (name, str(value)))

    def explain(self, statement: str, params: dict) -> str:
        with self.conn.cursor() as cur:
            cur.execute("EXPLAIN " + statement, params)
            return "\n".join(r[0] for r in cur.fetchall())

    def run(self, statement: str, vectors, k: int = K, extra: dict | None = None) -> tuple[list[list[tuple[str, float]]], list[float]]:
        rows, times = [], []
        with self.conn.cursor() as cur:
            for vec in vectors:
                params = {"qvec": literal(vec), "k": int(k), **(extra or {})}
                started = time.perf_counter()
                cur.execute(statement, params)
                fetched = cur.fetchall()
                times.append((time.perf_counter() - started) * 1000.0)
                rows.append([(r[0], float(r[1]) if r[1] is not None else None) for r in fetched])
        return rows, times

    def close(self) -> None:
        self.conn.close()


def served_run(vsm, vectors, k: int = K) -> tuple[list[list[tuple[str, float]]], list[float], str]:
    """The dense CTE as hybrid_search sends it: its settings in the same execute, one transaction per query."""
    statement = served_dense_sql(vsm)
    settings, settings_params = vsm.settings_sql(k)
    rows, times = [], []
    plan = None
    with vsm.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(settings + "EXPLAIN " + statement, {**settings_params, "qvec": literal(vectors[0]), "query": "x", "k": int(k), "rrf_k": 60.0})
            plan = "\n".join(r[0] for r in cur.fetchall())
        conn.commit()
        for vec in vectors:
            params = {**settings_params, "qvec": literal(vec), "query": "x", "k": int(k), "rrf_k": 60.0}
            with conn.cursor() as cur:
                started = time.perf_counter()
                cur.execute(settings + statement, params)
                fetched = cur.fetchall()
                times.append((time.perf_counter() - started) * 1000.0)
            conn.commit()  # the settings end with the transaction, as they do for the served statement
            rows.append([(r[0], float(r[1])) for r in fetched])
    return rows, times, plan


def hnsw_sweep(vsm, vectors, idx, *, hnsw_index: str, ef_values: Sequence[int] = EF_SEARCH_VALUES, k: int = K) -> dict:
    """The exact scan, the index at each ef_search, and the served CTE, on the same queries (see the module docstring)."""
    session = Session(vsm)
    table = session.table
    plain_table = vsm.data_table
    try:
        exact_stmt, ann_stmt = exact_sql(table), float32_sql(table)
        probe = {"qvec": literal(vectors[0]), "k": k}
        session.configure({"enable_indexscan": "off", "enable_bitmapscan": "off"})
        exact_node = check_plan(session.explain(exact_stmt, probe), expect_index=None, table=plain_table)
        session.run(exact_stmt, vectors, k)  # untimed: the heap into the page cache
        exact, exact_times = session.run(exact_stmt, vectors, k)
        session.configure({"enable_seqscan": "off", "hnsw.ef_search": max(ef_values), "hnsw.iterative_scan": "off"})
        session.run(ann_stmt, vectors, k)  # untimed: the index into the page cache
        rows = [{"config": "exact", "plan_node": exact_node, "settings": {"enable_indexscan": "off", "enable_bitmapscan": "off"}, "latency": latency(exact_times),
                 "recall@10": {"mean": 1.0}, "recall@50": {"mean": 1.0}}]
        for ef in ef_values:
            settings = {"enable_seqscan": "off", "hnsw.ef_search": ef, "hnsw.iterative_scan": "off"}
            session.configure(settings)
            node = check_plan(session.explain(ann_stmt, probe), expect_index=hnsw_index, table=plain_table)
            returned, times = session.run(ann_stmt, vectors, k)
            rows.append({**summarize(f"hnsw ef_search={ef}", returned, exact, times, idx, ef_search=ef, iterative_scan="off"), "plan_node": node, "settings": settings})
    finally:
        session.close()
    returned, times, plan = served_run(vsm, vectors, k)
    served_scans = [s for s in scans(plan) if s["relation"] == plain_table]
    node = f"{served_scans[0]['node']}" + (f" using {served_scans[0]['index']}" if served_scans and served_scans[0]["index"] else "") if served_scans else None
    rows.append({**summarize("served dense CTE", returned, exact, times, idx, **{"settings": vsm.search_settings(k)}), "plan_node": node, "forced": False,
                 "reads": f"{2 * k} rows from the scan, keeps {k} (app/database/sql/hybrid_tsrank.sql)"})
    return {"k": k, "tie_eps": TIE_EPS, "rows": rows, "exact": exact}


VARIANT_EF_VALUES = (100, 1000)  # the served setting for k = 50 (2k) and pgvector's maximum
CANDIDATES = (50, 100, 200, 400, 1000)  # binary candidates per query before the float32 rescoring


def variants(vsm, vectors, exact, idx, *, ef_values: Sequence[int] = VARIANT_EF_VALUES, candidates: Sequence[int] = CANDIDATES, k: int = K) -> dict:
    """float32, halfvec and binary + rescore on one table state (see the module docstring); `exact` from hnsw_sweep."""
    from app.eval.served_scale import table_sizes

    names = {"float32": f"{vsm.table_name}_embedding_hnsw", "halfvec": f"{vsm.table_name}_embedding_half_hnsw", "binary": f"{vsm.table_name}_embedding_bit_hnsw"}
    seconds: dict[str, float] = {}
    sizes_before = table_sizes(vsm)
    for name, ensure in (("ensure_halfvec", vsm.ensure_halfvec), ("ensure_binary", vsm.ensure_binary)):
        started = time.perf_counter()
        if not ensure():  # the helpers log a failure and return False
            raise RuntimeError(f"{name} failed; see the log")
        seconds[name] = round(time.perf_counter() - started, 3)
    session = Session(vsm)
    try:
        with session.conn.cursor() as cur:
            started = time.perf_counter()
            cur.execute(f"VACUUM (ANALYZE) {session.table};")
            seconds["vacuum_analyze"] = round(time.perf_counter() - started, 3)
            for variant, index in names.items():
                started = time.perf_counter()
                cur.execute(f'REINDEX INDEX "{vsm.schema_name}"."{index}";')
                seconds[f"reindex_{variant}"] = round(time.perf_counter() - started, 3)
        sizes_after = table_sizes(vsm)
        table, plain = session.table, vsm.data_table
        probe = {"qvec": literal(vectors[0]), "k": k}
        session.configure({"enable_indexscan": "off", "enable_bitmapscan": "off"})
        check_plan(session.explain(exact_sql(table), probe), expect_index=None, table=plain)
        again, _ = session.run(exact_sql(table), vectors, k)  # also the untimed pass over the rewritten heap
        unchanged = [[n for n, _ in a] for a in again] == [[n for n, _ in e] for e in exact]
        rows = []
        for variant, statement in (("float32", float32_sql(table)), ("halfvec", halfvec_sql(table))):
            for ef in ef_values:
                settings = {"enable_seqscan": "off", "hnsw.ef_search": ef, "hnsw.iterative_scan": "off"}
                session.configure(settings)
                node = check_plan(session.explain(statement, probe), expect_index=names[variant], table=plain)
                session.run(statement, vectors[: min(20, len(vectors))], k)  # untimed: this index into the page cache
                returned, times = session.run(statement, vectors, k)
                rows.append({**summarize(f"{variant} ef_search={ef}", returned, exact, times, idx, variant=variant, ef_search=ef, iterative_scan="off"), "plan_node": node, "settings": settings})
        statement = binary_rescore_sql(table)
        for depth in candidates:
            ef = min(1000, max(int(depth), 40))
            settings = {"enable_seqscan": "off", "hnsw.ef_search": ef, "hnsw.iterative_scan": "off"}
            session.configure(settings)
            node = check_plan(session.explain(statement, {**probe, "candidates": int(depth)}), expect_index=names["binary"], table=plain)
            session.run(statement, vectors[: min(20, len(vectors))], k, extra={"candidates": int(depth)})
            returned, times = session.run(statement, vectors, k, extra={"candidates": int(depth)})
            rows.append({**summarize(f"binary+rescore candidates={depth}", returned, exact, times, idx, variant="binary+rescore", candidates=int(depth), ef_search=ef, iterative_scan="off"),
                         "plan_node": node, "settings": settings})
    finally:
        session.close()
    return {"k": k, "indexes": names, "build_seconds": seconds, "sizes_before": sizes_before, "sizes_after": sizes_after, "exact_unchanged": unchanged, "rows": rows}
