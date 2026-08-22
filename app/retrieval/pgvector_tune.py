"""pgvector tuning: halfvec, binary quantization + rescore, HNSW m / ef_search.

In-memory analogues produce the recall-vs-latency curve without a live Postgres.
SQL helpers on VectorStoreManager apply the same knobs when pgvector is present.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from app.retrieval.embedders import l2_normalize


@dataclass
class SweepPoint:
    ef_search: int
    recall_10: float
    latency_p50_ms: float
    latency_p95_ms: float
    bytes_per_vec: int
    mode: str

    def as_dict(self) -> dict:
        return {
            "ef_search": self.ef_search,
            "recall@10": round(self.recall_10, 4),
            "p50_ms": round(self.latency_p50_ms, 3),
            "p95_ms": round(self.latency_p95_ms, 3),
            "bytes_per_vec": self.bytes_per_vec,
            "mode": self.mode,
        }


def to_halfvec(mat: np.ndarray) -> np.ndarray:
    """Simulate pgvector halfvec: store f16, compute in f32."""
    return mat.astype(np.float16).astype(np.float32)


def to_binary(mat: np.ndarray) -> np.ndarray:
    """Sign-bit packing analogue of binary quantization (1 bit/dim)."""
    return (l2_normalize(mat) >= 0).astype(np.uint8)


def hamming_search(query_bits: np.ndarray, doc_bits: np.ndarray, k: int) -> np.ndarray:
    # popcount of xor via sum of mismatches
    dists = np.abs(doc_bits.astype(np.int16) - query_bits.reshape(1, -1)).sum(axis=1)
    return np.argsort(dists)[:k]


def rescore_cosine(query: np.ndarray, docs: np.ndarray, cand_idx: np.ndarray, k: int) -> np.ndarray:
    q = l2_normalize(query.reshape(1, -1))
    subset = l2_normalize(docs[cand_idx])
    scores = (subset @ q.T).ravel()
    order = np.argsort(-scores)[:k]
    return cand_idx[order]


def brute_knn(query: np.ndarray, docs: np.ndarray, k: int) -> np.ndarray:
    q = l2_normalize(query.reshape(1, -1))
    d = l2_normalize(docs)
    scores = (d @ q.T).ravel()
    return np.argsort(-scores)[:k]


def build_nsw(docs: np.ndarray, m: int = 16, seed: int = 0) -> list[np.ndarray]:
    """Undirected NSW: each node links to its m nearest (brute, fine for personal corpora)."""
    n = docs.shape[0]
    m = max(1, min(m, n - 1))
    d = l2_normalize(docs)
    sims = d @ d.T
    np.fill_diagonal(sims, -np.inf)
    graph = []
    for i in range(n):
        graph.append(np.argpartition(-sims[i], m)[:m].astype(np.int32))
    return graph


def hnsw_search(
    query: np.ndarray,
    docs: np.ndarray,
    graph: list[np.ndarray],
    *,
    ef_search: int = 40,
    k: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """Best-first NSW search. `ef_search` is the candidate-list size (same knob as pgvector)."""
    import heapq

    n = docs.shape[0]
    if n == 0:
        return np.array([], dtype=np.int32)
    ef = max(k, min(int(ef_search), n))
    q = l2_normalize(query.reshape(1, -1)).ravel()
    d = l2_normalize(docs)
    rng = np.random.RandomState(seed)

    def score(i: int) -> float:
        return float(d[int(i)] @ q)

    entries = [int(rng.randint(0, n)) for _ in range(min(3, n))]
    starts = []
    for start in entries:
        cur = start
        cur_s = score(cur)
        improved = True
        seen_walk = {cur}
        while improved:
            improved = False
            for nb in graph[cur]:
                nb = int(nb)
                if nb in seen_walk:
                    continue
                s = score(nb)
                if s > cur_s:
                    cur, cur_s = nb, s
                    seen_walk.add(nb)
                    improved = True
        starts.append(cur)

    visited: set[int] = set()
    candidates: list[tuple[float, int]] = []
    w: list[tuple[float, int]] = []
    for s0 in starts:
        if s0 in visited:
            continue
        visited.add(s0)
        s = score(s0)
        heapq.heappush(candidates, (-s, s0))
        heapq.heappush(w, (s, s0))
    while w and len(w) > ef:
        heapq.heappop(w)

    while candidates:
        neg_s, idx = heapq.heappop(candidates)
        if w and -neg_s < w[0][0] and len(w) >= ef:
            continue
        for nb in graph[idx]:
            nb = int(nb)
            if nb in visited:
                continue
            visited.add(nb)
            s = score(nb)
            if len(w) < ef or s > w[0][0]:
                heapq.heappush(candidates, (-s, nb))
                heapq.heappush(w, (s, nb))
                if len(w) > ef:
                    heapq.heappop(w)

    ranked = sorted(w, key=lambda item: item[0], reverse=True)
    ids = [i for _, i in ranked[:k]]
    if len(ids) < k:
        extra = brute_knn(query, docs, k)
        for i in extra:
            if int(i) not in ids:
                ids.append(int(i))
            if len(ids) >= k:
                break
    return np.array(ids[:k], dtype=np.int32)


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((p / 100.0) * (len(ordered) - 1)))
    return ordered[idx]


def sweep_ef_search(
    docs: np.ndarray,
    queries: np.ndarray,
    *,
    k: int = 10,
    m: int = 16,
    efs: tuple[int, ...] = (10, 20, 40, 80, 160),
) -> list[SweepPoint]:
    gold = [brute_knn(queries[i], docs, k) for i in range(len(queries))]
    graph = build_nsw(docs, m=m)
    dim = docs.shape[1]
    if len(queries) and efs:
        hnsw_search(queries[0], docs, graph, ef_search=efs[0], k=k, seed=0)
    points = []
    for ef in efs:
        recs = []
        times = []
        for i in range(len(queries)):
            t0 = time.perf_counter()
            hit = hnsw_search(queries[i], docs, graph, ef_search=ef, k=k, seed=i)
            times.append((time.perf_counter() - t0) * 1000.0)
            recs.append(len(set(hit.tolist()) & set(gold[i].tolist())) / float(k))
        points.append(
            SweepPoint(
                ef_search=ef,
                recall_10=float(np.mean(recs)),
                latency_p50_ms=_pct(times, 50),
                latency_p95_ms=_pct(times, 95),
                bytes_per_vec=dim * 4,
                mode="float32+hnsw",
            )
        )
    return points


def compare_storage(
    docs: np.ndarray,
    queries: np.ndarray,
    *,
    k: int = 10,
    binary_oversample: int = 4,
) -> list[SweepPoint]:
    """float32 vs halfvec vs binary+rescore. Reports recall@10 and bytes/vec."""
    gold = [brute_knn(queries[i], docs, k) for i in range(len(queries))]
    dim = docs.shape[1]
    out: list[SweepPoint] = []

    def measure(mode: str, search_fn, bytes_per: int) -> SweepPoint:
        recs, times = [], []
        for i in range(len(queries)):
            t0 = time.perf_counter()
            hit = search_fn(i)
            times.append((time.perf_counter() - t0) * 1000.0)
            recs.append(len(set(hit.tolist()) & set(gold[i].tolist())) / float(k))
        return SweepPoint(
            ef_search=0,
            recall_10=float(np.mean(recs)),
            latency_p50_ms=_pct(times, 50),
            latency_p95_ms=_pct(times, 95),
            bytes_per_vec=bytes_per,
            mode=mode,
        )

    half = to_halfvec(docs)
    bits = to_binary(docs)

    out.append(measure("float32", lambda i: brute_knn(queries[i], docs, k), dim * 4))
    out.append(measure("halfvec", lambda i: brute_knn(queries[i], half, k), dim * 2))

    def binary_then_rescore(i: int) -> np.ndarray:
        qbits = to_binary(queries[i].reshape(1, -1))[0]
        cand = hamming_search(qbits, bits, k=min(docs.shape[0], k * binary_oversample))
        return rescore_cosine(queries[i], docs, cand, k)

    out.append(measure("binary+rescore", binary_then_rescore, max(1, dim // 8)))
    return out


HNSW_SQL = """
-- pgvector HNSW (cosine). Tune m / ef_construction at build; ef_search at query.
CREATE INDEX IF NOT EXISTS {table}_embedding_hnsw
ON {schema}.{table}
USING hnsw (embedding vector_cosine_ops)
WITH (m = {m}, ef_construction = {ef_construction});

-- Query-time beam. Sweep this; plot recall against latency.
SET hnsw.ef_search = {ef_search};

-- Filtered queries: don't silently under-return.
SET hnsw.iterative_scan = relaxed_order;
"""

HALFVEC_SQL = """
ALTER TABLE {schema}.{table}
    ADD COLUMN IF NOT EXISTS embedding_half halfvec({dim});
UPDATE {schema}.{table}
    SET embedding_half = embedding::halfvec
    WHERE embedding_half IS NULL;
CREATE INDEX IF NOT EXISTS {table}_embedding_half_hnsw
ON {schema}.{table}
USING hnsw (embedding_half halfvec_cosine_ops)
WITH (m = {m}, ef_construction = {ef_construction});
"""

BINARY_SQL = """
-- Binary quantization + rescoring pass (pgvector 0.7+).
ALTER TABLE {schema}.{table}
    ADD COLUMN IF NOT EXISTS embedding_bit bit({dim});
UPDATE {schema}.{table}
    SET embedding_bit = binary_quantize(embedding)::bit({dim})
    WHERE embedding_bit IS NULL;
CREATE INDEX IF NOT EXISTS {table}_embedding_bit_hnsw
ON {schema}.{table}
USING hnsw (embedding_bit bit_hamming_ops);
-- Retrieve wide on bits, rescore with <=> on the float column.
"""
