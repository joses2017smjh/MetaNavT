"""BEIR qrels scored through the served API (POST /api/retrieve/), at the document level.

The served path ranks chunks: a document longer than one chunk can come back as
several hits. A BEIR qrel names a document, so a hit counts as its document, and a
document's rank is the rank of its best-ranked chunk (later chunks of the same
document are dropped from the ranking, not counted twice). The API returns at most
MAX_K (50) chunks, so a served ranking holds at most 50 documents and usually
fewer: its Recall@50 is "within the 50 returned chunks", and every row records how
many distinct documents the chunks held.

A document is identified by its file: the corpus is written one file per document
(write_corpus, `<doc_id>.txt`), so a hit's `path` names its document. Both sides
of a comparison are scored by score_ranking, with BEIR's convention of dropping a
retrieved document whose id equals the query id (app/eval/beir.py run_config).

A request that fails (no response, a non-200 status, a body without hits) keeps
its place in the per-query list with an empty ranking: it scores zero, stays in
every interval, and is counted with its id and error. Nothing is retried.
"""

from __future__ import annotations

import hashlib
import json
import re
import socket
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Mapping, Sequence

from app.eval.beir import graded_ndcg_at_k
from app.eval.latency import percentile
from app.eval.metrics import mrr_at_k, recall_at_k
from app.eval.provenance import scrub
from app.eval.stats import mean_ci, paired_delta_ci, resample_indices

DOC_SUFFIX = ".txt"  # the served chunker gives .txt its fixed window (app/chunking/structure.py chunk_text)
CI_METRICS = ("ndcg@10", "recall@10", "recall@50")
METRICS = (*CI_METRICS, "mrr@10")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

PostFn = Callable[[str, int], tuple]


# ---------------------------------------------------------------- corpus as files


def doc_file_name(doc_id: str) -> str:
    """`<doc_id>.txt`; an id that is not a plain file name is refused rather than escaped."""
    if not _SAFE_ID.fullmatch(doc_id) or doc_id in {".", ".."}:
        raise ValueError(f"document id {doc_id!r} is not a plain file name")
    return doc_id + DOC_SUFFIX


def doc_id_from_path(path: str | None) -> str | None:
    """The document a served hit belongs to: its file name without DOC_SUFFIX (None for anything else)."""
    if not path:
        return None
    name = PurePosixPath(str(path).replace("\\", "/")).name
    return name[: -len(DOC_SUFFIX)] if name.endswith(DOC_SUFFIX) and len(name) > len(DOC_SUFFIX) else None


def write_corpus(corpus: Mapping[str, str], directory: Path) -> dict:
    """One UTF-8 file per document, its text exactly as the in-memory rows read it; nothing filtered here.

    Empty and duplicate documents are written too: the served scanner rejects them and counts them,
    which is part of what the run measures. Returns counts and a digest of (file name, text) pairs.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    listing = hashlib.sha256()
    n_bytes = 0
    for doc_id in sorted(corpus):
        name = doc_file_name(doc_id)
        data = corpus[doc_id].encode("utf-8")
        (directory / name).write_bytes(data)
        n_bytes += len(data)
        listing.update(json.dumps([name, hashlib.sha256(data).hexdigest()]).encode())
    return {"n_files": len(corpus), "bytes": n_bytes, "listing_sha256": listing.hexdigest()}


# ---------------------------------------------------------------- the API


def post_retrieve(url: str, query: str, k: int, timeout: float = 120.0) -> tuple[int | None, dict | None, str | None]:
    """(status, body, error) for one POST /api/retrieve/; never raises for an HTTP or network failure."""
    req = urllib.request.Request(url.rstrip("/") + "/api/retrieve/", data=json.dumps({"query": query, "k": k}).encode(), headers={"content-type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode()), None
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode()).get("detail")
        except (ValueError, OSError, AttributeError):
            detail = None
        return exc.code, None, f"http_{exc.code}" + (f": {str(detail)[:300]}" if detail else "")
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as exc:
        return None, None, f"{type(exc).__name__}: {str(exc)[:300]}"


def doc_ranking(hits: Sequence[Mapping]) -> tuple[list[str], int]:
    """(document ids by their best-ranked chunk, hits that named no document)."""
    ranking: list[str] = []
    seen: set[str] = set()
    unmapped = 0
    for hit in hits:
        doc = doc_id_from_path(hit.get("path"))
        if doc is None:
            unmapped += 1
        elif doc not in seen:
            seen.add(doc)
            ranking.append(doc)
    return ranking, unmapped


def score_ranking(ranking: Sequence[str], relevant: Mapping[str, int], query_id: str) -> dict:
    """nDCG@10 (graded, linear gain), Recall@10/@50 and MRR@10 of one document ranking, query id removed."""
    ranked = [d for d in ranking if d != query_id]
    positive = [d for d, grade in relevant.items() if grade > 0]
    return {
        "ndcg@10": graded_ndcg_at_k(ranked, dict(relevant), 10),
        "recall@10": recall_at_k(ranked, positive, 10),
        "recall@50": recall_at_k(ranked, positive, 50),
        "mrr@10": mrr_at_k(ranked, positive, 10),
    }


def replay(post: PostFn, queries: Mapping[str, str], qrels: Mapping[str, Mapping[str, int]], k: int = 50, progress: Callable[[int, dict], None] | None = None) -> list[dict]:
    """Every query once, in the order given, through `post(query, k)`; one scored row per query."""
    rows = []
    for i, (qid, text) in enumerate(queries.items()):
        started = time.perf_counter()
        status, body, error = post(text, k)
        e2e_ms = (time.perf_counter() - started) * 1000.0
        row: dict = {"id": qid, "status": status, "error": error, "e2e_ms": round(e2e_ms, 3)}
        hits = (body or {}).get("hits") if status == 200 else None
        if status == 200 and not isinstance(hits, list):
            row["error"] = "invalid_response: no hits list"
            hits = None
        ranking, unmapped = doc_ranking(hits or [])
        row.update(
            ok=hits is not None,
            route=(body or {}).get("route"),
            counts=(body or {}).get("counts"),
            staleness=(body or {}).get("staleness"),
            degraded=(body or {}).get("degraded") or [],
            latency_ms=(body or {}).get("latency_ms") or {},
            n_hits=len(hits or []),
            n_docs=len(ranking),
            unmapped_hits=unmapped,
            ranking=ranking,
            **score_ranking(ranking, qrels[qid], qid),
        )
        rows.append(row)
        if progress:
            progress(i, row)
    return rows


# ---------------------------------------------------------------- summaries


def distribution(values: Iterable[float]) -> dict:
    v = list(values)
    if not v:
        return {"n": 0}
    return {"n": len(v), "p50_ms": round(percentile(v, 50), 2), "p95_ms": round(percentile(v, 95), 2), "mean_ms": round(sum(v) / len(v), 2), "max_ms": round(max(v), 2)}


def _ci_block(values: Sequence[float], idx) -> dict:
    mean, lo, hi = mean_ci(values, idx)
    return {"mean": round(mean, 4), "lo": round(lo, 4), "hi": round(hi, 4)}


def summarize(rows: Sequence[Mapping], idx) -> dict:
    """Means with bootstrap intervals (CI_METRICS), MRR@10's mean, and what the run did per query."""
    n = len(rows)
    out: dict = {
        "n_queries": n,
        "retrieval": {m: round(sum(r[m] for r in rows) / n, 4) for m in METRICS},
        "ci": {m: _ci_block([r[m] for r in rows], idx) for m in CI_METRICS},
    }
    failed = [r for r in rows if not r.get("ok", True)]
    if any("status" in r for r in rows):
        docs = [r["n_docs"] for r in rows if r.get("ok", True)]
        out["requests"] = {
            "failed": len(failed),
            "failed_ids": [r["id"] for r in failed],
            "errors": dict(Counter(str(r.get("error")) for r in failed)),
            "status_counts": dict(Counter(str(r.get("status")) for r in rows)),
            "degraded": sum(1 for r in rows if r.get("degraded")),
            "degraded_components": dict(Counter(d.get("component", "?") for r in rows for d in r.get("degraded") or [])),
            "zero_hit_ids": [r["id"] for r in rows if r.get("ok", True) and not r.get("n_hits")],
            "unmapped_hits": sum(r.get("unmapped_hits", 0) for r in rows),
        }
        out["documents_per_response"] = {
            "mean": round(sum(docs) / len(docs), 2) if docs else None,
            "min": min(docs) if docs else None,
            "queries_under_10": sum(1 for d in docs if d < 10),
            "queries_under_50": sum(1 for d in docs if d < 50),
        }
        out["routes"] = dict(Counter(str(r.get("route")) for r in rows if r.get("ok", True)))
        out["latency"] = {"e2e": distribution(r["e2e_ms"] for r in rows if r.get("ok", True))}
        stages = sorted({s for r in rows if r.get("ok", True) for s in (r.get("latency_ms") or {})})
        out["latency"]["server_stages"] = {s: distribution(float(r["latency_ms"][s]) for r in rows if r.get("ok", True) and s in (r.get("latency_ms") or {})) for s in stages}
    return out


def paired(a: Sequence[Mapping], b: Sequence[Mapping], idx) -> dict:
    """mean(a) - mean(b) per CI metric with the paired bootstrap interval; rows must be in the same query order."""
    if [r["id"] for r in a] != [r["id"] for r in b]:
        raise ValueError("paired rows must cover the same queries in the same order")
    out = {}
    for m in CI_METRICS:
        delta, lo, hi = paired_delta_ci([r[m] for r in a], [r[m] for r in b], idx)
        out[m] = {"delta": round(delta, 4), "lo": round(lo, 4), "hi": round(hi, 4), "covers_zero": lo <= 0.0 <= hi}
    return out


def by_route(served: Sequence[Mapping], reference: Sequence[Mapping], *, seed: int = 0, n_boot: int = 2000) -> dict:
    """The paired comparison restricted to each served route (a failed request has no route: 'failed')."""
    out = {}
    routes = sorted({str(r.get("route")) if r.get("ok", True) else "failed" for r in served})
    for route in routes:
        keep = [i for i, r in enumerate(served) if (str(r.get("route")) if r.get("ok", True) else "failed") == route]
        sub_a, sub_b = [served[i] for i in keep], [reference[i] for i in keep]
        idx = resample_indices(len(keep), n_boot=n_boot, seed=seed)
        out[route] = {"n": len(keep), "served": {m: round(sum(r[m] for r in sub_a) / len(keep), 4) for m in CI_METRICS},
                      "reference": {m: round(sum(r[m] for r in sub_b) / len(keep), 4) for m in CI_METRICS}, "delta": paired(sub_a, sub_b, idx)}
    return out


def public_rows(rows: Sequence[Mapping]) -> list[dict]:
    """Per-query rows as a results file keeps them: no ranking (the TREC run holds it), errors scrubbed."""
    keep = ("id", "status", "error", "route", "n_hits", "n_docs", "e2e_ms", *METRICS)
    return [scrub({k: r.get(k) for k in keep if k in r}) for r in rows]
