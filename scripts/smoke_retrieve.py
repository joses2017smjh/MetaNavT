"""POST one query to a running API and require hits with non-null paths.

    python scripts/smoke_retrieve.py [--url http://localhost:8000] [--query "..."] [--require-bm25]

Waits for GET /health to return 200 (up to --timeout seconds), then POSTs
/api/retrieve/ and exits 1 unless the response has at least one hit and every
hit carries a path. Standard library only, so it runs on the CI runner itself.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def _get(url: str):
    with urllib.request.urlopen(url, timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode())


POST_TIMEOUT = 60.0  # raised by --post-timeout; a CPU reranker can take longer than a minute per query


def _post(url: str, payload: dict):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"content-type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=POST_TIMEOUT) as resp:
        return resp.status, json.loads(resp.read().decode())


def wait_for_health(base: str, timeout: float) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            status, body = _get(f"{base}/health")
            if status == 200:
                return body
            last = body
        except urllib.error.HTTPError as exc:
            last = exc.read().decode()[:300]
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            last = str(exc)
        time.sleep(3)
    raise SystemExit(f"API not healthy after {timeout:.0f}s; last: {last}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--query", default="current learning rate for run 47")
    p.add_argument("--timeout", type=float, default=180, help="seconds to wait for /health")
    p.add_argument("--require-bm25", action="store_true", help="also require counts.bm25 > 0")
    p.add_argument("--require-reranker", action="store_true", help="require /health.reranker.loaded and no degraded components")
    p.add_argument("--latency", default=None, metavar="GOLD_JSONL", help="replay these questions (k=8) and report p50/p95 of server total and e2e ms")
    p.add_argument("--latency-out", default=None, help="write the latency summary (JSON) here")
    p.add_argument("--post-timeout", type=float, default=60.0, help="seconds to wait for one POST (raise for CPU rerankers)")
    args = p.parse_args()
    global POST_TIMEOUT
    POST_TIMEOUT = args.post_timeout

    health = wait_for_health(args.url.rstrip("/"), args.timeout)
    print("health:", json.dumps(health))

    status, body = _post(f"{args.url.rstrip('/')}/api/retrieve/", {"query": args.query})
    hits = body.get("hits", [])
    print(f"POST /api/retrieve/ -> {status}; route={body.get('route')} counts={body.get('counts')} "
          f"bm25_backend={body.get('bm25_backend')} embedding_provider={body.get('embedding_provider')} "
          f"reranker_loaded={body.get('reranker_loaded')}")
    for hit in hits:
        print(f"  {hit.get('score'):.4f}  {hit.get('path')}")

    problems = []
    if status != 200:
        problems.append(f"status {status}")
    if not hits:
        problems.append("no hits")
    null_paths = [h for h in hits if not h.get("path")]
    if null_paths:
        problems.append(f"{len(null_paths)} hit(s) with a null path")
    if args.require_bm25 and not (body.get("counts") or {}).get("bm25"):
        problems.append("counts.bm25 == 0")
    if args.require_reranker:
        rr = health.get("reranker") or {}
        if not rr.get("loaded"):
            problems.append(f"reranker not loaded: {rr.get('error')}")
        if health.get("degraded") or body.get("degraded"):
            problems.append(f"degraded components: {health.get('degraded') or body.get('degraded')}")
    indexing = health.get("indexing") or {}
    if indexing.get("indexed") and health.get("n_nodes") != indexing.get("n_nodes"):
        problems.append(f"table has {health.get('n_nodes')} rows but indexing wrote {indexing.get('n_nodes')} (stray rows?)")
    if problems:
        print("SMOKE FAILED:", "; ".join(problems))
        return 1
    print(f"SMOKE OK: {len(hits)} hits, all with paths")
    if args.latency:
        summary = replay_latency(args.url.rstrip("/"), args.latency, health)
        print(f"latency over {summary['n']} questions (k={summary['k']}): server total p50/p95 {summary['server_total_ms']['p50']}/{summary['server_total_ms']['p95']} ms; "
              f"e2e p50/p95 {summary['e2e_ms']['p50']}/{summary['e2e_ms']['p95']} ms; stages p50 {summary['stages_p50_ms']}")
        if args.latency_out:
            import os
            os.makedirs(os.path.dirname(args.latency_out) or ".", exist_ok=True)
            with open(args.latency_out, "w") as fh:
                json.dump(summary, fh, indent=2)
            print(f"wrote {args.latency_out}")
    return 0


def _pct(values, p):
    if not values:
        return None
    s = sorted(values)
    i = (len(s) - 1) * p / 100.0
    lo, hi = int(i), min(int(i) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (i - lo), 2)


def replay_latency(base: str, gold_path: str, health: dict, k: int = 8) -> dict:
    """POST every gold question; p50/p95 of the server's total stage time and of e2e wall time."""
    import datetime

    questions = []
    with open(gold_path) as fh:
        for line in fh:
            if line.strip():
                questions.append(json.loads(line)["question"])
    _post(f"{base}/api/retrieve/", {"query": questions[0], "k": k})  # warm-up, excluded
    totals, e2e, stages = [], [], {}
    for q in questions:
        t = time.time()
        _, body = _post(f"{base}/api/retrieve/", {"query": q, "k": k})
        e2e.append((time.time() - t) * 1000.0)
        lat = body.get("latency_ms") or {}
        totals.append(float(lat.get("total", 0.0)))
        for name, ms in lat.items():
            stages.setdefault(name, []).append(float(ms))
    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "url": base,
        "n": len(questions),
        "k": k,
        "warm_up_excluded": True,
        "health": {key: health.get(key) for key in ("embedding_provider", "embed_model", "bm25_backend", "retrieval_mode", "reranker", "degraded", "n_nodes")},
        "server_total_ms": {"p50": _pct(totals, 50), "p95": _pct(totals, 95)},
        "e2e_ms": {"p50": _pct(e2e, 50), "p95": _pct(e2e, 95)},
        "stages_p50_ms": {name: _pct(v, 50) for name, v in stages.items()},
    }


if __name__ == "__main__":
    sys.exit(main())
