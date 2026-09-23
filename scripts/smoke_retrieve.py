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


def _post(url: str, payload: dict):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"content-type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
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
    args = p.parse_args()

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
    if problems:
        print("SMOKE FAILED:", "; ".join(problems))
        return 1
    print(f"SMOKE OK: {len(hits)} hits, all with paths")
    return 0


if __name__ == "__main__":
    sys.exit(main())
