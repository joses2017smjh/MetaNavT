"""Accounted, bounded HTTP load measurements against a running retrieval service.

python -m app.eval.http_load --url http://127.0.0.1:8000 \
    --queries bench/gold/questions.jsonl --out bench/results/api_latency_load.json

Records every cold-probe, warm-up and measured attempt, including non-2xx,
timeouts and malformed/degraded responses. Cold means the first request this
client sends after readiness; process/model startup is not measured. Use
--fresh-service only after an independently confirmed service restart.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any
from collections.abc import Callable

import httpx

from app.eval.latency import percentile


def distribution(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "p50_ms": round(percentile(values, 50), 3) if values else None,
        "p95_ms": round(percentile(values, 95), 3) if values else None,
        "mean_ms": round(sum(values) / len(values), 3) if values else None,
    }


def readiness_problems(health: dict, *, require_reranker: bool = False, require_device: str | None = None) -> list[str]:
    problems = []
    if health.get("status") != "ok":
        problems.append(f"health status {health.get('status')!r}")
    if not isinstance(health.get("n_nodes"), int) or health["n_nodes"] < 1:
        problems.append("empty or unreported index")
    if health.get("degraded"):
        problems.append("health reports degraded components")
    reranker = health.get("reranker") or {}
    if require_reranker and not reranker.get("loaded"):
        problems.append("required reranker not loaded")
    if require_device:
        devices = health.get("devices") or {}
        embedding = devices.get("embedding") or health.get("embedding_device")
        if not str(embedding or "").startswith(require_device):
            problems.append(f"embedding device {embedding!r} does not verify {require_device}")
        if require_reranker and not str(reranker.get("device") or "").startswith(require_device):
            problems.append(f"reranker device {reranker.get('device')!r} does not verify {require_device}")
    return problems


async def attempt(client: httpx.AsyncClient, *, query: str, query_id: str, k: int, phase: str, sequence: int, require_reranker: bool) -> dict:
    started = time.perf_counter()
    row: dict[str, Any] = {"phase": phase, "sequence": sequence, "query_id": query_id, "status_code": None, "success": False, "error": None, "stages_ms": {}}
    try:
        response = await client.post("/api/retrieve/", json={"query": query, "k": k})
        row["status_code"] = response.status_code
        if response.status_code != 200:
            row["error"] = f"http_{response.status_code}"
        else:
            try:
                body = response.json()
                if not isinstance(body, dict):
                    raise ValueError("response must be an object")
                hits = body.get("hits")
                if not isinstance(hits, list) or not hits:
                    raise ValueError("response has no hits")
                if any(not isinstance(h, dict) or not h.get("path") for h in hits):
                    raise ValueError("response hit lacks a source path")
                timings = body.get("latency_ms")
                if not isinstance(timings, dict) or "total" not in timings:
                    raise ValueError("response lacks stage timings/total")
                row["stages_ms"] = {name: float(ms) for name, ms in timings.items()}
                row["route"] = body.get("route")
                row["counts"] = body.get("counts")
                row["bm25_backend"] = body.get("bm25_backend")
                if any(not math.isfinite(ms) or ms < 0 for ms in row["stages_ms"].values()):
                    raise ValueError("invalid stage timing")
                if body.get("degraded"):
                    raise ValueError("response reports degraded components")
                if require_reranker and not (body.get("reranker") or {}).get("loaded"):
                    raise ValueError("response reranker is not loaded")
                row["success"] = True
            except (ValueError, TypeError, KeyError) as exc:
                row["error"] = f"invalid_response: {exc}"
    except httpx.TimeoutException:
        row["error"] = "timeout"
    except httpx.HTTPError as exc:
        row["error"] = type(exc).__name__
    finally:
        row["e2e_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    return row


def summarize(rows: list[dict], wall_seconds: float, concurrency: int) -> dict:
    successful = [r for r in rows if r["success"]]
    stages = sorted({name for r in successful for name in r["stages_ms"]})
    return {
        "concurrency": concurrency,
        "attempted": len(rows), "completed": len(rows),
        "succeeded": len(successful), "failed": len(rows) - len(successful),
        "error_rate": (len(rows) - len(successful)) / len(rows) if rows else None,
        "status_counts": dict(Counter(str(r["status_code"]) if r["status_code"] is not None else "no_response" for r in rows)),
        "error_counts": dict(Counter(r["error"] for r in rows if r["error"])),
        "unique_query_ids": len({r["query_id"] for r in rows}),
        "route_counts": dict(Counter(str(r.get("route")) for r in successful)),
        "e2e_successful_by_route": {str(route): distribution([r["e2e_ms"] for r in successful if r.get("route") == route]) for route in {r.get("route") for r in successful}},
        "wall_seconds": round(wall_seconds, 6),
        "successful_requests_per_second": len(successful) / wall_seconds if wall_seconds else None,
        "completed_requests_per_second": len(rows) / wall_seconds if wall_seconds else None,
        "e2e_all_attempts": distribution([r["e2e_ms"] for r in rows]),
        "e2e_successful": distribution([r["e2e_ms"] for r in successful]),
        "server_stages_successful": {name: distribution([r["stages_ms"][name] for r in successful if name in r["stages_ms"]]) for name in stages},
    }


async def run_load(base_url: str, queries: list[tuple[str, str]], *, concurrency_levels: tuple[int, ...] = (1, 4, 16), requests_per_level: int = 64, warmup_requests: int = 4, timeout: float = 60.0, require_reranker: bool = False, require_device: str | None = None, fresh_service: bool = False, transport=None, checkpoint: Callable[[dict], None] | None = None) -> dict:
    if not queries or requests_per_level < 1 or warmup_requests < 0 or not concurrency_levels or any(c < 1 for c in concurrency_levels) or timeout <= 0:
        raise ValueError("nonempty queries, positive requests/concurrency/timeout and nonnegative warmup required")
    started_at = datetime.now(timezone.utc).isoformat()
    maximum = max(concurrency_levels)
    limits = httpx.Limits(max_connections=maximum, max_keepalive_connections=maximum)
    records: list[dict] = []
    report: dict[str, Any] = {"schema_version": 1, "measurement": "HTTP client end-to-end retrieval latency; server stage times reported separately", "timestamp": started_at, "url": base_url, "k": 8, "query_count": len(queries), "query_sha256": hashlib.sha256(json.dumps(queries, ensure_ascii=False).encode()).hexdigest(), "timeout_seconds": timeout, "cold_definition": "first request by this client after health check; model load/startup excluded", "fresh_service_confirmed_by_operator": fresh_service, "warmup_excluded_from_measured": True, "health": None, "readiness_errors": [], "runs": [], "requests": records}
    report["planned_requests"] = 1 + warmup_requests + requests_per_level * len(concurrency_levels)
    report["workload"] = {"concurrency_levels": list(concurrency_levels), "requests_per_level": requests_per_level, "warmup_requests": warmup_requests}

    def save_progress() -> None:
        done = [r for r in records if r.get("state") == "completed"]
        report["all_attempts"] = {"attempted": len(records), "completed": len(done), "in_flight": len(records) - len(done), "succeeded": sum(r["success"] for r in done), "failed": sum(not r["success"] for r in done)}
        report["outcome"] = "in_progress"
        if checkpoint:
            checkpoint(report)

    async def recorded_attempt(client: httpx.AsyncClient, *, query: str, query_id: str, phase: str, sequence: int) -> dict:
        row = {"phase": phase, "sequence": sequence, "query_id": query_id, "state": "in_flight", "status_code": None, "success": False, "error": None, "stages_ms": {}}
        records.append(row)
        save_progress()
        try:
            row.update(await attempt(client, query=query, query_id=query_id, k=8, phase=phase, sequence=sequence, require_reranker=require_reranker))
        except asyncio.CancelledError:
            row.update({"error": "cancelled", "state": "completed"})
            save_progress()
            raise
        row["state"] = "completed"
        save_progress()
        return row
    async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout, limits=limits, transport=transport, follow_redirects=False, trust_env=False) as client:
        try:
            health = await client.get("/health")
            if health.status_code != 200:
                report["readiness_errors"] = [f"health HTTP {health.status_code}"]
            else:
                report["health"] = health.json()
                if not isinstance(report["health"], dict):
                    raise ValueError("health must be an object")
                report["readiness_errors"] = readiness_problems(report["health"], require_reranker=require_reranker, require_device=require_device)
        except (httpx.HTTPError, ValueError) as exc:
            report["readiness_errors"] = [f"{type(exc).__name__}: {exc}"]
        if report["readiness_errors"]:
            report["outcome"] = "readiness_failed"
            report["all_attempts"] = summarize([], 0.0, 0)
            if checkpoint:
                checkpoint(report)
            return report
        qid, text = queries[0]
        cold = await recorded_attempt(client, query=text, query_id=qid, phase="cold_probe", sequence=0)
        report["cold_probe"] = cold
        for i in range(warmup_requests):
            qid, text = queries[i % len(queries)]
            await recorded_attempt(client, query=text, query_id=qid, phase="warmup", sequence=i)
        measured_wall = 0.0
        for concurrency in concurrency_levels:
            semaphore = asyncio.Semaphore(concurrency)
            async def one(i: int) -> dict:
                async with semaphore:
                    qid, text = queries[i % len(queries)]
                    return await recorded_attempt(client, query=text, query_id=qid, phase=f"warm_c{concurrency}", sequence=i)
            start = time.perf_counter()
            batch = await asyncio.gather(*(one(i) for i in range(requests_per_level)))
            elapsed = time.perf_counter() - start
            measured_wall += elapsed
            report["runs"].append(summarize(batch, elapsed, concurrency))
        report["measured_attempts"] = summarize([r for r in records if r["phase"].startswith("warm_c")], measured_wall, maximum)
        report["all_attempts"] = {"attempted": len(records), "completed": len(records), "in_flight": 0, "succeeded": sum(r["success"] for r in records), "failed": sum(not r["success"] for r in records)}
        report["outcome"] = "ok" if report["all_attempts"]["failed"] == 0 else "requests_failed"
        if checkpoint:
            checkpoint(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--concurrency", default="1,4,16")
    parser.add_argument("--requests", type=int, default=64, help="requests per concurrency level")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--require-reranker", action="store_true")
    parser.add_argument("--require-device", choices=["cpu", "cuda"])
    parser.add_argument("--fresh-service", action="store_true", help="operator confirms service was restarted before this run")
    args = parser.parse_args(argv)
    queries = []
    for line in args.queries.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            query = row.get("question") or row.get("query") or row.get("text")
            if not isinstance(query, str) or not query.strip():
                parser.error("every query row requires nonempty question, query or text")
            queries.append((str(row.get("id", len(queries))), query))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def checkpoint(report):
        temporary = args.out.with_suffix(args.out.suffix + ".partial")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.out)
    report = asyncio.run(run_load(args.url, queries, concurrency_levels=tuple(int(c) for c in args.concurrency.split(",")), requests_per_level=args.requests, warmup_requests=args.warmup, timeout=args.timeout, require_reranker=args.require_reranker, require_device=args.require_device, fresh_service=args.fresh_service, checkpoint=checkpoint))
    print(json.dumps({"outcome": report["outcome"], "all_attempts": report["all_attempts"], "out": str(args.out)}))
    return 0 if report["outcome"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
