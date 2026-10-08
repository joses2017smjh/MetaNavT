"""Load-runner accounting and readiness, with deterministic HTTP failures."""

import asyncio
import json

import httpx
import pytest

from app.eval.http_load import distribution, readiness_problems, run_load


HEALTH = {"status": "ok", "n_nodes": 8, "devices": {"embedding": "cpu"}, "reranker": {"loaded": False}, "degraded": []}
BODY = {"hits": [{"path": "experiment/metrics.csv"}], "latency_ms": {"total": 2.0, "embed": 1.0}, "degraded": []}


def test_all_statuses_timeouts_and_invalid_responses_are_accounted():
    async def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json=HEALTH)
        query = json.loads(request.content)["query"]
        if query == "timeout":
            raise httpx.ReadTimeout("injected", request=request)
        if query == "error":
            return httpx.Response(503, json={"detail": "unavailable"})
        if query == "malformed":
            return httpx.Response(200, json={"hits": []})
        return httpx.Response(200, json=BODY)
    report = asyncio.run(run_load("http://service", [("q1", "ok"), ("q2", "timeout"), ("q3", "error"), ("q4", "malformed")], concurrency_levels=(1, 4, 16), requests_per_level=4, warmup_requests=0, transport=httpx.MockTransport(handler)))
    assert report["outcome"] == "requests_failed"
    assert report["all_attempts"] == {"attempted": 13, "completed": 13, "in_flight": 0, "succeeded": 4, "failed": 9}
    assert len(report["requests"]) == 13  # cold probe + all measured requests
    for run in report["runs"]:
        assert (run["attempted"], run["completed"], run["succeeded"], run["failed"]) == (4, 4, 1, 3)
        assert run["status_counts"] == {"200": 2, "no_response": 1, "503": 1}
        assert run["e2e_all_attempts"]["n"] == 4
        assert run["e2e_successful"]["n"] == 1
        assert run["server_stages_successful"]["total"]["n"] == 1
        assert run["error_counts"]["timeout"] == 1
        assert run["error_rate"] == 0.75


def test_concurrency_is_bounded_and_warmup_excluded():
    active = 0
    peak = 0
    async def handler(request):
        nonlocal active, peak
        if request.url.path == "/health":
            return httpx.Response(200, json=HEALTH)
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.005)
        active -= 1
        return httpx.Response(200, json=BODY)
    report = asyncio.run(run_load("http://service", [("q1", "ok")], concurrency_levels=(4,), requests_per_level=12, warmup_requests=2, transport=httpx.MockTransport(handler)))
    assert peak == 4
    assert report["all_attempts"]["attempted"] == 15
    assert report["measured_attempts"]["attempted"] == 12
    assert report["cold_probe"]["phase"] == "cold_probe"
    assert report["fresh_service_confirmed_by_operator"] is False


def test_degraded_or_unreported_cuda_does_not_produce_measurements():
    async def handler(request):
        assert request.url.path == "/health"
        return httpx.Response(200, json=HEALTH)
    report = asyncio.run(run_load("http://service", [("q1", "ok")], require_device="cuda", require_reranker=True, transport=httpx.MockTransport(handler)))
    assert report["outcome"] == "readiness_failed"
    assert report["requests"] == [] and report["runs"] == []
    assert report["all_attempts"]["e2e_successful"]["p95_ms"] is None
    assert len(report["readiness_errors"]) == 3


def test_degraded_success_response_is_failure_not_fast_success():
    async def handler(request):
        return httpx.Response(200, json=HEALTH if request.url.path == "/health" else {**BODY, "degraded": [{"component": "reranker"}]})
    report = asyncio.run(run_load("http://service", [("q", "ok")], concurrency_levels=(1,), requests_per_level=2, warmup_requests=0, transport=httpx.MockTransport(handler)))
    assert report["all_attempts"]["failed"] == 3
    assert report["runs"][0]["e2e_successful"]["p50_ms"] is None


def test_readiness_accepts_only_actual_reported_devices():
    health = {**HEALTH, "devices": {"embedding": "cuda:0"}, "reranker": {"loaded": True, "device": "cuda:0"}}
    assert readiness_problems(health, require_device="cuda", require_reranker=True) == []
    assert readiness_problems({**health, "degraded": ["unavailable"]})
    assert distribution([]) == {"n": 0, "p50_ms": None, "p95_ms": None, "mean_ms": None}


def test_empty_or_unbounded_invalid_workloads_are_refused():
    with pytest.raises(ValueError):
        asyncio.run(run_load("http://service", []))
    with pytest.raises(ValueError):
        asyncio.run(run_load("http://service", [("q", "ok")], concurrency_levels=(0,)))


def test_checkpoints_preserve_started_and_completed_attempts_on_cancellation():
    snapshots = []
    async def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json=HEALTH)
        await asyncio.sleep(60)
        return httpx.Response(200, json=BODY)
    async def execute():
        task = asyncio.create_task(run_load("http://service", [("q", "ok")], concurrency_levels=(1,), requests_per_level=1, transport=httpx.MockTransport(handler), checkpoint=lambda blob: snapshots.append(json.loads(json.dumps(blob)))))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(execute())
    assert snapshots[0]["all_attempts"] == {"attempted": 1, "completed": 0, "in_flight": 1, "succeeded": 0, "failed": 0}
    assert snapshots[-1]["all_attempts"]["completed"] == 1
    assert snapshots[-1]["requests"][0]["error"] == "cancelled"
    assert snapshots[-1]["outcome"] == "in_progress"  # a partial report cannot become success
