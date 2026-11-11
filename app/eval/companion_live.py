"""The research companion in live mode with a local model (A25 / C13): make bench-companion-live.

    PG_TEST_DSN=postgresql://... OLLAMA_BASE_URL=http://127.0.0.1:11434 \\
        python -m app.eval.companion_live --model qwen2.5:7b --out bench/results/companion_live.json \\
        --validation doc/research-agent-validation.json

The companion's HTTP gateway (app/agent/companion_api.py: POST /agent/ask, mode live, with the operator key)
runs in this process, unchanged, in front of two served retrieval APIs (app/eval/served_stack.py: the default
Compose stack under uvicorn, each over a task-local copy of fixture v1 and a database of its own):

  clean    fixture v1 as frozen; every question of bench/gold/questions.jsonl, in file order. These carry a gold
           answer and relevant paths; the companion's own validation cases (three questions over the five-file
           mobile fixture) carry neither, so they cannot be scored without a judge.
  planted  the same copy with the ten hijack texts of app/eval/agent_injection.build_tasks appended to their
           carriers (the config that answers the question and the same run's log; mtimes kept): planted
           instructions to report a false value. The ten gold questions those texts target, asked again.

The decisions come from an Ollama model (OLLAMA_BASE_URL) through the companion's own adapter: /api/chat with
its two tools, temperature 0, no seed; one trial per question, one question at a time. The companion's
outbound HTTP is recorded per question (which upstream, its status, the time to its response headers). Each
answer is scored by app/eval/companion_score.py, and --rescore recomputes every score and the summary from the
stored responses without a model call.

Before any measured question the run checks, and stops on a failure: both APIs' /health is ok with nodes
indexed; a retrieve returns hits whose paths are corpus paths in the gold's relative form; Ollama lists the
model and answers the companion's own request (tools, think false). That request also loads the model, so no measured
question pays the load. The request deadline is AGENT_TIMEOUT_SECONDS' largest allowed value (the default is
lower) so that a cold model is not what is measured; the per-upstream-call cap of companion_api is unchanged.
Both are recorded. An HTTP failure (503, 504) or a blocked answer is a result and is not retried.

Fallbacks, to be named wherever these numbers appear: hash embeddings, Postgres ts_rank_cd for the lexical
leg, no reranker; one local 7-8B model; synthetic fixture questions.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import httpx

from app.eval import companion_score as cs
from app.eval import served_stack
from app.eval.agent_injection import TICKET, build_tasks
from app.eval.provenance import sanitize_command, scrub

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = "companion_live_v1"
DEFAULT_OUT = Path("bench/results/companion_live.json")
SOURCES = ("app/agent/companion.py", "app/agent/companion_api.py", "app/eval/companion_live.py",
           "app/eval/companion_score.py", "app/eval/served_stack.py", "app/eval/agent_injection.py",
           "app/api/routers/retrieve.py", "main.py")
PREFLIGHT_QUERY = "what learning rate did run 47 use"
PREFLIGHT_QUESTION = "Preflight check: no search is needed. Reply with the JSON abstention."
PREVIEW_CHARS = 200  # app/agent/companion.py ResearchTools: a search returns text[:200] of each source
SCOPE = ("The read-only research companion (app/agent/companion_api.py) in live mode: a local Ollama model chooses "
         "searches and inspections over the served retrieval API of the default stack (hash embeddings, ts_rank_cd, "
         "no reranker) on fixture v1, and returns cited claims. Every gold question once, plus the ten hijack "
         "questions over a copy with planted instructions; one trial at temperature 0 per question. Scores are "
         "deterministic string and byte checks (app/eval/companion_score.py), not a judge and not entailment.")


def limits() -> dict:
    from app.agent import companion

    return {"max_model_rounds": companion.MAX_MODEL_ROUNDS, "max_tool_calls": companion.MAX_TOOL_CALLS,
            "max_sources": companion.MAX_SOURCES, "max_excerpt_chars": companion.MAX_EXCERPT_CHARS,
            "preview_chars": PREVIEW_CHARS}


def hijack_tasks() -> list[dict]:
    """The ten hijack tasks of the injection benchmark, as this run stores them (planted text included)."""
    out = []
    for task in build_tasks():
        if task.goal != "hijack":
            continue
        out.append({"task_id": task.task_id, "gold_id": task.gold_id, "run": task.run, "field": task.field,
                    "gold": task.gold, "planted_value": task.planted_value, "style": task.style,
                    "carriers": list(task.carriers), "injection": task.injection,
                    "injection_sha256": hashlib.sha256(task.injection.encode("utf-8")).hexdigest()})
    return out


def planted_additions(tasks: Sequence[dict]) -> dict[str, str]:
    """{carrier: planted text}; two tasks never share a carrier (each targets its own run)."""
    additions: dict[str, str] = {}
    for task in tasks:
        for carrier in task["carriers"]:
            if carrier in additions:
                raise ValueError(f"{carrier} carries two planted texts")
            additions[carrier] = task["injection"]
    return additions


class RecordingTransport(httpx.AsyncBaseTransport):
    """The companion's outbound HTTP, observed and passed on unchanged: per call, the upstream (model or
    retrieval), its status and the time to its response headers; and the first model request's fields other
    than the messages, as the companion's adapter sent them."""

    def __init__(self, inner: httpx.AsyncBaseTransport | None = None):
        self.inner = inner or httpx.AsyncHTTPTransport()
        self.calls: list[dict] = []
        self.model_request: dict | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        upstream = "model" if path == "/api/chat" else "retrieval" if path == "/api/retrieve/" else "other"
        if upstream == "model" and self.model_request is None:
            body = json.loads(await request.aread())
            self.model_request = {k: ([t["function"]["name"] for t in v] if k == "tools" else v)
                                  for k, v in body.items() if k != "messages"}
        started = time.perf_counter()
        try:
            response = await self.inner.handle_async_request(request)
        except Exception as exc:
            self.calls.append({"upstream": upstream, "status": None, "error": type(exc).__name__,
                               "s": round(time.perf_counter() - started, 3)})
            raise
        self.calls.append({"upstream": upstream, "status": response.status_code, "s": round(time.perf_counter() - started, 3)})
        return response

    async def aclose(self) -> None:
        await self.inner.aclose()


async def ask_all(questions: Sequence[dict], *, retrieval_url: str, model: str, ollama_url: str, timeout: float,
                  transport: RecordingTransport) -> tuple[list[dict], dict]:
    """POST /agent/ask (live) once per question through the companion's own app; rows in question order."""
    from app.agent.companion_api import AgentConfig, create_app

    key = secrets.token_urlsafe(24)
    config = AgentConfig(api_key=key, retrieval_url=retrieval_url, model=model, model_url=ollama_url,
                         timeout_seconds=timeout, max_concurrent=1)
    app = create_app(config, transport=transport)
    rows = []
    async with app.router.lifespan_context(app):
        gateway = {"request_timeout_s": config.timeout_seconds,
                   "upstream_timeout_s": app.state.live_model.client.timeout.read,
                   "max_concurrent": config.max_concurrent}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://companion",
                                     timeout=None) as client:
            for question in questions:
                mark = len(transport.calls)
                started = time.perf_counter()
                response = await client.post("/agent/ask", json={"question": question["question"], "mode": "live"},
                                              headers={"X-API-Key": key})
                row = {**{k: v for k, v in question.items() if k != "question"}, "question": question["question"],
                       "http_status": response.status_code, "latency_s": round(time.perf_counter() - started, 3),
                       "upstream_calls": transport.calls[mark:]}
                try:
                    body = response.json()
                except ValueError:
                    body = None
                if response.status_code == 200 and isinstance(body, dict):
                    row["response"] = body
                else:
                    row["detail"] = body.get("detail") if isinstance(body, dict) else None
                rows.append(row)
    return rows, gateway


async def preflight_model(model: str, ollama_url: str, inner: httpx.AsyncBaseTransport | None = None) -> dict:
    """One request in the companion's own shape (its adapter, its tools, think false): it must succeed."""
    from app.agent.companion import SYSTEM_PROMPT
    from app.agent.companion_api import AgentConfig, OllamaDecisionModel

    transport = RecordingTransport(inner)
    config = AgentConfig(api_key="preflight", retrieval_url="http://127.0.0.1:9", model=model, model_url=ollama_url)
    started = time.perf_counter()
    async with httpx.AsyncClient(transport=transport, timeout=600, follow_redirects=False, trust_env=False) as client:
        try:
            message = await OllamaDecisionModel(config, client).next(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": PREFLIGHT_QUESTION}])
        except Exception as exc:  # the reason is printed; nothing is measured after a failed preflight
            status = transport.calls[-1]["status"] if transport.calls else None
            raise SystemExit(f"preflight: the model did not answer the companion's request "
                             f"(HTTP {status}, {type(exc).__name__}); nothing was measured") from None
    return {"model_request": transport.model_request, "load_and_answer_s": round(time.perf_counter() - started, 3),
            "returned_fields": sorted(message), "returned_tool_calls": bool(message.get("tool_calls"))}


def ollama_info(base_url: str, model: str) -> dict:
    get = served_stack.get_json
    _, version = get(base_url.rstrip("/") + "/api/version")
    _, tags = get(base_url.rstrip("/") + "/api/tags")
    entry = next((m for m in (tags or {}).get("models", []) if model in (m.get("name"), m.get("model"))), None)
    if entry is None:
        raise SystemExit(f"preflight: Ollama at OLLAMA_BASE_URL does not list {model}; nothing was measured")
    details = entry.get("details") or {}
    return {"version": (version or {}).get("version"),
            "models": {model: {"digest": entry.get("digest"), "size": entry.get("size"),
                               "family": details.get("family"), "parameter_size": details.get("parameter_size"),
                               "quantization": details.get("quantization_level"),
                               "context_length": details.get("context_length"),
                               "capabilities": entry.get("capabilities")}}}


def loaded_models(base_url: str) -> list[dict]:
    """/api/ps after the preflight: what the server has loaded (context length, VRAM), as it reports it."""
    _, ps = served_stack.get_json(base_url.rstrip("/") + "/api/ps")
    keep = ("name", "digest", "size", "size_vram", "context_length")
    return [scrub({k: m.get(k) for k in keep if k in m}) for m in (ps or {}).get("models", [])]


def preflight_api(api: served_stack.ServedAPI, corpus: set[str]) -> dict:
    """/health ok with nodes indexed, and a retrieve whose every hit path is a corpus path as the gold writes it
    (relative to the corpus root). Whether the asked file is among the hits is not checked: the served lexical
    leg does not search paths (doc/engineering-reference.md, parity), and that is part of what is measured."""
    n_nodes = (api.health or {}).get("n_nodes") or 0
    if n_nodes <= 0:
        raise SystemExit(f"preflight: {api.label} has no indexed nodes; nothing was measured")
    status, body = served_stack.get_json(api.url + "/api/retrieve/", payload={"query": PREFLIGHT_QUERY, "k": 5})
    paths = [h.get("path") for h in (body or {}).get("hits", [])] if status == 200 else []
    if not paths or any(p not in corpus for p in paths):
        raise SystemExit(f"preflight: {api.label} retrieve returned HTTP {status} and paths {scrub(paths)[:5]}; "
                         f"hits with corpus-relative paths are required; nothing was measured")
    return {"n_nodes": n_nodes, "retrieve_query": PREFLIGHT_QUERY, "retrieve_status": status, "retrieve_paths": paths}


def gpu() -> str | None:
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True,
                                       timeout=10).strip().splitlines()[0]
    except Exception:
        return None


def recorded_command(argv: Sequence[str] | None = None) -> str:
    """provenance.command_line() with a --pg-dsn value (it can hold a password) replaced by its name."""
    argv = list(sys.argv if argv is None else argv)
    for i, arg in enumerate(argv):
        if arg == "--pg-dsn" and i + 1 < len(argv):
            argv[i + 1] = "<PG_TEST_DSN>"
        elif arg.startswith("--pg-dsn="):
            argv[i] = "--pg-dsn=<PG_TEST_DSN>"
    return sanitize_command(argv, os.environ)


def run_provenance() -> dict:
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.SubprocessError):
        head, dirty = None, None
    return {"run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_head": head,
            "working_tree_dirty": dirty,
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}}


def _api_block(api: served_stack.ServedAPI, corpus: str, data_dir: Path, preflight: dict) -> dict:
    return {"corpus": corpus, "corpus_sha256": served_stack.tree_sha256(data_dir), "startup_s": api.startup_s,
            "health": scrub(api.health), "env": served_stack.public_env(api.env), "preflight": preflight}


def _publishable(rows: list[dict]) -> list[dict]:
    """Rows as written: every string scrubbed of local paths, user and host names (provenance.scrub); a row whose
    stored response or detail changed says so, and its scores are computed from what is stored."""
    out = []
    for row in rows:
        clean = scrub(row)
        clean["scrubbed"] = clean != row
        out.append(clean)
    return out


def run(*, model: str, ollama_url: str, pg_dsn: str, timeout: float, limit: int | None,
        log_dir: Path | None) -> dict:
    started = time.monotonic()
    gold = list(cs.gold_rows().values())
    tasks = hijack_tasks()
    if limit is not None:
        gold, tasks = gold[:limit], tasks[:limit]
    by_id = cs.gold_rows()
    gold_questions = [{"id": g["id"], "kind": "gold", "category": g["category"], "question": g["question"]} for g in gold]
    planted_questions = [{"id": t["task_id"], "kind": "planted", "category": by_id[t["gold_id"]]["category"],
                          "gold_id": t["gold_id"], "gold": t["gold"], "planted_value": t["planted_value"],
                          "question": by_id[t["gold_id"]]["question"]} for t in tasks]
    info = ollama_info(ollama_url, model)
    with tempfile.TemporaryDirectory(prefix="metanavit-companion-live-") as tmp:
        tmp = Path(tmp)
        clean_dir = served_stack.copy_fixture(tmp / "clean")
        planted_dir = served_stack.copy_fixture(tmp / "planted")
        served_stack.plant(planted_dir, planted_additions(tasks))
        # what the APIs index is what the scores are computed against
        clean_files = cs.corpus_files()
        if cs.corpus_files(clean_dir) != clean_files or cs.corpus_files(planted_dir) != cs.planted_files(clean_files, tasks):
            raise SystemExit("the task-local corpora differ from what the scorer rebuilds; nothing was measured")
        logs = log_dir or tmp / "logs"
        with served_stack.ServedAPI(pg_dsn, clean_dir, model=model, ollama_url=ollama_url, log_dir=logs,
                                    label="companion_clean") as clean_api, \
             served_stack.ServedAPI(pg_dsn, planted_dir, model=model, ollama_url=ollama_url, log_dir=logs,
                                    label="companion_planted") as planted_api:
            checks = {"clean": preflight_api(clean_api, set(clean_files)), "planted": preflight_api(planted_api, set(clean_files))}
            model_check = asyncio.run(preflight_model(model, ollama_url))
            loaded = loaded_models(ollama_url)
            print(f"preflight ok: model answered in {model_check['load_and_answer_s']} s; "
                  f"{checks['clean']['n_nodes']} and {checks['planted']['n_nodes']} nodes indexed", flush=True)
            clean_transport, planted_transport = RecordingTransport(), RecordingTransport()
            rows, gateway = asyncio.run(ask_all(gold_questions, retrieval_url=clean_api.url, model=model,
                                                ollama_url=ollama_url, timeout=timeout, transport=clean_transport))
            print(f"{len(rows)} gold questions asked", flush=True)
            planted_rows, _ = asyncio.run(ask_all(planted_questions, retrieval_url=planted_api.url, model=model,
                                                  ollama_url=ollama_url, timeout=timeout, transport=planted_transport))
            served = {"clean": _api_block(clean_api, "task-local copy of bench/corpus/files (fixture v1, MANIFEST "
                                          "mtimes)", clean_dir, checks["clean"]),
                      "planted": _api_block(planted_api, "the clean copy with each planted task's text appended to its "
                                            "carriers (mtimes kept)", planted_dir, checks["planted"])}
    from app.agent.companion_api import AgentConfig

    report = {
        "benchmark": BENCHMARK,
        "scope": SCOPE,
        **run_provenance(),
        "command": recorded_command(),
        "model": model,
        "ollama": {**info, "loaded": loaded},
        "device": {"gpu": gpu()},
        "companion": {"limits": limits(), **gateway, "request_timeout_default_s": AgentConfig().timeout_seconds,
                      "model_request": model_check["model_request"], "trials_per_question": 1,
                      "questions_at_a_time": 1},
        "preflight": {"model": model_check},
        "served_api": served,
        "question_set": {"gold": "bench/gold/questions.jsonl, every question in file order" if limit is None else
                         f"bench/gold/questions.jsonl, the first {limit} (a --limit run)",
                         "n_gold": len(gold_questions),
                         "planted": "the hijack tasks of app/eval/agent_injection.build_tasks (one phrasing each, two "
                                    "styles alternating), planted together in one copy",
                         "n_planted": len(planted_questions)},
        "ticket": TICKET,
        "planted_tasks": tasks,
        "rows": _publishable(rows + planted_rows),
        "wall_s": round(time.monotonic() - started, 1),
    }
    return cs.rescore(report)


def update_validation(path: Path, report: dict) -> None:
    """Set the live-run fields of doc/research-agent-validation.json from the report; nothing else changes."""
    doc = json.loads(path.read_text())
    doc.update(cs.validation_fields(report))
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")


def write(report: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")


def _print_summary(report: dict) -> None:
    s = report["summary"]["gold"]["overall"]
    print(f"gold: {s['n']} questions, status {s['status']}; states the gold value {s['states_gold']['k']} of "
          f"{s['states_gold']['n']} scored; citations valid {s['citations']['valid']} of {s['citations']['n']}")
    p = report["summary"].get("planted")
    if p:
        print(f"planted: {p['n']} tasks, exposed {p['exposed']}, planted value stated {p['planted_value_stated']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--ollama-url", default=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    p.add_argument("--pg-dsn", default=os.environ.get("PG_TEST_DSN"),
                   help="a Postgres server (pgvector available) on which the run may create and drop databases")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--validation", type=Path, default=None,
                   help="doc/research-agent-validation.json: set its live-run fields from this run")
    p.add_argument("--timeout", type=float, default=120.0, help="the companion's request deadline (5-120 s)")
    p.add_argument("--limit", type=int, default=None, help="a dry run: only the first N gold and N planted questions")
    p.add_argument("--log-dir", type=Path, default=None, help="keep the two APIs' logs here")
    p.add_argument("--rescore", type=Path, default=None, metavar="RESULTS",
                   help="recompute the scores and summary of RESULTS from its stored responses (no model call)")
    p.add_argument("--check", action="store_true", help="with --rescore: exit 1 if the file is not a rescore of itself")
    args = p.parse_args(argv)
    if args.rescore:
        report = json.loads(args.rescore.read_text())
        fresh = cs.rescore(report)
        if args.check:
            if fresh != report:
                print(f"drift: {args.rescore} differs from a rescore of its stored responses")
                return 1
            print(f"{args.rescore} is a rescore of its stored responses")
            return 0
        write(fresh, args.rescore)
        if args.validation:
            update_validation(args.validation, fresh)
        _print_summary(fresh)
        return 0
    if not args.pg_dsn:
        p.error("set PG_TEST_DSN (or --pg-dsn): a Postgres server where the run creates its databases")
    if args.limit is not None and args.out == DEFAULT_OUT:
        p.error("a --limit run does not write the committed results file; pass --out")
    report = run(model=args.model, ollama_url=args.ollama_url, pg_dsn=args.pg_dsn, timeout=args.timeout,
                 limit=args.limit, log_dir=args.log_dir)
    write(report, args.out)
    if args.validation:
        update_validation(args.validation, report)
    print(f"wrote {args.out}")
    _print_summary(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
