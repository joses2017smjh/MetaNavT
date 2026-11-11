"""The chat route with a real model on the default stack (C23): make chat-route-smoke.

    PG_TEST_DSN=postgresql://... OLLAMA_BASE_URL=http://127.0.0.1:11434 \\
        python -m app.eval.chat_smoke --model qwen2.5:7b --out bench/results/chat_route_smoke.json

One `uvicorn main:app` with docker-compose.yml's api environment (app/eval/served_stack.py: hash embedder, no
reranker, a task-local copy of fixture v1, a database of its own on PG_TEST_DSN) and MODEL set to the Ollama
model under test (Compose names qwen2.5:14b). Code execution stays off: the API's environment has
ENABLE_PYTHON_EXEC=false and no METANAVIT_API_KEY, so get_chat_engine() builds the file-access agent alone
(app/engine/engine.py). The results file records that environment and counts tool calls named like the code
tool; the run fails if there is one.

Questions, fixed by a rule before the run: the first gold question of each category, in file order. Each goes
to both chat routes, one request at a time, after the model is loaded:

  POST /api/chat          the streaming route the web UI calls (a Vercel data stream). It sends nothing until
                          the whole agent run is over and then streams the text one word per 0.1 s (a dummy
                          stream), so the time to the first byte and the total time are both recorded. Tool calls
                          are not in the stream; the source nodes it sends are counted.
  POST /api/chat/request  the non-streaming route: the answer, the source nodes and the tool calls (name,
                          arguments, error flag).

Answered means HTTP 200, no error line (3:) and non-empty text. Whether the answer states the gold value
(app/eval/companion_score.states_value) is recorded as an aside: this is a smoke run of a route, not an accuracy
measurement. The chat LLM's settings are read back from app/settings.init_ollama under the API's environment
(temperature, request timeout), not typed here; the run does not change them.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.eval import companion_score as cs
from app.eval import served_stack
from app.eval.companion_live import (gpu, loaded_models, local_paths, ollama_info, publishable, recorded_command,
                                     run_provenance)
from app.eval.provenance import scrub

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = "chat_route_smoke_v1"
DEFAULT_OUT = Path("bench/results/chat_route_smoke.json")
ROUTES = ("/api/chat", "/api/chat/request")
CODE_TOOLS = ("execute_code", "python_exec")
QUESTION_RULE = "the first gold question of each category of bench/gold/questions.jsonl, in file order"
SOURCES = ("app/eval/chat_smoke.py", "app/eval/served_stack.py", "app/api/routers/chat.py", "app/engine/engine.py",
           "app/engine/agents/file_access_agent.py", "app/engine/agents/multi_agent_workflow.py", "app/settings.py",
           "config/tools.yaml", "main.py")
SCOPE = ("POST /api/chat and POST /api/chat/request of main:app with a local Ollama model, on the default stack "
         "(hash embeddings, ts_rank_cd, no reranker) over fixture v1, code execution off. One request per question "
         "and route, one at a time, after a warm-up. A smoke run of the routes: status, latency, tool calls; not an "
         "accuracy measurement.")


def questions() -> list[dict]:
    first: dict[str, dict] = {}
    for row in cs.gold_rows().values():
        first.setdefault(row["category"], row)
    return list(first.values())


def parse_stream(body: str) -> dict:
    """A Vercel data stream: the text of the 0: lines, the 8: data items, the 3: error lines."""
    text, data, errors = "", [], []
    for line in body.splitlines():
        prefix, _, payload = line.partition(":")
        try:
            value = json.loads(payload) if payload else None
        except ValueError:
            errors.append(f"unparsed {prefix}: line")
            continue
        if prefix == "0":
            text += value or ""
        elif prefix == "8":
            data.extend(value if isinstance(value, list) else [value])
        elif prefix == "3":
            errors.append(value)
    return {"text": text, "data": data, "errors": errors}


def _source_nodes(data: Sequence[Any]) -> int:
    return sum(len((item.get("data") or {}).get("nodes") or []) for item in data
               if isinstance(item, dict) and item.get("type") == "sources")


def tool_calls(result: Mapping[str, Any]) -> list[dict]:
    """The tool calls /api/chat/request returns (llama-index ToolOutput objects): name, arguments, error flag."""
    out = []
    for tool in result.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        raw = tool.get("raw_input") or {}
        out.append({"name": tool.get("tool_name"), "arguments": raw.get("kwargs", raw) if isinstance(raw, dict) else raw,
                    "is_error": bool(tool.get("is_error")), "output_chars": len(str(tool.get("content") or ""))})
    return out


def ask(client, url: str, route: str, question: Mapping[str, Any]) -> dict:
    """One request; every field is a result (an HTTP error, an empty answer and an exception are rows)."""
    payload = {"messages": [{"role": "user", "content": question["question"]}]}
    row: dict[str, Any] = {"id": question["id"], "category": question["category"], "question": question["question"],
                           "route": route}
    started = time.perf_counter()
    first_byte = None
    try:
        with client.stream("POST", url + route, json=payload) as response:
            chunks = []
            for chunk in response.iter_bytes():
                if first_byte is None:
                    first_byte = time.perf_counter() - started
                chunks.append(chunk)
            body = b"".join(chunks).decode("utf-8", errors="replace")
            row["http_status"] = response.status_code
    except Exception as exc:  # a dropped connection or a client timeout is recorded, not retried
        row.update(http_status=None, error=f"{type(exc).__name__}", total_s=round(time.perf_counter() - started, 3),
                   ttfb_s=None, answered=False, answer="", tools=None)
        return row
    row["ttfb_s"] = round(first_byte, 3) if first_byte is not None else None
    row["total_s"] = round(time.perf_counter() - started, 3)
    if row["http_status"] != 200:
        try:
            detail = json.loads(body).get("detail")
        except (ValueError, AttributeError):
            detail = body[:500]
        row.update(answered=False, answer="", detail=detail, tools=None)
        return row
    if route == "/api/chat":
        stream = parse_stream(body)
        row.update(answer=stream["text"].strip(), errors=stream["errors"], source_nodes=_source_nodes(stream["data"]),
                   tools=None)
        row["answered"] = not stream["errors"] and bool(row["answer"])
    else:
        result = json.loads(body)
        row.update(answer=str(((result.get("result") or {}).get("content")) or "").strip(),
                   source_nodes=len(result.get("nodes") or []), tools=tool_calls(result))
        row["answered"] = bool(row["answer"])
    return row


def chat_llm_settings(env: Mapping[str, str]) -> dict:
    """The chat LLM as app/settings.init_ollama builds it under the API's environment, read back from the object."""
    from llama_index.core.settings import Settings

    from app.settings import init_ollama

    keys = ("MODEL", "OLLAMA_BASE_URL", "LLM_TEMPERATURE", "OLLAMA_REQUEST_TIMEOUT", "LLM_MAX_TOKENS",
            "EMBEDDING_PROVIDER", "EMBEDDING_DIM")
    saved = {k: os.environ.get(k) for k in keys}
    llm, embed = Settings._llm, Settings._embed_model
    try:
        for k in keys:
            if k in env:
                os.environ[k] = env[k]
            else:
                os.environ.pop(k, None)
        init_ollama()
        built = Settings.llm
        return {"class": type(built).__name__, "model": built.model, "temperature": built.temperature,
                "request_timeout_s": built.request_timeout, "additional_kwargs": dict(built.additional_kwargs or {}),
                "set_in_api_environment": sorted(k for k in ("LLM_TEMPERATURE", "OLLAMA_REQUEST_TIMEOUT", "LLM_MAX_TOKENS")
                                                 if k in env)}
    finally:
        Settings._llm, Settings._embed_model = llm, embed
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def warm_up(ollama_url: str, model: str) -> float:
    """Load the model (Ollama loads it on an empty prompt) so no measured request pays the load."""
    started = time.perf_counter()
    status, _ = served_stack.get_json(ollama_url.rstrip("/") + "/api/generate", timeout=600,
                                      payload={"model": model, "prompt": "", "stream": False})
    if status != 200:
        raise SystemExit(f"warm-up: Ollama answered HTTP {status} for {model}; nothing was measured")
    return round(time.perf_counter() - started, 3)


def working_tree_changes() -> list[str] | None:
    """`git status --porcelain` paths when the report is built: what makes working_tree_dirty true, if it is."""
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return [line[3:] for line in out.splitlines() if line.strip()]


def _median(values: Sequence[float | None]) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 3) if values else None


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict:
    out = {}
    for route in ROUTES:
        mine = [r for r in rows if r["route"] == route]
        scored = [r for r in mine if r.get("gold_value_stated") is not None]
        names = [t["name"] for r in mine for t in (r.get("tools") or [])]
        out[route] = {
            "n": len(mine),
            "http_status": dict(sorted(Counter(str(r.get("http_status")) for r in mine).items())),
            "answered": sum(bool(r.get("answered")) for r in mine),
            "ttfb_s": {"p50": _median([r.get("ttfb_s") for r in mine]),
                       "max": max((r["ttfb_s"] for r in mine if r.get("ttfb_s") is not None), default=None)},
            "total_s": {"p50": _median([r.get("total_s") for r in mine]),
                        "max": max((r["total_s"] for r in mine if r.get("total_s") is not None), default=None)},
            "tool_calls": dict(sorted(Counter(names).items())) if route == "/api/chat/request" else None,
            "questions_with_a_tool_call": (sum(1 for r in mine if r.get("tools")) if route == "/api/chat/request"
                                           else None),
            "code_tool_calls": sum(n in CODE_TOOLS for n in names),
            "states_gold_value_aside": {"k": sum(bool(r["gold_value_stated"]) for r in scored), "n": len(scored)},
        }
    return out


def run(*, model: str, ollama_url: str, pg_dsn: str, limit: int | None, log_dir: Path | None,
        request_timeout: float) -> dict:
    import httpx

    started = time.monotonic()
    chosen = questions()[: limit if limit is not None else None]
    gold = cs.gold_rows()
    info = ollama_info(ollama_url, model)
    with tempfile.TemporaryDirectory(prefix="metanavit-chat-smoke-") as tmp:
        tmp = Path(tmp)
        local = local_paths(tmp, *([log_dir] if log_dir else []))
        data_dir = served_stack.copy_fixture(tmp / "files")
        with served_stack.ServedAPI(pg_dsn, data_dir, model=model, ollama_url=ollama_url, log_dir=log_dir or tmp / "logs",
                                    label="chat_smoke") as api:
            if api.env["ENABLE_PYTHON_EXEC"] != "false" or api.env["METANAVIT_API_KEY"]:
                raise SystemExit("code execution must be off for this run")
            llm = chat_llm_settings(api.env)
            warm = warm_up(ollama_url, model)
            loaded = loaded_models(ollama_url)
            rows = []
            with httpx.Client(timeout=request_timeout, trust_env=False) as client:
                for question in chosen:
                    for route in ROUTES:
                        row = ask(client, api.url, route, question)
                        parts = cs.gold_parts(gold[question["id"]]["answer"])
                        row["gold_value_stated"] = (cs.answer_score(row["answer"], gold[question["id"]]["answer"], [])["label"] == "correct"
                                                    if parts is not None and row.get("answered") else
                                                    (False if parts is not None else None))
                        rows.append(publishable(row, local))
                        print(f"{question['id']} {route}: HTTP {row['http_status']} answered={row['answered']} "
                              f"in {row['total_s']} s", flush=True)
            served = {"corpus": "task-local copy of bench/corpus/files (fixture v1, MANIFEST mtimes)",
                      "corpus_sha256": served_stack.tree_sha256(data_dir), "startup_s": api.startup_s,
                      "health": scrub(api.health), "env": served_stack.public_env(api.env)}
    report = {
        "benchmark": BENCHMARK,
        "scope": SCOPE,
        **run_provenance(SOURCES),
        "working_tree_changes": working_tree_changes(),
        "command": recorded_command(),
        "model": model,
        "ollama": {**info, "loaded": loaded, "warm_up_s": warm},
        "device": {"gpu": gpu()},
        "chat_llm": llm,
        "code_execution": {"ENABLE_PYTHON_EXEC": served["env"].get("ENABLE_PYTHON_EXEC"),
                           "METANAVIT_API_KEY_set": bool(api.env.get("METANAVIT_API_KEY")),
                           "code_tools_looked_for": list(CODE_TOOLS)},
        "served_api": served,
        "question_rule": QUESTION_RULE if limit is None else f"{QUESTION_RULE}, the first {limit} (a --limit run)",
        "request_timeout_s": request_timeout,
        "rows": rows,
        "summary": summarize(rows),
        "wall_s": round(time.monotonic() - started, 1),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--ollama-url", default=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    p.add_argument("--pg-dsn", default=os.environ.get("PG_TEST_DSN"),
                   help="a Postgres server (pgvector available) on which the run may create and drop a database")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--limit", type=int, default=None, help="a dry run: only the first N questions")
    p.add_argument("--log-dir", type=Path, default=None, help="keep the API's log here")
    p.add_argument("--request-timeout", type=float, default=600.0, help="the client's limit per request (seconds)")
    args = p.parse_args(argv)
    if not args.pg_dsn:
        p.error("set PG_TEST_DSN (or --pg-dsn): a Postgres server where the run creates its database")
    if args.limit is not None and args.out == DEFAULT_OUT:
        p.error("a --limit run does not write the committed results file; pass --out")
    report = run(model=args.model, ollama_url=args.ollama_url, pg_dsn=args.pg_dsn, limit=args.limit,
                 log_dir=args.log_dir, request_timeout=args.request_timeout)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    for route, s in report["summary"].items():
        print(f"{route}: answered {s['answered']} of {s['n']}, total p50 {s['total_s']['p50']} s, "
              f"tools {s['tool_calls']}, code tool calls {s['code_tool_calls']}")
    print(f"wrote {args.out}")
    if any(s["code_tool_calls"] for s in report["summary"].values()):
        print("a code tool was called although code execution is off")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
