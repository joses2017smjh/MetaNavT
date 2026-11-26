"""M50's probe (P0): the format and speed facts BAR's tables decide from, on questions that carry no planted text.

    python -m app.eval.agent_injection_probe --model qwen3.5:27b --out bench/results/agent_injection_v3_models_2026_probe.json \
        [--prior <an earlier probe file>] [--server-log <the job's Ollama log>]

For one model on a running Ollama server (the job's, gated before this runs), this sends agent episodes through the
harness's own run_episode on the probe questions of BAR (q001, q003, q009: runs 40 and 41, the test's fields, no
planted text, never scored), under none and answer_check with seeds 100 and 101, first with request A (think: false)
and then with request B (the committed request: no think key), each batch through 8 concurrent workers on one server
slot, as the shards send theirs. Every chat call is recorded: the request, the probe question, the reasoning detector
(the length of message.thinking and any listed marker in the content), done_reason, eval_count, prompt_eval_count, the
tool-call format (structured, recovered from text, an unparsed listed form, or none) and its latency. Then one
deliberately oversized prompt (fixture text above the pinned context) shows what the server does with a prompt it
cannot hold, and the server log, when given, supplies the gate lines (load_tensors offload, memory breakdown) and the
per-request truncation lines. The decisions BAR's tables give follow mechanically (agent_injection_models): the
reasoning case and think setting, whether one parser addition is due, the workers, H and the shard split, and the
request block every shard of the model must record.

The file holds one entry per model; --prior names an earlier probe file (read, never changed) whose models are kept,
so two jobs can probe one model each. It is written through provenance.py's scrub, holds BAR under `bar`, and outside
that copy names no question of the test set (USER_TASKS).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from app.eval import agent_injection as ai
from app.eval import agent_injection_models as m50
from app.eval.m4_provenance import provenance
from app.eval.provenance import command_line, scrub

DEFAULT_OUT = Path("bench/results/agent_injection_v3_models_2026_probe.json")
# the probe questions BAR names: (gold id, run, config field)
PROBE_TASKS = (("q001", 40, "learning_rate"), ("q003", 40, "bark_type"), ("q009", 41, "encoder"))
PROBE_DEFENSES = ("none", ai.ANSWER_CHECK)
PROBE_SEEDS = (100, 101)
REQUESTS = {"A": "off", "B": "default"}  # A: think false; B: the committed request, no think key
WORKERS = 8
OVERSIZED_CHARS = 140_000  # fixture text, about 40,000 tokens: above the pinned 32,768-token context
CONTENT_KEPT = 2000  # characters of each call's content kept (the parser rule reads the probe's stored contents)
DOES_NOT_SHOW = [
    "Format and speed on three questions without planted text, not attack or utility rates: nothing here is scored.",
    ("Speed at 8 workers on one server slot over 12 episodes per request; probe episodes have no planted text and run "
     "fewer turns than test episodes, so H uses the committed maximum mean turns (5.23), not the probe's own."),
    "One GPU of the job's type; the shards may run on other types (recorded per segment).",
]


def gold(gold_id: str) -> str:
    for line in (ai.ROOT / "bench" / "gold" / "questions.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["id"] == gold_id:
            return row["answer"]
    raise KeyError(gold_id)


def probe_tasks() -> list[ai.Task]:
    """The probe questions as harness tasks: a question, its run's config and field, no carrier and no planted text."""
    tasks = []
    for gold_id, run, field in PROBE_TASKS:
        path = f"configs/run_{run:03d}.yaml"
        if gold_id in {q for q, *_ in ai.USER_TASKS}:
            raise ValueError(f"{gold_id} is a test question")
        tasks.append(ai.Task(task_id=f"{gold_id}-probe", gold_id=gold_id, question=ai._question(gold_id), run=run,
                             field=field, gold=gold(gold_id), gold_path=path))
    return tasks


def call_format(message: dict) -> str:
    """How a reply carries a tool call: structured (message.tool_calls), recovered_from_text (text_tool_call finds
    one), unparsed_form (a listed form text_tool_call does not recover), or none (a plain answer)."""
    content = message.get("content") or ""
    if message.get("tool_calls"):
        return "structured"
    if ai.text_tool_call(content) is not None:
        return "recovered_from_text"
    return "unparsed_form" if m50.unparsed_forms_in(content) else "none"


def recording(chat: ai.ChatFn, calls: list[dict], lock: threading.Lock, env: ai.TaskEnv, **tags: Any) -> ai.ChatFn:
    """`chat` recording each call of one episode in `calls`."""
    count = [0]

    def call(messages: list[dict], tools: list[dict], seed: int) -> dict:
        out = chat(messages, tools, seed)
        message = out["message"]
        content = message.get("content") or ""
        count[0] += 1
        record = {**tags, "call": count[0], "done_reason": out.get("done_reason"),
                  "eval_count": out.get("output_tokens"), "prompt_eval_count": out.get("prompt_tokens"),
                  "thinking_chars": out.get("thinking_chars", 0), "reasoning_markers": m50.reasoning_markers_in(content),
                  "unparsed_forms": m50.unparsed_forms_in(content), "tool_call_format": call_format(message),
                  "tool_names": [str((c.get("function") or {}).get("name")) for c in message.get("tool_calls") or []],
                  "wall_s": out.get("wall_s"), "content_chars": len(content),
                  "content": env.publishable(content[:CONTENT_KEPT])}
        with lock:
            calls.append(record)
        return out
    return call


def batch(model: str, label: str, base_url: str, tasks: list[ai.Task], envs: dict[str, ai.TaskEnv],
          chat_factory=None) -> tuple[dict, list[dict], list[dict]]:
    """One request's probe episodes through WORKERS concurrent workers: the batch's wall time, its calls and episodes."""
    think = REQUESTS[label]
    rules = m50.BAR["request"]
    chat = (chat_factory or (lambda: ai.OllamaChat(model, base_url, temperature=rules["temperature"],
                                                    num_predict=rules["num_predict"], timeout=rules["chat_timeout_s"],
                                                    think=False if think == "off" else None)))()
    calls: list[dict] = []
    lock = threading.Lock()
    jobs = [(t, d, s) for t in tasks for d in PROBE_DEFENSES for s in PROBE_SEEDS]

    def one(job):
        task, defense, seed = job
        env = envs[task.task_id]
        tags = {"request": label, "question": task.gold_id, "defense": defense, "seed": seed}
        try:
            ep = ai.run_episode(recording(chat, calls, lock, env, **tags), env, defense, seed=seed,
                                max_turns=rules["max_turns"])
            return {**tags, "turns": ep["turns"], "answer": (ep["answer"] or "")[:CONTENT_KEPT],
                    "length_stops": ep["length_stops"], "thinking_calls": ep["thinking_calls"], "error": None}
        except Exception as exc:  # recorded, never dropped
            return {**tags, "turns": None, "answer": None, "error": f"{type(exc).__name__}: {exc}"}

    started = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        episodes = list(pool.map(one, jobs))
    wall = time.time() - started
    calls.sort(key=lambda c: (c["question"], PROBE_DEFENSES.index(c["defense"]), c["seed"], c["call"]))
    return ({"request": label, "think": think, "workers": WORKERS, "episodes": len(episodes),
             "episode_errors": sum(bool(e["error"]) for e in episodes), "calls": len(calls), "wall_s": round(wall, 1)},
            calls, episodes)


def oversized(model: str, base_url: str) -> dict:
    """One prompt of fixture text above the pinned context, sent as think false with a tiny cap: what the server does
    with it (its reply's counts, or the error), next to the size sent."""
    files = [p.read_text(errors="replace") for p in sorted(ai.CORPUS.rglob("*")) if p.is_file()]
    text = ""
    while len(text) < OVERSIZED_CHARS:
        text += "\n".join(files)
    text = text[:OVERSIZED_CHARS] + "\n\nReply with the single word OK."
    body = {"model": model, "stream": False, "think": False, "messages": [{"role": "user", "content": text}],
            "options": {"temperature": 0.0, "seed": 0, "num_predict": 8}}
    started = time.monotonic()
    out: dict[str, Any] = {"characters_sent": len(text)}
    try:
        req = urllib.request.Request(f"{base_url}/api/chat", data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=m50.BAR["request"]["chat_timeout_s"]) as r:
            reply = json.load(r)
        out.update(status="ok", prompt_eval_count=reply.get("prompt_eval_count"), eval_count=reply.get("eval_count"),
                   done_reason=reply.get("done_reason"), content=(reply.get("message") or {}).get("content"))
    except Exception as exc:
        detail = ""
        if hasattr(exc, "read"):
            try:
                detail = exc.read().decode(errors="replace")[:500]
            except Exception:
                pass
        out.update(status="error", error=f"{type(exc).__name__}: {exc} {detail}".strip())
    out["wall_s"] = round(time.monotonic() - started, 1)
    return out


TRUNCATED = re.compile(r"truncated = (\d+)")


def server_log_facts(path: Path | None) -> dict | None:
    """The gate lines and the truncation lines of the job's server log: every 'load_tensors: offloaded N/M layers to
    GPU' line and whether each reads N = M, the 'memory breakdown' lines, the count of per-request status lines
    ('truncated = N') and of those with N > 0, and every other line that mentions truncation."""
    if not path or not Path(path).exists():
        return None
    lines = Path(path).read_text(errors="replace").splitlines()
    offload = [ln.strip() for ln in lines if "load_tensors: offloaded" in ln]
    pairs = [tuple(map(int, m.groups())) for ln in offload if (m := re.search(r"offloaded (\d+)/(\d+) layers", ln))]
    status = [ln for ln in lines if TRUNCATED.search(ln)]
    nonzero = [ln.strip() for ln in status if int(TRUNCATED.search(ln).group(1)) > 0]
    other = [ln.strip()[:400] for ln in lines if "truncat" in ln.lower() and not TRUNCATED.search(ln)]
    return {"load_tensors": [ln[-200:] for ln in offload], "offload_complete": bool(pairs) and all(n == m for n, m in pairs),
            "memory_breakdown": [ln.strip()[-300:] for ln in lines if "memory breakdown" in ln][:4],
            "status_lines": len(status), "truncated_nonzero": len(nonzero), "truncated_nonzero_lines": nonzero[:10],
            "other_truncation_lines": other[:20]}


def loaded_context(base_url: str, model: str) -> int | None:
    try:
        with urllib.request.urlopen(f"{base_url}/api/ps", timeout=10) as r:
            return next((m.get("context_length") for m in json.load(r).get("models", []) if m.get("name") == model), None)
    except Exception:
        return None


def decisions(calls: list[dict], batches: dict[str, dict]) -> dict:
    """What BAR's tables give from these calls: the reasoning case, the speed rules on the chosen request's batch, the
    parser-addition trigger on its calls, and the request block every shard of the model must record."""
    reasoning = m50.reasoning_decision(calls)
    label = m50.chosen_request_label(reasoning["think"])
    out: dict[str, Any] = {"reasoning": reasoning, "chosen_request": label, "speed": None,
                           "parser_addition": None}
    if label is None:
        return out
    chosen = [c for c in calls if c["request"] == label]
    out["speed"] = m50.speed_decisions(batches[label]["wall_s"], chosen)
    missed = [c for c in chosen if c["tool_call_format"] == "unparsed_form"]
    out["parser_addition"] = {"due": bool(missed), "forms": sorted({f for c in missed for f in c["unparsed_forms"]}),
                              "calls": len(missed)}
    return out


def probe_model(model: str, base_url: str, server_log: Path | None = None, chat_factory=None) -> dict:
    """Every probe record of one model (see the module docstring)."""
    prov = provenance()
    if chat_factory is None:
        ai.check_pins(base_url, m50.BAR["models"]["digests"], [model])
    tasks = probe_tasks()
    envs = {t.task_id: ai.TaskEnv(t) for t in tasks}
    try:
        batches, calls, episodes = {}, [], []
        for label in REQUESTS:
            info, batch_calls, batch_episodes = batch(model, label, base_url, tasks, envs, chat_factory)
            batches[label], calls, episodes = info, calls + batch_calls, episodes + batch_episodes
    finally:
        for env in envs.values():
            env.close()
    entry: dict[str, Any] = {
        "started_utc": prov["run_at_utc"], "git_head": prov["git_head"], "working_tree_dirty": prov["working_tree_dirty"],
        "command": command_line(), "device": {"gpu": ai._gpu()},
        "ollama": ai._ollama_info(base_url, [model]) if chat_factory is None else None,
        "batches": batches, "calls": calls, "episodes": episodes,
        "oversized_prompt": oversized(model, base_url) if chat_factory is None else None,
    }
    time.sleep(2 if chat_factory is None else 0)  # let the server log the oversized request's status line
    entry["server"] = {"context_length": loaded_context(base_url, model) if chat_factory is None else None,
                       "pinned_context": m50.BAR["request"]["server_context"], "log": server_log_facts(server_log)}
    entry["decisions"] = decisions(calls, batches)
    entry["decisions"]["request"] = m50.request_for(entry) if entry["decisions"]["reasoning"]["think"] else None
    entry["finished_utc"] = provenance()["run_at_utc"]
    return entry


def write(entry: dict, model: str, out: Path, prior: Path | None = None) -> dict:
    """The probe file with this model's entry, keeping the entries of `prior` (an earlier probe file at the same commit
    and BAR); refuses a prior file of another commit or BAR."""
    blob: dict[str, Any] = {
        "what": ("M50's probe (P0): format and speed facts on the probe questions, and the decisions BAR's tables give "
                 "from them, per model; no planted text and nothing scored"),
        "bar": m50.BAR, "probe_questions": [q for q, *_ in PROBE_TASKS], "probe_defenses": list(PROBE_DEFENSES),
        "probe_seeds": list(PROBE_SEEDS), "requests": dict(REQUESTS), "workers": WORKERS,
        "git_head": entry["git_head"], "models": {}, "superseded_probes": [], "does_not_show": DOES_NOT_SHOW}
    if prior is not None:
        earlier = json.loads(Path(prior).read_text())
        if earlier["bar"] != m50.BAR or earlier["git_head"] != entry["git_head"]:
            raise SystemExit(f"{prior} was probed at another commit or BAR; a re-probe at a new commit supersedes it "
                             "(superseded_probes), it is not merged")
        blob["models"].update(earlier["models"])
        blob["superseded_probes"] = earlier.get("superseded_probes") or []
        blob["prior_sha256"] = hashlib.sha256(Path(prior).read_bytes()).hexdigest()
    blob["models"][model] = entry
    blob["models"] = {m: blob["models"][m] for m in sorted(blob["models"], key=lambda m: (
        m50.BAR["models"]["tier1"].index(m) if m in m50.BAR["models"]["tier1"] else 99, m))}
    blob = scrub(blob)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=2, ensure_ascii=False) + "\n")
    return blob


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--prior", type=Path, default=None, help="an earlier probe file whose models are kept")
    p.add_argument("--server-log", type=Path, default=None, help="the job's Ollama server log")
    p.add_argument("--base-url", default=ai.factory_url(None))
    args = p.parse_args(argv)
    entry = probe_model(args.model, args.base_url.rstrip("/"), args.server_log)
    write(entry, args.model, args.out, args.prior)
    d = entry["decisions"]
    print(f"wrote {args.out}: {args.model} reasoning case {d['reasoning']['case']} (think {d['reasoning']['think']}), "
          f"speed {d['speed']}, parser addition {d['parser_addition']}, request {d['request']}")
    for label, b in entry["batches"].items():
        print(f"  batch {label}: {b}")
    print(f"  oversized prompt: { {k: v for k, v in (entry['oversized_prompt'] or {}).items() if k != 'content'} }")
    print(f"  server: {entry['server']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
