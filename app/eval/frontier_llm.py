"""LLM-written generations for frontier arms, stored so their rows can be rescored without the model.

    python -m app.eval.frontier_llm --model qwen2.5:7b --out bench/results/frontier_generations.json
    (make frontier-generations; needs a local Ollama server at OLLAMA_BASE_URL)

Two kinds, each from the prototype's own prompt, at temperature 0 with seed 0:

- hyde: one hypothetical document per gold question (app.retrieval.hyde.HYDE_PROMPT). Only the
  question text reaches the model: no answer, category or relevant path.
- context: one situating description per fixture chunk (app.chunking.contextual.DEFAULT_PROMPT, as
  situate_chunk fills it: the file's path, its first 400 characters and the chunk).

Every item stores the text the model returned. A call that failed or returned nothing is stored with
ok=false and an empty text and counted; the arm that reads it then does what the prototype does with an
empty completion (HyDE: the keyword hypothesis; context: no prefix), and the row says how often.
`make bench-frontier-neural` reads this file and calls no model.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from app.chunking.contextual import DEFAULT_PROMPT, situate_chunk
from app.chunking.structure import Span
from app.eval.gold import GoldQuestion, load_gold
from app.eval.hashing import content_hash
from app.eval.ingest import BENCH_POLICY, file_chunks, scan_directory
from app.retrieval.hyde import HYDE_PROMPT

ROOT = Path(__file__).resolve().parents[2]
GENERATIONS_FILE = "bench/results/frontier_generations.json"
OPTIONS = {"temperature": 0.0, "seed": 0}  # what app.eval.judge.ollama_complete sends (temperature passed explicitly)

Complete = Callable[[str], "str | None"]


def fixture_chunks(files_root: Path) -> list[dict[str, Any]]:
    """Every chunk the benchmark indexes (same scan and chunker as app.eval.index_loader), with its file's text."""
    scan = scan_directory(Path(files_root), BENCH_POLICY)
    out = []
    for source in scan.files:
        for piece in file_chunks(source.path, source.text, "auto"):
            out.append({"chunk_id": piece.chunk_id, "path": source.path, "text": piece.text, "kind": piece.kind,
                        "span_sha256": content_hash(piece.raw), "doc_text": source.text})
    return out


def _timed(complete: Complete, prompt: str) -> tuple[str, bool, float]:
    t0 = time.perf_counter()
    try:
        raw = complete(prompt)
    except Exception:
        raw = None
    ms = (time.perf_counter() - t0) * 1000.0
    text = (raw or "").strip()
    return text, bool(text), round(ms, 1)


def generate(complete: Complete, gold: Iterable[GoldQuestion], chunks: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """The `hyde` and `context` blocks of a generations file."""
    hyde_items = []
    for q in gold:
        text, ok, ms = _timed(complete, HYDE_PROMPT.format(query=q.question))
        hyde_items.append({"id": q.id, "question": q.question, "text": text, "ok": ok, "ms": ms})
    context_items = []
    for c in chunks:
        prompts: list[str] = []

        def capture(prompt: str) -> str:
            prompts.append(prompt)
            return ""

        # situate_chunk builds the prompt exactly as the prototype does; the model is called once, outside it.
        situate_chunk(c["path"], c["doc_text"], Span(0, len(c["text"]), c["text"], c["kind"]), capture)
        text, ok, ms = _timed(complete, prompts[0])
        context_items.append({"chunk_id": c["chunk_id"], "path": c["path"], "span_sha256": c["span_sha256"], "text": text, "ok": ok, "ms": ms})
    return {
        "hyde": {"prompt": HYDE_PROMPT, "source": "app.retrieval.hyde.HYDE_PROMPT (llm_hypothesis)", "n": len(hyde_items),
                 "n_failed": sum(1 for i in hyde_items if not i["ok"]), "items": hyde_items},
        "context": {"prompt": DEFAULT_PROMPT, "source": "app.chunking.contextual.DEFAULT_PROMPT (situate_chunk)", "n": len(context_items),
                    "n_failed": sum(1 for i in context_items if not i["ok"]), "items": context_items},
    }


# ---- reading a generations file (no model) ----------------------------------------------------

def load_generations(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run `make frontier-generations` (Ollama) first")
    return json.loads(path.read_text())


def hyde_completion(gen: dict[str, Any]) -> Callable[[str], str]:
    """A completion function that answers each HyDE prompt with the stored hypothesis ('' for a failed call).
    A question with no stored hypothesis is an error, not a silent fallback."""
    stored = {HYDE_PROMPT.format(query=i["question"]): i["text"] for i in gen["hyde"]["items"]}

    def complete(prompt: str) -> str:
        if prompt not in stored:
            raise KeyError("no stored HyDE hypothesis for this question; re-run `make frontier-generations`")
        return stored[prompt]

    return complete


def context_prefixes(gen: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """chunk_id -> {span_sha256, text, ok}: the stored prefix of every chunk."""
    return {i["chunk_id"]: {"span_sha256": i["span_sha256"], "text": i["text"], "ok": i["ok"]} for i in gen["context"]["items"]}


def header(gen: dict[str, Any], rel: str, text: str) -> dict[str, Any]:
    """What a results file that used the generations records about them (model, prompts, options, failures)."""
    return {
        "file": rel,
        "sha256": content_hash(text),
        "git_sha": gen.get("git_sha"),
        "model": gen["model"],
        "ollama": gen.get("ollama"),
        "options": gen["options"],
        "prompts": {"hyde": gen["hyde"]["prompt"], "context": gen["context"]["prompt"]},
        "n": {"hyde": gen["hyde"]["n"], "context": gen["context"]["n"]},
        "n_failed": {"hyde": gen["hyde"]["n_failed"], "context": gen["context"]["n_failed"]},
    }


# ---- the generation run (Ollama) ---------------------------------------------------------------

def ollama_info(base_url: str, model: str) -> dict[str, Any]:
    """Server version and the model's digest and size (best effort; None where the server does not say)."""
    info: dict[str, Any] = {"version": None, "digest": None, "details": None}
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/api/version", timeout=10) as resp:
            info["version"] = json.loads(resp.read().decode()).get("version")
        with urllib.request.urlopen(base_url.rstrip("/") + "/api/tags", timeout=10) as resp:
            for m in json.loads(resp.read().decode()).get("models", []):
                if m.get("name") == model or m.get("model") == model:
                    info["digest"], info["details"] = m.get("digest"), m.get("details")
    except Exception:
        pass
    return info


def main(argv: list[str] | None = None) -> int:
    import os

    from app.eval.corpus import load_manifest
    from app.eval.harness import git_sha
    from app.eval.judge import ollama_complete
    from app.eval.provenance import command_line, device_info

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--out", type=Path, default=Path(GENERATIONS_FILE))
    args = p.parse_args(argv)
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    info = ollama_info(base_url, args.model)
    if info["version"] is None:
        raise SystemExit(f"no Ollama server answers at OLLAMA_BASE_URL ({base_url}); start one first")
    gold = load_gold(ROOT / "bench" / "gold" / "questions.jsonl")
    chunks = fixture_chunks(ROOT / "bench" / "corpus" / "files")

    def complete(prompt: str) -> str | None:
        return ollama_complete(prompt, model=args.model, temperature=OPTIONS["temperature"])

    t0 = time.perf_counter()
    blocks = generate(complete, gold, chunks)
    blob = {
        "git_sha": git_sha(ROOT),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command_line(),
        "device": device_info(),
        "model": args.model,
        "ollama": info,
        "options": dict(OPTIONS),
        "corpus_sha256": load_manifest(ROOT / "bench" / "corpus" / "MANIFEST.json")["aggregate_sha256"],
        "n_gold": len(gold),
        "wall_s": round(time.perf_counter() - t0, 1),
        **blocks,
    }
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=2) + "\n")
    print(f"hyde: {blocks['hyde']['n']} prompts, {blocks['hyde']['n_failed']} failed; "
          f"context: {blocks['context']['n']} prompts, {blocks['context']['n_failed']} failed; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
