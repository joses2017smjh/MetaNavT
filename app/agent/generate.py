"""Cited answers from a local LLM (Ollama), with the citation contract enforced.

The generator receives the query and the top-k retrieved chunks, shows each
chunk as a numbered source tagged `[path:start-end]` (the bench's byte-range
citation), and asks the model to answer only from the sources and to cite every
claim with a tag. parse_citations() reads the tags back; cited_answer() checks
them against the retrieved set. An answer without a valid citation is failed
loudly by the caller (RetrievalAgent / app.eval.llm_eval), never served as if
it were grounded.

    gen = OllamaGenerator(model="qwen2.5:7b")
    text = gen(query, hits)             # raw model text
    info = cited_answer(text, hits)     # {"citations": [...], "uncited": bool, ...}
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Sequence

from app.retrieval.types import RetrievalHit

CITE_RE = re.compile(r"\[([^\[\]\s]+?):(\d+)-(\d+)\]")
NOT_IN_SOURCES = "NOT IN SOURCES"
DEFAULT_MODEL = "qwen2.5:7b"


def source_tag(hit: RetrievalHit) -> str:
    return f"[{hit.chunk.path}:{hit.chunk.start_byte}-{hit.chunk.end_byte}]"


def sources_block(hits: Sequence[RetrievalHit], n: int = 8, max_chars: int = 1200) -> str:
    lines = []
    for i, hit in enumerate(hits[:n], start=1):
        text = hit.chunk.text.strip()
        if len(text) > max_chars:
            text = text[:max_chars] + " ..."
        lines.append(f"Source {i} {source_tag(hit)}\n{text}")
    return "\n\n".join(lines)


def build_prompt(query: str, hits: Sequence[RetrievalHit], n: int = 8) -> str:
    return (
        "You answer questions about a research project's files using ONLY the sources below.\n"
        "Rules:\n"
        "1. Every sentence that states a fact must end with the tag of the source it comes from, "
        "copied exactly, e.g. [configs/run_047.yaml:0-255].\n"
        "2. Copy numbers, names and paths exactly as they appear in the sources.\n"
        f"3. If the sources do not contain the answer, reply exactly: {NOT_IN_SOURCES}\n"
        "4. Be brief: one to three sentences.\n\n"
        f"{sources_block(hits, n=n)}\n\n"
        f"Question: {query}\nAnswer:"
    )


@dataclass
class ParsedCitation:
    path: str
    start_byte: int
    end_byte: int

    def as_dict(self) -> dict:
        return {"path": self.path, "start_byte": self.start_byte, "end_byte": self.end_byte}


def parse_citations(text: str) -> list[ParsedCitation]:
    out: list[ParsedCitation] = []
    seen = set()
    for m in CITE_RE.finditer(text or ""):
        key = (m.group(1), int(m.group(2)), int(m.group(3)))
        if key not in seen:
            seen.add(key)
            out.append(ParsedCitation(*key))
    return out


def strip_citations(text: str) -> str:
    return re.sub(r"\s*\[[^\[\]\s]+?:\d+-\d+\]", "", text or "").strip()


def cited_answer(text: str, hits: Sequence[RetrievalHit]) -> dict:
    """Check the model's citations against the retrieved chunks.

    A citation is valid when its path is a retrieved path and its byte range is
    one of that path's retrieved chunks (the tag the prompt showed). `uncited`
    is true when a substantive answer carries no valid citation; a NOT IN
    SOURCES reply is an abstention, not an uncited answer.
    """
    parsed = parse_citations(text)
    retrieved = {(h.chunk.path, h.chunk.start_byte, h.chunk.end_byte) for h in hits}
    retrieved_paths = {h.chunk.path for h in hits}
    valid = [c for c in parsed if (c.path, c.start_byte, c.end_byte) in retrieved]
    unknown = [c for c in parsed if (c.path, c.start_byte, c.end_byte) not in retrieved]
    abstained = NOT_IN_SOURCES in (text or "")
    return {
        "citations": [c.as_dict() for c in valid],
        "unknown_citations": [c.as_dict() for c in unknown],
        "cited_paths_in_retrieved": all(c.path in retrieved_paths for c in parsed) if parsed else False,
        "abstained": abstained,
        "uncited": (not valid) and not abstained,
        "answer": strip_citations(text),
    }


@dataclass
class GenerationMeta:
    model: str
    seconds: float
    eval_count: int | None = None
    prompt_eval_count: int | None = None
    load_seconds: float | None = None


class OllamaGenerator:
    """Callable (query, hits) -> raw answer text, via Ollama's /api/generate."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        *,
        temperature: float = 0.0,
        keep_alive: str = "30m",
        timeout: float = 600.0,
        n_sources: int = 8,
    ):
        self.model = model or os.environ.get("GENERATOR_MODEL", DEFAULT_MODEL)
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.temperature = temperature
        self.keep_alive = keep_alive
        self.timeout = timeout
        self.n_sources = n_sources
        self.last_meta: GenerationMeta | None = None
        self.history: list[GenerationMeta] = []
        self._last_payload: dict | None = None

    def complete(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": self.keep_alive,
                # No num_ctx: every caller must use the server's default context, because
                # Ollama reloads the model (about 50 s) whenever the requested size changes.
                "options": {"temperature": self.temperature, "seed": 0},
            }
        ).encode()
        req = urllib.request.Request(f"{self.base_url}/api/generate", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode())
        self._last_payload = payload
        return (payload.get("response") or "").strip()

    def __call__(self, query: str, hits: Sequence[RetrievalHit]) -> str:
        self._last_payload = None
        t0 = time.perf_counter()
        text = self.complete(build_prompt(query, hits, n=self.n_sources))
        payload = self._last_payload or {}
        meta = GenerationMeta(
            model=self.model,
            seconds=round(time.perf_counter() - t0, 3),
            eval_count=payload.get("eval_count"),
            prompt_eval_count=payload.get("prompt_eval_count"),
            load_seconds=round(payload.get("load_duration", 0) / 1e9, 3) if payload.get("load_duration") else None,
        )
        self.last_meta = meta
        self.history.append(meta)
        return text


def ollama_available(base_url: str | None = None, timeout: float = 5.0) -> list[str]:
    """Names of the models the Ollama server has, or [] when it is not reachable."""
    base = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
        return [m.get("name") for m in payload.get("models", []) if m.get("name")]
    except Exception:
        return []
