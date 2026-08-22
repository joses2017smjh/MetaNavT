"""Contextual retrieval (Anthropic): prepend a short situating prefix before embed.

Cost: one small-LLM call per chunk at index time. The prefix is the only
index-time mutation; retrieval is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.chunking.structure import Span

DEFAULT_PROMPT = """Write a 50-100 token description of what this chunk is and where it sits
in the document. Do not answer questions; just situate the chunk.

Document path: {path}
Document excerpt (first 400 chars): {doc_head}

Chunk:
{chunk}

Situated description:"""


@dataclass
class ContextualizedChunk:
    span: Span
    context: str
    embedded_text: str


def prepend_context(chunk_text: str, context: str) -> str:
    context = (context or "").strip()
    if not context:
        return chunk_text
    return f"{context}\n\n{chunk_text}"


def situate_chunk(
    path: str,
    doc_text: str,
    span: Span,
    llm: Callable[[str], str],
    prompt_template: str = DEFAULT_PROMPT,
) -> ContextualizedChunk:
    prompt = prompt_template.format(
        path=path,
        doc_head=(doc_text or "")[:400],
        chunk=span.text[:1500],
    )
    context = (llm(prompt) or "").strip()
    return ContextualizedChunk(
        span=span,
        context=context,
        embedded_text=prepend_context(span.text, context),
    )
