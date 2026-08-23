"""Corrective RAG (CRAG): post-retrieval quality evaluation + corrective actions.

Yan et al. 2024: after retrieving, classify each document as Correct /
Ambiguous / Incorrect.  If the retrieval is poor, trigger corrective
actions — query rewrite, knowledge refinement, or decomposition fallback.

This extends the existing grade_chunk() relevance check in retrieval_loop.py
with a three-tier classification and explicit corrective strategies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence

from app.retrieval.types import Chunk


class RelevanceLabel(Enum):
    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


@dataclass
class EvaluatedChunk:
    chunk: Chunk
    score: float
    label: RelevanceLabel
    reason: str = ""


@dataclass
class CorrectionAction:
    strategy: str  # "none" | "refine" | "rewrite" | "decompose"
    rewritten_query: str | None = None
    sub_queries: list[str] = field(default_factory=list)
    refined_chunks: list[Chunk] = field(default_factory=list)


def classify_chunk(
    query: str,
    chunk: Chunk,
    score: float,
    threshold_correct: float = 0.6,
    threshold_ambiguous: float = 0.3,
) -> EvaluatedChunk:
    """Classify a retrieved chunk's relevance to the query.

    Uses token overlap + structural signals. Swap for an LLM classifier
    when a GPU is available.
    """
    q_tokens = _extract_tokens(query)
    c_tokens = _extract_tokens(chunk.text)

    if not q_tokens:
        return EvaluatedChunk(chunk, score, RelevanceLabel.AMBIGUOUS, "empty query")

    overlap = len(q_tokens & c_tokens) / len(q_tokens)

    path_tokens = _extract_tokens(chunk.path)
    path_bonus = 0.15 if q_tokens & path_tokens else 0.0

    run_ids_q = set(re.findall(r"run[_\s]?(\d+)", query, re.I))
    run_ids_c = set(re.findall(r"run[_\s]?(\d+)", chunk.text, re.I))
    run_bonus = 0.2 if run_ids_q and run_ids_q & run_ids_c else 0.0

    relevance = overlap + path_bonus + run_bonus

    if relevance >= threshold_correct:
        label = RelevanceLabel.CORRECT
    elif relevance >= threshold_ambiguous:
        label = RelevanceLabel.AMBIGUOUS
    else:
        label = RelevanceLabel.INCORRECT

    return EvaluatedChunk(
        chunk, score, label,
        f"overlap={overlap:.2f} path={path_bonus:.2f} run={run_bonus:.2f}",
    )


def evaluate_retrieval(
    query: str,
    chunks: Sequence[tuple[Chunk, float]],
    threshold_correct: float = 0.6,
    threshold_ambiguous: float = 0.3,
) -> list[EvaluatedChunk]:
    """Classify all retrieved chunks."""
    return [
        classify_chunk(query, chunk, score, threshold_correct, threshold_ambiguous)
        for chunk, score in chunks
    ]


def determine_correction(
    query: str,
    evaluated: list[EvaluatedChunk],
    min_correct_ratio: float = 0.3,
) -> CorrectionAction:
    """Decide the corrective strategy based on retrieval quality.

    Strategies:
    - "none": enough correct chunks, proceed normally
    - "refine": some ambiguous chunks, extract key sentences only
    - "rewrite": mostly incorrect, rewrite the query
    - "decompose": complex query with poor retrieval, break it down
    """
    if not evaluated:
        return CorrectionAction(strategy="rewrite", rewritten_query=query)

    n_correct = sum(1 for e in evaluated if e.label == RelevanceLabel.CORRECT)
    n_ambiguous = sum(1 for e in evaluated if e.label == RelevanceLabel.AMBIGUOUS)
    n_incorrect = sum(1 for e in evaluated if e.label == RelevanceLabel.INCORRECT)
    total = len(evaluated)

    correct_ratio = n_correct / total

    if correct_ratio >= min_correct_ratio:
        return CorrectionAction(strategy="none")

    if n_ambiguous > n_incorrect:
        refined = _refine_ambiguous(query, evaluated)
        return CorrectionAction(strategy="refine", refined_chunks=refined)

    is_complex = (
        len(query.split()) > 8
        or " and " in query.lower()
        or any(w in query.lower() for w in ("compare", "which", "between"))
    )
    if is_complex:
        from app.agent.decompose import heuristic_decompose
        subs = heuristic_decompose(query)
        if len(subs) > 1:
            return CorrectionAction(strategy="decompose", sub_queries=subs)

    rewritten = _rewrite_query(query)
    return CorrectionAction(strategy="rewrite", rewritten_query=rewritten)


def _refine_ambiguous(
    query: str, evaluated: list[EvaluatedChunk]
) -> list[Chunk]:
    """Extract the most relevant sentences from ambiguous chunks."""
    q_tokens = _extract_tokens(query)
    refined: list[Chunk] = []
    for ec in evaluated:
        if ec.label != RelevanceLabel.AMBIGUOUS:
            if ec.label == RelevanceLabel.CORRECT:
                refined.append(ec.chunk)
            continue
        sentences = re.split(r"[.\n]+", ec.chunk.text)
        best_sents = []
        for sent in sentences:
            s_tokens = _extract_tokens(sent)
            if q_tokens & s_tokens:
                best_sents.append(sent.strip())
        if best_sents:
            refined_text = ". ".join(best_sents)
            refined.append(Chunk(
                chunk_id=ec.chunk.chunk_id + "_refined",
                path=ec.chunk.path,
                text=refined_text,
            ))
    return refined


def _rewrite_query(query: str) -> str:
    """Heuristic query rewrite: expand abbreviations, add context terms."""
    q = query
    q = re.sub(r"\bLR\b", "learning_rate", q, flags=re.I)
    q = re.sub(r"\bval\b", "validation", q, flags=re.I)
    q = re.sub(r"\bconfig\b", "configuration yaml", q, flags=re.I)

    run_match = re.search(r"run[_\s]?(\d+)", q, re.I)
    if run_match and "config" not in q.lower() and "yaml" not in q.lower():
        q += f" configs/run_{run_match.group(1).zfill(3)}.yaml"

    return q


def _extract_tokens(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[A-Za-z0-9_]+", text or "") if len(t) > 1}
