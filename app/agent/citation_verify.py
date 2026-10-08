"""Literal citation/value checks, not semantic entailment or fact verification.

Strict mode checks only the exact cited UTF-8 byte range. Structure-aware chunks
may include synthetic search headers; their raw source span is used instead.
Matching a number or token does not establish its subject, unit, polarity or
currency. Use evidence_checks for explicit structured fact comparisons.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Sequence

from app.retrieval.types import Chunk


@dataclass
class Claim:
    text: str
    value: str | None = None
    cited_path: str | None = None
    verified: bool = False
    evidence_snippet: str = ""


@dataclass
class VerificationResult:
    claims: list[Claim]
    verified_count: int
    total_count: int
    missing_citations: list[str]
    hallucinated_values: list[str]
    pass_threshold: bool

    @property
    def verification_ratio(self) -> float:
        return self.verified_count / self.total_count if self.total_count > 0 else 0.0

    def as_dict(self) -> dict:
        return {
            "verified": self.verified_count,
            "total": self.total_count,
            "ratio": round(self.verification_ratio, 4),
            "pass": self.pass_threshold,
            "missing_citations": self.missing_citations,
            "hallucinated_values": self.hallucinated_values,
            "claims": [
                {
                    "text": c.text,
                    "value": c.value,
                    "cited_path": c.cited_path,
                    "verified": c.verified,
                }
                for c in self.claims
            ],
        }


def extract_claims(answer: str) -> list[Claim]:
    """Extract factual claims from an answer text.

    A claim is a sentence or fragment that asserts a specific value,
    name, path, or configuration detail.
    """
    claims: list[Claim] = []
    # Sentence boundaries only: a period followed by whitespace or a newline. Splitting on
    # every period cut decimals like 0.055 in half, so those values were never checked.
    sentences = re.split(r"(?<=[.!?])\s+|\n+", answer or "")

    for sent in sentences:
        sent = sent.strip()
        if not sent or len(sent) < 10:
            continue

        values = _extract_values(sent)
        paths = re.findall(r"[\w/]+\.(?:yaml|py|csv|sbatch|out|md|json|txt)", sent)

        if values or paths:
            for val in values:
                claims.append(Claim(
                    text=sent,
                    value=val,
                    cited_path=paths[0] if paths else None,
                ))
            if paths and not values:
                claims.append(Claim(
                    text=sent,
                    cited_path=paths[0],
                ))
        elif _has_factual_assertion(sent):
            claims.append(Claim(text=sent))

    return claims


def cited_chunks(
    evidence: Sequence[Chunk],
    citations: Sequence[dict | tuple | str] | None,
) -> list[Chunk]:
    """Return raw evidence sliced to the requested range, never its whole chunk.

    Missing/invalid ranges, UTF-8 mid-character boundaries and transformed
    chunks without a raw span fail closed. A bare path selects its raw chunks.
    """
    keep: list[Chunk] = []
    for cit in citations or []:
        if isinstance(cit, str):
            path, start, end = cit, None, None
        elif isinstance(cit, dict):
            path, start, end = cit.get("path"), cit.get("start_byte"), cit.get("end_byte")
        elif isinstance(cit, (tuple, list)):
            path, start, end = (list(cit) + [None, None])[:3]
        else:
            continue
        ranged = start is not None or end is not None
        if ranged and (type(start) is not int or type(end) is not int or not 0 <= start < end):
            continue
        for chunk in evidence:
            if chunk.path != path:
                continue
            raw = chunk.metadata.get("evidence_raw")
            if raw is None:
                if chunk.metadata.get("text_is_verbatim") is False:
                    continue
                raw = chunk.text
            if not isinstance(raw, str):
                continue
            lo, hi = chunk.start_byte, chunk.end_byte
            data = raw.encode("utf-8")
            if ranged:
                if len(data) != hi - lo or not lo <= start < end <= hi:
                    continue
                try:
                    raw = data[start-lo:end-lo].decode("utf-8", errors="strict")
                except UnicodeDecodeError:
                    continue
                lo, hi = start, end
            sliced = replace(chunk, text=raw, start_byte=lo, end_byte=hi)
            if sliced not in keep:
                keep.append(sliced)
    return keep


def _literal_found(value: str, text: str) -> bool:
    # A cited 0.0559 or run 147 must not support the literal 0.055 or 47.
    return bool(re.search(r"(?<![\w.])" + re.escape(value) + r"(?![\w.])", text, re.I))


def verify_claims(
    claims: list[Claim],
    evidence: Sequence[Chunk],
    cited_paths: Sequence[str] | None = None,
    threshold: float = 0.5,
    *,
    citations: Sequence[dict | tuple | str] | None = None,
    cited_only: bool = False,
) -> VerificationResult:
    """Verify each claim against the retrieved evidence chunks.

    cited_only=True restricts value checks to the chunks named in `citations`
    (falling back to `cited_paths`): a number that is in the evidence but not in
    the cited bytes is reported as hallucinated_values, because the citation
    does not literally contain it. This is not an entailment check.
    """
    evidence_paths = {c.path for c in evidence}
    cited_set = set(cited_paths or [])
    if cited_only:
        scope = cited_chunks(evidence, citations if citations is not None else list(cited_set))
    else:
        scope = list(evidence)
    evidence_text = " ".join(c.text for c in scope)

    missing_citations: list[str] = []
    hallucinated_values: list[str] = []
    verified_count = 0

    for claim in claims:
        claim.verified = False
        claim.evidence_snippet = ""
        claim_scope = ([chunk for chunk in scope if chunk.path == claim.cited_path]
                       if cited_only and claim.cited_path else scope)
        claim_evidence = " ".join(chunk.text for chunk in claim_scope)
        if claim.cited_path:
            if claim.cited_path not in evidence_paths:
                missing_citations.append(claim.cited_path)
                claim.verified = False
                continue

        if claim.value:
            if _literal_found(claim.value, claim_evidence):
                claim.verified = True
                for chunk in claim_scope:
                    if _literal_found(claim.value, chunk.text):
                        claim.evidence_snippet = chunk.text[:200]
                        break
            else:
                claim.verified = False
                hallucinated_values.append(claim.value)
                continue
        elif claim.cited_path and claim.cited_path in {chunk.path for chunk in claim_scope}:
            claim.verified = True
        elif _fuzzy_match_claim(claim.text, claim_evidence):
            claim.verified = True
        else:
            claim.verified = False

        if claim.verified:
            verified_count += 1

    total = len(claims)
    return VerificationResult(
        claims=claims,
        verified_count=verified_count,
        total_count=total,
        missing_citations=missing_citations,
        hallucinated_values=hallucinated_values,
        pass_threshold=verified_count >= total * threshold if total > 0 else True,
    )


def _extract_values(text: str) -> list[str]:
    """Extract specific values (numbers, configs, identifiers) from text.

    Citation tags ([path:start-end]) are removed first so their byte offsets are
    not mistaken for claimed values.
    """
    text = re.sub(r"\[[^\[\]\s]+?:\d+-\d+\]", " ", text or "")
    values: list[str] = []
    values.extend(re.findall(r"\b\d+(?:\.\d+)?[eE][+-]?\d+\b", text))
    values.extend(re.findall(r"\b0\.\d+\b", text))
    values.extend(re.findall(r"\b\d+\.\d+\b", text))
    values.extend(m.group(1) for m in re.finditer(r"\brun[_\s]?(\d+)\b", text, re.I))  # the run number, not "Run 47" literally
    values.extend(re.findall(r"\b(?:dinov2|resnet\d+|clip|vit\w*)\b", text, re.I))
    for m in re.finditer(r"(\d+)\s*(?:epochs?|iterations?|steps?|pairs?)", text, re.I):
        values.append(m.group(1))
    return list(set(values))


def _has_factual_assertion(text: str) -> bool:
    """Check if a sentence makes a factual assertion (not just filler)."""
    assertion_patterns = [
        r"\b(?:is|are|was|were|has|had|uses?|set\s+to|equals?|=)\b",
        r"\b(?:achieved|produced|resulted|scored|measured)\b",
        r"\b\d+\b",
    ]
    return any(re.search(p, text, re.I) for p in assertion_patterns)


def _fuzzy_match_claim(claim_text: str, evidence_text: str) -> bool:
    """Check if the key tokens from a claim appear in the evidence."""
    claim_tokens = {
        t.lower()
        for t in re.findall(r"[A-Za-z0-9_]+", claim_text)
        if len(t) > 2 and t.lower() not in _FILLER
    }
    if not claim_tokens:
        return True
    evidence_lower = evidence_text.lower()
    matches = sum(1 for t in claim_tokens if t in evidence_lower)
    return matches / len(claim_tokens) >= 0.6 if claim_tokens else True


_FILLER = frozenset({
    "the", "and", "for", "that", "this", "with", "from", "are", "was",
    "were", "has", "had", "been", "have", "not", "but", "also", "its",
    "which", "about", "into", "than", "can", "will", "would", "could",
    "should", "does", "did", "may", "might",
})
