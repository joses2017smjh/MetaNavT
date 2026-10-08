"""Deterministic checks for explicit structured facts, not natural-language NLI.

Facts are read from raw, cited JSON evidence, never invented from token overlap.
Supported schema: {"facts": [{"subject": ..., "field": ..., "value": ...,
"unit": ..., "negated": false, "current": true}]}. Missing subject/field,
unknown units, absent currency metadata and contradictory sources fail closed.
This deliberately narrow contract does not parse or entail arbitrary prose.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from decimal import Decimal, InvalidOperation
import json
from typing import Any, Sequence

from app.agent.citation_verify import cited_chunks
from app.retrieval.types import Chunk

# Canonical dimensions/factors; unknown units are never silently stripped.
_UNITS = {"": ("scalar", Decimal(1)), "1": ("scalar", Decimal(1)),
          "%": ("scalar", Decimal("0.01")),
          "m": ("length", Decimal(1)), "cm": ("length", Decimal("0.01")),
          "mm": ("length", Decimal("0.001")),
          "s": ("time", Decimal(1)), "ms": ("time", Decimal("0.001")),
          "us": ("time", Decimal("0.000001")), "µs": ("time", Decimal("0.000001")),
          "rad": ("angle", Decimal(1)), "N": ("force", Decimal(1)),
          "Hz": ("frequency", Decimal(1)), "kHz": ("frequency", Decimal(1000))}


@dataclass(frozen=True)
class StructuredFact:
    subject: str
    field: str
    value: str | int | float | bool
    unit: str = ""
    negated: bool = False
    current: bool | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "StructuredFact":
        if not isinstance(data, dict) or set(data) - {"subject", "field", "value", "unit", "negated", "current"}:
            raise ValueError("invalid structured fact fields")
        fact = cls(**data)
        if not isinstance(fact.subject, str) or not fact.subject or not isinstance(fact.field, str) or not fact.field:
            raise ValueError("fact needs an explicit subject and field")
        if type(fact.value) not in (str, int, float, bool) or not isinstance(fact.unit, str) or type(fact.negated) is not bool:
            raise ValueError("invalid value, unit or polarity")
        if fact.current is not None and type(fact.current) is not bool:
            raise ValueError("current must be boolean or unknown")
        return fact


@dataclass(frozen=True)
class FactCheck:
    label: str  # supported | conflict | unknown
    reason: str
    source_paths: tuple[str, ...] = ()

    def as_dict(self):
        return asdict(self)


def _quantity(fact: StructuredFact):
    if type(fact.value) is bool:
        return ("boolean", fact.value) if not fact.unit else None
    try:
        number = Decimal(str(fact.value))
    except InvalidOperation:
        return ("text", fact.value) if isinstance(fact.value, str) and not fact.unit else None
    unit = _UNITS.get(fact.unit)
    if not number.is_finite() or unit is None:
        return None
    return unit[0], number * unit[1]


def compare_fact(claim: StructuredFact, fact: StructuredFact) -> FactCheck:
    """Compare exact fields, physical dimensions, value, polarity and currency."""
    if (claim.subject, claim.field) != (fact.subject, fact.field):
        return FactCheck("unknown", "different_subject_or_field")
    if claim.current is True:
        if fact.current is None:
            return FactCheck("unknown", "currency_unknown")
        if fact.current is False:
            return FactCheck("conflict", "superseded_evidence")
    lhs, rhs = _quantity(claim), _quantity(fact)
    if lhs is None or rhs is None:
        return FactCheck("unknown", "unknown_unit_or_nonfinite_value")
    if lhs[0] != rhs[0]:
        return FactCheck("conflict", "unit_dimension_mismatch")
    if lhs[1] != rhs[1]:
        # "not 5" cannot establish or contradict "not 6". A negative
        # observation also cannot establish an unrelated positive value.
        if claim.negated or fact.negated:
            return FactCheck("unknown", "different_value_with_negation")
        return FactCheck("conflict", "value_mismatch")
    if claim.negated != fact.negated:
        return FactCheck("conflict", "polarity_mismatch")
    return FactCheck("supported", "structured_fields_agree")


def verify_structured_claim(claim: StructuredFact, evidence: Sequence[Chunk],
                            citations: Sequence[dict | tuple | str]) -> FactCheck:
    """Read explicit fact objects from raw cited JSON, then compare the claim.

    The entire JSON value must fit the cited byte range. Invalid/non-JSON prose
    yields unknown. Evidence currency remains source-supplied metadata; this
    does not discover publication dates or prove an asserted `current` label.
    """
    matches: list[tuple[FactCheck, str]] = []
    for chunk in cited_chunks(evidence, citations):
        try:
            document = json.loads(chunk.text)
            rows = document.get("facts") if isinstance(document, dict) else None
            if not isinstance(rows, list):
                continue
            facts = [StructuredFact.from_dict(row) for row in rows]
        except (ValueError, TypeError, KeyError):
            continue
        for fact in facts:
            if (claim.subject, claim.field) == (fact.subject, fact.field):
                matches.append((compare_fact(claim, fact), chunk.path))
    if not matches:
        return FactCheck("unknown", "no_valid_cited_structured_fact")
    paths = tuple(sorted({path for _, path in matches}))
    labels = {result.label for result, _ in matches}
    if "supported" in labels and "conflict" in labels:
        return FactCheck("conflict", "contradictory_cited_facts", paths)
    if "conflict" in labels:
        return FactCheck("conflict", next(result.reason for result, _ in matches if result.label == "conflict"), paths)
    if labels == {"supported"}:
        return FactCheck("supported", "structured_fields_agree", paths)
    return FactCheck("unknown", "incomplete_structured_evidence", paths)
