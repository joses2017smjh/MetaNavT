"""Balanced synthetic structured-evidence and literal byte-scope regression.

Labels follow explicit fixture construction, not invented human annotation or
an LLM/NLI judge. Results quantify only this narrow JSON-fact contract.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import time

from app.agent.evidence_checks import StructuredFact, verify_structured_claim
from app.retrieval.types import Chunk
from app.eval.m4_provenance import provenance

FAMILIES = {
    "supported": ("numeric_equal", "unit_conversion", "numeric_notation", "boolean_equal", "text_equal", "same_negation", "current_evidence", "unicode_raw_evidence"),
    "conflict": ("numeric_mismatch", "unit_dimension", "opposite_negation", "superseded_current_claim", "contradictory_sources", "boolean_mismatch", "text_mismatch", "unequal_converted_value"),
    "unknown": ("different_subject", "different_field", "unsupported_unit", "currency_missing", "invalid_json", "invalid_citation_range", "transformed_without_raw", "nonfinite_value"),
}


def evidence_case(label, family, variant):
    fact = StructuredFact(f"robot_{variant}", "latency", str(variant + 1), "s", current=True)
    claim = fact
    facts = [fact]
    raw_override = None
    metadata = {}
    ranged = True
    if family == "unit_conversion":
        claim = replace(fact, value=str((variant+1)*1000), unit="ms")
    elif family == "numeric_notation":
        claim = replace(fact, value=f"{variant+1}e0")
    elif family in {"boolean_equal", "boolean_mismatch"}:
        fact = replace(fact, field="enabled", value=bool(variant % 2), unit="")
        facts = [fact]
        claim = fact if family == "boolean_equal" else replace(fact, value=not fact.value)
    elif family in {"text_equal", "text_mismatch"}:
        fact = replace(fact, field="encoder", value=f"encoder_{variant}", unit="")
        facts = [fact]
        claim = fact if family == "text_equal" else replace(fact, value=f"different_encoder_{variant}")
    elif family == "same_negation":
        facts = [replace(fact, negated=True)]
        claim = facts[0]
    elif family == "unicode_raw_evidence":
        facts = [replace(fact, subject=f"röbot_α_{variant}")]
        claim = facts[0]
        metadata["text_is_verbatim"] = False
    elif family == "numeric_mismatch":
        claim = replace(fact, value=str(variant+2))
    elif family == "unit_dimension":
        claim = replace(fact, unit="m")
    elif family == "opposite_negation":
        claim = replace(fact, negated=True)
    elif family == "superseded_current_claim":
        facts = [replace(fact, current=False)]
    elif family == "contradictory_sources":
        facts.append(replace(fact, value=str(variant+2)))
    elif family == "unequal_converted_value":
        claim = replace(fact, value=str((variant+1)*1000+1), unit="ms")
    elif family == "different_subject":
        claim = replace(fact, subject=f"different_robot_{variant}")
    elif family == "different_field":
        claim = replace(fact, field="different_metric")
    elif family == "unsupported_unit":
        claim = replace(fact, unit="furlongs_per_fortnight")
    elif family == "currency_missing":
        facts = [replace(fact, current=None)]
    elif family == "invalid_json":
        raw_override = f"The robot's latency is {variant+1} s."
    elif family == "transformed_without_raw":
        metadata["text_is_verbatim"] = False
    elif family == "nonfinite_value":
        claim = replace(fact, value="NaN")
    elif family not in {"numeric_equal", "current_evidence", "invalid_citation_range"}:
        raise ValueError(family)
    raw = raw_override or json.dumps({"facts": [asdict(item) for item in facts]}, ensure_ascii=False)
    # Byte coordinates begin after a Unicode prefix in the original file.
    start = len("α-calibration\n".encode("utf-8"))
    end = start + len(raw.encode("utf-8"))
    if family == "unicode_raw_evidence":
        metadata["evidence_raw"] = raw
    text = "synthetic search header\n" + raw if metadata else raw
    chunk = Chunk(f"fixture_{variant}", f"facts/robot_{variant}.json", text, start, end, metadata=metadata)
    citation = {"path": chunk.path, "start_byte": start, "end_byte": end}
    if family == "invalid_citation_range":
        citation["end_byte"] += variant+1
    result = verify_structured_claim(claim, [chunk], [citation])
    return {"family": family, "variant": variant, "expected": label, "actual": result.label, "reason": result.reason, "passed": result.label == label, "fixture_sha256": hashlib.sha256(raw.encode()).hexdigest()}


def run(variants=10):
    started = time.time()
    rows = [evidence_case(label, family, variant) for label, families in FAMILIES.items() for family in families for variant in range(variants)]
    confusion = {label: {pred: sum(row["expected"] == label and row["actual"] == pred for row in rows) for pred in FAMILIES} for label in FAMILIES}
    return {**provenance(), "benchmark": "structured_evidence_consistency_v1", "scope": "balanced synthetic JSON-fact fixtures; deterministic subject/field/value/unit/polarity/source-currency contract, not natural-language entailment or human-annotated citation correctness", "label_origin": "programmatic fixture construction", "cases": len(rows), "passed": sum(row["passed"] for row in rows), "families": sum(map(len, FAMILIES.values())), "variants_per_family": variants, "class_counts": dict(Counter(row["expected"] for row in rows)), "confusion_matrix": confusion, "failed_cases": [row for row in rows if not row["passed"]], "elapsed_seconds": round(time.time()-started, 4), "python": platform.python_version(), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("bench/results/evidence_consistency.json"))
    parser.add_argument("--variants", type=int, default=10)
    args = parser.parse_args()
    if args.variants < 1:
        parser.error("variants must be positive")
    result = run(args.variants)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    return bool(result["failed_cases"])


if __name__ == "__main__":
    raise SystemExit(main())
