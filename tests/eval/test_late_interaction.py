"""M49 (app/eval/late_interaction.py): what was declared before any late-interaction row was measured.

No torch here. Every pinned constant is pinned by value (a sha256 of its canonical JSON) and the important parts are
asserted in words too, so a bar, a row list or a threshold that changed after a number existed would fail here. The
fix commit of an ATTEMPT 2 may change ATTEMPT only: this file is then untouched.
"""

import ast
import hashlib
import json
import re
from pathlib import Path

from app.eval import late_interaction as li

ROOT = Path(__file__).resolve().parents[2]
# sha256 of json.dumps(li.pinned(), sort_keys=True, separators=(",", ":"), ensure_ascii=True), written with the
# declarations and never edited afterwards.
PINNED_SHA256 = "19fdce5cd65881636c50b25d7332a07776fc73c9722d8996b0d2c25bc5883a75"


def _digest() -> str:
    return hashlib.sha256(json.dumps(li.pinned(), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def test_every_pinned_constant_is_the_declared_one():
    assert set(li.PINNED) == {"CANDIDATES", "STRETCH", "PUBLISHED", "FINGERPRINT", "SETTINGS", "SENSITIVITY_LENGTH", "MAXSIM_BLOCK_BYTES",
                              "BAR", "MULTIPLICITY", "CARD_CLAIM", "ROWS", "PLAN_FLAGS", "TRIPWIRE", "PERMISSIVE_LICENCES", "CARD_URL"}
    assert _digest() == PINNED_SHA256


def test_the_attempt_is_the_first_or_the_single_defect_driven_second():
    assert li.ATTEMPT in (1, 2)


def test_the_constants_are_plain_literals_that_a_later_commit_can_read_back():
    """`make late-interaction-bar` reads BAR, SETTINGS and CANDIDATES out of the declarations commit with ast.literal_eval."""
    tree = ast.parse((ROOT / "app" / "eval" / "late_interaction.py").read_text())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in li.PINNED:
            found[node.targets[0].id] = ast.literal_eval(node.value)
    assert found == li.pinned()


def test_the_named_models_and_the_stretch_models_are_permissive_and_pinned():
    assert list(li.CANDIDATES) == ["answerai-colbert-small", "colbertv2"]
    assert list(li.STRETCH) == ["gte-moderncolbert", "mxbai-edge-colbert-17m"]
    for tag, c in {**li.CANDIDATES, **li.STRETCH}.items():
        assert c["licence"] in li.PERMISSIVE_LICENCES, tag
        assert re.fullmatch(r"[0-9a-f]{40}", c["revision"]), tag
        assert re.fullmatch(r"[0-9a-f]{64}", c["weights_sha256"]) and re.fullmatch(r"[0-9a-f]{64}", c["readme_sha256"]), tag
        assert c["weights_bytes"] < 1024 ** 3, tag  # COMMON rule 6: no model over 1 GB
    assert li.CANDIDATES["answerai-colbert-small"]["model"] == "answerdotai/answerai-colbert-small-v1"
    assert li.CANDIDATES["colbertv2"]["model"] == "colbert-ir/colbertv2.0"
    saved = {tag: c["saved"] for tag, c in li.CANDIDATES.items()}
    assert (saved["answerai-colbert-small"]["dim"], saved["answerai-colbert-small"]["document_length"]) == (96, 300)
    assert (saved["colbertv2"]["dim"], saved["colbertv2"]["document_length"]) == (128, 180)
    for s in saved.values():  # both named models: [MASK] expansion to 32, not attended, punctuation skiplist
        assert (s["query_length"], s["query_expansion"], s["expansion_attended"], s["skiplist"]) == (32, True, False, "punctuation")
    assert li.SENSITIVITY_LENGTH == li.SETTINGS["sensitivity_document_length"] == 512
    assert li.MAXSIM_BLOCK_BYTES == li.SETTINGS["maxsim_block_bytes"] == 1 << 30
    assert (li.SETTINGS["precision"], li.SETTINGS["batch_size"], li.SETTINGS["maxsim_dtype"]) == ("fp32", 32, "float64")


def test_the_published_figures_and_the_fingerprint_name_the_pinned_card():
    card = li.CANDIDATES["answerai-colbert-small"]
    for block in (li.PUBLISHED["card"], li.FINGERPRINT["answerai-colbert-small"]["card"]):
        assert block["url"] == li.CARD_URL and block["revision"] == card["revision"] and block["readme_sha256"] == card["readme_sha256"]
    for dataset, line in li.PUBLISHED["lines"].items():
        for tag in li.CANDIDATES:
            assert li.PUBLISHED["values"][tag][dataset] in line, (tag, dataset)
    assert "colbertv2" not in li.FINGERPRINT  # its card prints no scores
    for tag, f in li.FINGERPRINT.items():
        source = {**li.CANDIDATES, **li.STRETCH}[tag]
        assert f["card"]["revision"] == source["revision"] and f["card"]["readme_sha256"] == source["readme_sha256"]
        assert len(f["documents"]) == len(f["scores"]) == 4
        assert f"({f['query_shape'][0]}, {f['query_shape'][1]}) ({f['document_0_shape'][0]}, {f['document_0_shape'][1]})" in f["shapes_line"]
        assert ", ".join(f["scores"]) in f["scores_line"]
        assert f["query_shape"][1] == source["saved"]["dim"]


def test_the_rows_give_the_interval_counts_the_multiplicity_statement_names():
    counts = li.MULTIPLICITY["counts"]
    for dataset in ("fixture", "beir"):
        parts = li.ROWS[dataset]
        names = [r["name"] for part in parts.values() for r in part]
        for part in parts.values():
            assert [r["name"] for r in part[:2]] == [("dense_only" if dataset == "fixture" else "dense") + "@bge-small", "hybrid@bge-small"]
            assert all(r["parent"] is None for r in part[:2]) and all(r["parent"] for r in part[2:])
            assert len({r["name"] for r in part}) == len(part)
        primary, extra = parts["primary"], parts["extra"]
        bar = [r for r in primary if r["kind"] == "bar"]
        assert len(bar) == counts["bar_per_dataset"] == 4 and {r["parent"] for r in bar} == {"hybrid@bge-small"}
        other_primary = sum(1 for r in primary if r["parent"] and r["kind"] != "bar") + sum(len(r.get("references", [])) for r in primary)
        assert other_primary == counts["other_primary_per_dataset"] == 11
        assert sum(1 for r in extra if r["parent"]) == counts["other_extra_per_dataset"] == 12
        assert not any(r["kind"] == "bar" or r.get("references") for r in extra)
        assert not [n for n in names if n.startswith("bm25")]  # no bm25 row: no delta against it
    assert counts["bar_total"] == 3 * counts["bar_per_dataset"] == 12
    assert counts["other_total"] == 3 * (counts["other_primary_per_dataset"] + counts["other_extra_per_dataset"]) == 69
    assert li.MULTIPLICITY["bonferroni"]["alpha"] == 0.05 / 4 and li.MULTIPLICITY["bonferroni"]["deciding"] is False


def test_the_row_names_follow_the_declared_patterns():
    for dataset, dense in (("fixture", "dense_only"), ("beir", "dense")):
        for part, rows in li.ROWS[dataset].items():
            for r in rows:
                if r["role"] == "b":
                    assert r["name"] == f"hybrid+{r['model']}@bge-small/fp32-doc{r['document_length']}/top20"
                elif r["role"] == "a":
                    suffix = f"/doc{li.SENSITIVITY_LENGTH}" if r["kind"] == "sensitivity" else ""
                    assert r["name"] in (f"{dense}@{r['model']}{suffix}", f"hybrid@{r['model']}{suffix}")
                    saved = {**li.CANDIDATES, **li.STRETCH}[r["model"]]["saved"]["document_length"]
                    assert r["document_length"] == (li.SENSITIVITY_LENGTH if r["kind"] == "sensitivity" else saved)
                if r["kind"] == "stretch":
                    assert r["model"] in li.STRETCH and part == "extra"
                if r["kind"] in ("bar", "secondary"):
                    assert r["model"] in li.CANDIDATES and part == "primary"
    assert li.ROWS["fixture"]["primary"][-3:] == li.ROWS["beir"]["primary"][-3:]  # the same three references, by name


def test_the_eighteen_plan_flagged_claims_are_the_bar_and_the_gte_comparisons():
    flags = [tuple(f) for f in li.PLAN_FLAGS]
    assert len(flags) == len(set(flags)) == 18
    gte = "hybrid+gte-modernbert-base@bge-small/fp32-512/top20"
    for tag, c in li.CANDIDATES.items():
        doc = f"hybrid+{tag}@bge-small/fp32-doc{c['saved']['document_length']}/top20"
        for d in ("fixture", "scifact", "fiqa"):
            for pair in ((f"{d}:hybrid@{tag}", f"{d}:hybrid@bge-small"), (f"{d}:{doc}", f"{d}:hybrid@bge-small"), (f"{d}:{doc}", f"{d}:{gte}")):
                assert pair in flags, pair
    bar_rows = {r["name"] for r in li.ROWS["beir"]["primary"] if r["kind"] == "bar"}
    assert {f[0].split(":", 1)[1] for f in flags} == bar_rows


def test_the_bar_and_the_tripwire_say_what_was_declared():
    assert li.BAR["metric"] == "ndcg@10" and li.BAR["parent"] == "hybrid@bge-small"
    assert "2000 resamples, seed 0, alpha 0.05" in li.BAR["interval"]
    assert "exactly 0.0000" in li.BAR["words"]
    assert set(li.BAR["datasets"]) == {"fixture", "scifact", "fiqa"} and "held-out" in li.BAR["datasets"]["fixture"]
    assert li.BAR["named_models"] == list(li.CANDIDATES)
    assert "OPEN is never a pass" in li.BAR["open"]
    assert li.TRIPWIRE["low"]["below_published_by"] == li.TRIPWIRE["high"]["above_published_by"] == 0.05
    assert "beir_scifact.json" in li.TRIPWIRE["low"]["below_bm25"]
    assert len(li.TRIPWIRE["checklist"]) == 6
    assert li.CARD_CLAIM["row"] == "hybrid+answerai-colbert-small@bge-small/fp32-doc300/top20"
    assert li.CARD_CLAIM["comparator"] == "hybrid+minilm-l12@bge-small/fp32-512/top20"
    assert "vastly outperforms cross-encoders its size" in li.CARD_CLAIM["card_line"]
