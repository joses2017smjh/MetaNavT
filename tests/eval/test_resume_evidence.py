"""docs/RESUME_EVIDENCE.md must equal `make evidence` output; every claim row carries its provenance."""

from pathlib import Path

from app.eval.evidence import claims, render

ROOT = Path(__file__).resolve().parents[2]


def test_document_is_in_sync():
    assert (ROOT / "docs" / "RESUME_EVIDENCE.md").read_text() == render(), "run `make evidence`"


def test_every_claim_has_command_source_and_limits():
    rows = claims()
    assert len(rows) >= 10
    for r in rows:
        assert r["command"].startswith(("make ", "docker ")) and "bench/results/" in r["source"] and r["not_shown"]


def test_prose_sections_type_no_metric_digits():
    from app.eval.evidence import BULLETS, QA
    import re

    # a metric typed by hand looks like a decimal with two or more places, a signed delta, or scientific notation
    metric = re.compile(r"(?<![\w.])[-+]?\d*\.\d{2,}(?![\w.])|(?<![\w.])\d+(?:\.\d+)?[eE][-+]?\d+(?![\w.])")
    for text in (BULLETS, QA):
        found = metric.findall(text)
        assert not found, f"metric-like numbers typed by hand in prose: {found}"


def test_rrf_sweep_row_reads_the_held_out_delta_and_says_what_happens_to_the_default():
    """B12: one row for the RRF sweep, read from rrf_sweep.json: the tune-half winner minus k = 60 on the held-out half,
    the verdict word from its interval, and that the default stays unless that interval excludes zero in its favour."""
    import json

    from app.eval.evidence import _ci
    from app.eval.report import verdict

    blob = json.loads((ROOT / "bench" / "results" / "rrf_sweep.json").read_text())
    rows = [c for c in claims() if c["command"] == "make bench-rrf-sweep"]
    assert len(rows) == 1
    row, d = rows[0], blob["decision"]
    assert row["source"] == f"bench/results/rrf_sweep.json @ {blob['git_sha']}"
    if d["chosen_is_default"]:  # not the recorded run; this branch is also tested on a copy below
        assert row["ci"] == "—" and row["n"] == len(blob["split"]["tune_ids"])
        return
    block = d["held_out_ndcg@10"]
    assert (row["value"], row["ci"]) == _ci(block, "delta", digits=4) and row["n"] == len(blob["split"]["held_out_ids"])
    assert f"k = {d['chosen']['k']}, dense weight {d['chosen']['weights']['dense']:g}" in row["claim"]
    assert f"verdict from the interval: {verdict(block)}" in row["not_shown"]
    assert ("the default stays k = 60 with equal weights" in row["not_shown"]) == (not d["owner_decision_needed"])


def test_rrf_sweep_row_when_the_tune_half_chose_the_default(tmp_path):
    import json

    from app.eval.evidence import _rrf_sweep_claims

    blob = json.loads((ROOT / "bench" / "results" / "rrf_sweep.json").read_text())
    blob["decision"]["chosen"] = {"k": 60, "weights": {"bm25": 1.0, "dense": 1.0}}
    blob["decision"]["chosen_is_default"] = True
    blob["decision"]["held_out_ndcg@10"] = None
    blob["decision"]["owner_decision_needed"] = False
    (tmp_path / "bench" / "results").mkdir(parents=True)
    (tmp_path / "bench" / "results" / "rrf_sweep.json").write_text(json.dumps(blob))
    (row,) = _rrf_sweep_claims(tmp_path)
    assert row["ci"] == "—" and row["n"] == len(blob["split"]["tune_ids"]) and "chose the default" in row["claim"]
    assert "nothing to compare on the held-out half" in row["not_shown"]


def test_rrf_sweep_row_names_the_default_the_file_records(tmp_path):
    """The default setting is read from the file's decision block, not typed: a sweep recorded after the owner changed the
    default must not be reported against k = 60."""
    import json

    from app.eval.evidence import _rrf_sweep_claims

    blob = json.loads((ROOT / "bench" / "results" / "rrf_sweep.json").read_text())
    blob["decision"]["default"] = {"k": 100, "weights": {"bm25": 1.0, "dense": 1.0}}
    (tmp_path / "bench" / "results").mkdir(parents=True)
    (tmp_path / "bench" / "results" / "rrf_sweep.json").write_text(json.dumps(blob))
    (row,) = _rrf_sweep_claims(tmp_path)
    assert "minus k = 100 with equal weights" in row["claim"]
    assert "k = 60" not in row["claim"] + row["not_shown"]
    assert ("the default stays k = 100 with equal weights" in row["not_shown"]) == (not blob["decision"]["owner_decision_needed"])
    blob["decision"].update({"chosen": {"k": 100, "weights": {"bm25": 1.0, "dense": 1.0}}, "chosen_is_default": True,
                             "held_out_ndcg@10": None, "owner_decision_needed": False})
    (tmp_path / "bench" / "results" / "rrf_sweep.json").write_text(json.dumps(blob))
    (row,) = _rrf_sweep_claims(tmp_path)
    assert row["claim"].endswith("the tune half chose the default, k = 100 with equal weights")
    assert "no evidence that k = 100 is best" in row["not_shown"] and "k = 60" not in row["claim"] + row["not_shown"]


def test_hybrid_drops_row_reads_its_counts_and_ranks_from_the_file():
    """B17: the diagnosis row counts the gold files hybrid@bge-base drops and gains against dense_only@bge-base, from
    hybrid_drops_bge_base.json, with the rank ranges that place the drops just past the fused cut."""
    import json

    blob = json.loads((ROOT / "bench" / "results" / "hybrid_drops_bge_base.json").read_text())
    rows = [c for c in claims() if c["command"] == "make bench-hybrid-drops"]
    assert len(rows) == 1
    row, s = rows[0], blob["summary"]
    assert row["source"] == f"bench/results/hybrid_drops_bge_base.json @ {blob['git_sha']}" and row["n"] == s["n_questions"]
    assert row["value"] == f"{s['n_dropped']} dropped on {s['n_questions_with_drops']} questions, {s['n_gained']} gained"
    fused = [d["fused"]["chunk_rank"] for q in blob["questions"] for d in q["dropped"]]
    if fused:  # a recording with no drops has no rank range to state
        assert f"{min(fused)}-{max(fused)} in the fused list, which is cut at {blob['retrieve_k']}" in row["not_shown"]
    assert f"Recall@50 delta {s['recall@50']['delta']:+.4f}" in row["not_shown"] and "not a retrieval-quality claim" in row["not_shown"]


DOCS = ("README.md", "docs/RESUME_EVIDENCE.md", "doc/engineering-reference.md")


def test_the_docs_say_the_parity_gap_is_reported_not_gated():
    """Since #10 the parity job reports the API-minus-in-memory gap with its interval and fails only when the API is
    broken or the fixture on disk differs from its MANIFEST: no doc may call it a gate on that gap."""
    for rel in DOCS:
        text = (ROOT / rel).read_text().lower()
        for phrase in ("parity gate", "gated by the same parity job", "gated by the parity job"):
            assert phrase not in text, (rel, phrase)


def test_ts_rank_cd_is_labelled_the_default_lexical_backend_not_a_fallback():
    """M11 made the lexical backend an explicit setting (LEXICAL_BACKEND) whose default is ts_rank_cd; nothing falls
    back to it. Every doc still labels it as not BM25 wherever its numbers appear (owner rule 3), without calling it a
    fallback."""
    for rel in DOCS:
        for line in (ROOT / rel).read_text().splitlines():
            if "ts_rank_cd" in line:
                assert not ("ts_rank_cd (fallback)" in line or "a fallback, not BM25" in line
                            or "lexical fallback" in line or "default stack's fallbacks" in line), (rel, line[:160])
    served = [line for line in (ROOT / "README.md").read_text().splitlines() if line.startswith("| api:")]
    assert served and all("ts_rank_cd (default LEXICAL_BACKEND, not BM25)" in line for line in served)
