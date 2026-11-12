"""The agent-path runs (A25 / C13, C23) have generated rows in docs/RESUME_EVIDENCE.md once their files exist,
and every number in those rows comes from the results file."""

import json
from pathlib import Path

import pytest

from app.eval.evidence import FILES, claims

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("key,command", [("companion_live", "make bench-companion-live"),
                                         ("chat_route_smoke", "make chat-route-smoke")])
def test_rows_follow_the_results_file(key, command, tmp_path):
    rows = [r for r in claims() if r["command"] == command]
    if not (ROOT / FILES[key]).exists():
        assert rows == []
        pytest.skip(f"{FILES[key]} not recorded yet")
    assert rows and all(r["source"].startswith(FILES[key] + " @ ") for r in rows)
    # rendered from the file: a copy with other numbers renders other values
    blob = json.loads((ROOT / FILES[key]).read_text())
    if key == "companion_live":
        blob["summary"]["gold"]["overall"]["status"] = {"complete": 1}
    else:
        blob["summary"]["/api/chat"]["answered"] = 99
    target = tmp_path / FILES[key]
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(blob))
    changed = [r for r in claims(tmp_path) if r["command"] == command]
    assert [r["value"] for r in changed] != [r["value"] for r in rows]


@pytest.mark.parametrize("key,command", [("companion_live", "make bench-companion-live"),
                                         ("chat_route_smoke", "make chat-route-smoke")])
def test_every_row_names_the_fallback_retrieval(key, command):
    rows = [r for r in claims() if r["command"] == command]
    if not (ROOT / FILES[key]).exists():
        pytest.skip(f"{FILES[key]} not recorded yet")
    for row in rows:
        assert all(label in row["not_shown"] for label in ("hash embeddings", "ts_rank_cd", "no reranker")), row["claim"]


def test_the_citation_row_counts_the_answers_the_companion_blocked_for_citing(tmp_path):
    if not (ROOT / FILES["companion_live"]).exists():
        pytest.skip("bench/results/companion_live.json not recorded yet")
    blob = json.loads((ROOT / FILES["companion_live"]).read_text())

    def citation_row(root):
        return next(r for r in claims(root) if r["command"] == "make bench-companion-live" and "citation" in r["claim"])

    blocked = blob["summary"]["gold"]["overall"]["stop_reason"].get("invalid_citations", 0)
    assert f"({blocked} here, stop reason invalid_citations)" in citation_row(ROOT)["not_shown"]
    assert "stopped at the model-round limit" in citation_row(ROOT)["not_shown"]
    blob["summary"]["gold"]["overall"]["stop_reason"]["invalid_citations"] = blocked + 5
    target = tmp_path / FILES["companion_live"]
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(blob))
    assert f"({blocked + 5} here, stop reason invalid_citations)" in citation_row(tmp_path)["not_shown"]
