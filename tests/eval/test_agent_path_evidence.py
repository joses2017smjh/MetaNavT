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
