"""The README's benchmark blocks must equal `make bench-table` over bench/results/main.json."""

import json
from pathlib import Path

from app.eval.report import BLOCKS, FILES, check, check_files, render, wrap, write

ROOT = Path(__file__).resolve().parents[2]


def test_generated_blocks_match_the_committed_baseline():
    blob = json.loads((ROOT / "bench" / "results" / "main.json").read_text())
    problems = check_files(ROOT, blob)
    assert problems == [], "\n".join(problems) + "\n\nrun `make bench-table-write`"
    assert set(FILES) == {"README.md", "doc/demo.html"}


def test_write_replaces_blocks_in_place():
    blob = json.loads((ROOT / "bench" / "results" / "main.json").read_text())
    stale = "\n".join(wrap(name, "STALE") for name in BLOCKS) + "\ntrailing text"
    fresh = write(stale, blob, BLOCKS)
    assert "STALE" not in fresh and fresh.endswith("trailing text")
    assert check(fresh, blob, BLOCKS) == []


def test_render_covers_every_block_and_names_the_seed():
    blob = json.loads((ROOT / "bench" / "results" / "main.json").read_text())
    blocks = render(blob)
    assert set(blocks) == set(BLOCKS)
    assert "seed" in blocks["bench-table"] and "bm25_only" in blocks["bench-table"]
    assert wrap("bench-table", "x").startswith("<!-- bench-table:start")


def test_check_reports_missing_markers_and_drift():
    blob = json.loads((ROOT / "bench" / "results" / "main.json").read_text())
    assert any("markers missing" in p for p in check("no blocks here", blob))
    drifted = "\n".join(wrap(name, body + " EDITED") for name, body in render(blob).items())
    assert any("differs" in p for p in check(drifted, blob, BLOCKS))
