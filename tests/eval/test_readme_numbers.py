"""The README's benchmark blocks must equal `make bench-table` over bench/results/main.json."""

import json
from pathlib import Path

from app.eval.report import BLOCKS, check, render, wrap

ROOT = Path(__file__).resolve().parents[2]


def test_readme_blocks_match_the_committed_baseline():
    blob = json.loads((ROOT / "bench" / "results" / "main.json").read_text())
    problems = check((ROOT / "README.md").read_text(), blob)
    assert problems == [], "\n".join(problems) + "\n\nrun `make bench-table` and paste the blocks into README.md"


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
    assert any("differs" in p for p in check(drifted, blob))
