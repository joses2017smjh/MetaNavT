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
    fresh = write(stale, blob, BLOCKS, ROOT)
    assert "STALE" not in fresh and fresh.endswith("trailing text")
    assert check(fresh, blob, BLOCKS, ROOT) == []


def test_render_covers_every_block_and_names_the_seed():
    blob = json.loads((ROOT / "bench" / "results" / "main.json").read_text())
    blocks = render(blob, ROOT)
    assert set(blocks) == set(BLOCKS)
    assert "seed" in blocks["bench-table"] and "bm25_only" in blocks["bench-table"]
    assert wrap("bench-table", "x").startswith("<!-- bench-table:start")
    assert "beats BM25?" in blocks["bench-neural"] and "bge-reranker-v2-m3" in blocks["bench-neural"]
    assert "Published BM25" in blocks["bench-beir"] and "beats BM25?" in blocks["bench-beir"]
    assert "api:postgres" in blocks["bench-parity"] and "Gate:" in blocks["bench-parity"]
    assert "api-latency" in blocks and ("| stack |" in blocks["api-latency"] or "no api_latency" in blocks["api-latency"])
    assert "bench-jury" in blocks and ("kappa" in blocks["bench-jury"] or "make bench-jury" in blocks["bench-jury"])


def test_missing_neural_files_render_a_pointer_not_a_crash(tmp_path):
    blob = json.loads((ROOT / "bench" / "results" / "main.json").read_text())
    blocks = render(blob, tmp_path)
    assert "make bench-neural" in blocks["bench-neural"] and "make bench-beir" in blocks["bench-beir"]
    assert "make parity" in blocks["bench-parity"]


def test_check_reports_missing_markers_and_drift():
    blob = json.loads((ROOT / "bench" / "results" / "main.json").read_text())
    assert any("markers missing" in p for p in check("no blocks here", blob))
    drifted = "\n".join(wrap(name, body + " EDITED") for name, body in render(blob).items())
    assert any("differs" in p for p in check(drifted, blob, BLOCKS, ROOT))
