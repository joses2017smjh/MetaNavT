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
    assert "api:postgres" in blocks["bench-parity"] and "reported, not gated" in blocks["bench-parity"] and "Gate:" not in blocks["bench-parity"]
    assert "api-latency" in blocks and ("| stack |" in blocks["api-latency"] or "no api_latency" in blocks["api-latency"])
    assert "bench-jury" in blocks and ("kappa" in blocks["bench-jury"] or "make bench-jury" in blocks["bench-jury"])


def test_missing_neural_files_render_a_pointer_not_a_crash(tmp_path):
    blob = json.loads((ROOT / "bench" / "results" / "main.json").read_text())
    blocks = render(blob, tmp_path)
    assert "make bench-neural" in blocks["bench-neural"] and "make bench-beir" in blocks["bench-beir"]
    assert "make bench-beir-bm25-ablation" in blocks["bench-beir-bm25"] and "make bench-beir-fiqa" in blocks["bench-beir-fiqa"]
    assert "make parity" in blocks["bench-parity"]


def test_check_reports_missing_markers_and_drift():
    blob = json.loads((ROOT / "bench" / "results" / "main.json").read_text())
    assert any("markers missing" in p for p in check("no blocks here", blob))
    drifted = "\n".join(wrap(name, body + " EDITED") for name, body in render(blob).items())
    assert any("differs" in p for p in check(drifted, blob, BLOCKS, ROOT))


def test_beir_published_comparison_prints_the_score_it_subtracts():
    """B05: the SciFact sentence and the evidence row quote our nDCG@10 to four decimals, the precision of the
    difference printed next to it, so the subtraction closes."""
    from app.eval.evidence import claims
    from app.eval.report import beir_table

    blob = json.loads((ROOT / "bench" / "results" / "beir_scifact.json").read_text())
    bm25 = {r["config"]: r for r in blob["results"]}["bm25"]
    score, vp = bm25["retrieval"]["ndcg@10"], bm25["vs_published"]
    assert round(score - vp["published_bm25_ndcg@10"], 4) == vp["delta"]
    sentence = f"scores nDCG@10 {score:.4f} against Anserini's published {vp['published_bm25_ndcg@10']:.4f} ({vp['delta']:+.4f})"
    assert sentence in beir_table(blob)
    row = next(c for c in claims() if c["claim"].startswith("BEIR SciFact, our BM25"))
    assert row["value"] == f"{score:.4f}" and f"{vp['published_bm25_ndcg@10']:.4f} ({vp['delta']:+.4f})" in row["not_shown"]


def test_router_rows_show_the_slices_where_the_router_loses():
    """Keep losers visible: next to the router's exact-path line, the generated blocks list every other gold category
    where its paired nDCG@10 delta is negative, with the interval and how many questions rose and fell, read from
    delta_vs_parent_by_category_all. With real models that includes the aggregation slice (neural.json)."""
    from app.eval.report import neural_table, negative_slices, table

    for rel, render_block in (("main.json", table), ("neural.json", neural_table)):
        blob = json.loads((ROOT / "bench" / "results" / rel).read_text())
        text = render_block(blob)
        for row in blob["results"]:
            if not row.get("delta_vs_parent_by_category"):
                continue
            losers = negative_slices(row)
            assert losers == [b for b in row.get("delta_vs_parent_by_category_all") or []
                              if b["category"] != row["delta_vs_parent_by_category"]["category"] and b["ndcg@10"]["delta"] < 0]
            for block in losers:
                moved = block["questions_changed"]
                line = next(x for x in text.splitlines() if x.startswith(f"  - `{row['config']}` on `{block['category']}`"))
                d = block["ndcg@10"]
                assert f"{d['delta']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]" in line
                assert f"rose on {moved['up']} of the {block['n']} questions and fell on {moved['down']}" in line
            if not losers:
                assert f"  - `{row['config']}`: no other category has a negative nDCG@10 delta." in text
    neural = json.loads((ROOT / "bench" / "results" / "neural.json").read_text())
    router = next(r for r in neural["results"] if r["config"] == "hybrid+bge-rerank+router@bge-small")
    assert "aggregation" in [b["category"] for b in negative_slices(router)]
    readme = (ROOT / "README.md").read_text()
    assert "  - `hybrid+bge-rerank+router@bge-small` on `aggregation`" in readme and "  - `hybrid+rerank+router` on `aggregation`" in readme
