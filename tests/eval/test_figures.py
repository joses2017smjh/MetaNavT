"""The committed figures equal `make figures` over the committed results files, and the hops figure's subtitle reads
its numbers from bench/results/sweeps.json (M34: they were typed into the generator)."""

import json
import shutil
from pathlib import Path

from app.eval import figures

ROOT = Path(__file__).resolve().parents[2]


def test_hops_figure_is_regenerated_from_sweeps_json(tmp_path):
    (tmp_path / "bench" / "results").mkdir(parents=True)
    for name in ("main.json", "sweeps.json"):
        shutil.copy(ROOT / "bench" / "results" / name, tmp_path / "bench" / "results" / name)
    figures.write_from_results(tmp_path)
    svg = (tmp_path / "doc" / "figures" / "hops.svg").read_text()
    assert svg == (ROOT / "doc" / "figures" / "hops.svg").read_text(), "run `make figures`"
    hops = {p["config"]: p["retrieval"] for p in json.loads((ROOT / "bench" / "results" / "sweeps.json").read_text())["graph_hops"]}
    zero, one = hops["hybrid+hops=0"], hops["hybrid+hops=1"]
    assert f"One hop {zero['recall@50']:.3f} → {one['recall@50']:.3f} recall. nDCG@10 falls {zero['ndcg@10']:.3f} → {one['ndcg@10']:.3f}." in svg

    sweeps = json.loads((ROOT / "bench" / "results" / "sweeps.json").read_text())
    sweeps["graph_hops"][1]["retrieval"]["ndcg@10"] = 0.1234  # a re-recorded sweep moves the subtitle with it
    (tmp_path / "bench" / "results" / "sweeps.json").write_text(json.dumps(sweeps))
    figures.write_from_results(tmp_path)
    assert "falls 0.495 → 0.123." in (tmp_path / "doc" / "figures" / "hops.svg").read_text()


def test_the_m41_charts_are_what_make_figures_draws_from_the_committed_results(tmp_path):
    """M41: the two model charts the README shows equal `make figures` over the committed results files, byte for byte
    (the SVG is plain text formatting), mark the served models, state each dataset's size and keep every row."""
    (tmp_path / "bench" / "results").mkdir(parents=True)
    names = ("models_2026.json", "models_2026_harrier.json", "beir_scifact_models_2026.json", "beir_scifact_models_2026_harrier.json",
             "beir_fiqa_models_2026.json", "beir_fiqa_models_2026_harrier.json")
    for name in names:
        if (ROOT / "bench" / "results" / name).exists():
            shutil.copy(ROOT / "bench" / "results" / name, tmp_path / "bench" / "results" / name)
    written = figures.write_models_2026_figures(tmp_path, tmp_path / "out")
    assert [p.name for p in written] == list(figures.MODELS_2026_FIGURES)
    readme = (ROOT / "README.md").read_text()
    for path in written:
        assert path.read_text() == (ROOT / "doc" / "figures" / path.name).read_text(), "run `make figures`"
        assert f"](doc/figures/{path.name})" in readme  # linked from the README, with alt text
    embedders, rerankers = (p.read_text() for p in written)
    scifact = json.loads((ROOT / "bench" / "results" / "beir_scifact_models_2026.json").read_text())
    assert f"{scifact['n_docs']} documents, {scifact['n_queries']} test queries" in embedders
    assert embedders.count(">served<") == 3 and rerankers.count(">served<") == 3  # one served row per dataset panel
    assert "▼ below" in embedders and "= tie" in rerankers  # losers and ties stay visible
    fixture = json.loads((ROOT / "bench" / "results" / "models_2026.json").read_text())
    for row in fixture["results"]:
        if row["config"].startswith(("dense_only@", "hybrid@")):
            assert f"{row['retrieval']['ndcg@10']:.3f}" in embedders, row["config"]


def test_the_m41_charts_draw_nothing_without_their_results(tmp_path):
    assert figures.write_models_2026_figures(tmp_path, tmp_path / "out") == []
