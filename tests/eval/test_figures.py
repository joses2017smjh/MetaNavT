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
