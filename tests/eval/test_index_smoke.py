"""End-to-end smoke of the bench index over a tiny frozen tree."""

from pathlib import Path

from app.eval.index_loader import build_index, load_chunks
from app.eval.metrics import recall_at_k


def test_index_loader_and_recall(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "run_047.yaml").write_text(
        "run_id: 47\nencoder: dinov2\nlearning_rate: 3.0e-4\nfusion: true\n"
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "fusion.py").write_text(
        '"""fusion concatenates per-view features."""\n'
    )
    chunks = load_chunks(tmp_path)
    assert chunks
    index = build_index(tmp_path, embedder_name="hash", enable_router=False)
    result = index.retrieve("learning_rate run 47 dinov2")
    rec = recall_at_k(result.paths, ["configs/run_047.yaml"], k=50)
    assert rec == 1.0
