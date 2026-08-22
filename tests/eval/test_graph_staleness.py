from app.graph.file_graph import build_file_graph, expand_with_graph
from app.graph.staleness import cluster_versions, prefer_current, content_hash
from app.graph.hierarchy import build_hierarchy, route_level
from app.retrieval.hybrid import Chunk, RetrievalHit


def _chunk(path, text, mtime, cid=None):
    return Chunk(
        chunk_id=cid or path,
        path=path,
        text=text,
        start_byte=0,
        end_byte=len(text),
        mtime=mtime,
        content_hash=content_hash(text),
    )


def test_file_graph_contains_and_same_run():
    files = [
        ("configs/run_047.yaml", "run_id: 47\ncheckpoint: checkpoints/run_047.ckpt.meta.json\n", 10),
        ("checkpoints/run_047.ckpt.meta.json", '{"run_id": 47}', 11),
        ("src/fusion.py", "import dinov2_encoder\n", 12),
        ("src/dinov2_encoder.py", "MODEL_NAME='dinov2'\n", 12),
    ]
    g = build_file_graph(files)
    kinds = {e.kind for e in g.edges}
    assert "contains" in kinds
    assert "same_run" in kinds
    expanded = expand_with_graph(["configs/run_047.yaml"], g, hops=1)
    assert "checkpoints/run_047.ckpt.meta.json" in expanded


def test_version_cluster_prefers_newest_non_archive():
    chunks = [
        _chunk("configs/archive/run_047_v1.yaml", "learning_rate: 1e-5", mtime=1),
        _chunk("configs/run_047.yaml", "learning_rate: 3e-4", mtime=9),
    ]
    clusters = cluster_versions(chunks)
    assert clusters
    cluster = next(iter(clusters.values()))
    assert cluster.current_member.path == "configs/run_047.yaml"
    hits = [
        RetrievalHit(chunk=chunks[0], score=1.0, rank=1),
        RetrievalHit(chunk=chunks[1], score=0.9, rank=2),
    ]
    kept = prefer_current(hits, clusters, "what's the current learning rate for run 47")
    assert [h.chunk.path for h in kept] == ["configs/run_047.yaml"]
    both = prefer_current(hits, clusters, "what changed between run 40 and 47")
    assert len(both) == 2


def test_hierarchy_levels():
    chunks = [
        _chunk("configs/run_047.yaml", "lr: 3e-4", 1, "c1"),
        _chunk("src/fusion.py", "fusion module", 1, "c2"),
    ]
    nodes = build_hierarchy(chunks)
    assert nodes["__corpus__"].level == "corpus"
    assert nodes["configs/run_047.yaml"].level == "file"
    assert route_level("what's in this corpus") == "corpus"
    assert route_level("what learning rate did run 47 use") == "chunk"
