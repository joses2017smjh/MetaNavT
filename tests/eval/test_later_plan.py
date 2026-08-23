import numpy as np

from app.graph.conflicts import annotate_for_generator, detect_semantic_conflicts, detect_structural_conflicts
from app.graph.graphrag import answer_global, build_graphrag, extract_triples
from app.multimodal.colpali import image_to_patches, index_page, query_to_patches, search_pages
from app.multimodal.fusion import fuse_modalities
from app.multimodal.images import pixel_embed, search_images
from app.retrieval.types import Chunk, RetrievalHit
from app.retrieval.pgvector_tune import compare_storage, sweep_ef_search, to_binary, to_halfvec


def test_halfvec_and_binary_keep_most_recall():
    rng = np.random.RandomState(0)
    docs = rng.randn(40, 32).astype(np.float32)
    queries = rng.randn(8, 32).astype(np.float32)
    points = compare_storage(docs, queries, k=5)
    by_mode = {p.mode: p for p in points}
    assert by_mode["halfvec"].recall_10 >= 0.8 * by_mode["float32"].recall_10
    assert by_mode["halfvec"].bytes_per_vec == by_mode["float32"].bytes_per_vec // 2
    assert by_mode["binary+rescore"].bytes_per_vec < by_mode["halfvec"].bytes_per_vec
    assert to_halfvec(docs).shape == docs.shape
    assert to_binary(docs).shape == docs.shape


def test_hnsw_ef_search_sweep_runs():
    rng = np.random.RandomState(1)
    docs = rng.randn(30, 16).astype(np.float32)
    queries = rng.randn(6, 16).astype(np.float32)
    pts = sweep_ef_search(docs, queries, k=5, m=4, efs=(8, 16, 32))
    assert [p.ef_search for p in pts] == [8, 16, 32]
    assert all(0.0 <= p.recall_10 <= 1.0 for p in pts)


def test_graphrag_communities_from_runs():
    chunks = [
        Chunk("a", "configs/run_047.yaml", "run_id: 47 encoder: dinov2 learning_rate: 3e-4 bark_type: birch", 0, 80),
        Chunk("b", "configs/run_046.yaml", "run_id: 46 encoder: dinov2 fusion: true", 0, 50),
        Chunk("c", "src/fusion.py", "fusion concatenates features", 0, 40),
    ]
    triples = extract_triples(chunks[0].path, chunks[0].text)
    assert any(t.relation == "uses_encoder" for t in triples)
    comm = build_graphrag(chunks, min_community=1)
    assert comm
    text = answer_global("which runs used dinov2", comm)
    assert "run" in text.lower() or "community" in text.lower()


def test_tier2_flags_conflicting_learning_rates():
    hits = [
        RetrievalHit(Chunk("n", "configs/run_047.yaml", "run_id: 47 learning_rate: 3e-4", 0, 40), 1.0, 1),
        RetrievalHit(Chunk("o", "configs/archive/run_047_v1.yaml", "run_id: 47 learning_rate: 1e-5", 0, 40), 0.9, 2),
    ]
    conflicts = detect_structural_conflicts(hits)
    assert conflicts
    assert any(c.field == "learning_rate" for c in conflicts)
    paper = [
        RetrievalHit(Chunk("p1", "paper/draft_v1.md", "fusion does not help", 0, 20), 1.0, 1),
        RetrievalHit(Chunk("p2", "paper/draft_v2.md", "stereo fusion **does** help", 0, 20), 0.8, 2),
    ]
    sem = detect_semantic_conflicts(paper)
    assert sem
    note = annotate_for_generator(hits + paper)["generator_note"]
    assert "Conflicting" in note


def test_colpali_maxsim_page_search():
    page_a = np.zeros((64, 64), dtype=np.uint8)
    page_a[10:20, 10:40] = 255
    page_b = np.zeros((64, 64), dtype=np.uint8)
    page_b[:, :] = 8
    indexed = [
        index_page("ablation#p0", "paper/table.png", 0, page_a, patch=16, dim=32, caption="ablation table fusion off"),
        index_page("blank#p0", "paper/blank.png", 0, page_b, patch=16, dim=32, caption="empty page"),
    ]
    hits = search_pages("ablation table fusion off", indexed, k=2)
    assert hits[0][0].page_id == "ablation#p0"
    q = query_to_patches("fusion off", dim=32)
    assert q.shape[1] == 32
    patches = image_to_patches(page_a, patch=16)
    assert patches.shape[0] >= 4


def test_image_search_and_modality_fusion():
    img = np.linspace(0, 255, 16 * 16, dtype=np.float32).reshape(16, 16)
    vec = pixel_embed(img, dim=32)
    hits = search_images("trellis wires", [("renders/trellis.png", vec)], k=1)
    assert hits[0][0] == "renders/trellis.png"
    fused = fuse_modalities(["t1"], ["p1"], ["renders/trellis.png"])
    assert fused
