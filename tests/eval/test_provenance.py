from pathlib import Path

from app.eval.provenance import device_info, hf_revision, model_provenance


def test_hf_revision_reads_refs_main_from_the_cache(tmp_path, monkeypatch):
    slug = tmp_path / "models--BAAI--bge-small-en-v1.5"
    (slug / "refs").mkdir(parents=True)
    (slug / "refs" / "main").write_text("5c38ec7c405ec4b44bffa4de0e3f7d5d7b3d1c7f\n")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    assert hf_revision("BAAI/bge-small-en-v1.5") == "5c38ec7c405ec4b44bffa4de0e3f7d5d7b3d1c7f"
    assert hf_revision("BAAI/not-cached") is None


def test_model_provenance_labels_non_neural_backends():
    assert model_provenance("hash", "embedder") == {"role": "embedder", "model": "hash", "revision": None, "neural": False}
    p = model_provenance("st:BAAI/bge-small-en-v1.5", "embedder")
    assert p["model"] == "BAAI/bge-small-en-v1.5" and p["neural"] is True and "revision" in p


def test_device_info_works_without_cuda():
    info = device_info("cpu")
    assert info["device"] == "cpu" and "python" in info and "torch" in info
