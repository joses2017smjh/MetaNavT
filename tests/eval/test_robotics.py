import json
import subprocess

import pytest

from app.eval.robotics import confined, evaluate, freeze, group_bootstrap, scalar_questions, verify


@pytest.fixture
def snapshot(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    def git(*args):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.name", "Test fixture")
    git("config", "user.email", "fixture@example.invalid")
    git("remote", "add", "origin", "https://example.invalid/robot.git")
    for i in range(12):
        p = repo / "results" / f"campaign-{i}" / "summary.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"success": bool(i % 2), "steps": 100 + i, "verdict": "PASS", "seed": i}))
    git("add", "results")
    git("commit", "-qm", "fixture")
    destination = tmp_path / "snapshot"
    freeze(repo, destination, n_files=12, dev_n=4, test_n=8)
    return destination


def test_utf8_evidence_and_nested_collision():
    data = '{"note":"café", "nested":{"success":false}, "success":true}'.encode()
    q = scalar_questions("results/run/a.json", data, "results/run")[0]
    a, b = q["evidence"]["start_byte"], q["evidence"]["end_byte"]
    assert data[a:b] == b"true"
    assert q["answer"] is True
    assert not q["human_reviewed"]


def test_manifest_split_and_content_integrity(snapshot):
    m, questions = verify(snapshot)
    dev = {q["family"] for q in questions if q["split"] == "dev"}
    test = {q["family"] for q in questions if q["split"] == "test"}
    assert dev.isdisjoint(test)
    assert m["n_files"] == 12
    target = snapshot / "corpus" / m["files"][0]["path"]
    target.write_text('{"success": false}')
    with pytest.raises(ValueError, match="corpus changed"):
        verify(snapshot)


def test_gold_tamper_is_rejected(snapshot):
    q = snapshot / "questions.jsonl"
    q.write_bytes(q.read_bytes().replace(b'"human_reviewed": false', b'"human_reviewed": true', 1))
    with pytest.raises(ValueError, match="questions changed"):
        verify(snapshot)


def test_symlink_and_parent_escape_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "escape").symlink_to(tmp_path)
    for path in ("../outside", "/etc/passwd", "escape/file"):
        with pytest.raises(ValueError):
            confined(root, path)


def test_extra_file_invalidates_inventory(snapshot):
    (snapshot / "corpus" / "unexpected.txt").write_text("not frozen")
    with pytest.raises(ValueError, match="inventory mismatch"):
        verify(snapshot)


def test_family_resampling_retains_cluster_dependence():
    # One campaign contributes ten identical queries; another contributes one.
    rows = [{"family": "a", "x": 1., "y": 0.} for _ in range(10)] + [{"family": "b", "x": 0., "y": 0.}]
    result = group_bootstrap(rows, "x", paired_field="y")
    assert result["mean"] == pytest.approx(10 / 11)
    assert result["n_families"] == 2
    assert result["ci95"] == [0., 1.]


def test_insufficient_source_fails_without_partial_snapshot(tmp_path):
    source = tmp_path / "empty"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    target = tmp_path / "out"
    with pytest.raises(ValueError, match="unique eligible"):
        freeze(source, target, n_files=1)
    assert not target.exists()


def test_search_context_keeps_citation_text_unchanged(tmp_path):
    from app.eval.index_loader import load_chunks
    from app.retrieval.hybrid import InMemoryHybridIndex
    source = tmp_path / "results" / "walk-push" / "summary.json"
    source.parent.mkdir(parents=True)
    source.write_text('{"success": true}')
    (tmp_path / "other.json").write_text('{"success": false}')
    plain = load_chunks(tmp_path)
    enriched = load_chunks(tmp_path, include_path_context=True)
    assert [(c.text, c.start_byte, c.end_byte, c.content_hash) for c in plain] == [(c.text, c.start_byte, c.end_byte, c.content_hash) for c in enriched]
    index = InMemoryHybridIndex(enriched, enable_rerank=False)
    assert index.search_bm25("walk push", k=1)[0][0].path == "results/walk-push/summary.json"


def test_loaded_citations_use_utf8_bytes(tmp_path):
    from app.eval.index_loader import load_chunks
    source = tmp_path / "notes.md"
    source.write_text("# café\nRésumé résumé.\n\n# robot\n歩行 success.\n")
    blob = source.read_bytes()
    chunks = load_chunks(tmp_path)
    assert len(chunks) == 2
    for chunk in chunks:
        raw = blob[chunk.start_byte:chunk.end_byte]
        assert raw.decode() == chunk.text
        assert chunk.metadata["text_is_verbatim"]


def test_loaded_citations_preserve_crlf_bytes(tmp_path):
    from app.eval.hashing import content_hash
    from app.eval.index_loader import load_chunks
    source = tmp_path / "notes.md"
    blob = "# café\r\nRésumé résumé.\r\n\r\n# robot\r\n歩行 success.\r\n".encode()
    source.write_bytes(blob)
    chunks = load_chunks(tmp_path)
    assert len(chunks) == 2
    for chunk in chunks:
        raw = blob[chunk.start_byte:chunk.end_byte]
        assert raw.decode() == chunk.metadata["evidence_raw"]
        assert chunk.content_hash == content_hash(blob.decode())


def test_evaluation_rejects_snapshot_changed_after_indexing(snapshot, tmp_path, monkeypatch):
    from app.retrieval.bm25 import BM25Index
    manifest, _ = verify(snapshot)
    original = BM25Index.search
    changed = False
    def search(index, *args, **kwargs):
        nonlocal changed
        if not changed:
            source = snapshot / "corpus" / manifest["files"][0]["path"]
            source.write_bytes(source.read_bytes() + b"\n")
            changed = True
        return original(index, *args, **kwargs)
    monkeypatch.setattr(BM25Index, "search", search)
    report = tmp_path / "result.json"
    with pytest.raises(ValueError, match="corpus changed"):
        evaluate(snapshot, report)
    assert not report.exists()
