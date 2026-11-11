"""The benchmark gives the same rows on any machine: mtimes come from the fixture's MANIFEST."""

import json
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path

import pytest

from app.eval.gold import load_gold
from app.eval.harness import DEFAULT_CONFIGS, FRONTIER_CONFIGS, run_bench, run_config
from app.eval.index_loader import fixture_manifest, load_chunks, load_chunks_with_report, mtime_provenance

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "bench" / "corpus"
VOLATILE = ("latency", "wall_ms")  # timings: the only row fields allowed to differ between runs


def _fixture_copy(base: Path, mtime_of) -> Path:
    """bench/corpus/{files,MANIFEST.json} and bench/gold under `base`; each file's mtime set by mtime_of(rel, entry)."""
    corpus = base / "bench" / "corpus"
    shutil.copytree(CORPUS / "files", corpus / "files")
    shutil.copy2(CORPUS / "MANIFEST.json", corpus / "MANIFEST.json")
    shutil.copytree(ROOT / "bench" / "gold", base / "bench" / "gold")
    for entry in json.loads((corpus / "MANIFEST.json").read_text())["files"]:
        t = mtime_of(entry["path"], entry)
        os.utime(corpus / "files" / entry["path"], (t, t))
    return base


def _rows(blob: dict) -> list[dict]:
    return [{k: v for k, v in row.items() if k not in VOLATILE} for row in blob["results"]]


STALENESS = DEFAULT_CONFIGS[-1]  # the row whose filter picks the newest member of each version cluster
HOPS = next(c for c in FRONTIER_CONFIGS if c.name == "hybrid+hops=1")  # co-modified edges compare mtimes too


def test_loader_reads_fixture_mtimes_from_the_manifest_not_the_disk(tmp_path):
    now = time.time()
    files = _fixture_copy(tmp_path, lambda rel, entry: now) / "bench" / "corpus" / "files"
    listed = {e["path"]: float(e["mtime"]) for e in json.loads((files.parent / "MANIFEST.json").read_text())["files"]}

    chunks, report = load_chunks_with_report(files)

    assert {c.path: c.mtime for c in chunks} == {p: listed[p] for p in {c.path for c in chunks}}
    assert report["mtimes"] == {"source": "manifest", "file": "corpus/MANIFEST.json", "field": "files[].mtime"}
    assert mtime_provenance(files, base=tmp_path)["file"] == "bench/corpus/MANIFEST.json"
    assert all(c.mtime == pytest.approx(now, abs=1.0) for c in load_chunks(files, mtimes="disk"))  # what the disk says


def test_same_rows_whatever_the_disk_mtimes(tmp_path):
    """A checkout sets every mtime to "now"; another order makes the archived versions newest.
    Every default row and the graph-hops row equal the run on the repository's own fixture."""
    now = time.time()
    fresh = _fixture_copy(tmp_path / "fresh", lambda rel, entry: now)
    reversed_ = _fixture_copy(tmp_path / "reversed", lambda rel, entry: 2_000_000_000 - entry["mtime"])
    configs = [*DEFAULT_CONFIGS, HOPS]

    def run(root, name):
        return run_bench(root=root, configs=[replace(c) for c in configs], out=tmp_path / f"{name}.json")

    reference = run(ROOT, "repository")
    assert reference["mtimes"] == {"source": "manifest", "file": "bench/corpus/MANIFEST.json", "field": "files[].mtime"}
    assert json.loads((tmp_path / "repository.json").read_text())["mtimes"] == reference["mtimes"]
    assert _rows(run(fresh, "fresh")) == _rows(reference)
    assert _rows(run(reversed_, "reversed")) == _rows(reference)


def test_disk_mtimes_would_change_the_rows(tmp_path):
    """Why the manifest matters: with the disk's mtimes, a fresh checkout (every file "now") scores differently."""
    now = time.time()
    files = _fixture_copy(tmp_path, lambda rel, entry: now) / "bench" / "corpus" / "files"
    gold = load_gold(tmp_path / "bench" / "gold" / "questions.jsonl")

    from_manifest = run_config(replace(STALENESS), gold, files, mtimes="manifest")["retrieval"]
    from_disk = run_config(replace(STALENESS), gold, files, mtimes="disk")["retrieval"]

    assert run_config(replace(STALENESS), gold, files)["retrieval"] == from_manifest  # the default ("auto") finds the fixture
    assert from_disk != from_manifest


def test_fixture_manifest_needs_the_fixture_layout(tmp_path):
    (tmp_path / "files").mkdir()
    (tmp_path / "files" / "a.md").write_text("# a\n")
    assert fixture_manifest(tmp_path / "files") is None
    assert load_chunks_with_report(tmp_path / "files")[1]["mtimes"] == {"source": "disk", "field": "st_mtime"}
    (tmp_path / "MANIFEST.json").write_text(json.dumps([{"path": "a.md", "mtime": 1}]))
    assert fixture_manifest(tmp_path / "files") is None  # not a manifest object
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"root": "elsewhere/corpus", "files": [{"path": "a.md", "mtime": 1}]}))
    assert fixture_manifest(tmp_path / "files") is None  # a manifest for another directory name
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"root": "x/files", "files": [{"path": "a.md"}]}))
    assert fixture_manifest(tmp_path / "files") is None  # an entry without an mtime
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"root": "x/files", "files": [{"path": "a.md", "mtime": 1}]}))
    assert fixture_manifest(tmp_path / "files") == tmp_path / "MANIFEST.json"
    assert [c.mtime for c in load_chunks(tmp_path / "files")] == [1.0]
    (tmp_path / "files" / "b.md").write_text("# b\n")
    with pytest.raises(ValueError, match="not in its MANIFEST.json"):
        load_chunks(tmp_path / "files")
    (tmp_path / "plain").mkdir()
    (tmp_path / "plain" / "a.md").write_text("# a\n")
    with pytest.raises(ValueError, match="no fixture MANIFEST.json"):
        load_chunks(tmp_path / "plain", mtimes="manifest")
    with pytest.raises(ValueError, match="unknown mtimes mode"):
        load_chunks(tmp_path / "plain", mtimes="checkout")
