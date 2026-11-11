"""The benchmark gives the same rows on any machine: mtimes from the fixture's MANIFEST, a defined dense order."""

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.eval.gold import load_gold
from app.eval.harness import DEFAULT_CONFIGS, FRONTIER_CONFIGS, run_bench, run_config
from app.eval.index_loader import fixture_manifest, load_chunks, load_chunks_with_report, mtime_provenance
from app.retrieval.embedders import HashEmbedder
from app.retrieval.hybrid import DENSE_SCORE_DECIMALS, InMemoryHybridIndex, dense_scores, rank_by_score
from app.retrieval.types import Chunk, RetrievalHit

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


def test_rank_by_score_keeps_row_order_for_ties():
    assert list(rank_by_score(np.array([0.3, 0.7, 0.3, 0.5, 0.3]), 5)) == [1, 3, 0, 2, 4]
    mixed = np.full(200, 0.5)
    mixed[::7] = 0.25  # two large tie groups: numpy's default argsort (introsort, SIMD) need not keep row order
    assert list(rank_by_score(mixed, 200)) == [i for i in range(200) if i % 7] + list(range(0, 200, 7))
    assert list(rank_by_score(mixed, 3)) == [1, 2, 3]
    assert list(rank_by_score(np.array([0.0, -0.0, 0.0]), 3)) == [0, 1, 2]  # signed zeros tie


def test_dense_scores_round_away_last_bit_differences():
    """Rows that tie in exact arithmetic: summed in another order their floats can differ in the last bit; rounded, they tie."""
    assert (0.1 + 0.2) + 0.3 != (0.3 + 0.2) + 0.1  # the same three numbers, another summation order
    permuted = np.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1], [0.3, 0.1, 0.2]], dtype=np.float32)
    s = dense_scores(np.ones(3, dtype=np.float32), permuted)
    assert s.dtype == np.float64 and s[0] == s[1] == s[2]
    assert list(rank_by_score(s, 3)) == [0, 1, 2]
    rng = np.random.RandomState(0)
    docs, q = rng.randn(5, 64).astype(np.float32), rng.randn(64).astype(np.float32)
    s = dense_scores(q, docs)
    assert np.array_equal(s, np.round(s, DENSE_SCORE_DECIMALS))
    d64, q64 = docs.astype(np.float64), q.astype(np.float64)
    exact = (d64 / np.linalg.norm(d64, axis=1, keepdims=True)) @ (q64 / np.linalg.norm(q64))
    assert np.allclose(s, exact, rtol=0, atol=10.0 ** -DENSE_SCORE_DECIMALS)


def test_dense_search_breaks_exact_ties_by_corpus_order():
    """Forty chunks in two groups of identical text: every score ties with its group, so corpus order decides."""
    texts = ("learning_rate: 3e-4 encoder dinov2", "bark_type: birch fusion true")
    corpus = [Chunk(chunk_id=f"c{i:02d}", path=f"f{i:02d}.yaml", text=texts[i % 3 == 0], start_byte=0, end_byte=9) for i in range(40)]
    query = "learning_rate for the dinov2 encoder"
    first, second = [c.chunk_id for c in corpus if c.text == texts[0]], [c.chunk_id for c in corpus if c.text == texts[1]]

    forward = InMemoryHybridIndex(corpus, embedder=HashEmbedder())
    backward = InMemoryHybridIndex(corpus[::-1], embedder=HashEmbedder())

    assert [c.chunk_id for c, _ in forward.search_dense(query, k=40)] == first + second
    assert [c.chunk_id for c, _ in backward.search_dense(query, k=40)] == first[::-1] + second[::-1]
    assert [c.chunk_id for c, _ in forward.search_dense(query, k=5)] == first[:5]


def test_hipporag_orders_exact_ties_by_path_not_the_last_bit():
    """Two path masses equal in exact arithmetic, one ulp apart as floats (BLAS on another CPU can do this)."""
    from app.graph.hipporag import boost_hits

    x, y = (0.1 + 0.2) + 0.3, (0.3 + 0.2) + 0.1
    assert x != y and round(x, 12) == round(y, 12)
    hits = [RetrievalHit(chunk=Chunk(chunk_id=p, path=p, text="", start_byte=0, end_byte=1), score=1.0, rank=i)
            for i, p in enumerate(["paper/draft_v2.md", "logs/run_046.out", "logs/run_047.out"], start=1)]

    one = boost_hits(hits, {"logs/run_046.out": x, "logs/run_047.out": y}, query="which runs")
    two = boost_hits(hits, {"logs/run_046.out": y, "logs/run_047.out": x}, query="which runs")

    assert [h.chunk.path for h in one] == [h.chunk.path for h in two]


def test_graphrag_community_order_does_not_depend_on_the_hash_seed():
    """Equal-size communities: their order (and so their ids and the boost they give) is the same for every PYTHONHASHSEED."""
    code = (
        "from app.graph.graphrag import build_graphrag\n"
        "from app.retrieval.types import Chunk\n"
        "chunks = [Chunk(chunk_id=f'c{i}', path=f'configs/run_00{i}.yaml', text=f'encoder: {e}\\nlearning_rate: 0.00{i}\\n',"
        " start_byte=0, end_byte=1) for i, e in ((1, 'dinov2'), (2, 'resnet'), (3, 'vit'), (4, 'convnext'))]\n"
        "print([(c.community_id, c.members) for c in build_graphrag(chunks)])\n"
    )
    outputs = {
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, env={**os.environ, "PYTHONHASHSEED": str(seed)},
                       capture_output=True, text=True, check=True).stdout
        for seed in range(8)
    }
    assert len(outputs) == 1, outputs
    assert outputs.pop().count("(") == 4  # four communities, three of them the same size


def test_fixture_mtimes_puts_back_the_manifest_mtimes_and_refuses_on_drift(tmp_path):
    """make fixture-mtimes: for the served indexer, which reads disk mtimes (the API side of the parity run)."""
    from app.eval.corpus import apply_manifest_mtimes, load_manifest, mtime_mismatches

    files = _fixture_copy(tmp_path, lambda rel, entry: time.time()) / "bench" / "corpus" / "files"
    manifest = load_manifest(files.parent / "MANIFEST.json")
    contents = {e["path"]: (files / e["path"]).read_bytes() for e in manifest["files"]}
    assert len(mtime_mismatches(files, manifest)) == manifest["n_files"]

    assert apply_manifest_mtimes(files, manifest) == manifest["n_files"]

    assert mtime_mismatches(files, manifest) == []
    assert {p: (files / p).read_bytes() for p in contents} == contents
    (files / "README.md").write_text("edited\n")
    os.utime(files / "src" / "fusion.py", (0, 0))
    with pytest.raises(RuntimeError, match="corpus drift"):
        apply_manifest_mtimes(files, manifest)
    assert (files / "src" / "fusion.py").stat().st_mtime == 0  # nothing was set while the fixture disagreed


def test_rewritten_line_endings_are_drift_not_other_numbers(tmp_path):
    """core.autocrlf=true (the Git for Windows default) checks LF files out as CRLF. The MANIFEST's sha256 is of
    read_text(), which turns CRLF back into LF, but the loader reads bytes, so the rows would change silently."""
    from app.eval.corpus import load_manifest, verify_manifest

    files = _fixture_copy(tmp_path, lambda rel, entry: entry["mtime"]) / "bench" / "corpus" / "files"
    manifest = load_manifest(files.parent / "MANIFEST.json")
    assert verify_manifest(files, manifest) == []
    target = files / "src" / "fusion.py"
    lf = target.read_bytes()
    crlf = lf.replace(b"\n", b"\r\n")
    assert b"\r" not in lf and len(crlf) > len(lf)
    target.write_bytes(crlf)

    assert target.read_text() == lf.decode()  # what the sha256 sees: unchanged
    assert verify_manifest(files, manifest) == [
        f"size mismatch: src/fusion.py ({len(crlf)} bytes on disk, {len(lf)} in the manifest; line endings rewritten?)"]
    with pytest.raises(RuntimeError, match="size mismatch: src/fusion.py"):
        run_bench(root=tmp_path, configs=[replace(DEFAULT_CONFIGS[0])], out=tmp_path / "x.json")


def test_git_checks_the_fixture_out_byte_for_byte():
    """.gitattributes: no end-of-line conversion under bench/corpus, whatever core.autocrlf says."""
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    paths = ["bench/corpus/files/src/fusion.py", "bench/corpus/MANIFEST.json"]
    out = subprocess.run(["git", "check-attr", "text", "--", *paths], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    assert out.splitlines() == [f"{p}: text: unset" for p in paths]
