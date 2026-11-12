"""Pure ingestion logic: scan gates, content-hash diff, accounting, rejection limit. No database."""

import hashlib
import json
from pathlib import Path

import pytest

from app.chunking.structure import chunk_text
from app.eval.hashing import content_hash
from app.eval.index_loader import build_index, load_chunks, load_chunks_with_report
from app.eval.ingest import (
    BENCH_POLICY,
    REJECT_REASONS,
    SERVED_POLICY,
    IngestPolicy,
    IngestRejected,
    file_chunks,
    plan_and_report,
    plan_ingest,
    row_mismatches,
    scan_directory,
)
from app.eval.reindex import files_to_remove, sha256_file
from app.retrieval.hybrid import chunk_id_for

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "bench" / "corpus" / "files"
# sha256 of fixture v1's chunks (id, path, byte range, kind, searched text, cited text) as the chunker of
# bench/results/main.json cuts them, computed at fe666cb. A chunker change that moves a fixture chunk fails here.
FIXTURE_CHUNKS_SHA256 = "c49edb286b982303b8fb5fdeead02ab3c6bfd9c75fea5a95881d8f8972eeaad3"


def _messy_corpus(root: Path) -> None:
    (root / "configs").mkdir(parents=True)
    (root / "configs" / "run_001.yaml").write_text("run_id: 1\nlearning_rate: 0.0003\n")
    (root / "configs" / "run_001_copy.yaml").write_text("run_id: 1\nlearning_rate: 0.0003\n")  # duplicate_content
    (root / "notes.md").write_text("# Notes\nfusion helps\n")
    (root / "broken.out").write_bytes(b"epoch 1\n\xff\xfe")  # undecodable
    (root / "blank.txt").write_text("\n  \n")  # empty
    (root / "empty.csv").write_bytes(b"")  # empty
    (root / "weights.ckpt").write_bytes(b"\x00\x01")  # unsupported_type
    (root / "long.log").write_text("x" * 300)  # too_large under the test policy


def test_scan_counts_every_rejection_by_reason(tmp_path):
    _messy_corpus(tmp_path)
    policy = IngestPolicy(suffixes=SERVED_POLICY.suffixes, max_bytes=200, dedup=True)

    scan = scan_directory(tmp_path, policy)

    assert [f.path for f in scan.files] == ["configs/run_001.yaml", "notes.md"]
    assert {row["path"]: row["reason"] for row in scan.rejected} == {
        "blank.txt": "empty",
        "broken.out": "undecodable",
        "configs/run_001_copy.yaml": "duplicate_content",  # the first path in sorted order keeps the content
        "empty.csv": "empty",
        "long.log": "too_large",
        "weights.ckpt": "unsupported_type",
    }
    assert scan.seen == 8
    assert {row["reason"] for row in scan.rejected} <= set(REJECT_REASONS)


def test_file_identity_is_the_sha256_of_the_raw_bytes(tmp_path):
    path = tmp_path / "ablation.csv"
    path.write_bytes("ablation,rmse\r\nfusion_on,0.0542\r\nλ,0.1\r\n".encode("utf-8"))
    (source,) = scan_directory(tmp_path).files
    assert source.sha256 == sha256_file(path) == content_hash(source.text)  # CRLF survives the decode
    assert source.size_bytes == path.stat().st_size


def test_plan_separates_new_changed_unchanged_and_removed():
    manifest = {"a.yaml": "h1", "b.yaml": "h2", "gone.yaml": "h3"}
    current = {"a.yaml": "h1", "b.yaml": "h2-edited", "c.yaml": "h4"}

    plan = plan_ingest(current, manifest)

    assert (plan.new, plan.changed, plan.unchanged, plan.removed) == (["c.yaml"], ["b.yaml"], ["a.yaml"], ["gone.yaml"])
    assert plan.to_index == ["b.yaml", "c.yaml"]
    assert files_to_remove(current, manifest) == ["gone.yaml"]
    same = plan_ingest(current, current)
    assert (same.to_index, same.removed, same.unchanged) == ([], [], sorted(current))  # an unchanged re-run does nothing


def test_report_accounts_for_every_file_and_the_limit_fails_the_run(tmp_path):
    _messy_corpus(tmp_path)
    policy = IngestPolicy(suffixes=SERVED_POLICY.suffixes, max_bytes=200, dedup=True)
    scan = scan_directory(tmp_path, policy)
    manifest = {"notes.md": scan.hashes()["notes.md"], "deleted.md": "old"}

    plan, report = plan_and_report(scan, manifest, max_rejected=6)

    assert report["files_seen"] == report["files_indexed"] + report["files_unchanged"] + report["files_rejected"] == 8
    assert (report["files_indexed"], report["files_unchanged"], report["files_removed"], report["files_rejected"]) == (1, 1, 1, 6)
    assert report["rejected_by_reason"] == {
        "outside_root": 0, "unsupported_type": 1, "too_large": 1, "undecodable": 1, "empty": 2, "duplicate_content": 1,
    }
    assert plan.removed == ["deleted.md"]
    with pytest.raises(IngestRejected) as refused:
        plan_and_report(scan, manifest, max_rejected=5)
    assert refused.value.report["files_rejected"] == 6 and "limit 5" in str(refused.value)
    assert plan_and_report(scan, manifest, max_rejected=None)[1]["max_rejected"] is None  # no limit: counted, not fatal


def test_a_file_that_stops_being_accepted_is_removed_from_the_index(tmp_path):
    (tmp_path / "a.md").write_text("# A\n")
    manifest = scan_directory(tmp_path, SERVED_POLICY).hashes()
    (tmp_path / "a.md").write_bytes(b"\xff")

    plan, report = plan_and_report(scan_directory(tmp_path, SERVED_POLICY), manifest)

    assert plan.removed == ["a.md"] and report["rejected_by_reason"]["undecodable"] == 1


def test_row_mismatches_compares_rows_per_path_with_the_manifest_chunk_counts():
    listed = {"a.yaml": 1, "b.csv": 9, "c.md": 2, "no_chunks.md": 0}
    rows = {"a.yaml": 2, "b.csv": 8, "c.md": 2, "unlisted.out": 3}

    assert row_mismatches(listed, rows) == [
        {"path": "a.yaml", "manifest_chunks": 1, "rows": 2},
        {"path": "b.csv", "manifest_chunks": 9, "rows": 8},
        {"path": "unlisted.out", "manifest_chunks": 0, "rows": 3},
    ]
    assert row_mismatches(listed, {"a.yaml": 1, "b.csv": 9, "c.md": 2}) == []  # a path missing on one side counts as 0 there


def test_bench_loader_counts_the_file_it_used_to_skip_silently(tmp_path):
    _messy_corpus(tmp_path)

    chunks, report = load_chunks_with_report(tmp_path)

    assert BENCH_POLICY == IngestPolicy()  # the benchmark keeps accepting every decodable, non-empty file
    assert report["rejected_by_reason"] == {
        "outside_root": 0, "unsupported_type": 0, "too_large": 0, "undecodable": 1, "empty": 2, "duplicate_content": 0,
    }
    assert report["files_seen"] == 8 and report["files_indexed"] == 5 and report["chunks_embedded"] == len(chunks)
    assert "broken.out" not in {c.path for c in chunks}
    assert build_index(tmp_path, embedder_name="hash", enable_rerank=False).ingest_report == report


def _root_with_links_out_of_it(tmp_path: Path) -> Path:
    """One plain file, links whose real path leaves the scanned tree, and a link that stays inside it."""
    root, outside = tmp_path / "root", tmp_path / "outside"
    (outside / "dir").mkdir(parents=True)
    (root / ".git").mkdir(parents=True)
    (root / "notes.md").write_text("# Notes\nfusion helps\n")
    (outside / "secret.txt").write_text("operator secret zqxjfilemarker\n")
    (outside / "weights.ckpt").write_bytes(b"zqxjckptmarker")
    (outside / "dir" / "log.out").write_text("outside directory zqxjdirmarker\n")
    (root / ".git" / "config").write_text("[remote] url = https://zqxjgitmarker@example.invalid/repo\n")
    (root / "link.txt").symlink_to(outside / "secret.txt")
    (root / "link.ckpt").symlink_to(outside / "weights.ckpt")  # confinement is the first gate, before the type
    (root / "linkdir").symlink_to(outside / "dir", target_is_directory=True)  # a directory link is not walked
    (root / "gitconfig.txt").symlink_to(root / ".git" / "config")  # inside the root, but in a directory the scan skips
    (root / "notes_alias.md").symlink_to(root / "notes.md")  # stays inside the scanned tree: read like any file
    return root


def test_a_symlink_out_of_the_root_is_rejected_as_outside_root_and_its_text_is_in_no_chunk(tmp_path):
    root = _root_with_links_out_of_it(tmp_path)
    outside = {"gitconfig.txt": "outside_root", "link.ckpt": "outside_root", "link.txt": "outside_root"}

    bench, served = scan_directory(root, BENCH_POLICY), scan_directory(root, SERVED_POLICY)

    assert [f.path for f in bench.files] == ["notes.md", "notes_alias.md"]
    assert {row["path"]: row["reason"] for row in bench.rejected} == outside
    assert [f.path for f in served.files] == ["notes.md"]
    assert {row["path"]: row["reason"] for row in served.rejected} == {**outside, "notes_alias.md": "duplicate_content"}
    assert REJECT_REASONS[0] == "outside_root" and bench.seen == served.seen == 5  # the directory link is never walked
    chunks, report = load_chunks_with_report(root)
    served_chunks = [piece for source in served.files for piece in file_chunks(source.path, source.text)]
    assert chunks and served_chunks
    assert not [c for c in chunks if "zqxj" in c.text or "zqxj" in c.metadata["evidence_raw"]]
    assert not [piece for piece in served_chunks if "zqxj" in piece.text or "zqxj" in piece.raw]
    assert report["rejected_by_reason"]["outside_root"] == 3 and report["files_seen"] == report["files_indexed"] + 3
    assert {row["path"] for row in report["rejected_sample"]} == set(outside)  # the link's own path, never its target


def test_a_root_reached_through_a_symlinked_prefix_is_scanned_normally(tmp_path):
    real = tmp_path / "real"
    (real / "configs").mkdir(parents=True)
    (real / "configs" / "run_001.yaml").write_text("run_id: 1\n")
    (tmp_path / "via").symlink_to(real, target_is_directory=True)

    scan = scan_directory(tmp_path / "via", SERVED_POLICY)

    assert [f.path for f in scan.files] == ["configs/run_001.yaml"] and scan.rejected == []


def _chunks_as_loaded_before_m8(root: Path) -> list[tuple]:
    """The loader body that produced bench/results/main.json, kept as the reference for the fixture."""
    out = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root)).replace("\\", "/")
        text = path.read_bytes().decode("utf-8")
        digest = content_hash(text)
        for span in chunk_text(text, path=rel, strategy="auto") or chunk_text(text, path=rel, strategy="fixed"):
            start, end = len(text[:span.start].encode("utf-8")), len(text[:span.end].encode("utf-8"))
            out.append((chunk_id_for(rel, start, end), rel, span.text, start, end, digest, span.kind, text[span.start:span.end]))
    return out


def test_fixture_chunks_are_the_ones_the_recorded_benchmark_used():
    chunks, report = load_chunks_with_report(FIXTURE)
    got = [(c.chunk_id, c.path, c.text, c.start_byte, c.end_byte, c.content_hash, c.metadata["kind"], c.metadata["evidence_raw"]) for c in chunks]
    assert got == _chunks_as_loaded_before_m8(FIXTURE)
    recorded = json.loads((ROOT / "bench" / "results" / "main.json").read_text())
    assert {row["n_chunks"] for row in recorded["results"]} == {len(chunks)}
    assert (report["files_seen"], report["files_indexed"], report["files_rejected"]) == (recorded["n_files"], recorded["n_files"], 0)
    assert [c.chunk_id for c in load_chunks(FIXTURE)] == [c[0] for c in got]


def test_fixture_chunks_match_the_digest_pinned_from_the_recorded_benchmark():
    """Independent of the live chunker: the reference loader above calls the same chunk_text."""
    rows = [[c.chunk_id, c.path, c.start_byte, c.end_byte, c.metadata["kind"], c.text, c.metadata["evidence_raw"]]
            for c in load_chunks(FIXTURE)]
    assert hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode("utf-8")).hexdigest() == FIXTURE_CHUNKS_SHA256


def test_file_chunks_addresses_bytes_and_keeps_the_searched_text():
    text = "ablation,rmse\r\nλ_on,0.0542\r\n"
    header, row = file_chunks("results/ablation.csv", text)
    assert (header.start_byte, header.end_byte, row.start_byte, row.end_byte) == (0, 15, 15, len(text.encode("utf-8")))
    assert row.text == "ablation,rmse\r\nλ_on,0.0542\r\n" and row.raw == "λ_on,0.0542\r\n"  # header prepended for search only
    assert text.encode("utf-8")[row.start_byte:row.end_byte].decode("utf-8") == row.raw
