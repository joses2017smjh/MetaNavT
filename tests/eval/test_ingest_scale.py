"""The scaled corpus `make ingest-scale` indexes, and how it keeps rows from earlier runs (no database)."""

import os

from app.engine.ingest_scale import FIXTURE, MANIFEST, build_corpus, merge, perturb
from app.eval.corpus import load_manifest
from app.eval.ingest import SERVED_POLICY, file_chunks, scan_directory


def test_every_copy_is_accepted_by_the_served_scan_and_chunked_like_the_fixture(tmp_path):
    info = build_corpus(tmp_path, 3)
    scan = scan_directory(tmp_path, SERVED_POLICY)
    n_fixture = sum(1 for p in FIXTURE.rglob("*") if p.is_file())

    assert info["files"] == len(scan.files) == 3 * n_fixture and scan.rejected == []  # no duplicate bytes, nothing refused
    by_path = {f.path: f for f in scan.files}
    mtimes = {e["path"]: float(e["mtime"]) for e in load_manifest(MANIFEST)["files"]}
    for source in sorted(p for p in FIXTURE.rglob("*") if p.is_file()):
        rel = source.relative_to(FIXTURE).as_posix()
        original = by_path[f"copy_000/{rel}"]
        assert (tmp_path / "copy_000" / rel).read_bytes() == source.read_bytes()  # copy 0 is the fixture
        base = [c.kind for c in file_chunks(original.path, original.text)]
        for copy in (1, 2):
            copied = by_path[f"copy_{copy:03d}/{rel}"]
            assert os.stat(copied.abs_path).st_mtime == mtimes[rel] + copy
            kinds = [c.kind for c in file_chunks(copied.path, copied.text)]
            # the added line keeps the file's format: the same chunks, and a CSV gains the one row it was given
            assert kinds == base + (["csv_row"] if rel.endswith(".csv") else []), rel
    assert build_corpus(tmp_path / "again", 3) == info  # the same bytes on every run


def test_perturb_leaves_copy_zero_alone_and_adds_one_line_in_the_files_format():
    yaml = b"run_id: run_047\nlearning_rate: 0.0003\n"
    assert perturb("configs/run_047.yaml", yaml, 0) == yaml
    assert perturb("configs/run_047.yaml", yaml, 7) == yaml + b"# copy 7\n"
    assert perturb("src/fusion.py", b"import torch\n", 2) == b"# copy 2\nimport torch\n"
    assert perturb("logs/run_047.out", b"epoch 1", 3) == b"epoch 1\ncopy 3\n"
    assert perturb("results/ablation.csv", b"a,b,c\r\n1,2,3\r\n", 4) == b"a,b,c\r\n1,2,3\r\ncopy 4,,\r\n"
    assert perturb("checkpoints/x.json", b'{"step": 1}', 5) == b'{\n  "step": 1,\n  "copy": 5\n}\n'


def test_a_run_replaces_its_own_rows_and_keeps_every_other_row_as_written():
    before = [{"label": "before", "copies": c, "seconds": 1.0 + c} for c in (1, 10)]
    first = merge(None, "before", before)
    assert first["rows"] == before and first["fixture_sha256"] == load_manifest(MANIFEST)["aggregate_sha256"]

    after = [{"label": "after", "copies": 1, "seconds": 0.5}]
    second = merge(first, "after", after)
    assert second["rows"] == before + after  # before rows untouched, in the order the file keeps

    redone = [{"label": "after", "copies": 1, "seconds": 0.4}, {"label": "after", "copies": 10, "seconds": 0.9}]
    third = merge(second, "after", redone)
    assert third["rows"] == before + redone  # the earlier after row is replaced, the before rows are not
