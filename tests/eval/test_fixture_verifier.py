"""Fixture v1 verification: every way a copy can drift is reported, and the parity run refuses a drifted copy.

All tampering happens on a temporary copy of bench/corpus; the committed fixture is only read.
"""

import json
import shutil
from pathlib import Path

import pytest

from app.eval import parity
from app.eval.corpus import aggregate_sha256, load_manifest, verify_manifest
from app.eval.hashing import content_hash

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "bench" / "corpus"


def _copy(base: Path) -> tuple[Path, Path]:
    """bench/corpus/{files,MANIFEST.json} and bench/gold under `base`; returns (files root, manifest path)."""
    corpus = base / "bench" / "corpus"
    shutil.copytree(CORPUS / "files", corpus / "files")
    shutil.copy2(CORPUS / "MANIFEST.json", corpus / "MANIFEST.json")
    shutil.copytree(ROOT / "bench" / "gold", base / "bench" / "gold")
    return corpus / "files", corpus / "MANIFEST.json"


def _rewrite_manifest(path: Path, change) -> None:
    manifest = json.loads(path.read_text())
    change(manifest)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def test_the_aggregate_is_recomputed_from_the_entries_the_way_the_freeze_wrote_it():
    """bench/corpus/build.py hashes "path:sha256" of every file in path order; the committed MANIFEST agrees."""
    manifest = load_manifest(CORPUS / "MANIFEST.json")
    assert aggregate_sha256(manifest["files"]) == manifest["aggregate_sha256"]
    assert aggregate_sha256(list(reversed(manifest["files"]))) == manifest["aggregate_sha256"]  # order of listing is not a change
    assert manifest["n_files"] == len(manifest["files"])


def test_the_untouched_copy_verifies(tmp_path):
    files, manifest_path = _copy(tmp_path)
    assert verify_manifest(files, load_manifest(manifest_path)) == []


def _crlf(files: Path, manifest_path: Path) -> list[str]:
    """What core.autocrlf=true does on checkout. The digest is of read_text(), which reads CRLF as LF: only the size sees it."""
    target = files / "README.md"
    lf = target.read_bytes()
    crlf = lf.replace(b"\n", b"\r\n")
    assert b"\r" not in lf and len(crlf) > len(lf)
    target.write_bytes(crlf)
    return [f"size mismatch: README.md ({len(crlf)} bytes on disk, {len(lf)} in the manifest; line endings rewritten?)"]


def _zeroed_aggregate(files: Path, manifest_path: Path) -> list[str]:
    _rewrite_manifest(manifest_path, lambda m: m.update(aggregate_sha256="0" * 64))
    recomputed = aggregate_sha256(load_manifest(manifest_path)["files"])
    return [f"aggregate mismatch: aggregate_sha256 is {'0' * 64}, the listed entries hash to {recomputed}"]


def _extra(files: Path, manifest_path: Path) -> list[str]:
    (files / "notes").mkdir()
    (files / "notes" / "extra.md").write_text("not in the manifest\n")
    return ["untracked file: notes/extra.md"]


def _missing(files: Path, manifest_path: Path) -> list[str]:
    (files / "src" / "fusion.py").unlink()  # the temporary copy's file, not the fixture's
    return ["missing file: src/fusion.py"]


def _edited(files: Path, manifest_path: Path) -> list[str]:
    target = files / "configs" / "run_047.yaml"
    text = target.read_text()
    assert text.count("val_rmse: 0.0521") == 1
    target.write_text(text.replace("val_rmse: 0.0521", "val_rmse: 0.0522"))  # same size: only the digest sees it
    return ["hash mismatch: configs/run_047.yaml"]


def _entry_rewritten_to_match(files: Path, manifest_path: Path) -> list[str]:
    """An edit whose manifest entry was updated to match: every per-file check passes, the aggregate does not."""
    _edited(files, manifest_path)
    digest = content_hash((files / "configs" / "run_047.yaml").read_text())

    def change(m):
        for entry in m["files"]:
            if entry["path"] == "configs/run_047.yaml":
                entry["sha256"] = digest

    _rewrite_manifest(manifest_path, change)
    recomputed = aggregate_sha256(load_manifest(manifest_path)["files"])
    stored = load_manifest(manifest_path)["aggregate_sha256"]
    return [f"aggregate mismatch: aggregate_sha256 is {stored}, the listed entries hash to {recomputed}"]


def _n_files(files: Path, manifest_path: Path) -> list[str]:
    _rewrite_manifest(manifest_path, lambda m: m.update(n_files=m["n_files"] + 1))
    n = len(load_manifest(manifest_path)["files"])
    return [f"n_files mismatch: n_files is {n + 1}, the manifest lists {n} files"]


@pytest.mark.parametrize("tamper", [_crlf, _zeroed_aggregate, _extra, _missing, _edited, _entry_rewritten_to_match, _n_files],
                         ids=lambda f: f.__name__.strip("_"))
def test_each_kind_of_drift_is_reported_and_nothing_else(tmp_path, tamper):
    files, manifest_path = _copy(tmp_path)
    expected = tamper(files, manifest_path)
    assert verify_manifest(files, load_manifest(manifest_path)) == expected


def test_parity_refuses_a_drifted_copy_before_it_contacts_the_api(tmp_path, monkeypatch):
    files, manifest_path = _copy(tmp_path)
    _crlf(files, manifest_path)

    def must_not_run(*args, **kwargs):
        raise AssertionError("parity went past the corpus check")

    for name in ("wait_for_health", "run_config", "replay_api"):
        monkeypatch.setattr(parity, name, must_not_run)
    with pytest.raises(RuntimeError, match="corpus drift:\nsize mismatch: README.md"):
        parity.run("http://127.0.0.1:9", root=tmp_path)


def test_parity_checks_the_copy_and_then_waits_for_the_api(tmp_path, monkeypatch):
    _copy(tmp_path)

    class Reached(Exception):
        pass

    def reached(*args, **kwargs):
        raise Reached

    monkeypatch.setattr(parity, "wait_for_health", reached)
    with pytest.raises(Reached):
        parity.run("http://127.0.0.1:9", root=tmp_path)
