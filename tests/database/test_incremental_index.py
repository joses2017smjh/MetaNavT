"""Incremental indexing against a live Postgres: manifest diff, per-file replace, advisory lock.

Needs a live Postgres with pgvector: set PG_TEST_DSN (the CI parity service, or a
local instance). Skipped otherwise. Every test indexes a temporary copy of
fixture v1 (never bench/corpus/files itself) into its own table, built by the
schema migrations as the deploy step builds the served one, with the hash
embedder, so there are no model downloads.
"""

import os
import shutil
import threading
import time
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("PG_TEST_DSN"), reason="PG_TEST_DSN not set")

FIXTURE = Path(__file__).resolve().parents[2] / "bench" / "corpus" / "files"
EDITED = "configs/run_047.yaml"
DELETED = "logs/run_046.out"


class Run:
    """One table, one corpus copy, and as many managers (replicas) as a test opens."""

    def __init__(self, dsn: str, corpus: Path):
        from app.retrieval.embedders import HashEmbedder

        self.dsn = dsn
        self.corpus = corpus
        self.embedder = HashEmbedder(256)
        self.embedded: list[str] = []  # every chunk text sent to the embedder
        self.managers = []

    def manager(self):
        from app.database.vector_store_manager import VectorStoreManager

        vsm = VectorStoreManager(conn_string=self.dsn)
        self.managers.append(vsm)
        return vsm

    def embed(self, texts):
        self.embedded.extend(texts)
        return self.embedder.encode(texts).tolist()

    def index(self, vsm=None, **kwargs):
        from app.engine.ingest import index_directory

        vsm = vsm or self.managers[0]
        kwargs.setdefault("embed_fn", self.embed)
        return index_directory(vsm, str(self.corpus), **kwargs)

    def rows(self) -> list[tuple]:
        """(serial id, node_id, path) for every chunk row; the serial id changes when a row is rewritten."""
        vsm = self.managers[0]
        with vsm.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f'SELECT id, node_id, metadata_->>\'path\' FROM "{vsm.schema_name}"."{vsm.data_table}" ORDER BY id;')
                return cur.fetchall()

    def expected_ids(self) -> list[str]:
        """Row ids for the chunks the benchmark loader cuts from the same directory."""
        from app.engine.ingest import node_id_for
        from app.eval.index_loader import load_chunks

        return sorted(node_id_for(c.path, c.start_byte, c.end_byte) for c in load_chunks(self.corpus))

    def assert_manifest_matches_rows(self):
        """The manifest lists exactly the files that have rows, with the right chunk counts."""
        per_path: dict[str, int] = {}
        for _, _, path in self.rows():
            per_path[path] = per_path.get(path, 0) + 1
        manifest = {p: row["n_chunks"] for p, row in self.managers[0].ingest_manifest().items()}
        assert manifest == per_path


@pytest.fixture
def run(tmp_path, monkeypatch):
    dsn = os.environ["PG_TEST_DSN"]
    monkeypatch.setenv("PG_CONNECTION_STRING", dsn)
    monkeypatch.setenv("PSYCOPG2_CONNECTION_STRING", dsn)
    monkeypatch.setenv("PGVECTOR_TABLE", "ingesttest_" + uuid.uuid4().hex[:8])
    monkeypatch.setenv("EMBEDDING_DIM", "256")
    for name in ("INGEST_SUFFIXES", "INGEST_MAX_FILE_BYTES", "INGEST_MAX_REJECTED"):
        monkeypatch.delenv(name, raising=False)
    corpus = tmp_path / "files"
    shutil.copytree(FIXTURE, corpus)
    from app.database.migrate import drop_vector_table, migrate_all

    r = Run(dsn, corpus)
    vsm = r.manager()
    with vsm.get_connection() as conn:
        migrate_all(conn, vsm.vector_table)  # the data table and the manifest: index_directory runs no DDL
    yield r
    with vsm.get_connection() as conn:
        drop_vector_table(conn, vsm.vector_table)
    for manager in r.managers:
        manager.close()


def test_first_run_indexes_everything_and_an_unchanged_rerun_embeds_nothing(run):
    n_files = sum(1 for p in run.corpus.rglob("*") if p.is_file())
    first = run.index()
    expected = run.expected_ids()
    assert (first["files_seen"], first["files_indexed"], first["files_unchanged"], first["files_rejected"]) == (n_files, n_files, 0, 0)
    assert first["chunks_embedded"] == len(run.embedded) == len(expected) == first["n_nodes"]
    assert sorted(node_id for _, node_id, _ in run.rows()) == expected  # one row per benchmark chunk
    run.assert_manifest_matches_rows()

    before, run.embedded = run.rows(), []
    second = run.index()
    assert run.embedded == [] and second["chunks_embedded"] == 0 and second["indexed"] is False
    assert (second["files_seen"], second["files_indexed"], second["files_unchanged"], second["files_removed"]) == (n_files, 0, n_files, 0)
    assert run.rows() == before  # not one row rewritten

    third = run.index(embedder="another-model")  # rows made by another embedder or chunker version are stale
    assert (third["files_indexed"], third["indexed_changed"], third["files_unchanged"]) == (n_files, n_files, 0)
    assert sorted(node_id for _, node_id, _ in run.rows()) == expected


def test_an_edit_reembeds_only_that_file_and_a_delete_removes_its_rows(run):
    from app.eval.ingest import file_chunks

    run.index()
    before, run.embedded = run.rows(), []
    target = run.corpus / EDITED
    target.write_text(target.read_text().replace("0.0003", "0.0004") + "note: edited\n")
    (run.corpus / "notes_new.md").write_text("# New note\nadded after the first run\n")
    (run.corpus / DELETED).unlink()

    report = run.index()

    assert (report["files_indexed"], report["indexed_changed"], report["indexed_new"], report["files_removed"]) == (2, 1, 1, 1)
    assert report["files_unchanged"] == report["files_seen"] - 2
    edited_texts = [c.text for c in file_chunks(EDITED, target.read_text())]
    assert run.embedded == edited_texts + ["# New note\nadded after the first run\n"]  # only the two files' chunks
    after = run.rows()
    untouched = [row for row in before if row[2] not in {EDITED, DELETED}]
    assert [row for row in after if row[2] not in {EDITED, "notes_new.md"}] == untouched  # same serial ids: rows not rewritten
    assert not [row for row in after if row[2] == DELETED]
    assert sorted(node_id for _, node_id, _ in after) == run.expected_ids()
    assert DELETED not in run.managers[0].ingest_manifest()
    run.assert_manifest_matches_rows()


def test_rejections_are_counted_by_reason_and_a_limit_fails_the_run_before_any_write(run):
    from app.eval.ingest import SERVED_SUFFIXES, IngestPolicy, IngestRejected

    run.index()
    before = run.rows()
    (run.corpus / "binary.yaml").write_bytes(b"\xff\xfe\x00not utf-8")
    (run.corpus / "blank.md").write_text(" \n\n")
    (run.corpus / "weights.bin").write_bytes(b"\x00" * 16)
    (run.corpus / "huge.out").write_text("x" * 5000)
    shutil.copyfile(run.corpus / EDITED, run.corpus / "zz_copy_of_run_047.yaml")
    (run.corpus / DELETED).unlink()  # a pending change the refused run must not apply
    policy = IngestPolicy(suffixes=SERVED_SUFFIXES, max_bytes=4096, dedup=True)

    with pytest.raises(IngestRejected) as refused:
        run.index(policy=policy, max_rejected=4)
    assert refused.value.report["files_rejected"] == 5
    assert run.rows() == before  # nothing deleted, nothing written

    report = run.index(policy=policy, max_rejected=5)
    assert report["rejected_by_reason"] == {
        "outside_root": 0, "unsupported_type": 1, "too_large": 1, "undecodable": 1, "empty": 1, "duplicate_content": 1,
    }
    assert report["files_seen"] == report["files_indexed"] + report["files_unchanged"] + report["files_rejected"]
    assert report["files_removed"] == 1
    assert {row["path"] for row in report["rejected_sample"]} == {"binary.yaml", "blank.md", "weights.bin", "huge.out", "zz_copy_of_run_047.yaml"}
    run.assert_manifest_matches_rows()


def test_a_link_out_of_the_data_dir_is_rejected_and_no_row_holds_its_text(run, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.yaml").write_text("api_token: zqxjsecretmarker\n")
    (run.corpus / "configs" / "linked.yaml").symlink_to(outside / "secret.yaml")

    report = run.index()

    assert report["rejected_by_reason"]["outside_root"] == report["files_rejected"] == 1
    assert report["rejected_sample"] == [{"path": "configs/linked.yaml", "reason": "outside_root"}]
    assert run.embedded and not [text for text in run.embedded if "zqxj" in text]  # never sent to the embedder
    assert "configs/linked.yaml" not in {path for _, _, path in run.rows()} | set(run.managers[0].ingest_manifest())
    run.assert_manifest_matches_rows()


def test_an_interrupted_run_never_leaves_the_manifest_ahead_of_the_rows(run):
    calls = {"n": 0}

    def failing_embed(texts):
        calls["n"] += 1
        if calls["n"] == 10:
            raise RuntimeError("embedder died")
        return run.embed(texts)

    with pytest.raises(RuntimeError, match="embedder died"):
        run.index(embed_fn=failing_embed)
    assert len(run.managers[0].ingest_manifest()) == 9
    run.assert_manifest_matches_rows()

    # Crash between inserting a file's chunks and listing it: rows exist that the manifest does not know.
    vsm = run.managers[0]
    real_record, state = vsm.record_file, {"armed": True}

    def record_once_fails(*args, **kwargs):
        if state.pop("armed", False):
            raise RuntimeError("killed before the manifest write")
        return real_record(*args, **kwargs)

    vsm.record_file = record_once_fails
    with pytest.raises(RuntimeError, match="killed before the manifest write"):
        run.index()
    assert len(run.rows()) > sum(row["n_chunks"] for row in vsm.ingest_manifest().values())  # the unlisted file's rows

    report = run.index()  # the repair run
    assert report["files_unchanged"] == 9 and report["files_indexed"] == report["files_seen"] - 9
    assert sorted(node_id for _, node_id, _ in run.rows()) == run.expected_ids()  # one set of rows, no duplicates
    run.assert_manifest_matches_rows()

    # An embedder failure on an edited file leaves that file's previous rows and manifest entry in place.
    before, old_hash = run.rows(), vsm.ingest_manifest()[EDITED]["content_hash"]
    with (run.corpus / EDITED).open("a") as handle:
        handle.write("note: edited\n")
    calls["n"] = 9  # the next embed call fails
    with pytest.raises(RuntimeError, match="embedder died"):
        run.index(embed_fn=failing_embed)
    assert run.rows() == before and vsm.ingest_manifest()[EDITED]["content_hash"] == old_hash
    assert run.index()["indexed_changed"] == 1
    assert sorted(node_id for _, node_id, _ in run.rows()) == run.expected_ids()


def test_a_database_failure_while_a_file_is_written_leaves_it_unlisted_until_the_next_run(run, monkeypatch):
    """The file in flight does not keep its old rows: they and its manifest entry go just before the insert."""
    run.index()
    vsm = run.managers[0]
    store_type, state = type(vsm.get_vector_store()), {"armed": True}
    real_add = store_type.add

    def add_once_fails(self, nodes, **kwargs):
        if state.pop("armed", False):
            raise RuntimeError("database gone during the insert")
        return real_add(self, nodes, **kwargs)

    with (run.corpus / EDITED).open("a") as handle:
        handle.write("note: edited\n")
    monkeypatch.setattr(store_type, "add", add_once_fails)
    with pytest.raises(RuntimeError, match="database gone during the insert"):
        run.index()
    assert EDITED not in vsm.ingest_manifest() and EDITED not in {path for _, _, path in run.rows()}

    report = run.index()  # the next start indexes it again, as a new file

    assert (report["files_indexed"], report["indexed_new"], report["reconciliation"]["files_mismatched_at_start"]) == (1, 1, 0)
    assert sorted(node_id for _, node_id, _ in run.rows()) == run.expected_ids()
    run.assert_manifest_matches_rows()


def _table_sql(run, statement: str, *params) -> None:
    """Change the data table directly, the way a manual edit or another writer would (behind the manifest's back)."""
    vsm = run.managers[0]
    with vsm.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(statement.format(table=f'"{vsm.schema_name}"."{vsm.data_table}"'), params)


def test_rows_changed_behind_the_manifests_back_are_reported_and_repaired_by_the_next_run(run):
    from app.eval.ingest import file_chunks

    run.index()
    short, extra = "results/ablation.csv", EDITED  # one loses a row, the other gains a duplicate
    _table_sql(run, "DELETE FROM {table} WHERE id = (SELECT max(id) FROM {table} WHERE metadata_->>'path' = %s);", short)
    _table_sql(run, "INSERT INTO {table} (text, metadata_, node_id, embedding) "
                    "SELECT text, metadata_, node_id || '-copy', embedding FROM {table} WHERE metadata_->>'path' = %s;", extra)
    listed = {p: row["n_chunks"] for p, row in run.managers[0].ingest_manifest().items()}
    run.embedded = []

    report = run.index()

    assert report["reconciliation"]["files_mismatched_at_start"] == 2
    assert report["reconciliation"]["mismatched_sample"] == [
        {"path": extra, "manifest_chunks": listed[extra], "rows": listed[extra] + 1},
        {"path": short, "manifest_chunks": listed[short], "rows": listed[short] - 1},
    ]
    assert report["reconciliation"]["files_mismatched_after_run"] == 0
    assert (report["files_indexed"], report["indexed_changed"], report["files_unchanged"]) == (2, 2, report["files_seen"] - 2)
    assert run.embedded == [c.text for path in (extra, short) for c in file_chunks(path, (run.corpus / path).read_bytes().decode("utf-8"))]
    assert sorted(node_id for _, node_id, _ in run.rows()) == run.expected_ids()
    run.assert_manifest_matches_rows()
    again = run.index()
    assert again["reconciliation"]["files_mismatched_at_start"] == 0 and again["chunks_embedded"] == 0


def test_a_run_whose_rows_do_not_match_its_manifest_fails_and_the_next_run_repairs_it(run):
    from app.eval.ingest import IngestUnreconciled

    run.index()
    vsm = run.managers[0]
    real_record = vsm.record_file

    def record_one_chunk_too_many(path, content_hash, n_chunks, pipeline="", **staleness):
        return real_record(path, content_hash, n_chunks + 1, pipeline, **staleness)

    with (run.corpus / EDITED).open("a") as handle:
        handle.write("note: edited\n")
    vsm.record_file = record_one_chunk_too_many  # the manifest claims a chunk the table does not hold
    with pytest.raises(IngestUnreconciled) as unreconciled:
        run.index()
    vsm.record_file = real_record
    (row,) = unreconciled.value.mismatches
    assert row["path"] == EDITED and row["manifest_chunks"] == row["rows"] + 1
    assert unreconciled.value.report["reconciliation"]["files_mismatched_after_run"] == 1

    report = run.index()  # the next start finds the mismatch and indexes that file again

    assert report["reconciliation"]["files_mismatched_at_start"] == 1 and report["files_indexed"] == 1
    assert sorted(node_id for _, node_id, _ in run.rows()) == run.expected_ids()
    run.assert_manifest_matches_rows()


def test_two_replicas_starting_together_index_once(run):
    first, second = run.managers[0], run.manager()
    barrier = threading.Barrier(2)
    reports: dict[str, dict] = {}
    errors: list[BaseException] = []

    overlapped = threading.Event()

    def wait_until_the_other_replica_is_blocked(texts):
        # Whichever run holds the lock keeps it until the other one is waiting on it, so the two runs really overlap.
        import psycopg2

        if not overlapped.is_set():
            probe = psycopg2.connect(run.dsn)
            probe.autocommit = True
            deadline = time.time() + 30
            with probe.cursor() as cur:
                while time.time() < deadline and not overlapped.is_set():
                    cur.execute("SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND NOT granted;")
                    if cur.fetchone()[0] > 0:
                        overlapped.set()
                    else:
                        time.sleep(0.05)
            probe.close()
        return run.embed(texts)

    def start(name, vsm):
        try:
            barrier.wait(timeout=30)
            reports[name] = run.index(vsm, embed_fn=wait_until_the_other_replica_is_blocked)
        except BaseException as exc:  # noqa: BLE001 - reported by the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=start, args=("a", first)), threading.Thread(target=start, args=("b", second))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert not errors, errors
    assert overlapped.is_set(), "the second run never waited on the lock"

    n_files = reports["a"]["files_seen"]
    winner, waiter = sorted(reports.values(), key=lambda r: r["lock"]["waited"])
    assert (winner["lock"]["waited"], waiter["lock"]["waited"]) == (False, True)
    assert (winner["files_indexed"], waiter["files_indexed"], waiter["files_unchanged"]) == (n_files, 0, n_files)
    assert waiter["chunks_embedded"] == 0
    node_ids = [node_id for _, node_id, _ in run.rows()]
    assert sorted(node_ids) == run.expected_ids() and len(set(node_ids)) == len(node_ids)  # a single set of rows
    run.assert_manifest_matches_rows()


def test_committed_ingest_runs_match_a_fresh_run(monkeypatch):
    """bench/results/ingest_runs.json (make ingest-runs) is what the four runs produce today."""
    import json

    from app.engine.ingest_runs import OUT, record_runs

    dsn = os.environ["PG_TEST_DSN"]
    for name in ("PG_CONNECTION_STRING", "PSYCOPG2_CONNECTION_STRING", "PGVECTOR_TABLE", "EMBEDDING_DIM"):
        monkeypatch.setenv(name, dsn)  # record_runs sets its own; monkeypatch restores the originals afterwards
    committed = json.loads(OUT.read_text())

    runs = record_runs(dsn)

    assert runs == committed["runs"]
    by_name = {row["run"]: row for row in runs}
    n_files = by_name["first_start"]["report"]["files_seen"]
    assert by_name["first_start"]["report"]["files_indexed"] == n_files
    assert by_name["unchanged_restart"]["texts_sent_to_embedder"] == 0 and by_name["unchanged_restart"]["report"]["files_unchanged"] == n_files
    edit = by_name["edit_one_file"]["report"]
    assert (edit["files_indexed"], edit["indexed_changed"], edit["files_unchanged"]) == (1, 1, n_files - 1)
    assert by_name["edit_one_file"]["texts_sent_to_embedder"] == edit["chunks_embedded"] < by_name["first_start"]["texts_sent_to_embedder"]
    delete = by_name["delete_one_file"]["report"]
    assert (delete["files_seen"], delete["files_removed"], delete["files_indexed"]) == (n_files - 1, 1, 0)
    assert by_name["delete_one_file"]["rows_in_table"] == by_name["edit_one_file"]["rows_in_table"] - delete["chunks_deleted"]
