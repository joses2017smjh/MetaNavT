"""The served indexer and the benchmark loader must cut the same chunks.

No database: build_nodes() is what app/engine/ingest.index_directory embeds and
writes to pgvector, load_chunks() is what the in-memory benchmark indexes. Both
are compared on the frozen fixture. tests/database/test_incremental_index.py
repeats the id comparison on the rows a live Postgres holds.
"""

from pathlib import Path

from llama_index.core.schema import MetadataMode

from app.engine.ingest import build_nodes, max_rejected_from_env, policy_from_env
from app.eval.index_loader import load_chunks
from app.eval.ingest import SERVED_MAX_BYTES, SERVED_POLICY, SERVED_SUFFIXES, scan_directory

FIXTURE = Path(__file__).resolve().parents[2] / "bench" / "corpus" / "files"


def test_served_nodes_have_the_benchmark_chunk_boundaries_on_the_fixture():
    scan = scan_directory(FIXTURE, SERVED_POLICY)
    assert scan.rejected == []  # the served policy accepts every fixture file
    nodes = [node for source in scan.files for node in build_nodes(source)]
    bench = load_chunks(FIXTURE)

    served = [(n.node_id, n.metadata["path"], n.metadata["start_byte"], n.metadata["end_byte"], n.text) for n in nodes]
    assert served == [(c.chunk_id, c.path, c.start_byte, c.end_byte, c.text) for c in bench]
    assert [n.metadata["content_hash"] for n in nodes] == [c.content_hash for c in bench]
    assert [n.metadata["text_is_verbatim"] for n in nodes] == [c.metadata["text_is_verbatim"] for c in bench]
    assert len({n.node_id for n in nodes}) == len(nodes)


def test_nodes_embed_the_chunk_text_alone_and_cite_bytes():
    (source,) = [f for f in scan_directory(FIXTURE, SERVED_POLICY).files if f.path == "results/ablation.csv"]
    raw = (FIXTURE / "results" / "ablation.csv").read_bytes()
    nodes = build_nodes(source)
    header = nodes[0].text
    for node in nodes:
        assert node.get_content(metadata_mode=MetadataMode.EMBED) == node.text  # metadata stays out of the vectors
        assert node.ref_doc_id == "results/ablation.csv"
        cited = raw[node.metadata["start_byte"]:node.metadata["end_byte"]].decode("utf-8")
        assert node.text == (cited if node.metadata["text_is_verbatim"] else header + cited)
    assert [n.metadata["kind"] for n in nodes][:2] == ["csv_header", "csv_row"]


def test_policy_and_limit_come_from_the_environment(monkeypatch):
    for name in ("INGEST_SUFFIXES", "INGEST_MAX_FILE_BYTES", "INGEST_MAX_REJECTED"):
        monkeypatch.delenv(name, raising=False)
    assert policy_from_env() == SERVED_POLICY and max_rejected_from_env() is None
    assert (SERVED_POLICY.suffixes, SERVED_POLICY.max_bytes, SERVED_POLICY.dedup) == (SERVED_SUFFIXES, SERVED_MAX_BYTES, True)
    monkeypatch.setenv("INGEST_SUFFIXES", "yaml, .MD")
    monkeypatch.setenv("INGEST_MAX_FILE_BYTES", "2048")
    monkeypatch.setenv("INGEST_MAX_REJECTED", "0")
    policy = policy_from_env()
    assert (policy.suffixes, policy.max_bytes, max_rejected_from_env()) == (frozenset({".yaml", ".md"}), 2048, 0)
    monkeypatch.setenv("INGEST_SUFFIXES", "*")
    assert policy_from_env().suffixes is None
