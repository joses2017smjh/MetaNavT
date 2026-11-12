"""The chat agent's file-access tools (app/engine/tools/index_file_access.py), called the way the agent workflow
calls them: through get_tools() and the FunctionTool, with only the arguments the model gave.

No database: the vector store manager the tools read is a stand-in (or points at a closed port).
tests/database/test_file_access_tools.py runs the same tools over the fixture indexed through the served path.
"""

import asyncio
import inspect
import random
from pathlib import Path

import pytest

from app.engine.tools import index_file_access as files
from app.eval.ingest import SERVED_POLICY, file_chunks, scan_directory

MANIFEST = ["src/fusion.py", "configs/run_047.yaml", "configs/archive/run_047_v1.yaml", "README.md", "Zeta.md"]
FIXTURE = Path(__file__).resolve().parents[2] / "bench" / "corpus" / "files"


def _tool(name):
    return next(tool for tool in files.get_tools() if tool.metadata.name == name)


def _call(name, **kwargs):
    """What the workflow's _call_tool does with a tool call: await tool.acall(**the model's arguments)."""
    return asyncio.run(_tool(name).acall(**kwargs))


def stored_chunks(path: str, text: str) -> list[dict]:
    """A file's chunks as the served indexer stores them (app/engine/ingest.build_nodes): the chunk text, the UTF-8
    byte range and whether the text is the source slice itself, in a shuffled order."""
    rows = [{"text": c.text, "start_byte": c.start_byte, "end_byte": c.end_byte, "verbatim": c.text == c.raw}
            for c in file_chunks(path, text)]
    random.Random(path).shuffle(rows)
    return rows


class FakeManager:
    """The ingest manifest the list tool reads ({path: row}) and the chunk rows the get tool reads, without a database."""

    def __init__(self, paths=MANIFEST, error=None, texts=None):
        self.paths, self.error, self.texts, self.asked = list(paths), error, dict(texts or {}), []

    def ingest_manifest(self):
        if self.error:
            raise self.error
        return {path: {"content_hash": "0" * 64, "n_chunks": 1, "pipeline": "test"} for path in self.paths}

    def chunks_of_path(self, path):
        self.asked.append(path)
        if self.error:
            raise self.error
        return stored_chunks(path, self.texts[path]) if path in self.texts else []


@pytest.fixture
def manifest(monkeypatch):
    manager = FakeManager()
    monkeypatch.setattr(files, "get_vector_store_manager", lambda: manager)
    return manager


def test_the_pattern_default_is_a_string_not_a_pydantic_field():
    # The default used to be pydantic's Field(default="*"): the FieldInfo object itself, not "*".
    assert inspect.signature(files.list_available_files).parameters["pattern"].default == "*"


def test_list_available_files_called_without_a_pattern_returns_the_paths(manifest):
    out = _call("list_available_files")  # the model gave no pattern: the agent passes no arguments
    assert not out.is_error
    assert out.raw_output == sorted(MANIFEST)  # code point order: upper case before lower case


def test_a_pattern_is_a_glob_over_the_whole_relative_path(manifest):
    assert _call("list_available_files", pattern="configs/*.yaml").raw_output == [
        "configs/archive/run_047_v1.yaml", "configs/run_047.yaml"]  # `*` also crosses "/", as LIKE '%' did
    assert _call("list_available_files", pattern="*.py").raw_output == ["src/fusion.py"]
    assert _call("list_available_files", pattern="configs/run_04?.yaml").raw_output == ["configs/run_047.yaml"]
    assert _call("list_available_files", pattern="run_047*").raw_output == []  # matched against the whole path
    assert _call("list_available_files", pattern="*.MD").raw_output == []  # case-sensitive
    assert _call("list_available_files", pattern="").raw_output == sorted(MANIFEST)
    assert _call("list_available_files", pattern=" * ").raw_output == sorted(MANIFEST)


def test_a_bare_name_pattern_matches_only_paths_that_start_with_it(monkeypatch):
    # What doc/engineering-reference.md (Backend) says: `fusion*` does not find src/fusion.py, and since `*` crosses
    # "/" it does find a file below a top-level directory whose name starts with "fusion".
    monkeypatch.setattr(files, "get_vector_store_manager",
                        lambda: FakeManager(["src/fusion.py", "fusion.md", "fusion_x/a.py"]))
    assert _call("list_available_files", pattern="fusion*").raw_output == ["fusion.md", "fusion_x/a.py"]


def test_one_call_lists_at_most_the_first_hundred_paths(monkeypatch):
    many = [f"logs/run_{i:03d}.out" for i in range(150)]
    monkeypatch.setattr(files, "get_vector_store_manager", lambda: FakeManager(many))
    assert _call("list_available_files").raw_output == sorted(many)[: files.MAX_LISTED_FILES]
    assert files.MAX_LISTED_FILES == 100  # the team's LIMIT 100


def test_a_failed_manifest_read_returns_a_fixed_text_without_the_reason(monkeypatch, caplog):
    internals = RuntimeError('relation "public.ingest_manifest_llamaindex_embedding" does not exist')
    monkeypatch.setattr(files, "get_vector_store_manager", lambda: FakeManager(error=internals))
    out = _call("list_available_files")
    assert not out.is_error and out.raw_output == files.LIST_FAILED and out.raw_output.startswith("Error")
    assert "ingest_manifest" not in str(out.content)
    assert "ingest_manifest_llamaindex_embedding" in caplog.text  # the reason is in the API's log


def test_an_unreachable_database_is_not_described_to_the_model(monkeypatch):
    # The real accessor and manager, pointed at a closed port: psycopg2's message names the host and port.
    from app.database import vector_store

    monkeypatch.setenv("PG_CONNECTION_STRING", "postgresql://someone@127.0.0.1:1/metanavit")
    monkeypatch.setenv("PSYCOPG2_CONNECTION_STRING", "dbname=metanavit user=someone host=127.0.0.1 port=1")
    vector_store.get_vector_store_manager.cache_clear()
    try:
        out = _call("list_available_files")
    finally:
        vector_store.get_vector_store_manager.cache_clear()
    assert not out.is_error and out.raw_output == files.LIST_FAILED
    assert not any(word in str(out.content) for word in ("127.0.0.1", "port", "someone", "refused"))


# ---------------------------------------------------------------- get_file_content


def test_every_fixture_file_is_rebuilt_from_its_stored_chunks():
    scan = scan_directory(FIXTURE, SERVED_POLICY)
    assert len(scan.files) > 1
    for source in scan.files:  # a CSV whose rows carry the header line, Python files cut in two, single chunks
        assert files.file_text(stored_chunks(source.path, source.text)) == source.text, source.path


@pytest.mark.parametrize("path,text", [
    # non-ASCII text in fixed windows that overlap: byte offsets differ from character offsets
    ("notes/long.txt", "".join(f"línea {i}: café 日本語 ü\n" for i in range(120))),
    # a long Markdown section re-cut into overlapping windows
    ("docs/long.md", "# Title\n\n" + "".join(f"Paragraph {i} with ümlauts and 日本 text.\n" for i in range(120))),
    # a large YAML file split before every top-level key: each block is stored behind the run_id block
    ("configs/big.yaml", "run_id: 47\n" + "".join(f"key_{i}:\n  value: {'é' * 40} {i}\n" for i in range(60))),
    # CSV rows stored behind the header line
    ("results/x.csv", "náme,value\n" + "".join(f"ñame_{i},{i}.5\n" for i in range(5))),
], ids=["fixed-windows", "markdown-recut", "yaml-split", "csv-rows"])
def test_overlapping_and_header_prefixed_chunks_rebuild_the_exact_text(path, text):
    rows = stored_chunks(path, text)
    ordered = sorted(rows, key=lambda r: r["start_byte"])
    # the case is really there: windows that overlap, or chunks whose text is not the source slice
    assert any(b["start_byte"] < a["end_byte"] for a, b in zip(ordered, ordered[1:])) or not all(r["verbatim"] for r in rows)
    assert files.file_text(rows) == text


def test_whitespace_the_chunker_did_not_store_is_left_out_with_a_line_break_where_needed():
    # Blank JSON Lines records have no chunk: the lines around them are joined, each still ending its line.
    assert files.file_text(stored_chunks("logs/events.jsonl", '{"a": 1}\n\n\n{"b": 2}\n')) == '{"a": 1}\n{"b": 2}\n'
    # A fixed window of spaces only has no chunk either: a line break stands in where the text before goes on.
    text = "a" * 600 + " " * 1200 + "b" * 600
    rows = stored_chunks("notes/spaced.txt", text)
    covered, gaps = 0, 0
    for row in sorted(rows, key=lambda r: r["start_byte"]):
        gaps += row["start_byte"] > covered
        covered = max(covered, row["end_byte"])
    assert gaps == 1
    rebuilt = files.file_text(rows)
    assert rebuilt.count("\n") == 1 and rebuilt.replace("\n", "").strip(" ") != text.strip(" ")
    assert rebuilt.startswith("a" * 600 + " ") and rebuilt.endswith(" " + "b" * 600)


def test_blank_lines_between_top_level_python_definitions_do_not_come_back():
    # What doc/engineering-reference.md (Backend) says: python_ast_chunks stores no chunk for the blank lines between
    # two top-level definitions (none of the fixture's Python files has two), and the definition before them ends
    # its line, so nothing stands in for them.
    text = "import os\n\n\ndef a():\n    return 1\n\n\ndef b():\n    return 2\n"
    assert files.file_text(stored_chunks("src/m.py", text)) == "import os\n\n\ndef a():\n    return 1\ndef b():\n    return 2\n"


def test_get_file_content_returns_the_file_by_its_relative_path(monkeypatch):
    text = (FIXTURE / "results" / "ablation.csv").read_bytes().decode("utf-8")
    manager = FakeManager(texts={"results/ablation.csv": text})
    monkeypatch.setattr(files, "get_vector_store_manager", lambda: manager)
    out = _call("get_file_content", file_path=" results/ablation.csv ")
    assert not out.is_error and out.raw_output == text
    assert manager.asked == ["results/ablation.csv"]


def test_a_path_the_index_does_not_hold_is_not_found(monkeypatch):
    manager = FakeManager(texts={"configs/run_047.yaml": "run_id: 47\n"})
    monkeypatch.setattr(files, "get_vector_store_manager", lambda: manager)
    for path in ("configs/run_048.yaml", "../configs/run_047.yaml", "/data/configs/run_047.yaml", "", "   "):
        out = _call("get_file_content", file_path=path)
        assert not out.is_error and out.raw_output == files.NOT_FOUND, path
    assert files.NOT_FOUND.startswith("Not found") and "list_available_files" in files.NOT_FOUND
    assert manager.asked == ["configs/run_048.yaml", "../configs/run_047.yaml", "/data/configs/run_047.yaml"]


def test_get_file_content_called_without_a_path_says_not_found():
    # The default used to be pydantic's Field(description=...), a FieldInfo, as with list_available_files' pattern.
    assert inspect.signature(files.get_file_by_path).parameters["file_path"].default == ""
    out = _call("get_file_content")
    assert not out.is_error and out.raw_output == files.NOT_FOUND


def test_a_failed_read_returns_a_fixed_text_without_the_reason(monkeypatch, caplog):
    internals = RuntimeError('relation "public.data_llamaindex_embedding" does not exist')
    monkeypatch.setattr(files, "get_vector_store_manager", lambda: FakeManager(error=internals))
    out = _call("get_file_content", file_path="configs/run_047.yaml")
    assert not out.is_error and out.raw_output == files.READ_FAILED and out.raw_output.startswith("Error")
    assert "data_llamaindex_embedding" in caplog.text and "data_llamaindex_embedding" not in str(out.content)


def test_get_file_content_with_an_unreachable_database_is_not_described_to_the_model(monkeypatch):
    from app.database import vector_store

    monkeypatch.setenv("PG_CONNECTION_STRING", "postgresql://someone@127.0.0.1:1/metanavit")
    monkeypatch.setenv("PSYCOPG2_CONNECTION_STRING", "dbname=metanavit user=someone host=127.0.0.1 port=1")
    vector_store.get_vector_store_manager.cache_clear()
    try:
        out = _call("get_file_content", file_path="configs/run_047.yaml")
    finally:
        vector_store.get_vector_store_manager.cache_clear()
    assert not out.is_error and out.raw_output == files.READ_FAILED
    assert not any(word in str(out.content) for word in ("127.0.0.1", "port", "someone", "refused"))


def test_a_long_file_is_cut_with_a_note(monkeypatch):
    text = "".join(f"step {i}: loss 0.{i:04d}\n" for i in range(3000))
    assert len(text) > files.MAX_CONTENT_CHARS
    monkeypatch.setattr(files, "get_vector_store_manager", lambda: FakeManager(texts={"logs/train.log": text}))
    out = _call("get_file_content", file_path="logs/train.log").raw_output
    assert out == text[: files.MAX_CONTENT_CHARS] + f"\n[truncated: {len(text) - files.MAX_CONTENT_CHARS} more characters]"
