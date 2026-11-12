"""
Index File Access Tool

Provides capabilities for accessing file contents from the document index
rather than directly from the file system.
"""

import fnmatch
import logging
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from llama_index.core.tools import FunctionTool, BaseTool

from app.database.vector_store import get_vector_store_manager
from app.engine.tools import create_tool_callback

logger = logging.getLogger(__name__)

# The most paths one list_available_files call returns (the LIMIT 100 of the team's query).
MAX_LISTED_FILES = 100
# The longest text one get_file_content call returns; the rest is cut with a note. The tool used
# to return one chunk at most, and the index step accepts files up to 1 MiB (INGEST_MAX_FILE_BYTES).
MAX_CONTENT_CHARS = 20_000
# What the model gets when the index cannot be read; the reason (it can name the host, the port
# or the table) goes to the API's log only.
LIST_FAILED = "Error: the list of indexed files could not be read."
READ_FAILED = "Error: the file could not be read from the index."
# A path the index holds no chunk of, or none given: an answer, not a failure. The path itself is
# not repeated back.
NOT_FOUND = "Not found: the index holds no file at that path. list_available_files lists the indexed paths."

class ListFileSchema(BaseModel):
    pattern: str = Field(
        default="*",
        description="Filter pattern to match file paths (e.g., *.py for Python files)"
    )

class GetFileSchema(BaseModel):
    file_path: str = Field(
        description="Path to the file that should be retrieved from the index"
    )

def file_text(chunks: List[Dict[str, Any]]) -> str:
    """
    A file's text rebuilt from its stored chunks ({"text", "start_byte", "end_byte",
    "verbatim"} each, in any order), as VectorStoreManager.chunks_of_path returns them.

    The byte ranges are UTF-8 offsets into the file (app/eval/ingest.file_chunks). A chunk
    that is not verbatim was stored with context in front of its source slice (a CSV row
    behind the header line, a block of a split YAML file behind the header block;
    app/chunking/structure.py), so its slice is the last end_byte - start_byte bytes of
    its text. Overlapping windows (the fixed window, a re-cut long section) are not
    repeated. Whitespace-only stretches the chunker left out are not stored: where one
    was, a line break is added if the text before it does not end a line.
    """
    out = bytearray()
    covered = 0
    for chunk in sorted(chunks, key=lambda c: (c["start_byte"], c["end_byte"])):
        data = chunk["text"].encode("utf-8")
        start, end = chunk["start_byte"], chunk["end_byte"]
        if not chunk["verbatim"] and 0 <= end - start <= len(data):
            data = data[len(data) - (end - start):]
        if out and start > covered and out[-1:] not in (b"\n", b"\r"):
            out += b"\n"
        out += data[max(0, covered - start):]
        covered = max(covered, end)
    return out.decode("utf-8", errors="replace")


def get_file_by_path(file_path: str = "") -> str:
    """
    Get file content from the index based on file path.

    The file is looked up by the path the ingest manifest and list_available_files give
    (relative to DATA_DIR, e.g. configs/run_047.yaml) with a parameterized query on the
    chunk rows (VectorStoreManager.chunks_of_path), and its text is rebuilt from all of
    its chunks (file_text). This used to pass a plain dict as metadata filters to a query
    engine, which raised on every call (the error came back as text), compared the
    absolute file_path the rows carry, and returned one chunk after an LLM synthesis
    call. The vector store's own MetadataFilters are not used: the installed
    PGVectorStore formats a filter value into the SQL text, and this value is whatever
    path the model sends.

    Args:
        file_path: Path of the file to retrieve. A plain "" default, not a pydantic
            Field: the agent passes only the arguments the model gave.

    Returns:
        Content of the file as a string (cut after MAX_CONTENT_CHARS characters, with a
        note), NOT_FOUND when no chunk carries that path or none is given, READ_FAILED
        when the index cannot be read.
    """
    path = file_path.strip() if isinstance(file_path, str) else ""
    if not path:
        return NOT_FOUND
    try:
        chunks = get_vector_store_manager().chunks_of_path(path)
    except Exception as exc:  # noqa: BLE001 - READ_FAILED says what failed; raised, the workflow would give the model only the class and a request id
        logger.error(f"get_file_content: the file's chunks could not be read: {type(exc).__name__}: {exc}")
        return READ_FAILED
    if not chunks:
        return NOT_FOUND
    text = file_text(chunks)
    if len(text) > MAX_CONTENT_CHARS:
        return text[:MAX_CONTENT_CHARS] + f"\n[truncated: {len(text) - MAX_CONTENT_CHARS} more characters]"
    return text

def list_available_files(pattern: str = "*") -> Union[List[str], str]:
    """
    List available files in the index with optional pattern matching.

    The files are the ones the served index holds: the paths its ingest manifest lists
    (one row per indexed file, relative to DATA_DIR, written by the index step once the
    file's chunks are stored; app/engine/ingest.py). This used to read the indexed_files
    table, which only the team's crawler (app/engine/loaders/file_system.py) writes and
    no indexing path runs, so every listing was empty.

    Args:
        pattern: Filter pattern for file paths. The model sees its description in
            ListFileSchema. The default is a plain "*": the agent calls the function with
            only the arguments the model gave, so a pydantic Field(default="*") here was
            the default itself (a FieldInfo, not a string) and a call without a pattern
            raised. A glob matched against the whole relative path (fnmatch, case-sensitive;
            `*` also matches "/", as the old LIKE '%' did): configs/*.yaml, *.py. Empty or
            "*" lists every file.

    Returns:
        At most MAX_LISTED_FILES matching paths, in code point order (whatever the database's
        collation), or LIST_FAILED when the manifest cannot be read.
    """
    logger.info("Listing available files in the index")
    pattern = (pattern.strip() if isinstance(pattern, str) else "") or "*"
    try:
        # The API's cached manager (query_index reads through the same one), resolved per call:
        # building the tools needs no database.
        indexed = get_vector_store_manager().ingest_manifest()
    except Exception as exc:  # noqa: BLE001 - LIST_FAILED says what failed; raised, the workflow would give the model only the class and a request id
        logger.error(f"list_available_files: the ingest manifest could not be read: {type(exc).__name__}: {exc}")
        return LIST_FAILED
    file_paths = sorted(path for path in indexed if fnmatch.fnmatchcase(path, pattern))
    return file_paths[:MAX_LISTED_FILES]

def get_tools() -> List[BaseTool]:
    """
    Create and return index-based file access tools.
    
    Returns:
        List of file access function tools
    """
    file_tool_name = "get_file_content"
    file_tool_description = "Get file content from the index based on file path"
    file_get_tool = FunctionTool.from_defaults(
        fn=get_file_by_path,
        name=file_tool_name,
        description=file_tool_description,
        fn_schema=GetFileSchema,
        callback=create_tool_callback(file_tool_name, file_tool_description)
    )
    
    file_list_tool_name ="list_available_files"
    file_list_tool_description = "List available files in the index with optional pattern matching"
    file_list_tool = FunctionTool.from_defaults(
        fn=list_available_files,
        name=file_list_tool_name,
        description=file_list_tool_description,
        fn_schema=ListFileSchema,
        callback=create_tool_callback(file_list_tool_name, file_list_tool_description)
    )
    
    return [file_get_tool, file_list_tool]