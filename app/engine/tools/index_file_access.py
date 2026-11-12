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

from app.engine.index import get_index
from app.database.vector_store import get_vector_store_manager
from app.engine.tools import create_tool_callback

logger = logging.getLogger(__name__)

# The most paths one list_available_files call returns (the LIMIT 100 of the team's query).
MAX_LISTED_FILES = 100
# What the model gets when the manifest cannot be read; the reason (it can name the host,
# the port or the table) goes to the API's log only.
LIST_FAILED = "Error: the list of indexed files could not be read."

class ListFileSchema(BaseModel):
    pattern: str = Field(
        default="*",
        description="Filter pattern to match file paths (e.g., *.py for Python files)"
    )

class GetFileSchema(BaseModel):
    file_path: str = Field(
        description="Path to the file that should be retrieved from the index"
    )

def get_file_by_path(
    file_path: str = Field(
        description="Path to the file that should be retrieved from the index"
    )
) -> str:
    """
    Get file content from the index based on file path.
    
    This tool searches the document index for a document with the specified file path
    in its metadata and returns the content.
    
    Args:
        file_path: Path of the file to retrieve
        
    Returns:
        Content of the file as a string
    """
    # Get the index
    index = get_index()
    if not index:
        return "Error: Document index not available"
    
    # Query for documents with matching file_path in metadata
    query_engine = index.as_query_engine(
        filters={"file_path": {"$eq": file_path}},
        similarity_top_k=1
    )
    
    # Get the document
    try:
        response = query_engine.query("")  # Empty query to just get the document
        
        if not response.source_nodes:
            return f"Error: No document found with path {file_path}"
        
        # Return the content of the first matching document
        return response.source_nodes[0].get_content()
        
    except Exception as e:
        logger.error(f"Error retrieving file from index: {str(e)}")
        return f"Error: {str(e)}"

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
    except Exception as exc:  # noqa: BLE001 - the workflow would hand str(exc) to the model
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