# tests/test_file.py

import sys
from unittest.mock import MagicMock

import app.database.vector_store  # noqa: F401 - imported at collection, before conftest swaps in the module mocks
import app.engine.loaders  # noqa: F401 - imported at collection, before conftest swaps in the module mocks; patched below
from app.engine import ingest
from app.engine.generate import get_doc_store, run_pipeline, generate_datasource


def test_get_doc_store_no_storage_dir(mocker):
    # Arrange
    mocker.patch.dict('os.environ', {}, clear=True)  # Clear STORAGE_DIR
    mocker.patch('os.path.exists', return_value=False)  # Mock path.exists to return False
    mock_simple_store = mocker.patch('app.engine.generate.SimpleDocumentStore')
    
    # Act
    _ = get_doc_store()
    
    # Assert
    mock_simple_store.assert_called_once()


def test_get_doc_store_with_storage_dir(mocker):
    # Arrange
    mocker.patch('app.engine.generate.os.path.exists', return_value=True)
    mock_simple_store = mocker.patch('app.engine.generate.SimpleDocumentStore')
    
    # Act
    _ = get_doc_store()
    
    # Assert
    mock_simple_store.from_persist_dir.assert_called_once()


def test_run_pipeline_success(mocker):
    # Arrange
    # Mock the imports and Settings
    # Patch the whole Settings singleton: patching Settings.embed_model directly
    # reads the real property first, which resolves the default OpenAI embedder.
    settings = mocker.patch('app.engine.generate.Settings')
    settings.embed_model = MagicMock()
    settings.chunk_size = 1000
    settings.chunk_overlap = 200
    
    # Mock the pipeline
    mock_pipeline = mocker.patch('app.engine.generate.IngestionPipeline')
    mock_pipeline_instance = mock_pipeline.return_value
    mock_pipeline_instance.run.return_value = ["processed_doc1", "processed_doc2"]
    
    docstore = MagicMock()
    vector_store = MagicMock()
    documents = ["doc1", "doc2"]
    
    # Act
    _ = run_pipeline(docstore, vector_store, documents)
    
    # Assert
    mock_pipeline.assert_called_once()
    mock_pipeline_instance.run.assert_called_once_with(
        show_progress=True,
        documents=documents
    )


def test_generate_datasource_delegates_to_index_directory(mocker, monkeypatch, tmp_path):
    """The CLI is the index step: DATA_DIR through app/engine/ingest.index_directory (the benchmark's chunker,
    the manifest, the lock), with the configured embedder. The crawler and the SentenceSplitter pipeline,
    which wrote a second copy of every chunk on a second run, are not called."""
    settings = sys.modules["llama_index.core.settings"].Settings  # this suite's stand-in (tests/conftest.py)
    settings.embed_model.model_name = "hash-8"
    init_settings = mocker.patch("app.settings.init_settings")
    vsm = MagicMock()
    mocker.patch("app.database.vector_store.get_vector_store_manager", return_value=vsm)
    index_directory = mocker.patch.object(ingest, "index_directory", return_value={"files_indexed": 2, "chunks_embedded": 3})
    for name in ("IngestionPipeline", "SentenceSplitter", "get_doc_store", "run_pipeline", "persist_storage"):
        mocker.patch(f"app.engine.generate.{name}", side_effect=AssertionError(f"{name} is not the served path"))
    crawler = mocker.patch("app.engine.loaders.get_documents", side_effect=AssertionError("the crawler is not the served path"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("INGEST_MAX_REJECTED", "0")

    report = generate_datasource()

    assert report == {"files_indexed": 2, "chunks_embedded": 3}
    init_settings.assert_called_once()
    vsm.check_migrations.assert_called_once()  # the deploy step comes first
    index_directory.assert_called_once()
    (manager, data_dir), kwargs = index_directory.call_args
    assert (manager, data_dir, kwargs["embedder"], kwargs["max_rejected"]) == (vsm, str(tmp_path), "hash-8", 0)
    assert kwargs["embed_fn"] is settings.embed_model.get_text_embedding_batch
    crawler.assert_not_called()


def test_python_m_app_engine_generate_runs_the_index_step(mocker):
    """`python -m app.engine.generate` (scripts/run.sh generate) has the index step's output and exit status."""
    import runpy

    import pytest

    step = mocker.patch("app.engine.ingest.main", return_value=0)
    with pytest.raises(SystemExit) as exited:
        runpy.run_module("app.engine.generate", run_name="__main__")
    assert exited.value.code == 0
    step.assert_called_once_with()
