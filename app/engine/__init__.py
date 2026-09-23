"""
Engine Module

Provides document processing and index generation capabilities.

`generate_datasource` is resolved lazily: importing app.engine (or any
submodule such as app.engine.retriever) must not import the ingestion pipeline,
which loads .env and the LlamaIndex ingestion stack as a side effect.
"""

__all__ = ["generate_datasource"]


def __getattr__(name: str):
    if name == "generate_datasource":
        from app.engine.generate import generate_datasource

        return generate_datasource
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
