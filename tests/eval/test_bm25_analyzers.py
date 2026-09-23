"""The default tokenizer is what fixture v1 is scored with; the BEIR analyzer is opt-in."""

import pytest

from app.retrieval.bm25 import BM25Index, LUCENE_STOPWORDS, get_tokenizer, tokenize

snowballstemmer = pytest.importorskip("snowballstemmer")
from app.retrieval.bm25 import tokenize_beir  # noqa: E402


def test_default_tokenizer_keeps_paths_and_does_not_stem():
    assert tokenize("Open configs/run_047.yaml: learning_rate") == ["open", "configs/run_047.yaml", "learning_rate"]
    assert tokenize("The cells are running") == ["the", "cells", "are", "running"]


def test_beir_analyzer_lowercases_stems_and_drops_stopwords():
    assert tokenize_beir("The cells are running in scientific studies") == ["cell", "run", "scientif", "studi"]
    assert "the" in LUCENE_STOPWORDS and "cells" not in LUCENE_STOPWORDS
    assert tokenize_beir("configs/run_047.yaml") == ["config", "run", "047", "yaml"]


def test_index_uses_the_requested_analyzer():
    docs = ["cells divide rapidly", "the study of stars"]
    default = BM25Index().fit(["a", "b"], docs)
    beir = BM25Index(k1=0.9, b=0.4, analyzer="beir").fit(["a", "b"], docs)
    assert default.search("cell") == []              # no stemming: 'cell' != 'cells'
    assert beir.search("cell")[0][0] == "a"          # stemmed match
    assert beir.k1 == 0.9 and beir.b == 0.4 and beir.analyzer == "beir"


def test_unknown_analyzer_is_an_error():
    with pytest.raises(ValueError):
        get_tokenizer("lucene")
