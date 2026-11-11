"""Score values of the in-memory Okapi BM25 (app/retrieval/bm25.py), not only its rank order.

Every expected value is written out from the formula the module implements,

    idf(t)      = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))
    score(q, d) = sum over distinct query terms t of
                  idf(t) * f * (k1 + 1) / (f + k1 * (1 - b + b * |d| / avgdl)) * (count of t in q)

on corpora small enough to do by hand. CORPUS has N = 5 documents of lengths 2, 3, 2, 1, 2, so
avgdl = 2 and, at the defaults k1 = 1.5, b = 0.75, the length term k1 * (1 - b + b * |d| / 2) is
0.9375, 1.5 and 2.0625 for lengths 1, 2 and 3. A term seen once in a document of average length
contributes exactly its idf for any k1 and b.
"""

import math

import pytest

from app.retrieval.bm25 import BM25Index

REL = 1e-12
IDS = ["d1", "d2", "d3", "d4", "d5"]
CORPUS = [
    "common rare",  # |d| = 2
    "common shared shared",  # |d| = 3
    "common shared",  # |d| = 2
    "common",  # |d| = 1
    "common filler",  # |d| = 2
]
IDF_RARE = math.log(4)  # in 1 of 5 documents: ln(1 + 4.5 / 1.5)
IDF_SHARED = math.log(12 / 5)  # in 2 of 5: ln(1 + 3.5 / 2.5)
IDF_COMMON = math.log(12 / 11)  # in all 5: ln(1 + 0.5 / 5.5), small but positive
IDF_UNSEEN = math.log(12)  # in none: ln(1 + 5.5 / 0.5)


def _index(**params) -> BM25Index:
    return BM25Index(**params).fit(IDS, CORPUS)


def _approx(pairs):
    return [(doc_id, pytest.approx(score, rel=REL)) for doc_id, score in pairs]


def test_corpus_statistics():
    index = _index()
    assert (index.n, index.doc_len, index.avgdl) == (5, [2, 3, 2, 1, 2], 2.0)
    assert dict(index.df) == {"common": 5, "rare": 1, "shared": 2, "filler": 1}


def test_idf_values():
    index = _index()
    assert index.idf("rare") == pytest.approx(IDF_RARE, rel=REL)
    assert index.idf("shared") == pytest.approx(IDF_SHARED, rel=REL)
    assert index.idf("common") == pytest.approx(IDF_COMMON, rel=REL)
    assert index.idf("zeta") == pytest.approx(IDF_UNSEEN, rel=REL)


def test_scores_at_the_defaults():
    index = _index()
    assert (index.k1, index.b) == (1.5, 0.75)
    # A single occurrence in a document of average length: (1 * 2.5) / (1 + 1.5) = 1.
    assert index.search("rare") == _approx([("d1", IDF_RARE)])
    # d2: two occurrences, |d| = 3: (2 * 2.5) / (2 + 2.0625) = 16 / 13.
    assert index.search("shared") == _approx([("d2", IDF_SHARED * 16 / 13), ("d3", IDF_SHARED)])
    # Query terms add up: d1 scores rare and common, d2 shared (twice in it) and common (|d| = 3: 40 / 49).
    assert index.search("rare shared common", k=2) == _approx(
        [("d1", IDF_RARE + IDF_COMMON), ("d2", IDF_SHARED * 16 / 13 + IDF_COMMON * 40 / 49)])


def test_scores_in_every_document_and_exact_ties_keep_corpus_order():
    index = _index()
    # |d| = 1: 2.5 / 1.9375 = 40 / 31; |d| = 2: 1; |d| = 3: 2.5 / 3.0625 = 40 / 49.
    expected = [IDF_COMMON * 40 / 31, IDF_COMMON * 40 / 49, IDF_COMMON, IDF_COMMON, IDF_COMMON]
    tokens = ["common"]
    assert [index.score_doc(tokens, i) for i in (3, 1, 0, 2, 4)] == pytest.approx(expected, rel=REL)
    hits = index.search("common")
    assert [doc_id for doc_id, _ in hits] == ["d4", "d1", "d3", "d5", "d2"]
    assert hits[1][1] == hits[2][1] == hits[3][1]  # d1, d3 and d5 tie exactly and stay in corpus order


@pytest.mark.parametrize(
    "k1,b,d2_tf",
    [
        pytest.param(1.5, 0.0, 10 / 7, id="b=0, no length normalisation: (2 * 2.5) / (2 + 1.5)"),
        pytest.param(1.5, 1.0, 20 / 17, id="b=1, full length normalisation: (2 * 2.5) / (2 + 1.5 * 3 / 2)"),
        pytest.param(0.9, 0.4, 95 / 77, id="SciFact settings: (2 * 1.9) / (2 + 0.9 * (0.6 + 0.4 * 3 / 2))"),
        pytest.param(3.0, 0.75, 64 / 49, id="k1=3: (2 * 4) / (2 + 3 * (0.25 + 0.75 * 3 / 2))"),
    ],
)
def test_k1_and_b(k1, b, d2_tf):
    """d2 holds `shared` twice at |d| = 3; d3 once at the average length, so it scores the idf whatever k1 and b are."""
    index = _index(k1=k1, b=b)
    assert index.search("shared") == _approx([("d2", IDF_SHARED * d2_tf), ("d3", IDF_SHARED)])


def test_k1_zero_is_binary_term_presence():
    index = _index(k1=0.0, b=0.75)
    assert index.search("shared") == _approx([("d2", IDF_SHARED), ("d3", IDF_SHARED)])


def test_a_repeated_query_term_counts_linearly():
    index = _index()
    assert index.search("rare rare") == _approx([("d1", 2 * IDF_RARE)])
    assert index.search("rare rare rare RARE") == _approx([("d1", 4 * IDF_RARE)])  # the default analyzer lowercases


def test_empty_documents_count_in_n_and_avgdl_and_never_match():
    index = BM25Index().fit(["e1", "e2", "e3", "e4"], ["", "alpha", "", "alpha beta beta"])
    assert (index.n, index.doc_len, index.avgdl) == (4, [0, 1, 0, 3], 1.0)
    idf_alpha = math.log(2)  # in 2 of 4: ln(1 + 2.5 / 2.5)
    idf_beta = math.log(10 / 3)  # in 1 of 4: ln(1 + 3.5 / 1.5)
    # Length term at avgdl = 1: 1.5 for |d| = 1, 3.75 for |d| = 3.
    assert index.search("alpha") == _approx([("e2", idf_alpha), ("e4", idf_alpha * 10 / 19)])
    assert index.search("beta") == _approx([("e4", idf_beta * 20 / 23)])
    assert index.score_doc(["alpha"], 0) == 0.0 and index.score_doc(["alpha"], 2) == 0.0


def test_empty_index_empty_query_and_unseen_terms():
    empty = BM25Index().fit([], [])
    assert (empty.n, empty.avgdl) == (0, 0.0)
    assert empty.search("anything") == []
    index = _index()
    assert index.search("") == []
    assert index.search("!!! ???") == []  # no token survives the analyzer
    assert index.search("zeta") == []  # a positive idf, but no document holds the term
    assert len(index.search("common", k=2)) == 2
