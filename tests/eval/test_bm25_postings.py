"""BM25 search over postings lists with a top-k heap returns exactly what scoring every document did.

The reference is the search this module replaced: score every document with score_doc, sort the whole list by
score (a stable sort, so ties keep corpus order), keep the first k and drop zero scores. Equal means the same ids
in the same order with the same float scores, compared with ==.
"""

import random

import pytest

from app.retrieval.bm25 import BM25Index

VOCAB = ["cell", "cells", "star", "the", "of", "fibre", "sleep", "vaccine", "configs/run_047.yaml", "learning_rate", "dinov2", "heart"]


def brute_force(index: BM25Index, query: str, k: int) -> list[tuple[str, float]]:
    tokens = index._tokenize(query)
    if not tokens or index.n == 0:
        return []
    scored = [(index.doc_ids[i], index.score_doc(tokens, i)) for i in range(index.n)]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [(doc_id, score) for doc_id, score in scored[:k] if score > 0]


def random_corpus(rng: random.Random, n: int) -> list[str]:
    docs: list[str] = []
    for _ in range(n):
        if docs and rng.random() < 0.3:
            docs.append(rng.choice(docs))  # an identical document: every score ties with the earlier copy
        else:
            docs.append(" ".join(rng.choice(VOCAB) for _ in range(rng.randint(0, 12))))  # empty documents too
    return docs


def random_query(rng: random.Random) -> str:
    words = [rng.choice(VOCAB + ["absent", "nowhere"]) for _ in range(rng.randint(1, 5))]
    if rng.random() < 0.5:
        words.append(words[0])  # a repeated query term (qtf 2)
    return " ".join(words)


@pytest.mark.parametrize("analyzer", ["default", "beir"])
@pytest.mark.parametrize("k1,b", [(1.5, 0.75), (0.9, 0.4), (1.2, 0.0), (0.0, 1.0)])
def test_postings_search_equals_scoring_every_document(analyzer, k1, b):
    if analyzer == "beir":
        pytest.importorskip("snowballstemmer")
    rng = random.Random(f"{analyzer}-{k1}-{b}")
    for _ in range(12):
        docs = random_corpus(rng, rng.randint(1, 60))
        index = BM25Index(k1=k1, b=b, analyzer=analyzer).fit([f"d{i:02d}" for i in range(len(docs))], docs)
        for _ in range(15):
            query = random_query(rng)
            for k in (1, 3, 10, len(docs), len(docs) + 5):
                assert index.search(query, k=k) == brute_force(index, query, k), (docs, query, k)


def test_ties_across_the_cutoff_keep_corpus_order():
    docs = ["star cell", "heart", "cell star", "star cell", "star cell", "fibre", "star cell"]
    index = BM25Index().fit([f"d{i}" for i in range(len(docs))], docs)
    hits = index.search("cell star cell", k=3)  # five documents tie on every score; the cutoff falls inside the group
    assert [d for d, _ in hits] == ["d0", "d2", "d3"] and len({s for _, s in hits}) == 1
    assert index.search("cell star cell", k=3) == brute_force(index, "cell star cell", 3)
    assert index.search("absent", k=3) == [] and index.search("", k=3) == [] and index.search("cell", k=0) == []


def test_search_touches_only_documents_in_the_query_terms_postings(monkeypatch):
    docs = ["cells divide rapidly", "the study of stars", "sleep and memory", "cells and stars"]
    index = BM25Index().fit(["a", "b", "c", "d"], docs)
    expected = brute_force(index, "cells stars", 10)

    def scored_every_document(self, query_tokens, doc_index):
        raise AssertionError("search scored a document one at a time")

    monkeypatch.setattr(BM25Index, "score_doc", scored_every_document)
    assert index.search("cells stars", k=10) == expected
    assert [d for d, _ in expected] == ["d", "a", "b"]


def test_refit_replaces_the_postings_and_new_parameters_apply():
    index = BM25Index().fit(["a", "b"], ["cell cell star", "star"])
    index.fit(["x", "y", "z"], ["heart", "heart sleep", "sleep"])
    assert index.search("cell star", k=5) == []
    assert [d for d, _ in index.search("sleep", k=5)] == ["z", "y"]
    index.k1, index.b = 0.9, 0.4  # parameters changed after fit: the length norms follow, as score_doc's do
    assert index.search("heart sleep", k=5) == brute_force(index, "heart sleep", 5)
