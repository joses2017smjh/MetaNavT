"""M49: the batched MaxSim pass (app.multimodal.late_interaction.maxsim_scores) is maxsim's arithmetic.

numpy only in CI. The torch float64 test is skip-guarded (login and CI have no torch); it runs inside job J1, whose
output goes into the milestone notes.
"""

import numpy as np
import pytest

from app.eval import late_interaction as li
from app.multimodal.late_interaction import TokenStore, _blocks, maxsim, maxsim_scores
from app.retrieval.hybrid import DENSE_SCORE_DECIMALS


def _ragged(seed: int = 0, n_docs: int = 37, n_queries: int = 9, dim: int = 24):
    rng = np.random.RandomState(seed)
    docs = [rng.standard_normal((int(n), dim)).astype(np.float32) for n in rng.randint(1, 40, size=n_docs)]
    docs[5] = np.zeros((0, dim), dtype=np.float32)  # a document with no scored token
    queries = [rng.standard_normal((int(n), dim)).astype(np.float32) for n in rng.randint(1, 33, size=n_queries)]
    return queries, docs, dim


def test_the_batched_pass_equals_maxsim_document_by_document():
    queries, docs, dim = _ragged()
    store = TokenStore.from_matrices(docs, dim)
    scores = maxsim_scores(queries, store)
    assert scores.shape == (len(queries), len(docs)) and scores.dtype == np.float64
    for i, q in enumerate(queries):
        for j, d in enumerate(docs):
            assert abs(scores[i, j] - maxsim(q, d)) < 1e-5, (i, j)


def test_two_block_budgets_give_identical_rounded_scores():
    queries, docs, dim = _ragged(seed=1, n_docs=80)
    store = TokenStore.from_matrices(docs, dim)
    big = maxsim_scores(queries, store, block_bytes=1 << 30)
    small = maxsim_scores(queries, store, block_bytes=1 << 14, query_rows=40)  # many small blocks of both kinds
    assert len(_blocks(store.lengths(), (1 << 14) // (8 * 40))) > 10
    assert np.array_equal(np.round(big, DENSE_SCORE_DECIMALS), np.round(small, DENSE_SCORE_DECIMALS))


def test_a_document_or_a_query_with_no_tokens_scores_zero():
    queries, docs, dim = _ragged(seed=2)
    queries.append(np.zeros((0, dim), dtype=np.float32))
    store = TokenStore.from_matrices(docs, dim)
    scores = maxsim_scores(queries, store)
    assert (scores[:, 5] == 0.0).all() and (scores[-1] == 0.0).all()
    assert maxsim_scores([], store).shape == (0, len(docs))


def test_the_store_is_one_flat_float32_array_with_offsets():
    _, docs, dim = _ragged(seed=3)
    store = TokenStore.from_matrices(docs, dim)
    assert store.vectors.dtype == np.float32 and store.vectors.shape == (sum(len(d) for d in docs), dim)
    assert list(store.lengths()) == [len(d) for d in docs] and len(store) == len(docs) and store.dim == dim
    assert all(np.array_equal(store.doc(i), d) for i, d in enumerate(docs))
    assert store.nbytes == store.vectors.nbytes


def test_blocks_cover_every_item_once_within_the_cap():
    lengths = np.array([3, 9, 1, 0, 7, 7, 2, 30, 4])
    blocks = _blocks(lengths, 20)
    assert [i for s, e, _ in blocks for i in range(s, e)] == list(range(len(lengths)))
    for s, e, w in blocks:
        assert w == lengths[s:e].max() and ((e - s) * w <= 20 or e - s == 1)


def test_a_stores_cache_key_names_the_document_length_and_the_side():
    keys = {li.store_key("c" * 64, "colbertv2", L, side) for L in (180, 512) for side in ("document", "query")}
    assert len(keys) == 4
    assert li.store_key("c" * 64, "colbertv2", 180) != li.store_key("d" * 64, "colbertv2", 180)


def test_the_torch_float64_path_agrees_with_numpy():
    torch = pytest.importorskip("torch")  # login-app and CI have no torch: this runs inside job J1
    queries, docs, dim = _ragged(seed=4, n_docs=60)
    store = TokenStore.from_matrices(docs, dim)
    host = maxsim_scores(queries, store)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert np.max(np.abs(maxsim_scores(queries, store, device="cpu") - host)) < 1e-12
    assert np.max(np.abs(maxsim_scores(queries, store, device=device, block_bytes=1 << 15, query_rows=48) - host)) < 1e-12
