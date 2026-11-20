"""Equal file depth for the real-model hybrid and its legs (make bench-file-depth).

Recall@k scores the first k entries of a FILE list (app/eval/metrics.py). Every list the harness scores is cut at
retrieve_k = 50 CHUNKS before its files are taken (InMemoryHybridIndex.retrieve for the fused list, the BM25 and
dense legs likewise), and fixture v1 has 71 chunks in 61 files, so a list whose chunks share files holds fewer than
50 files and lists compared at Recall@50 can hold different numbers of files.
bench/results/hybrid_drops_bge_base.json shows what that costs hybrid@bge-base: 7 gold files that dense ranks 48th to
50th and BM25 never returns land at fused chunk ranks 51 to 53, so hybrid loses Recall@50 against dense alone.

FILE_DEPTH_RULE, declared before any run (M37), gives every compared list the same file depth. The rows recorded
under it are added beside the committed ones; nothing here changes the default path: InMemoryHybridIndex, the served
SQL and every committed row keep the 50-chunk cut.
"""

from __future__ import annotations

from typing import Callable, Sequence

FILE_DEPTH_RULE = ("file-depth cut: every list (the BM25 leg, the dense leg and the fused list) keeps chunks in rank "
                   "order until it holds retrieve_k distinct files (or its candidates run out); metrics are unchanged.")
CUT_AS_CODED = ("a list keeps its chunks in rank order and stops right after the chunk that brings it to retrieve_k "
                "distinct files, so a later chunk of a file it already holds is not kept; a leg's candidates are every "
                "chunk it returns when asked for all chunks of the index (BM25 returns only chunks that share a term "
                "with the question); the fused list is the two cut legs through rrf_score_map and the stable sort by "
                "score InMemoryHybridIndex.retrieve uses, cut again; files are taken in the order of their first chunk")


def file_depth_cut(chunk_ids: Sequence[str], path_of: Callable[[str], str], retrieve_k: int) -> list[str]:
    """The chunks of a ranked chunk list, in rank order, up to and including the one that brings the list to
    retrieve_k distinct files (FILE_DEPTH_RULE); every chunk when the list holds fewer files than that."""
    kept: list[str] = []
    files: set[str] = set()
    for cid in chunk_ids:
        if len(files) >= retrieve_k:
            break
        kept.append(cid)
        files.add(path_of(cid))
    return kept
