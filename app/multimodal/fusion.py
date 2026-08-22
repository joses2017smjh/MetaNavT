"""Cross-modal fusion: text + page-image + image retrievers, fused with RRF."""

from __future__ import annotations

from typing import Sequence

from app.retrieval.fuse import reciprocal_rank_fusion


def fuse_modalities(
    text_ids: Sequence[str],
    page_ids: Sequence[str] | None = None,
    image_ids: Sequence[str] | None = None,
    k: int = 60,
) -> list[tuple[str, float]]:
    lists = [list(text_ids)]
    if page_ids:
        lists.append(list(page_ids))
    if image_ids:
        lists.append(list(image_ids))
    return reciprocal_rank_fusion(lists, k=k)
