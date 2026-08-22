"""Image embeddings for renders, plots, screenshots.

Tries CLIP / SigLIP via sentence-transformers; falls back to a perceptual
hash of pixel blocks so 'find the trellis-wire render' still works in CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.retrieval.embedders import HashEmbedder, cosine_scores, l2_normalize

_clip_model = None


def pixel_embed(image: np.ndarray, dim: int = 64) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    h, w = arr.shape[:2]
    # 8x8 block averages → hashing trick
    bh, bw = max(1, h // 8), max(1, w // 8)
    blocks = []
    for i in range(8):
        for j in range(8):
            tile = arr[i * bh : (i + 1) * bh, j * bw : (j + 1) * bw]
            blocks.append(float(tile.mean()) if tile.size else 0.0)
    text = " ".join(f"{v:.3f}" for v in blocks)
    return HashEmbedder(dim=dim).encode([text])[0]


def embed_image(path: str | Path, image: np.ndarray | None = None) -> np.ndarray:
    global _clip_model
    if image is None:
        try:
            from PIL import Image

            image = np.asarray(Image.open(path).convert("RGB"))
        except Exception:
            return HashEmbedder(dim=64).encode([str(path)])[0]
    try:
        from sentence_transformers import SentenceTransformer
        from PIL import Image as PILImage

        if _clip_model is None:
            # SigLIP if present, else CLIP.
            for name in (
                "clip-ViT-B-32",
                "sentence-transformers/clip-ViT-B-32",
            ):
                try:
                    _clip_model = SentenceTransformer(name)
                    break
                except Exception:
                    continue
        if _clip_model is not None:
            vec = _clip_model.encode(
                PILImage.fromarray(np.asarray(image).astype("uint8")),
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return np.asarray(vec, dtype=np.float32).ravel()
    except Exception:
        pass
    return pixel_embed(image)


def embed_text_for_images(query: str, dim: int = 64) -> np.ndarray:
    """CLIP text tower when the image model loaded; hash embedder otherwise."""
    global _clip_model
    try:
        if _clip_model is not None:
            vec = _clip_model.encode(
                query, convert_to_numpy=True, normalize_embeddings=True
            )
            return np.asarray(vec, dtype=np.float32).ravel()
    except Exception:
        pass
    return HashEmbedder(dim=dim).encode([query])[0]


def search_images(
    query: str,
    items: list[tuple[str, np.ndarray]],
    k: int = 5,
) -> list[tuple[str, float]]:
    """Text query against image vectors. CLIP space if available; hash otherwise."""
    dim = int(items[0][1].shape[0]) if items else 64
    q = embed_text_for_images(query, dim=dim)
    if q.shape[0] != dim:
        q = HashEmbedder(dim=dim).encode([query])[0]
    mat = np.stack([v for _, v in items], axis=0) if items else np.zeros((0, 64))
    if mat.size == 0:
        return []
    scores = cosine_scores(q, mat)
    order = np.argsort(-scores)[:k]
    return [(items[i][0], float(scores[i])) for i in order]
