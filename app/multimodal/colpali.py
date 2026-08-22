"""ColPali-style page-image indexing: patches + MaxSim, no PDF-to-text required.

Pages are rendered (or synthesized) as images, split into a patch grid, each
patch embedded, scored with late-interaction MaxSim. Tables and figures stay
as pixels — that is the point.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.multimodal.late_interaction import maxsim
from app.retrieval.embedders import HashEmbedder, l2_normalize


@dataclass
class PageIndex:
    page_id: str
    path: str
    page_no: int
    patches: np.ndarray  # (n_patches, dim)


def image_to_patches(image: np.ndarray, patch: int = 32) -> np.ndarray:
    """image: (H, W) or (H, W, C) float/uint8 → (n_patches, patch, patch[, C])."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    h, w, c = arr.shape
    ph = max(1, h // patch)
    pw = max(1, w // patch)
    patches = []
    for i in range(ph):
        for j in range(pw):
            tile = arr[i * patch : (i + 1) * patch, j * patch : (j + 1) * patch]
            if tile.shape[0] < patch or tile.shape[1] < patch:
                pad = np.zeros((patch, patch, c), dtype=arr.dtype)
                pad[: tile.shape[0], : tile.shape[1]] = tile
                tile = pad
            patches.append(tile)
    if not patches:
        return np.zeros((1, patch, patch, arr.shape[2]), dtype=arr.dtype)
    return np.stack(patches, axis=0)


def embed_patches(patches: np.ndarray, dim: int = 64, caption: str = "") -> np.ndarray:
    """Deterministic patch embedder (hashing trick). Swap for ColPali/ColQwen."""
    flat = patches.reshape(patches.shape[0], -1)
    embedder = HashEmbedder(dim=dim)
    texts = [
        (caption + " " + " ".join(str(int(v)) for v in row[:: max(1, len(row) // 32)])).strip()
        for row in flat
    ]
    return embedder.encode(texts)


def index_page(
    page_id: str,
    path: str,
    page_no: int,
    image: np.ndarray,
    patch: int = 32,
    dim: int = 64,
    caption: str = "",
) -> PageIndex:
    patches = image_to_patches(image, patch=patch)
    vecs = embed_patches(patches, dim=dim, caption=caption or path)
    return PageIndex(page_id=page_id, path=path, page_no=page_no, patches=vecs)


def query_to_patches(query: str, dim: int = 64, n_tokens: int = 8) -> np.ndarray:
    """Bag of overlapping query spans as 'token' vectors for MaxSim."""
    q = query or ""
    spans = []
    words = q.split() or [q]
    for i in range(len(words)):
        spans.append(" ".join(words[max(0, i - 1) : i + 2]))
    if not spans:
        spans = [q]
    return HashEmbedder(dim=dim).encode(spans[:n_tokens])


def search_pages(query: str, pages: list[PageIndex], k: int = 5) -> list[tuple[PageIndex, float]]:
    dim = int(pages[0].patches.shape[1]) if pages and pages[0].patches.size else 64
    q_tok = query_to_patches(query, dim=dim)
    scored = [(page, maxsim(q_tok, page.patches)) for page in pages]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]


def render_text_page(text: str, size: tuple[int, int] = (256, 320)) -> np.ndarray:
    """Fallback page image when poppler/pdf2image is not installed."""
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("RGB", (size[1], size[0]), (194, 213, 255))
        draw = ImageDraw.Draw(img)
        y = 8
        for line in (text or "").splitlines()[:40]:
            draw.text((8, y), line[:48], fill=(18, 18, 18))
            y += 12
            if y > size[0] - 12:
                break
        return np.asarray(img)
    except Exception:
        # tiny synthetic page from character codes
        h, w = size
        canvas = np.full((h, w), 210, dtype=np.uint8)
        for i, ch in enumerate((text or "")[: h * 2]):
            canvas[i % h, (i * 3) % w] = ord(ch) % 256
        return canvas


def index_pdf_as_pages(path: str | Path, max_pages: int = 4) -> list[PageIndex]:
    path = Path(path)
    pages: list[PageIndex] = []
    rendered = []
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages[:max_pages]):
            text = page.extract_text() or ""
            rendered.append((i, render_text_page(text)))
    except Exception:
        rendered.append((0, render_text_page(path.read_text(errors="ignore")[:800])))
    for i, image in rendered:
        pages.append(
            index_page(f"{path.name}#p{i}", str(path), i, image)
        )
    return pages
