"""sha256 helpers with no numpy dependency (used at corpus freeze time)."""

from __future__ import annotations

import hashlib


def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()
