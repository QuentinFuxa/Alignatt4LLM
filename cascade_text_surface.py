from __future__ import annotations

import re


_EXTRA_SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"\s+([,.;:!?])")
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")


def normalize_incremental_target_text(text: str) -> str:
    """Stitch obviously broken sentence boundaries from incremental decoding."""

    text = text.strip()
    if not text:
        return ""

    text = _EXTRA_SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", text)
    text = _MULTISPACE_RE.sub(" ", text)
    return text.strip()
