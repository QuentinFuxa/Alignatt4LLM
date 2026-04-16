from __future__ import annotations

import re
import unicodedata


_EXTRA_SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"\s+([,.;:!?])")
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")

# Target-language codes that LongYAAL / XCOMET evaluate at character level and
# whose surface form uses no whitespace between emission units.
CHAR_LEVEL_TARGET_LANG_CODES: frozenset[str] = frozenset({"zh", "ja"})


def normalize_incremental_target_text(text: str) -> str:
    """Stitch obviously broken sentence boundaries from incremental decoding."""

    text = text.strip()
    if not text:
        return ""

    text = _EXTRA_SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", text)
    text = _MULTISPACE_RE.sub(" ", text)
    return text.strip()


def is_char_level_target_lang(lang_code: str | None) -> bool:
    if not lang_code:
        return False
    return lang_code.lower() in CHAR_LEVEL_TARGET_LANG_CODES


def split_target_emission_units(text: str, *, target_lang_code: str | None) -> list[str]:
    """Split a target surface string into the units used for latency accounting.

    For whitespace-separated languages (en->de, en->it, ...) one unit is one
    whitespace-delimited word, so the evaluator can line up ``delays_ms`` with
    per-word LongYAAL computations. For non-spacing scripts (en->zh, en->ja)
    each non-whitespace Unicode character is its own unit after NFKC
    normalization, which keeps the ``hypothesis.jsonl`` timestamps aligned
    with OmniSTEval's character-level resegmentation (which also NFKC
    normalizes before counting units).
    """
    if not text:
        return []
    if is_char_level_target_lang(target_lang_code):
        normalized = unicodedata.normalize("NFKC", text)
        return [char for char in normalized if not char.isspace()]
    return text.split()
