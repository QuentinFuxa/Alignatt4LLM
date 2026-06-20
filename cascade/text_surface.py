from __future__ import annotations

import re
import unicodedata


_EXTRA_SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"\s+([,.;:!?])")
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")
_LATENCY_BOUNDARY_PUNCTUATION = ",.;:!?"
_CJK_RANGES = (
    ("\u3400", "\u4dbf"),
    ("\u4e00", "\u9fff"),
    ("\uf900", "\ufaff"),
)

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


def _is_cjk_char(char: str) -> bool:
    return any(start <= char <= end for start, end in _CJK_RANGES)


def _has_enough_overlap_content(overlap: str) -> bool:
    content = [
        char
        for char in overlap
        if not char.isspace() and not unicodedata.category(char).startswith("P")
    ]
    if any(_is_cjk_char(char) for char in content):
        return len(content) >= 2
    return len(content) >= 4


def _has_enough_punctuation_bridged_overlap_content(overlap: str) -> bool:
    content = [
        char
        for char in overlap
        if not char.isspace() and not unicodedata.category(char).startswith("P")
    ]
    if any(_is_cjk_char(char) for char in content):
        return len(content) >= 3
    return len(content) >= 6


def _leading_punctuation_span(text: str) -> int:
    idx = 0
    while idx < len(text) and unicodedata.category(text[idx]).startswith("P"):
        idx += 1
    return idx


def _strip_leading_boundary_punctuation(text: str) -> str:
    text = text.lstrip()
    while text and unicodedata.category(text[0]).startswith("P"):
        text = text[1:].lstrip()
    return text


def strip_repeated_accepted_prefix(
    *,
    accepted_prefix: str,
    generated_continuation: str,
) -> str:
    """Remove model restarts that repeat the already accepted target suffix.

    MiLMMT sometimes treats a seeded target prefix as context and starts the
    continuation by repeating a short suffix of it. The streaming surface is
    append-only, so keep the accepted prefix fixed and trim only exact
    suffix/prefix overlap from the newly generated continuation.
    """

    prefix = normalize_incremental_target_text(accepted_prefix)
    continuation = normalize_incremental_target_text(generated_continuation)
    if not prefix or not continuation:
        return continuation

    max_overlap = min(len(prefix), len(continuation))
    for overlap_len in range(max_overlap, 0, -1):
        overlap = continuation[:overlap_len]
        if not prefix.endswith(overlap):
            continue
        if not _has_enough_overlap_content(overlap):
            continue
        return continuation[overlap_len:].lstrip()

    bridge_len = _leading_punctuation_span(continuation)
    if bridge_len <= 0:
        return continuation
    bridge = continuation[:bridge_len]
    bridged_continuation = continuation[bridge_len:].lstrip()
    max_overlap = min(len(prefix), len(bridged_continuation))
    for overlap_len in range(max_overlap, 0, -1):
        overlap = bridged_continuation[:overlap_len]
        if not prefix.endswith(overlap):
            continue
        if not _has_enough_punctuation_bridged_overlap_content(overlap):
            continue
        remainder = _strip_leading_boundary_punctuation(
            bridged_continuation[overlap_len:]
        )
        return (bridge + remainder).strip()
    return continuation


def is_char_level_target_lang(lang_code: str | None) -> bool:
    if not lang_code:
        return False
    return lang_code.lower() in CHAR_LEVEL_TARGET_LANG_CODES


def split_target_emission_units(text: str, *, target_lang_code: str | None) -> list[str]:
    """Split a target surface string into the units used for latency accounting.

    For whitespace-separated languages (en->de, en->it, ...) one unit is one
    whitespace-delimited lexical word, with boundary punctuation stripped so
    retroactive commas / sentence marks do not rewrite earlier latency units.
    For non-spacing scripts (en->zh, en->ja) each non-whitespace Unicode
    character is its own unit after NFKC normalization, which keeps the
    ``hypothesis.jsonl`` timestamps aligned with OmniSTEval's character-level
    resegmentation (which also NFKC normalizes before counting units).
    """
    if not text:
        return []
    if is_char_level_target_lang(target_lang_code):
        normalized = unicodedata.normalize("NFKC", text)
        return [char for char in normalized if not char.isspace()]
    normalized = normalize_incremental_target_text(text)
    units: list[str] = []
    for token in normalized.split():
        lexical = token.strip(_LATENCY_BOUNDARY_PUNCTUATION)
        if lexical:
            units.append(lexical)
    return units


def split_public_emission_units(text: str, *, target_lang_code: str | None) -> list[str]:
    """Split the public append-only surface into exact monotone string units.

    Public streaming output is append-only at the normalized string level, not
    at the lexical-word level. This lets the surface grow monotonically through
    intra-word extensions such as German compounds (``Sprechtext`` ->
    ``Sprechtextrahmen``) and hyphen bridges (``Kreuz`` ->
    ``Kreuz-Attention``) without treating them as rewrites of the past.
    """
    if not text:
        return []
    del target_lang_code
    return list(normalize_incremental_target_text(text))


def join_public_emission_units(
    units: list[str] | tuple[str, ...],
    *,
    target_lang_code: str | None,
) -> str:
    del target_lang_code
    return "".join(units)


def prediction_text_from_target_surface(
    text: str,
    *,
    target_lang_code: str | None,
) -> str:
    if is_char_level_target_lang(target_lang_code):
        return "".join(split_target_emission_units(text, target_lang_code=target_lang_code))
    return normalize_incremental_target_text(text)
