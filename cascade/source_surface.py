from __future__ import annotations

from dataclasses import dataclass
import re
import string


NONSPACE_PATTERN = re.compile(r"\S+")
LEADING_TRIM_CHARS = "\"'`“”‘’([{"
TRAILING_TRIM_CHARS = string.punctuation + "”’)]}"


@dataclass(frozen=True)
class LexicalWordSpan:
    """One whitespace-delimited lexical unit on the source side.

    Qwen's forced aligner returns one timing per punctuation-stripped lexical
    word. The capture and replay code must therefore recover word boundaries
    from the surface string the same way every time, or we silently attach
    timings to the wrong source units.
    """

    surface: str
    normalized_core: str
    char_start: int
    char_end: int


def _normalize_raw_token(raw_token: str) -> str:
    start = 0
    end = len(raw_token)
    while start < end and raw_token[start] in LEADING_TRIM_CHARS:
        start += 1
    while end > start and raw_token[end - 1] in TRAILING_TRIM_CHARS:
        end -= 1
    return raw_token[start:end]


def lexical_word_spans(text: str) -> list[LexicalWordSpan]:
    spans: list[LexicalWordSpan] = []
    for match in NONSPACE_PATTERN.finditer(text):
        raw = match.group(0)
        normalized_core = _normalize_raw_token(raw)
        if not normalized_core:
            continue
        spans.append(
            LexicalWordSpan(
                surface=raw,
                normalized_core=normalized_core,
                char_start=match.start(),
                char_end=match.end(),
            )
        )
    return spans


def lexical_word_count(text: str) -> int:
    return len(lexical_word_spans(text))


def lexical_word_surfaces(text: str) -> list[str]:
    return [span.surface for span in lexical_word_spans(text)]


def slice_text_to_word_count(text: str, n_words: int) -> str:
    if n_words <= 0:
        return ""
    spans = lexical_word_spans(text)
    if not spans:
        return ""
    if n_words >= len(spans):
        return text.strip()
    return text[: spans[n_words - 1].char_end].rstrip()


def word_count_from_char_prefix(text: str, char_limit: int) -> int:
    if char_limit <= 0:
        return 0
    count = 0
    for span in lexical_word_spans(text):
        if span.char_end <= char_limit:
            count += 1
        else:
            break
    return count


def project_char_lcp_to_word_prefix_text(current_text: str, lcp_text: str) -> str:
    """Project a raw char-level LCP to a whole-word source prefix.

    We keep the exact surface from ``current_text`` instead of rebuilding the
    prefix from normalized word pieces. This preserves quotes / punctuation and
    makes the reinjected prompt identical to what the model itself produced.
    """

    return slice_text_to_word_count(
        current_text,
        word_count_from_char_prefix(current_text, len(lcp_text)),
    )
