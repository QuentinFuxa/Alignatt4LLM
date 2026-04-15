from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from cascade_source_frontier import iter_source_word_spans


_MULTISPACE_RE = re.compile(r"[ \t]{2,}")
_EXTRA_SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"\s+([,.;:!?])")


@dataclass(frozen=True)
class NormalizedSourceText:
    text: str
    word_timestamps_ms: tuple[tuple[float | None, float | None], ...]


def normalize_source_text_for_mt(
    source_text: str,
    *,
    word_timestamps_ms: Sequence[tuple[float | None, float | None]] | None = None,
) -> NormalizedSourceText:
    text = source_text.strip()
    if not text:
        return NormalizedSourceText(text="", word_timestamps_ms=())

    words, separators = _decompose_source_text(text)
    timestamps = _align_word_timestamps(words, word_timestamps_ms)

    normalized_text = _compose_source_text(words, separators)
    normalized_text = _normalize_source_surface(normalized_text)

    return NormalizedSourceText(
        text=normalized_text,
        word_timestamps_ms=tuple(timestamps),
    )


def _decompose_source_text(text: str) -> tuple[list[str], list[str]]:
    spans = list(iter_source_word_spans(text))
    if not spans:
        return [], [text]

    words: list[str] = []
    separators: list[str] = []
    previous_end = 0
    for start, end, word in spans:
        separators.append(text[previous_end:start])
        words.append(word)
        previous_end = end
    separators.append(text[previous_end:])
    return words, separators


def _align_word_timestamps(
    words: Sequence[str],
    word_timestamps_ms: Sequence[tuple[float | None, float | None]] | None,
) -> list[tuple[float | None, float | None]]:
    timestamps = list(word_timestamps_ms or [])
    aligned: list[tuple[float | None, float | None]] = []
    for idx, _ in enumerate(words):
        if idx < len(timestamps):
            aligned.append(tuple(timestamps[idx]))
        else:
            aligned.append((None, None))
    return aligned


def _compose_source_text(words: Sequence[str], separators: Sequence[str]) -> str:
    if not words:
        return separators[0] if separators else ""

    chunks: list[str] = []
    for idx, word in enumerate(words):
        chunks.append(separators[idx])
        chunks.append(word)
    chunks.append(separators[-1])
    return "".join(chunks).strip()


def _normalize_source_surface(text: str) -> str:
    text = _EXTRA_SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", text)
    text = _MULTISPACE_RE.sub(" ", text)
    return text.strip()
