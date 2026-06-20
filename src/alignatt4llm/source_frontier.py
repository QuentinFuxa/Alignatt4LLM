from __future__ import annotations

from dataclasses import dataclass
import re
import string
from typing import Iterable, Sequence


LEADING_TRIM_CHARS = "\"'`“”‘’([{"
TRAILING_TRIM_CHARS = string.punctuation + "”’)]}"
NONSPACE_PATTERN = re.compile(r"\S+")


@dataclass(frozen=True)
class SourceUnit:
    text: str
    char_start: int
    char_end: int
    start_ms: float | None
    end_ms: float | None
    is_accessible: bool
    is_final: bool


@dataclass(frozen=True)
class SourceAccessibilityFrontier:
    source_text: str
    units: tuple[SourceUnit, ...]
    accessible_unit_count: int
    current_source_ms: float
    inaccessible_ms: float
    is_final: bool

    @property
    def current_audio_ms(self) -> float:
        """Compatibility alias for the historical cascade runtime."""
        return self.current_source_ms


def iter_source_word_spans(text: str) -> Iterable[tuple[int, int, str]]:
    for match in NONSPACE_PATTERN.finditer(text):
        start, end = match.span()
        while start < end and text[start] in LEADING_TRIM_CHARS:
            start += 1
        while end > start and text[end - 1] in TRAILING_TRIM_CHARS:
            end -= 1
        if start < end:
            yield start, end, text[start:end]


def normalize_word_timestamps_ms(time_stamps: Sequence[object] | None) -> list[tuple[float | None, float | None]]:
    if not time_stamps:
        return []
    normalized: list[tuple[float | None, float | None]] = []
    for stamp in time_stamps:
        start_time = getattr(stamp, "start_time", None)
        end_time = getattr(stamp, "end_time", None)
        normalized.append(
            (
                None if start_time is None else float(start_time) * 1000.0,
                None if end_time is None else float(end_time) * 1000.0,
            )
        )
    return normalized


def build_source_accessibility_frontier(
    source_text: str,
    *,
    word_timestamps_ms: Sequence[tuple[float | None, float | None]] | None = None,
    current_source_ms: float | None = None,
    current_audio_ms: float | None = None,
    inaccessible_ms: float,
    is_final: bool,
    max_accessible_chars: int | None = None,
) -> SourceAccessibilityFrontier:
    """Build the source frontier over the full MT-visible source prefix.

    Callers pass the complete source prefix they want the MT to condition on.
    The returned frontier does not ask callers to truncate that source prefix;
    it only marks which source units AlignAtt may treat as currently accessible
    when deciding how much target text to accept.
    """
    if current_source_ms is None:
        if current_audio_ms is None:
            raise TypeError("current_source_ms is required")
        current_source_ms = float(current_audio_ms)

    spans = list(iter_source_word_spans(source_text))
    timestamps = list(word_timestamps_ms or [])
    units: list[SourceUnit] = []

    if is_final:
        accessible_unit_count = len(spans)
    elif not spans:
        accessible_unit_count = 0
    elif not timestamps:
        accessible_unit_count = max(0, len(spans) - 1)
    else:
        accessible_until_ms = max(0.0, float(current_source_ms) - float(inaccessible_ms))
        accessible_unit_count = 0
        for idx, _ in enumerate(spans):
            if idx >= len(timestamps):
                break
            _, end_ms = timestamps[idx]
            if end_ms is None or end_ms > accessible_until_ms:
                break
            accessible_unit_count += 1
    if max_accessible_chars is not None and not is_final:
        char_cap = max(0, int(max_accessible_chars))
        lcp_accessible_unit_count = 0
        for _, char_end, _ in spans:
            if int(char_end) > char_cap:
                break
            lcp_accessible_unit_count += 1
        accessible_unit_count = min(accessible_unit_count, lcp_accessible_unit_count)

    for idx, (char_start, char_end, token_text) in enumerate(spans):
        start_ms = None
        end_ms = None
        if idx < len(timestamps):
            start_ms, end_ms = timestamps[idx]
        units.append(
            SourceUnit(
                text=token_text,
                char_start=char_start,
                char_end=char_end,
                start_ms=start_ms,
                end_ms=end_ms,
                is_accessible=idx < accessible_unit_count,
                is_final=is_final,
            )
        )

    return SourceAccessibilityFrontier(
        source_text=source_text,
        units=tuple(units),
        accessible_unit_count=accessible_unit_count,
        current_source_ms=float(current_source_ms),
        inaccessible_ms=float(inaccessible_ms),
        is_final=bool(is_final),
    )
