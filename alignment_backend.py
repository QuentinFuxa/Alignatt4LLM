"""Shared contract for streaming source-side ASR + forced alignment.

The cascade consumes a transcript + per-word start/end timestamps from an
ASR/aligner pair. Today this comes from Qwen3-ASR + Qwen3-ForcedAligner;
the goal of the Gemma-only aligner research path is to provide the same
contract from Gemma's own internals. This module defines the narrowest
interface that both implementations must satisfy so that the cascade
does not encode Qwen-specific assumptions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WordAlignment:
    """A single source-side word with its onset and offset in seconds.

    Field names mirror Qwen3-ForcedAligner's ``TimeStamp`` (``start_time``,
    ``end_time``) so that downstream helpers that read these via
    ``getattr`` (e.g. ``normalize_word_timestamps_ms``) work with either
    backend unchanged.
    """

    text: str
    start_time: float
    end_time: float


@dataclass(frozen=True)
class AlignmentResult:
    text: str
    words: tuple[WordAlignment, ...]
    audio_duration_s: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


class AlignmentBackend(ABC):
    """Transcribe an audio chunk and return per-word timestamps.

    Implementations own their models and caches. ``transcribe_and_align``
    is called on the active utterance tail, exactly like the current Qwen
    path; the returned text and per-word timestamps must use the same
    punctuation-stripped word unit as ``Qwen3ForcedAligner`` so the
    cascade's utterance-cut and source-frontier logic keeps working
    unchanged.
    """

    name: str = "alignment_backend"

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def transcribe_and_align(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        language: str,
    ) -> AlignmentResult | None:
        """Return ``None`` if the backend detected an invalid output (e.g.
        timestamp running past the end of the provided audio).
        """
        raise NotImplementedError

    def reset_caches(self) -> None:
        return None
