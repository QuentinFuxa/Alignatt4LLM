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
class AlignAttProvenanceBreakdown:
    """Compact prompt-region mass summary for one observed token.

    This is the defensible engine-to-runtime contract for future AlignAtt
    backends: expose only the provenance summary the runtime can actually
    use, not full attention tensors or per-head rows in Python.
    """

    source_accessible: float
    source_inaccessible: float
    non_source_prompt: float
    suffix: float


@dataclass(frozen=True)
class AlignAttObserverToken:
    """Minimal per-token AlignAtt observer signal for runtime decisions.

    The runtime-side acceptance policy only needs compact observer outputs:
    token identity, aligned source/audio argmax, optional accessible-source
    mass, optional compact provenance, and optional blocked-source metadata.
    It must *not* depend on full attention matrices or per-head rows.
    """

    token_id: int
    token_str: str
    aligned_source_position: int | None
    source_accessible_mass: float | None = None
    blocked_source_local_position: int | None = None
    blocked_source_unit_index: int | None = None
    provenance: AlignAttProvenanceBreakdown | None = None


@dataclass(frozen=True)
class AlignmentResult:
    """Backend contract for ASR text, word timings, and compact observer data."""

    text: str
    words: tuple[WordAlignment, ...]
    audio_duration_s: float
    observer_tokens: tuple[AlignAttObserverToken, ...] = field(default_factory=tuple)
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
