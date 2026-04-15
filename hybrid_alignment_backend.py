"""Hybrid alignment backend: Qwen3-ASR transcript + Gemma attention timings.

This is the realistic deployment path implied by the PLAN.md research:
Gemma-4 E4B's free-run ASR is unreliable on streaming-quality clips, but
its self-attention to the audio-placeholder span provides a usable
forced-alignment signal once heads and systematic offset are calibrated.
So we keep Qwen3-ASR as the text source and replace the Qwen3-Forced
Aligner-0.6B dependency with Gemma's attention — the downstream cascade
sees the same ``WordAlignment`` contract either way.
"""

from __future__ import annotations

import numpy as np

from alignment_backend import AlignmentBackend, AlignmentResult


class HybridQwenAsrGemmaAlignerBackend(AlignmentBackend):
    name = "hybrid_qwen_asr_gemma_aligner"

    def __init__(
        self,
        *,
        asr_backend: AlignmentBackend,
        gemma_backend,
    ):
        """``asr_backend`` produces the transcript; ``gemma_backend`` produces timings.

        ``gemma_backend`` must implement
        :meth:`gemma_alignment_probe.GemmaAttentionAlignmentBackend.align_transcript`.
        """
        self.asr_backend = asr_backend
        self.gemma_backend = gemma_backend

    def load(self) -> None:
        self.asr_backend.load()
        self.gemma_backend.load()

    def reset_caches(self) -> None:
        self.asr_backend.reset_caches()
        self.gemma_backend.reset_caches()

    def transcribe_and_align(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        language: str,
    ) -> AlignmentResult | None:
        asr_result = self.asr_backend.transcribe_and_align(
            audio, sample_rate=sample_rate, language=language
        )
        if asr_result is None:
            return None
        transcript = asr_result.text.strip()
        if not transcript:
            return asr_result

        gemma_result = self.gemma_backend.align_transcript(
            audio,
            sample_rate=sample_rate,
            language=language,
            transcript=transcript,
        )
        if gemma_result is None or not gemma_result.words:
            # Fall back to whatever the ASR backend produced (may be empty
            # when the ASR is run without its own forced aligner).
            return AlignmentResult(
                text=transcript,
                words=asr_result.words,
                audio_duration_s=asr_result.audio_duration_s,
                diagnostics={
                    "backend": self.name,
                    "fallback": "asr_backend_timings",
                    "asr_backend": self.asr_backend.name,
                    "aligner_backend": self.gemma_backend.name,
                },
            )

        # Gemma may have split the transcript into a slightly different
        # word count than Qwen's forced aligner. The downstream cascade
        # assumes one timestamp per punctuation-stripped word of the
        # ``text`` field, so we align counts by truncating whichever side
        # is longer. This is a generic shape guarantee, not a content fix.
        words = gemma_result.words
        if asr_result.words and len(words) != len(asr_result.words):
            n = min(len(words), len(asr_result.words))
            words = words[:n]

        return AlignmentResult(
            text=transcript,
            words=words,
            audio_duration_s=asr_result.audio_duration_s,
            diagnostics={
                "backend": self.name,
                "asr_backend": self.asr_backend.name,
                "aligner_backend": self.gemma_backend.name,
                "asr_word_count": len(asr_result.words),
                "gemma_word_count": len(gemma_result.words),
                "gemma_monotonicity": gemma_result.diagnostics.get("monotonicity"),
                "gemma_offset_s": gemma_result.diagnostics.get("word_end_offset_s"),
            },
        )
