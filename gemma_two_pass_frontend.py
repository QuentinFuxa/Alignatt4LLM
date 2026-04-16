"""Two-pass full-Gemma alignment backend: default-attention ASR + eager-attention alignment.

Pass 1 uses default attention for free-run ASR (WER 0.03-0.26).
Pass 2 uses eager attention for forced alignment on the transcript from pass 1.
This removes the Qwen ASR dependency while keeping word-level timing via
Gemma's own attention-based alignment.
"""

from __future__ import annotations

import numpy as np

from alignment_backend import AlignmentBackend, AlignmentResult
from gemma_alignment_probe import GemmaAttentionAlignmentBackend


class GemmaTwoPassAlignmentBackend(AlignmentBackend):
    name = "gemma_two_pass"

    def __init__(self, *, gemma_backend: GemmaAttentionAlignmentBackend):
        self.gemma_backend = gemma_backend

    def load(self) -> None:
        self.gemma_backend.load()

    def reset_caches(self) -> None:
        self.gemma_backend.reset_caches()

    def transcribe_and_align(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        language: str,
    ) -> AlignmentResult | None:
        transcript = self.gemma_backend.transcribe(
            audio, sample_rate=sample_rate, language=language
        )
        if not transcript:
            return None

        alignment = self.gemma_backend.align_transcript(
            audio,
            sample_rate=sample_rate,
            language=language,
            transcript=transcript,
        )
        if alignment is None or not alignment.words:
            audio_duration_s = float(len(audio)) / float(sample_rate)
            alignment_diag = {} if alignment is None else alignment.diagnostics
            return AlignmentResult(
                text=transcript,
                words=(),
                audio_duration_s=audio_duration_s,
                diagnostics={
                    "backend": self.name,
                    "asr_attention": "default",
                    "alignment_attention": alignment_diag.get("alignment_attention"),
                    "gemma_probe_backend": alignment_diag.get("probe_backend"),
                    "alignment_failed": True,
                },
            )

        return AlignmentResult(
            text=transcript,
            words=alignment.words,
            audio_duration_s=alignment.audio_duration_s,
            diagnostics={
                "backend": self.name,
                "asr_attention": "default",
                "alignment_attention": alignment.diagnostics.get("alignment_attention"),
                "gemma_probe_backend": alignment.diagnostics.get("probe_backend"),
                "gemma_monotonicity": alignment.diagnostics.get("monotonicity"),
                "gemma_offset_s": alignment.diagnostics.get("word_end_offset_s"),
                "gemma_audio_span_length": alignment.diagnostics.get("audio_span_length"),
                "gemma_qk_fast_reconstruction_succeeded": alignment.diagnostics.get(
                    "qk_fast_reconstruction_succeeded"
                ),
                "word_count": len(alignment.words),
            },
        )
