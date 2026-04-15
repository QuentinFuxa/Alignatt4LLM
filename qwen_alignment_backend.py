"""Qwen3-ASR + Qwen3-ForcedAligner alignment backend.

Wraps the current ``Qwen3ASRModel.transcribe(..., return_time_stamps=True)``
path as an :class:`AlignmentBackend`. This is the baseline that the
Gemma-only aligner research path must match or explain-why-not.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from alignment_backend import AlignmentBackend, AlignmentResult, WordAlignment


class QwenAlignmentBackend(AlignmentBackend):
    name = "qwen_asr_qwen_aligner"

    def __init__(
        self,
        *,
        asr_model_path: str,
        forced_aligner_model_path: str,
        runtime_config: SimpleNamespace,
    ):
        self.asr_model_path = asr_model_path
        self.forced_aligner_model_path = forced_aligner_model_path
        self.runtime_config = runtime_config
        self.asr = None

    def load(self) -> None:
        if self.asr is not None:
            return
        from qwen_asr import Qwen3ASRModel

        self.asr = Qwen3ASRModel.LLM(
            model=self.asr_model_path,
            gpu_memory_utilization=self.runtime_config.asr_gpu_memory_utilization,
            max_inference_batch_size=1,
            max_model_len=1024,
            max_new_tokens=1024,
            forced_aligner=self.forced_aligner_model_path,
            forced_aligner_kwargs={
                "dtype": torch.bfloat16,
                "device_map": "cuda",
            },
        )

    def adopt_loaded_asr(self, asr) -> None:
        """Attach an already-loaded Qwen3ASRModel instance (hot-reload path)."""
        self.asr = asr

    def transcribe_and_align(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        language: str,
    ) -> AlignmentResult | None:
        if self.asr is None:
            raise RuntimeError("QwenAlignmentBackend is not loaded. Call load() first.")

        audio = np.asarray(audio, dtype=np.float32)
        audio_duration_s = float(len(audio)) / float(sample_rate)

        asr_outputs = self.asr.transcribe(
            (audio, sample_rate),
            language=language,
            context="",
            return_time_stamps=True,
        )
        output = asr_outputs[0]

        time_stamps = getattr(output, "time_stamps", None) or ()
        if time_stamps and float(time_stamps[-1].end_time) > audio_duration_s:
            return None

        words = tuple(
            WordAlignment(
                text=str(stamp.text),
                start_time=float(stamp.start_time),
                end_time=float(stamp.end_time),
            )
            for stamp in time_stamps
        )

        return AlignmentResult(
            text=str(output.text),
            words=words,
            audio_duration_s=audio_duration_s,
            diagnostics={"backend": self.name},
        )
