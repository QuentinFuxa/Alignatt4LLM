"""Qwen3-ASR + Qwen3-ForcedAligner alignment backend.

Wraps the current ``Qwen3ASRModel.transcribe(..., return_time_stamps=True)``
path as an :class:`AlignmentBackend`. This is the baseline that the
Gemma-only aligner research path must match or explain-why-not.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from cascade.alignment.base import AlignmentBackend, AlignmentResult, WordAlignment


_QWEN_RUNTIME_PATCHED = False


def _qwen3_asr_default_rope_init(config, device=None, seq_len=None, layer_type=None):
    standardize = getattr(config, "standardize_rope_params", None)
    if callable(standardize):
        standardize()

    rope_parameters = getattr(config, "rope_parameters", None) or {}
    if layer_type is not None and isinstance(rope_parameters, dict) and layer_type in rope_parameters:
        rope_parameters = rope_parameters[layer_type]

    base = rope_parameters.get("rope_theta", getattr(config, "rope_theta", 10000.0))
    partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
            / dim
        )
    )
    return inv_freq, 1.0


def _patched_qwen3_asr_get_text_config(self, decoder=False):
    del decoder
    thinker_config = getattr(self, "thinker_config", None)
    if thinker_config is not None:
        return thinker_config.get_text_config()

    text_config = getattr(self, "text_config", None)
    if text_config is not None:
        return text_config

    return self


def ensure_qwen_runtime_patched() -> None:
    global _QWEN_RUNTIME_PATCHED
    if _QWEN_RUNTIME_PATCHED:
        return

    import patch_qwen_asr_for_transformers5
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from qwen_asr.core.transformers_backend import modeling_qwen3_asr
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig

    patch_qwen_asr_for_transformers5.main()
    modeling_qwen3_asr._qwen3_asr_default_rope_init = _qwen3_asr_default_rope_init
    if "default" not in ROPE_INIT_FUNCTIONS:
        ROPE_INIT_FUNCTIONS["default"] = _qwen3_asr_default_rope_init
    Qwen3ASRConfig.get_text_config = _patched_qwen3_asr_get_text_config
    _QWEN_RUNTIME_PATCHED = True


class QwenAlignmentBackend(AlignmentBackend):
    name = "qwen_forced"

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
        ensure_qwen_runtime_patched()
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
        streaming_prefix_text: str = "",
        streaming_prefix_words: tuple[WordAlignment, ...] = (),
    ) -> AlignmentResult | None:
        if self.asr is None:
            raise RuntimeError("QwenAlignmentBackend is not loaded. Call load() first.")
        if streaming_prefix_text or streaming_prefix_words:
            raise NotImplementedError(
                "QwenAlignmentBackend does not yet support prompt-prefix streaming; "
                "the Qwen3-ASR vLLM path has its own streaming_transcribe() API that "
                "the cascade does not route through this backend."
            )

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
