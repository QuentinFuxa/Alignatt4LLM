"""Custom vLLM worker for the experimental Gemma AlignAtt ASR observer.

Inherits the shared ``QKObserverWorkerLifecycle`` (defer compile/cudagraph
warmup until the observer is armed) from the ``vllm_qk`` base, and adds the
ASR-specific arming: install the audio-K tensor observer patch, then
configure/prepare/fetch the audio observer. The audio observer Module and its
argmax reconstruction live in ``gemma_vllm_asr_backend`` because the ASR capture
(audio-K only, raw-score argmax) is a different task from the MT 4-way
provenance observer.
"""

from __future__ import annotations

from typing import Any, Sequence

from vllm.logger import init_logger

from alignatt4llm.vllm_compat import compilation_time_seconds, ensure_compilation_times
from alignatt4llm.vllm_qk.worker import QKObserverWorkerLifecycle
from alignatt4llm.alignment.gemma_vllm_asr_backend import (
    _configure_audio_qk_tensor_observer_on_model,
    _decode_tensor_observer_bootstrap_from_env,
    _fetch_audio_qk_tensor_observer_from_model,
    _prepare_audio_qk_tensor_observer_on_model,
    _resolve_tensor_observer_bindings,
    install_global_gemma4_attention_tensor_patch,
)

logger = init_logger(__name__)


class GemmaVLLMASRWorker(QKObserverWorkerLifecycle):
    """Single-GPU worker that defers observer-aware warmup until request prep."""

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        install_global_gemma4_attention_tensor_patch()
        super().load_model(load_dummy_weights=load_dummy_weights)
        bootstrap = _decode_tensor_observer_bootstrap_from_env()
        if bootstrap is not None:
            self.configure_audio_observer(
                selected_heads=bootstrap["selected_heads"],
                max_audio_tokens=int(bootstrap["max_audio_tokens"]),
                max_decode_tokens=int(bootstrap["max_decode_tokens"]),
            )

    def configure_audio_observer(
        self,
        selected_heads: Sequence[dict[str, int]],
        max_audio_tokens: int,
        max_decode_tokens: int,
    ) -> dict[str, Any]:
        result = _configure_audio_qk_tensor_observer_on_model(
            self.get_model(),
            selected_heads=selected_heads,
            max_audio_tokens=int(max_audio_tokens),
            max_decode_tokens=int(max_decode_tokens),
        )
        self._observer_configured = True
        self._observer_prepared = False
        self._observer_warm = False
        return result

    def prepare_audio_observer(
        self,
        prompt_length: int,
        audio_prompt_start: int,
        audio_prompt_length: int,
    ) -> dict[str, Any]:
        if not self._observer_configured:
            raise RuntimeError("configure_audio_observer must be called before prepare.")
        result = _prepare_audio_qk_tensor_observer_on_model(
            self.get_model(),
            prompt_length=int(prompt_length),
            audio_prompt_start=int(audio_prompt_start),
            audio_prompt_length=int(audio_prompt_length),
        )
        self._observer_prepared = True
        if not self._observer_warm:
            warmup_time = ensure_compilation_times(super().compile_or_warm_up_model())
            self._observer_warm = True
            self._verify_observer_integrity("post-warmup")
            result = {
                **result,
                "warmup_triggered": True,
                "warmup_compilation_time_s": compilation_time_seconds(warmup_time),
                "observer_intact_after_warmup": True,
            }
        else:
            result = {
                **result,
                "warmup_triggered": False,
                "warmup_compilation_time_s": compilation_time_seconds(
                    self.vllm_config.compilation_config.compilation_time
                ),
            }
        return result

    def _verify_observer_integrity(self, label: str) -> None:
        """Verify tensor observer modules survived compile/cudagraph."""
        bindings = _resolve_tensor_observer_bindings(self.get_model())
        if not bindings:
            raise RuntimeError(
                f"Observer integrity check failed ({label}): no tensor observer "
                "bindings found on the model after warmup. The compile/cudagraph "
                "pass may have replaced the attention modules."
            )
        for layer_idx, observer in bindings:
            if observer.prompt_audio_k_buffer is None:
                raise RuntimeError(
                    f"Observer integrity check failed ({label}): layer {layer_idx} "
                    "observer has no prompt_audio_k_buffer after warmup."
                )
        logger.info(
            "Observer integrity verified (%s): %d layer bindings intact.",
            label,
            len(bindings),
        )

    def fetch_audio_observer_payload(self) -> dict[str, Any] | None:
        return _fetch_audio_qk_tensor_observer_from_model(self.get_model())
