"""Custom vLLM worker for the experimental Gemma AlignAtt observer.

The goal of this worker is narrow and research-oriented:

- install the tensor-buffer observer on the Gemma attention modules before any
  observer-aware warmup/cudagraph capture happens
- defer compile/cudagraph warmup until the backend has armed the observer with
  the real prompt/audio span for the current request

This keeps the experiment clean: the graph we compile or capture is built with
the observer already present, instead of trying to patch Python state after the
engine has stabilized its execution path.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Sequence

from vllm.config import CUDAGraphMode
from vllm.config.compilation import CompilationMode
from vllm.logger import init_logger
from vllm.v1.worker.gpu_worker import Worker as VLLMGPUWorker

from cascade.vllm_compat import compilation_time_seconds, ensure_compilation_times
from cascade.alignment.gemma_vllm_asr_backend import (
    _decode_tensor_observer_bootstrap_from_env,
    _configure_audio_qk_tensor_observer_on_model,
    _fetch_audio_qk_tensor_observer_from_model,
    _prepare_audio_qk_tensor_observer_on_model,
    _resolve_tensor_observer_bindings,
    install_global_gemma4_attention_tensor_patch,
)

logger = init_logger(__name__)


class GemmaVLLMASRWorker(VLLMGPUWorker):
    """Single-GPU worker that defers observer-aware warmup until request prep."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._alignatt_observer_configured = False
        self._alignatt_observer_prepared = False
        self._alignatt_observer_warm = False

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

    @contextmanager
    def _temporarily_disable_compile_and_cudagraph(self):
        compilation_config = self.vllm_config.compilation_config
        model_config = self.vllm_config.model_config
        saved_mode = compilation_config.mode
        saved_cudagraph_mode = compilation_config.cudagraph_mode
        saved_max_capture = compilation_config.max_cudagraph_capture_size
        saved_capture_sizes = compilation_config.cudagraph_capture_sizes
        saved_num_warmups = compilation_config.cudagraph_num_of_warmups
        saved_enforce_eager = model_config.enforce_eager
        compilation_config.mode = CompilationMode.NONE
        compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        compilation_config.max_cudagraph_capture_size = 0
        compilation_config.cudagraph_capture_sizes = []
        compilation_config.cudagraph_num_of_warmups = 0
        model_config.enforce_eager = True
        try:
            yield
        finally:
            compilation_config.mode = saved_mode
            compilation_config.cudagraph_mode = saved_cudagraph_mode
            compilation_config.max_cudagraph_capture_size = saved_max_capture
            compilation_config.cudagraph_capture_sizes = saved_capture_sizes
            compilation_config.cudagraph_num_of_warmups = saved_num_warmups
            model_config.enforce_eager = saved_enforce_eager

    def determine_available_memory(self) -> int:
        # Avoid building the execution path before the observer is armed with the
        # real prompt/audio metadata. The later explicit warmup is the one we care
        # about for this experiment.
        with self._temporarily_disable_compile_and_cudagraph():
            return super().determine_available_memory()

    def compile_or_warm_up_model(self):
        if not self._alignatt_observer_prepared:
            logger.info(
                "Deferring compile/warmup until prepare_audio_observer arms the "
                "Gemma AlignAtt observer."
            )
            return ensure_compilation_times(0.0)
        if self._alignatt_observer_warm:
            return ensure_compilation_times(
                self.vllm_config.compilation_config.compilation_time
            )
        warmup_time = ensure_compilation_times(super().compile_or_warm_up_model())
        self._alignatt_observer_warm = True
        return warmup_time

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
        self._alignatt_observer_configured = True
        self._alignatt_observer_prepared = False
        self._alignatt_observer_warm = False
        return result

    def prepare_audio_observer(
        self,
        prompt_length: int,
        audio_prompt_start: int,
        audio_prompt_length: int,
    ) -> dict[str, Any]:
        if not self._alignatt_observer_configured:
            raise RuntimeError("configure_audio_observer must be called before prepare.")
        result = _prepare_audio_qk_tensor_observer_on_model(
            self.get_model(),
            prompt_length=int(prompt_length),
            audio_prompt_start=int(audio_prompt_start),
            audio_prompt_length=int(audio_prompt_length),
        )
        self._alignatt_observer_prepared = True
        if not self._alignatt_observer_warm:
            warmup_time = ensure_compilation_times(super().compile_or_warm_up_model())
            self._alignatt_observer_warm = True
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
