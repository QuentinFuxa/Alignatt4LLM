"""Custom vLLM worker for the Gemma-family MT AlignAtt observer.

Mirrors ``cascade.alignment.gemma_vllm_asr_worker.GemmaVLLMASRWorker``
(ASR-side) but installs the
MT-specific Q/K tensor observer so the backend captures K at **both** prompt
and decode positions, which is what the 4-way MT provenance partition
requires.

The two workers are kept independent because the observer state is stored on
different attributes on Gemma-family attention modules and the patched forward
call targets a different capture function. The stable route pairs
``qwen_forced`` ASR with ``gemma_vllm_alignatt`` MT; MiLMMT is the active
improvement route.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Sequence

from vllm.config import CUDAGraphMode
from vllm.config.compilation import CompilationMode
from vllm.logger import init_logger
from vllm.v1.worker.gpu_worker import Worker as VLLMGPUWorker

from alignatt4llm.vllm_compat import compilation_time_seconds, ensure_compilation_times
from alignatt4llm.mt.gemma_vllm_observer import (
    _configure_mt_qk_observer_on_model,
    _decode_mt_observer_bootstrap_from_env,
    _fetch_mt_qk_observer_from_model,
    _prepare_mt_qk_observer_on_model,
    _resolve_mt_observer_bindings,
    install_global_gemma_attention_mt_patch,
    install_stub_observers_on_model,
)

logger = init_logger(__name__)


class GemmaVLLMMTWorker(VLLMGPUWorker):
    """Single-GPU worker that defers observer-aware warmup until request prep."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mt_observer_configured = False
        self._mt_observer_prepared = False
        self._mt_observer_warm = False

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        install_global_gemma_attention_mt_patch()
        super().load_model(load_dummy_weights=load_dummy_weights)
        # Seed every attention layer with a None observer stub so the
        # AOT-compiled forward's __dict__ lookup succeeds even before
        # configure_mt_observer arms real observers. Fixes a KeyError
        # that surfaced when vLLM's memory-profiling dummy_run replayed
        # a cached compiled graph on a fresh input shape (e.g. cs->en
        # as the first inference-time direction of the process).
        stubbed = install_stub_observers_on_model(self.get_model())
        # Print (not logger) so the install shows up in stdout for the
        # batch runners even when the vLLM logger is quieted — it's the
        # only way to verify the stub count from outside the worker
        # process.
        print(
            f"[gemma_vllm_mt_worker] Installed None-observer stubs "
            f"on {stubbed} Gemma-family attention layers.",
            flush=True,
        )
        bootstrap = _decode_mt_observer_bootstrap_from_env()
        if bootstrap is not None:
            self.configure_mt_observer(
                selected_heads=bootstrap["selected_heads"],
                max_prompt_tokens=int(bootstrap["max_prompt_tokens"]),
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
        with self._temporarily_disable_compile_and_cudagraph():
            return super().determine_available_memory()

    def compile_or_warm_up_model(self):
        if not self._mt_observer_prepared:
            logger.info(
                "Deferring MT compile/warmup until prepare_mt_observer arms the "
                "observer."
            )
            return ensure_compilation_times(0.0)
        if self._mt_observer_warm:
            return ensure_compilation_times(
                self.vllm_config.compilation_config.compilation_time
            )
        warmup_time = ensure_compilation_times(super().compile_or_warm_up_model())
        self._mt_observer_warm = True
        return warmup_time

    def configure_mt_observer(
        self,
        selected_heads: Sequence[dict[str, int]],
        max_prompt_tokens: int,
        max_decode_tokens: int,
    ) -> dict[str, Any]:
        result = _configure_mt_qk_observer_on_model(
            self.get_model(),
            selected_heads=selected_heads,
            max_prompt_tokens=int(max_prompt_tokens),
            max_decode_tokens=int(max_decode_tokens),
        )
        self._mt_observer_configured = True
        self._mt_observer_prepared = False
        self._mt_observer_warm = False
        return result

    def prepare_mt_observer(self, prompt_length: int) -> dict[str, Any]:
        if not self._mt_observer_configured:
            raise RuntimeError("configure_mt_observer must be called before prepare.")
        result = _prepare_mt_qk_observer_on_model(
            self.get_model(),
            prompt_length=int(prompt_length),
        )
        self._mt_observer_prepared = True
        if not self._mt_observer_warm:
            warmup_time = ensure_compilation_times(super().compile_or_warm_up_model())
            self._mt_observer_warm = True
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
        bindings = _resolve_mt_observer_bindings(self.get_model())
        if not bindings:
            raise RuntimeError(
                f"MT observer integrity check failed ({label}): no tensor observer "
                "bindings found on the model after warmup."
            )
        for layer_idx, observer in bindings:
            if observer.prompt_k_buffer is None:
                raise RuntimeError(
                    f"MT observer integrity check failed ({label}): layer {layer_idx} "
                    "observer has no prompt_k_buffer after warmup."
                )
        logger.info(
            "MT observer integrity verified (%s): %d layer bindings intact.",
            label,
            len(bindings),
        )

    def fetch_mt_observer_payload(self) -> dict[str, Any] | None:
        return _fetch_mt_qk_observer_from_model(self.get_model())
