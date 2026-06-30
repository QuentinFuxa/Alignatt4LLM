"""Reusable vLLM worker base for spec-driven Q/K observers.

``QKObserverWorkerLifecycle`` holds the model- and task-agnostic worker
lifecycle shared by every observer worker: defer compile/cudagraph warmup until
the observer is armed, and disable compile during the memory-profiling dummy run
so the graph is built with the observer present. Both the MT worker
(``BaseQKObserverWorker`` below) and the ASR worker
(``alignatt4llm.alignment.gemma_vllm_asr_worker``) inherit it.

``BaseQKObserverWorker`` adds the MT-specific arming: install the spec's
attention patch + stub observers, then configure/prepare/fetch the MT observer.
A new MT model's worker is just a subclass that sets ``spec``.

Imports ``vllm`` at module load, so it is only importable inside a vLLM worker
process (GPU).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Sequence

from vllm.config import CUDAGraphMode
from vllm.config.compilation import CompilationMode
from vllm.logger import init_logger
from vllm.v1.worker.gpu_worker import Worker as VLLMGPUWorker

from alignatt4llm.vllm_compat import compilation_time_seconds, ensure_compilation_times
from alignatt4llm.vllm_qk.observer import (
    _decode_mt_observer_bootstrap_from_env,
    _fetch_mt_qk_observer_from_model,
    _prepare_mt_qk_observer_on_model,
    _resolve_mt_observer_bindings,
    configure_mt_qk_observer_on_model,
    install_stub_observers_on_model,
)
from alignatt4llm.vllm_qk.patch import install_global_attention_mt_patch
from alignatt4llm.vllm_qk.spec import VLLMAttentionSpec

logger = init_logger(__name__)


class QKObserverWorkerLifecycle(VLLMGPUWorker):
    """Shared lifecycle for observer workers: defer warmup until the observer arms.

    Task-agnostic. Subclasses install/configure their own observer and flip
    ``self._observer_prepared`` / ``self._observer_warm`` when armed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._observer_configured = False
        self._observer_prepared = False
        self._observer_warm = False

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
        # Avoid building the execution path before the observer is armed; the
        # later explicit warmup (in prepare_*) is the one that matters.
        with self._temporarily_disable_compile_and_cudagraph():
            return super().determine_available_memory()

    def compile_or_warm_up_model(self):
        if not self._observer_prepared:
            logger.info("Deferring compile/warmup until the observer is armed.")
            return ensure_compilation_times(0.0)
        if self._observer_warm:
            return ensure_compilation_times(
                self.vllm_config.compilation_config.compilation_time
            )
        warmup_time = ensure_compilation_times(super().compile_or_warm_up_model())
        self._observer_warm = True
        return warmup_time


class BaseQKObserverWorker(QKObserverWorkerLifecycle):
    """Single-GPU MT observer worker. Subclasses set ``spec``.

    The worker_cls string a backend passes to vLLM must point at the subclass.
    """

    spec: VLLMAttentionSpec | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.spec is None:
            raise RuntimeError(
                f"{type(self).__name__}.spec must be set to a VLLMAttentionSpec."
            )

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        install_global_attention_mt_patch(self.spec)
        super().load_model(load_dummy_weights=load_dummy_weights)
        stubbed = install_stub_observers_on_model(self.get_model(), self.spec)
        print(
            f"[{self.spec.family}_vllm_mt_worker] Installed None-observer stubs "
            f"on {stubbed} attention layers.",
            flush=True,
        )
        bootstrap = _decode_mt_observer_bootstrap_from_env()
        if bootstrap is not None:
            self.configure_mt_observer(
                selected_heads=bootstrap["selected_heads"],
                max_prompt_tokens=int(bootstrap["max_prompt_tokens"]),
                max_decode_tokens=int(bootstrap["max_decode_tokens"]),
            )

    def configure_mt_observer(
        self,
        selected_heads: Sequence[dict[str, int]],
        max_prompt_tokens: int,
        max_decode_tokens: int,
    ) -> dict[str, Any]:
        result = configure_mt_qk_observer_on_model(
            self.get_model(),
            self.spec,
            selected_heads=selected_heads,
            max_prompt_tokens=int(max_prompt_tokens),
            max_decode_tokens=int(max_decode_tokens),
        )
        self._observer_configured = True
        self._observer_prepared = False
        self._observer_warm = False
        return result

    def prepare_mt_observer(self, prompt_length: int) -> dict[str, Any]:
        if not self._observer_configured:
            raise RuntimeError("configure_mt_observer must be called before prepare.")
        result = _prepare_mt_qk_observer_on_model(
            self.get_model(),
            prompt_length=int(prompt_length),
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
