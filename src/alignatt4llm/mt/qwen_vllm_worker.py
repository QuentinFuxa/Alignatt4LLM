"""Custom vLLM worker for the Qwen3 MT AlignAtt observer.

The reference example of how thin a new model's worker is once the generic base
exists: set ``spec`` and inherit the entire warmup/compile-deferral lifecycle
from :class:`BaseQKObserverWorker`. Imports ``vllm`` (via the base), so it is
only importable inside a vLLM worker process.
"""
from __future__ import annotations

from alignatt4llm.mt.qwen_vllm_backend import QWEN_SPEC
from alignatt4llm.vllm_qk.worker import BaseQKObserverWorker


class QwenVLLMMTWorker(BaseQKObserverWorker):
    """Single-GPU Qwen3 observer worker (Qwen3Attention, standard forward + QK-norm)."""

    spec = QWEN_SPEC
