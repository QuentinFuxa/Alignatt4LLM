"""Custom vLLM worker for the Gemma-family MT AlignAtt observer.

Thin :class:`BaseQKObserverWorker` subclass: the full warmup / compile-deferral
lifecycle lives in the generic ``vllm_qk`` base. Gemma only supplies
``GEMMA_SPEC`` (its vLLM attention classes plus the QK-norm / Gemma4 KV-sharing
forward). Used by both the Gemma MT route and the MiLMMT route, which share the
Gemma attention classes.
"""
from __future__ import annotations

from alignatt4llm.mt.gemma_vllm_backend import GEMMA_SPEC
from alignatt4llm.vllm_qk.worker import BaseQKObserverWorker


class GemmaVLLMMTWorker(BaseQKObserverWorker):
    """Single-GPU Gemma-family observer worker."""

    spec = GEMMA_SPEC
