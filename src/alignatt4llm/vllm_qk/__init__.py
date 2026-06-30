"""Generic vLLM Q/K observer base for porting AlignAtt to other decoder-only LLMs.

AlignAtt4LLM reconstructs target-to-source attention from selected decoder heads
captured at runtime inside vLLM. That capture machinery (a torch custom op, a
fixed-buffer per-layer observer, and a Q@K^T reconstruction) is model-agnostic;
only two things are model-specific:

  1. which vLLM attention class to patch, and
  2. how that class's ``forward`` recomputes Q/K (e.g. whether it applies
     per-head QK-norm before the rotary embedding).

This package captures (1) and (2) in a :class:`VLLMAttentionSpec` and provides a
spec-driven patch installer plus a reusable worker base, so a new model plugs in
by supplying a spec and a thin backend subclass. The model-agnostic primitives
live here: ``observer`` (per-layer buffers, custom op, configure/fetch,
reconstruction), ``patch`` (the installer + standard forward), and ``worker``
(``BaseQKObserverWorker``). Gemma keeps only its bespoke forward + ``GEMMA_SPEC``
in ``alignatt4llm.mt.gemma_vllm_backend``.

Only ``spec`` is imported here so the package can be imported without torch/vLLM
(the other submodules pull torch/vLLM and are imported by their consumers).

See ``docs/adding_a_model.md`` and ``alignatt4llm.mt.qwen_vllm_backend`` for the
worked reference (Qwen3, standard attention with QK-norm).
"""
from __future__ import annotations

from alignatt4llm.vllm_qk.spec import (
    VLLMAttentionSpec,
    assert_supported_attention_module,
    resolve_attention_classes,
)

__all__ = [
    "VLLMAttentionSpec",
    "assert_supported_attention_module",
    "resolve_attention_classes",
]
