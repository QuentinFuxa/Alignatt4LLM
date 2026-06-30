"""Tests for the Qwen3 reference MT backend and its observer wiring.

Heavy imports (torch/vLLM) are done *inside* the tests so this file collects
cleanly under ``.venv-dev`` and simply skips when those deps are absent — the
patch-installability check only runs where vLLM's Qwen2 model is importable.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


def _runtime_config(**overrides) -> SimpleNamespace:
    base = dict(
        mt_backend_name="qwen_vllm_alignatt",
        mt_max_model_len=1024,
        max_new_tokens=160,
        partial_max_new_tokens=48,
        mt_vllm_enforce_eager=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_backend_registration_dispatches_to_qwen():
    pytest.importorskip("torch")
    from alignatt4llm.mt.base import build_mt_backend
    from alignatt4llm.mt.qwen_vllm_backend import QwenVLLMMTBackend

    backend = build_mt_backend(
        model_name="Qwen/Qwen3-1.7B",
        runtime_config=_runtime_config(),
    )
    assert isinstance(backend, QwenVLLMMTBackend)
    assert backend.backend_name == "qwen_vllm_alignatt"
    # The worker_cls must be the canonical (non-cascade) module path.
    assert (
        backend.build_llm_init_kwargs()["worker_cls"]
        == "alignatt4llm.mt.qwen_vllm_worker.QwenVLLMMTWorker"
    )


def test_runtime_registration_resolves_model_and_heads_path():
    pytest.importorskip("torch")
    from alignatt4llm.runtime import (
        VALID_MT_BACKEND_NAMES,
        alignatt_heads_path_for,
        mt_model_name_for_backend,
    )

    assert "qwen_vllm_alignatt" in VALID_MT_BACKEND_NAMES
    assert "Qwen3" in mt_model_name_for_backend("qwen_vllm_alignatt")
    path = alignatt_heads_path_for(
        "English", "German", mt_backend_name="qwen_vllm_alignatt"
    )
    assert path.endswith(
        "translation_heads_Qwen_Qwen3-1_7B_en-de.json"
    )


def test_qwen_spec_targets_qwen3_attention_with_standard_forward():
    pytest.importorskip("torch")
    from alignatt4llm.mt.qwen_vllm_backend import QWEN_SPEC
    from alignatt4llm.vllm_qk.patch import make_standard_decoder_patched_forward

    assert QWEN_SPEC.family == "qwen3"
    assert QWEN_SPEC.attention_import_paths == (
        ("vllm.model_executor.models.qwen3", "Qwen3Attention"),
    )
    # Qwen3 enables QK-norm; the standard forward handles that branch.
    assert "q_norm" in QWEN_SPEC.required_attrs
    assert callable(QWEN_SPEC.make_patched_forward)
    assert QWEN_SPEC.make_patched_forward is make_standard_decoder_patched_forward
    assert callable(make_standard_decoder_patched_forward(QWEN_SPEC))


def test_qwen3_attention_patch_is_installable():
    # Only runs where vLLM's Qwen3 attention class is importable (GPU/inference
    # env). Mirrors test_milmmt_mt's Gemma patch-installability check.
    pytest.importorskip("vllm.model_executor.models.qwen3")
    from alignatt4llm.mt.qwen_vllm_backend import QWEN_SPEC
    from alignatt4llm.vllm_qk.patch import install_global_attention_mt_patch
    from vllm.model_executor.models.qwen3 import Qwen3Attention

    install_global_attention_mt_patch(QWEN_SPEC)
    assert hasattr(Qwen3Attention, "_alignatt_qwen3_mt_qk_original_forward")
