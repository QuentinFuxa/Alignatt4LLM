"""Unit tests for the torch-free vLLM Q/K spec plug-in point.

Runnable under ``.venv-dev`` (no torch/vLLM): the spec and its class resolver
must import and behave without any heavy dependency.
"""
from __future__ import annotations

from collections import OrderedDict

from alignatt4llm.vllm_qk.spec import VLLMAttentionSpec, resolve_attention_classes


def _spec() -> VLLMAttentionSpec:
    return VLLMAttentionSpec(
        family="qwen3",
        attention_import_paths=(("vllm.model_executor.models.qwen3", "Qwen3Attention"),),
        required_attrs=("qkv_proj", "rotary_emb", "attn", "o_proj"),
        make_patched_forward=lambda spec: (lambda *a, **k: None),
    )


def test_original_forward_attr_is_family_namespaced():
    assert _spec().original_forward_attr() == "_alignatt_qwen3_mt_qk_original_forward"


def test_resolve_attention_classes_skips_missing_modules():
    # vLLM is absent in the dev env; the resolver must swallow ImportError and
    # return an empty tuple rather than raising.
    assert resolve_attention_classes(
        [("vllm.model_executor.models.qwen3", "Qwen3Attention")]
    ) == ()


def test_resolve_attention_classes_returns_real_classes():
    # Prove the resolver actually imports and returns a class when present.
    resolved = resolve_attention_classes([("collections", "OrderedDict")])
    assert resolved == (OrderedDict,)


def test_resolve_attention_classes_skips_non_class_attributes():
    # A name that exists but is not a class must be skipped, not returned.
    assert resolve_attention_classes([("collections", "namedtuple")]) == ()
