"""Spec-driven vLLM attention patch + observer wiring (model-agnostic).

This reuses the proven, model-agnostic primitives from
``alignatt4llm.mt.gemma_vllm_observer`` — the ``alignatt::capture_mt_qk`` custom
op, the per-layer ``_MTPromptDecodeQKTensorObserver`` buffers, the global layer
registry, and ``reconstruct_mt_attention_rows`` — and adds only the parts that
must vary per model:

  * ``make_standard_decoder_patched_forward`` — the common no-QK-norm attention
    forward (Llama/Qwen2 shape): qkv_proj → rotary → capture → attn → o_proj.
  * ``install_global_attention_mt_patch(spec)`` — patch the spec's attention
    class(es) with the spec's forward.
  * ``install_stub_observers_on_model`` / ``configure_mt_qk_observer_on_model`` —
    spec-asserted variants of the Gemma helpers (the originals hard-assert
    Gemma's ``q_norm``/``k_norm``; these assert ``spec.required_attrs`` instead).

The Gemma MT path keeps its own bespoke forward (it applies QK-norm and handles
Gemma4 KV-sharing) and is intentionally left untouched here; converging it onto
this base is a separate, GPU-validated step.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch

from alignatt4llm.vllm_qk.spec import VLLMAttentionSpec, resolve_attention_classes

# Reused, model-agnostic primitives from the proven Gemma MT observer.
from alignatt4llm.mt.gemma_vllm_observer import (
    _LAYER_OBSERVER_REGISTRY,
    _MTPromptDecodeQKTensorObserver,
    _ensure_custom_op_registered,
    _get_mt_qk_tensor_observer,
    _register_layer_observer,
    _resolve_vllm_gemma_decoder_layers as resolve_decoder_layers,
)


def assert_supported_attention_module(
    attn_module: Any, *, family: str, required_attrs: Sequence[str]
) -> None:
    """Fail loudly at install time if the vLLM attention module lost an attr."""
    missing = [name for name in required_attrs if not hasattr(attn_module, name)]
    if missing:
        raise RuntimeError(
            f"{family} MT AlignAtt requires a vLLM attention module exposing "
            f"{tuple(required_attrs)}; missing attributes: {missing}. "
            "A vLLM version bump may have renamed these — update the "
            "VLLMAttentionSpec.required_attrs and the patched forward."
        )


def make_standard_decoder_patched_forward(spec: VLLMAttentionSpec):
    """Patched ``forward`` for standard decoder attention without QK-norm.

    Correct for Llama- and Qwen2-class attention: split the fused QKV, apply the
    rotary embedding, capture post-rotary Q/K through the custom op, run the
    paged attention, and project out. Models that apply per-head QK-norm (Gemma,
    Qwen3) must supply their own forward that norms before the rotary, otherwise
    the captured Q/K would be pre-norm and the reconstructed attention wrong.
    """
    family = spec.family
    required_attrs = spec.required_attrs

    def _patched_forward(self, positions, hidden_states, **kwargs):
        assert_supported_attention_module(
            self, family=family, required_attrs=required_attrs
        )
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)

        # Dispatch capture through the opaque custom op so AOT-compile can't
        # trace into the observer scatters; the zero-sentinel return is added
        # to attn_output to create a data dependency inductor can't DCE under
        # cudagraph=full. See gemma_vllm_observer._ensure_custom_op_registered.
        layer_idx = int(getattr(self, "_alignatt_mt_layer_idx", -1))
        observer_sentinel = torch.ops.alignatt.capture_mt_qk(layer_idx, positions, q, k)

        attn_output = self.attn(q, k, v)
        attn_output = attn_output + observer_sentinel
        output, _ = self.o_proj(attn_output)
        return output

    return _patched_forward


def install_global_attention_mt_patch(spec: VLLMAttentionSpec) -> None:
    """Patch the spec's vLLM attention class(es) with the spec's forward.

    Idempotent: the original forward is preserved under
    ``spec.original_forward_attr()`` and re-patching is skipped if present.
    """
    _ensure_custom_op_registered()
    classes = resolve_attention_classes(spec.attention_import_paths)
    if not classes:
        raise RuntimeError(
            f"{spec.family} MT AlignAtt: none of {spec.attention_import_paths} "
            "are importable from this vLLM build."
        )
    original_attr = spec.original_forward_attr()
    patched_forward = spec.make_patched_forward(spec)
    for cls in classes:
        if not hasattr(cls, original_attr):
            setattr(cls, original_attr, cls.forward)
            cls.forward = patched_forward


def install_stub_observers_on_model(model, spec: VLLMAttentionSpec) -> int:
    """Seed every attention layer with a ``None`` observer stub + layer index.

    Spec-asserted variant of the Gemma helper: keeps the
    ``_alignatt_mt_qk_tensor_observer`` attribute present in ``__dict__`` (so the
    AOT-compiled forward's lookup succeeds before configure arms real observers)
    and tags each module with its integer layer index for the custom-op lookup.
    """
    layers = resolve_decoder_layers(model)
    stubbed = 0
    for layer_idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        assert_supported_attention_module(
            attn, family=spec.family, required_attrs=spec.required_attrs
        )
        if "_alignatt_mt_qk_tensor_observer" not in attn.__dict__:
            attn._alignatt_mt_qk_tensor_observer = None
            stubbed += 1
        attn._alignatt_mt_layer_idx = int(layer_idx)
        _LAYER_OBSERVER_REGISTRY.setdefault(int(layer_idx), None)
    return stubbed


def configure_mt_qk_observer_on_model(
    model,
    spec: VLLMAttentionSpec,
    *,
    selected_heads: Sequence[dict[str, int]],
    max_prompt_tokens: int,
    max_decode_tokens: int,
) -> dict[str, Any]:
    """Arm per-layer Q/K observers for the selected heads.

    Spec-asserted variant of ``_configure_mt_qk_observer_on_model``: reuses the
    proven ``_MTPromptDecodeQKTensorObserver`` and the global registry; only the
    attention-module assertion is spec-driven (rather than Gemma-specific).
    """
    from alignatt4llm.mt.base import map_attention_head_to_key_value_head

    layers = resolve_decoder_layers(model)
    heads_by_layer: dict[int, list[int]] = {}
    for head in selected_heads:
        heads_by_layer.setdefault(int(head["layer"]), []).append(int(head["head"]))
    if not heads_by_layer:
        raise ValueError("selected_heads must not be empty for MT tensor observer")

    layer_indices: list[int] = []
    for layer_idx, layer_head_indices in heads_by_layer.items():
        layer = layers[int(layer_idx)]
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            raise RuntimeError(f"Layer {layer_idx} has no self_attn module.")
        assert_supported_attention_module(
            attn, family=spec.family, required_attrs=spec.required_attrs
        )
        device = attn.qkv_proj.weight.device

        selected_kv_heads = [
            map_attention_head_to_key_value_head(
                int(head_index),
                num_attention_heads=int(attn.num_heads),
                num_key_value_heads=int(attn.num_kv_heads),
            )
            for head_index in layer_head_indices
        ]
        observer = _get_mt_qk_tensor_observer(attn)
        if observer is None or any(
            (
                tuple(observer.selected_heads)
                != tuple(int(h) for h in layer_head_indices),
                tuple(int(h) for h in observer.selected_kv_heads)
                != tuple(int(h) for h in selected_kv_heads),
                int(observer.prompt_k_buffer.shape[1]) != int(max_prompt_tokens),
                int(observer.decode_q_buffer.shape[0]) != int(max_decode_tokens),
                int(observer.head_dim) != int(attn.head_dim),
            )
        ):
            observer = _MTPromptDecodeQKTensorObserver(
                selected_heads=list(layer_head_indices),
                selected_kv_heads=selected_kv_heads,
                max_prompt_tokens=int(max_prompt_tokens),
                max_decode_tokens=int(max_decode_tokens),
                head_dim=int(attn.head_dim),
                scaling=float(getattr(attn, "scaling", 1.0)),
                device=device,
            )
        attn._alignatt_mt_qk_tensor_observer = observer
        attn._alignatt_mt_layer_idx = int(layer_idx)
        _register_layer_observer(int(layer_idx), observer)
        layer_indices.append(int(layer_idx))

    model._alignatt_mt_qk_state = {
        "storage_mode": "tensor_buffers",
        "layer_indices": tuple(sorted(layer_indices)),
    }
    return {
        "layer_count": len(heads_by_layer),
        "max_prompt_tokens": int(max_prompt_tokens),
        "max_decode_tokens": int(max_decode_tokens),
    }
