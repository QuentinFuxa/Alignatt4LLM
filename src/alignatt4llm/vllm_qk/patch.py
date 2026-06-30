"""Spec-driven vLLM attention patch + the standard decoder forward.

The patch installer swaps a model family's vLLM attention ``forward`` for one
that captures post-rotary Q/K through the ``alignatt::capture_mt_qk`` custom op
(registered in :mod:`alignatt4llm.vllm_qk.observer`). The observer buffers,
registry, configure/stub/prepare/fetch helpers, and the reconstruction all live
in ``observer.py``; this module only installs the patch and provides the common
non-Gemma forward. Gemma supplies its own forward (it reshapes differently and
has a Gemma4 KV-sharing branch).
"""
from __future__ import annotations

import torch

from alignatt4llm.vllm_qk.observer import _ensure_custom_op_registered
from alignatt4llm.vllm_qk.spec import (
    VLLMAttentionSpec,
    assert_supported_attention_module,
    resolve_attention_classes,
)


def make_standard_decoder_patched_forward(spec: VLLMAttentionSpec):
    """Patched ``forward`` for the standard vLLM decoder attention shape.

    Mirrors the standard vLLM decoder attention (Qwen3 / Llama-class): split the
    fused QKV, apply optional per-head QK-norm (``self.qk_norm`` flag, e.g. Qwen3
    uses it, Llama-class does not), apply the rotary embedding, then capture the
    post-rotary Q/K through the custom op before paged attention and the output
    projection. Capturing post-norm-post-rotary is what makes the reconstructed
    attention match what the model actually attends with.

    Gemma keeps its own bespoke forward (it uses a different reshape and a
    Gemma4 KV-sharing branch), so this covers the common, non-Gemma case.
    """
    family = spec.family
    required_attrs = spec.required_attrs

    def _patched_forward(self, positions, hidden_states, **kwargs):
        assert_supported_attention_module(self, spec)
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Optional per-head QK-norm before RoPE (same reshape vLLM uses). Applied
        # when the module carries q_norm/k_norm: Qwen3 always norms (no flag),
        # Qwen2-class gates on self.qk_norm, Llama has no norms. Capturing the
        # post-norm Q/K is required or the reconstructed attention is wrong.
        q_norm = getattr(self, "q_norm", None)
        k_norm = getattr(self, "k_norm", None)
        if q_norm is not None and k_norm is not None and getattr(self, "qk_norm", True):
            total_tokens = q.shape[0]
            q = q_norm(q.view(total_tokens, self.num_heads, self.head_dim)).view(
                total_tokens, self.q_size
            )
            k = k_norm(k.view(total_tokens, self.num_kv_heads, self.head_dim)).view(
                total_tokens, self.kv_size
            )

        q, k = self.rotary_emb(positions, q, k)

        # Dispatch capture through the opaque custom op so AOT-compile can't
        # trace into the observer scatters; the zero-sentinel return is added
        # to attn_output to create a data dependency inductor can't DCE under
        # cudagraph=full. See observer._ensure_custom_op_registered.
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
