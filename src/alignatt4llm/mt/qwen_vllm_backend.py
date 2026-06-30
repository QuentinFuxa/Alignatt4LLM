"""Reference MT AlignAtt backend for a non-Gemma decoder-only LLM (Qwen2.5).

This is the worked "bring your own LLM" example referenced by
``docs/adding_a_model.md``. It shows the minimal surface to add a new model:

  * a :class:`VLLMAttentionSpec` naming the vLLM attention class and the
    standard (no-QK-norm) patched forward, and
  * a ~30-line backend subclass that reuses the entire MiLMMT/Gemma AlignAtt
    ``translate`` loop, acceptance policy, prompt rendering, and observer
    reconstruction unchanged.

Qwen2.5 is chosen deliberately: it is openly downloadable (Apache-2.0), uses
grouped-query attention (exercising the head->KV-head mapping on a non-Gemma
layout), and its vLLM attention does NOT apply per-head QK-norm, so the generic
``make_standard_decoder_patched_forward`` is exactly correct. (Qwen3 and Gemma
add QK-norm and therefore need a norm-aware forward instead.)

NOTE: like every vLLM-touching path here, the runtime correctness of the
captured Q/K must be validated on a GPU box. The attribute names and forward
shape are asserted at patch-install time and will fail loudly on a vLLM bump.
"""
from __future__ import annotations

from typing import Any

from alignatt4llm.mt.base import (
    BaseMTBackend,
    RenderedPromptWithSourceMap,
)
from alignatt4llm.mt.gemma_vllm_backend import MiLMMTVLLMMTBackend
from alignatt4llm.translation_variants import RenderedTranslationPrompt
from alignatt4llm.vllm_qk.patch import make_standard_decoder_patched_forward
from alignatt4llm.vllm_qk.spec import VLLMAttentionSpec

# The plug-in point: which vLLM attention class to patch and how. Standard
# Llama/Qwen2-class attention (qkv_proj -> rotary -> attn -> o_proj), no QK-norm.
QWEN_SPEC = VLLMAttentionSpec(
    family="qwen2",
    attention_import_paths=(
        ("vllm.model_executor.models.qwen2", "Qwen2Attention"),
    ),
    required_attrs=(
        "qkv_proj",
        "q_size",
        "kv_size",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "rotary_emb",
        "attn",
        "o_proj",
    ),
    make_patched_forward=make_standard_decoder_patched_forward,
)


class QwenVLLMMTBackend(MiLMMTVLLMMTBackend):
    """Qwen2.5 MT AlignAtt backend — the reference "bring your own LLM" example.

    Reuses the full MiLMMT/Gemma AlignAtt ``translate`` loop and acceptance
    policy; only the chat-template prompt contract, the vLLM attention spec, and
    the observer worker differ.
    """

    backend_name = "qwen_vllm_alignatt"
    model_family = "Qwen"
    context_config_attr = "mt_max_model_len"
    qk_spec = QWEN_SPEC

    def _validate_alignatt_heads(self) -> None:
        # Head/layer bounds are model-specific. The observer asserts the model's
        # actual attention shape at configure time and the head loader clips to
        # top-k, so an out-of-range (layer, head) fails loudly there. A model
        # author may add explicit bounds (num_layers, num_attention_heads) here.
        return None

    def build_llm_init_kwargs(self) -> dict[str, Any]:
        kwargs = super().build_llm_init_kwargs()
        # Qwen uses its own observer worker (Qwen2Attention, not Gemma classes).
        # Canonical module path (new code does not use the legacy `cascade.`
        # namespace shim).
        kwargs["worker_cls"] = "alignatt4llm.mt.qwen_vllm_worker.QwenVLLMMTWorker"
        return kwargs

    def build_sampling_params_kwargs(
        self,
        *,
        max_new_tokens: int,
        stop_token_ids: list[int],
    ) -> dict[str, Any]:
        return {
            "temperature": 0.0,
            "top_k": 1,
            "max_tokens": int(max_new_tokens),
            "repetition_penalty": float(
                getattr(self.runtime_config, "repetition_penalty", 1.0)
            ),
            "stop_token_ids": stop_token_ids or None,
            "skip_special_tokens": False,
        }

    def resolve_generation_stop_token_ids(self) -> tuple[int, ...]:
        return BaseMTBackend.resolve_generation_stop_token_ids(self)

    # Qwen2.5-Instruct ships a chat template, so reuse the chat-template prompt
    # contract (same path Gemma uses) rather than MiLMMT's raw-completion prompt.
    def render_prompt_token_ids(
        self,
        rendered_prompt: RenderedTranslationPrompt,
    ) -> list[int]:
        return BaseMTBackend.render_prompt_token_ids(self, rendered_prompt)

    def render_prompt_text(self, rendered_prompt: RenderedTranslationPrompt) -> str:
        return BaseMTBackend.render_prompt_text(self, rendered_prompt)

    def render_prompt_package(
        self,
        rendered_prompt: RenderedTranslationPrompt,
    ) -> RenderedPromptWithSourceMap:
        return BaseMTBackend.render_prompt_package(self, rendered_prompt)
