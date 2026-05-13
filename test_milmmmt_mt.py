from __future__ import annotations

from types import SimpleNamespace

import pytest

from cascade.mt.milmmmt_vllm_backend import (
    MiLMMTVLLMMTBackend,
    render_milmmmt_prompt_text,
)
from cascade.source_frontier import build_source_accessibility_frontier
from cascade.translation_variants import TRANSLATION_VARIANTS


class CharOffsetTokenizer:
    all_special_ids = [1]
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        del add_special_tokens
        text = str(text)
        payload = {"input_ids": list(range(len(text)))}
        if return_offsets_mapping:
            payload["offset_mapping"] = [(idx, idx + 1) for idx in range(len(text))]
        return payload

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(65 + (int(i) % 26)) for i in ids)

    def convert_ids_to_tokens(self, ids):
        return [str(i) for i in ids]

    def convert_tokens_to_ids(self, token):
        return 106 if token == "<end_of_turn>" else -1


def test_milmmmt_direct_prompt_matches_model_card_shape_and_source_map():
    runtime_config = SimpleNamespace(
        source_lang="English",
        target_lang="Chinese",
        milmmmt_prompt_mode="direct",
        mt_max_model_len=1024,
        gemma_max_model_len=1024,
    )
    backend = MiLMMTVLLMMTBackend(
        model_name="xiaomi-research/MiLMMT-46-4B-v0.1",
        runtime_config=runtime_config,
    )
    backend.tokenizer = CharOffsetTokenizer()

    variant = TRANSLATION_VARIANTS["alignatt_prefix"]
    source_text = "It is on the house"
    frontier = build_source_accessibility_frontier(
        source_text,
        word_timestamps_ms=None,
        current_audio_ms=10_000.0,
        inaccessible_ms=0.0,
        is_final=True,
    )
    rendered = variant.render_messages(
        source_lang="English",
        target_lang="Chinese",
        text=source_text,
        source_frontier=frontier,
        source_history=[],
        translation_history=[],
        is_partial=True,
        assistant_prefill="这是",
    )

    package = backend.render_prompt_package(rendered)

    assert package.prompt_text == (
        "Translate this from English to Chinese (Simplified):\n"
        "English: It is on the house\n"
        "Chinese (Simplified):这是"
    )
    assert package.source_map is not None
    assert package.source_map.source_text == source_text
    assert package.source_map.source_token_positions
    assert 0 < package.source_map.accessible_source_token_count <= len(
        package.source_map.source_token_positions
    )


def test_milmmmt_direct_preserve_is_general_instruction_not_source_hack():
    prompt_text, source_span = render_milmmmt_prompt_text(
        source_lang="en",
        target_lang="zh",
        source_text="Mateo Negri, BLEU and chrF are metrics.",
        preserve_names_numbers_tags=True,
    )

    assert "Preserve names, numbers, acronyms, symbols, and tags" in prompt_text
    assert prompt_text[source_span[0] : source_span[1]] == (
        "Mateo Negri, BLEU and chrF are metrics."
    )
    assert prompt_text.endswith("Chinese (Simplified):")


def test_milmmmt_sampling_uses_model_card_greedy_defaults_and_end_of_turn_stop():
    runtime_config = SimpleNamespace(
        milmmmt_temperature=0.0,
        milmmmt_top_p=1.0,
        milmmmt_top_k=1,
        milmmmt_repetition_penalty=1.0,
        repetition_penalty=1.05,
    )
    backend = MiLMMTVLLMMTBackend(
        model_name="xiaomi-research/MiLMMT-46-4B-v0.1",
        runtime_config=runtime_config,
    )
    backend.tokenizer = CharOffsetTokenizer()

    kwargs = backend.build_sampling_params_kwargs(max_new_tokens=8, stop_token_ids=[1])

    assert backend.resolve_generation_stop_token_ids() == (1, 106)
    assert kwargs["temperature"] == pytest.approx(0.0)
    assert kwargs["top_p"] == pytest.approx(1.0)
    assert kwargs["top_k"] == 1
    assert kwargs["repetition_penalty"] == pytest.approx(1.0)
    assert kwargs["max_tokens"] == 8


def test_gemma3_mt_attention_patch_is_installable():
    pytest.importorskip("vllm.model_executor.models.gemma3")
    from cascade.mt.gemma_vllm_observer import (
        install_global_gemma_attention_mt_patches,
    )
    from vllm.model_executor.models.gemma3 import Gemma3Attention

    install_global_gemma_attention_mt_patches()

    assert hasattr(Gemma3Attention, "_alignatt_mt_qk_original_forward")
