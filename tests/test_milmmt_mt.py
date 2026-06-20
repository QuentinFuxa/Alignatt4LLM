from __future__ import annotations

from types import SimpleNamespace

import pytest

from alignatt4llm.mt.gemma_vllm_backend import (
    MiLMMTVLLMMTBackend,
    render_milmmt_prompt_text,
)
from alignatt4llm.source_frontier import build_source_accessibility_frontier
from alignatt4llm.text_surface import strip_repeated_accepted_prefix
from alignatt4llm.translation_variants import TRANSLATION_VARIANTS


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


def test_milmmt_direct_prompt_matches_model_card_shape_and_source_map():
    runtime_config = SimpleNamespace(
        source_lang="English",
        target_lang="Chinese",
        milmmt_prompt_mode="direct",
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


def test_source_frontier_caps_accessibility_to_stable_prefix_chars():
    frontier = build_source_accessibility_frontier(
        "In everyday life. Hilma's office",
        word_timestamps_ms=[
            (0.0, 100.0),
            (100.0, 200.0),
            (200.0, 300.0),
            (300.0, 400.0),
            (400.0, 500.0),
        ],
        current_audio_ms=2_000.0,
        inaccessible_ms=0.0,
        is_final=False,
        max_accessible_chars=len("In everyday life"),
    )

    assert [unit.text for unit in frontier.units] == [
        "In",
        "everyday",
        "life",
        "Hilma's",
        "office",
    ]
    assert frontier.accessible_unit_count == 3
    assert [unit.is_accessible for unit in frontier.units] == [
        True,
        True,
        True,
        False,
        False,
    ]


def test_milmmt_prefill_restart_after_punctuation_keeps_bridge_only():
    continuation = strip_repeated_accepted_prefix(
        accepted_prefix="大家好,我是珍妮",
        generated_continuation="。我是珍妮,卡内基梅隆大学的一名博士生",
    )

    assert continuation == "。卡内基梅隆大学的一名博士生"


def test_milmmt_direct_prompt_has_source_span_without_extra_instructions():
    prompt_text, source_span = render_milmmt_prompt_text(
        source_lang="en",
        target_lang="zh",
        source_text="Mateo Negri, BLEU and chrF are metrics.",
    )

    assert "Preserve names, numbers, acronyms, symbols, and tags" not in prompt_text
    assert prompt_text.startswith(
        "Translate this from English to Chinese (Simplified):\nEnglish: "
    )
    assert prompt_text[source_span[0] : source_span[1]] == (
        "Mateo Negri, BLEU and chrF are metrics."
    )
    assert prompt_text.endswith("Chinese (Simplified):")


def test_milmmt_direct_prompt_has_no_paper_context_surface():
    source_text = "CoScript improves constrained language planning."
    prompt_text, source_span = render_milmmt_prompt_text(
        source_lang="en",
        target_lang="zh",
        source_text=source_text,
    )

    assert prompt_text[source_span[0] : source_span[1]] == source_text
    assert "[Paper context]" not in prompt_text
    assert prompt_text.endswith("Chinese (Simplified):")


def test_milmmt_direct_renderer_ignores_structured_paper_context():
    runtime_config = SimpleNamespace(
        source_lang="English",
        target_lang="Chinese",
        milmmt_prompt_mode="direct",
        mt_max_model_len=1024,
        gemma_max_model_len=1024,
    )
    backend = MiLMMTVLLMMTBackend(
        model_name="xiaomi-research/MiLMMT-46-4B-v0.1",
        runtime_config=runtime_config,
    )
    backend.tokenizer = CharOffsetTokenizer()

    source_text = "CoScript improves constrained language planning."
    rendered = TRANSLATION_VARIANTS["alignatt_prefix"].render_messages(
        source_lang="English",
        target_lang="Chinese",
        text=source_text,
        source_frontier=None,
        source_history=[],
        translation_history=[],
        is_partial=True,
        paper_context_block="[Paper context]\nTitle: CoScript",
    )

    prompt_text = backend.render_prompt_text(rendered)

    assert rendered.paper_context_block.startswith("[Paper context]")
    assert "[Paper context]" not in prompt_text
    assert f"English: {source_text}" in prompt_text


def test_milmmt_sampling_uses_model_card_greedy_defaults_and_end_of_turn_stop():
    runtime_config = SimpleNamespace(
        milmmt_temperature=0.0,
        milmmt_top_p=1.0,
        milmmt_top_k=1,
        milmmt_repetition_penalty=1.0,
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
    from alignatt4llm.mt.gemma_vllm_observer import (
        install_global_gemma_attention_mt_patches,
    )
    from vllm.model_executor.models.gemma3 import Gemma3Attention

    install_global_gemma_attention_mt_patches()

    assert hasattr(Gemma3Attention, "_alignatt_mt_qk_original_forward")


class CharOffsetTokenizerWithBos(CharOffsetTokenizer):
    bos_token_id = 777


def _render_simple_prefix_prompt(source_text: str, assistant_prefill: str = ""):
    variant = TRANSLATION_VARIANTS["alignatt_prefix"]
    frontier = build_source_accessibility_frontier(
        source_text,
        word_timestamps_ms=None,
        current_audio_ms=10_000.0,
        inaccessible_ms=0.0,
        is_final=True,
    )
    return variant.render_messages(
        source_lang="English",
        target_lang="Chinese",
        text=source_text,
        source_frontier=frontier,
        source_history=[],
        translation_history=[],
        is_partial=True,
        assistant_prefill=assistant_prefill,
    )


def _milmmt_backend(*, add_bos: bool, tokenizer):
    runtime_config = SimpleNamespace(
        source_lang="English",
        target_lang="Chinese",
        milmmt_prompt_mode="direct",
        milmmt_prompt_add_bos=add_bos,
        mt_max_model_len=1024,
        gemma_max_model_len=1024,
    )
    backend = MiLMMTVLLMMTBackend(
        model_name="xiaomi-research/MiLMMT-46-4B-v0.1",
        runtime_config=runtime_config,
    )
    backend.tokenizer = tokenizer
    return backend


def test_milmmt_prompt_add_bos_prepends_bos_and_shifts_source_map():
    rendered = _render_simple_prefix_prompt("It is on the house", "这是")
    package_off = _milmmt_backend(
        add_bos=False, tokenizer=CharOffsetTokenizerWithBos()
    ).render_prompt_package(rendered)
    package_on = _milmmt_backend(
        add_bos=True, tokenizer=CharOffsetTokenizerWithBos()
    ).render_prompt_package(rendered)

    assert list(package_on.prompt_token_ids) == [777] + list(
        package_off.prompt_token_ids
    )
    assert package_on.prompt_text == package_off.prompt_text
    assert package_on.source_map is not None and package_off.source_map is not None
    assert package_on.source_map.source_token_positions == tuple(
        position + 1 for position in package_off.source_map.source_token_positions
    )
    for span_on, span_off in zip(
        package_on.source_map.source_unit_spans,
        package_off.source_map.source_unit_spans,
    ):
        assert span_on.prompt_token_positions == tuple(
            position + 1 for position in span_off.prompt_token_positions
        )
    assert (
        package_on.source_map.accessible_source_token_count
        == package_off.source_map.accessible_source_token_count
    )


def test_milmmt_prompt_add_bos_is_inert_without_bos_token_or_by_default():
    rendered = _render_simple_prefix_prompt("It is on the house")
    no_bos_tokenizer = _milmmt_backend(
        add_bos=True, tokenizer=CharOffsetTokenizer()
    ).render_prompt_package(rendered)
    default_off = _milmmt_backend(
        add_bos=False, tokenizer=CharOffsetTokenizerWithBos()
    ).render_prompt_package(rendered)
    assert 777 not in no_bos_tokenizer.prompt_token_ids
    assert 777 not in default_off.prompt_token_ids
