from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_backend_registry_exposes_only_two_supported_frontends():
    from cascade_runtime import CascadeRuntimeConfig, VALID_ALIGNMENT_BACKEND_NAMES

    assert VALID_ALIGNMENT_BACKEND_NAMES == (
        "qwen_forced",
        "gemma_onepass_qk_fast",
    )
    assert CascadeRuntimeConfig(alignment_backend_name="qwen_forced").alignment_backend_name == "qwen_forced"
    assert (
        CascadeRuntimeConfig(alignment_backend_name="gemma_onepass_qk_fast").alignment_backend_name
        == "gemma_onepass_qk_fast"
    )
    with pytest.raises(ValueError):
        CascadeRuntimeConfig(alignment_backend_name="hybrid_qwen_asr_gemma_aligner")


def test_new_sessions_keep_mutable_state_isolated():
    from cascade_runtime import CascadeRuntimeConfig, LoadedModelBundle

    bundle = LoadedModelBundle(CascadeRuntimeConfig())
    first = bundle.new_session()
    second = bundle.new_session()

    first.state.utt_sources.append("hello.")
    first.state.partial_translation.accepted_target = "Hallo."
    first.mt_prompt_cache.full_prompt_ids.append(42)

    assert second.state.utt_sources == [""]
    assert second.state.partial_translation.accepted_target == ""
    assert second.mt_prompt_cache.full_prompt_ids == []
    assert first.translation_units is not second.translation_units


def test_processor_runtime_config_propagates_backend_and_audio_probe_settings():
    from cascade_simulstream_processor import CascadeAlignAttProcessor

    runtime_config = CascadeAlignAttProcessor._build_runtime_config(
        SimpleNamespace(
            source_lang_code="en",
            target_lang_code="de",
            alignment_backend_name="gemma_onepass_qk_fast",
            min_start_seconds=2.0,
            max_history_utterances=1,
            partial_max_new_tokens=16,
            partial_followup_max_new_tokens=8,
            translation_alignatt_inaccessible_ms=0.0,
            translation_alignatt_rewind_threshold=8,
            translation_alignatt_min_source_mass=0.2,
            translation_alignatt_top_k_heads=8,
            translation_alignatt_filter_width=7,
            translation_alignatt_probe_mode="qk_fast",
            translation_scheduler_stall_seconds=1.2,
            temperature=0.0,
            repetition_penalty=1.05,
            gemma_audio_align_probe_mode="qk_fast",
            gemma_audio_alignment_heads_path="assets/attention_heads/audio_alignment_heads_google_gemma-4-E4B-it_en_forced.json",
            gemma_audio_alignment_top_k_heads=8,
            gemma_audio_alignment_filter_width=7,
            gemma_audio_alignment_max_new_tokens=256,
        )
    )

    assert runtime_config.alignment_backend_name == "gemma_onepass_qk_fast"
    assert runtime_config.source_lang == "English"
    assert runtime_config.target_lang == "German"
    assert runtime_config.gemma_audio_align_probe_mode == "qk_fast"
    assert runtime_config.translation_alignatt_probe_mode == "qk_fast"
    assert runtime_config.translation_alignatt_min_source_mass == 0.2
