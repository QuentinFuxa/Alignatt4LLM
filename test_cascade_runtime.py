from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_alignment_backend_registry_shape():
    from cascade_runtime import (
        CascadeRuntimeConfig,
        STABLE_ALIGNMENT_BACKEND_NAMES,
        VALID_ALIGNMENT_BACKEND_NAMES,
    )

    # Three frontends are currently supported; the experimental vLLM ASR path
    # must stay opt-in and NOT appear in the stable comparison set.
    assert VALID_ALIGNMENT_BACKEND_NAMES == (
        "qwen_forced",
        "gemma_onepass_qk_fast",
        "gemma_vllm_qk_fast",
    )
    assert STABLE_ALIGNMENT_BACKEND_NAMES == ("qwen_forced", "gemma_onepass_qk_fast")
    assert "gemma_vllm_qk_fast" not in STABLE_ALIGNMENT_BACKEND_NAMES

    for name in VALID_ALIGNMENT_BACKEND_NAMES:
        assert CascadeRuntimeConfig(alignment_backend_name=name).alignment_backend_name == name

    with pytest.raises(ValueError):
        CascadeRuntimeConfig(alignment_backend_name="hybrid_qwen_asr_gemma_aligner")


def test_mt_backend_registry_shape_and_defaults():
    # PLAN.md Phase 0: MT backend selection is a separate runtime axis.
    # Default must remain the stable Transformers MT path.
    from cascade_runtime import (
        CascadeRuntimeConfig,
        STABLE_MT_BACKEND_NAMES,
        VALID_MT_BACKEND_NAMES,
    )

    assert VALID_MT_BACKEND_NAMES == (
        "gemma_transformers_alignatt",
        "gemma_vllm_alignatt",
    )
    assert STABLE_MT_BACKEND_NAMES == ("gemma_transformers_alignatt",)
    assert "gemma_vllm_alignatt" not in STABLE_MT_BACKEND_NAMES

    default = CascadeRuntimeConfig()
    assert default.mt_backend_name == "gemma_transformers_alignatt"

    for name in VALID_MT_BACKEND_NAMES:
        assert CascadeRuntimeConfig(mt_backend_name=name).mt_backend_name == name


def test_mt_backend_name_rejects_unknown_values():
    from cascade_runtime import CascadeRuntimeConfig

    with pytest.raises(ValueError):
        CascadeRuntimeConfig(mt_backend_name="gemma_hybrid_alignatt")


def test_mt_vllm_backend_keeps_prefix_caching_off_by_default():
    # PLAN.md: "start with prefix caching disabled" on the MT vLLM backend.
    from cascade_runtime import CascadeRuntimeConfig

    config = CascadeRuntimeConfig(mt_backend_name="gemma_vllm_alignatt")
    assert config.mt_vllm_enable_prefix_caching is False


def test_qwen_forced_plus_vllm_mt_is_a_valid_combination():
    # PLAN.md Phase 5 target: qwen_forced ASR + gemma_vllm_alignatt MT.
    # The two axes are independent so this combination must be accepted, while
    # still not the default.
    from cascade_runtime import (
        CascadeRuntimeConfig,
        STABLE_ALIGNMENT_BACKEND_NAMES,
        STABLE_MT_BACKEND_NAMES,
    )

    config = CascadeRuntimeConfig(
        alignment_backend_name="qwen_forced",
        mt_backend_name="gemma_vllm_alignatt",
    )
    assert config.alignment_backend_name == "qwen_forced"
    assert config.mt_backend_name == "gemma_vllm_alignatt"
    # Defense-in-depth: the experimental MT backend must not silently become
    # the default.
    assert config.mt_backend_name not in STABLE_MT_BACKEND_NAMES
    # And the paired ASR backend IS the stable default.
    assert config.alignment_backend_name in STABLE_ALIGNMENT_BACKEND_NAMES


def test_mt_vllm_observer_max_decode_covers_all_runtime_caps():
    # The MT vLLM observer sizes decode_q_buffer from the config's new_tokens
    # caps at load time. If any per-call cap exceeds the observer size at
    # translate time, the backend raises (verified by the runtime check).
    # This test pins the invariant used to derive that size.
    from cascade_runtime import CascadeRuntimeConfig

    config = CascadeRuntimeConfig(mt_backend_name="gemma_vllm_alignatt")
    expected_observer_cap = max(
        int(config.max_new_tokens),
        int(config.partial_max_new_tokens),
        int(config.partial_followup_max_new_tokens),
    )
    assert expected_observer_cap == config.max_new_tokens
    assert expected_observer_cap >= config.partial_max_new_tokens
    assert expected_observer_cap >= config.partial_followup_max_new_tokens


def test_build_mt_backend_dispatches_on_name():
    # Avoids loading any model: dispatch is inspected via the class name of the
    # returned backend instance (no .load() called).
    from cascade_mt_backend import TransformersAlignAttGemmaMTBackend, build_mt_backend
    from cascade_runtime import CascadeRuntimeConfig

    tf_config = CascadeRuntimeConfig(mt_backend_name="gemma_transformers_alignatt")
    tf_backend = build_mt_backend(model_name="stub", runtime_config=tf_config)
    assert isinstance(tf_backend, TransformersAlignAttGemmaMTBackend)

    vllm_config = CascadeRuntimeConfig(mt_backend_name="gemma_vllm_alignatt")
    vllm_backend = build_mt_backend(model_name="stub", runtime_config=vllm_config)
    assert type(vllm_backend).__name__ == "VLLMAlignAttGemmaMTBackend"


def test_build_mt_backend_rejects_unknown_name():
    from cascade_mt_backend import build_mt_backend

    with pytest.raises(ValueError):
        build_mt_backend(
            model_name="stub",
            runtime_config=SimpleNamespace(mt_backend_name="gemma_unknown"),
        )


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
            mt_backend_name="gemma_transformers_alignatt",
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
    assert runtime_config.mt_backend_name == "gemma_transformers_alignatt"
    assert runtime_config.source_lang == "English"
    assert runtime_config.target_lang == "German"
    assert runtime_config.gemma_audio_align_probe_mode == "qk_fast"
    assert runtime_config.translation_alignatt_probe_mode == "qk_fast"
    assert runtime_config.translation_alignatt_min_source_mass == 0.2
