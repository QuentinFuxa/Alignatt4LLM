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


def test_language_code_to_name_includes_czech():
    # cs->en is a first-class direction. Historical bug: LANGUAGE_CODE_TO_NAME
    # was built from an early map that didn't include Czech, so cs silently
    # passed through as the raw code and broke downstream consumers that
    # expected an English-name label.
    from cascade_runtime import LANGUAGE_CODE_TO_NAME, LANGUAGE_NAME_TO_CODE
    assert LANGUAGE_CODE_TO_NAME["cs"] == "Czech"
    assert LANGUAGE_NAME_TO_CODE["Czech"] == "cs"
    # Reverse map must be derived from the one source of truth.
    for name, code in LANGUAGE_NAME_TO_CODE.items():
        assert LANGUAGE_CODE_TO_NAME[code] == name


def test_apply_overrides_recomputes_heads_path_on_source_or_target_change():
    # Historical bug: heads-path recompute keyed only off target_lang, so a
    # cs->en switch kept the English-source heads file.
    from cascade_runtime import CascadeRuntimeConfig

    config = CascadeRuntimeConfig(source_lang="English", target_lang="German")
    original = config.translation_alignatt_heads_path
    assert "en-de" in original

    config.apply_overrides(target_lang="Italian")
    assert "en-it" in config.translation_alignatt_heads_path

    config.apply_overrides(source_lang="Czech", target_lang="English")
    assert "cs-en" in config.translation_alignatt_heads_path

    config.apply_overrides(source_lang="English")
    assert "en-en" in config.translation_alignatt_heads_path

    # Explicit override must still win.
    config.apply_overrides(
        source_lang="German",
        translation_alignatt_heads_path="assets/custom.json",
    )
    assert config.translation_alignatt_heads_path == "assets/custom.json"


def test_runtime_config_overrides_context_manager_refreshes_on_source_lang():
    # Mirrors the apply_overrides fix for the temporary-overrides surface.
    from cascade_runtime import CascadeRuntimeConfig, temporary_runtime_config

    config = CascadeRuntimeConfig(source_lang="English", target_lang="German")
    baseline = config.translation_alignatt_heads_path

    with temporary_runtime_config(config, source_lang="Czech", target_lang="English"):
        assert "cs-en" in config.translation_alignatt_heads_path
    # Context must restore the original heads path.
    assert config.translation_alignatt_heads_path == baseline

    with temporary_runtime_config(config, target_lang="Italian"):
        assert "en-it" in config.translation_alignatt_heads_path
    assert config.translation_alignatt_heads_path == baseline

    with temporary_runtime_config(config, source_lang="German"):
        assert "de-de" in config.translation_alignatt_heads_path
    assert config.translation_alignatt_heads_path == baseline


def test_backend_fingerprints_flip_on_engine_knobs_but_not_policy_knobs():
    # The bundle reuse check must discriminate on engine-construction config
    # (memory budget, prefix caching, cudagraph mode, max_model_len,
    # prompt-KV reuse, ...) but not on live policy (commit mode, frontier
    # margin, heads path, rewind thresholds).
    from cascade_runtime import CascadeRuntimeConfig

    base = CascadeRuntimeConfig(
        alignment_backend_name="qwen_forced",
        mt_backend_name="gemma_vllm_alignatt",
    )
    base_asr_fp = base.alignment_backend_fingerprint()
    base_mt_fp = base.mt_backend_fingerprint()

    # Engine knobs must flip fingerprints:
    assert (
        CascadeRuntimeConfig(
            alignment_backend_name="qwen_forced",
            mt_backend_name="gemma_vllm_alignatt",
            asr_gpu_memory_utilization=0.3,
        ).alignment_backend_fingerprint()
        != base_asr_fp
    )
    assert (
        CascadeRuntimeConfig(
            alignment_backend_name="qwen_forced",
            mt_backend_name="gemma_vllm_alignatt",
            mt_vllm_gpu_memory_utilization=0.6,
        ).mt_backend_fingerprint()
        != base_mt_fp
    )
    assert (
        CascadeRuntimeConfig(
            alignment_backend_name="qwen_forced",
            mt_backend_name="gemma_vllm_alignatt",
            mt_vllm_enable_prefix_caching=True,
        ).mt_backend_fingerprint()
        != base_mt_fp
    )
    assert (
        CascadeRuntimeConfig(
            alignment_backend_name="qwen_forced",
            mt_backend_name="gemma_vllm_alignatt",
            gemma_max_model_len=2048,
        ).mt_backend_fingerprint()
        != base_mt_fp
    )

    # Policy / session-level knobs must NOT flip fingerprints: changing them
    # should keep the hot backend.
    for policy_override in [
        dict(asr_commit_mode="alignatt_frontier"),
        dict(asr_alignatt_frontier_margin_ms=250.0),
        dict(translation_alignatt_rewind_threshold=4),
        dict(translation_alignatt_min_source_mass=0.3),
        dict(translation_alignatt_inaccessible_ms=500.0),
        dict(translation_alignatt_filter_width=5),
        dict(translation_alignatt_heads_path="assets/other.json"),
        dict(source_lang="Czech", target_lang="English"),
    ]:
        cfg = CascadeRuntimeConfig(
            alignment_backend_name="qwen_forced",
            mt_backend_name="gemma_vllm_alignatt",
            **policy_override,
        )
        assert cfg.alignment_backend_fingerprint() == base_asr_fp, policy_override
        assert cfg.mt_backend_fingerprint() == base_mt_fp, policy_override

    # Changing the backend name itself obviously flips the fingerprint.
    alt = CascadeRuntimeConfig(
        alignment_backend_name="qwen_forced",
        mt_backend_name="gemma_transformers_alignatt",
    )
    assert alt.mt_backend_fingerprint() != base_mt_fp


def test_translation_source_frontier_mode_validated():
    # New runtime knob: scalar vs discrete source_frontier gate.
    # Config-only check — unknown values raise.
    import pytest
    from cascade_runtime import CascadeRuntimeConfig

    default = CascadeRuntimeConfig()
    assert default.translation_source_frontier_mode == "discrete"
    assert default.translation_source_frontier_scalar_threshold == 0.015

    CascadeRuntimeConfig(translation_source_frontier_mode="scalar")
    CascadeRuntimeConfig(
        translation_source_frontier_mode="scalar",
        translation_source_frontier_scalar_threshold=0.02,
    )

    with pytest.raises(ValueError, match="translation_source_frontier_mode"):
        CascadeRuntimeConfig(translation_source_frontier_mode="continuous")


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
