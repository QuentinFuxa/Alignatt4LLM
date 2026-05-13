from __future__ import annotations

from types import SimpleNamespace

import pytest

from cascade.mt.base import AlignAttDecoderPolicy
from cascade.mt.gemma_vllm_backend import GemmaVLLMMTBackend
from cascade.runtime import CascadeRuntimeConfig
from cascade.simulstream_processor import CascadeAlignAttProcessor
from cascade.submission import get_submission_preset


class TokenListTokenizer:
    def __init__(self, tokens: list[str]):
        self.tokens = list(tokens)

    def convert_ids_to_tokens(self, ids):
        return [self.tokens[int(i)] for i in ids]


def _runtime_config(**overrides):
    defaults = {
        "gemma_max_model_len": 1024,
        "max_new_tokens": 160,
        "partial_max_new_tokens": 16,
        "mt_vllm_enforce_eager": False,
        "mt_vllm_enable_prefix_caching": False,
        "mt_vllm_cudagraph_mode": "full",
        "mt_vllm_gpu_memory_utilization": 0.5,
        "mt_vllm_enable_speculative_decoding": False,
        "mt_vllm_speculative_assistant_model": None,
        "mt_vllm_num_speculative_tokens": 4,
        "repetition_penalty": 1.05,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_cut_last_target_stability_units_handles_whitespace_words():
    policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["▁Das", "▁ist", "▁sehr", "▁gut", "."]),
        runtime_config=SimpleNamespace(),
    )

    assert policy.cut_last_target_stability_units([0, 1, 2, 3, 4], cutoff_units=0) == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert policy.cut_last_target_stability_units([0, 1, 2, 3, 4], cutoff_units=1) == [
        0,
        1,
        2,
    ]
    assert policy.cut_last_target_stability_units([0, 1, 2, 3, 4], cutoff_units=3) == [
        0,
    ]


def test_cut_last_target_stability_units_handles_cjk_no_space_units():
    policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["你", "好", "世", "界"]),
        runtime_config=SimpleNamespace(),
    )

    assert policy.cut_last_target_stability_units([0, 1, 2, 3], cutoff_units=2) == [
        0,
        1,
    ]
    assert policy.cut_last_target_stability_units([0, 1, 2, 3], cutoff_units=4) == []


def test_alignatt_soft_frontier_ignores_tiny_future_mass():
    policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["▁gut"]),
        runtime_config=SimpleNamespace(
            translation_alignatt_border_margin=0,
            translation_alignatt_frontier_min_inaccessible_mass=0.05,
        ),
    )

    assert policy.should_stop_in_loop(
        current_source_local_position=5,
        accessible_source_token_count=5,
        source_inaccessible_mass=0.01,
    ) == (None, 5)
    assert policy.should_stop_in_loop(
        current_source_local_position=5,
        accessible_source_token_count=5,
        source_inaccessible_mass=0.08,
    ) == ("source_frontier", 5)


def test_alignatt_provenance_confidence_gates():
    policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["▁gut"]),
        runtime_config=SimpleNamespace(
            translation_alignatt_max_inaccessible_source_mass=0.20,
            translation_alignatt_min_accessible_inaccessible_margin=0.10,
        ),
    )

    assert (
        policy.should_stop_for_provenance_mass(
            source_accessible_mass=0.35,
            source_inaccessible_mass=0.25,
        )
        == "provenance_inaccessible_high"
    )
    assert (
        policy.should_stop_for_provenance_mass(
            source_accessible_mass=0.20,
            source_inaccessible_mass=0.15,
        )
        == "provenance_margin_weak"
    )
    assert (
        policy.should_stop_for_provenance_mass(
            source_accessible_mass=0.35,
            source_inaccessible_mass=0.15,
        )
        is None
    )


def test_gemma_mt_llm_kwargs_do_not_include_speculative_config_by_default():
    backend = GemmaVLLMMTBackend(
        model_name="/models/gemma-4-E4B-it",
        runtime_config=_runtime_config(),
    )

    kwargs = backend.build_llm_init_kwargs()

    assert "speculative_config" not in kwargs
    assert kwargs["worker_cls"] == "cascade.mt.gemma_vllm_worker.GemmaVLLMMTWorker"


def test_gemma_mt_llm_kwargs_include_explicit_speculative_config():
    backend = GemmaVLLMMTBackend(
        model_name="/models/gemma-4-E4B-it",
        runtime_config=_runtime_config(
            mt_vllm_enable_speculative_decoding=True,
            mt_vllm_speculative_assistant_model="/models/gemma-4-E4B-it-assistant",
            mt_vllm_num_speculative_tokens=4,
        ),
    )

    kwargs = backend.build_llm_init_kwargs()

    assert kwargs["speculative_config"] == {
        "model": "/models/gemma-4-E4B-it-assistant",
        "num_speculative_tokens": 4,
    }


def test_gemma_mt_speculative_config_rejects_nonpositive_token_count():
    backend = GemmaVLLMMTBackend(
        model_name="/models/gemma-4-E4B-it",
        runtime_config=_runtime_config(
            mt_vllm_enable_speculative_decoding=True,
            mt_vllm_speculative_assistant_model="/models/gemma-4-E4B-it-assistant",
            mt_vllm_num_speculative_tokens=0,
        ),
    )

    with pytest.raises(ValueError, match="mt_vllm_num_speculative_tokens"):
        backend.build_speculative_config()


def test_runtime_and_processor_fingerprints_include_speculative_engine_knobs():
    base = CascadeRuntimeConfig()
    speculative = CascadeRuntimeConfig(
        mt_vllm_enable_speculative_decoding=True,
        mt_vllm_speculative_assistant_model="/models/gemma-4-E4B-it-assistant",
        mt_vllm_num_speculative_tokens=4,
    )

    assert base.mt_backend_fingerprint() != speculative.mt_backend_fingerprint()

    processor_config = SimpleNamespace(
        source_lang_code="en",
        target_lang_code="de",
        mt_vllm_enable_speculative_decoding=True,
        mt_vllm_speculative_assistant_model="/models/gemma-4-E4B-it-assistant",
        mt_vllm_num_speculative_tokens=2,
    )
    resolved = CascadeAlignAttProcessor._build_runtime_config(processor_config)

    assert resolved.mt_vllm_enable_speculative_decoding is True
    assert resolved.mt_vllm_speculative_assistant_model == "/models/gemma-4-E4B-it-assistant"
    assert resolved.mt_vllm_num_speculative_tokens == 2


def test_runtime_validates_alignatt_mass_gates():
    with pytest.raises(ValueError, match="translation_alignatt_max_inaccessible_source_mass"):
        CascadeRuntimeConfig(translation_alignatt_max_inaccessible_source_mass=1.1)
    with pytest.raises(
        ValueError,
        match="translation_alignatt_min_accessible_inaccessible_margin",
    ):
        CascadeRuntimeConfig(translation_alignatt_min_accessible_inaccessible_margin=-1.1)


def test_frozen_main_low_latency_alignatt_policy_is_the_runtime_default():
    runtime = CascadeRuntimeConfig()
    preset_config = get_submission_preset("main_low_latency").build_speech_processor_config(
        source_lang_code="en",
        target_lang_code="de",
    )

    expected = {
        "translation_alignatt_top_k_heads": 4,
        "translation_alignatt_border_margin": 1,
        "translation_alignatt_min_source_mass": 0.003,
        "translation_alignatt_frontier_min_inaccessible_mass": 0.03,
        "translation_alignatt_max_inaccessible_source_mass": 0.15,
    }
    for name, value in expected.items():
        assert getattr(runtime, name) == value
        assert getattr(preset_config, name) == value
