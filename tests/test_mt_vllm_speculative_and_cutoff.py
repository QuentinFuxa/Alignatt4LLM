from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from alignatt4llm.mt.base import AlignAttDecoderPolicy
from alignatt4llm.mt.base import PromptSourceMap, PromptSourceUnitSpan
from alignatt4llm.mt.base import TokenProvenanceBreakdown
from alignatt4llm.mt.gemma_vllm_backend import GemmaVLLMMTBackend
from alignatt4llm.runtime import (
    CascadeRuntimeConfig,
    CascadeState,
    PartialTranslationState,
    TranslationUnitManager,
    source_bounded_prefill_generated_token_ids,
)
from alignatt4llm.source_frontier import build_source_accessibility_frontier
from alignatt4llm.simulstream_processor import CascadeAlignAttProcessor
from alignatt4llm.presets import get_runtime_preset


class TokenListTokenizer:
    def __init__(self, tokens: list[str]):
        self.tokens = list(tokens)

    def convert_ids_to_tokens(self, ids):
        return [self.tokens[int(i)] for i in ids]

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(self.tokens[int(i)] for i in ids)


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


def _prompt_source_map(
    *,
    accessible_source_token_count: int = 8,
    is_final: bool = False,
) -> PromptSourceMap:
    return PromptSourceMap(
        source_text="source",
        source_token_positions=tuple(range(accessible_source_token_count)),
        source_unit_spans=(),
        accessible_source_token_count=accessible_source_token_count,
        accessible_unit_count=accessible_source_token_count,
        total_unit_count=accessible_source_token_count,
        current_source_ms=0.0,
        inaccessible_ms=0.0,
        is_final=is_final,
    )


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


def test_keep_first_target_stability_units_caps_cjk_units():
    policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["你", "好", "世", "界"]),
        runtime_config=SimpleNamespace(target_lang="Simplified Chinese"),
    )

    assert policy.keep_first_target_stability_units([0, 1, 2, 3], max_units=2) == [
        0,
        1,
    ]
    assert policy.keep_first_target_stability_units([0, 1, 2, 3], max_units=8) == [
        0,
        1,
        2,
        3,
    ]
    assert policy.keep_first_target_stability_units([0, 1, 2, 3], max_units=0) == []


def test_unit_consensus_uses_online_normalized_per_head_positions():
    policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["你"]),
        runtime_config=SimpleNamespace(
            target_lang="Simplified Chinese",
            translation_alignatt_acceptance_variant="unit_consensus",
            translation_alignatt_border_margin=0,
            translation_alignatt_unit_consensus_min_head_ratio=0.75,
            translation_alignatt_online_normalization="zscore",
        ),
    )
    # Raw rows would put every head beyond the frontier. The supplied per-head
    # positions are the online-normalized AlignAtt positions, where 3/4 heads
    # remain inside the accessible frontier.
    raw_rows = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    decision = policy.accept_complete_target_units(
        generated_ids=[0],
        aligned_source_local_positions=[0],
        source_attention_rows=[raw_rows],
        provenance_mass=None,
        source_map=_source_map_with_accessible_units(2),
        finish_reason="length",
        per_head_aligned_source_local_positions=[[0, 1, 0, 3]],
    )

    assert decision.accepted_candidate_ids == [0]
    assert decision.metadata["alignatt_unit_policy_accepted_unit_count"] == 1
    assert decision.metadata["alignatt_unit_policy_last_consensus_ratio"] == 0.75


def test_unit_consensus_falls_back_to_raw_rows_without_online_positions():
    policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["你"]),
        runtime_config=SimpleNamespace(
            target_lang="Simplified Chinese",
            translation_alignatt_acceptance_variant="unit_consensus",
            translation_alignatt_border_margin=0,
            translation_alignatt_unit_consensus_min_head_ratio=0.75,
        ),
    )
    raw_rows = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    decision = policy.accept_complete_target_units(
        generated_ids=[0],
        aligned_source_local_positions=[0],
        source_attention_rows=[raw_rows],
        provenance_mass=None,
        source_map=_source_map_with_accessible_units(2),
        finish_reason="length",
    )

    assert decision.accepted_candidate_ids == []
    assert decision.unsafe_reason == "head_consensus_frontier"
    assert decision.metadata["alignatt_unit_policy_last_consensus_ratio"] == 0.0


def test_unit_source_bearing_requires_source_bearing_token():
    policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["介", "绍"]),
        runtime_config=SimpleNamespace(
            target_lang="Simplified Chinese",
            translation_alignatt_acceptance_variant="unit_mass_source_bearing",
            translation_alignatt_source_bearing_min_source_mass=0.05,
            translation_alignatt_source_bearing_hard_inaccessible_cap=0.75,
        ),
    )

    decision = policy.accept_complete_target_units(
        generated_ids=[0, 1],
        aligned_source_local_positions=[0, 1],
        source_attention_rows=None,
        provenance_mass=[
            TokenProvenanceBreakdown(
                source_accessible=0.001,
                source_inaccessible=0.0,
                non_source_prompt=0.99,
                suffix=0.009,
            ),
            TokenProvenanceBreakdown(
                source_accessible=0.001,
                source_inaccessible=0.0,
                non_source_prompt=0.99,
                suffix=0.009,
            ),
        ],
        source_map=_source_map_with_accessible_units(2),
        finish_reason="length",
    )

    assert decision.accepted_candidate_ids == []
    assert decision.unsafe_reason == "source_bearing_missing"
    assert decision.metadata["alignatt_source_bearing_last_unit_token_count"] == 0


def test_unit_source_bearing_accepts_source_bearing_unit():
    policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["介"]),
        runtime_config=SimpleNamespace(
            target_lang="Simplified Chinese",
            translation_alignatt_acceptance_variant="unit_mass_source_bearing",
            translation_alignatt_source_bearing_min_source_mass=0.05,
            translation_alignatt_source_bearing_hard_inaccessible_cap=0.75,
            translation_alignatt_max_inaccessible_source_mass=1.0,
            translation_alignatt_min_accessible_inaccessible_margin=-1.0,
        ),
    )

    decision = policy.accept_complete_target_units(
        generated_ids=[0],
        aligned_source_local_positions=[0],
        source_attention_rows=None,
        provenance_mass=[
            TokenProvenanceBreakdown(
                source_accessible=0.06,
                source_inaccessible=0.01,
                non_source_prompt=0.9,
                suffix=0.03,
            ),
        ],
        source_map=_source_map_with_accessible_units(2),
        finish_reason="length",
    )

    assert decision.accepted_candidate_ids == [0]
    assert decision.unsafe_reason is None
    assert decision.metadata["alignatt_source_bearing_last_unit_token_count"] == 1


def test_provenance_non_source_prompt_gate_is_disabled_by_default():
    policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["介"]),
        runtime_config=SimpleNamespace(
            translation_alignatt_max_inaccessible_source_mass=1.0,
            translation_alignatt_min_accessible_inaccessible_margin=-1.0,
        ),
    )

    assert (
        policy.should_stop_for_provenance_mass(
            source_accessible_mass=0.06,
            source_inaccessible_mass=0.01,
            non_source_prompt_mass=0.90,
        )
        is None
    )


def test_provenance_non_source_prompt_gate_blocks_when_enabled():
    policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["介"]),
        runtime_config=SimpleNamespace(
            translation_alignatt_max_inaccessible_source_mass=1.0,
            translation_alignatt_max_non_source_prompt_mass=0.80,
            translation_alignatt_min_accessible_inaccessible_margin=-1.0,
        ),
    )

    assert policy.should_stop_for_provenance_mass(
        source_accessible_mass=0.06,
        source_inaccessible_mass=0.01,
        non_source_prompt_mass=0.90,
    ) == "provenance_non_source_high"


def test_unit_source_bearing_reports_non_source_prompt_gate_when_enabled():
    policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["介"]),
        runtime_config=SimpleNamespace(
            target_lang="Simplified Chinese",
            translation_alignatt_acceptance_variant="unit_mass_source_bearing",
            translation_alignatt_source_bearing_min_source_mass=0.05,
            translation_alignatt_source_bearing_hard_inaccessible_cap=1.0,
            translation_alignatt_max_inaccessible_source_mass=1.0,
            translation_alignatt_max_non_source_prompt_mass=0.80,
            translation_alignatt_min_accessible_inaccessible_margin=-1.0,
        ),
    )

    decision = policy.accept_complete_target_units(
        generated_ids=[0],
        aligned_source_local_positions=[0],
        source_attention_rows=None,
        provenance_mass=[
            TokenProvenanceBreakdown(
                source_accessible=0.06,
                source_inaccessible=0.01,
                non_source_prompt=0.90,
                suffix=0.03,
            ),
        ],
        source_map=_source_map_with_accessible_units(2),
        finish_reason="length",
    )

    assert decision.accepted_candidate_ids == []
    assert decision.unsafe_reason == "provenance_non_source_high"


def test_unit_mass_remains_source_mass_permissive_without_argmax_frontier_gate():
    policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["介"]),
        runtime_config=SimpleNamespace(
            target_lang="Simplified Chinese",
            translation_alignatt_acceptance_variant="unit_mass",
            translation_alignatt_min_source_mass=0.05,
            translation_alignatt_max_inaccessible_source_mass=1.0,
            translation_alignatt_min_accessible_inaccessible_margin=-1.0,
            translation_alignatt_border_margin=0,
            translation_alignatt_frontier_min_inaccessible_mass=0.0,
        ),
    )

    decision = policy.accept_complete_target_units(
        generated_ids=[0],
        aligned_source_local_positions=[3],
        source_attention_rows=None,
        provenance_mass=[
            TokenProvenanceBreakdown(
                source_accessible=0.06,
                source_inaccessible=0.01,
                non_source_prompt=0.9,
                suffix=0.03,
            ),
        ],
        source_map=_source_map_with_accessible_units(1),
        finish_reason="length",
    )

    assert decision.accepted_candidate_ids == [0]
    assert decision.unsafe_reason is None


def test_unit_source_bearing_blocks_future_frontier_token():
    policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["介"]),
        runtime_config=SimpleNamespace(
            target_lang="Simplified Chinese",
            translation_alignatt_acceptance_variant="unit_mass_source_bearing",
            translation_alignatt_source_bearing_min_source_mass=0.05,
            translation_alignatt_source_bearing_hard_inaccessible_cap=0.75,
            translation_alignatt_border_margin=0,
            translation_alignatt_frontier_min_inaccessible_mass=0.0,
        ),
    )

    decision = policy.accept_complete_target_units(
        generated_ids=[0],
        aligned_source_local_positions=[3],
        source_attention_rows=None,
        provenance_mass=[
            TokenProvenanceBreakdown(
                source_accessible=0.06,
                source_inaccessible=0.01,
                non_source_prompt=0.9,
                suffix=0.03,
            ),
        ],
        source_map=_source_map_with_accessible_units(1),
        finish_reason="length",
    )

    assert decision.accepted_candidate_ids == []
    assert decision.unsafe_reason == "source_frontier"
    assert decision.blocked_source_local_position == 3


def test_unit_source_bearing_keeps_soft_frontier_permissiveness():
    policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["介"]),
        runtime_config=SimpleNamespace(
            target_lang="Simplified Chinese",
            translation_alignatt_acceptance_variant="unit_mass_source_bearing",
            translation_alignatt_source_bearing_min_source_mass=0.05,
            translation_alignatt_source_bearing_hard_inaccessible_cap=0.75,
            translation_alignatt_border_margin=0,
            translation_alignatt_frontier_min_inaccessible_mass=0.03,
            translation_alignatt_max_inaccessible_source_mass=1.0,
            translation_alignatt_min_accessible_inaccessible_margin=-1.0,
        ),
    )

    decision = policy.accept_complete_target_units(
        generated_ids=[0],
        aligned_source_local_positions=[3],
        source_attention_rows=None,
        provenance_mass=[
            TokenProvenanceBreakdown(
                source_accessible=0.06,
                source_inaccessible=0.01,
                non_source_prompt=0.9,
                suffix=0.03,
            ),
        ],
        source_map=_source_map_with_accessible_units(1),
        finish_reason="length",
    )

    assert decision.accepted_candidate_ids == [0]
    assert decision.unsafe_reason is None
    assert decision.metadata["alignatt_unit_policy_last_safe_source_position"] == 3


def _source_map_with_accessible_units(accessible_units: int) -> PromptSourceMap:
    source_unit_spans = tuple(
        PromptSourceUnitSpan(
            unit_index=index,
            text=f"u{index}",
            prompt_token_positions=(index,),
            is_accessible=index < accessible_units,
            start_ms=None,
            end_ms=None,
        )
        for index in range(6)
    )
    return PromptSourceMap(
        source_text="u0 u1 u2 u3 u4 u5",
        source_token_positions=tuple(range(6)),
        source_unit_spans=source_unit_spans,
        accessible_source_token_count=accessible_units,
        accessible_unit_count=accessible_units,
        total_unit_count=6,
        current_source_ms=640.0,
        inaccessible_ms=0.0,
        is_final=False,
    )


def test_source_context_target_unit_cap_keeps_small_prefix_before_min_source_units():
    policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["你", "好", "世", "界"]),
        runtime_config=SimpleNamespace(
            target_lang="Simplified Chinese",
            translation_alignatt_min_accessible_source_units=6,
            translation_alignatt_min_accessible_source_units_mode="target_unit_cap",
        ),
    )

    acceptance = policy.finalize_partial(
        accepted_candidate_ids=[0, 1, 2, 3],
        aligned_source_local_positions=[0, 1, 2, 3],
        source_map=_source_map_with_accessible_units(2),
        unsafe_reason=None,
        unsafe_target_token_index=None,
        unsafe_token_id=None,
        blocked_source_local_position=None,
        blocked_source_unit_index=None,
        stop_reason="length",
        probe_backend="test",
    )

    assert acceptance.accepted_generated_ids == [0, 1]
    assert acceptance.alignatt_metadata["alignatt_source_context_under_min"] is True
    assert acceptance.alignatt_metadata["alignatt_source_context_blocked"] is False
    assert acceptance.alignatt_metadata["alignatt_source_context_cap_applied"] is True
    assert acceptance.alignatt_metadata["alignatt_source_context_cap_target_units"] == 2


def test_source_context_block_mode_preserves_hard_gate():
    policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["你", "好", "世", "界"]),
        runtime_config=SimpleNamespace(
            target_lang="Simplified Chinese",
            translation_alignatt_min_accessible_source_units=6,
            translation_alignatt_min_accessible_source_units_mode="block",
        ),
    )

    acceptance = policy.finalize_partial(
        accepted_candidate_ids=[0, 1, 2, 3],
        aligned_source_local_positions=[0, 1, 2, 3],
        source_map=_source_map_with_accessible_units(2),
        unsafe_reason=None,
        unsafe_target_token_index=None,
        unsafe_token_id=None,
        blocked_source_local_position=None,
        blocked_source_unit_index=None,
        stop_reason="length",
        probe_backend="test",
    )

    assert acceptance.accepted_generated_ids == []
    assert acceptance.alignatt_metadata["alignatt_source_context_under_min"] is True
    assert acceptance.alignatt_metadata["alignatt_source_context_blocked"] is True
    assert acceptance.alignatt_metadata["alignatt_source_context_cap_applied"] is False


def test_source_context_cap_composes_with_unrecovered_regression_trim():
    policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["甲", "乙", "丙", "丁", "戊", "己"]),
        runtime_config=SimpleNamespace(
            target_lang="Simplified Chinese",
            translation_alignatt_max_source_regression=0,
            translation_alignatt_source_regression_action="trim_unrecovered",
            translation_alignatt_min_accessible_source_units=6,
            translation_alignatt_min_accessible_source_units_mode="target_unit_cap",
        ),
    )

    acceptance = policy.finalize_partial(
        accepted_candidate_ids=[0, 1, 2, 3, 4, 5],
        aligned_source_local_positions=[0, 5, 8, 2, 9, 3],
        source_map=_source_map_with_accessible_units(4),
        unsafe_reason=None,
        unsafe_target_token_index=None,
        unsafe_token_id=None,
        blocked_source_local_position=None,
        blocked_source_unit_index=None,
        stop_reason="length",
        probe_backend="test",
    )

    assert acceptance.accepted_generated_ids == [0, 1, 2, 3]
    assert acceptance.alignatt_metadata["alignatt_source_regression_action"] == (
        "trim_unrecovered"
    )
    assert acceptance.alignatt_metadata["alignatt_source_regression_trimmed"] is True
    assert acceptance.alignatt_metadata["alignatt_source_regression_trimmed_token_count"] == 1
    assert acceptance.alignatt_metadata["alignatt_source_regression_trim_bypassed_count"] == 2
    assert acceptance.alignatt_metadata["alignatt_source_context_under_min"] is True
    assert acceptance.alignatt_metadata["alignatt_source_context_blocked"] is False
    assert acceptance.alignatt_metadata["alignatt_source_context_cap_applied"] is True
    assert acceptance.alignatt_metadata["alignatt_source_context_cap_target_units"] == 4


def test_source_frontier_unrecovered_trim_keeps_recovered_frontier_hits():
    policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["甲", "乙", "丙", "丁"]),
        runtime_config=SimpleNamespace(
            target_lang="Simplified Chinese",
            translation_alignatt_source_frontier_action="trim_unrecovered",
        ),
    )

    acceptance = policy.finalize_partial(
        accepted_candidate_ids=[0, 1, 2, 3],
        aligned_source_local_positions=[0, 5, 2, 5],
        source_map=_source_map_with_accessible_units(4),
        unsafe_reason=None,
        unsafe_target_token_index=None,
        unsafe_token_id=None,
        blocked_source_local_position=None,
        blocked_source_unit_index=None,
        stop_reason="length",
        probe_backend="test",
    )

    assert acceptance.accepted_generated_ids == [0, 1, 2]
    assert acceptance.alignatt_metadata["alignatt_source_frontier_action"] == (
        "trim_unrecovered"
    )
    assert acceptance.alignatt_metadata["alignatt_source_frontier_trimmed"] is True
    assert acceptance.alignatt_metadata["alignatt_source_frontier_trimmed_token_count"] == 1
    assert acceptance.alignatt_metadata["alignatt_source_frontier_trimmed_unit_count"] == 1
    assert acceptance.alignatt_metadata["alignatt_source_frontier_trim_reason"] == (
        "source_frontier"
    )
    assert acceptance.alignatt_metadata["alignatt_source_frontier_trim_bypassed_count"] == 2


def test_source_frontier_unrecovered_trim_accepts_within_unit_recovery():
    policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["▁hel", "lo", "▁world"]),
        runtime_config=SimpleNamespace(
            translation_alignatt_source_frontier_action="trim_unrecovered",
        ),
    )

    acceptance = policy.finalize_partial(
        accepted_candidate_ids=[0, 1],
        aligned_source_local_positions=[5, 2],
        source_map=_source_map_with_accessible_units(4),
        unsafe_reason=None,
        unsafe_target_token_index=None,
        unsafe_token_id=2,
        blocked_source_local_position=None,
        blocked_source_unit_index=None,
        stop_reason="length",
        probe_backend="test",
    )

    assert acceptance.accepted_generated_ids == [0, 1]
    assert acceptance.alignatt_metadata["alignatt_source_frontier_trimmed"] is False
    assert acceptance.alignatt_metadata["alignatt_source_frontier_trim_bypassed_count"] == 1


def test_runtime_rejects_unknown_source_frontier_action():
    with pytest.raises(ValueError, match="source_frontier_action"):
        CascadeRuntimeConfig(
            translation_alignatt_source_frontier_action="local_agreement_wait"
        )


def test_runtime_rejects_unknown_source_context_mode():
    with pytest.raises(ValueError, match="min_accessible_source_units_mode"):
        CascadeRuntimeConfig(
            translation_alignatt_min_accessible_source_units_mode="wait_forever"
        )


def test_runtime_rejects_unknown_source_regression_reference_mode():
    with pytest.raises(ValueError, match="source_regression_reference_mode"):
        CascadeRuntimeConfig(
            translation_alignatt_source_regression_reference_mode="latest_spike"
        )


def test_runtime_rejects_unknown_source_regression_activation_mode():
    with pytest.raises(ValueError, match="source_regression_activation_mode"):
        CascadeRuntimeConfig(
            translation_alignatt_source_regression_activation_mode="lateish"
        )


def test_runtime_rejects_unknown_source_regression_action():
    with pytest.raises(ValueError, match="source_regression_action"):
        CascadeRuntimeConfig(
            translation_alignatt_source_regression_action="local_agreementish"
        )


def test_runtime_rejects_nonpositive_source_regression_patience():
    with pytest.raises(ValueError, match="source_regression_patience_tokens"):
        CascadeRuntimeConfig(
            translation_alignatt_source_regression_patience_tokens=0
        )


def test_runtime_rejects_negative_source_regression_activation_slack():
    with pytest.raises(ValueError, match="source_regression_activation_slack_tokens"):
        CascadeRuntimeConfig(
            translation_alignatt_source_regression_activation_slack_tokens=-1
        )


def test_runtime_rejects_invalid_source_regression_future_mass_threshold():
    with pytest.raises(ValueError, match="source_regression_min_inaccessible_mass"):
        CascadeRuntimeConfig(
            translation_alignatt_source_regression_min_inaccessible_mass=1.5
        )


def test_runtime_rejects_invalid_non_source_prompt_mass_cap():
    with pytest.raises(ValueError, match="max_non_source_prompt_mass"):
        CascadeRuntimeConfig(translation_alignatt_max_non_source_prompt_mass=1.5)


def test_runtime_rejects_nonpositive_token_argmax_frontier_patience():
    with pytest.raises(ValueError, match="token_argmax_frontier_patience_tokens"):
        CascadeRuntimeConfig(
            translation_alignatt_token_argmax_frontier_patience_tokens=0
        )


def test_source_regression_patience_requires_consecutive_regressions():
    default_policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["▁a"]),
        runtime_config=SimpleNamespace(),
    )
    patient_policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["▁a"]),
        runtime_config=SimpleNamespace(
            translation_alignatt_source_regression_patience_tokens=2,
        ),
    )

    assert default_policy.should_stop_after_source_regression_patience(
        current_streak=0,
        source_regression_stop_reason="source_regression",
    ) == (True, 1)
    assert patient_policy.should_stop_after_source_regression_patience(
        current_streak=0,
        source_regression_stop_reason="source_regression",
    ) == (False, 1)
    assert patient_policy.should_stop_after_source_regression_patience(
        current_streak=1,
        source_regression_stop_reason="source_regression",
    ) == (True, 2)
    assert patient_policy.should_stop_after_source_regression_patience(
        current_streak=1,
        source_regression_stop_reason=None,
    ) == (False, 0)


def test_token_argmax_frontier_patience_requires_consecutive_hits():
    default_policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["▁a"]),
        runtime_config=SimpleNamespace(),
    )
    patient_policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["▁a"]),
        runtime_config=SimpleNamespace(
            translation_alignatt_token_argmax_frontier_patience_tokens=2,
        ),
    )

    assert default_policy.should_stop_after_token_argmax_frontier_patience(
        current_streak=0,
        token_argmax_stop_reason="token_argmax_source_frontier",
    ) == (True, 1)
    assert patient_policy.should_stop_after_token_argmax_frontier_patience(
        current_streak=0,
        token_argmax_stop_reason="token_argmax_source_frontier",
    ) == (False, 1)
    assert patient_policy.should_stop_after_token_argmax_frontier_patience(
        current_streak=1,
        token_argmax_stop_reason="token_argmax_source_frontier",
    ) == (True, 2)
    assert patient_policy.should_stop_after_token_argmax_frontier_patience(
        current_streak=1,
        token_argmax_stop_reason=None,
    ) == (False, 0)


def test_source_regression_can_activate_only_after_source_frontier_reached():
    always_policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["▁a"]),
        runtime_config=SimpleNamespace(
            translation_alignatt_max_source_regression=1,
            translation_alignatt_source_regression_activation_mode="always",
        ),
    )
    frontier_policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["▁a"]),
        runtime_config=SimpleNamespace(
            translation_alignatt_max_source_regression=1,
            translation_alignatt_source_regression_activation_mode="frontier_reached",
        ),
    )

    assert always_policy.should_stop_for_source_regression(
        current_source_local_position=2,
        max_accepted_source_local_position=5,
        accessible_source_token_count=10,
    ) == "source_regression"
    assert frontier_policy.should_stop_for_source_regression(
        current_source_local_position=2,
        max_accepted_source_local_position=5,
        accessible_source_token_count=10,
    ) is None
    assert frontier_policy.should_stop_for_source_regression(
        current_source_local_position=6,
        max_accepted_source_local_position=9,
        accessible_source_token_count=10,
    ) == "source_regression"


def test_source_regression_frontier_activation_allows_near_frontier_slack():
    strict_frontier_policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["▁a"]),
        runtime_config=SimpleNamespace(
            translation_alignatt_max_source_regression=1,
            translation_alignatt_source_regression_activation_mode="frontier_reached",
            translation_alignatt_source_regression_activation_slack_tokens=0,
        ),
    )
    slack_policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["▁a"]),
        runtime_config=SimpleNamespace(
            translation_alignatt_max_source_regression=1,
            translation_alignatt_source_regression_activation_mode="frontier_reached",
            translation_alignatt_source_regression_activation_slack_tokens=3,
        ),
    )

    assert strict_frontier_policy.should_stop_for_source_regression(
        current_source_local_position=3,
        max_accepted_source_local_position=6,
        accessible_source_token_count=10,
    ) is None
    assert slack_policy.should_stop_for_source_regression(
        current_source_local_position=3,
        max_accepted_source_local_position=6,
        accessible_source_token_count=10,
    ) == "source_regression"


def test_source_regression_can_require_future_bearing_mass():
    policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["▁a"]),
        runtime_config=SimpleNamespace(
            translation_alignatt_max_source_regression=1,
            translation_alignatt_source_regression_min_inaccessible_mass=0.03,
        ),
    )

    assert policy.should_stop_for_source_regression(
        current_source_local_position=3,
        max_accepted_source_local_position=6,
        source_accessible_mass=0.40,
        source_inaccessible_mass=0.0,
    ) is None
    assert policy.should_stop_for_source_regression(
        current_source_local_position=3,
        max_accepted_source_local_position=6,
        source_accessible_mass=0.40,
        source_inaccessible_mass=0.05,
    ) == "source_regression"


def test_source_lcp_stability_caps_partial_source_frontier():
    class FakeSession:
        def __init__(self):
            self.config = CascadeRuntimeConfig(
                translation_alignatt_source_lcp_stability=True,
                translation_alignatt_inaccessible_ms=0.0,
            )
            self.state = CascadeState(
                asr_hypotheses=[
                    "In everyday life",
                    "In everyday life. Hilma's office",
                ],
                partial_word_timestamps_ms=[
                    (0.0, 100.0),
                    (100.0, 200.0),
                    (200.0, 300.0),
                    (300.0, 400.0),
                    (400.0, 500.0),
                ],
            )

        def current_audio_seconds(self):
            return 2.0

    manager = TranslationUnitManager(FakeSession())
    normalized = manager.normalize_source_text(
        "In everyday life. Hilma's office",
        is_final=False,
    )
    frontier = manager.build_source_frontier(normalized, is_final=False)

    assert [unit.text for unit in frontier.units] == [
        "In",
        "everyday",
        "life",
        "Hilma's",
        "office",
    ]
    assert frontier.accessible_unit_count == 3


def test_source_lcp_append_slack_allows_bounded_append_only_source_units():
    class FakeSession:
        def __init__(self):
            self.config = CascadeRuntimeConfig(
                translation_alignatt_source_lcp_stability=True,
                translation_alignatt_source_lcp_append_slack_units=2,
                translation_alignatt_inaccessible_ms=0.0,
            )
            self.state = CascadeState(
                asr_hypotheses=[
                    "In everyday life",
                    "In everyday life humans usually",
                ],
                partial_word_timestamps_ms=[
                    (0.0, 100.0),
                    (100.0, 200.0),
                    (200.0, 300.0),
                    (300.0, 400.0),
                    (400.0, 500.0),
                ],
            )

        def current_audio_seconds(self):
            return 2.0

    manager = TranslationUnitManager(FakeSession())
    normalized = manager.normalize_source_text(
        "In everyday life humans usually",
        is_final=False,
    )
    frontier = manager.build_source_frontier(normalized, is_final=False)

    assert [unit.text for unit in frontier.units] == [
        "In",
        "everyday",
        "life",
        "humans",
        "usually",
    ]
    assert frontier.accessible_unit_count == 5


def test_source_lcp_append_slack_requires_source_lcp_stability():
    with pytest.raises(ValueError, match="source_lcp_append_slack_units"):
        CascadeRuntimeConfig(
            translation_alignatt_source_lcp_stability=False,
            translation_alignatt_source_lcp_append_slack_units=1,
        )


def test_source_lcp_append_slack_does_not_expand_rewritten_source_tail():
    class FakeSession:
        def __init__(self):
            self.config = CascadeRuntimeConfig(
                translation_alignatt_source_lcp_stability=True,
                translation_alignatt_source_lcp_append_slack_units=2,
                translation_alignatt_inaccessible_ms=0.0,
            )
            self.state = CascadeState(
                asr_hypotheses=[
                    "In everyday life",
                    "In everyday life. Hilma's office",
                ],
                partial_word_timestamps_ms=[
                    (0.0, 100.0),
                    (100.0, 200.0),
                    (200.0, 300.0),
                    (300.0, 400.0),
                    (400.0, 500.0),
                ],
            )

        def current_audio_seconds(self):
            return 2.0

    manager = TranslationUnitManager(FakeSession())
    normalized = manager.normalize_source_text(
        "In everyday life. Hilma's office",
        is_final=False,
    )
    frontier = manager.build_source_frontier(normalized, is_final=False)

    assert frontier.accessible_unit_count == 3


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


def test_accepted_prefix_source_mass_floor_checks_recent_units_individually():
    policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["介", "绍", "工"]),
        runtime_config=SimpleNamespace(
            target_lang="Simplified Chinese",
            translation_alignatt_min_accepted_accessible_source_mass=0.001,
            translation_alignatt_accepted_accessible_source_mass_recent_units=2,
        ),
    )

    trimmed, mean_mass, recent_mean_mass, recent_min_unit_mass, was_trimmed = (
        policy.trim_for_accepted_prefix_provenance(
            [0, 1, 2],
            provenance_mass=[
                TokenProvenanceBreakdown(
                    source_accessible=0.009,
                    source_inaccessible=0.0,
                    non_source_prompt=0.99,
                    suffix=0.001,
                ),
                TokenProvenanceBreakdown(
                    source_accessible=0.00001,
                    source_inaccessible=0.0,
                    non_source_prompt=0.999,
                    suffix=0.00099,
                ),
                TokenProvenanceBreakdown(
                    source_accessible=0.003,
                    source_inaccessible=0.0,
                    non_source_prompt=0.996,
                    suffix=0.001,
                ),
            ],
        )
    )

    assert trimmed == [0]
    assert was_trimmed is True
    assert mean_mass == pytest.approx(0.009)
    assert recent_mean_mass == pytest.approx(0.009)
    assert recent_min_unit_mass == pytest.approx(0.009)


def test_accepted_prefix_source_mass_floor_keeps_source_bearing_recent_units():
    policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["介", "绍", "工"]),
        runtime_config=SimpleNamespace(
            target_lang="Simplified Chinese",
            translation_alignatt_min_accepted_accessible_source_mass=0.001,
            translation_alignatt_accepted_accessible_source_mass_recent_units=2,
        ),
    )

    trimmed, mean_mass, recent_mean_mass, recent_min_unit_mass, was_trimmed = (
        policy.trim_for_accepted_prefix_provenance(
            [0, 1, 2],
            provenance_mass=[
                TokenProvenanceBreakdown(
                    source_accessible=0.009,
                    source_inaccessible=0.0,
                    non_source_prompt=0.99,
                    suffix=0.001,
                ),
                TokenProvenanceBreakdown(
                    source_accessible=0.002,
                    source_inaccessible=0.0,
                    non_source_prompt=0.997,
                    suffix=0.001,
                ),
                TokenProvenanceBreakdown(
                    source_accessible=0.003,
                    source_inaccessible=0.0,
                    non_source_prompt=0.996,
                    suffix=0.001,
                ),
            ],
        )
    )

    assert trimmed == [0, 1, 2]
    assert was_trimmed is False
    assert mean_mass == pytest.approx((0.009 + 0.002 + 0.003) / 3)
    assert recent_mean_mass == pytest.approx((0.002 + 0.003) / 2)
    assert recent_min_unit_mass == pytest.approx(0.002)


def test_source_regression_reference_can_use_recent_accepted_positions():
    global_policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["▁a"]),
        runtime_config=SimpleNamespace(
            translation_alignatt_source_regression_recent_tokens=0,
        ),
    )
    recent_policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["▁a"]),
        runtime_config=SimpleNamespace(
            translation_alignatt_source_regression_recent_tokens=3,
        ),
    )
    median_recent_policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["▁a"]),
        runtime_config=SimpleNamespace(
            translation_alignatt_source_regression_recent_tokens=3,
            translation_alignatt_source_regression_reference_mode="median_recent",
        ),
    )
    accepted_positions = [1, 12, 8, 7, 6]

    assert (
        global_policy.source_regression_reference_position(
            accepted_source_local_positions=accepted_positions,
            max_accepted_source_local_position=12,
        )
        == 12
    )
    assert (
        recent_policy.source_regression_reference_position(
            accepted_source_local_positions=accepted_positions,
            max_accepted_source_local_position=12,
        )
        == 8
    )
    spiky_positions = [1, 7, 8, 11]
    assert (
        recent_policy.source_regression_reference_position(
            accepted_source_local_positions=spiky_positions,
            max_accepted_source_local_position=11,
        )
        == 11
    )
    assert (
        median_recent_policy.source_regression_reference_position(
            accepted_source_local_positions=spiky_positions,
            max_accepted_source_local_position=11,
        )
        == 8
    )


def test_source_regression_target_unit_trim_allows_intra_unit_argmax_blips():
    policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["▁A", "▁B", "x", "▁C"]),
        runtime_config=SimpleNamespace(
            translation_alignatt_max_source_regression=0,
            translation_alignatt_source_regression_action="trim_target_unit",
        ),
    )

    trimmed, was_trimmed, *_ = policy.trim_for_source_regression_target_units(
        [0, 1, 2, 3],
        aligned_source_local_positions=[5, 4, 6, 7],
        provenance_mass=None,
        source_map=_prompt_source_map(),
    )

    assert trimmed == [0, 1, 2, 3]
    assert was_trimmed is False


def test_source_regression_target_unit_trim_cuts_regressive_suffix():
    policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["▁A", "▁B", "▁C"]),
        runtime_config=SimpleNamespace(
            translation_alignatt_max_source_regression=0,
            translation_alignatt_source_regression_action="trim_target_unit",
        ),
    )

    (
        trimmed,
        was_trimmed,
        trimmed_token_count,
        trimmed_unit_count,
        reason,
        reference_position,
        unit_position,
        bypassed_count,
    ) = policy.trim_for_source_regression_target_units(
        [0, 1, 2],
        aligned_source_local_positions=[0, 2, 1],
        provenance_mass=None,
        source_map=_prompt_source_map(),
    )

    assert trimmed == [0, 1]
    assert was_trimmed is True
    assert trimmed_token_count == 1
    assert trimmed_unit_count == 1
    assert reason == "source_regression"
    assert reference_position == 2
    assert unit_position == 1
    assert bypassed_count == 0


def test_source_regression_trim_unrecovered_keeps_recovered_local_regression():
    policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["▁A", "▁B", "▁C", "▁D", "▁E", "▁F"]),
        runtime_config=SimpleNamespace(
            translation_alignatt_max_source_regression=0,
            translation_alignatt_source_regression_action="trim_unrecovered",
        ),
    )

    (
        trimmed,
        was_trimmed,
        trimmed_token_count,
        trimmed_unit_count,
        reason,
        reference_position,
        unit_position,
        bypassed_count,
    ) = policy.trim_for_source_regression_target_units(
        [0, 1, 2, 3, 4, 5],
        aligned_source_local_positions=[0, 5, 8, 2, 4, 9],
        provenance_mass=None,
        source_map=_prompt_source_map(),
    )

    assert trimmed == [0, 1, 2, 3, 4, 5]
    assert was_trimmed is False
    assert trimmed_token_count == 0
    assert trimmed_unit_count == 0
    assert reason is None
    assert reference_position is None
    assert unit_position is None
    assert bypassed_count == 2


def test_source_regression_trim_unrecovered_cuts_unresolved_suffix():
    policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["▁A", "▁B", "▁C", "▁D", "▁E"]),
        runtime_config=SimpleNamespace(
            translation_alignatt_max_source_regression=0,
            translation_alignatt_source_regression_action="trim_unrecovered",
        ),
    )

    trimmed, was_trimmed, trimmed_token_count, trimmed_unit_count, reason, *_ = (
        policy.trim_for_source_regression_target_units(
            [0, 1, 2, 3, 4],
            aligned_source_local_positions=[0, 5, 8, 2, 4],
            provenance_mass=None,
            source_map=_prompt_source_map(),
        )
    )

    assert trimmed == [0, 1, 2]
    assert was_trimmed is True
    assert trimmed_token_count == 2
    assert trimmed_unit_count == 2
    assert reason == "source_regression"


def test_finalize_partial_records_source_regression_target_unit_trim():
    policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["甲", "乙", "丙"]),
        runtime_config=SimpleNamespace(
            target_lang="Simplified Chinese",
            translation_alignatt_max_source_regression=0,
            translation_alignatt_source_regression_action="trim_target_unit",
        ),
    )

    acceptance = policy.finalize_partial(
        accepted_candidate_ids=[0, 1, 2],
        aligned_source_local_positions=[0, 2, 1],
        provenance_mass=None,
        source_map=_prompt_source_map(),
        unsafe_reason=None,
        unsafe_target_token_index=None,
        unsafe_token_id=None,
        blocked_source_local_position=None,
        blocked_source_unit_index=None,
        stop_reason="length",
        probe_backend="test",
    )

    assert acceptance.accepted_generated_ids == [0, 1]
    assert acceptance.alignatt_metadata["alignatt_source_regression_action"] == (
        "trim_target_unit"
    )
    assert acceptance.alignatt_metadata["alignatt_source_regression_trimmed"] is True
    assert (
        acceptance.alignatt_metadata["alignatt_source_regression_trim_reason"]
        == "source_regression"
    )


def test_alignatt_metadata_records_source_unit_indices_for_tokens():
    policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["大", "家", "好"]),
        runtime_config=SimpleNamespace(target_lang="Simplified Chinese"),
    )
    source_map = PromptSourceMap(
        source_text="Hi everyone",
        source_token_positions=(10, 11, 12),
        source_unit_spans=(
            PromptSourceUnitSpan(
                unit_index=0,
                text="Hi",
                prompt_token_positions=(10,),
                is_accessible=True,
                start_ms=None,
                end_ms=None,
            ),
            PromptSourceUnitSpan(
                unit_index=1,
                text="everyone",
                prompt_token_positions=(11, 12),
                is_accessible=True,
                start_ms=None,
                end_ms=None,
            ),
        ),
        accessible_source_token_count=3,
        accessible_unit_count=2,
        total_unit_count=2,
        current_source_ms=1000.0,
        inaccessible_ms=0.0,
        is_final=False,
    )

    acceptance = policy.finalize_partial(
        accepted_candidate_ids=[0, 1, 2],
        aligned_source_local_positions=[0, 2, None],
        source_map=source_map,
        unsafe_reason=None,
        unsafe_target_token_index=None,
        unsafe_token_id=None,
        blocked_source_local_position=None,
        blocked_source_unit_index=None,
        stop_reason="stop",
        probe_backend="test",
    )

    assert acceptance.alignatt_metadata["aligned_source_unit_indices"] == [0, 1, None]


def test_source_bounded_prefill_keeps_only_tokens_inside_final_source_segment():
    bounded = source_bounded_prefill_generated_token_ids(
        accepted_generated_token_ids=(10, 11, 12, 13),
        aligned_source_unit_indices=(0, 1, 2, 2),
        final_source_unit_count=2,
    )

    assert bounded == (10, 11)


def test_source_bounded_prefill_preserves_legacy_when_unit_metadata_missing():
    bounded = source_bounded_prefill_generated_token_ids(
        accepted_generated_token_ids=(10, 11, 12),
        aligned_source_unit_indices=(),
        final_source_unit_count=1,
    )

    assert bounded == (10, 11, 12)


def test_translation_unit_manager_decodes_source_bounded_prefill():
    class FakeBackend:
        pieces = {10: "大", 11: "家好", 12: "，我是", 13: "珍"}

        def decode_candidate_text(self, *, generated_ids, assistant_prefill, variant, is_partial):
            del assistant_prefill, variant, is_partial
            return "".join(self.pieces[int(token_id)] for token_id in generated_ids)

        def encode_semantic_target_token_ids(self, text):
            return tuple(ord(char) for char in text)

    class FakeBundle:
        def ensure_mt_backend(self):
            return FakeBackend()

    class FakeSession:
        config = CascadeRuntimeConfig(target_lang="Simplified Chinese")
        bundle = FakeBundle()
        state = CascadeState()

        def current_audio_seconds(self):
            return 10.0

        def current_live_asr_tail_text(self):
            return self.state.asr_hypotheses[-1]

    manager = TranslationUnitManager(FakeSession())
    manager.state.partial_translation = PartialTranslationState(
        accepted_target="大家好，我是珍",
        accepted_token_ids=(101, 102, 103, 104),
        accepted_generated_token_ids=(10, 11, 12, 13),
        last_alignatt_metadata={
            "aligned_source_unit_indices": [0, 1, 2, 2],
        },
    )

    assert manager.source_bounded_accepted_prefill(final_source_unit_count=2) == "大家好"


def test_committed_split_carries_future_aligned_partial_suffix():
    class FakeBackend:
        pieces = {10: "大", 11: "家好", 12: "，我是", 13: "珍"}

        def decode_candidate_text(self, *, generated_ids, assistant_prefill, variant, is_partial):
            del assistant_prefill, variant, is_partial
            return "".join(self.pieces[int(token_id)] for token_id in generated_ids)

        def encode_semantic_target_token_ids(self, text):
            return tuple(ord(char) for char in text)

    class FakeBundle:
        def ensure_mt_backend(self):
            return FakeBackend()

    class FakeSession:
        config = CascadeRuntimeConfig(target_lang="Simplified Chinese")
        bundle = FakeBundle()

        def __init__(self):
            self.state = CascadeState()

        def current_audio_seconds(self):
            return 10.0

        def current_live_asr_tail_text(self):
            return self.state.asr_hypotheses[-1]

    session = FakeSession()
    session.state.utt_sources = ["", "Hi everyone."]
    session.state.utt_translations = [""]
    session.state.asr_hypotheses = ["I'm Jenny, a first-year"]
    session.state.partial_translation = PartialTranslationState(
        source_prefix="Hi everyone. I'm Jenny, a first-year",
        accepted_target="大家好，我是珍",
        accepted_token_ids=(101, 102, 103, 104),
        accepted_generated_token_ids=(10, 11, 12, 13),
        last_alignatt_metadata={
            "aligned_source_unit_indices": [0, 1, 2, 2],
        },
    )
    manager = TranslationUnitManager(session)
    final_frontier = build_source_accessibility_frontier(
        "Hi everyone.",
        word_timestamps_ms=None,
        current_audio_ms=10_000.0,
        inaccessible_ms=0.0,
        is_final=True,
    )

    result = manager.commit_source_bounded_partial_acceptance(
        final_source_frontier=final_frontier,
    )

    assert result is not None
    assert session.state.utt_translations == ["", "大家好"]
    assert session.state.partial_translation.source_prefix == "I'm Jenny, a first-year"
    assert session.state.partial_translation.accepted_target == "，我是珍"
    assert session.state.partial_translation.accepted_generated_token_ids == (12, 13)


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


def test_runtime_validates_milmmt_prompt_mode():
    with pytest.raises(ValueError, match="milmmt_prompt_mode"):
        CascadeRuntimeConfig(milmmt_prompt_mode="chat")


def test_gemma_low_latency_alignatt_policy_is_the_runtime_default():
    runtime = CascadeRuntimeConfig()
    preset_config = get_runtime_preset("gemma_low_latency").build_speech_processor_config(
        source_lang_code="en",
        target_lang_code="de",
    )

    expected = {
        "translation_alignatt_top_k_heads": 4,
        "translation_alignatt_border_margin": 1,
        "translation_alignatt_min_source_mass": 0.0,
        "translation_alignatt_frontier_min_inaccessible_mass": 0.03,
        "translation_alignatt_max_inaccessible_source_mass": 1.0,
        "translation_alignatt_source_bearing_min_source_mass": 0.005,
        "translation_alignatt_source_bearing_hard_inaccessible_cap": 1.0,
    }
    for name, value in expected.items():
        assert getattr(runtime, name) == value
        assert getattr(preset_config, name) == value


def test_legacy_runtime_preset_alias_resolves_to_runtime_preset():
    legacy_config = get_runtime_preset("main_low_latency").build_speech_processor_config(
        source_lang_code="en",
        target_lang_code="de",
    )
    active_config = get_runtime_preset("gemma_low_latency").build_speech_processor_config(
        source_lang_code="en",
        target_lang_code="de",
    )

    assert legacy_config.chunk_ms == active_config.chunk_ms == 850
    assert legacy_config.mt_backend_name == active_config.mt_backend_name


def test_refresh_alignatt_artifacts_rebinds_policy_runtime_config():
    """Hot-swapped runtime configs must reach the decoder policy.

    The policy sweep reuses one loaded bundle across policy points by setting
    ``backend.runtime_config`` and calling ``refresh_alignatt_artifacts``.
    The decoder policy holds its own ``runtime_config`` reference, so the
    refresh must rebind it; otherwise every point after the first executes
    the first point's acceptance knobs while manifests record the new ones.
    """
    backend = GemmaVLLMMTBackend.__new__(GemmaVLLMMTBackend)
    old_config = SimpleNamespace(
        translation_alignatt_heads_path=None,
        translation_alignatt_top_k_heads=4,
        translation_alignatt_min_accepted_accessible_source_mass=0.001,
        translation_alignatt_frontier_min_inaccessible_mass=0.005,
    )
    new_config = SimpleNamespace(
        translation_alignatt_heads_path=None,
        translation_alignatt_top_k_heads=4,
        translation_alignatt_min_accepted_accessible_source_mass=0.02,
        translation_alignatt_frontier_min_inaccessible_mass=0.03,
    )
    backend.runtime_config = old_config
    backend.policy = SimpleNamespace(runtime_config=old_config)

    backend.runtime_config = new_config
    backend.refresh_alignatt_artifacts()

    assert backend.policy.runtime_config is new_config


def _unit_conf_policy(min_confidence: float) -> AlignAttDecoderPolicy:
    return AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["你"]),
        runtime_config=SimpleNamespace(
            target_lang="Simplified Chinese",
            translation_alignatt_acceptance_variant="unit_conf",
            translation_alignatt_border_margin=0,
            translation_alignatt_min_alignment_confidence=min_confidence,
        ),
    )


def test_unit_conf_accepts_confident_unit_within_frontier():
    policy = _unit_conf_policy(0.6)

    decision = policy.accept_complete_target_units(
        generated_ids=[0],
        aligned_source_local_positions=[0],
        source_attention_rows=None,
        provenance_mass=None,
        source_map=_source_map_with_accessible_units(2),
        finish_reason="length",
        per_head_aligned_source_local_positions=[[0, 0, 1, 0]],
    )

    assert decision.accepted_candidate_ids == [0]
    assert decision.metadata["alignatt_unit_policy_accepted_unit_count"] == 1
    assert decision.metadata["alignatt_unit_policy_last_unit_confidence"] == 1.0
    assert decision.metadata["alignatt_unit_conf_unit_confidences"] == [1.0]


def test_unit_conf_defers_unit_on_weak_alignment_confidence():
    policy = _unit_conf_policy(0.75)

    # Heads disagree wildly: only 1/4 within one token of the consensus (0).
    decision = policy.accept_complete_target_units(
        generated_ids=[0],
        aligned_source_local_positions=[0],
        source_attention_rows=None,
        provenance_mass=None,
        source_map=_source_map_with_accessible_units(8),
        finish_reason="length",
        per_head_aligned_source_local_positions=[[0, 4, 6, 7]],
    )

    assert decision.accepted_candidate_ids == []
    assert decision.unsafe_reason == "unit_confidence_weak"
    assert decision.stop_reason == "alignatt:unit_confidence_weak"


def test_unit_conf_frontier_failure_precedes_confidence():
    policy = _unit_conf_policy(0.1)

    # Consensus argmax beyond the accessible frontier: the frontier reason
    # must win even though every head agrees.
    decision = policy.accept_complete_target_units(
        generated_ids=[0],
        aligned_source_local_positions=[9],
        source_attention_rows=None,
        provenance_mass=None,
        source_map=_source_map_with_accessible_units(2),
        finish_reason="length",
        per_head_aligned_source_local_positions=[[9, 9, 9, 9]],
    )

    assert decision.accepted_candidate_ids == []
    assert decision.unsafe_reason == "source_frontier"


def test_unit_conf_zero_threshold_matches_unit_argmax():
    conf_policy = _unit_conf_policy(0.0)
    argmax_policy = AlignAttDecoderPolicy(
        tokenizer=TokenListTokenizer(["你"]),
        runtime_config=SimpleNamespace(
            target_lang="Simplified Chinese",
            translation_alignatt_acceptance_variant="unit_argmax",
            translation_alignatt_border_margin=0,
        ),
    )

    kwargs = dict(
        generated_ids=[0],
        aligned_source_local_positions=[0],
        source_attention_rows=None,
        provenance_mass=None,
        source_map=_source_map_with_accessible_units(2),
        finish_reason="length",
    )
    conf_decision = conf_policy.accept_complete_target_units(
        per_head_aligned_source_local_positions=[[0, 5, 6, 7]],
        **kwargs,
    )
    argmax_decision = argmax_policy.accept_complete_target_units(**kwargs)

    assert conf_decision.accepted_candidate_ids == argmax_decision.accepted_candidate_ids
    assert conf_decision.unsafe_reason == argmax_decision.unsafe_reason


def test_unit_conf_missing_per_head_positions_defers():
    policy = _unit_conf_policy(0.6)

    decision = policy.accept_complete_target_units(
        generated_ids=[0],
        aligned_source_local_positions=[0],
        source_attention_rows=None,
        provenance_mass=None,
        source_map=_source_map_with_accessible_units(2),
        finish_reason="length",
        per_head_aligned_source_local_positions=None,
    )

    assert decision.accepted_candidate_ids == []
    assert decision.unsafe_reason == "attention_missing"


def test_runtime_validates_min_alignment_confidence_interval():
    with pytest.raises(ValueError):
        CascadeRuntimeConfig(translation_alignatt_min_alignment_confidence=1.5)
    config = CascadeRuntimeConfig(
        translation_alignatt_acceptance_variant="unit_conf",
        translation_alignatt_min_alignment_confidence=0.75,
    )
    assert config.translation_alignatt_min_alignment_confidence == 0.75


def test_attention_confidence_summary_per_token_features():
    from alignatt4llm.mt.gemma_vllm_backend import _summarize_attention_confidence

    rows = [
        # Two heads, four source positions: concentrated on position 0.
        torch.tensor([[0.7, 0.1, 0.1, 0.1], [0.9, 0.05, 0.03, 0.02]]),
        # Non-finite row: entropy/concentration must be None.
        torch.tensor([[float("nan"), 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
    ]
    features = _summarize_attention_confidence(
        rows,
        aligned_source_local_positions=[0, 3],
        per_head_aligned_source_local_positions=[[0, 1], [0, 3]],
        argmax_raw_mass_per_token=[0.8, None],
    )

    assert len(features) == 2
    first = features[0]
    assert first["consensus_ratio"] == 1.0
    assert first["argmax_mass"] == 0.8
    assert 0.0 < first["entropy_norm"] < 1.0
    mean_row = torch.stack([rows[0][0], rows[0][1]]).mean(dim=0)
    top2 = torch.topk(mean_row, k=2).values
    assert abs(first["concentration"] - float((top2[0] - top2[1]).item())) < 1e-6
    second = features[1]
    assert second["consensus_ratio"] == 0.5
    assert second["entropy_norm"] is None
    assert second["concentration"] is None
    assert second["argmax_mass"] is None
