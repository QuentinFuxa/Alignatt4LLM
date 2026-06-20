from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tools.research.run_mt_cutoff_policy_sweep import (
    build_processor_config,
    parse_args,
    point_value,
    policy_points,
    policy_point_record,
)


def _record_args() -> SimpleNamespace:
    return SimpleNamespace(
        mt_backend_name="milmmt_vllm_alignatt",
        alignatt_inaccessible_ms=160.0,
        alignatt_frontier_min_inaccessible_mass=0.03,
        alignatt_source_frontier_action="trim_unrecovered",
        alignatt_max_inaccessible_source_mass=0.15,
        alignatt_min_accessible_inaccessible_margin=0.0,
        alignatt_min_accessible_source_units=6,
        alignatt_min_accessible_source_units_mode="target_unit_cap",
        alignatt_online_normalization="raw",
        alignatt_source_bearing_min_source_mass=0.04,
        alignatt_source_bearing_hard_inaccessible_cap=0.60,
        alignatt_max_non_source_prompt_mass=0.80,
        alignatt_min_accepted_accessible_source_mass=0.001,
        alignatt_accepted_accessible_source_mass_recent_units=3,
        alignatt_max_source_regression=0,
        alignatt_source_regression_min_inaccessible_mass=0.03,
        alignatt_source_regression_recent_tokens=3,
        alignatt_source_regression_reference_mode="median_recent",
        alignatt_source_regression_activation_mode="frontier_reached",
        alignatt_source_regression_activation_slack_tokens=4,
        alignatt_source_regression_patience_tokens=2,
        alignatt_source_regression_action="trim_target_unit",
        alignatt_token_argmax_frontier_gate=True,
        alignatt_token_argmax_frontier_patience_tokens=2,
        alignatt_source_lcp_stability=True,
        alignatt_source_lcp_append_slack_units=1,
        asr_gpu_memory_utilization=0.4,
        mt_vllm_enforce_eager=True,
        mt_vllm_enable_speculative_decoding=False,
        mt_vllm_speculative_assistant_model=None,
        mt_vllm_num_speculative_tokens=4,
    )


def test_cutoff_sweep_can_build_milmmt_policy_config():
    config = build_processor_config(
        base_preset_name="gemma_low_latency",
        source="en",
        target="zh",
        chunk_ms=640,
        mt_backend_name="milmmt_vllm_alignatt",
        policy="alignatt",
        cutoff_units=0,
        alignatt_border_margin=1,
        alignatt_top_k_heads=8,
        alignatt_acceptance_variant="unit_mass",
        alignatt_min_source_mass=0.003,
        alignatt_unit_consensus_min_head_ratio=0.55,
        alignatt_online_normalization="raw",
        alignatt_source_bearing_min_source_mass=0.04,
        alignatt_source_bearing_hard_inaccessible_cap=0.60,
        alignatt_max_non_source_prompt_mass=0.80,
        alignatt_min_accepted_accessible_source_mass=0.001,
        alignatt_accepted_accessible_source_mass_recent_units=3,
        alignatt_inaccessible_ms=160.0,
        alignatt_frontier_min_inaccessible_mass=0.03,
        alignatt_source_frontier_action="trim_unrecovered",
        alignatt_max_inaccessible_source_mass=0.15,
        alignatt_min_accessible_inaccessible_margin=0.0,
        alignatt_min_accessible_source_units=6,
        alignatt_min_accessible_source_units_mode="target_unit_cap",
        alignatt_max_source_regression=0,
        alignatt_source_regression_min_inaccessible_mass=0.03,
        alignatt_source_regression_recent_tokens=3,
        alignatt_source_regression_reference_mode="median_recent",
        alignatt_source_regression_activation_mode="frontier_reached",
        alignatt_source_regression_activation_slack_tokens=4,
        alignatt_source_regression_patience_tokens=2,
        alignatt_source_regression_action="trim_target_unit",
        alignatt_token_argmax_frontier_gate=True,
        alignatt_token_argmax_frontier_patience_tokens=2,
        alignatt_source_lcp_stability=True,
        alignatt_source_lcp_append_slack_units=1,
        asr_gpu_memory_utilization=0.4,
        mt_vllm_gpu_memory_utilization=0.5,
        gemma_vllm_gpu_memory_utilization=None,
        mt_vllm_enable_speculative_decoding=False,
        mt_vllm_speculative_assistant_model=None,
        mt_vllm_num_speculative_tokens=4,
    )

    assert config.mt_backend_name == "milmmt_vllm_alignatt"
    assert config.translation_acceptance_policy == "alignatt"
    assert config.translation_alignatt_acceptance_variant == "unit_mass"
    assert config.translation_alignatt_unit_consensus_min_head_ratio == 0.55
    assert config.translation_alignatt_online_normalization == "raw"
    assert config.translation_alignatt_source_bearing_min_source_mass == 0.04
    assert config.translation_alignatt_source_bearing_hard_inaccessible_cap == 0.60
    assert config.translation_alignatt_max_non_source_prompt_mass == 0.80
    assert config.translation_alignatt_min_accepted_accessible_source_mass == 0.001
    assert config.translation_alignatt_accepted_accessible_source_mass_recent_units == 3
    assert config.translation_alignatt_inaccessible_ms == 160.0
    assert config.translation_alignatt_frontier_min_inaccessible_mass == 0.03
    assert config.translation_alignatt_source_frontier_action == "trim_unrecovered"
    assert config.translation_alignatt_min_accessible_source_units == 6
    assert config.translation_alignatt_min_accessible_source_units_mode == "target_unit_cap"
    assert config.translation_alignatt_max_source_regression == 0
    assert config.translation_alignatt_source_regression_min_inaccessible_mass == 0.03
    assert config.translation_alignatt_source_regression_recent_tokens == 3
    assert config.translation_alignatt_source_regression_reference_mode == "median_recent"
    assert (
        config.translation_alignatt_source_regression_activation_mode
        == "frontier_reached"
    )
    assert config.translation_alignatt_source_regression_activation_slack_tokens == 4
    assert config.translation_alignatt_source_regression_patience_tokens == 2
    assert config.translation_alignatt_source_regression_action == "trim_target_unit"
    assert config.translation_alignatt_token_argmax_frontier_gate is True
    assert config.translation_alignatt_token_argmax_frontier_patience_tokens == 2
    assert config.translation_alignatt_source_lcp_stability is True
    assert config.translation_alignatt_source_lcp_append_slack_units == 1
    assert config.asr_gpu_memory_utilization == 0.4
    assert config.translation_alignatt_top_k_heads == 8


def test_cutoff_sweep_default_source_bearing_threshold_is_permissive(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mt_cutoff_policy_sweep.py",
            "--inputs",
            "data/smoke/mt_parity_set.json",
            "--output-root",
            "outputs/tmp",
        ],
    )

    args = parse_args()

    assert args.alignatt_source_bearing_min_source_mass == 0.005
    assert args.alignatt_source_bearing_hard_inaccessible_cap == 1.0
    assert args.alignatt_source_frontier_action == "stop"
    assert args.alignatt_source_regression_action == "stop"


def test_cutoff_sweep_accepts_source_regression_target_unit_trim_action(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mt_cutoff_policy_sweep.py",
            "--inputs",
            "data/smoke/mt_parity_set.json",
            "--output-root",
            "outputs/tmp",
            "--alignatt-source-regression-action",
            "trim_target_unit",
        ],
    )

    args = parse_args()

    assert args.alignatt_source_regression_action == "trim_target_unit"


def test_policy_points_tag_source_regression_unrecovered_trim():
    args = SimpleNamespace(
        skip_alignatt=False,
        skip_cutoffs=True,
        cutoffs=[],
        base_preset="gemma_low_latency",
        alignatt_border_margins=None,
        alignatt_top_k_heads=None,
        alignatt_acceptance_variants=None,
        alignatt_min_source_masses=None,
        alignatt_unit_consensus_min_head_ratio=0.60,
        alignatt_unit_consensus_min_head_ratios=None,
        alignatt_online_normalization="zscore",
        alignatt_online_normalizations=None,
        alignatt_source_bearing_min_source_mass=0.005,
        alignatt_source_bearing_min_source_masses=None,
        alignatt_source_bearing_hard_inaccessible_cap=1.0,
        alignatt_source_bearing_hard_inaccessible_caps=None,
        alignatt_max_non_source_prompt_mass=1.0,
        alignatt_max_non_source_prompt_masses=None,
        alignatt_inaccessible_ms=0.0,
        alignatt_inaccessible_ms_values=None,
        alignatt_frontier_min_inaccessible_mass=0.03,
        alignatt_frontier_min_inaccessible_masses=None,
        alignatt_source_frontier_action="stop",
        alignatt_min_accessible_source_units_mode="block",
        alignatt_min_accessible_source_units_modes=None,
        alignatt_source_regression_min_inaccessible_mass=0.0,
        alignatt_source_regression_min_inaccessible_masses=None,
        alignatt_source_regression_action="trim_unrecovered",
        alignatt_source_lcp_stability=False,
        alignatt_source_lcp_append_slack_units=0,
    )

    point = policy_points(args)[0]

    assert point["tag"] == "alignatt_srtrim_unrecovered"
    assert point["alignatt_source_regression_action"] == "trim_unrecovered"


def test_policy_points_tag_source_frontier_unrecovered_trim():
    args = SimpleNamespace(
        skip_alignatt=False,
        skip_cutoffs=True,
        cutoffs=[],
        base_preset="gemma_low_latency",
        alignatt_border_margins=None,
        alignatt_top_k_heads=None,
        alignatt_acceptance_variants=None,
        alignatt_min_source_masses=None,
        alignatt_unit_consensus_min_head_ratio=0.60,
        alignatt_unit_consensus_min_head_ratios=None,
        alignatt_online_normalization="zscore",
        alignatt_online_normalizations=None,
        alignatt_source_bearing_min_source_mass=0.005,
        alignatt_source_bearing_min_source_masses=None,
        alignatt_source_bearing_hard_inaccessible_cap=1.0,
        alignatt_source_bearing_hard_inaccessible_caps=None,
        alignatt_max_non_source_prompt_mass=1.0,
        alignatt_max_non_source_prompt_masses=None,
        alignatt_inaccessible_ms=0.0,
        alignatt_inaccessible_ms_values=None,
        alignatt_frontier_min_inaccessible_mass=0.03,
        alignatt_frontier_min_inaccessible_masses=None,
        alignatt_source_frontier_action="trim_unrecovered",
        alignatt_min_accessible_source_units_mode="block",
        alignatt_min_accessible_source_units_modes=None,
        alignatt_source_regression_min_inaccessible_mass=0.0,
        alignatt_source_regression_min_inaccessible_masses=None,
        alignatt_source_regression_action="stop",
        alignatt_source_lcp_stability=False,
        alignatt_source_lcp_append_slack_units=0,
    )

    point = policy_points(args)[0]

    assert point["tag"] == "alignatt_sftrim_unrecovered"
    assert point["alignatt_source_frontier_action"] == "trim_unrecovered"


def test_cutoff_sweep_rejects_inactive_source_lcp_append_slack():
    with pytest.raises(ValueError, match="source_lcp_append_slack_units"):
        build_processor_config(
            base_preset_name="gemma_low_latency",
            source="en",
            target="zh",
            chunk_ms=640,
            mt_backend_name="milmmt_vllm_alignatt",
            policy="alignatt",
            cutoff_units=0,
            alignatt_border_margin=1,
            alignatt_top_k_heads=8,
            alignatt_acceptance_variant=None,
            alignatt_min_source_mass=0.003,
            alignatt_unit_consensus_min_head_ratio=0.60,
            alignatt_online_normalization="zscore",
            alignatt_source_bearing_min_source_mass=0.05,
            alignatt_source_bearing_hard_inaccessible_cap=0.75,
            alignatt_inaccessible_ms=0.0,
            alignatt_frontier_min_inaccessible_mass=0.03,
            alignatt_source_frontier_action="stop",
            alignatt_max_inaccessible_source_mass=1.0,
            alignatt_min_accessible_inaccessible_margin=-1.0,
            alignatt_min_accessible_source_units=None,
            alignatt_min_accessible_source_units_mode="block",
            alignatt_max_source_regression=None,
            alignatt_source_regression_min_inaccessible_mass=0.0,
            alignatt_source_regression_recent_tokens=0,
            alignatt_source_regression_reference_mode="max",
            alignatt_source_regression_activation_mode="always",
            alignatt_source_regression_activation_slack_tokens=0,
            alignatt_source_regression_patience_tokens=1,
            alignatt_source_regression_action="stop",
            alignatt_token_argmax_frontier_gate=False,
            alignatt_token_argmax_frontier_patience_tokens=1,
            alignatt_source_lcp_stability=False,
            alignatt_source_lcp_append_slack_units=1,
            asr_gpu_memory_utilization=None,
            mt_vllm_gpu_memory_utilization=None,
            gemma_vllm_gpu_memory_utilization=None,
            mt_vllm_enable_speculative_decoding=False,
            mt_vllm_speculative_assistant_model=None,
            mt_vllm_num_speculative_tokens=4,
        )


def test_cutoff_only_policy_point_coalesces_alignatt_defaults():
    args = SimpleNamespace(
        skip_alignatt=True,
        skip_cutoffs=False,
        cutoffs=[3],
    )

    point = policy_points(args)[0]
    config = build_processor_config(
        base_preset_name="gemma_low_latency",
        source="en",
        target="zh",
        chunk_ms=640,
        mt_backend_name="milmmt_vllm_alignatt",
        policy=point["policy"],
        cutoff_units=point["cutoff_units"],
        alignatt_border_margin=point.get("alignatt_border_margin"),
        alignatt_top_k_heads=point.get("alignatt_top_k_heads"),
        alignatt_acceptance_variant=point.get("alignatt_acceptance_variant"),
        alignatt_min_source_mass=point.get("alignatt_min_source_mass"),
        alignatt_unit_consensus_min_head_ratio=point_value(
            point, "alignatt_unit_consensus_min_head_ratio", 0.60
        ),
        alignatt_online_normalization=point_value(
            point, "alignatt_online_normalization", "zscore"
        ),
        alignatt_source_bearing_min_source_mass=point_value(
            point, "alignatt_source_bearing_min_source_mass", 0.05
        ),
        alignatt_source_bearing_hard_inaccessible_cap=point_value(
            point, "alignatt_source_bearing_hard_inaccessible_cap", 1.0
        ),
        alignatt_max_non_source_prompt_mass=point_value(
            point, "alignatt_max_non_source_prompt_mass", 1.0
        ),
        alignatt_inaccessible_ms=point_value(point, "alignatt_inaccessible_ms", 0.0),
        alignatt_frontier_min_inaccessible_mass=point_value(
            point, "alignatt_frontier_min_inaccessible_mass", 0.0
        ),
        alignatt_source_frontier_action=point_value(
            point, "alignatt_source_frontier_action", "stop"
        ),
        alignatt_max_inaccessible_source_mass=1.0,
        alignatt_min_accessible_inaccessible_margin=-1.0,
        alignatt_min_accessible_source_units=None,
        alignatt_min_accessible_source_units_mode="block",
        alignatt_max_source_regression=None,
        alignatt_source_regression_min_inaccessible_mass=point_value(
            point, "alignatt_source_regression_min_inaccessible_mass", 0.0
        ),
        alignatt_source_regression_recent_tokens=0,
        alignatt_source_regression_reference_mode="max",
        alignatt_source_regression_activation_mode="always",
        alignatt_source_regression_activation_slack_tokens=0,
        alignatt_source_regression_patience_tokens=1,
        alignatt_source_regression_action="stop",
        alignatt_token_argmax_frontier_gate=False,
        alignatt_token_argmax_frontier_patience_tokens=1,
        alignatt_source_lcp_stability=False,
        alignatt_source_lcp_append_slack_units=0,
        asr_gpu_memory_utilization=None,
        mt_vllm_gpu_memory_utilization=None,
        gemma_vllm_gpu_memory_utilization=None,
        mt_vllm_enable_speculative_decoding=False,
        mt_vllm_speculative_assistant_model=None,
        mt_vllm_num_speculative_tokens=4,
    )

    assert config.translation_acceptance_policy == "cut_last_target_units"
    assert config.translation_static_cutoff_units == 3
    assert config.translation_alignatt_max_non_source_prompt_mass == 1.0


def test_policy_point_record_includes_reproducibility_knobs():
    point = {
        "tag": "alignatt_b1_top8_m0p003",
        "policy": "alignatt",
        "cutoff_units": 0,
        "alignatt_border_margin": 1,
        "alignatt_top_k_heads": 8,
        "alignatt_acceptance_variant": "unit_mass",
        "alignatt_min_source_mass": 0.003,
        "alignatt_unit_consensus_min_head_ratio": 0.55,
        "alignatt_online_normalization": "raw",
        "alignatt_source_bearing_min_source_mass": 0.04,
        "alignatt_source_bearing_hard_inaccessible_cap": 0.60,
        "alignatt_max_non_source_prompt_mass": 0.80,
        "alignatt_min_accepted_accessible_source_mass": 0.001,
        "alignatt_inaccessible_ms": 160.0,
        "alignatt_frontier_min_inaccessible_mass": 0.03,
        "alignatt_source_frontier_action": "trim_unrecovered",
        "alignatt_min_alignment_confidence": 0.65,
    }

    row = policy_point_record(
        args=_record_args(),
        point=point,
        output_dir=Path("outputs/policy/alignatt"),
        input_count=3,
        model_load_ms=None,
        resumed=True,
    )

    assert row["mt_backend_name"] == "milmmt_vllm_alignatt"
    assert row["alignatt_min_alignment_confidence"] == 0.65
    assert row["alignatt_inaccessible_ms"] == 160.0
    assert row["alignatt_acceptance_variant"] == "unit_mass"
    assert row["alignatt_unit_consensus_min_head_ratio"] == 0.55
    assert row["alignatt_online_normalization"] == "raw"
    assert row["alignatt_source_bearing_min_source_mass"] == 0.04
    assert row["alignatt_source_bearing_hard_inaccessible_cap"] == 0.60
    assert row["alignatt_max_non_source_prompt_mass"] == 0.80
    assert row["alignatt_min_accepted_accessible_source_mass"] == 0.001
    assert row["alignatt_accepted_accessible_source_mass_recent_units"] == 3
    assert row["alignatt_frontier_min_inaccessible_mass"] == 0.03
    assert row["alignatt_source_frontier_action"] == "trim_unrecovered"
    assert row["alignatt_min_accessible_source_units"] == 6
    assert row["alignatt_min_accessible_source_units_mode"] == "target_unit_cap"
    assert row["alignatt_max_source_regression"] == 0
    assert row["alignatt_source_regression_min_inaccessible_mass"] == 0.03
    assert row["alignatt_source_regression_recent_tokens"] == 3
    assert row["alignatt_source_regression_reference_mode"] == "median_recent"
    assert row["alignatt_source_regression_activation_mode"] == "frontier_reached"
    assert row["alignatt_source_regression_activation_slack_tokens"] == 4
    assert row["alignatt_source_regression_patience_tokens"] == 2
    assert row["alignatt_source_regression_action"] == "trim_target_unit"
    assert row["alignatt_token_argmax_frontier_gate"] is True
    assert row["alignatt_token_argmax_frontier_patience_tokens"] == 2
    assert row["alignatt_source_lcp_stability"] is True
    assert row["alignatt_source_lcp_append_slack_units"] == 1
    assert row["asr_gpu_memory_utilization"] == 0.4
    assert row["resumed"] is True


def test_default_alignatt_policy_point_records_effective_preset_values():
    args = SimpleNamespace(
        skip_alignatt=False,
        skip_cutoffs=True,
        cutoffs=[],
        base_preset="gemma_low_latency",
        alignatt_border_margins=None,
        alignatt_top_k_heads=None,
        alignatt_acceptance_variants=None,
        alignatt_min_source_masses=None,
        alignatt_unit_consensus_min_head_ratio=0.60,
        alignatt_unit_consensus_min_head_ratios=None,
        alignatt_online_normalization="zscore",
        alignatt_online_normalizations=None,
        alignatt_source_bearing_min_source_mass=0.05,
        alignatt_source_bearing_min_source_masses=None,
        alignatt_source_bearing_hard_inaccessible_cap=1.0,
        alignatt_source_bearing_hard_inaccessible_caps=None,
        alignatt_max_non_source_prompt_mass=1.0,
        alignatt_max_non_source_prompt_masses=None,
        alignatt_inaccessible_ms=0.0,
        alignatt_inaccessible_ms_values=None,
        alignatt_frontier_min_inaccessible_mass=0.03,
        alignatt_frontier_min_inaccessible_masses=None,
        alignatt_source_frontier_action="stop",
        alignatt_min_accessible_source_units_mode="block",
        alignatt_min_accessible_source_units_modes=None,
        alignatt_source_regression_min_inaccessible_mass=0.0,
        alignatt_source_regression_min_inaccessible_masses=None,
        alignatt_source_regression_action="stop",
        alignatt_source_lcp_stability=False,
        alignatt_source_lcp_append_slack_units=0,
    )

    point = policy_points(args)[0]

    assert point["tag"] == "alignatt"
    assert point["alignatt_border_margin"] == 1
    assert point["alignatt_top_k_heads"] == 4
    assert point["alignatt_acceptance_variant"] == "token"
    assert point["alignatt_min_source_mass"] == 0.0
    assert point["alignatt_max_non_source_prompt_mass"] == 1.0
    assert point["alignatt_min_accepted_accessible_source_mass"] == 0.0


def test_policy_points_can_sweep_source_regression_future_mass():
    args = SimpleNamespace(
        skip_alignatt=False,
        skip_cutoffs=True,
        cutoffs=[],
        base_preset="gemma_low_latency",
        alignatt_border_margins=[1],
        alignatt_top_k_heads=[8],
        alignatt_acceptance_variants=None,
        alignatt_min_source_masses=[0.003],
        alignatt_unit_consensus_min_head_ratio=0.60,
        alignatt_unit_consensus_min_head_ratios=None,
        alignatt_online_normalization="zscore",
        alignatt_online_normalizations=None,
        alignatt_inaccessible_ms=0.0,
        alignatt_inaccessible_ms_values=None,
        alignatt_frontier_min_inaccessible_mass=0.03,
        alignatt_frontier_min_inaccessible_masses=None,
        alignatt_source_frontier_action="stop",
        alignatt_source_regression_min_inaccessible_mass=0.0,
        alignatt_source_regression_min_inaccessible_masses=[0.001, 0.003],
        alignatt_source_regression_action="stop",
        alignatt_source_lcp_stability=True,
        alignatt_source_lcp_append_slack_units=1,
    )

    points = policy_points(args)

    assert [point["tag"] for point in points] == [
        "alignatt_b1_top8_m0p003_frontier0p03_future0p001_lcp_aslack1",
        "alignatt_b1_top8_m0p003_frontier0p03_future0p003_lcp_aslack1",
    ]
    assert [
        point["alignatt_source_regression_min_inaccessible_mass"]
        for point in points
    ] == [0.001, 0.003]


def test_policy_points_can_sweep_clean_frontier_axes():
    args = SimpleNamespace(
        skip_alignatt=False,
        skip_cutoffs=True,
        cutoffs=[],
        base_preset="gemma_low_latency",
        alignatt_border_margins=[1],
        alignatt_top_k_heads=[8],
        alignatt_acceptance_variants=["token", "unit_mass"],
        alignatt_min_source_masses=[0.0, 0.003],
        alignatt_unit_consensus_min_head_ratio=0.60,
        alignatt_unit_consensus_min_head_ratios=None,
        alignatt_online_normalization="zscore",
        alignatt_online_normalizations=None,
        alignatt_inaccessible_ms=0.0,
        alignatt_inaccessible_ms_values=[0.0, 160.0],
        alignatt_frontier_min_inaccessible_mass=0.03,
        alignatt_frontier_min_inaccessible_masses=[0.0, 0.03],
        alignatt_source_frontier_action="stop",
        alignatt_source_regression_min_inaccessible_mass=0.0,
        alignatt_source_regression_min_inaccessible_masses=None,
        alignatt_source_regression_action="stop",
        alignatt_source_lcp_stability=False,
        alignatt_source_lcp_append_slack_units=0,
    )

    points = policy_points(args)

    assert len(points) == 16
    assert points[0]["tag"] == "alignatt_b1_top8"
    assert points[-1]["tag"] == (
        "alignatt_b1_top8_unit_mass_m0p003_inacc160_frontier0p03"
    )
    assert sorted({point["alignatt_acceptance_variant"] for point in points}) == [
        "token",
        "unit_mass",
    ]
    assert sorted({point["alignatt_inaccessible_ms"] for point in points}) == [
        0.0,
        160.0,
    ]
    assert sorted(
        {point["alignatt_frontier_min_inaccessible_mass"] for point in points}
    ) == [0.0, 0.03]


def test_policy_points_can_sweep_unit_consensus_ratio():
    args = SimpleNamespace(
        skip_alignatt=False,
        skip_cutoffs=True,
        cutoffs=[],
        base_preset="gemma_low_latency",
        alignatt_border_margins=[1],
        alignatt_top_k_heads=[8],
        alignatt_acceptance_variants=["unit_consensus"],
        alignatt_min_source_masses=[0.0],
        alignatt_unit_consensus_min_head_ratio=0.60,
        alignatt_unit_consensus_min_head_ratios=[0.45, 0.60],
        alignatt_online_normalization="zscore",
        alignatt_online_normalizations=None,
        alignatt_inaccessible_ms=0.0,
        alignatt_inaccessible_ms_values=[0.0, 160.0],
        alignatt_frontier_min_inaccessible_mass=0.0,
        alignatt_frontier_min_inaccessible_masses=[0.0],
        alignatt_source_frontier_action="stop",
        alignatt_source_regression_min_inaccessible_mass=0.0,
        alignatt_source_regression_min_inaccessible_masses=None,
        alignatt_source_regression_action="stop",
        alignatt_source_lcp_stability=False,
        alignatt_source_lcp_append_slack_units=0,
    )

    points = policy_points(args)

    assert [point["tag"] for point in points] == [
        "alignatt_b1_top8_unit_consensus_consensus0p45",
        "alignatt_b1_top8_unit_consensus_consensus0p45_inacc160",
        "alignatt_b1_top8_unit_consensus_consensus0p6",
        "alignatt_b1_top8_unit_consensus_consensus0p6_inacc160",
    ]
    assert sorted(
        {point["alignatt_unit_consensus_min_head_ratio"] for point in points}
    ) == [0.45, 0.60]


def test_policy_points_can_sweep_min_source_unit_modes():
    args = SimpleNamespace(
        skip_alignatt=False,
        skip_cutoffs=True,
        cutoffs=[],
        base_preset="gemma_low_latency",
        alignatt_border_margins=[1],
        alignatt_top_k_heads=[8],
        alignatt_acceptance_variants=["unit_consensus"],
        alignatt_min_source_masses=[0.0],
        alignatt_unit_consensus_min_head_ratio=0.60,
        alignatt_unit_consensus_min_head_ratios=[0.60],
        alignatt_online_normalization="zscore",
        alignatt_online_normalizations=None,
        alignatt_inaccessible_ms=0.0,
        alignatt_inaccessible_ms_values=[0.0],
        alignatt_frontier_min_inaccessible_mass=0.0,
        alignatt_frontier_min_inaccessible_masses=[0.0],
        alignatt_source_frontier_action="stop",
        alignatt_min_accessible_source_units_mode="block",
        alignatt_min_accessible_source_units_modes=["block", "target_unit_cap"],
        alignatt_source_regression_min_inaccessible_mass=0.0,
        alignatt_source_regression_min_inaccessible_masses=None,
        alignatt_source_regression_action="stop",
        alignatt_source_lcp_stability=False,
        alignatt_source_lcp_append_slack_units=0,
    )

    points = policy_points(args)

    assert [point["tag"] for point in points] == [
        "alignatt_b1_top8_unit_consensus_consensus0p6",
        "alignatt_b1_top8_unit_consensus_consensus0p6_srcunitcap",
    ]
    assert [
        point["alignatt_min_accessible_source_units_mode"] for point in points
    ] == ["block", "target_unit_cap"]


def test_policy_points_tag_target_unit_cap_with_unrecovered_trim():
    args = SimpleNamespace(
        skip_alignatt=False,
        skip_cutoffs=True,
        cutoffs=[],
        base_preset="gemma_low_latency",
        alignatt_border_margins=[1],
        alignatt_top_k_heads=[8],
        alignatt_acceptance_variants=["token"],
        alignatt_min_source_masses=[0.003],
        alignatt_unit_consensus_min_head_ratio=0.60,
        alignatt_unit_consensus_min_head_ratios=None,
        alignatt_online_normalization="zscore",
        alignatt_online_normalizations=None,
        alignatt_source_bearing_min_source_mass=0.005,
        alignatt_source_bearing_min_source_masses=None,
        alignatt_source_bearing_hard_inaccessible_cap=1.0,
        alignatt_source_bearing_hard_inaccessible_caps=None,
        alignatt_max_non_source_prompt_mass=1.0,
        alignatt_max_non_source_prompt_masses=None,
        alignatt_min_accepted_accessible_source_mass=0.0,
        alignatt_min_accepted_accessible_source_masses=None,
        alignatt_inaccessible_ms=0.0,
        alignatt_inaccessible_ms_values=[0.0],
        alignatt_frontier_min_inaccessible_mass=0.03,
        alignatt_frontier_min_inaccessible_masses=[0.03],
        alignatt_source_frontier_action="stop",
        alignatt_min_accessible_source_units_mode="block",
        alignatt_min_accessible_source_units_modes=["target_unit_cap"],
        alignatt_source_regression_min_inaccessible_mass=0.0,
        alignatt_source_regression_min_inaccessible_masses=None,
        alignatt_source_regression_action="trim_unrecovered",
        alignatt_source_lcp_stability=False,
        alignatt_source_lcp_append_slack_units=0,
    )

    point = policy_points(args)[0]

    assert point["tag"] == "alignatt_b1_top8_m0p003_frontier0p03_srcunitcap_srtrim_unrecovered"
    assert point["alignatt_min_accessible_source_units_mode"] == "target_unit_cap"
    assert point["alignatt_source_regression_action"] == "trim_unrecovered"


def test_policy_points_can_sweep_online_normalization():
    args = SimpleNamespace(
        skip_alignatt=False,
        skip_cutoffs=True,
        cutoffs=[],
        base_preset="gemma_low_latency",
        alignatt_border_margins=[1],
        alignatt_top_k_heads=[8],
        alignatt_acceptance_variants=["unit_consensus"],
        alignatt_min_source_masses=[0.0],
        alignatt_unit_consensus_min_head_ratio=0.60,
        alignatt_unit_consensus_min_head_ratios=[0.60],
        alignatt_online_normalization="zscore",
        alignatt_online_normalizations=["zscore", "raw"],
        alignatt_inaccessible_ms=0.0,
        alignatt_inaccessible_ms_values=[0.0],
        alignatt_frontier_min_inaccessible_mass=0.0,
        alignatt_frontier_min_inaccessible_masses=[0.0],
        alignatt_source_frontier_action="stop",
        alignatt_source_regression_min_inaccessible_mass=0.0,
        alignatt_source_regression_min_inaccessible_masses=None,
        alignatt_source_regression_action="stop",
        alignatt_source_lcp_stability=False,
        alignatt_source_lcp_append_slack_units=0,
    )

    points = policy_points(args)

    assert [point["tag"] for point in points] == [
        "alignatt_b1_top8_unit_consensus_consensus0p6_raw",
        "alignatt_b1_top8_unit_consensus_consensus0p6",
    ]
    assert [point["alignatt_online_normalization"] for point in points] == [
        "raw",
        "zscore",
    ]


def test_policy_points_can_sweep_source_bearing_unit_thresholds():
    args = SimpleNamespace(
        skip_alignatt=False,
        skip_cutoffs=True,
        cutoffs=[],
        base_preset="gemma_low_latency",
        alignatt_border_margins=[1],
        alignatt_top_k_heads=[8],
        alignatt_acceptance_variants=["unit_mass_source_bearing"],
        alignatt_min_source_masses=[0.0],
        alignatt_unit_consensus_min_head_ratio=0.60,
        alignatt_unit_consensus_min_head_ratios=None,
        alignatt_online_normalization="zscore",
        alignatt_online_normalizations=None,
        alignatt_source_bearing_min_source_mass=0.05,
        alignatt_source_bearing_min_source_masses=[0.03, 0.05],
        alignatt_source_bearing_hard_inaccessible_cap=0.75,
        alignatt_source_bearing_hard_inaccessible_caps=[0.60, 0.75, 1.0],
        alignatt_inaccessible_ms=0.0,
        alignatt_inaccessible_ms_values=[0.0],
        alignatt_frontier_min_inaccessible_mass=0.0,
        alignatt_frontier_min_inaccessible_masses=[0.0],
        alignatt_source_frontier_action="stop",
        alignatt_min_accessible_source_units_mode="block",
        alignatt_min_accessible_source_units_modes=None,
        alignatt_source_regression_min_inaccessible_mass=0.0,
        alignatt_source_regression_min_inaccessible_masses=None,
        alignatt_source_regression_action="stop",
        alignatt_source_lcp_stability=False,
        alignatt_source_lcp_append_slack_units=0,
    )

    points = policy_points(args)

    assert [point["tag"] for point in points] == [
        "alignatt_b1_top8_unit_mass_source_bearing_sb0p03_sbcap0p6",
        "alignatt_b1_top8_unit_mass_source_bearing_sb0p03_sbcap0p75",
        "alignatt_b1_top8_unit_mass_source_bearing_sb0p03",
        "alignatt_b1_top8_unit_mass_source_bearing_sb0p05_sbcap0p6",
        "alignatt_b1_top8_unit_mass_source_bearing_sb0p05_sbcap0p75",
        "alignatt_b1_top8_unit_mass_source_bearing_sb0p05",
    ]
    assert sorted(
        {point["alignatt_source_bearing_min_source_mass"] for point in points}
    ) == [0.03, 0.05]
    assert sorted(
        {point["alignatt_source_bearing_hard_inaccessible_cap"] for point in points}
    ) == [0.60, 0.75, 1.0]


def test_policy_points_can_sweep_non_source_prompt_cap():
    args = SimpleNamespace(
        skip_alignatt=False,
        skip_cutoffs=True,
        cutoffs=[],
        base_preset="gemma_low_latency",
        alignatt_border_margins=[1],
        alignatt_top_k_heads=[8],
        alignatt_acceptance_variants=["token"],
        alignatt_min_source_masses=[0.0],
        alignatt_unit_consensus_min_head_ratio=0.60,
        alignatt_unit_consensus_min_head_ratios=None,
        alignatt_online_normalization="zscore",
        alignatt_online_normalizations=None,
        alignatt_source_bearing_min_source_mass=0.05,
        alignatt_source_bearing_min_source_masses=None,
        alignatt_source_bearing_hard_inaccessible_cap=1.0,
        alignatt_source_bearing_hard_inaccessible_caps=None,
        alignatt_max_non_source_prompt_mass=1.0,
        alignatt_max_non_source_prompt_masses=[0.80, 1.0],
        alignatt_inaccessible_ms=0.0,
        alignatt_inaccessible_ms_values=[0.0],
        alignatt_frontier_min_inaccessible_mass=0.0,
        alignatt_frontier_min_inaccessible_masses=[0.0],
        alignatt_source_frontier_action="stop",
        alignatt_min_accessible_source_units_mode="block",
        alignatt_min_accessible_source_units_modes=None,
        alignatt_source_regression_min_inaccessible_mass=0.0,
        alignatt_source_regression_min_inaccessible_masses=None,
        alignatt_source_regression_action="stop",
        alignatt_source_lcp_stability=False,
        alignatt_source_lcp_append_slack_units=0,
    )

    points = policy_points(args)

    assert [point["tag"] for point in points] == [
        "alignatt_b1_top8_nsp0p8",
        "alignatt_b1_top8",
    ]
    assert [point["alignatt_max_non_source_prompt_mass"] for point in points] == [
        0.80,
        1.0,
    ]


def test_policy_points_can_sweep_accepted_prefix_source_mass_floor():
    args = SimpleNamespace(
        skip_alignatt=False,
        skip_cutoffs=True,
        cutoffs=[],
        base_preset="gemma_low_latency",
        alignatt_border_margins=[1],
        alignatt_top_k_heads=[8],
        alignatt_acceptance_variants=["token"],
        alignatt_min_source_masses=[0.0],
        alignatt_unit_consensus_min_head_ratio=0.60,
        alignatt_unit_consensus_min_head_ratios=None,
        alignatt_online_normalization="zscore",
        alignatt_online_normalizations=None,
        alignatt_source_bearing_min_source_mass=0.005,
        alignatt_source_bearing_min_source_masses=None,
        alignatt_source_bearing_hard_inaccessible_cap=1.0,
        alignatt_source_bearing_hard_inaccessible_caps=None,
        alignatt_max_non_source_prompt_mass=1.0,
        alignatt_max_non_source_prompt_masses=None,
        alignatt_min_accepted_accessible_source_mass=0.0,
        alignatt_min_accepted_accessible_source_masses=[0.0, 0.0005, 0.001],
        alignatt_inaccessible_ms=0.0,
        alignatt_inaccessible_ms_values=[0.0],
        alignatt_frontier_min_inaccessible_mass=0.03,
        alignatt_frontier_min_inaccessible_masses=[0.03],
        alignatt_source_frontier_action="stop",
        alignatt_min_accessible_source_units_mode="block",
        alignatt_min_accessible_source_units_modes=None,
        alignatt_source_regression_min_inaccessible_mass=0.0,
        alignatt_source_regression_min_inaccessible_masses=None,
        alignatt_source_regression_action="stop",
        alignatt_source_lcp_stability=False,
        alignatt_source_lcp_append_slack_units=0,
    )

    points = policy_points(args)

    assert [point["tag"] for point in points] == [
        "alignatt_b1_top8_frontier0p03",
        "alignatt_b1_top8_apm0p0005_frontier0p03",
        "alignatt_b1_top8_apm0p001_frontier0p03",
    ]
    assert [
        point["alignatt_min_accepted_accessible_source_mass"] for point in points
    ] == [0.0, 0.0005, 0.001]


def test_source_bearing_axes_only_expand_source_bearing_variant():
    args = SimpleNamespace(
        skip_alignatt=False,
        skip_cutoffs=True,
        cutoffs=[],
        base_preset="gemma_low_latency",
        alignatt_border_margins=[1],
        alignatt_top_k_heads=[8],
        alignatt_acceptance_variants=["token", "unit_mass_source_bearing"],
        alignatt_min_source_masses=[0.0],
        alignatt_unit_consensus_min_head_ratio=0.60,
        alignatt_unit_consensus_min_head_ratios=None,
        alignatt_online_normalization="zscore",
        alignatt_online_normalizations=None,
        alignatt_source_bearing_min_source_mass=0.05,
        alignatt_source_bearing_min_source_masses=[0.03, 0.05],
        alignatt_source_bearing_hard_inaccessible_cap=0.75,
        alignatt_source_bearing_hard_inaccessible_caps=[0.60, 0.75, 1.0],
        alignatt_inaccessible_ms=0.0,
        alignatt_inaccessible_ms_values=[0.0],
        alignatt_frontier_min_inaccessible_mass=0.0,
        alignatt_frontier_min_inaccessible_masses=[0.0],
        alignatt_source_frontier_action="stop",
        alignatt_min_accessible_source_units_mode="block",
        alignatt_min_accessible_source_units_modes=None,
        alignatt_source_regression_min_inaccessible_mass=0.0,
        alignatt_source_regression_min_inaccessible_masses=None,
        alignatt_source_regression_action="stop",
        alignatt_source_lcp_stability=False,
        alignatt_source_lcp_append_slack_units=0,
    )

    points = policy_points(args)

    assert [point["tag"] for point in points] == [
        "alignatt_b1_top8",
        "alignatt_b1_top8_unit_mass_source_bearing_sb0p03_sbcap0p6",
        "alignatt_b1_top8_unit_mass_source_bearing_sb0p03_sbcap0p75",
        "alignatt_b1_top8_unit_mass_source_bearing_sb0p03",
        "alignatt_b1_top8_unit_mass_source_bearing_sb0p05_sbcap0p6",
        "alignatt_b1_top8_unit_mass_source_bearing_sb0p05_sbcap0p75",
        "alignatt_b1_top8_unit_mass_source_bearing_sb0p05",
    ]
