from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.plot_enzh_quality_latency_tradeoff import (
    TradeoffPoint,
    alignatt_same_chunk_permissiveness_summary,
    baseline_anchor_dominance_summary,
    baseline_cu_for_chunk_ms,
    baseline_equivalent_segment_ms,
    baseline_points,
    classify_alignatt_policy,
    format_stop_reason_counts,
    summarize_alignatt_stream_updates,
    write_gap_table,
)


def test_baseline_latency_helpers_use_public_no_context_points():
    baseline = baseline_points("baseline")

    assert baseline_cu_for_chunk_ms(baseline, 640) == 1760.0
    assert baseline_equivalent_segment_ms(baseline, 2680.0) == 960.0
    assert baseline_equivalent_segment_ms(baseline, 3450.0) == 1280.0


def test_gap_table_flags_alignatt_that_is_not_more_permissive(tmp_path: Path):
    baseline = baseline_points("baseline")
    context = baseline_points("with_context")
    gap_path = tmp_path / "gap.tsv"
    point = TradeoffPoint(
        system="MiLMMT AlignAtt",
        label="chunk640_slow",
        longyaal_cu_ms=2296.104,
        xcometxl=76.149,
        chunk_ms=640,
    )

    write_gap_table(
        gap_path,
        points=[point],
        baseline=baseline,
        context_baseline=context,
    )

    with gap_path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))

    assert row["public_baseline_same_chunk_cu_ms"] == "1760.000"
    assert row["alignatt_more_permissive_than_same_chunk_baseline"] == "false"
    assert float(row["delta_cu_vs_public_same_chunk_ms"]) > 0.0
    assert float(row["public_baseline_latency_equivalent_segment_ms"]) > 640.0


def test_dominance_summary_requires_beating_each_public_anchor_latency():
    baseline = baseline_points("baseline")
    points = [
        TradeoffPoint(
            system="MiLMMT AlignAtt",
            label="slow_640",
            longyaal_cu_ms=2296.0,
            xcometxl=76.0,
            chunk_ms=640,
        ),
        TradeoffPoint(
            system="MiLMMT AlignAtt",
            label="better_960",
            longyaal_cu_ms=2600.0,
            xcometxl=78.5,
            chunk_ms=960,
        ),
    ]

    dominance = baseline_anchor_dominance_summary(points=points, baseline=baseline)
    permissiveness = alignatt_same_chunk_permissiveness_summary(
        points=points,
        baseline=baseline,
    )

    assert dominance["all_public_baseline_anchors_dominated"] is False
    assert dominance["anchors"][0]["segment_ms"] == 640
    assert dominance["anchors"][0]["dominates"] is False
    assert dominance["anchors"][1]["segment_ms"] == 960
    assert dominance["anchors"][1]["dominates"] is True
    assert permissiveness["all_checked_alignatt_points_more_permissive"] is False
    assert permissiveness["checks"][0]["more_permissive"] is False


def test_gap_table_reports_alignatt_blocking_diagnostics(tmp_path: Path):
    artifact_dir = tmp_path / "run"
    artifact_dir.mkdir()
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "runtime_config": {
                    "translation_acceptance_policy": "alignatt",
                    "translation_alignatt_min_accessible_source_units": 6,
                    "translation_alignatt_max_inaccessible_source_mass": 0.15,
                    "translation_alignatt_min_accessible_inaccessible_margin": 0.0,
                    "translation_alignatt_max_source_regression": 1,
                    "translation_alignatt_source_regression_activation_mode": "frontier_reached",
                    "translation_alignatt_source_regression_activation_slack_tokens": 4,
                    "translation_alignatt_source_regression_min_inaccessible_mass": 0.03,
                    "translation_alignatt_source_regression_patience_tokens": 2,
                    "translation_alignatt_token_argmax_frontier_gate": True,
                    "translation_alignatt_token_argmax_frontier_patience_tokens": 2,
                    "asr_punctuation_min_commit_words": 4,
                }
            }
        ),
        encoding="utf-8",
    )
    stream_rows = [
        {
            "alignatt_metadata": {
                "stop_reason": "alignatt:source_regression",
                "accepted_token_count": 2,
            }
        },
        {
            "alignatt_metadata": {
                "stop_reason": "alignatt:source_regression",
                "accepted_token_count": 0,
            }
        },
        {
            "alignatt_metadata": {
                "unsafe_reason": "provenance_weak",
                "accepted_token_count": 0,
            }
        },
    ]
    (artifact_dir / "stream_updates.jsonl").write_text(
        "\n".join(json.dumps(row) for row in stream_rows) + "\n",
        encoding="utf-8",
    )
    point = TradeoffPoint(
        system="MiLMMT AlignAtt",
        label="chunk640_guarded",
        longyaal_cu_ms=2000.0,
        xcometxl=76.0,
        chunk_ms=640,
        manifest_dir=str(artifact_dir),
    )
    gap_path = tmp_path / "gap.tsv"

    write_gap_table(
        gap_path,
        points=[point],
        baseline=baseline_points("baseline"),
        context_baseline=baseline_points("with_context"),
    )

    diagnostics = summarize_alignatt_stream_updates(artifact_dir)
    assert diagnostics.update_count == 3
    assert diagnostics.zero_accept_update_count == 2
    assert format_stop_reason_counts(diagnostics.stop_reason_counts).startswith(
        "alignatt:source_regression=2"
    )
    with gap_path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))
    assert row["alignatt_update_count"] == "3"
    assert row["alignatt_zero_accept_update_count"] == "2"
    assert row["alignatt_stop_reason_counts"] == (
        "alignatt:source_regression=2,provenance_weak=1"
    )
    assert row["alignatt_policy_class"] == "guarded_alignatt"
    assert row["alignatt_guard_flags"] == (
        "min_accessible_source_units=6:block,max_inaccessible_source_mass=0.15,"
        "accessible_inaccessible_margin=0.0,"
        "source_regression=1:frontier_reached+slack4+future0.03+patience2,"
        "token_argmax_frontier_gate:patience2,asr_punctuation_min_commit_words=4"
    )


def test_recoverable_source_frontier_is_clean_explicit_variant(tmp_path: Path):
    artifact_dir = tmp_path / "run"
    artifact_dir.mkdir()
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "runtime_config": {
                    "translation_acceptance_policy": "alignatt",
                    "translation_alignatt_acceptance_variant": "token",
                    "translation_alignatt_frontier_min_inaccessible_mass": 0.03,
                    "translation_alignatt_source_frontier_action": "trim_unrecovered",
                    "translation_alignatt_min_source_mass": 0.0,
                    "translation_alignatt_max_source_regression": -1,
                    "translation_alignatt_token_argmax_frontier_gate": False,
                }
            }
        ),
        encoding="utf-8",
    )

    classification = classify_alignatt_policy(artifact_dir)

    assert classification.policy_class == "clean_recoverable_soft_frontier"
    assert classification.guard_flags == ()


def test_gap_table_prefers_chunk_decision_diagnostics(tmp_path: Path):
    artifact_dir = tmp_path / "run"
    artifact_dir.mkdir()
    (artifact_dir / "manifest.json").write_text(
        json.dumps({"runtime_config": {"translation_acceptance_policy": "alignatt"}}),
        encoding="utf-8",
    )
    (artifact_dir / "stream_updates.jsonl").write_text(
        json.dumps(
            {
                "alignatt_metadata": {
                    "stop_reason": "stop",
                    "accepted_token_count": 3,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    chunk_decisions = [
        {
            "emitted": False,
            "alignatt_metadata_current_chunk": True,
            "alignatt_decision": {
                "stop_reason": "alignatt:source_frontier",
                "accepted_token_count": 0,
            },
        },
        {
            "emitted": True,
            "alignatt_metadata_current_chunk": True,
            "alignatt_decision": {
                "stop_reason": "stop",
                "accepted_token_count": 4,
            },
        },
        {
            "emitted": False,
            "alignatt_metadata_current_chunk": False,
            "alignatt_decision": {
                "stop_reason": "stale",
                "accepted_token_count": 0,
            },
        },
    ]
    (artifact_dir / "chunk_decisions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in chunk_decisions) + "\n",
        encoding="utf-8",
    )
    gap_path = tmp_path / "gap.tsv"

    write_gap_table(
        gap_path,
        points=[
            TradeoffPoint(
                system="MiLMMT AlignAtt",
                label="chunk640",
                longyaal_cu_ms=1500.0,
                xcometxl=76.0,
                chunk_ms=640,
                manifest_dir=str(artifact_dir),
            )
        ],
        baseline=baseline_points("baseline"),
        context_baseline=baseline_points("with_context"),
    )

    diagnostics = summarize_alignatt_stream_updates(artifact_dir)
    assert diagnostics.diagnostic_source == "chunk_decisions"
    assert diagnostics.chunk_count == 3
    assert diagnostics.emitted_chunk_count == 1
    assert diagnostics.update_count == 2
    assert diagnostics.zero_accept_update_count == 1
    assert diagnostics.zero_emit_current_mt_decision_count == 1
    with gap_path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))
    assert row["alignatt_diagnostic_source"] == "chunk_decisions"
    assert row["alignatt_chunk_count"] == "3"
    assert row["alignatt_emitted_chunk_count"] == "1"
    assert row["alignatt_update_count"] == "2"
    assert row["alignatt_zero_emit_current_mt_decision_count"] == "1"
    assert row["alignatt_stop_reason_counts"] == (
        "alignatt:source_frontier=1,stop=1"
    )


def test_alignatt_policy_classification_separates_clean_and_guarded(tmp_path: Path):
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    (clean_dir / "manifest.json").write_text(
        json.dumps(
            {
                "runtime_config": {
                    "translation_acceptance_policy": "alignatt",
                    "translation_alignatt_acceptance_variant": "token",
                    "translation_alignatt_frontier_min_inaccessible_mass": 0.03,
                    "translation_alignatt_max_inaccessible_source_mass": 1.0,
                    "translation_alignatt_min_accessible_inaccessible_margin": -1.0,
                    "translation_alignatt_min_accessible_source_units": 0,
                    "translation_alignatt_max_source_regression": -1,
                    "translation_alignatt_token_argmax_frontier_gate": False,
                    "asr_punctuation_min_commit_words": 0,
                }
            }
        ),
        encoding="utf-8",
    )

    guarded_dir = tmp_path / "guarded"
    guarded_dir.mkdir()
    (guarded_dir / "manifest.json").write_text(
        json.dumps(
            {
                "runtime_config": {
                    "translation_acceptance_policy": "alignatt",
                    "translation_alignatt_acceptance_variant": "unit_mass",
                    "translation_alignatt_source_lcp_append_slack_units": 2,
                    "translation_alignatt_min_accepted_accessible_source_mass": 0.1,
                }
            }
        ),
        encoding="utf-8",
    )
    unit_mass_dir = tmp_path / "unit_mass"
    unit_mass_dir.mkdir()
    (unit_mass_dir / "manifest.json").write_text(
        json.dumps(
            {
                "runtime_config": {
                    "translation_acceptance_policy": "alignatt",
                    "translation_alignatt_acceptance_variant": "unit_mass",
                    "translation_alignatt_min_source_mass": 0.001,
                    "translation_alignatt_max_source_regression": -1,
                    "translation_alignatt_token_argmax_frontier_gate": False,
                }
            }
        ),
        encoding="utf-8",
    )
    unit_mass_zero_dir = tmp_path / "unit_mass_zero"
    unit_mass_zero_dir.mkdir()
    (unit_mass_zero_dir / "manifest.json").write_text(
        json.dumps(
            {
                "runtime_config": {
                    "translation_acceptance_policy": "alignatt",
                    "translation_alignatt_acceptance_variant": "unit_mass",
                    "translation_alignatt_min_source_mass": 0.0,
                    "translation_alignatt_max_source_regression": -1,
                    "translation_alignatt_token_argmax_frontier_gate": False,
                }
            }
        ),
        encoding="utf-8",
    )
    source_bearing_dir = tmp_path / "source_bearing"
    source_bearing_dir.mkdir()
    (source_bearing_dir / "manifest.json").write_text(
        json.dumps(
            {
                "runtime_config": {
                    "translation_acceptance_policy": "alignatt",
                    "translation_alignatt_acceptance_variant": (
                        "unit_mass_source_bearing"
                    ),
                    "translation_alignatt_min_source_mass": 0.0,
                    "translation_alignatt_source_bearing_min_source_mass": 0.005,
                    "translation_alignatt_source_bearing_hard_inaccessible_cap": 1.0,
                    "translation_alignatt_max_source_regression": -1,
                    "translation_alignatt_token_argmax_frontier_gate": False,
                }
            }
        ),
        encoding="utf-8",
    )
    source_bearing_guarded_dir = tmp_path / "source_bearing_guarded"
    source_bearing_guarded_dir.mkdir()
    (source_bearing_guarded_dir / "manifest.json").write_text(
        json.dumps(
            {
                "runtime_config": {
                    "translation_acceptance_policy": "alignatt",
                    "translation_alignatt_acceptance_variant": (
                        "unit_mass_source_bearing"
                    ),
                    "translation_alignatt_min_source_mass": 0.0,
                    "translation_alignatt_source_bearing_min_source_mass": 0.005,
                    "translation_alignatt_source_bearing_hard_inaccessible_cap": 0.75,
                    "translation_alignatt_max_source_regression": -1,
                    "translation_alignatt_token_argmax_frontier_gate": False,
                }
            }
        ),
        encoding="utf-8",
    )
    non_source_guarded_dir = tmp_path / "non_source_guarded"
    non_source_guarded_dir.mkdir()
    (non_source_guarded_dir / "manifest.json").write_text(
        json.dumps(
            {
                "runtime_config": {
                    "translation_acceptance_policy": "alignatt",
                    "translation_alignatt_acceptance_variant": "token",
                    "translation_alignatt_min_source_mass": 0.0,
                    "translation_alignatt_max_non_source_prompt_mass": 0.80,
                    "translation_alignatt_max_source_regression": -1,
                    "translation_alignatt_token_argmax_frontier_gate": False,
                }
            }
        ),
        encoding="utf-8",
    )
    unit_argmax_dir = tmp_path / "unit_argmax"
    unit_argmax_dir.mkdir()
    (unit_argmax_dir / "manifest.json").write_text(
        json.dumps(
            {
                "runtime_config": {
                    "translation_acceptance_policy": "alignatt",
                    "translation_alignatt_acceptance_variant": "unit_argmax",
                    "translation_alignatt_min_source_mass": 0.0,
                    "translation_alignatt_max_source_regression": -1,
                    "translation_alignatt_token_argmax_frontier_gate": False,
                }
            }
        ),
        encoding="utf-8",
    )
    unit_consensus_dir = tmp_path / "unit_consensus"
    unit_consensus_dir.mkdir()
    (unit_consensus_dir / "manifest.json").write_text(
        json.dumps(
            {
                "runtime_config": {
                    "translation_acceptance_policy": "alignatt",
                    "translation_alignatt_acceptance_variant": "unit_consensus",
                    "translation_alignatt_min_source_mass": 0.0,
                    "translation_alignatt_unit_consensus_min_head_ratio": 0.45,
                    "translation_alignatt_max_source_regression": -1,
                    "translation_alignatt_token_argmax_frontier_gate": False,
                }
            }
        ),
        encoding="utf-8",
    )
    unit_argmax_floor_dir = tmp_path / "unit_argmax_floor"
    unit_argmax_floor_dir.mkdir()
    (unit_argmax_floor_dir / "manifest.json").write_text(
        json.dumps(
            {
                "runtime_config": {
                    "translation_acceptance_policy": "alignatt",
                    "translation_alignatt_acceptance_variant": "unit_argmax",
                    "translation_alignatt_min_source_mass": 0.001,
                    "translation_alignatt_max_source_regression": -1,
                    "translation_alignatt_token_argmax_frontier_gate": False,
                }
            }
        ),
        encoding="utf-8",
    )

    clean = classify_alignatt_policy(clean_dir)
    guarded = classify_alignatt_policy(guarded_dir)
    unit_mass = classify_alignatt_policy(unit_mass_dir)
    unit_mass_zero = classify_alignatt_policy(unit_mass_zero_dir)
    source_bearing = classify_alignatt_policy(source_bearing_dir)
    source_bearing_guarded = classify_alignatt_policy(source_bearing_guarded_dir)
    non_source_guarded = classify_alignatt_policy(non_source_guarded_dir)
    unit_argmax = classify_alignatt_policy(unit_argmax_dir)
    unit_consensus = classify_alignatt_policy(unit_consensus_dir)
    unit_argmax_floor = classify_alignatt_policy(unit_argmax_floor_dir)

    assert clean.policy_class == "pure_soft_frontier"
    assert clean.guard_flags == ()
    assert guarded.policy_class == "guarded_alignatt"
    assert guarded.guard_flags == (
        "acceptance_variant=unit_mass_without_source_mass_floor",
        "source_lcp_append_slack_units=2",
    )
    assert unit_mass.policy_class == "clean_unit_source_mass_floor"
    assert unit_mass.guard_flags == ()
    assert unit_mass_zero.policy_class == "guarded_alignatt"
    assert unit_mass_zero.guard_flags == (
        "acceptance_variant=unit_mass_without_source_mass_floor",
    )
    assert source_bearing.policy_class == "clean_unit_source_bearing"
    assert source_bearing.guard_flags == ()
    assert source_bearing_guarded.policy_class == "guarded_alignatt"
    assert source_bearing_guarded.guard_flags == (
        "source_bearing_hard_inaccessible_cap=0.75",
    )
    assert non_source_guarded.policy_class == "guarded_alignatt"
    assert non_source_guarded.guard_flags == ("max_non_source_prompt_mass=0.8",)
    assert unit_argmax.policy_class == "clean_unit_argmax_frontier"
    assert unit_argmax.guard_flags == ()
    assert unit_consensus.policy_class == "clean_unit_consensus_frontier"
    assert unit_consensus.guard_flags == ()
    assert unit_argmax_floor.policy_class == "guarded_alignatt"
    assert unit_argmax_floor.guard_flags == ("unused_source_mass_floor",)


def test_accepted_prefix_source_mass_floor_is_clean_token_source_mass_axis(
    tmp_path: Path,
):
    run_dir = tmp_path / "accepted_prefix_source_mass"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "runtime_config": {
                    "translation_acceptance_policy": "alignatt",
                    "translation_alignatt_acceptance_variant": "token",
                    "translation_alignatt_min_source_mass": 0.0,
                    "translation_alignatt_min_accepted_accessible_source_mass": 0.001,
                    "translation_alignatt_accepted_accessible_source_mass_recent_units": 2,
                    "translation_alignatt_frontier_min_inaccessible_mass": 0.03,
                    "translation_alignatt_max_inaccessible_source_mass": 1.0,
                    "translation_alignatt_min_accessible_inaccessible_margin": -1.0,
                    "translation_alignatt_min_accessible_source_units": 0,
                    "translation_alignatt_max_source_regression": -1,
                    "translation_alignatt_token_argmax_frontier_gate": False,
                }
            }
        ),
        encoding="utf-8",
    )

    classification = classify_alignatt_policy(run_dir)

    assert classification.policy_class == "clean_soft_frontier_source_mass_floor"
    assert classification.guard_flags == ()


def test_surface_dedup_named_runs_are_guarded_diagnostics(tmp_path: Path):
    run_dir = tmp_path / "enzh_milmmt_chunk640_surface_dedup_smoke"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "runtime_config": {
                    "translation_acceptance_policy": "alignatt",
                    "translation_alignatt_acceptance_variant": "token",
                    "translation_alignatt_frontier_min_inaccessible_mass": 0.03,
                    "translation_alignatt_max_inaccessible_source_mass": 1.0,
                    "translation_alignatt_min_accessible_inaccessible_margin": -1.0,
                    "translation_alignatt_min_accessible_source_units": 0,
                    "translation_alignatt_max_source_regression": -1,
                    "translation_alignatt_token_argmax_frontier_gate": False,
                }
            }
        ),
        encoding="utf-8",
    )

    classification = classify_alignatt_policy(run_dir)

    assert classification.policy_class == "guarded_alignatt"
    assert classification.guard_flags == ("surface_dedup_diagnostic",)


def test_provenance_cap_alignatt_is_guarded_not_clean(tmp_path: Path):
    capped_dir = tmp_path / "capped"
    capped_dir.mkdir()
    (capped_dir / "manifest.json").write_text(
        json.dumps(
            {
                "runtime_config": {
                    "translation_acceptance_policy": "alignatt",
                    "translation_alignatt_acceptance_variant": "token",
                    "translation_alignatt_min_source_mass": 0.003,
                    "translation_alignatt_frontier_min_inaccessible_mass": 0.03,
                    "translation_alignatt_max_inaccessible_source_mass": 0.15,
                    "translation_alignatt_min_accessible_inaccessible_margin": -1.0,
                    "translation_alignatt_min_accessible_source_units": 0,
                    "translation_alignatt_max_source_regression": -1,
                    "translation_alignatt_token_argmax_frontier_gate": False,
                    "asr_punctuation_min_commit_words": 0,
                }
            }
        ),
        encoding="utf-8",
    )

    capped = classify_alignatt_policy(capped_dir)

    assert capped.policy_class == "guarded_alignatt"
    assert capped.guard_flags == ("max_inaccessible_source_mass=0.15",)
