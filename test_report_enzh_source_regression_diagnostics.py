from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.report_enzh_source_regression_diagnostics import (
    analyze_run,
    gate_aware_trim_unrecovered_accept_count,
    report_rows,
    source_regression_reference_position,
    source_regression_trim_unrecovered_accept_count,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_source_regression_reference_position_uses_recent_median():
    assert source_regression_reference_position(
        [1, 9, 7, 6],
        max_accepted_position=9,
        recent_tokens=3,
        reference_mode="median_recent",
    ) == 7
    assert source_regression_reference_position(
        [1, 9, 7, 6],
        max_accepted_position=9,
        recent_tokens=3,
        reference_mode="max",
    ) == 9


def test_trim_unrecovered_simulation_keeps_recovered_suffix_only():
    assert (
        source_regression_trim_unrecovered_accept_count(
            [0, 5, 8, 2, 4, 9],
            accepted_count=3,
            max_regression=0,
            recent_tokens=0,
            reference_mode="max",
            patience_tokens=1,
        )
        == 6
    )
    assert (
        source_regression_trim_unrecovered_accept_count(
            [0, 5, 8, 2, 4],
            accepted_count=3,
            max_regression=0,
            recent_tokens=0,
            reference_mode="max",
            patience_tokens=1,
        )
        == 3
    )


def test_gate_aware_trim_unrecovered_stops_on_other_alignatt_gates():
    accept_count, blocked_by_other_gate = gate_aware_trim_unrecovered_accept_count(
        [0, 5, 8, 2, 4, 9],
        provenance=[
            {},
            {},
            {},
            {"source_accessible": 0.1, "source_inaccessible": 0.0},
            {"source_accessible": 0.1, "source_inaccessible": 0.0},
            {"source_accessible": 0.1, "source_inaccessible": 0.2},
        ],
        metadata={"accessible_source_local_end_exclusive": 8},
        runtime_config={
            "translation_alignatt_border_margin": 1,
            "translation_alignatt_frontier_min_inaccessible_mass": 0.03,
            "translation_alignatt_token_argmax_frontier_gate": False,
            "translation_alignatt_min_source_mass": 0.0,
            "translation_alignatt_max_inaccessible_source_mass": 1.0,
            "translation_alignatt_min_accessible_inaccessible_margin": -1.0,
        },
        accepted_count=3,
        max_regression=0,
        recent_tokens=0,
        reference_mode="max",
        patience_tokens=1,
    )

    assert accept_count == 5
    assert blocked_by_other_gate is True


def test_analyze_run_counts_future_recovery_after_hard_stop(tmp_path):
    run_dir = tmp_path / "run"
    write_json(
        run_dir / "manifest.json",
        {
            "num_inputs": 3,
            "runtime_config": {
                "translation_alignatt_max_source_regression": 0,
                "translation_alignatt_source_regression_recent_tokens": 0,
                "translation_alignatt_source_regression_reference_mode": "max",
                "translation_alignatt_source_regression_action": "stop",
                "translation_alignatt_source_regression_patience_tokens": 1,
                "translation_alignatt_border_margin": 1,
                "translation_alignatt_frontier_min_inaccessible_mass": 0.03,
                "translation_alignatt_token_argmax_frontier_gate": False,
                "translation_alignatt_min_source_mass": 0.0,
                "translation_alignatt_max_inaccessible_source_mass": 1.0,
                "translation_alignatt_min_accessible_inaccessible_margin": -1.0,
            },
        },
    )
    write_jsonl(
        run_dir / "stream_updates.jsonl",
        [
            {
                "alignatt_metadata": {
                    "stop_reason": "alignatt:source_regression",
                    "accepted_candidate_token_count": 3,
                    "unsafe_target_token_index": 3,
                    "blocked_source_local_position": 2,
                    "accessible_source_local_end_exclusive": 12,
                    "aligned_source_local_positions": [0, 5, 8, 2, 4, 9],
                    "provenance_per_draft_token": [
                        {},
                        {},
                        {},
                        {"source_accessible": 0.1, "source_inaccessible": 0.0},
                        {"source_accessible": 0.1, "source_inaccessible": 0.0},
                        {"source_accessible": 0.1, "source_inaccessible": 0.0},
                    ],
                }
            },
            {
                "alignatt_metadata": {
                    "stop_reason": "alignatt:token_argmax_source_frontier",
                    "accepted_candidate_token_count": 1,
                    "aligned_source_local_positions": [0, 1],
                }
            },
        ],
    )

    row = analyze_run(
        {
            "relative_dir": "run",
            "copied_dir": str(run_dir),
            "valid_for_claims": "True",
            "target_language_code": "zh",
            "mt_backend_name": "milmmt_vllm_alignatt",
            "num_inputs": "3",
            "chunk_ms": "640",
            "alignatt_guard_flags": "source_regression",
            "longyaal_cu_ms": "2000.0",
            "xcometxl": "0.75",
        }
    )

    assert row is not None
    assert row["source_regression_stop_count"] == 1
    assert row["source_regression_stop_ratio"] == pytest.approx(0.5)
    assert row["mean_extra_draft_tokens_after_stop"] == pytest.approx(3.0)
    assert row["updates_with_future_recovery"] == 1
    assert row["future_recovery_ratio"] == pytest.approx(1.0)
    assert row["mean_future_recovery_token_gap"] == pytest.approx(2.0)
    assert row["simulated_trim_unrecovered_gain_update_count"] == 1
    assert row["simulated_trim_unrecovered_token_gain_sum"] == 3
    assert row["mean_simulated_trim_unrecovered_token_gain"] == pytest.approx(3.0)
    assert row["simulated_trim_unrecovered_full_draft_count"] == 1
    assert row["simulated_trim_unrecovered_unrecovered_suffix_count"] == 0
    assert row["simulated_gate_aware_gain_update_count"] == 1
    assert row["simulated_gate_aware_token_gain_sum"] == 3
    assert row["mean_simulated_gate_aware_token_gain"] == pytest.approx(3.0)
    assert row["simulated_gate_aware_blocked_by_other_gate_count"] == 0
    assert row["mean_reference_position"] == pytest.approx(8.0)
    assert row["mean_blocked_position"] == pytest.approx(2.0)


def test_report_rows_filters_to_enzh_milmmt_runs(tmp_path):
    run_dir = tmp_path / "run"
    write_json(
        run_dir / "manifest.json",
        {
            "runtime_config": {
                "translation_alignatt_max_source_regression": 1,
            },
        },
    )
    write_jsonl(
        run_dir / "stream_updates.jsonl",
        [
            {
                "alignatt_metadata": {
                    "stop_reason": "alignatt:source_regression",
                    "accepted_candidate_token_count": 1,
                    "unsafe_target_token_index": 1,
                    "blocked_source_local_position": 0,
                    "aligned_source_local_positions": [3, 0, 4],
                }
            }
        ],
    )
    rows = report_rows(
        [
            {
                "relative_dir": "run",
                "copied_dir": str(run_dir),
                "target_language_code": "zh",
                "mt_backend_name": "milmmt_vllm_alignatt",
            },
            {
                "relative_dir": "ende",
                "copied_dir": str(run_dir),
                "target_language_code": "de",
                "mt_backend_name": "milmmt_vllm_alignatt",
            },
        ]
    )

    assert [row["relative_dir"] for row in rows] == ["run"]
