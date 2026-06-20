from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.report_enzh_token_argmax_frontier_diagnostics import (
    analyze_run,
    recoverable_token_argmax_accept_count,
    report_rows,
    simulate_token_argmax_patience_accept_count,
    token_argmax_frontier_blocks,
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


def runtime_config() -> dict:
    return {
        "translation_alignatt_token_argmax_frontier_gate": True,
        "translation_alignatt_token_argmax_min_source_mass": 0.05,
        "translation_alignatt_token_argmax_frontier_margin": 0,
        "translation_alignatt_token_argmax_frontier_patience_tokens": 1,
        "translation_alignatt_border_margin": 1,
        "translation_alignatt_frontier_min_inaccessible_mass": 0.03,
        "translation_alignatt_min_source_mass": 0.003,
        "translation_alignatt_max_inaccessible_source_mass": 0.15,
        "translation_alignatt_max_non_source_prompt_mass": 1.0,
        "translation_alignatt_min_accessible_inaccessible_margin": -1.0,
    }


def test_token_argmax_frontier_blocks_only_future_source_with_enough_mass():
    metadata = {"accessible_source_local_end_exclusive": 4}
    config = runtime_config()

    assert token_argmax_frontier_blocks(
        position=4,
        provenance={"source_accessible": 0.04, "source_inaccessible": 0.02},
        metadata=metadata,
        runtime_config=config,
    )
    assert not token_argmax_frontier_blocks(
        position=4,
        provenance={"source_accessible": 0.02, "source_inaccessible": 0.01},
        metadata=metadata,
        runtime_config=config,
    )
    assert not token_argmax_frontier_blocks(
        position=3,
        provenance={"source_accessible": 0.08, "source_inaccessible": 0.01},
        metadata=metadata,
        runtime_config=config,
    )


def test_patience_simulation_uses_existing_frontier_patience_semantics():
    metadata = {"accessible_source_local_end_exclusive": 4}
    config = runtime_config()
    provenance = [
        {},
        {"source_accessible": 0.04, "source_inaccessible": 0.02},
        {"source_accessible": 0.004, "source_inaccessible": 0.001},
        {"source_accessible": 0.08, "source_inaccessible": 0.0},
    ]

    accept_count, blocked_by_other_gate = simulate_token_argmax_patience_accept_count(
        [0, 6, 0, 1],
        provenance=provenance,
        metadata=metadata,
        runtime_config=config,
        accepted_count=1,
        patience_tokens=2,
    )

    assert accept_count == 4
    assert blocked_by_other_gate is False


def test_recoverable_simulation_trims_unrecovered_frontier_suffix():
    metadata = {"accessible_source_local_end_exclusive": 4}
    config = runtime_config()
    provenance = [
        {},
        {"source_accessible": 0.04, "source_inaccessible": 0.02},
        {"source_accessible": 0.07, "source_inaccessible": 0.01},
    ]

    accept_count, blocked_by_other_gate = recoverable_token_argmax_accept_count(
        [0, 6, 7],
        provenance=provenance,
        metadata=metadata,
        runtime_config=config,
        accepted_count=1,
    )

    assert accept_count == 1
    assert blocked_by_other_gate is False


def test_analyze_run_counts_recovery_and_simulated_gain(tmp_path):
    run_dir = tmp_path / "run"
    write_json(
        run_dir / "manifest.json",
        {
            "num_inputs": 3,
            "runtime_config": runtime_config(),
        },
    )
    write_jsonl(
        run_dir / "stream_updates.jsonl",
        [
            {
                "alignatt_metadata": {
                    "stop_reason": "alignatt:token_argmax_source_frontier",
                    "accepted_candidate_token_count": 1,
                    "unsafe_target_token_index": 1,
                    "blocked_source_local_position": 6,
                    "accessible_source_local_end_exclusive": 4,
                    "aligned_source_local_positions": [0, 6, 0, 1],
                    "provenance_per_draft_token": [
                        {},
                        {"source_accessible": 0.04, "source_inaccessible": 0.02},
                        {"source_accessible": 0.004, "source_inaccessible": 0.001},
                        {"source_accessible": 0.08, "source_inaccessible": 0.0},
                    ],
                }
            },
            {
                "alignatt_metadata": {
                    "stop_reason": "alignatt:source_regression",
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
            "alignatt_guard_flags": "token_argmax_frontier",
            "longyaal_cu_ms": "2000.0",
            "xcometxl": "0.75",
        }
    )

    assert row is not None
    assert row["token_argmax_frontier_stop_count"] == 1
    assert row["token_argmax_frontier_stop_ratio"] == pytest.approx(0.5)
    assert row["mean_extra_draft_tokens_after_stop"] == pytest.approx(3.0)
    assert row["updates_with_future_recovery"] == 1
    assert row["future_recovery_ratio"] == pytest.approx(1.0)
    assert row["mean_future_recovery_token_gap"] == pytest.approx(1.0)
    assert row["simulated_patience2_gain_update_count"] == 1
    assert row["simulated_patience2_token_gain_sum"] == 3
    assert row["simulated_recoverable_gain_update_count"] == 1
    assert row["simulated_recoverable_token_gain_sum"] == 3
    assert row["simulated_recoverable_full_draft_count"] == 1
    assert row["mean_blocked_position"] == pytest.approx(6.0)


def test_report_rows_filters_to_enzh_milmmt_with_token_argmax_gate(tmp_path):
    gated_dir = tmp_path / "gated"
    write_json(gated_dir / "manifest.json", {"runtime_config": runtime_config()})
    write_jsonl(
        gated_dir / "stream_updates.jsonl",
        [{"alignatt_metadata": {"stop_reason": "stop"}}],
    )
    ungated_dir = tmp_path / "ungated"
    config = runtime_config()
    config["translation_alignatt_token_argmax_frontier_gate"] = False
    write_json(ungated_dir / "manifest.json", {"runtime_config": config})
    write_jsonl(
        ungated_dir / "stream_updates.jsonl",
        [{"alignatt_metadata": {"stop_reason": "stop"}}],
    )

    rows = report_rows(
        [
            {
                "relative_dir": "gated",
                "copied_dir": str(gated_dir),
                "target_language_code": "zh",
                "mt_backend_name": "milmmt_vllm_alignatt",
            },
            {
                "relative_dir": "ungated",
                "copied_dir": str(ungated_dir),
                "target_language_code": "zh",
                "mt_backend_name": "milmmt_vllm_alignatt",
            },
            {
                "relative_dir": "ende",
                "copied_dir": str(gated_dir),
                "target_language_code": "de",
                "mt_backend_name": "milmmt_vllm_alignatt",
            },
        ]
    )

    assert [row["relative_dir"] for row in rows] == ["gated"]
