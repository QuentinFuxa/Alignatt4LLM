from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.report_enzh_source_frontier_diagnostics import (
    analyze_run,
    recoverable_source_frontier_accept_count,
    report_rows,
    source_frontier_blocks,
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
        "translation_alignatt_border_margin": 1,
        "translation_alignatt_frontier_min_inaccessible_mass": 0.03,
        "translation_alignatt_token_argmax_frontier_gate": False,
        "translation_alignatt_min_source_mass": 0.0,
        "translation_alignatt_max_inaccessible_source_mass": 1.0,
        "translation_alignatt_max_non_source_prompt_mass": 1.0,
        "translation_alignatt_min_accessible_inaccessible_margin": -1.0,
    }


def test_source_frontier_blocks_respects_soft_future_mass():
    metadata = {"accessible_source_local_end_exclusive": 4}
    config = runtime_config()

    assert source_frontier_blocks(
        position=5,
        provenance={"source_inaccessible": 0.05},
        metadata=metadata,
        runtime_config=config,
    )
    assert not source_frontier_blocks(
        position=5,
        provenance={"source_inaccessible": 0.01},
        metadata=metadata,
        runtime_config=config,
    )
    assert not source_frontier_blocks(
        position=4,
        provenance={"source_inaccessible": 0.05},
        metadata=metadata,
        runtime_config=config,
    )


def test_recoverable_source_frontier_keeps_recovered_prefix_only():
    accept_count, blocked_by_other_gate = recoverable_source_frontier_accept_count(
        [0, 5, 0, 1],
        provenance=[
            {},
            {"source_accessible": 0.2, "source_inaccessible": 0.05},
            {"source_accessible": 0.2, "source_inaccessible": 0.0},
            {"source_accessible": 0.2, "source_inaccessible": 0.0},
        ],
        metadata={"accessible_source_local_end_exclusive": 4},
        runtime_config=runtime_config(),
        accepted_count=1,
    )

    assert accept_count == 4
    assert blocked_by_other_gate is False

    accept_count, blocked_by_other_gate = recoverable_source_frontier_accept_count(
        [0, 5, 6],
        provenance=[
            {},
            {"source_accessible": 0.2, "source_inaccessible": 0.05},
            {"source_accessible": 0.2, "source_inaccessible": 0.05},
        ],
        metadata={"accessible_source_local_end_exclusive": 4},
        runtime_config=runtime_config(),
        accepted_count=1,
    )

    assert accept_count == 1
    assert blocked_by_other_gate is False


def test_analyze_run_counts_source_frontier_future_recovery(tmp_path):
    run_dir = tmp_path / "run"
    write_json(run_dir / "manifest.json", {"runtime_config": runtime_config()})
    write_jsonl(
        run_dir / "stream_updates.jsonl",
        [
            {
                "alignatt_metadata": {
                    "stop_reason": "alignatt:source_frontier",
                    "accepted_candidate_token_count": 1,
                    "unsafe_target_token_index": 1,
                    "blocked_source_local_position": 5,
                    "accessible_source_local_end_exclusive": 4,
                    "aligned_source_local_positions": [0, 5, 0, 1],
                    "provenance_per_draft_token": [
                        {},
                        {"source_accessible": 0.2, "source_inaccessible": 0.05},
                        {"source_accessible": 0.2, "source_inaccessible": 0.0},
                        {"source_accessible": 0.2, "source_inaccessible": 0.0},
                    ],
                }
            },
            {"alignatt_metadata": {"stop_reason": "stop"}},
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
            "alignatt_guard_flags": "",
            "longyaal_cu_ms": "1000.0",
            "xcometxl": "0.60",
        }
    )

    assert row is not None
    assert row["source_frontier_stop_count"] == 1
    assert row["source_frontier_stop_ratio"] == pytest.approx(0.5)
    assert row["updates_with_future_recovery"] == 1
    assert row["future_recovery_ratio"] == pytest.approx(1.0)
    assert row["simulated_recoverable_gain_update_count"] == 1
    assert row["simulated_recoverable_token_gain_sum"] == 3
    assert row["translation_alignatt_source_frontier_action"] == "stop"


def test_report_rows_filters_to_enzh_milmmt(tmp_path):
    run_dir = tmp_path / "run"
    write_json(run_dir / "manifest.json", {"runtime_config": runtime_config()})
    write_jsonl(
        run_dir / "stream_updates.jsonl",
        [{"alignatt_metadata": {"stop_reason": "stop"}}],
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
