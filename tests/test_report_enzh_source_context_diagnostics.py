from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.reports.report_enzh_source_context_diagnostics import analyze_run, report_rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def index_row(run_dir: Path) -> dict[str, str]:
    return {
        "relative_dir": "run",
        "copied_dir": str(run_dir),
        "target_language_code": "zh",
        "mt_backend_name": "milmmt_vllm_alignatt",
        "valid_for_claims": "True",
        "num_inputs": "3",
        "chunk_ms": "640",
        "alignatt_guard_flags": "min_accessible_source_units=6",
        "longyaal_cu_ms": "1800.0",
        "xcometxl": "0.70",
    }


def test_analyze_run_uses_chunk_decisions_for_zero_emit_blocks(tmp_path):
    run_dir = tmp_path / "run"
    write_json(
        run_dir / "manifest.json",
        {
            "runtime_config": {
                "translation_alignatt_min_accessible_source_units": 6,
                "translation_alignatt_min_accessible_source_units_mode": "block",
            }
        },
    )
    write_jsonl(
        run_dir / "chunk_decisions.jsonl",
        [
            {
                "audio_processed_ms": 1280.0,
                "emitted": False,
                "alignatt_metadata_current_chunk": True,
                "alignatt_decision": {
                    "stop_reason": "stop",
                    "accepted_token_count": 0,
                    "accepted_candidate_token_count": 5,
                    "target_stability_unit_end_token_indices": [1, 2, 4, 5],
                    "accessible_source_unit_count": 3,
                    "alignatt_source_context_accessible_units": 3,
                    "alignatt_source_context_under_min": True,
                    "alignatt_source_context_blocked": True,
                },
            },
            {
                "audio_processed_ms": 1920.0,
                "emitted": True,
                "alignatt_metadata_current_chunk": True,
                "alignatt_decision": {
                    "stop_reason": "stop",
                    "accepted_token_count": 2,
                    "accessible_source_unit_count": 6,
                    "alignatt_source_context_under_min": False,
                    "alignatt_source_context_blocked": False,
                },
            },
            {
                "audio_processed_ms": 2240.0,
                "emitted": True,
                "alignatt_metadata_current_chunk": True,
                "alignatt_decision": {
                    "stop_reason": "stop",
                    "accepted_token_count": 3,
                    "accessible_source_unit_count": 4,
                    "alignatt_source_context_accessible_units": 4,
                    "alignatt_source_context_under_min": True,
                    "alignatt_source_context_blocked": False,
                    "alignatt_source_context_cap_applied": True,
                    "alignatt_source_context_cap_target_units": 4,
                },
            },
            {
                "audio_processed_ms": 2560.0,
                "emitted": False,
                "alignatt_metadata_current_chunk": False,
                "alignatt_decision": {
                    "alignatt_source_context_under_min": True,
                    "alignatt_source_context_blocked": True,
                },
            },
        ],
    )
    write_jsonl(run_dir / "stream_updates.jsonl", [{"alignatt_metadata": {}}])

    row = analyze_run(index_row(run_dir))

    assert row is not None
    assert row["observability"] == "chunk_decisions"
    assert row["chunk_decision_count"] == 4
    assert row["current_mt_decision_count"] == 3
    assert row["source_context_under_min_count"] == 2
    assert row["source_context_blocked_count"] == 1
    assert row["source_context_cap_applied_count"] == 1
    assert row["zero_emit_source_context_blocked_count"] == 1
    assert row["zero_accept_source_context_blocked_count"] == 1
    assert row["mean_under_min_accessible_source_units"] == pytest.approx(3.5)
    assert row["mean_source_context_cap_target_units"] == pytest.approx(4.0)
    assert row["target_unit_cap_replay_source"] == "chunk_decisions"
    assert row["simulated_target_unit_cap_opportunity_count"] == 1
    assert row["simulated_target_unit_cap_token_gain_sum"] == 4
    assert row["mean_simulated_target_unit_cap_token_gain"] == pytest.approx(4.0)
    assert row["simulated_target_unit_cap_full_candidate_count"] == 0
    assert row["first_current_mt_audio_ms"] == pytest.approx(1280.0)
    assert row["first_source_context_blocked_audio_ms"] == pytest.approx(1280.0)
    assert row["first_source_context_cap_audio_ms"] == pytest.approx(2240.0)
    assert row["translation_alignatt_min_accessible_source_units"] == 6


def test_analyze_run_marks_stream_update_only_observability(tmp_path):
    run_dir = tmp_path / "run"
    write_json(
        run_dir / "manifest.json",
        {
            "runtime_config": {
                "translation_alignatt_min_accessible_source_units": 6,
            }
        },
    )
    write_jsonl(
        run_dir / "stream_updates.jsonl",
        [
            {
                "alignatt_metadata": {
                    "accepted_token_count": 0,
                    "accepted_candidate_token_count": 3,
                    "target_stability_unit_end_token_indices": [1, 2, 3],
                    "alignatt_source_context_accessible_units": 4,
                    "alignatt_source_context_under_min": True,
                    "alignatt_source_context_blocked": True,
                }
            },
            {
                "alignatt_metadata": {
                    "alignatt_source_context_under_min": True,
                    "alignatt_source_context_blocked": False,
                    "alignatt_source_context_cap_applied": True,
                    "alignatt_source_context_cap_target_units": 2,
                }
            },
        ],
    )

    row = analyze_run(index_row(run_dir))

    assert row is not None
    assert row["observability"] == "stream_updates_only_no_zero_emit_visibility"
    assert row["chunk_decision_count"] == 0
    assert row["current_mt_decision_count"] == 0
    assert row["stream_update_count"] == 2
    assert row["stream_update_source_context_under_min_count"] == 2
    assert row["stream_update_source_context_blocked_count"] == 1
    assert row["stream_update_source_context_cap_applied_count"] == 1
    assert row["stream_update_mean_source_context_cap_target_units"] == pytest.approx(2.0)
    assert row["zero_emit_source_context_blocked_count"] == 0
    assert row["target_unit_cap_replay_source"] == "stream_updates"
    assert row["simulated_target_unit_cap_opportunity_count"] == 1
    assert row["simulated_target_unit_cap_token_gain_sum"] == 3
    assert row["simulated_target_unit_cap_full_candidate_count"] == 1


def test_report_rows_filters_to_enzh_milmmt(tmp_path):
    run_dir = tmp_path / "run"
    write_json(run_dir / "manifest.json", {"runtime_config": {}})
    write_jsonl(run_dir / "stream_updates.jsonl", [{"alignatt_metadata": {}}])

    rows = report_rows(
        [
            index_row(run_dir),
            {
                **index_row(run_dir),
                "relative_dir": "ende",
                "target_language_code": "de",
            },
            {
                **index_row(run_dir),
                "relative_dir": "gemma",
                "mt_backend_name": "gemma_vllm_alignatt",
            },
        ]
    )

    assert [row["relative_dir"] for row in rows] == ["run"]
