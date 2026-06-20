from __future__ import annotations

import json
from pathlib import Path

from scripts.report_enzh_combined_guard_diagnostics import combine_run, report_rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def index_row(run_dir: Path, *, relative_dir: str = "run") -> dict[str, str]:
    return {
        "relative_dir": relative_dir,
        "copied_dir": str(run_dir),
        "target_language_code": "zh",
        "mt_backend_name": "milmmt_vllm_alignatt",
        "valid_for_claims": "True",
        "num_inputs": "3",
        "chunk_ms": "640",
        "alignatt_guard_flags": "source_regression,min_accessible_source_units",
        "longyaal_cu_ms": "1800.0",
        "xcometxl": "0.70",
    }


def write_guarded_run(run_dir: Path) -> None:
    write_json(
        run_dir / "manifest.json",
        {
            "num_inputs": 3,
            "runtime_config": {
                "translation_alignatt_min_accessible_source_units": 6,
                "translation_alignatt_min_accessible_source_units_mode": "block",
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
                    "stop_reason": "stop",
                    "accepted_token_count": 0,
                    "accepted_candidate_token_count": 4,
                    "target_stability_unit_end_token_indices": [1, 2, 4],
                    "alignatt_source_context_accessible_units": 3,
                    "alignatt_source_context_under_min": True,
                    "alignatt_source_context_blocked": True,
                }
            },
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
        ],
    )


def test_combine_run_sums_source_context_and_regression_opportunities(tmp_path):
    run_dir = tmp_path / "run"
    write_guarded_run(run_dir)

    row = combine_run(index_row(run_dir))

    assert row is not None
    assert row["target_unit_cap_token_gain_sum"] == 4
    assert row["target_unit_cap_opportunity_count"] == 1
    assert row["trim_unrecovered_gate_aware_token_gain_sum"] == 3
    assert row["trim_unrecovered_gate_aware_update_count"] == 1
    assert row["combined_permissive_token_gain_sum"] == 7
    assert row["source_regression_stop_count"] == 1
    assert row["source_context_blocked_update_count"] == 1


def test_report_rows_filters_and_sorts_by_combined_gain(tmp_path):
    run_dir = tmp_path / "run"
    low_dir = tmp_path / "low"
    write_guarded_run(run_dir)
    write_json(low_dir / "manifest.json", {"runtime_config": {}})
    write_jsonl(low_dir / "stream_updates.jsonl", [{"alignatt_metadata": {}}])

    rows = report_rows(
        [
            index_row(low_dir, relative_dir="low"),
            index_row(run_dir, relative_dir="run"),
            {
                **index_row(run_dir, relative_dir="ende"),
                "target_language_code": "de",
            },
        ]
    )

    assert [row["relative_dir"] for row in rows] == ["run", "low"]
