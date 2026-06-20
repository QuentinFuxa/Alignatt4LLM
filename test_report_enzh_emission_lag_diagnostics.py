from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.report_enzh_emission_lag_diagnostics import analyze_run, report_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_analyze_run_counts_final_tail_delay_units(tmp_path):
    run_dir = tmp_path / "run"
    write_jsonl(
        run_dir / "hypothesis.jsonl",
        [
            {
                "source": ["a.wav"],
                "source_length": 1000.0,
                "prediction": "你好啊",
                "delays": [320.0, 680.0, 1000.0],
            },
            {
                "source": ["b.wav"],
                "source_length": 2000.0,
                "prediction": "世界",
                "delays": [1200.0, 1500.0],
            },
        ],
    )

    row = analyze_run(
        {
            "relative_dir": "run",
            "copied_dir": str(run_dir),
            "chunk_ms": "640",
            "valid_for_claims": "True",
        }
    )

    assert row is not None
    assert row["target_unit_count"] == 5
    assert row["final_delay_unit_count"] == 1
    assert row["final_delay_unit_ratio"] == pytest.approx(0.2)
    assert row["final_chunk_delay_unit_count"] == 3
    assert row["final_chunk_delay_unit_ratio"] == pytest.approx(0.6)
    assert row["p90_delay_ms"] == pytest.approx(1500.0)


def test_report_rows_filters_to_enzh_milmmt(tmp_path):
    run_dir = tmp_path / "run"
    write_jsonl(
        run_dir / "hypothesis.jsonl",
        [{"source_length": 1000.0, "prediction": "你", "delays": [1000.0]}],
    )

    rows = report_rows(
        [
            {
                "relative_dir": "keep",
                "copied_dir": str(run_dir),
                "target_language_code": "zh",
                "mt_backend_name": "milmmt_vllm_alignatt",
            },
            {
                "relative_dir": "drop",
                "copied_dir": str(run_dir),
                "target_language_code": "de",
                "mt_backend_name": "milmmt_vllm_alignatt",
            },
        ]
    )

    assert [row["relative_dir"] for row in rows] == ["keep"]
