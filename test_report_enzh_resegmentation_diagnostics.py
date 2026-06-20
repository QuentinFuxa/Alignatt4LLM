from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.report_enzh_resegmentation_diagnostics import (
    candidate_index_rows,
    segment_diagnostics,
    summarize_run,
    worst_segments,
    write_outputs,
)


def test_segment_diagnostics_counts_shape_errors():
    diagnostics = segment_diagnostics(
        [
            {"index": 0, "prediction": "。", "reference": "这是一句话。"},
            {"index": 1, "prediction": "这是一个很长很长的错误扩展", "reference": "短句"},
            {"index": 2, "prediction": "准确翻译", "reference": "准确翻译"},
        ]
    )

    row = summarize_run(
        {
            "relative_dir": "run",
            "num_inputs": 3,
            "chunk_ms": 640,
            "longyaal_cu_ms": 1000,
            "xcometxl": 0.5,
            "alignatt_policy_family": "pure_soft_frontier",
            "alignatt_guard_flags": "",
        },
        diagnostics,
    )

    assert row["punctuation_only_prediction_count"] == 1
    assert row["overlong_prediction_count"] == 1
    assert row["underlong_prediction_count"] == 1
    assert row["low_similarity_count"] >= 1
    assert worst_segments(diagnostics, limit=1)[0]["index"] == 1


def test_candidate_index_rows_and_outputs(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "instances.resegmented.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"prediction": "你好", "reference": "你好"}, ensure_ascii=False),
                json.dumps({"prediction": "。", "reference": "世界"}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            [
                {
                    "relative_dir": "run",
                    "copied_dir": str(run_dir),
                    "valid_for_claims": True,
                    "target_language_code": "zh",
                    "mt_backend_name": "milmmt_vllm_alignatt",
                    "num_inputs": 3,
                    "chunk_ms": 640,
                    "longyaal_cu_ms": 1000,
                    "xcometxl": 0.5,
                    "alignatt_policy_family": "pure_soft_frontier",
                    "alignatt_guard_flags": "",
                },
                {
                    "relative_dir": "bad",
                    "copied_dir": str(tmp_path / "bad"),
                    "valid_for_claims": False,
                    "target_language_code": "zh",
                    "mt_backend_name": "milmmt_vllm_alignatt",
                },
            ]
        ),
        encoding="utf-8",
    )

    rows = candidate_index_rows(index_path, relative_dirs=set())
    assert [row["relative_dir"] for row in rows] == ["run"]

    diagnostics = segment_diagnostics(
        json.loads(line)
        for line in (run_dir / "instances.resegmented.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    )
    summary = [summarize_run(rows[0], diagnostics)]
    paths = write_outputs(
        summary,
        {"run": worst_segments(diagnostics, limit=1)},
        output_dir=tmp_path,
        output_stem="diag",
    )

    with paths["tsv"].open(encoding="utf-8", newline="") as handle:
        report_row = next(csv.DictReader(handle, delimiter="\t"))
    assert report_row["relative_dir"] == "run"
    assert report_row["punctuation_only_prediction_count"] == "1"
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["worst_segments_by_run"]["run"][0]["prediction"] == "。"
