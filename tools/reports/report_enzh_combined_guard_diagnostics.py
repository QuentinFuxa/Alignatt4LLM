#!/usr/bin/env python3
"""Rank EN->ZH guarded AlignAtt runs by combined permissiveness opportunities."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.reports.report_enzh_source_context_diagnostics import (  # noqa: E402
    analyze_run as analyze_source_context_run,
)
from tools.reports.report_enzh_source_regression_diagnostics import (  # noqa: E402
    analyze_run as analyze_source_regression_run,
    load_index,
)


DEFAULT_INDEX = Path(
    "outputs/diagnostics_jarvislab_20260606/diagnostics_artifact_index.tsv"
)
DEFAULT_OUTPUT_DIR = Path("outputs/plots")

REPORT_COLUMNS = (
    "relative_dir",
    "num_inputs",
    "chunk_ms",
    "valid_for_claims",
    "alignatt_guard_flags",
    "longyaal_cu_ms",
    "xcometxl",
    "target_unit_cap_token_gain_sum",
    "target_unit_cap_opportunity_count",
    "target_unit_cap_visible_cap_count",
    "trim_unrecovered_gate_aware_token_gain_sum",
    "trim_unrecovered_gate_aware_update_count",
    "combined_permissive_token_gain_sum",
    "source_regression_stop_count",
    "source_context_blocked_update_count",
    "source_context_cap_applied_update_count",
    "translation_alignatt_min_accessible_source_units",
    "translation_alignatt_min_accessible_source_units_mode",
    "translation_alignatt_max_source_regression",
    "translation_alignatt_source_regression_action",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-stem",
        default="enzh_combined_guard_diagnostics_20260607",
    )
    return parser.parse_args()


def int_value(row: dict[str, Any] | None, key: str) -> int:
    if not row:
        return 0
    try:
        return int(row.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def text_value(row: dict[str, Any] | None, key: str, default: str = "") -> str:
    if not row:
        return default
    value = row.get(key)
    if value is None:
        return default
    return str(value)


def combine_run(index_row: dict[str, str]) -> dict[str, Any] | None:
    source_context = analyze_source_context_run(index_row)
    source_regression = analyze_source_regression_run(index_row)
    if source_context is None and source_regression is None:
        return None
    cap_gain = int_value(source_context, "simulated_target_unit_cap_token_gain_sum")
    trim_gain = int_value(source_regression, "simulated_gate_aware_token_gain_sum")
    return {
        "relative_dir": index_row.get("relative_dir") or text_value(
            source_context or source_regression, "relative_dir"
        ),
        "num_inputs": index_row.get("num_inputs")
        or text_value(source_context or source_regression, "num_inputs"),
        "chunk_ms": index_row.get("chunk_ms")
        or text_value(source_context or source_regression, "chunk_ms"),
        "valid_for_claims": index_row.get("valid_for_claims")
        or text_value(source_context or source_regression, "valid_for_claims"),
        "alignatt_guard_flags": index_row.get("alignatt_guard_flags")
        or text_value(source_context or source_regression, "alignatt_guard_flags"),
        "longyaal_cu_ms": index_row.get("longyaal_cu_ms")
        or text_value(source_context or source_regression, "longyaal_cu_ms"),
        "xcometxl": index_row.get("xcometxl")
        or text_value(source_context or source_regression, "xcometxl"),
        "target_unit_cap_token_gain_sum": cap_gain,
        "target_unit_cap_opportunity_count": int_value(
            source_context,
            "simulated_target_unit_cap_opportunity_count",
        ),
        "target_unit_cap_visible_cap_count": int_value(
            source_context,
            "stream_update_source_context_cap_applied_count",
        ),
        "trim_unrecovered_gate_aware_token_gain_sum": trim_gain,
        "trim_unrecovered_gate_aware_update_count": int_value(
            source_regression,
            "simulated_gate_aware_gain_update_count",
        ),
        "combined_permissive_token_gain_sum": cap_gain + trim_gain,
        "source_regression_stop_count": int_value(
            source_regression,
            "source_regression_stop_count",
        ),
        "source_context_blocked_update_count": int_value(
            source_context,
            "stream_update_source_context_blocked_count",
        ),
        "source_context_cap_applied_update_count": int_value(
            source_context,
            "stream_update_source_context_cap_applied_count",
        ),
        "translation_alignatt_min_accessible_source_units": text_value(
            source_context,
            "translation_alignatt_min_accessible_source_units",
            "0",
        ),
        "translation_alignatt_min_accessible_source_units_mode": text_value(
            source_context,
            "translation_alignatt_min_accessible_source_units_mode",
            "block",
        ),
        "translation_alignatt_max_source_regression": text_value(
            source_regression,
            "translation_alignatt_max_source_regression",
            "-1",
        ),
        "translation_alignatt_source_regression_action": text_value(
            source_regression,
            "translation_alignatt_source_regression_action",
            "stop",
        ),
    }


def report_rows(index_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in index_rows:
        if row.get("target_language_code") != "zh":
            continue
        if row.get("mt_backend_name") != "milmmt_vllm_alignatt":
            continue
        combined = combine_run(row)
        if combined is not None:
            rows.append(combined)
    rows.sort(
        key=lambda row: (
            -int(row["combined_permissive_token_gain_sum"]),
            -int(row["trim_unrecovered_gate_aware_token_gain_sum"]),
            -int(row["target_unit_cap_token_gain_sum"]),
            str(row["relative_dir"]),
        )
    )
    return rows


def write_outputs(rows: list[dict[str, Any]], output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = output_dir / f"{stem}.tsv"
    json_path = output_dir / f"{stem}.json"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in REPORT_COLUMNS})
    summary = {
        "run_count": len(rows),
        "total_target_unit_cap_token_gain_sum": sum(
            int(row["target_unit_cap_token_gain_sum"]) for row in rows
        ),
        "total_trim_unrecovered_gate_aware_token_gain_sum": sum(
            int(row["trim_unrecovered_gate_aware_token_gain_sum"]) for row in rows
        ),
        "total_combined_permissive_token_gain_sum": sum(
            int(row["combined_permissive_token_gain_sum"]) for row in rows
        ),
        "rows": rows,
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(tsv_path.resolve())
    print(json_path.resolve())


def main() -> None:
    args = parse_args()
    rows = report_rows(load_index(args.index))
    write_outputs(rows, args.output_dir, args.output_stem)


if __name__ == "__main__":
    main()
