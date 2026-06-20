#!/usr/bin/env python3
"""Report EN->ZH target units emitted in the final tail of each hypothesis."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_INDEX = Path(
    "outputs/diagnostics_jarvislab_20260606/diagnostics_artifact_index.tsv"
)
DEFAULT_OUTPUT_DIR = Path("outputs/plots")

REPORT_COLUMNS = (
    "relative_dir",
    "num_inputs",
    "chunk_ms",
    "valid_for_claims",
    "alignatt_policy_family",
    "alignatt_guard_flags",
    "longyaal_cu_ms",
    "xcometxl",
    "target_unit_count",
    "final_delay_unit_count",
    "final_delay_unit_ratio",
    "final_chunk_delay_unit_count",
    "final_chunk_delay_unit_ratio",
    "mean_delay_ms",
    "mean_tail_lag_ms",
    "p90_delay_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default="enzh_emission_lag_diagnostics_20260607")
    return parser.parse_args()


def finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def finite_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def load_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def iter_hypothesis_records(path: Path):
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


def analyze_run(index_row: dict[str, str]) -> dict[str, Any] | None:
    copied_dir = Path(index_row.get("copied_dir") or "")
    chunk_ms = finite_int(index_row.get("chunk_ms"), default=0)
    total_units = 0
    final_delay_units = 0
    final_chunk_delay_units = 0
    delays: list[float] = []
    tail_lags: list[float] = []
    for record in iter_hypothesis_records(copied_dir / "hypothesis.jsonl") or []:
        source_length = finite_float(record.get("source_length"))
        record_delays = record.get("delays")
        if source_length is None or not isinstance(record_delays, list):
            continue
        final_threshold = source_length - 1e-6
        final_chunk_threshold = source_length - max(0, chunk_ms) - 1e-6
        for delay_value in record_delays:
            delay = finite_float(delay_value)
            if delay is None:
                continue
            total_units += 1
            delays.append(delay)
            tail_lags.append(max(0.0, source_length - delay))
            if delay >= final_threshold:
                final_delay_units += 1
            if delay >= final_chunk_threshold:
                final_chunk_delay_units += 1
    if total_units <= 0:
        return None
    return {
        "relative_dir": index_row.get("relative_dir"),
        "num_inputs": index_row.get("num_inputs"),
        "chunk_ms": index_row.get("chunk_ms"),
        "valid_for_claims": index_row.get("valid_for_claims"),
        "alignatt_policy_family": index_row.get("alignatt_policy_family"),
        "alignatt_guard_flags": index_row.get("alignatt_guard_flags"),
        "longyaal_cu_ms": index_row.get("longyaal_cu_ms"),
        "xcometxl": index_row.get("xcometxl"),
        "target_unit_count": total_units,
        "final_delay_unit_count": final_delay_units,
        "final_delay_unit_ratio": final_delay_units / float(total_units),
        "final_chunk_delay_unit_count": final_chunk_delay_units,
        "final_chunk_delay_unit_ratio": final_chunk_delay_units / float(total_units),
        "mean_delay_ms": sum(delays) / float(len(delays)) if delays else None,
        "mean_tail_lag_ms": sum(tail_lags) / float(len(tail_lags)) if tail_lags else None,
        "p90_delay_ms": percentile(delays, 0.90),
    }


def report_rows(index_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in index_rows:
        if row.get("target_language_code") != "zh":
            continue
        if row.get("mt_backend_name") != "milmmt_vllm_alignatt":
            continue
        analyzed = analyze_run(row)
        if analyzed is not None:
            rows.append(analyzed)
    rows.sort(
        key=lambda row: (
            -float(row["final_chunk_delay_unit_ratio"]),
            -float(row["final_delay_unit_ratio"]),
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
        "total_target_unit_count": sum(int(row["target_unit_count"]) for row in rows),
        "total_final_delay_unit_count": sum(
            int(row["final_delay_unit_count"]) for row in rows
        ),
        "total_final_chunk_delay_unit_count": sum(
            int(row["final_chunk_delay_unit_count"]) for row in rows
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
