#!/usr/bin/env python3
"""Report EN->ZH minimum-source-context gate decisions.

This is the audit for AlignAtt runs that become too close to local agreement:
`stream_updates.jsonl` can only show emitted updates, while `chunk_decisions.jsonl`
can also expose zero-emission MT decisions blocked by
`translation_alignatt_min_accessible_source_units`.
"""

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

from tools.reports.report_enzh_source_regression_diagnostics import (  # noqa: E402
    finite_int,
    iter_stream_updates,
    load_index,
    load_json,
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
    "observability",
    "chunk_decision_count",
    "current_mt_decision_count",
    "scheduler_skip_current_count",
    "source_context_under_min_count",
    "source_context_blocked_count",
    "source_context_cap_applied_count",
    "mean_source_context_cap_target_units",
    "zero_emit_source_context_blocked_count",
    "zero_accept_source_context_blocked_count",
    "stream_update_count",
    "stream_update_source_context_under_min_count",
    "stream_update_source_context_blocked_count",
    "stream_update_source_context_cap_applied_count",
    "stream_update_mean_source_context_cap_target_units",
    "target_unit_cap_replay_source",
    "simulated_target_unit_cap_opportunity_count",
    "simulated_target_unit_cap_token_gain_sum",
    "mean_simulated_target_unit_cap_token_gain",
    "simulated_target_unit_cap_full_candidate_count",
    "mean_under_min_accessible_source_units",
    "first_current_mt_audio_ms",
    "first_source_context_blocked_audio_ms",
    "first_source_context_cap_audio_ms",
    "translation_alignatt_min_accessible_source_units",
    "translation_alignatt_min_accessible_source_units_mode",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-stem",
        default="enzh_source_context_diagnostics_20260607",
    )
    return parser.parse_args()


def iter_jsonl(path: Path):
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def decision_metadata(decision: dict[str, Any]) -> dict[str, Any]:
    metadata = decision.get("alignatt_decision")
    return metadata if isinstance(metadata, dict) else {}


def update_metadata(update: dict[str, Any]) -> dict[str, Any]:
    metadata = update.get("alignatt_metadata")
    return metadata if isinstance(metadata, dict) else {}


def int_list(values: Any) -> list[int]:
    if not isinstance(values, list):
        return []
    ints: list[int] = []
    for value in values:
        try:
            ints.append(int(value))
        except (TypeError, ValueError):
            continue
    return ints


def simulate_target_unit_cap_gain(metadata: dict[str, Any]) -> tuple[int, bool]:
    if not boolish(metadata.get("alignatt_source_context_blocked")):
        return 0, False
    accepted_count = finite_int(metadata.get("accepted_token_count"), default=0)
    candidate_count = finite_int(
        metadata.get("accepted_candidate_token_count"),
        default=accepted_count,
    )
    accessible_units = finite_int(
        metadata.get("alignatt_source_context_accessible_units")
        or metadata.get("accessible_source_unit_count"),
        default=0,
    )
    unit_ends = int_list(metadata.get("target_stability_unit_end_token_indices"))
    if candidate_count <= accepted_count or accessible_units <= 0 or not unit_ends:
        return 0, False
    if len(unit_ends) <= accessible_units:
        capped_count = candidate_count
    else:
        capped_count = min(candidate_count, max(0, unit_ends[accessible_units - 1]))
    gain = max(0, capped_count - accepted_count)
    return gain, bool(gain > 0 and capped_count >= candidate_count)


def analyze_run(index_row: dict[str, str]) -> dict[str, Any] | None:
    copied_dir = Path(index_row.get("copied_dir") or "")
    manifest = load_json(copied_dir / "manifest.json")
    runtime_config = manifest.get("runtime_config") or {}
    chunk_decisions = list(iter_jsonl(copied_dir / "chunk_decisions.jsonl") or [])
    stream_updates = list(iter_stream_updates(copied_dir / "stream_updates.jsonl") or [])
    if not chunk_decisions and not stream_updates:
        return None

    current_mt_count = 0
    scheduler_skip_current_count = 0
    under_min_count = 0
    blocked_count = 0
    cap_count = 0
    zero_emit_blocked_count = 0
    zero_accept_blocked_count = 0
    under_min_accessible_counts: list[int] = []
    cap_target_units: list[int] = []
    first_current_mt_audio_ms: float | None = None
    first_blocked_audio_ms: float | None = None
    first_cap_audio_ms: float | None = None
    decision_replay_metadata: list[dict[str, Any]] = []

    for decision in chunk_decisions:
        if not boolish(decision.get("alignatt_metadata_current_chunk")):
            continue
        current_mt_count += 1
        if first_current_mt_audio_ms is None:
            first_current_mt_audio_ms = float(decision.get("audio_processed_ms") or 0.0)
        metadata = decision_metadata(decision)
        decision_replay_metadata.append(metadata)
        if boolish(metadata.get("scheduler_skipped")):
            scheduler_skip_current_count += 1
        if boolish(metadata.get("alignatt_source_context_under_min")):
            under_min_count += 1
            under_min_accessible_counts.append(
                finite_int(
                    metadata.get("alignatt_source_context_accessible_units")
                    or metadata.get("accessible_source_unit_count"),
                    default=0,
                )
            )
        if boolish(metadata.get("alignatt_source_context_blocked")):
            blocked_count += 1
            if first_blocked_audio_ms is None:
                first_blocked_audio_ms = float(decision.get("audio_processed_ms") or 0.0)
            if not boolish(decision.get("emitted")):
                zero_emit_blocked_count += 1
            if finite_int(metadata.get("accepted_token_count"), default=-1) == 0:
                zero_accept_blocked_count += 1
        if boolish(metadata.get("alignatt_source_context_cap_applied")):
            cap_count += 1
            if first_cap_audio_ms is None:
                first_cap_audio_ms = float(decision.get("audio_processed_ms") or 0.0)
            cap_target_units.append(
                finite_int(
                    metadata.get("alignatt_source_context_cap_target_units"),
                    default=0,
                )
            )

    update_under_min_count = 0
    update_blocked_count = 0
    update_cap_count = 0
    update_cap_target_units: list[int] = []
    update_replay_metadata: list[dict[str, Any]] = []
    for update in stream_updates:
        metadata = update_metadata(update)
        update_replay_metadata.append(metadata)
        if boolish(metadata.get("alignatt_source_context_under_min")):
            update_under_min_count += 1
        if boolish(metadata.get("alignatt_source_context_blocked")):
            update_blocked_count += 1
        if boolish(metadata.get("alignatt_source_context_cap_applied")):
            update_cap_count += 1
            update_cap_target_units.append(
                finite_int(
                    metadata.get("alignatt_source_context_cap_target_units"),
                    default=0,
                )
            )

    if chunk_decisions:
        observability = "chunk_decisions"
        replay_source = "chunk_decisions"
        replay_metadata = decision_replay_metadata
    else:
        observability = "stream_updates_only_no_zero_emit_visibility"
        replay_source = "stream_updates"
        replay_metadata = update_replay_metadata

    cap_gains: list[int] = []
    cap_full_candidate_count = 0
    for metadata in replay_metadata:
        gain, full_candidate = simulate_target_unit_cap_gain(metadata)
        if gain > 0:
            cap_gains.append(gain)
        if full_candidate:
            cap_full_candidate_count += 1

    return {
        "relative_dir": index_row.get("relative_dir") or copied_dir.name,
        "num_inputs": index_row.get("num_inputs") or manifest.get("num_inputs"),
        "chunk_ms": index_row.get("chunk_ms"),
        "valid_for_claims": index_row.get("valid_for_claims"),
        "alignatt_guard_flags": index_row.get("alignatt_guard_flags"),
        "longyaal_cu_ms": index_row.get("longyaal_cu_ms"),
        "xcometxl": index_row.get("xcometxl"),
        "observability": observability,
        "chunk_decision_count": len(chunk_decisions),
        "current_mt_decision_count": current_mt_count,
        "scheduler_skip_current_count": scheduler_skip_current_count,
        "source_context_under_min_count": under_min_count,
        "source_context_blocked_count": blocked_count,
        "source_context_cap_applied_count": cap_count,
        "mean_source_context_cap_target_units": (
            sum(cap_target_units) / float(len(cap_target_units))
            if cap_target_units
            else None
        ),
        "zero_emit_source_context_blocked_count": zero_emit_blocked_count,
        "zero_accept_source_context_blocked_count": zero_accept_blocked_count,
        "stream_update_count": len(stream_updates),
        "stream_update_source_context_under_min_count": update_under_min_count,
        "stream_update_source_context_blocked_count": update_blocked_count,
        "stream_update_source_context_cap_applied_count": update_cap_count,
        "stream_update_mean_source_context_cap_target_units": (
            sum(update_cap_target_units) / float(len(update_cap_target_units))
            if update_cap_target_units
            else None
        ),
        "target_unit_cap_replay_source": replay_source,
        "simulated_target_unit_cap_opportunity_count": len(cap_gains),
        "simulated_target_unit_cap_token_gain_sum": sum(cap_gains),
        "mean_simulated_target_unit_cap_token_gain": (
            sum(cap_gains) / float(len(cap_gains)) if cap_gains else None
        ),
        "simulated_target_unit_cap_full_candidate_count": cap_full_candidate_count,
        "mean_under_min_accessible_source_units": (
            sum(under_min_accessible_counts) / float(len(under_min_accessible_counts))
            if under_min_accessible_counts
            else None
        ),
        "first_current_mt_audio_ms": first_current_mt_audio_ms,
        "first_source_context_blocked_audio_ms": first_blocked_audio_ms,
        "first_source_context_cap_audio_ms": first_cap_audio_ms,
        "translation_alignatt_min_accessible_source_units": finite_int(
            runtime_config.get("translation_alignatt_min_accessible_source_units"),
            default=0,
        ),
        "translation_alignatt_min_accessible_source_units_mode": runtime_config.get(
            "translation_alignatt_min_accessible_source_units_mode",
            "block",
        ),
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
            str(row["observability"]) != "chunk_decisions",
            -int(row["simulated_target_unit_cap_token_gain_sum"]),
            -int(row["simulated_target_unit_cap_opportunity_count"]),
            -int(row["zero_emit_source_context_blocked_count"]),
            -int(row["source_context_blocked_count"]),
            row["relative_dir"],
        )
    )
    return rows


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


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
        "runs_with_chunk_decisions": sum(
            1 for row in rows if row["observability"] == "chunk_decisions"
        ),
        "runs_stream_updates_only": sum(
            1
            for row in rows
            if row["observability"] == "stream_updates_only_no_zero_emit_visibility"
        ),
        "total_source_context_blocked_count": sum(
            int(row["source_context_blocked_count"]) for row in rows
        ),
        "total_zero_emit_source_context_blocked_count": sum(
            int(row["zero_emit_source_context_blocked_count"]) for row in rows
        ),
        "total_stream_update_source_context_cap_applied_count": sum(
            int(row["stream_update_source_context_cap_applied_count"]) for row in rows
        ),
        "total_simulated_target_unit_cap_opportunity_count": sum(
            int(row["simulated_target_unit_cap_opportunity_count"]) for row in rows
        ),
        "total_simulated_target_unit_cap_token_gain_sum": sum(
            int(row["simulated_target_unit_cap_token_gain_sum"]) for row in rows
        ),
        "total_simulated_target_unit_cap_full_candidate_count": sum(
            int(row["simulated_target_unit_cap_full_candidate_count"]) for row in rows
        ),
        "rows": rows,
    }
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    print(tsv_path.resolve())
    print(json_path.resolve())


def main() -> None:
    args = parse_args()
    rows = report_rows(load_index(args.index))
    write_outputs(rows, args.output_dir, args.output_stem)


if __name__ == "__main__":
    main()
