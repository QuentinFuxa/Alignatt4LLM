#!/usr/bin/env python3
"""Summarize EN->ZH token-argmax frontier hard-stop recovery opportunities."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.report_enzh_source_regression_diagnostics import (
    finite_bool,
    finite_float,
    finite_int,
    int_positions,
    iter_stream_updates,
    load_index,
    load_json,
    provenance_rows,
    token_passes_non_regression_gates,
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
    "token_argmax_frontier_stop_count",
    "token_argmax_frontier_stop_ratio",
    "mean_extra_draft_tokens_after_stop",
    "updates_with_future_recovery",
    "future_recovery_ratio",
    "mean_future_recovery_token_gap",
    "updates_blocked_by_other_gate_before_recovery",
    "simulated_patience2_gain_update_count",
    "simulated_patience2_token_gain_sum",
    "simulated_patience3_gain_update_count",
    "simulated_patience3_token_gain_sum",
    "simulated_patience4_gain_update_count",
    "simulated_patience4_token_gain_sum",
    "simulated_recoverable_gain_update_count",
    "simulated_recoverable_token_gain_sum",
    "mean_simulated_recoverable_token_gain",
    "simulated_recoverable_full_draft_count",
    "simulated_recoverable_unrecovered_suffix_count",
    "simulated_recoverable_blocked_by_other_gate_count",
    "mean_blocked_position",
    "translation_alignatt_token_argmax_frontier_gate",
    "translation_alignatt_token_argmax_min_source_mass",
    "translation_alignatt_token_argmax_frontier_margin",
    "translation_alignatt_token_argmax_frontier_patience_tokens",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-stem",
        default="enzh_token_argmax_frontier_diagnostics_20260607",
    )
    return parser.parse_args()


def token_argmax_frontier_blocks(
    *,
    position: int | None,
    provenance: dict[str, float] | None,
    metadata: dict[str, Any],
    runtime_config: dict[str, Any],
) -> bool:
    if not finite_bool(runtime_config.get("translation_alignatt_token_argmax_frontier_gate")):
        return False
    if position is None:
        return False
    source_accessible = None if provenance is None else provenance.get("source_accessible")
    source_inaccessible = (
        None if provenance is None else provenance.get("source_inaccessible")
    )
    if source_accessible is None or source_inaccessible is None:
        return False
    source_mass = float(source_accessible) + float(source_inaccessible)
    min_source_mass = finite_float(
        runtime_config.get("translation_alignatt_token_argmax_min_source_mass")
    )
    min_source_mass = 0.05 if min_source_mass is None else min_source_mass
    if source_mass < min_source_mass:
        return False
    accessible_count = finite_int(
        metadata.get("accessible_source_local_end_exclusive"),
        default=0,
    )
    frontier_margin = finite_int(
        runtime_config.get("translation_alignatt_token_argmax_frontier_margin"),
        default=0,
    )
    frontier = max(0, accessible_count) + max(0, frontier_margin)
    return int(position) >= frontier


def token_passes_other_alignatt_gates(
    *,
    position: int | None,
    provenance: dict[str, float] | None,
    metadata: dict[str, Any],
    runtime_config: dict[str, Any],
) -> bool:
    return token_passes_non_regression_gates(
        position=position,
        provenance=provenance,
        metadata=metadata,
        runtime_config=runtime_config,
        include_token_argmax_frontier_gate=False,
    )


def simulate_token_argmax_patience_accept_count(
    positions: list[int | None],
    *,
    provenance: list[dict[str, float]],
    metadata: dict[str, Any],
    runtime_config: dict[str, Any],
    accepted_count: int,
    patience_tokens: int,
) -> tuple[int, bool]:
    patience = max(1, int(patience_tokens))
    streak = 0
    for token_index in range(max(0, accepted_count), len(positions)):
        position = positions[token_index]
        row = provenance[token_index] if token_index < len(provenance) else {}
        if not token_passes_other_alignatt_gates(
            position=position,
            provenance=row,
            metadata=metadata,
            runtime_config=runtime_config,
        ):
            return token_index, True
        if token_argmax_frontier_blocks(
            position=position,
            provenance=row,
            metadata=metadata,
            runtime_config=runtime_config,
        ):
            streak += 1
            if streak >= patience:
                return token_index, False
            continue
        streak = 0
    return len(positions), False


def recoverable_token_argmax_accept_count(
    positions: list[int | None],
    *,
    provenance: list[dict[str, float]],
    metadata: dict[str, Any],
    runtime_config: dict[str, Any],
    accepted_count: int,
) -> tuple[int, bool]:
    pending_trim_index: int | None = None
    for token_index in range(max(0, accepted_count), len(positions)):
        position = positions[token_index]
        row = provenance[token_index] if token_index < len(provenance) else {}
        if not token_passes_other_alignatt_gates(
            position=position,
            provenance=row,
            metadata=metadata,
            runtime_config=runtime_config,
        ):
            return (
                pending_trim_index if pending_trim_index is not None else token_index,
                True,
            )
        if token_argmax_frontier_blocks(
            position=position,
            provenance=row,
            metadata=metadata,
            runtime_config=runtime_config,
        ):
            if pending_trim_index is None:
                pending_trim_index = token_index
            continue
        pending_trim_index = None
    if pending_trim_index is not None:
        return pending_trim_index, False
    return len(positions), False


def first_future_recovery_gap(
    positions: list[int | None],
    *,
    provenance: list[dict[str, float]],
    metadata: dict[str, Any],
    runtime_config: dict[str, Any],
    unsafe_index: int,
) -> tuple[int | None, bool]:
    for offset, token_index in enumerate(range(unsafe_index + 1, len(positions)), start=1):
        position = positions[token_index]
        row = provenance[token_index] if token_index < len(provenance) else {}
        if not token_passes_other_alignatt_gates(
            position=position,
            provenance=row,
            metadata=metadata,
            runtime_config=runtime_config,
        ):
            return None, True
        if not token_argmax_frontier_blocks(
            position=position,
            provenance=row,
            metadata=metadata,
            runtime_config=runtime_config,
        ):
            return offset, False
    return None, False


def analyze_run(index_row: dict[str, str]) -> dict[str, Any] | None:
    copied_dir = Path(index_row.get("copied_dir") or "")
    manifest = load_json(copied_dir / "manifest.json")
    runtime_config = manifest.get("runtime_config") or {}
    if not finite_bool(runtime_config.get("translation_alignatt_token_argmax_frontier_gate")):
        return None

    configured_patience = finite_int(
        runtime_config.get("translation_alignatt_token_argmax_frontier_patience_tokens"),
        default=1,
    )
    min_source_mass = finite_float(
        runtime_config.get("translation_alignatt_token_argmax_min_source_mass")
    )
    min_source_mass = 0.05 if min_source_mass is None else min_source_mass
    frontier_margin = finite_int(
        runtime_config.get("translation_alignatt_token_argmax_frontier_margin"),
        default=0,
    )

    update_count = 0
    stop_count = 0
    extra_draft_tokens: list[int] = []
    future_recovery_gaps: list[int] = []
    blocked_before_recovery_count = 0
    patience_gains: dict[int, list[int]] = {2: [], 3: [], 4: []}
    recoverable_gains: list[int] = []
    recoverable_full_draft_count = 0
    recoverable_unrecovered_suffix_count = 0
    recoverable_other_gate_blocks = 0
    blocked_positions: list[int] = []

    for update in iter_stream_updates(copied_dir / "stream_updates.jsonl") or []:
        update_count += 1
        metadata = update.get("alignatt_metadata") or {}
        if metadata.get("stop_reason") != "alignatt:token_argmax_source_frontier":
            continue
        stop_count += 1
        positions = int_positions(metadata.get("aligned_source_local_positions"))
        provenance = provenance_rows(metadata.get("provenance_per_draft_token"))
        accepted_count = finite_int(
            metadata.get("accepted_candidate_token_count"),
            default=0,
        )
        unsafe_index = finite_int(
            metadata.get("unsafe_target_token_index"),
            default=accepted_count,
        )
        blocked_position = finite_int(
            metadata.get("blocked_source_local_position"),
            default=-1,
        )
        if blocked_position >= 0:
            blocked_positions.append(blocked_position)
        extra_draft_tokens.append(max(0, len(positions) - accepted_count))

        recovery_gap, blocked_before_recovery = first_future_recovery_gap(
            positions,
            provenance=provenance,
            metadata=metadata,
            runtime_config=runtime_config,
            unsafe_index=unsafe_index,
        )
        if recovery_gap is not None:
            future_recovery_gaps.append(recovery_gap)
        if blocked_before_recovery:
            blocked_before_recovery_count += 1

        for patience in (2, 3, 4):
            simulated_accept_count, _ = simulate_token_argmax_patience_accept_count(
                positions,
                provenance=provenance,
                metadata=metadata,
                runtime_config=runtime_config,
                accepted_count=accepted_count,
                patience_tokens=patience,
            )
            gain = max(0, simulated_accept_count - accepted_count)
            if gain > 0:
                patience_gains[patience].append(gain)

        recoverable_accept_count, blocked_by_other_gate = (
            recoverable_token_argmax_accept_count(
                positions,
                provenance=provenance,
                metadata=metadata,
                runtime_config=runtime_config,
                accepted_count=accepted_count,
            )
        )
        recoverable_gain = max(0, recoverable_accept_count - accepted_count)
        if recoverable_gain > 0:
            recoverable_gains.append(recoverable_gain)
        if recoverable_accept_count >= len(positions):
            recoverable_full_draft_count += 1
        elif recoverable_accept_count > accepted_count:
            recoverable_unrecovered_suffix_count += 1
        if blocked_by_other_gate:
            recoverable_other_gate_blocks += 1

    if update_count <= 0:
        return None

    token_gain_sum = sum(recoverable_gains)
    return {
        "relative_dir": index_row.get("relative_dir") or copied_dir.name,
        "num_inputs": index_row.get("num_inputs") or manifest.get("num_inputs"),
        "chunk_ms": index_row.get("chunk_ms"),
        "valid_for_claims": index_row.get("valid_for_claims"),
        "alignatt_guard_flags": index_row.get("alignatt_guard_flags"),
        "longyaal_cu_ms": index_row.get("longyaal_cu_ms"),
        "xcometxl": index_row.get("xcometxl"),
        "token_argmax_frontier_stop_count": stop_count,
        "token_argmax_frontier_stop_ratio": (
            stop_count / float(update_count) if update_count else 0.0
        ),
        "mean_extra_draft_tokens_after_stop": (
            sum(extra_draft_tokens) / float(len(extra_draft_tokens))
            if extra_draft_tokens
            else 0.0
        ),
        "updates_with_future_recovery": len(future_recovery_gaps),
        "future_recovery_ratio": (
            len(future_recovery_gaps) / float(stop_count) if stop_count else 0.0
        ),
        "mean_future_recovery_token_gap": (
            sum(future_recovery_gaps) / float(len(future_recovery_gaps))
            if future_recovery_gaps
            else None
        ),
        "updates_blocked_by_other_gate_before_recovery": blocked_before_recovery_count,
        "simulated_patience2_gain_update_count": len(patience_gains[2]),
        "simulated_patience2_token_gain_sum": sum(patience_gains[2]),
        "simulated_patience3_gain_update_count": len(patience_gains[3]),
        "simulated_patience3_token_gain_sum": sum(patience_gains[3]),
        "simulated_patience4_gain_update_count": len(patience_gains[4]),
        "simulated_patience4_token_gain_sum": sum(patience_gains[4]),
        "simulated_recoverable_gain_update_count": len(recoverable_gains),
        "simulated_recoverable_token_gain_sum": token_gain_sum,
        "mean_simulated_recoverable_token_gain": (
            token_gain_sum / float(len(recoverable_gains))
            if recoverable_gains
            else None
        ),
        "simulated_recoverable_full_draft_count": recoverable_full_draft_count,
        "simulated_recoverable_unrecovered_suffix_count": (
            recoverable_unrecovered_suffix_count
        ),
        "simulated_recoverable_blocked_by_other_gate_count": (
            recoverable_other_gate_blocks
        ),
        "mean_blocked_position": (
            sum(blocked_positions) / float(len(blocked_positions))
            if blocked_positions
            else None
        ),
        "translation_alignatt_token_argmax_frontier_gate": True,
        "translation_alignatt_token_argmax_min_source_mass": min_source_mass,
        "translation_alignatt_token_argmax_frontier_margin": frontier_margin,
        "translation_alignatt_token_argmax_frontier_patience_tokens": (
            configured_patience
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
            -int(row["token_argmax_frontier_stop_count"]),
            -int(row["simulated_recoverable_token_gain_sum"]),
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
        "runs_with_token_argmax_frontier_stops": sum(
            1 for row in rows if int(row["token_argmax_frontier_stop_count"]) > 0
        ),
        "total_token_argmax_frontier_stops": sum(
            int(row["token_argmax_frontier_stop_count"]) for row in rows
        ),
        "total_future_recovery_updates": sum(
            int(row["updates_with_future_recovery"]) for row in rows
        ),
        "total_simulated_patience2_token_gain": sum(
            int(row["simulated_patience2_token_gain_sum"]) for row in rows
        ),
        "total_simulated_patience3_token_gain": sum(
            int(row["simulated_patience3_token_gain_sum"]) for row in rows
        ),
        "total_simulated_patience4_token_gain": sum(
            int(row["simulated_patience4_token_gain_sum"]) for row in rows
        ),
        "total_simulated_recoverable_token_gain": sum(
            int(row["simulated_recoverable_token_gain_sum"]) for row in rows
        ),
        "total_simulated_recoverable_gain_updates": sum(
            int(row["simulated_recoverable_gain_update_count"]) for row in rows
        ),
        "total_simulated_recoverable_other_gate_blocks": sum(
            int(row["simulated_recoverable_blocked_by_other_gate_count"])
            for row in rows
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
