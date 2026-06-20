#!/usr/bin/env python3
"""Summarize EN->ZH source-regression hard-stop recovery opportunities."""

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
    "alignatt_guard_flags",
    "longyaal_cu_ms",
    "xcometxl",
    "source_regression_stop_count",
    "source_regression_stop_ratio",
    "mean_extra_draft_tokens_after_stop",
    "updates_with_future_recovery",
    "future_recovery_ratio",
    "mean_future_recovery_token_gap",
    "simulated_trim_unrecovered_gain_update_count",
    "simulated_trim_unrecovered_token_gain_sum",
    "mean_simulated_trim_unrecovered_token_gain",
    "simulated_trim_unrecovered_full_draft_count",
    "simulated_trim_unrecovered_unrecovered_suffix_count",
    "simulated_gate_aware_gain_update_count",
    "simulated_gate_aware_token_gain_sum",
    "mean_simulated_gate_aware_token_gain",
    "simulated_gate_aware_blocked_by_other_gate_count",
    "mean_reference_position",
    "mean_blocked_position",
    "translation_alignatt_max_source_regression",
    "translation_alignatt_source_regression_recent_tokens",
    "translation_alignatt_source_regression_reference_mode",
    "translation_alignatt_source_regression_action",
    "translation_alignatt_source_regression_patience_tokens",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-stem",
        default="enzh_source_regression_diagnostics_20260607",
    )
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


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def median_int(values: list[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(int(value) for value in values)
    return int(ordered[(len(ordered) - 1) // 2])


def source_regression_reference_position(
    accepted_positions: list[int],
    *,
    max_accepted_position: int | None,
    recent_tokens: int,
    reference_mode: str,
) -> int | None:
    if recent_tokens > 0 and accepted_positions:
        recent = accepted_positions[-int(recent_tokens) :]
        if reference_mode == "median_recent":
            return median_int(recent)
        return max(recent)
    return max_accepted_position


def int_positions(values: Any) -> list[int | None]:
    if not isinstance(values, list):
        return []
    positions: list[int | None] = []
    for value in values:
        if value is None:
            positions.append(None)
            continue
        try:
            positions.append(int(value))
        except (TypeError, ValueError):
            positions.append(None)
    return positions


def finite_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def provenance_rows(values: Any) -> list[dict[str, float]]:
    if not isinstance(values, list):
        return []
    rows: list[dict[str, float]] = []
    for value in values:
        if not isinstance(value, dict):
            rows.append({})
            continue
        row: dict[str, float] = {}
        for key in ("source_accessible", "source_inaccessible", "non_source_prompt"):
            numeric = finite_float(value.get(key))
            if numeric is not None:
                row[key] = numeric
        rows.append(row)
    return rows


def token_passes_non_regression_gates(
    *,
    position: int | None,
    provenance: dict[str, float] | None,
    metadata: dict[str, Any],
    runtime_config: dict[str, Any],
    include_source_frontier_gate: bool = True,
    include_token_argmax_frontier_gate: bool = True,
) -> bool:
    if position is None:
        return True

    accessible_count = finite_int(
        metadata.get("accessible_source_local_end_exclusive"),
        default=0,
    )
    border_margin = finite_int(
        runtime_config.get("translation_alignatt_border_margin"),
        default=0,
    )
    source_inaccessible = None if provenance is None else provenance.get(
        "source_inaccessible"
    )
    if (
        include_source_frontier_gate
        and int(position) >= max(0, accessible_count) + border_margin
    ):
        frontier_min_inaccessible = finite_float(
            runtime_config.get("translation_alignatt_frontier_min_inaccessible_mass")
        )
        if not (
            frontier_min_inaccessible is not None
            and frontier_min_inaccessible > 0.0
            and source_inaccessible is not None
            and source_inaccessible < frontier_min_inaccessible
        ):
            return False

    source_accessible = None if provenance is None else provenance.get(
        "source_accessible"
    )
    source_mass = None
    if source_accessible is not None and source_inaccessible is not None:
        source_mass = source_accessible + source_inaccessible

    if (
        include_token_argmax_frontier_gate
        and finite_bool(runtime_config.get("translation_alignatt_token_argmax_frontier_gate"))
    ):
        token_argmax_min_source_mass = finite_float(
            runtime_config.get("translation_alignatt_token_argmax_min_source_mass")
        )
        token_argmax_min_source_mass = (
            0.05 if token_argmax_min_source_mass is None else token_argmax_min_source_mass
        )
        token_argmax_margin = finite_int(
            runtime_config.get("translation_alignatt_token_argmax_frontier_margin"),
            default=0,
        )
        if (
            source_mass is not None
            and source_mass >= token_argmax_min_source_mass
            and int(position) >= max(0, accessible_count) + token_argmax_margin
        ):
            return False

    min_source_mass = finite_float(
        runtime_config.get("translation_alignatt_min_source_mass")
    )
    if (
        min_source_mass is not None
        and min_source_mass > 0.0
        and source_accessible is not None
        and source_accessible < min_source_mass
    ):
        return False

    max_inaccessible = finite_float(
        runtime_config.get("translation_alignatt_max_inaccessible_source_mass")
    )
    if (
        max_inaccessible is not None
        and max_inaccessible < 1.0
        and source_inaccessible is not None
        and source_inaccessible > max_inaccessible
    ):
        return False

    max_non_source = finite_float(
        runtime_config.get("translation_alignatt_max_non_source_prompt_mass")
    )
    non_source = None if provenance is None else provenance.get("non_source_prompt")
    if (
        max_non_source is not None
        and max_non_source < 1.0
        and non_source is not None
        and non_source > max_non_source
    ):
        return False

    min_margin = finite_float(
        runtime_config.get("translation_alignatt_min_accessible_inaccessible_margin")
    )
    if (
        min_margin is not None
        and min_margin > -1.0
        and source_accessible is not None
        and source_inaccessible is not None
        and source_accessible - source_inaccessible < min_margin
    ):
        return False

    return True


def source_regression_trim_unrecovered_accept_count(
    positions: list[int | None],
    *,
    accepted_count: int,
    max_regression: int,
    recent_tokens: int,
    reference_mode: str,
    patience_tokens: int,
) -> int:
    accepted_positions = [
        int(position) for position in positions[: max(0, accepted_count)] if position is not None
    ]
    max_accepted_position = max(accepted_positions) if accepted_positions else None
    pending_trim_index: int | None = None
    regression_streak = 0
    for token_index in range(max(0, accepted_count), len(positions)):
        position = positions[token_index]
        reference = source_regression_reference_position(
            accepted_positions,
            max_accepted_position=max_accepted_position,
            recent_tokens=recent_tokens,
            reference_mode=reference_mode,
        )
        regressed = (
            position is not None
            and reference is not None
            and int(position) < int(reference) - int(max_regression)
        )
        if regressed:
            if pending_trim_index is None:
                pending_trim_index = token_index
            regression_streak += 1
            continue
        pending_trim_index = None
        regression_streak = 0
        if position is not None:
            accepted_positions.append(int(position))
            max_accepted_position = max(
                int(position),
                -1 if max_accepted_position is None else int(max_accepted_position),
            )
    if (
        pending_trim_index is not None
        and regression_streak >= max(1, int(patience_tokens))
    ):
        return pending_trim_index
    return len(positions)


def gate_aware_trim_unrecovered_accept_count(
    positions: list[int | None],
    *,
    provenance: list[dict[str, float]],
    metadata: dict[str, Any],
    runtime_config: dict[str, Any],
    accepted_count: int,
    max_regression: int,
    recent_tokens: int,
    reference_mode: str,
    patience_tokens: int,
) -> tuple[int, bool]:
    accepted_positions = [
        int(position) for position in positions[: max(0, accepted_count)] if position is not None
    ]
    max_accepted_position = max(accepted_positions) if accepted_positions else None
    pending_trim_index: int | None = None
    regression_streak = 0
    for token_index in range(max(0, accepted_count), len(positions)):
        position = positions[token_index]
        row = provenance[token_index] if token_index < len(provenance) else {}
        if not token_passes_non_regression_gates(
            position=position,
            provenance=row,
            metadata=metadata,
            runtime_config=runtime_config,
        ):
            return token_index, True

        reference = source_regression_reference_position(
            accepted_positions,
            max_accepted_position=max_accepted_position,
            recent_tokens=recent_tokens,
            reference_mode=reference_mode,
        )
        regressed = (
            position is not None
            and reference is not None
            and int(position) < int(reference) - int(max_regression)
        )
        if regressed:
            if pending_trim_index is None:
                pending_trim_index = token_index
            regression_streak += 1
            continue

        pending_trim_index = None
        regression_streak = 0
        if position is not None:
            accepted_positions.append(int(position))
            max_accepted_position = max(
                int(position),
                -1 if max_accepted_position is None else int(max_accepted_position),
            )
    if (
        pending_trim_index is not None
        and regression_streak >= max(1, int(patience_tokens))
    ):
        return pending_trim_index, False
    return len(positions), False


def iter_stream_updates(path: Path):
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def analyze_run(index_row: dict[str, str]) -> dict[str, Any] | None:
    copied_dir = Path(index_row.get("copied_dir") or "")
    manifest = load_json(copied_dir / "manifest.json")
    runtime_config = manifest.get("runtime_config") or {}
    max_regression = finite_int(
        runtime_config.get("translation_alignatt_max_source_regression"),
        default=-1,
    )
    if max_regression < 0:
        return None

    recent_tokens = finite_int(
        runtime_config.get("translation_alignatt_source_regression_recent_tokens"),
        default=0,
    )
    reference_mode = str(
        runtime_config.get("translation_alignatt_source_regression_reference_mode")
        or "max"
    )
    action = str(
        runtime_config.get("translation_alignatt_source_regression_action") or "stop"
    )
    patience_tokens = finite_int(
        runtime_config.get("translation_alignatt_source_regression_patience_tokens"),
        default=1,
    )

    update_count = 0
    stop_count = 0
    extra_draft_tokens: list[int] = []
    future_recovery_gaps: list[int] = []
    simulated_gains: list[int] = []
    simulated_full_draft_count = 0
    simulated_unrecovered_suffix_count = 0
    gate_aware_gains: list[int] = []
    gate_aware_other_gate_blocks = 0
    reference_positions: list[int] = []
    blocked_positions: list[int] = []
    for update in iter_stream_updates(copied_dir / "stream_updates.jsonl") or []:
        update_count += 1
        metadata = update.get("alignatt_metadata") or {}
        if metadata.get("stop_reason") != "alignatt:source_regression":
            continue
        stop_count += 1
        positions = int_positions(metadata.get("aligned_source_local_positions"))
        accepted_count = finite_int(
            metadata.get("accepted_candidate_token_count"),
            default=0,
        )
        unsafe_index = finite_int(
            metadata.get("unsafe_target_token_index"),
            default=accepted_count,
        )
        accepted_positions = [
            int(position)
            for position in positions[:accepted_count]
            if position is not None
        ]
        max_accepted_position = max(accepted_positions) if accepted_positions else None
        reference = source_regression_reference_position(
            accepted_positions,
            max_accepted_position=max_accepted_position,
            recent_tokens=recent_tokens,
            reference_mode=reference_mode,
        )
        blocked_position = metadata.get("blocked_source_local_position")
        blocked = finite_int(blocked_position, default=-1)
        if reference is not None:
            reference_positions.append(int(reference))
        if blocked >= 0:
            blocked_positions.append(blocked)
        extra_draft_tokens.append(max(0, len(positions) - accepted_count))
        if reference is None:
            continue
        recovery_floor = int(reference) - max_regression
        for offset, position in enumerate(positions[unsafe_index + 1 :], start=1):
            if position is not None and int(position) >= recovery_floor:
                future_recovery_gaps.append(offset)
                break
        simulated_accept_count = source_regression_trim_unrecovered_accept_count(
            positions,
            accepted_count=accepted_count,
            max_regression=max_regression,
            recent_tokens=recent_tokens,
            reference_mode=reference_mode,
            patience_tokens=patience_tokens,
        )
        gain = max(0, simulated_accept_count - accepted_count)
        if gain > 0:
            simulated_gains.append(gain)
        if simulated_accept_count >= len(positions):
            simulated_full_draft_count += 1
        elif simulated_accept_count > accepted_count:
            simulated_unrecovered_suffix_count += 1

        gate_aware_accept_count, blocked_by_other_gate = (
            gate_aware_trim_unrecovered_accept_count(
                positions,
                provenance=provenance_rows(metadata.get("provenance_per_draft_token")),
                metadata=metadata,
                runtime_config=runtime_config,
                accepted_count=accepted_count,
                max_regression=max_regression,
                recent_tokens=recent_tokens,
                reference_mode=reference_mode,
                patience_tokens=patience_tokens,
            )
        )
        gate_aware_gain = max(0, gate_aware_accept_count - accepted_count)
        if gate_aware_gain > 0:
            gate_aware_gains.append(gate_aware_gain)
        if blocked_by_other_gate:
            gate_aware_other_gate_blocks += 1

    if update_count <= 0:
        return None
    mean_extra = (
        sum(extra_draft_tokens) / float(len(extra_draft_tokens))
        if extra_draft_tokens
        else 0.0
    )
    mean_gap = (
        sum(future_recovery_gaps) / float(len(future_recovery_gaps))
        if future_recovery_gaps
        else None
    )
    token_gain_sum = sum(simulated_gains)
    gate_aware_token_gain_sum = sum(gate_aware_gains)
    return {
        "relative_dir": index_row.get("relative_dir") or copied_dir.name,
        "num_inputs": index_row.get("num_inputs") or manifest.get("num_inputs"),
        "chunk_ms": index_row.get("chunk_ms"),
        "valid_for_claims": index_row.get("valid_for_claims"),
        "alignatt_guard_flags": index_row.get("alignatt_guard_flags"),
        "longyaal_cu_ms": index_row.get("longyaal_cu_ms"),
        "xcometxl": index_row.get("xcometxl"),
        "source_regression_stop_count": stop_count,
        "source_regression_stop_ratio": stop_count / float(update_count),
        "mean_extra_draft_tokens_after_stop": mean_extra,
        "updates_with_future_recovery": len(future_recovery_gaps),
        "future_recovery_ratio": (
            len(future_recovery_gaps) / float(stop_count) if stop_count else 0.0
        ),
        "mean_future_recovery_token_gap": mean_gap,
        "simulated_trim_unrecovered_gain_update_count": len(simulated_gains),
        "simulated_trim_unrecovered_token_gain_sum": token_gain_sum,
        "mean_simulated_trim_unrecovered_token_gain": (
            token_gain_sum / float(len(simulated_gains))
            if simulated_gains
            else None
        ),
        "simulated_trim_unrecovered_full_draft_count": simulated_full_draft_count,
        "simulated_trim_unrecovered_unrecovered_suffix_count": (
            simulated_unrecovered_suffix_count
        ),
        "simulated_gate_aware_gain_update_count": len(gate_aware_gains),
        "simulated_gate_aware_token_gain_sum": gate_aware_token_gain_sum,
        "mean_simulated_gate_aware_token_gain": (
            gate_aware_token_gain_sum / float(len(gate_aware_gains))
            if gate_aware_gains
            else None
        ),
        "simulated_gate_aware_blocked_by_other_gate_count": (
            gate_aware_other_gate_blocks
        ),
        "mean_reference_position": (
            sum(reference_positions) / float(len(reference_positions))
            if reference_positions
            else None
        ),
        "mean_blocked_position": (
            sum(blocked_positions) / float(len(blocked_positions))
            if blocked_positions
            else None
        ),
        "translation_alignatt_max_source_regression": max_regression,
        "translation_alignatt_source_regression_recent_tokens": recent_tokens,
        "translation_alignatt_source_regression_reference_mode": reference_mode,
        "translation_alignatt_source_regression_action": action,
        "translation_alignatt_source_regression_patience_tokens": patience_tokens,
    }


def load_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


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
            -int(row["source_regression_stop_count"]),
            -float(row["future_recovery_ratio"]),
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
        "runs_with_source_regression_stops": sum(
            1 for row in rows if int(row["source_regression_stop_count"]) > 0
        ),
        "total_source_regression_stops": sum(
            int(row["source_regression_stop_count"]) for row in rows
        ),
        "total_future_recovery_updates": sum(
            int(row["updates_with_future_recovery"]) for row in rows
        ),
        "total_simulated_trim_unrecovered_token_gain": sum(
            int(row["simulated_trim_unrecovered_token_gain_sum"]) for row in rows
        ),
        "total_simulated_trim_unrecovered_gain_updates": sum(
            int(row["simulated_trim_unrecovered_gain_update_count"]) for row in rows
        ),
        "total_simulated_gate_aware_token_gain": sum(
            int(row["simulated_gate_aware_token_gain_sum"]) for row in rows
        ),
        "total_simulated_gate_aware_gain_updates": sum(
            int(row["simulated_gate_aware_gain_update_count"]) for row in rows
        ),
        "total_simulated_gate_aware_other_gate_blocks": sum(
            int(row["simulated_gate_aware_blocked_by_other_gate_count"])
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
