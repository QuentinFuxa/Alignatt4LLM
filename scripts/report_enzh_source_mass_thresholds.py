#!/usr/bin/env python3
"""Estimate clean AlignAtt source-mass thresholds from recovered stream traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = (
    REPO_ROOT
    / "outputs"
    / "diagnostics_jarvislab_20260606"
    / "diagnostics_artifact_index.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "plots"
DEFAULT_THRESHOLDS = (0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003)

REPORT_COLUMNS = (
    "relative_dir",
    "num_inputs",
    "chunk_ms",
    "longyaal_cu_ms",
    "xcometxl",
    "alignatt_policy_family",
    "alignatt_guard_flags",
    "configured_min_source_mass",
    "threshold",
    "stream_update_count",
    "updates_with_provenance",
    "updates_with_draft_unit_boundaries",
    "updates_without_draft_unit_boundaries",
    "draft_target_stability_unit_count_total",
    "original_accepted_token_count",
    "simulated_accepted_token_count",
    "simulated_retained_token_ratio",
    "accepted_prefix_simulated_token_count",
    "accepted_prefix_retained_token_ratio",
    "updates_trimmed_by_threshold",
    "updates_emptied_by_threshold",
    "updates_trimmed_by_accepted_prefix",
    "updates_emptied_by_accepted_prefix",
    "accepted_tokens_below_threshold",
    "accepted_tokens_source_mass_below_threshold",
    "accepted_tokens_non_source_dominant",
    "accepted_tokens_non_source_ge_0p80",
    "accepted_tokens_suffix_ge_0p10",
    "accepted_token_mean_source_accessible",
    "accepted_token_mean_source_mass",
    "accepted_token_mean_non_source_prompt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-stem",
        default="enzh_source_mass_thresholds",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        action="append",
        default=[],
        help="Source-accessible mass threshold to simulate; may be repeated.",
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


def finite_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def candidate_index_rows(index_path: Path) -> list[dict[str, Any]]:
    rows = json.loads(index_path.read_text(encoding="utf-8"))
    selected: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("valid_for_claims"):
            continue
        if row.get("target_language_code") != "zh":
            continue
        if row.get("mt_backend_name") != "milmmt_vllm_alignatt":
            continue
        copied_dir = Path(str(row.get("copied_dir") or ""))
        if not (copied_dir / "stream_updates.jsonl").is_file():
            continue
        selected.append(row)
    selected.sort(
        key=lambda row: (
            int(row.get("num_inputs") or 0) < 3,
            finite_float(row.get("longyaal_cu_ms")) or 999999.0,
            str(row.get("relative_dir") or ""),
        )
    )
    return selected


def source_accessible_values(provenance: Any) -> list[float | None]:
    if not isinstance(provenance, list):
        return []
    values: list[float | None] = []
    for row in provenance:
        if not isinstance(row, dict):
            values.append(None)
            continue
        values.append(finite_float(row.get("source_accessible")))
    return values


def provenance_value(
    row: Any,
    key: str,
) -> float | None:
    if not isinstance(row, dict):
        return None
    return finite_float(row.get(key))


def source_mass_value(row: Any) -> float | None:
    accessible = provenance_value(row, "source_accessible")
    inaccessible = provenance_value(row, "source_inaccessible")
    if accessible is None and inaccessible is None:
        return None
    return float(accessible or 0.0) + float(inaccessible or 0.0)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def safe_candidate_count(metadata: dict[str, Any], provenance_count: int) -> int:
    candidate_count = finite_int(metadata.get("accepted_candidate_token_count"))
    if candidate_count is None:
        candidate_count = finite_int(metadata.get("accepted_token_count"))
    if candidate_count is None:
        candidate_count = provenance_count
    return max(0, min(int(candidate_count), int(provenance_count)))


def stable_prefix_after_threshold(
    metadata: dict[str, Any],
    *,
    source_accessible: list[float | None],
    threshold: float,
) -> int:
    candidate_count = safe_candidate_count(metadata, len(source_accessible))
    prefix_count = candidate_count
    for index, value in enumerate(source_accessible[:candidate_count]):
        if value is None or float(value) < float(threshold):
            prefix_count = index
            break
    if bool(metadata.get("final_source_completed_full_accept")):
        return candidate_count
    unit_ends = [
        int(end)
        for end in metadata.get("target_stability_unit_end_token_indices", [])
        if 0 < int(end) <= prefix_count
    ]
    return max(unit_ends) if unit_ends else 0


def mean_accessible_range(
    source_accessible: list[float | None],
    *,
    start: int,
    end: int,
) -> float | None:
    start_index = max(0, int(start))
    end_index = int(end)
    if end_index <= start_index or len(source_accessible) < end_index:
        return None
    values = source_accessible[start_index:end_index]
    if any(value is None for value in values):
        return None
    return sum(float(value) for value in values) / float(len(values))


def accepted_prefix_after_threshold(
    metadata: dict[str, Any],
    *,
    source_accessible: list[float | None],
    threshold: float,
    recent_units: int = 2,
) -> int:
    candidate_count = safe_candidate_count(metadata, len(source_accessible))
    if bool(metadata.get("final_source_completed_full_accept")):
        return candidate_count
    unit_ends = [
        int(end)
        for end in metadata.get("target_stability_unit_end_token_indices", [])
        if 0 < int(end) <= candidate_count
    ]
    if not unit_ends:
        return 0

    def prefix_passes(end: int) -> bool:
        mean_accessible = mean_accessible_range(
            source_accessible,
            start=0,
            end=end,
        )
        if mean_accessible is None or mean_accessible < threshold:
            return False
        prefix_unit_ends = [unit_end for unit_end in unit_ends if unit_end <= end]
        if not prefix_unit_ends:
            return False
        units = int(recent_units)
        if units <= 0:
            return True
        recent_start = (
            0
            if len(prefix_unit_ends) <= units
            else prefix_unit_ends[-units - 1]
        )
        recent_mean_accessible = mean_accessible_range(
            source_accessible,
            start=recent_start,
            end=end,
        )
        if recent_mean_accessible is None or recent_mean_accessible < threshold:
            return False
        unit_starts = [0, *prefix_unit_ends[:-1]]
        for unit_start, unit_end in list(zip(unit_starts, prefix_unit_ends))[-units:]:
            unit_mean = mean_accessible_range(
                source_accessible,
                start=unit_start,
                end=unit_end,
            )
            if unit_mean is None or unit_mean < threshold:
                return False
        return True

    if prefix_passes(candidate_count):
        return candidate_count
    for end in reversed(unit_ends):
        if end >= candidate_count:
            continue
        if prefix_passes(end):
            return end
    return 0


def summarize_threshold(
    index_row: dict[str, Any],
    stream_updates: list[dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    update_count = len(stream_updates)
    updates_with_provenance = 0
    updates_with_draft_unit_boundaries = 0
    updates_without_draft_unit_boundaries = 0
    draft_target_stability_unit_count_total = 0
    original_accepted_total = 0
    simulated_accepted_total = 0
    accepted_prefix_simulated_total = 0
    trimmed_updates = 0
    emptied_updates = 0
    accepted_prefix_trimmed_updates = 0
    accepted_prefix_emptied_updates = 0
    accepted_tokens_below_threshold = 0
    accepted_tokens_source_mass_below_threshold = 0
    accepted_tokens_non_source_dominant = 0
    accepted_tokens_non_source_ge_0p80 = 0
    accepted_tokens_suffix_ge_0p10 = 0
    accepted_source_accessible_values: list[float] = []
    accepted_source_mass_values: list[float] = []
    accepted_non_source_values: list[float] = []
    for update in stream_updates:
        metadata = update.get("alignatt_metadata") or {}
        if not isinstance(metadata, dict):
            continue
        provenance_rows = metadata.get("provenance_per_draft_token")
        source_values = source_accessible_values(provenance_rows)
        if not source_values:
            continue
        updates_with_provenance += 1
        draft_unit_ends: list[int] = []
        draft_unit_end_values = metadata.get(
            "draft_target_stability_unit_end_token_indices"
        )
        if not isinstance(draft_unit_end_values, list):
            draft_unit_end_values = []
        for end in draft_unit_end_values:
            parsed_end = finite_int(end)
            if parsed_end is not None and parsed_end > 0:
                draft_unit_ends.append(parsed_end)
        if draft_unit_ends:
            updates_with_draft_unit_boundaries += 1
            draft_target_stability_unit_count_total += len(set(draft_unit_ends))
        else:
            updates_without_draft_unit_boundaries += 1
        original_count = max(0, finite_int(metadata.get("accepted_token_count")) or 0)
        simulated_count = stable_prefix_after_threshold(
            metadata,
            source_accessible=source_values,
            threshold=threshold,
        )
        accepted_prefix_count = accepted_prefix_after_threshold(
            metadata,
            source_accessible=source_values,
            threshold=threshold,
        )
        original_accepted_total += original_count
        simulated_accepted_total += simulated_count
        accepted_prefix_simulated_total += accepted_prefix_count
        if simulated_count < original_count:
            trimmed_updates += 1
        if original_count > 0 and simulated_count == 0:
            emptied_updates += 1
        if accepted_prefix_count < original_count:
            accepted_prefix_trimmed_updates += 1
        if original_count > 0 and accepted_prefix_count == 0:
            accepted_prefix_emptied_updates += 1
        accepted_provenance_rows = (
            provenance_rows[:original_count] if isinstance(provenance_rows, list) else []
        )
        for index, value in enumerate(source_values[:original_count]):
            if value is None or float(value) < float(threshold):
                accepted_tokens_below_threshold += 1
            if value is not None:
                accepted_source_accessible_values.append(float(value))
            source_mass = (
                source_mass_value(accepted_provenance_rows[index])
                if index < len(accepted_provenance_rows)
                else None
            )
            if source_mass is None or float(source_mass) < float(threshold):
                accepted_tokens_source_mass_below_threshold += 1
            if source_mass is not None:
                accepted_source_mass_values.append(float(source_mass))
            non_source = (
                provenance_value(accepted_provenance_rows[index], "non_source_prompt")
                if index < len(accepted_provenance_rows)
                else None
            )
            if non_source is not None:
                accepted_non_source_values.append(float(non_source))
                if source_mass is not None and float(non_source) > float(source_mass):
                    accepted_tokens_non_source_dominant += 1
                if float(non_source) >= 0.80:
                    accepted_tokens_non_source_ge_0p80 += 1
            suffix = (
                provenance_value(accepted_provenance_rows[index], "suffix")
                if index < len(accepted_provenance_rows)
                else None
            )
            if suffix is not None and float(suffix) >= 0.10:
                accepted_tokens_suffix_ge_0p10 += 1
    retained_ratio = (
        None
        if original_accepted_total <= 0
        else simulated_accepted_total / float(original_accepted_total)
    )
    accepted_prefix_retained_ratio = (
        None
        if original_accepted_total <= 0
        else accepted_prefix_simulated_total / float(original_accepted_total)
    )
    return {
        "relative_dir": index_row.get("relative_dir") or "",
        "num_inputs": index_row.get("num_inputs"),
        "chunk_ms": index_row.get("chunk_ms"),
        "longyaal_cu_ms": finite_float(index_row.get("longyaal_cu_ms")),
        "xcometxl": finite_float(index_row.get("xcometxl")),
        "alignatt_policy_family": index_row.get("alignatt_policy_family") or "",
        "alignatt_guard_flags": index_row.get("alignatt_guard_flags") or "",
        "configured_min_source_mass": finite_float(
            index_row.get("translation_alignatt_min_source_mass")
        ),
        "threshold": float(threshold),
        "stream_update_count": update_count,
        "updates_with_provenance": updates_with_provenance,
        "updates_with_draft_unit_boundaries": updates_with_draft_unit_boundaries,
        "updates_without_draft_unit_boundaries": updates_without_draft_unit_boundaries,
        "draft_target_stability_unit_count_total": (
            draft_target_stability_unit_count_total
        ),
        "original_accepted_token_count": original_accepted_total,
        "simulated_accepted_token_count": simulated_accepted_total,
        "simulated_retained_token_ratio": retained_ratio,
        "accepted_prefix_simulated_token_count": accepted_prefix_simulated_total,
        "accepted_prefix_retained_token_ratio": accepted_prefix_retained_ratio,
        "updates_trimmed_by_threshold": trimmed_updates,
        "updates_emptied_by_threshold": emptied_updates,
        "updates_trimmed_by_accepted_prefix": accepted_prefix_trimmed_updates,
        "updates_emptied_by_accepted_prefix": accepted_prefix_emptied_updates,
        "accepted_tokens_below_threshold": accepted_tokens_below_threshold,
        "accepted_tokens_source_mass_below_threshold": (
            accepted_tokens_source_mass_below_threshold
        ),
        "accepted_tokens_non_source_dominant": accepted_tokens_non_source_dominant,
        "accepted_tokens_non_source_ge_0p80": accepted_tokens_non_source_ge_0p80,
        "accepted_tokens_suffix_ge_0p10": accepted_tokens_suffix_ge_0p10,
        "accepted_token_mean_source_accessible": mean(
            accepted_source_accessible_values
        ),
        "accepted_token_mean_source_mass": mean(accepted_source_mass_values),
        "accepted_token_mean_non_source_prompt": mean(accepted_non_source_values),
    }


def format_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def write_outputs(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    output_stem: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = output_dir / f"{output_stem}.tsv"
    json_path = output_dir / f"{output_stem}.json"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: format_value(row.get(column)) for column in REPORT_COLUMNS}
            )
    json_path.write_text(
        json.dumps({"summary": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"tsv": tsv_path, "json": json_path}


def main() -> None:
    args = parse_args()
    thresholds = tuple(args.threshold) if args.threshold else DEFAULT_THRESHOLDS
    rows: list[dict[str, Any]] = []
    for index_row in candidate_index_rows(args.index):
        stream_path = Path(str(index_row["copied_dir"])) / "stream_updates.jsonl"
        stream_updates = load_jsonl(stream_path)
        for threshold in thresholds:
            rows.append(
                summarize_threshold(
                    index_row,
                    stream_updates,
                    threshold=float(threshold),
                )
            )
    if not rows:
        raise SystemExit("No matching EN->ZH MiLMMT stream updates found.")
    paths = write_outputs(rows, output_dir=args.output_dir, output_stem=args.output_stem)
    print(paths["tsv"])
    print(paths["json"])


if __name__ == "__main__":
    main()
