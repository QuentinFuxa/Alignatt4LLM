#!/usr/bin/env python3
"""Recover prior experiment artifacts and build a claim-safety index."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_ROOT = Path("outputs")
DEFAULT_OUTPUT_ROOT = Path("outputs/recovered_20260603_05")

EXACT_ARTIFACT_FILENAMES = (
    "manifest.json",
    "evaluation.json",
    "scores.tsv",
    "hypothesis.jsonl",
    "instances.resegmented.jsonl",
    "stream_updates.jsonl",
    "emission_events.jsonl",
    "chunk_decisions.jsonl",
    "summary.json",
    "policy_score_summary.json",
    "config.json",
    "records.jsonl",
    "predictions.txt",
)
GLOB_ARTIFACT_PATTERNS = ("*.log",)

INDEX_COLUMNS = (
    "relative_dir",
    "copied_dir",
    "valid_for_claims",
    "invalid_reasons",
    "artifact_files",
    "schema_version",
    "num_inputs",
    "source_language_code",
    "target_language_code",
    "mt_backend_name",
    "translation_acceptance_policy",
    "alignatt_policy_family",
    "alignatt_guard_flags",
    "translation_alignatt_acceptance_variant",
    "translation_alignatt_min_source_mass",
    "translation_alignatt_inaccessible_ms",
    "translation_alignatt_frontier_min_inaccessible_mass",
    "translation_alignatt_source_frontier_action",
    "translation_alignatt_source_regression_action",
    "translation_alignatt_max_non_source_prompt_mass",
    "translation_alignatt_min_accepted_accessible_source_mass",
    "translation_alignatt_accepted_accessible_source_mass_recent_units",
    "translation_alignatt_unit_consensus_min_head_ratio",
    "translation_alignatt_min_alignment_confidence",
    "translation_alignatt_source_bearing_min_source_mass",
    "translation_alignatt_source_bearing_hard_inaccessible_cap",
    "translation_alignatt_top_k_heads",
    "translation_alignatt_filter_width",
    "translation_alignatt_online_normalization",
    "translation_static_cutoff_units",
    "chunk_ms",
    "bleu",
    "chrf",
    "xcometxl",
    "longyaal_cu_ms",
    "longyaal_ca_ms",
    "run_script",
    "source_dir",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--index-name", default="recovered_artifact_index")
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Index artifacts in place without copying artifact files.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON artifact: {path}") from exc


def artifact_files(path: Path) -> list[Path]:
    files = [
        path / name for name in EXACT_ARTIFACT_FILENAMES if (path / name).is_file()
    ]
    for pattern in GLOB_ARTIFACT_PATTERNS:
        files.extend(
            sorted(candidate for candidate in path.glob(pattern) if candidate.is_file())
        )
    return sorted(files)


def iter_artifact_dirs(source_root: Path) -> list[Path]:
    dirs: list[Path] = []
    for path in sorted([source_root, *source_root.rglob("*")]):
        if path.is_dir() and artifact_files(path):
            dirs.append(path)
    return dirs


def copy_artifacts(source_dir: Path, dest_dir: Path) -> list[str]:
    copied: list[str] = []
    dest_dir.mkdir(parents=True, exist_ok=True)
    for source_file in artifact_files(source_dir):
        shutil.copy2(source_file, dest_dir / source_file.name)
        copied.append(source_file.name)
    return copied


def finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def parse_scores_tsv(path: Path) -> dict[str, float | None]:
    if not path.is_file():
        return {}
    scores: dict[str, float | None] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            metric = row.get("metric")
            if not metric:
                continue
            scores[metric] = finite_float(row.get("value"))
    return scores


def extract_scores(copied_dir: Path) -> dict[str, float | None]:
    evaluation = load_json(copied_dir / "evaluation.json")
    scores = dict(evaluation.get("contract_scores") or {})
    if not scores:
        scores.update(parse_scores_tsv(copied_dir / "scores.tsv"))
    if not scores:
        summary = load_json(copied_dir / "summary.json")
        metrics = summary.get("metrics") or {}
        scores = {
            "BLEU": metrics.get("bleu"),
            "CHRF": metrics.get("chrf"),
            "XCOMETXL": metrics.get("xcometxl"),
            "LongYAAL CU": metrics.get("longyaal_cu_ms"),
            "LongYAAL CA": metrics.get("longyaal_ca_ms"),
        }
    return {name: finite_float(value) for name, value in scores.items()}


def alignatt_guard_flags(runtime_config: dict[str, Any]) -> list[str]:
    """Return auxiliary policy knobs outside the clean frontier family."""
    if runtime_config.get("translation_acceptance_policy") != "alignatt":
        return []

    flags: list[str] = []
    if int(runtime_config.get("translation_alignatt_max_source_regression", -1)) >= 0:
        action = str(
            runtime_config.get("translation_alignatt_source_regression_action", "stop")
        )
        if action == "stop":
            flags.append("source_regression")
        else:
            flags.append(f"source_regression:{action}")
    if bool(runtime_config.get("translation_alignatt_token_argmax_frontier_gate", False)):
        flags.append("token_argmax_frontier")
    if int(runtime_config.get("translation_alignatt_min_accessible_source_units", 0)) > 0:
        flags.append("min_accessible_source_units")
    if bool(runtime_config.get("translation_alignatt_source_lcp_stability", False)):
        flags.append("source_lcp_stability")
    if int(runtime_config.get("translation_alignatt_source_lcp_append_slack_units", 0)) > 0:
        flags.append("source_lcp_append_slack")
    if float(runtime_config.get("translation_alignatt_max_inaccessible_source_mass", 1.0)) < 1.0:
        flags.append("max_inaccessible_source_mass")
    try:
        max_non_source_prompt_mass = float(
            runtime_config.get("translation_alignatt_max_non_source_prompt_mass", 1.0)
        )
    except (TypeError, ValueError):
        max_non_source_prompt_mass = 1.0
    if max_non_source_prompt_mass < 1.0:
        flags.append("max_non_source_prompt_mass")
    if float(runtime_config.get("translation_alignatt_min_accessible_inaccessible_margin", -1.0)) > -1.0:
        flags.append("accessible_inaccessible_margin")
    if float(runtime_config.get("translation_alignatt_argmax_mass_threshold", 0.0)) > 0.0:
        flags.append("argmax_mass_threshold")
    if int(runtime_config.get("translation_alignatt_hold_back_target_units", 0)) > 0:
        flags.append("target_holdback")
    if int(runtime_config.get("translation_alignatt_min_emit_target_units", 0)) > 0:
        flags.append("min_emit_target_units")
    if bool(runtime_config.get("translation_alignatt_source_lookback_holdback", False)):
        flags.append("source_lookback_holdback")
    if bool(runtime_config.get("translation_alignatt_defer_low_source_terminal_punctuation", False)):
        flags.append("terminal_punctuation_source_mass")

    variant = str(runtime_config.get("translation_alignatt_acceptance_variant", "token"))
    source_mass_floor = (
        float(runtime_config.get("translation_alignatt_min_source_mass", 0.0)) > 0.0
    )
    if variant == "unit_mass":
        if not source_mass_floor:
            flags.append("acceptance_variant:unit_mass_without_source_mass_floor")
    elif variant in {"unit_argmax", "unit_consensus", "unit_conf"}:
        if source_mass_floor:
            flags.append("unused_source_mass_floor")
    elif variant == "unit_mass_source_bearing":
        source_bearing_floor = (
            float(runtime_config.get("translation_alignatt_source_bearing_min_source_mass", 0.0))
            > 0.0
        )
        if source_mass_floor:
            flags.append("unit_source_bearing_with_source_mass_floor")
        if not source_bearing_floor:
            flags.append(
                "acceptance_variant:unit_mass_source_bearing_without_source_bearing_floor"
            )
        source_bearing_cap = float(
            runtime_config.get(
                "translation_alignatt_source_bearing_hard_inaccessible_cap",
                0.75,
            )
        )
        if source_bearing_cap < 1.0:
            flags.append(f"source_bearing_hard_inaccessible_cap:{source_bearing_cap:g}")
    elif variant != "token":
        flags.append(f"acceptance_variant:{variant}")
    return flags


def alignatt_policy_family(runtime_config: dict[str, Any]) -> str:
    policy = runtime_config.get("translation_acceptance_policy")
    if policy != "alignatt":
        return str(policy or "unknown")
    flags = alignatt_guard_flags(runtime_config)
    if flags:
        return "guarded_alignatt"
    source_mass_floor = (
        float(runtime_config.get("translation_alignatt_min_source_mass", 0.0)) > 0.0
    )
    accepted_prefix_source_mass_floor = (
        float(
            runtime_config.get(
                "translation_alignatt_min_accepted_accessible_source_mass",
                0.0,
            )
        )
        > 0.0
    )
    variant = str(runtime_config.get("translation_alignatt_acceptance_variant", "token"))
    soft_frontier = (
        float(runtime_config.get("translation_alignatt_frontier_min_inaccessible_mass", 0.0)) > 0.0
    )
    source_frontier_action = str(
        runtime_config.get("translation_alignatt_source_frontier_action", "stop")
    )
    recoverable_frontier = source_frontier_action == "trim_unrecovered"
    if variant == "unit_mass" and source_mass_floor:
        return "clean_unit_source_mass_floor"
    if variant == "unit_mass_source_bearing":
        return "clean_unit_source_bearing"
    if variant == "unit_argmax":
        return "clean_unit_argmax_frontier"
    if variant == "unit_consensus":
        return "clean_unit_consensus_frontier"
    if variant == "unit_conf":
        return "clean_unit_confidence_frontier"
    if soft_frontier and (source_mass_floor or accepted_prefix_source_mass_floor):
        if recoverable_frontier:
            return "clean_recoverable_soft_frontier_source_mass_floor"
        return "clean_soft_frontier_source_mass_floor"
    if source_mass_floor or accepted_prefix_source_mass_floor:
        if recoverable_frontier:
            return "clean_recoverable_argmax_frontier_source_mass_floor"
        return "clean_argmax_frontier_source_mass_floor"
    if soft_frontier:
        if recoverable_frontier:
            return "clean_recoverable_soft_frontier"
        return "pure_soft_frontier"
    if recoverable_frontier:
        return "clean_recoverable_argmax_frontier"
    return "pure_argmax_frontier"


def provenance_nonfinite_stop_ratio(manifest: dict[str, Any]) -> float | None:
    """Ratio of current-MT chunk decisions stopped by non-finite provenance.

    Batch manifests record per-input ``chunk_decision_summary.stop_reason_counts``.
    A non-trivial ``alignatt:provenance_nonfinite`` rate means the MT attention
    observer's captured q/k payload was corrupt for those chunks, so the run was
    not executing its declared acceptance policy (corrupt cycles act as an
    undeclared random throttle). Returns ``None`` when the manifest carries no
    decision summaries (older artifacts).
    """
    speed = manifest.get("speed") or {}
    per_input = speed.get("per_input") or speed.get("per_audio") or []
    bad = 0
    total = 0
    for entry in per_input:
        if not isinstance(entry, dict):
            continue
        counts = (entry.get("chunk_decision_summary") or {}).get("stop_reason_counts") or {}
        for reason, count in counts.items():
            try:
                count = int(count)
            except (TypeError, ValueError):
                continue
            total += count
            if str(reason) == "alignatt:provenance_nonfinite":
                bad += count
    if total <= 0:
        return None
    return bad / float(total)


# A claim-valid artifact must have executed its declared acceptance policy.
# Above this ratio of non-finite-provenance stops the observer capture was
# corrupt often enough to act as an undeclared latency guard.
PROVENANCE_NONFINITE_INVALID_RATIO = 0.01


def invalid_reasons(
    *,
    relative_dir: Path,
    copied_dir: Path,
    manifest: dict[str, Any],
    scores: dict[str, float | None],
) -> list[str]:
    reasons: list[str] = []
    has_manifest = (copied_dir / "manifest.json").is_file()
    has_score_file = (copied_dir / "evaluation.json").is_file() or (
        copied_dir / "scores.tsv"
    ).is_file()
    if not has_manifest:
        reasons.append("missing_manifest")
    if not has_score_file:
        reasons.append("missing_score_file")
    if "replay_sr" in relative_dir.as_posix():
        reasons.append("replay_sr_diagnostic")
    if "surface_dedup" in relative_dir.as_posix():
        reasons.append("surface_dedup_diagnostic")
    if "diagnostic" in relative_dir.as_posix():
        reasons.append("diagnostic_artifact")
    if "offline_replay" in relative_dir.as_posix():
        reasons.append("offline_replay_diagnostic")
    offline_replay = manifest.get("offline_replay") or {}
    if isinstance(offline_replay, dict) and bool(offline_replay.get("diagnostic_only")):
        reasons.append("diagnostic_only_manifest")
    longyaal_cu = scores.get("LongYAAL CU")
    longyaal_ca = scores.get("LongYAAL CA")
    if longyaal_cu is not None and longyaal_cu < 0.0:
        reasons.append("negative_longyaal_cu")
    if longyaal_ca is not None and longyaal_ca < 0.0:
        reasons.append("negative_longyaal_ca")
    nonfinite_ratio = provenance_nonfinite_stop_ratio(manifest)
    if (
        nonfinite_ratio is not None
        and nonfinite_ratio > PROVENANCE_NONFINITE_INVALID_RATIO
    ):
        reasons.append("provenance_nonfinite_capture_corruption")
    return reasons


def index_row(
    *,
    source_root: Path,
    source_dir: Path,
    copied_dir: Path,
    copied_files: list[str],
) -> dict[str, Any]:
    relative_dir = source_dir.relative_to(source_root)
    manifest = load_json(copied_dir / "manifest.json")
    runtime_config = manifest.get("runtime_config") or {}
    scores = extract_scores(copied_dir)
    reasons = invalid_reasons(
        relative_dir=relative_dir,
        copied_dir=copied_dir,
        manifest=manifest,
        scores=scores,
    )
    provenance = manifest.get("run_provenance") or {}
    guard_flags = alignatt_guard_flags(runtime_config)
    return {
        "relative_dir": relative_dir.as_posix(),
        "copied_dir": str(copied_dir),
        "valid_for_claims": not reasons,
        "invalid_reasons": ",".join(reasons),
        "artifact_files": ",".join(copied_files),
        "schema_version": manifest.get("schema_version"),
        "num_inputs": manifest.get("num_inputs"),
        "source_language_code": manifest.get("source_language_code"),
        "target_language_code": manifest.get("target_language_code"),
        "mt_backend_name": runtime_config.get("mt_backend_name"),
        "translation_acceptance_policy": runtime_config.get(
            "translation_acceptance_policy"
        ),
        "alignatt_policy_family": alignatt_policy_family(runtime_config),
        "alignatt_guard_flags": ",".join(guard_flags),
        "translation_alignatt_acceptance_variant": runtime_config.get(
            "translation_alignatt_acceptance_variant"
        ),
        "translation_alignatt_min_source_mass": runtime_config.get(
            "translation_alignatt_min_source_mass"
        ),
        "translation_alignatt_inaccessible_ms": runtime_config.get(
            "translation_alignatt_inaccessible_ms"
        ),
        "translation_alignatt_frontier_min_inaccessible_mass": runtime_config.get(
            "translation_alignatt_frontier_min_inaccessible_mass"
        ),
        "translation_alignatt_source_frontier_action": runtime_config.get(
            "translation_alignatt_source_frontier_action"
        ),
        "translation_alignatt_source_regression_action": runtime_config.get(
            "translation_alignatt_source_regression_action"
        ),
        "translation_alignatt_max_non_source_prompt_mass": runtime_config.get(
            "translation_alignatt_max_non_source_prompt_mass"
        ),
        "translation_alignatt_min_accepted_accessible_source_mass": runtime_config.get(
            "translation_alignatt_min_accepted_accessible_source_mass"
        ),
        "translation_alignatt_accepted_accessible_source_mass_recent_units": runtime_config.get(
            "translation_alignatt_accepted_accessible_source_mass_recent_units"
        ),
        "translation_alignatt_unit_consensus_min_head_ratio": runtime_config.get(
            "translation_alignatt_unit_consensus_min_head_ratio"
        ),
        "translation_alignatt_min_alignment_confidence": runtime_config.get(
            "translation_alignatt_min_alignment_confidence"
        ),
        "translation_alignatt_source_bearing_min_source_mass": runtime_config.get(
            "translation_alignatt_source_bearing_min_source_mass"
        ),
        "translation_alignatt_source_bearing_hard_inaccessible_cap": runtime_config.get(
            "translation_alignatt_source_bearing_hard_inaccessible_cap"
        ),
        "translation_alignatt_top_k_heads": runtime_config.get(
            "translation_alignatt_top_k_heads"
        ),
        "translation_alignatt_filter_width": runtime_config.get(
            "translation_alignatt_filter_width"
        ),
        "translation_alignatt_online_normalization": runtime_config.get(
            "translation_alignatt_online_normalization"
        ),
        "translation_static_cutoff_units": runtime_config.get(
            "translation_static_cutoff_units"
        ),
        "chunk_ms": runtime_config.get("chunk_ms"),
        "bleu": scores.get("BLEU"),
        "chrf": scores.get("CHRF"),
        "xcometxl": scores.get("XCOMETXL"),
        "longyaal_cu_ms": scores.get("LongYAAL CU"),
        "longyaal_ca_ms": scores.get("LongYAAL CA"),
        "run_script": provenance.get("script"),
        "source_dir": str(source_dir),
    }


def write_index(rows: list[dict[str, Any]], output_root: Path, index_name: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / f"{index_name}.json"
    tsv_path = output_root / f"{index_name}.tsv"
    json_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in INDEX_COLUMNS})


def recover_artifacts(
    *,
    source_root: Path,
    output_root: Path,
    index_name: str = "recovered_artifact_index",
    copy_files: bool = True,
) -> list[dict[str, Any]]:
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Missing source artifact root: {source_root}")
    rows: list[dict[str, Any]] = []
    for source_dir in iter_artifact_dirs(source_root):
        relative_dir = source_dir.relative_to(source_root)
        if copy_files:
            copied_dir = output_root / relative_dir
            copied_files = copy_artifacts(source_dir, copied_dir)
        else:
            copied_dir = source_dir
            copied_files = [
                source_file.name for source_file in artifact_files(source_dir)
            ]
        rows.append(
            index_row(
                source_root=source_root,
                source_dir=source_dir,
                copied_dir=copied_dir,
                copied_files=copied_files,
            )
        )
    rows.sort(key=lambda row: str(row["relative_dir"]))
    write_index(rows, output_root, index_name)
    return rows


def index_existing_artifacts(
    *,
    source_root: Path,
    output_root: Path,
    index_name: str = "recovered_artifact_index",
) -> list[dict[str, Any]]:
    return recover_artifacts(
        source_root=source_root,
        output_root=output_root,
        index_name=index_name,
        copy_files=False,
    )


def main() -> None:
    args = parse_args()
    rows = recover_artifacts(
        source_root=args.source_root,
        output_root=args.output_root,
        index_name=args.index_name,
        copy_files=not args.index_only,
    )
    valid_count = sum(1 for row in rows if row["valid_for_claims"])
    action = "Indexed" if args.index_only else "Recovered"
    print(
        f"{action} {len(rows)} artifact directories to {args.output_root} "
        f"({valid_count} valid_for_claims)."
    )
    print(args.output_root / f"{args.index_name}.tsv")


if __name__ == "__main__":
    main()
