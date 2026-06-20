#!/usr/bin/env python3
"""Replay recovered EN->ZH stream updates under clean source-mass thresholds.

This is a diagnostic replay, not a substitute for a real GPU run. It uses the
recorded draft provenance to truncate low-source continuations, then replays the
append-only public surface to produce hypothesis files that can be scored with
the local evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alignatt4llm.artifacts import (  # noqa: E402
    HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
    normalize_computation_aware_timestamps,
    utc_now_isoformat,
)
from alignatt4llm.emission import register_translation_timestamps, register_translation_words  # noqa: E402
from alignatt4llm.text_surface import (  # noqa: E402
    join_public_emission_units,
    prediction_text_from_target_surface,
    split_public_emission_units,
)
from tools.reports.report_enzh_source_mass_thresholds import (  # noqa: E402
    accepted_prefix_after_threshold,
    source_accessible_values,
    stable_prefix_after_threshold,
)


DEFAULT_ARTIFACT_DIR = (
    REPO_ROOT
    / "outputs"
    / "diagnostics_jarvislab_20260606"
    / "enzh_milmmt_chunk640_argmaxonly_soft003_mini3_20260606"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "outputs" / "diagnostics_jarvislab_20260606" / "offline_replay_source_mass"
)
DEFAULT_THRESHOLDS = (0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003)
UNIT_REPLAY_VARIANTS = {"unit_mass", "unit_mass_source_bearing"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--threshold", type=float, action="append", default=[])
    parser.add_argument("--target-lang-code", default="zh")
    parser.add_argument(
        "--replay-variant",
        default="token",
        choices=("token", "accepted_prefix", "unit_mass", "unit_mass_source_bearing"),
        help=(
            "Diagnostic policy to replay from recorded draft provenance. "
            "Unit variants use draft target-unit boundaries when present."
        ),
    )
    parser.add_argument(
        "--source-bearing-hard-inaccessible-cap",
        type=float,
        default=1.0,
        help=(
            "Hard future-source cap for `unit_mass_source_bearing` replay. "
            "Default 1.0 keeps the clean source-bearing replay uncapped; lower "
            "values are guarded diagnostics. Ignored by token and unit_mass "
            "replay."
        ),
    )
    parser.add_argument(
        "--allow-unit-mass-fallback-boundaries",
        action="store_true",
        help=(
            "Allow unit_mass replay to fall back to accepted-prefix target-unit "
            "boundaries when draft boundaries are absent. This is diagnostic "
            "only and should not be used for candidate promotion."
        ),
    )
    parser.add_argument(
        "--allow-unit-fallback-boundaries",
        action="store_true",
        help=(
            "Allow any unit replay variant to fall back to accepted-prefix "
            "target-unit boundaries when draft boundaries are absent. This is "
            "diagnostic only and should not be used for candidate promotion."
        ),
    )
    parser.add_argument(
        "--allow-surface-unit-boundaries",
        action="store_true",
        help=(
            "Allow unit replay variants to synthesize target-unit boundaries "
            "from the recorded partial_draft_target surface when draft token "
            "boundaries are absent. This is diagnostic only and must not be "
            "used for candidate promotion."
        ),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def threshold_tag(threshold: float) -> str:
    text = f"{float(threshold):.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def source_order(hypothesis_rows: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for row in hypothesis_rows:
        source = row.get("source")
        if isinstance(source, list) and source:
            names.append(str(source[0]))
        elif source:
            names.append(str(source))
    return names


def original_prefix_for_update(update: dict[str, Any]) -> str:
    translation = str(update.get("translation_text") or "")
    partial = str(update.get("partial_accepted_target") or "")
    if partial and translation.endswith(partial):
        return translation[: -len(partial)]
    return translation


def generated_unit_keep_count(metadata: dict[str, Any], *, token_prefix: int) -> int:
    ends = [
        int(end)
        for end in metadata.get("target_stability_unit_end_token_indices", [])
        if 0 < int(end) <= int(token_prefix)
    ]
    return len(ends)


def _positive_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        try:
            integer = int(item)
        except (TypeError, ValueError):
            continue
        if integer > 0:
            out.append(integer)
    return sorted(set(out))


def target_unit_end_indices_for_replay(
    metadata: dict[str, Any],
    *,
    replay_variant: str,
    surface_unit_end_indices: list[int] | None = None,
) -> list[int]:
    if replay_variant in UNIT_REPLAY_VARIANTS:
        draft_ends = _positive_int_list(
            metadata.get("draft_target_stability_unit_end_token_indices")
        )
        if draft_ends:
            return draft_ends
        if surface_unit_end_indices:
            return _positive_int_list(surface_unit_end_indices)
    return _positive_int_list(metadata.get("target_stability_unit_end_token_indices"))


def surface_unit_end_indices_for_replay(
    update: dict[str, Any],
    *,
    previous_partial_target: str,
    target_lang_code: str,
    provenance_count: int,
) -> list[int]:
    if provenance_count <= 0:
        return []
    draft = str(update.get("partial_draft_target") or "")
    if not draft:
        return []
    base = previous_partial_target if draft.startswith(previous_partial_target) else ""
    continuation = draft[len(base) :]
    continuation_units = split_public_emission_units(
        continuation,
        target_lang_code=target_lang_code,
    )
    unit_count = min(len(continuation_units), int(provenance_count))
    return list(range(1, unit_count + 1))


def has_unit_replay_draft_boundaries(stream_updates: list[dict[str, Any]]) -> bool:
    for update in stream_updates:
        metadata = update.get("alignatt_metadata") or {}
        if not isinstance(metadata, dict):
            continue
        if _positive_int_list(
            metadata.get("draft_target_stability_unit_end_token_indices")
        ):
            return True
    return False


def unit_mass_prefix_after_threshold(
    metadata: dict[str, Any],
    *,
    source_accessible: list[float | None],
    threshold: float,
    surface_unit_end_indices: list[int] | None = None,
) -> int:
    if bool(metadata.get("final_source_completed_full_accept")):
        return len(source_accessible)
    unit_ends = [
        end
        for end in target_unit_end_indices_for_replay(
            metadata,
            replay_variant="unit_mass",
            surface_unit_end_indices=surface_unit_end_indices,
        )
        if end <= len(source_accessible)
    ]
    accepted_end = 0
    unit_start = 0
    for unit_end in unit_ends:
        for value in source_accessible[unit_start:unit_end]:
            if value is None or float(value) < float(threshold):
                return accepted_end
        accepted_end = unit_end
        unit_start = unit_end
    return accepted_end


def source_bearing_values(
    provenance: Any,
) -> list[tuple[float | None, float | None]]:
    if not isinstance(provenance, list):
        return []
    values: list[tuple[float | None, float | None]] = []
    for row in provenance:
        if not isinstance(row, dict):
            values.append((None, None))
            continue
        accessible = row.get("source_accessible")
        inaccessible = row.get("source_inaccessible")
        try:
            accessible_value = float(accessible)
            inaccessible_value = float(inaccessible)
        except (TypeError, ValueError):
            values.append((None, None))
            continue
        values.append((accessible_value, inaccessible_value))
    return values


def unit_source_bearing_prefix_after_threshold(
    metadata: dict[str, Any],
    *,
    source_bearing: list[tuple[float | None, float | None]],
    threshold: float,
    hard_inaccessible_cap: float,
    surface_unit_end_indices: list[int] | None = None,
) -> int:
    if bool(metadata.get("final_source_completed_full_accept")):
        return len(source_bearing)
    unit_ends = [
        end
        for end in target_unit_end_indices_for_replay(
            metadata,
            replay_variant="unit_mass_source_bearing",
            surface_unit_end_indices=surface_unit_end_indices,
        )
        if end <= len(source_bearing)
    ]
    accepted_end = 0
    unit_start = 0
    for unit_end in unit_ends:
        source_bearing_token_seen = False
        for token_index, (accessible, inaccessible) in enumerate(
            source_bearing[unit_start:unit_end],
            start=unit_start,
        ):
            if accessible is None or inaccessible is None:
                return accepted_end
            source_mass = float(accessible) + float(inaccessible)
            if source_mass < float(threshold):
                continue
            if float(inaccessible) > float(hard_inaccessible_cap):
                return accepted_end
            if not source_bearing_frontier_safe(
                metadata,
                token_index=token_index,
                source_inaccessible_mass=float(inaccessible),
            ):
                return accepted_end
            source_bearing_token_seen = True
        if not source_bearing_token_seen:
            return accepted_end
        accepted_end = unit_end
        unit_start = unit_end
    return accepted_end


def source_bearing_frontier_safe(
    metadata: dict[str, Any],
    *,
    token_index: int,
    source_inaccessible_mass: float,
) -> bool:
    positions = metadata.get("aligned_source_local_positions")
    if not isinstance(positions, list) or int(token_index) >= len(positions):
        return False
    position = positions[int(token_index)]
    if position is None:
        return False
    try:
        current_source_local_position = int(position)
        accessible_source_end = int(metadata.get("accessible_source_local_end_exclusive"))
    except (TypeError, ValueError):
        return False
    try:
        border_margin = int(metadata.get("alignatt_unit_policy_border_margin", 0))
    except (TypeError, ValueError):
        border_margin = 0
    frontier = max(0, accessible_source_end) + border_margin
    if current_source_local_position < frontier:
        return True
    try:
        frontier_min_inaccessible_mass = float(
            metadata.get("alignatt_frontier_min_inaccessible_mass", 0.0)
        )
    except (TypeError, ValueError):
        frontier_min_inaccessible_mass = 0.0
    return (
        frontier_min_inaccessible_mass > 0.0
        and float(source_inaccessible_mass) < frontier_min_inaccessible_mass
    )


def replay_token_prefix_after_threshold(
    metadata: dict[str, Any],
    *,
    source_accessible: list[float | None],
    threshold: float,
    replay_variant: str,
    hard_inaccessible_cap: float = 1.0,
    surface_unit_end_indices: list[int] | None = None,
) -> int:
    if replay_variant == "unit_mass":
        return unit_mass_prefix_after_threshold(
            metadata,
            source_accessible=source_accessible,
            threshold=threshold,
            surface_unit_end_indices=surface_unit_end_indices,
        )
    if replay_variant == "unit_mass_source_bearing":
        source_bearing = source_bearing_values(metadata.get("provenance_per_draft_token"))
        return unit_source_bearing_prefix_after_threshold(
            metadata,
            source_bearing=source_bearing,
            threshold=threshold,
            hard_inaccessible_cap=hard_inaccessible_cap,
            surface_unit_end_indices=surface_unit_end_indices,
        )
    if replay_variant == "accepted_prefix":
        return accepted_prefix_after_threshold(
            metadata,
            source_accessible=source_accessible,
            threshold=float(threshold),
        )
    return stable_prefix_after_threshold(
        metadata,
        source_accessible=source_accessible,
        threshold=float(threshold),
    )


def unit_keep_count_for_replay(
    metadata: dict[str, Any],
    *,
    token_prefix: int,
    replay_variant: str,
    surface_unit_end_indices: list[int] | None = None,
) -> int:
    ends = [
        end
        for end in target_unit_end_indices_for_replay(
            metadata,
            replay_variant=replay_variant,
            surface_unit_end_indices=surface_unit_end_indices,
        )
        if end <= int(token_prefix)
    ]
    return len(ends)


def simulated_partial_target(
    update: dict[str, Any],
    *,
    previous_partial_target: str,
    threshold: float,
    target_lang_code: str,
    replay_variant: str,
    source_bearing_hard_inaccessible_cap: float = 1.0,
    allow_surface_unit_boundaries: bool = False,
) -> str:
    metadata = update.get("alignatt_metadata") or {}
    if not isinstance(metadata, dict):
        return str(update.get("partial_accepted_target") or "")
    if bool(metadata.get("final_source_completed_full_accept")):
        return str(update.get("partial_accepted_target") or "")
    source_values = source_accessible_values(metadata.get("provenance_per_draft_token"))
    if not source_values:
        return str(update.get("partial_accepted_target") or "")
    surface_unit_ends = (
        surface_unit_end_indices_for_replay(
            update,
            previous_partial_target=previous_partial_target,
            target_lang_code=target_lang_code,
            provenance_count=len(source_values),
        )
        if allow_surface_unit_boundaries and replay_variant in UNIT_REPLAY_VARIANTS
        else None
    )

    token_prefix = replay_token_prefix_after_threshold(
        metadata,
        source_accessible=source_values,
        threshold=float(threshold),
        replay_variant=replay_variant,
        hard_inaccessible_cap=source_bearing_hard_inaccessible_cap,
        surface_unit_end_indices=surface_unit_ends,
    )
    accepted_count = int(metadata.get("accepted_token_count") or 0)
    if replay_variant == "token" and token_prefix >= accepted_count:
        return str(update.get("partial_accepted_target") or "")

    draft = str(update.get("partial_draft_target") or "")
    base = previous_partial_target if draft.startswith(previous_partial_target) else ""
    continuation = draft[len(base) :]
    keep_units = unit_keep_count_for_replay(
        metadata,
        token_prefix=token_prefix,
        replay_variant=replay_variant,
        surface_unit_end_indices=surface_unit_ends,
    )
    continuation_units = split_public_emission_units(
        continuation,
        target_lang_code=target_lang_code,
    )
    return base + join_public_emission_units(
        continuation_units[:keep_units],
        target_lang_code=target_lang_code,
    )


def append_only_accepts(
    *,
    previous_translation: str,
    candidate_translation: str,
    target_lang_code: str,
) -> tuple[bool, list[str]]:
    previous_units = split_public_emission_units(
        previous_translation,
        target_lang_code=target_lang_code,
    )
    candidate_units = split_public_emission_units(
        candidate_translation,
        target_lang_code=target_lang_code,
    )
    if not candidate_units:
        return False, []
    if len(candidate_units) < len(previous_units):
        return False, []
    if candidate_units[: len(previous_units)] != previous_units:
        return False, []
    return len(candidate_units) > len(previous_units), candidate_units[len(previous_units) :]


def replay_updates_for_threshold(
    *,
    stream_updates: list[dict[str, Any]],
    original_hypothesis: list[dict[str, Any]],
    threshold: float,
    target_lang_code: str,
    replay_variant: str = "token",
    source_bearing_hard_inaccessible_cap: float = 1.0,
    allow_surface_unit_boundaries: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_input: dict[str, list[dict[str, Any]]] = {}
    for update in stream_updates:
        by_input.setdefault(str(update.get("input_name") or ""), []).append(update)

    original_by_input = {
        str(row["source"][0] if isinstance(row.get("source"), list) else row.get("source")): row
        for row in original_hypothesis
    }
    hypothesis_rows: list[dict[str, Any]] = []
    summary = {
        "threshold": float(threshold),
        "input_count": 0,
        "accepted_update_count": 0,
        "rejected_update_count": 0,
        "changed_partial_update_count": 0,
        "trimmed_partial_update_count": 0,
        "expanded_partial_update_count": 0,
        "emptied_partial_update_count": 0,
        "replay_variant": replay_variant,
        "unit_mass_draft_boundary_update_count": 0,
        "unit_mass_fallback_boundary_update_count": 0,
        "unit_replay_draft_boundary_update_count": 0,
        "unit_replay_fallback_boundary_update_count": 0,
        "unit_replay_surface_boundary_update_count": 0,
        "source_bearing_hard_inaccessible_cap": (
            float(source_bearing_hard_inaccessible_cap)
            if replay_variant == "unit_mass_source_bearing"
            else None
        ),
    }
    for input_name in source_order(original_hypothesis):
        updates = by_input.get(input_name, [])
        emitted_translation = ""
        previous_raw_translation = ""
        previous_partial_target = ""
        word_delays_ms: list[float] = []
        word_elapsed_ms: list[float] = []
        accepted_updates: list[dict[str, Any]] = []
        for update in updates:
            metadata = update.get("alignatt_metadata") or {}
            if replay_variant in UNIT_REPLAY_VARIANTS and isinstance(metadata, dict):
                if _positive_int_list(
                    metadata.get("draft_target_stability_unit_end_token_indices")
                ):
                    summary["unit_replay_draft_boundary_update_count"] += 1
                    if replay_variant == "unit_mass":
                        summary["unit_mass_draft_boundary_update_count"] += 1
                elif allow_surface_unit_boundaries and surface_unit_end_indices_for_replay(
                    update,
                    previous_partial_target=previous_partial_target,
                    target_lang_code=target_lang_code,
                    provenance_count=len(
                        source_accessible_values(
                            metadata.get("provenance_per_draft_token")
                        )
                    ),
                ):
                    summary["unit_replay_surface_boundary_update_count"] += 1
                else:
                    summary["unit_replay_fallback_boundary_update_count"] += 1
                    if replay_variant == "unit_mass":
                        summary["unit_mass_fallback_boundary_update_count"] += 1
            simulated_partial = simulated_partial_target(
                update,
                previous_partial_target=previous_partial_target,
                threshold=threshold,
                target_lang_code=target_lang_code,
                replay_variant=replay_variant,
                source_bearing_hard_inaccessible_cap=(
                    source_bearing_hard_inaccessible_cap
                ),
                allow_surface_unit_boundaries=allow_surface_unit_boundaries,
            )
            original_partial = str(update.get("partial_accepted_target") or "")
            if simulated_partial != original_partial:
                summary["changed_partial_update_count"] += 1
                original_units = split_public_emission_units(
                    original_partial,
                    target_lang_code=target_lang_code,
                )
                simulated_units = split_public_emission_units(
                    simulated_partial,
                    target_lang_code=target_lang_code,
                )
                if len(simulated_units) < len(original_units):
                    summary["trimmed_partial_update_count"] += 1
                elif len(simulated_units) > len(original_units):
                    summary["expanded_partial_update_count"] += 1
            if original_partial and not simulated_partial:
                summary["emptied_partial_update_count"] += 1
            previous_partial_target = simulated_partial
            candidate_translation = original_prefix_for_update(update) + simulated_partial
            accepts, added_units = append_only_accepts(
                previous_translation=emitted_translation,
                candidate_translation=candidate_translation,
                target_lang_code=target_lang_code,
            )
            if not accepts:
                summary["rejected_update_count"] += 1
                continue
            register_translation_timestamps(
                previous_raw_translation,
                candidate_translation,
                float(update["wallclock_elapsed_ms"]),
                word_elapsed_ms,
                target_lang_code=target_lang_code,
            )
            new_words = register_translation_words(
                emitted_translation,
                candidate_translation,
                float(update["audio_processed_ms"]),
                word_delays_ms,
                target_lang_code=target_lang_code,
            )
            emitted_translation = candidate_translation
            previous_raw_translation = candidate_translation
            accepted = dict(update)
            accepted["translation_text"] = candidate_translation
            accepted["new_words"] = new_words
            accepted["offline_replay_threshold"] = float(threshold)
            accepted["offline_replay_variant"] = replay_variant
            accepted["offline_replay_added_units"] = added_units
            accepted_updates.append(accepted)
            summary["accepted_update_count"] += 1

        original = original_by_input[input_name]
        normalized_elapsed_ms = normalize_computation_aware_timestamps(
            word_delays_ms,
            word_elapsed_ms,
        )
        hypothesis_rows.append(
            {
                "source": [input_name],
                "source_length": original["source_length"],
                "prediction": prediction_text_from_target_surface(
                    emitted_translation,
                    target_lang_code=target_lang_code,
                ),
                "delays": word_delays_ms,
                "elapsed": normalized_elapsed_ms,
                "elapsed_wallclock_ms": word_elapsed_ms,
                "elapsed_semantics": HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
                "offline_replay_update_count": len(accepted_updates),
            }
        )
        summary["input_count"] += 1
    return hypothesis_rows, summary


def write_replay_artifacts(
    *,
    output_dir: Path,
    artifact_dir: Path,
    threshold: float,
    target_lang_code: str,
    replay_variant: str,
    source_bearing_hard_inaccessible_cap: float,
    allow_surface_unit_boundaries: bool,
    hypothesis_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "hypothesis.jsonl", hypothesis_rows)
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    runtime_config = dict(manifest.get("runtime_config") or {})
    if replay_variant == "unit_mass_source_bearing":
        runtime_config["translation_alignatt_min_source_mass"] = 0.0
        runtime_config["translation_alignatt_source_bearing_min_source_mass"] = float(
            threshold
        )
        runtime_config["translation_alignatt_source_bearing_hard_inaccessible_cap"] = (
            float(source_bearing_hard_inaccessible_cap)
        )
        runtime_config["translation_alignatt_acceptance_variant"] = (
            "unit_mass_source_bearing"
        )
    elif replay_variant == "accepted_prefix":
        runtime_config["translation_alignatt_acceptance_variant"] = "token"
        runtime_config["translation_alignatt_min_source_mass"] = 0.0
        runtime_config["translation_alignatt_min_accepted_accessible_source_mass"] = (
            float(threshold)
        )
    else:
        runtime_config["translation_alignatt_min_source_mass"] = float(threshold)
        runtime_config["translation_alignatt_acceptance_variant"] = replay_variant
    runtime_config["offline_replay_from"] = str(artifact_dir)
    manifest.update(
        {
            "generated_at_utc": utc_now_isoformat(),
            "num_inputs": len(hypothesis_rows),
            "target_language_code": target_lang_code,
            "runtime_config": runtime_config,
            "offline_replay": {
                "kind": "source_mass_threshold",
                "variant": replay_variant,
                "threshold": float(threshold),
                "source_bearing_hard_inaccessible_cap": (
                    float(source_bearing_hard_inaccessible_cap)
                    if replay_variant == "unit_mass_source_bearing"
                    else None
                ),
                "source_artifact_dir": str(artifact_dir),
                "surface_unit_boundaries": bool(allow_surface_unit_boundaries),
                "diagnostic_only": True,
            },
        }
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "offline_replay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    stream_updates = load_jsonl(artifact_dir / "stream_updates.jsonl")
    if (
        args.replay_variant in UNIT_REPLAY_VARIANTS
        and not args.allow_unit_mass_fallback_boundaries
        and not args.allow_unit_fallback_boundaries
        and not args.allow_surface_unit_boundaries
        and not has_unit_replay_draft_boundaries(stream_updates)
    ):
        raise ValueError(
            f"{args.replay_variant} replay requires "
            "draft_target_stability_unit_end_token_indices. Use "
            "--allow-unit-fallback-boundaries only for explicit diagnostic "
            "fallback replays."
        )
    original_hypothesis = load_jsonl(artifact_dir / "hypothesis.jsonl")
    thresholds = tuple(args.threshold) if args.threshold else DEFAULT_THRESHOLDS
    summaries: list[dict[str, Any]] = []
    for threshold in thresholds:
        hypothesis_rows, summary = replay_updates_for_threshold(
            stream_updates=stream_updates,
            original_hypothesis=original_hypothesis,
            threshold=float(threshold),
            target_lang_code=args.target_lang_code,
            replay_variant=args.replay_variant,
            source_bearing_hard_inaccessible_cap=(
                args.source_bearing_hard_inaccessible_cap
            ),
            allow_surface_unit_boundaries=args.allow_surface_unit_boundaries,
        )
        output_prefix = "source_mass"
        if args.replay_variant != "token":
            output_prefix = args.replay_variant
        output_dir = args.output_root / f"{output_prefix}_{threshold_tag(float(threshold))}"
        write_replay_artifacts(
            output_dir=output_dir,
            artifact_dir=artifact_dir,
            threshold=float(threshold),
            target_lang_code=args.target_lang_code,
            replay_variant=args.replay_variant,
            source_bearing_hard_inaccessible_cap=(
                args.source_bearing_hard_inaccessible_cap
            ),
            allow_surface_unit_boundaries=args.allow_surface_unit_boundaries,
            hypothesis_rows=hypothesis_rows,
            summary=summary,
        )
        summaries.append({"output_dir": str(output_dir), **summary})
        print(output_dir)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "offline_replay_points.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output_root / "offline_replay_points.json")


if __name__ == "__main__":
    main()
