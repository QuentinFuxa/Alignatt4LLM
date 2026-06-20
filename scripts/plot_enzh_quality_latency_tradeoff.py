#!/usr/bin/env python3
"""Plot EN->ZH XCOMET/LongYAAL tradeoffs against the public IWSLT baseline."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECOVERED_INDEX = (
    REPO_ROOT
    / "outputs"
    / "diagnostics_jarvislab_20260606"
    / "diagnostics_artifact_index.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "plots"
BASELINE_SOURCE_URL = (
    "https://github.com/owaski/iwslt-2026-baselines/blob/master/"
    "quality_latency_tradeoff.png"
)
BASELINE_SOURCE_COMMIT = "7e2974bb3c850fde9bd62f3fa3103f9f345a56d0"

# Digitized from the EN->ZH panel of the public quality_latency_tradeoff.png.
# The source plot labels the five segment sizes but does not publish eval.txt
# files in the GitHub repository, so keep this compact table explicit and
# source-linked.
BASELINE_ENZH_POINTS = {
    "baseline": [
        {"segment_ms": 640, "longyaal_cu_ms": 1760.0, "xcometxl": 74.9},
        {"segment_ms": 960, "longyaal_cu_ms": 2680.0, "xcometxl": 78.2},
        {"segment_ms": 1280, "longyaal_cu_ms": 3450.0, "xcometxl": 79.5},
        {"segment_ms": 1600, "longyaal_cu_ms": 4190.0, "xcometxl": 79.9},
        {"segment_ms": 1920, "longyaal_cu_ms": 4960.0, "xcometxl": 80.8},
    ],
    "with_context": [
        {"segment_ms": 640, "longyaal_cu_ms": 1760.0, "xcometxl": 75.0},
        {"segment_ms": 960, "longyaal_cu_ms": 2680.0, "xcometxl": 79.5},
        {"segment_ms": 1280, "longyaal_cu_ms": 3450.0, "xcometxl": 80.1},
        {"segment_ms": 1600, "longyaal_cu_ms": 4190.0, "xcometxl": 81.3},
        {"segment_ms": 1920, "longyaal_cu_ms": 4960.0, "xcometxl": 81.9},
    ],
}


@dataclass(frozen=True)
class TradeoffPoint:
    system: str
    label: str
    longyaal_cu_ms: float
    xcometxl: float
    chunk_ms: int | None = None
    bleu: float | None = None
    chrf: float | None = None
    source_dir: str = ""
    manifest_dir: str = ""
    alignatt_policy_class: str = ""
    alignatt_guard_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class AlignAttRunDiagnostics:
    diagnostic_source: str = ""
    chunk_count: int = 0
    emitted_chunk_count: int = 0
    update_count: int = 0
    zero_accept_update_count: int = 0
    zero_emit_current_mt_decision_count: int = 0
    stop_reason_counts: dict[str, int] | None = None


@dataclass(frozen=True)
class AlignAttPolicyClassification:
    policy_class: str = ""
    guard_flags: tuple[str, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recovered-index",
        type=Path,
        default=DEFAULT_RECOVERED_INDEX,
        help="Recovered artifact index containing valid scored EN->ZH runs.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-stem",
        default="enzh_quality_latency_tradeoff_current",
    )
    return parser.parse_args()


def finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        return None
    return numeric


def xcomet_to_plot_scale(value: float) -> float:
    return value * 100.0 if value <= 2.0 else value


def load_recovered_points(index_path: Path) -> list[TradeoffPoint]:
    rows = json.loads(index_path.read_text(encoding="utf-8"))
    points: list[TradeoffPoint] = []
    for row in rows:
        if not row.get("valid_for_claims"):
            continue
        if row.get("target_language_code") != "zh":
            continue
        if row.get("mt_backend_name") not in {"milmmt_vllm_alignatt", "gemma_vllm_alignatt"}:
            continue
        if int(row.get("num_inputs") or 0) != 21:
            continue
        xcomet = finite_float(row.get("xcometxl"))
        latency = finite_float(row.get("longyaal_cu_ms"))
        if xcomet is None or latency is None:
            continue
        policy = row.get("translation_acceptance_policy") or "unknown"
        cut = row.get("translation_static_cutoff_units")
        chunk_ms = row.get("chunk_ms")
        rel = str(row.get("relative_dir") or "")
        label = Path(rel).name
        if policy == "cut_last_target_units":
            system = f"MiLMMT fixed cut={cut}"
        else:
            system = "MiLMMT AlignAtt"
        points.append(
            TradeoffPoint(
                system=system,
                label=label,
                longyaal_cu_ms=float(latency),
                xcometxl=xcomet_to_plot_scale(float(xcomet)),
                chunk_ms=None if chunk_ms in (None, "") else int(chunk_ms),
                bleu=finite_float(row.get("bleu")),
                chrf=finite_float(row.get("chrf")),
                source_dir=str(row.get("source_dir") or ""),
                manifest_dir=str(row.get("copied_dir") or ""),
                alignatt_policy_class=str(row.get("alignatt_policy_family") or ""),
                alignatt_guard_flags=tuple(
                    flag
                    for flag in str(row.get("alignatt_guard_flags") or "").split(",")
                    if flag
                ),
            )
        )
    points.sort(key=lambda point: (point.system, point.longyaal_cu_ms, point.xcometxl))
    return points


def baseline_points(name: str) -> list[TradeoffPoint]:
    label = "Public baseline" if name == "baseline" else "Public + context"
    return [
        TradeoffPoint(
            system=label,
            label=f"seg{int(row['segment_ms'])}",
            longyaal_cu_ms=float(row["longyaal_cu_ms"]),
            xcometxl=float(row["xcometxl"]),
            chunk_ms=int(row["segment_ms"]),
            source_dir=BASELINE_SOURCE_URL,
        )
        for row in BASELINE_ENZH_POINTS[name]
    ]


def interpolate_baseline(points: list[TradeoffPoint], latency_ms: float) -> float:
    ordered = sorted(points, key=lambda point: point.longyaal_cu_ms)
    if latency_ms <= ordered[0].longyaal_cu_ms:
        return ordered[0].xcometxl
    if latency_ms >= ordered[-1].longyaal_cu_ms:
        return ordered[-1].xcometxl
    for left, right in zip(ordered, ordered[1:]):
        if left.longyaal_cu_ms <= latency_ms <= right.longyaal_cu_ms:
            span = right.longyaal_cu_ms - left.longyaal_cu_ms
            ratio = (latency_ms - left.longyaal_cu_ms) / span
            return left.xcometxl + ratio * (right.xcometxl - left.xcometxl)
    raise AssertionError("unreachable interpolation state")


def baseline_cu_for_chunk_ms(
    points: list[TradeoffPoint],
    chunk_ms: int | None,
) -> float | None:
    if chunk_ms is None:
        return None
    for point in points:
        if point.chunk_ms == int(chunk_ms):
            return point.longyaal_cu_ms
    return None


def baseline_equivalent_segment_ms(
    points: list[TradeoffPoint],
    latency_ms: float,
) -> float:
    ordered = sorted(points, key=lambda point: point.longyaal_cu_ms)
    if latency_ms <= ordered[0].longyaal_cu_ms:
        return float(ordered[0].chunk_ms or 0)
    if latency_ms >= ordered[-1].longyaal_cu_ms:
        return float(ordered[-1].chunk_ms or 0)
    for left, right in zip(ordered, ordered[1:]):
        if left.longyaal_cu_ms <= latency_ms <= right.longyaal_cu_ms:
            span = right.longyaal_cu_ms - left.longyaal_cu_ms
            ratio = (latency_ms - left.longyaal_cu_ms) / span
            left_chunk = float(left.chunk_ms or 0)
            right_chunk = float(right.chunk_ms or 0)
            return left_chunk + ratio * (right_chunk - left_chunk)
    raise AssertionError("unreachable interpolation state")


def frontier(points: list[TradeoffPoint]) -> list[TradeoffPoint]:
    best = -1.0
    kept: list[TradeoffPoint] = []
    for point in sorted(points, key=lambda item: (item.longyaal_cu_ms, -item.xcometxl)):
        if point.xcometxl > best:
            kept.append(point)
            best = point.xcometxl
    return kept


def best_point_at_or_before_latency(
    points: list[TradeoffPoint],
    latency_ms: float,
) -> TradeoffPoint | None:
    eligible = [
        point
        for point in points
        if point.longyaal_cu_ms <= float(latency_ms)
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda point: (point.xcometxl, -point.longyaal_cu_ms))


def baseline_anchor_dominance_summary(
    *,
    points: list[TradeoffPoint],
    baseline: list[TradeoffPoint],
) -> dict[str, Any]:
    anchors: list[dict[str, Any]] = []
    for anchor in sorted(baseline, key=lambda point: point.longyaal_cu_ms):
        best = best_point_at_or_before_latency(points, anchor.longyaal_cu_ms)
        best_xcomet = None if best is None else best.xcometxl
        delta = None if best_xcomet is None else best_xcomet - anchor.xcometxl
        anchors.append(
            {
                "segment_ms": anchor.chunk_ms,
                "baseline_longyaal_cu_ms": anchor.longyaal_cu_ms,
                "baseline_xcometxl": anchor.xcometxl,
                "best_label_at_or_below_latency": None if best is None else best.label,
                "best_system_at_or_below_latency": None if best is None else best.system,
                "best_longyaal_cu_ms": None if best is None else best.longyaal_cu_ms,
                "best_xcometxl": best_xcomet,
                "delta_vs_baseline_xcometxl": delta,
                "dominates": bool(delta is not None and delta > 0.0),
            }
        )
    return {
        "all_public_baseline_anchors_dominated": all(
            bool(anchor["dominates"]) for anchor in anchors
        ),
        "covered_public_baseline_anchor_count": sum(
            1 for anchor in anchors if anchor["dominates"]
        ),
        "total_public_baseline_anchor_count": len(anchors),
        "anchors": anchors,
    }


def alignatt_same_chunk_permissiveness_summary(
    *,
    points: list[TradeoffPoint],
    baseline: list[TradeoffPoint],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for point in points:
        if point.system != "MiLMMT AlignAtt":
            continue
        same_chunk_cu = baseline_cu_for_chunk_ms(baseline, point.chunk_ms)
        if same_chunk_cu is None:
            continue
        delta = point.longyaal_cu_ms - same_chunk_cu
        checks.append(
            {
                "label": point.label,
                "chunk_ms": point.chunk_ms,
                "longyaal_cu_ms": point.longyaal_cu_ms,
                "public_baseline_same_chunk_cu_ms": same_chunk_cu,
                "delta_cu_vs_public_same_chunk_ms": delta,
                "more_permissive": delta < 0.0,
            }
        )
    return {
        "all_checked_alignatt_points_more_permissive": bool(checks)
        and all(bool(check["more_permissive"]) for check in checks),
        "checked_alignatt_point_count": len(checks),
        "more_permissive_count": sum(
            1 for check in checks if bool(check["more_permissive"])
        ),
        "checks": checks,
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summarize_alignatt_stream_updates(artifact_dir: str | Path) -> AlignAttRunDiagnostics:
    chunk_decisions_path = Path(artifact_dir) / "chunk_decisions.jsonl"
    if chunk_decisions_path.is_file():
        stop_reason_counts: Counter[str] = Counter()
        chunk_count = 0
        emitted_chunk_count = 0
        current_mt_decision_count = 0
        zero_emit_current_mt_decision_count = 0
        zero_accept_current_mt_decision_count = 0
        with chunk_decisions_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    decision = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunk_count += 1
                if decision.get("emitted"):
                    emitted_chunk_count += 1
                if not decision.get("alignatt_metadata_current_chunk"):
                    continue
                current_mt_decision_count += 1
                if not decision.get("emitted"):
                    zero_emit_current_mt_decision_count += 1
                metadata = decision.get("alignatt_decision") or {}
                if not isinstance(metadata, dict):
                    continue
                stop_reason = (
                    metadata.get("stop_reason")
                    or metadata.get("unsafe_reason")
                    or "missing"
                )
                stop_reason_counts[str(stop_reason)] += 1
                try:
                    accepted_token_count = int(metadata.get("accepted_token_count") or 0)
                except (TypeError, ValueError):
                    accepted_token_count = 0
                if accepted_token_count == 0:
                    zero_accept_current_mt_decision_count += 1
        return AlignAttRunDiagnostics(
            diagnostic_source="chunk_decisions",
            chunk_count=chunk_count,
            emitted_chunk_count=emitted_chunk_count,
            update_count=current_mt_decision_count,
            zero_accept_update_count=zero_accept_current_mt_decision_count,
            zero_emit_current_mt_decision_count=zero_emit_current_mt_decision_count,
            stop_reason_counts=dict(stop_reason_counts),
        )

    stream_path = Path(artifact_dir) / "stream_updates.jsonl"
    if not stream_path.is_file():
        return AlignAttRunDiagnostics(stop_reason_counts={})

    stop_reason_counts: Counter[str] = Counter()
    update_count = 0
    zero_accept_update_count = 0
    with stream_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                update = json.loads(line)
            except json.JSONDecodeError:
                continue
            metadata = update.get("alignatt_metadata") or {}
            if not isinstance(metadata, dict):
                continue
            update_count += 1
            stop_reason = (
                metadata.get("stop_reason")
                or metadata.get("unsafe_reason")
                or "missing"
            )
            stop_reason_counts[str(stop_reason)] += 1
            try:
                accepted_token_count = int(metadata.get("accepted_token_count") or 0)
            except (TypeError, ValueError):
                accepted_token_count = 0
            if accepted_token_count == 0:
                zero_accept_update_count += 1
    return AlignAttRunDiagnostics(
        diagnostic_source="stream_updates",
        update_count=update_count,
        zero_accept_update_count=zero_accept_update_count,
        stop_reason_counts=dict(stop_reason_counts),
    )


def format_stop_reason_counts(counts: dict[str, int] | None) -> str:
    if not counts:
        return ""
    ordered = sorted(counts.items(), key=lambda item: (-int(item[1]), item[0]))
    return ",".join(f"{reason}={count}" for reason, count in ordered)


def _manifest_runtime_config(artifact_dir: str | Path) -> dict[str, Any]:
    manifest_path = Path(artifact_dir) / "manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    runtime_config = payload.get("runtime_config")
    return dict(runtime_config) if isinstance(runtime_config, dict) else {}


def _positive_int(value: Any) -> bool:
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def _nonnegative_int(value: Any) -> bool:
    try:
        return int(value) >= 0
    except (TypeError, ValueError):
        return False


def _positive_float(value: Any) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def classify_alignatt_policy(artifact_dir: str | Path) -> AlignAttPolicyClassification:
    """Separate clean AlignAtt evidence from guarded policy diagnostics."""
    artifact_path = Path(artifact_dir)
    runtime_config = _manifest_runtime_config(artifact_dir)
    if not runtime_config:
        return AlignAttPolicyClassification()
    if runtime_config.get("translation_acceptance_policy") not in (None, "alignatt"):
        return AlignAttPolicyClassification(
            policy_class=str(runtime_config.get("translation_acceptance_policy")),
            guard_flags=(),
        )

    guard_flags: list[str] = []
    if "surface_dedup" in artifact_path.as_posix():
        guard_flags.append("surface_dedup_diagnostic")
    acceptance_variant = str(
        runtime_config.get("translation_alignatt_acceptance_variant") or "token"
    )
    source_mass_floor = _positive_float(
        runtime_config.get("translation_alignatt_min_source_mass")
    )
    accepted_prefix_source_mass_floor = _positive_float(
        runtime_config.get("translation_alignatt_min_accepted_accessible_source_mass")
    )
    source_bearing_floor = _positive_float(
        runtime_config.get("translation_alignatt_source_bearing_min_source_mass")
    )
    source_frontier_action = str(
        runtime_config.get("translation_alignatt_source_frontier_action", "stop")
    )
    recoverable_frontier = source_frontier_action == "trim_unrecovered"
    if acceptance_variant == "unit_mass":
        if not source_mass_floor:
            guard_flags.append("acceptance_variant=unit_mass_without_source_mass_floor")
    elif acceptance_variant == "unit_mass_source_bearing":
        if source_mass_floor:
            guard_flags.append("unit_source_bearing_with_source_mass_floor")
        if not source_bearing_floor:
            guard_flags.append(
                "acceptance_variant=unit_source_bearing_without_source_bearing_floor"
            )
        try:
            source_bearing_cap = float(
                runtime_config.get(
                    "translation_alignatt_source_bearing_hard_inaccessible_cap",
                    0.75,
                )
            )
        except (TypeError, ValueError):
            source_bearing_cap = 0.75
        if source_bearing_cap < 1.0:
            guard_flags.append(
                "source_bearing_hard_inaccessible_cap="
                f"{runtime_config.get('translation_alignatt_source_bearing_hard_inaccessible_cap', 0.75)}"
            )
    elif acceptance_variant in {"unit_argmax", "unit_consensus"}:
        if source_mass_floor:
            guard_flags.append("unused_source_mass_floor")
        if accepted_prefix_source_mass_floor:
            guard_flags.append("unused_accepted_prefix_source_mass_floor")
    elif acceptance_variant != "token":
        guard_flags.append(f"acceptance_variant={acceptance_variant}")
    if _positive_int(runtime_config.get("translation_alignatt_min_accessible_source_units")):
        source_context_mode = str(
            runtime_config.get(
                "translation_alignatt_min_accessible_source_units_mode",
                "block",
            )
        )
        guard_flags.append(
            "min_accessible_source_units="
            f"{runtime_config.get('translation_alignatt_min_accessible_source_units')}"
            f":{source_context_mode}"
        )
    if bool(runtime_config.get("translation_alignatt_source_lcp_stability")):
        guard_flags.append("source_lcp_stability")
    if _positive_int(runtime_config.get("translation_alignatt_source_lcp_append_slack_units")):
        guard_flags.append(
            "source_lcp_append_slack_units="
            f"{runtime_config.get('translation_alignatt_source_lcp_append_slack_units')}"
        )
    try:
        max_inaccessible = float(
            runtime_config.get("translation_alignatt_max_inaccessible_source_mass", 1.0)
        )
    except (TypeError, ValueError):
        max_inaccessible = 1.0
    if max_inaccessible < 1.0:
        guard_flags.append(
            "max_inaccessible_source_mass="
            f"{runtime_config.get('translation_alignatt_max_inaccessible_source_mass')}"
        )
    try:
        max_non_source_prompt = float(
            runtime_config.get("translation_alignatt_max_non_source_prompt_mass", 1.0)
        )
    except (TypeError, ValueError):
        max_non_source_prompt = 1.0
    if max_non_source_prompt < 1.0:
        guard_flags.append(
            "max_non_source_prompt_mass="
            f"{runtime_config.get('translation_alignatt_max_non_source_prompt_mass')}"
        )
    try:
        min_margin = float(
            runtime_config.get(
                "translation_alignatt_min_accessible_inaccessible_margin",
                -1.0,
            )
        )
    except (TypeError, ValueError):
        min_margin = -1.0
    if min_margin > -1.0:
        guard_flags.append(
            "accessible_inaccessible_margin="
            f"{runtime_config.get('translation_alignatt_min_accessible_inaccessible_margin')}"
        )
    if _nonnegative_int(runtime_config.get("translation_alignatt_max_source_regression")):
        source_regression_flag = (
            "source_regression="
            f"{runtime_config.get('translation_alignatt_max_source_regression')}"
        )
        source_regression_action = str(
            runtime_config.get("translation_alignatt_source_regression_action", "stop")
        )
        if source_regression_action != "stop":
            source_regression_flag += f"+{source_regression_action}"
        activation_mode = str(
            runtime_config.get(
                "translation_alignatt_source_regression_activation_mode",
                "always",
            )
        )
        if activation_mode != "always":
            source_regression_flag += f":{activation_mode}"
        if _positive_int(
            runtime_config.get(
                "translation_alignatt_source_regression_activation_slack_tokens"
            )
        ):
            source_regression_flag += (
                "+slack"
                f"{runtime_config.get('translation_alignatt_source_regression_activation_slack_tokens')}"
            )
        min_inaccessible = runtime_config.get(
            "translation_alignatt_source_regression_min_inaccessible_mass"
        )
        if _positive_float(min_inaccessible):
            source_regression_flag += f"+future{float(min_inaccessible):g}"
        patience = runtime_config.get(
            "translation_alignatt_source_regression_patience_tokens"
        )
        if _positive_int(patience) and int(patience) > 1:
            source_regression_flag += f"+patience{patience}"
        guard_flags.append(source_regression_flag)
    if bool(runtime_config.get("translation_alignatt_token_argmax_frontier_gate")):
        token_argmax_flag = "token_argmax_frontier_gate"
        patience = runtime_config.get(
            "translation_alignatt_token_argmax_frontier_patience_tokens"
        )
        if _positive_int(patience) and int(patience) > 1:
            token_argmax_flag += f":patience{patience}"
        guard_flags.append(token_argmax_flag)
    if _positive_float(runtime_config.get("translation_alignatt_argmax_mass_threshold")):
        guard_flags.append(
            "argmax_mass_threshold="
            f"{runtime_config.get('translation_alignatt_argmax_mass_threshold')}"
        )
    if _positive_int(runtime_config.get("translation_alignatt_hold_back_target_units")):
        guard_flags.append(
            "hold_back_target_units="
            f"{runtime_config.get('translation_alignatt_hold_back_target_units')}"
        )
    if _positive_int(runtime_config.get("translation_alignatt_min_emit_target_units")):
        guard_flags.append(
            "min_emit_target_units="
            f"{runtime_config.get('translation_alignatt_min_emit_target_units')}"
        )
    if bool(runtime_config.get("translation_alignatt_source_lookback_holdback")):
        guard_flags.append("source_lookback_holdback")
    if bool(runtime_config.get("translation_alignatt_defer_low_source_terminal_punctuation")):
        guard_flags.append("terminal_punctuation_holdback")
    if _positive_int(runtime_config.get("asr_punctuation_min_commit_words")):
        guard_flags.append(
            "asr_punctuation_min_commit_words="
            f"{runtime_config.get('asr_punctuation_min_commit_words')}"
        )

    if guard_flags:
        policy_class = "guarded_alignatt"
    elif acceptance_variant == "unit_mass" and source_mass_floor:
        policy_class = "clean_unit_source_mass_floor"
    elif acceptance_variant == "unit_mass_source_bearing" and source_bearing_floor:
        policy_class = "clean_unit_source_bearing"
    elif acceptance_variant == "unit_argmax":
        policy_class = "clean_unit_argmax_frontier"
    elif acceptance_variant == "unit_consensus":
        policy_class = "clean_unit_consensus_frontier"
    elif _positive_float(
        runtime_config.get("translation_alignatt_frontier_min_inaccessible_mass")
    ) and (source_mass_floor or accepted_prefix_source_mass_floor):
        if recoverable_frontier:
            policy_class = "clean_recoverable_soft_frontier_source_mass_floor"
        else:
            policy_class = "clean_soft_frontier_source_mass_floor"
    elif source_mass_floor or accepted_prefix_source_mass_floor:
        if recoverable_frontier:
            policy_class = "clean_recoverable_argmax_frontier_source_mass_floor"
        else:
            policy_class = "clean_argmax_frontier_source_mass_floor"
    elif _positive_float(
        runtime_config.get("translation_alignatt_frontier_min_inaccessible_mass")
    ):
        if recoverable_frontier:
            policy_class = "clean_recoverable_soft_frontier"
        else:
            policy_class = "pure_soft_frontier"
    elif recoverable_frontier:
        policy_class = "clean_recoverable_argmax_frontier"
    else:
        policy_class = "pure_argmax_frontier"
    return AlignAttPolicyClassification(policy_class=policy_class, guard_flags=tuple(guard_flags))


def write_gap_table(
    path: Path,
    *,
    points: list[TradeoffPoint],
    baseline: list[TradeoffPoint],
    context_baseline: list[TradeoffPoint],
) -> None:
    columns = [
        "label",
        "system",
        "chunk_ms",
        "longyaal_cu_ms",
        "xcometxl",
        "public_baseline_interp_xcometxl",
        "delta_vs_public_baseline_xcometxl",
        "beats_public_baseline",
        "public_baseline_same_chunk_cu_ms",
        "delta_cu_vs_public_same_chunk_ms",
        "alignatt_more_permissive_than_same_chunk_baseline",
        "public_baseline_latency_equivalent_segment_ms",
        "latency_equivalent_segment_minus_chunk_ms",
        "context_baseline_interp_xcometxl",
        "delta_vs_context_xcometxl",
        "beats_context_baseline",
        "bleu",
        "chrf",
        "alignatt_diagnostic_source",
        "alignatt_chunk_count",
        "alignatt_emitted_chunk_count",
        "alignatt_update_count",
        "alignatt_zero_accept_update_count",
        "alignatt_zero_emit_current_mt_decision_count",
        "alignatt_stop_reason_counts",
        "alignatt_policy_class",
        "alignatt_guard_flags",
        "manifest_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for point in sorted(points, key=lambda item: item.longyaal_cu_ms):
            alignatt_diagnostics = (
                summarize_alignatt_stream_updates(point.manifest_dir)
                if point.system == "MiLMMT AlignAtt" and point.manifest_dir
                else AlignAttRunDiagnostics(stop_reason_counts={})
            )
            alignatt_policy = (
                classify_alignatt_policy(point.manifest_dir)
                if point.system == "MiLMMT AlignAtt" and point.manifest_dir
                else
                AlignAttPolicyClassification(
                    policy_class=point.alignatt_policy_class,
                    guard_flags=point.alignatt_guard_flags,
                )
                if point.system == "MiLMMT AlignAtt" and point.alignatt_policy_class
                else AlignAttPolicyClassification()
            )
            public_reference = interpolate_baseline(baseline, point.longyaal_cu_ms)
            context_reference = interpolate_baseline(
                context_baseline, point.longyaal_cu_ms
            )
            public_delta = point.xcometxl - public_reference
            context_delta = point.xcometxl - context_reference
            same_chunk_cu = baseline_cu_for_chunk_ms(baseline, point.chunk_ms)
            cu_delta = (
                None if same_chunk_cu is None else point.longyaal_cu_ms - same_chunk_cu
            )
            equivalent_segment = baseline_equivalent_segment_ms(
                baseline,
                point.longyaal_cu_ms,
            )
            equivalent_delta = (
                None
                if point.chunk_ms is None
                else equivalent_segment - float(point.chunk_ms)
            )
            alignatt_more_permissive = (
                None
                if same_chunk_cu is None or point.system != "MiLMMT AlignAtt"
                else point.longyaal_cu_ms < same_chunk_cu
            )
            writer.writerow(
                {
                    "label": point.label,
                    "system": point.system,
                    "chunk_ms": point.chunk_ms,
                    "longyaal_cu_ms": f"{point.longyaal_cu_ms:.3f}",
                    "xcometxl": f"{point.xcometxl:.3f}",
                    "public_baseline_interp_xcometxl": f"{public_reference:.3f}",
                    "delta_vs_public_baseline_xcometxl": f"{public_delta:.3f}",
                    "beats_public_baseline": str(public_delta > 0.0).lower(),
                    "public_baseline_same_chunk_cu_ms": (
                        "" if same_chunk_cu is None else f"{same_chunk_cu:.3f}"
                    ),
                    "delta_cu_vs_public_same_chunk_ms": (
                        "" if cu_delta is None else f"{cu_delta:.3f}"
                    ),
                    "alignatt_more_permissive_than_same_chunk_baseline": (
                        "" if alignatt_more_permissive is None else str(alignatt_more_permissive).lower()
                    ),
                    "public_baseline_latency_equivalent_segment_ms": (
                        f"{equivalent_segment:.3f}"
                    ),
                    "latency_equivalent_segment_minus_chunk_ms": (
                        "" if equivalent_delta is None else f"{equivalent_delta:.3f}"
                    ),
                    "context_baseline_interp_xcometxl": f"{context_reference:.3f}",
                    "delta_vs_context_xcometxl": f"{context_delta:.3f}",
                    "beats_context_baseline": str(context_delta > 0.0).lower(),
                    "bleu": "" if point.bleu is None else f"{point.bleu:.6f}",
                    "chrf": "" if point.chrf is None else f"{point.chrf:.6f}",
                    "alignatt_diagnostic_source": alignatt_diagnostics.diagnostic_source,
                    "alignatt_chunk_count": (
                        "" if not alignatt_diagnostics.chunk_count else alignatt_diagnostics.chunk_count
                    ),
                    "alignatt_emitted_chunk_count": (
                        ""
                        if not alignatt_diagnostics.chunk_count
                        else alignatt_diagnostics.emitted_chunk_count
                    ),
                    "alignatt_update_count": (
                        "" if not alignatt_diagnostics.update_count else alignatt_diagnostics.update_count
                    ),
                    "alignatt_zero_accept_update_count": (
                        ""
                        if not alignatt_diagnostics.update_count
                        else alignatt_diagnostics.zero_accept_update_count
                    ),
                    "alignatt_zero_emit_current_mt_decision_count": (
                        ""
                        if not alignatt_diagnostics.update_count
                        else alignatt_diagnostics.zero_emit_current_mt_decision_count
                    ),
                    "alignatt_stop_reason_counts": format_stop_reason_counts(
                        alignatt_diagnostics.stop_reason_counts
                    ),
                    "alignatt_policy_class": alignatt_policy.policy_class,
                    "alignatt_guard_flags": ",".join(alignatt_policy.guard_flags),
                    "manifest_dir": point.manifest_dir,
                }
            )


def plot_tradeoff(
    path: Path,
    *,
    our_points: list[TradeoffPoint],
    baseline: list[TradeoffPoint],
    with_context: list[TradeoffPoint],
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    ax.plot(
        [p.longyaal_cu_ms for p in baseline],
        [p.xcometxl for p in baseline],
        "o-",
        color="#2196F3",
        linewidth=2,
        markersize=6,
        label="Public baseline",
    )
    ax.plot(
        [p.longyaal_cu_ms for p in with_context],
        [p.xcometxl for p in with_context],
        "s-",
        color="#FF5722",
        linewidth=2,
        markersize=6,
        label="Public + context",
    )

    alignatt = [p for p in our_points if p.system == "MiLMMT AlignAtt"]
    fixed = [p for p in our_points if p.system.startswith("MiLMMT fixed")]
    ax.scatter(
        [p.longyaal_cu_ms for p in alignatt],
        [p.xcometxl for p in alignatt],
        marker="D",
        s=58,
        color="#111827",
        label="MiLMMT AlignAtt recovered",
        zorder=5,
    )
    ax.plot(
        [p.longyaal_cu_ms for p in frontier(alignatt)],
        [p.xcometxl for p in frontier(alignatt)],
        "-",
        color="#111827",
        linewidth=1.5,
        alpha=0.75,
        zorder=4,
    )
    if fixed:
        ax.scatter(
            [p.longyaal_cu_ms for p in fixed],
            [p.xcometxl for p in fixed],
            marker="x",
            s=70,
            color="#6B7280",
            label="MiLMMT fixed cutoff",
            zorder=5,
        )

    for point in alignatt:
        if point.chunk_ms in {500, 640, 720, 960, 1280}:
            ax.annotate(
                f"{point.chunk_ms}ms",
                (point.longyaal_cu_ms, point.xcometxl),
                textcoords="offset points",
                xytext=(6, 7),
                fontsize=8,
                color="#111827",
            )

    ax.axvspan(0, 2000, color="#ECFDF5", alpha=0.55, label="IWSLT low regime")
    ax.axvspan(2000, 4000, color="#EFF6FF", alpha=0.35, label="IWSLT high regime")
    dominance = baseline_anchor_dominance_summary(points=our_points, baseline=baseline)
    ax.set_title(
        "EN→ZH quality-latency tradeoff "
        f"(public anchors dominated: "
        f"{dominance['covered_public_baseline_anchor_count']}/"
        f"{dominance['total_public_baseline_anchor_count']})"
    )
    ax.set_xlabel("LongYAAL CU (ms)")
    ax.set_ylabel("XCOMET-XL × 100")
    ax.set_xlim(1200, 5200)
    ax.set_ylim(68.5, 83.2)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    our_points = load_recovered_points(args.recovered_index)
    baseline = baseline_points("baseline")
    with_context = baseline_points("with_context")

    plot_path = args.output_dir / f"{args.output_stem}.png"
    data_path = args.output_dir / f"{args.output_stem}.json"
    gap_path = args.output_dir / f"{args.output_stem}_gap.tsv"
    plot_tradeoff(
        plot_path,
        our_points=our_points,
        baseline=baseline,
        with_context=with_context,
    )
    write_gap_table(
        gap_path,
        points=our_points,
        baseline=baseline,
        context_baseline=with_context,
    )
    write_json(
        data_path,
        {
            "baseline_source_url": BASELINE_SOURCE_URL,
            "baseline_source_commit": BASELINE_SOURCE_COMMIT,
            "baseline_points": BASELINE_ENZH_POINTS,
            "recovered_index": str(args.recovered_index),
            "public_baseline_anchor_dominance": baseline_anchor_dominance_summary(
                points=our_points,
                baseline=baseline,
            ),
            "alignatt_same_chunk_permissiveness": (
                alignatt_same_chunk_permissiveness_summary(
                    points=our_points,
                    baseline=baseline,
                )
            ),
            "our_points": [point.__dict__ for point in our_points],
        },
    )
    print(plot_path)
    print(gap_path)
    print(data_path)


if __name__ == "__main__":
    main()
