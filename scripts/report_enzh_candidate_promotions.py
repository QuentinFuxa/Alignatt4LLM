#!/usr/bin/env python3
"""Rank EN->ZH MiLMMT candidates for promotion without mixing claim scopes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_enzh_quality_latency_tradeoff import (
    BASELINE_SOURCE_COMMIT,
    BASELINE_SOURCE_URL,
    baseline_cu_for_chunk_ms,
    baseline_points,
    finite_float,
    interpolate_baseline,
    xcomet_to_plot_scale,
)


DEFAULT_INDEX = (
    REPO_ROOT
    / "outputs"
    / "diagnostics_jarvislab_20260606"
    / "diagnostics_artifact_index.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "plots"


REPORT_COLUMNS = (
    "rank",
    "scope",
    "relative_dir",
    "num_inputs",
    "chunk_ms",
    "longyaal_cu_ms",
    "xcometxl",
    "bleu",
    "chrf",
    "same_chunk_public_cu_ms",
    "delta_cu_vs_same_chunk_public_ms",
    "more_permissive_than_same_chunk_public",
    "public_interp_xcometxl",
    "delta_xcometxl_vs_public_interp",
    "same_chunk_public_xcometxl",
    "delta_xcometxl_vs_same_chunk_public",
    "promotable_latency_probe",
    "clean_alignatt_latency_probe",
    "same_chunk_quality_beat",
    "clean_alignatt_same_chunk_claim_candidate",
    "promotion_blockers",
    "clean_alignatt_subfamily",
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
    "translation_alignatt_source_bearing_min_source_mass",
    "translation_alignatt_source_bearing_hard_inaccessible_cap",
    "translation_alignatt_top_k_heads",
    "translation_alignatt_filter_width",
    "translation_alignatt_online_normalization",
)

ANCHOR_COVERAGE_COLUMNS = (
    "coverage_scope",
    "segment_ms",
    "baseline_longyaal_cu_ms",
    "baseline_xcometxl",
    "best_relative_dir",
    "best_scope",
    "best_chunk_ms",
    "best_longyaal_cu_ms",
    "best_xcometxl",
    "delta_xcometxl_vs_anchor",
    "dominates_anchor",
    "best_clean_alignatt_subfamily",
    "best_promotion_blockers",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default="enzh_candidate_promotions")
    return parser.parse_args()


def scope_for_num_inputs(num_inputs: int) -> str:
    if num_inputs >= 21:
        return "full21_claim"
    if num_inputs >= 3:
        return "mini3_probe"
    return "single_audio_probe"


def baseline_xcomet_for_chunk_ms(chunk_ms: int | None) -> float | None:
    if chunk_ms is None:
        return None
    for point in baseline_points("baseline"):
        if point.chunk_ms == int(chunk_ms):
            return point.xcometxl
    return None


def _format_float(value: float | None, *, digits: int = 6) -> str:
    return "" if value is None else f"{float(value):.{digits}f}"


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def is_clean_alignatt_policy(policy_family: str, guard_flags: str) -> bool:
    return policy_family in {
        "pure_soft_frontier",
        "pure_argmax_frontier",
        "clean_soft_frontier_source_mass_floor",
        "clean_argmax_frontier_source_mass_floor",
        "clean_unit_source_mass_floor",
        "clean_unit_source_bearing",
        "clean_unit_argmax_frontier",
        "clean_unit_consensus_frontier",
        "clean_recoverable_soft_frontier",
        "clean_recoverable_argmax_frontier",
        "clean_recoverable_soft_frontier_source_mass_floor",
        "clean_recoverable_argmax_frontier_source_mass_floor",
    } and not guard_flags


def clean_alignatt_subfamily(policy_family: str, guard_flags: str) -> str:
    if guard_flags:
        return "guarded_or_cutoff"
    if policy_family in {"pure_soft_frontier", "pure_argmax_frontier"}:
        return "pure_frontier"
    if policy_family in {
        "clean_soft_frontier_source_mass_floor",
        "clean_argmax_frontier_source_mass_floor",
        "clean_recoverable_soft_frontier_source_mass_floor",
        "clean_recoverable_argmax_frontier_source_mass_floor",
        "clean_unit_source_mass_floor",
    }:
        return "source_mass_floor"
    if policy_family in {
        "clean_recoverable_soft_frontier",
        "clean_recoverable_argmax_frontier",
    }:
        return "recoverable_frontier"
    if policy_family == "clean_unit_source_bearing":
        return "source_bearing_unit"
    if policy_family in {"clean_unit_argmax_frontier", "clean_unit_consensus_frontier"}:
        return "unit_frontier"
    return "guarded_or_cutoff"


def promotion_blockers(
    *,
    xcomet: float | None,
    same_chunk_cu: float | None,
    same_chunk_xcomet_delta: float | None,
    more_permissive: bool,
    clean_policy: bool,
) -> tuple[str, ...]:
    blockers: list[str] = []
    if xcomet is None:
        blockers.append("missing_xcomet")
    if same_chunk_cu is None:
        blockers.append("missing_same_chunk_public_latency")
    elif not more_permissive:
        blockers.append("not_more_permissive_than_same_chunk_public")
    if not clean_policy:
        blockers.append("guarded_or_cutoff_policy")
    if same_chunk_xcomet_delta is None:
        blockers.append("missing_same_chunk_public_quality")
    elif same_chunk_xcomet_delta <= 0.0:
        blockers.append("not_higher_quality_than_same_chunk_public")
    return tuple(blockers)


def candidate_rows(index_path: Path) -> list[dict[str, Any]]:
    rows = json.loads(index_path.read_text(encoding="utf-8"))
    baseline = baseline_points("baseline")
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("valid_for_claims"):
            continue
        if row.get("target_language_code") != "zh":
            continue
        if row.get("mt_backend_name") not in {"milmmt_vllm_alignatt", "gemma_vllm_alignatt"}:
            continue
        latency = finite_float(row.get("longyaal_cu_ms"))
        if latency is None:
            continue
        chunk_ms = _int_or_none(row.get("chunk_ms"))
        num_inputs = _int_or_none(row.get("num_inputs")) or 0
        xcomet_raw = finite_float(row.get("xcometxl"))
        xcomet = None if xcomet_raw is None else xcomet_to_plot_scale(xcomet_raw)
        same_chunk_cu = baseline_cu_for_chunk_ms(baseline, chunk_ms)
        same_chunk_xcomet = baseline_xcomet_for_chunk_ms(chunk_ms)
        cu_delta = None if same_chunk_cu is None else float(latency) - same_chunk_cu
        public_interp = interpolate_baseline(baseline, float(latency))
        interp_delta = None if xcomet is None else xcomet - public_interp
        same_chunk_xcomet_delta = (
            None
            if xcomet is None or same_chunk_xcomet is None
            else xcomet - same_chunk_xcomet
        )
        more_permissive = cu_delta is not None and cu_delta < 0.0
        scope = scope_for_num_inputs(num_inputs)
        policy_family = row.get("alignatt_policy_family") or ""
        guard_flags = row.get("alignatt_guard_flags") or ""
        clean_policy = is_clean_alignatt_policy(policy_family, guard_flags)
        clean_subfamily = clean_alignatt_subfamily(policy_family, guard_flags)
        same_chunk_quality_beat = (
            same_chunk_xcomet_delta is not None and same_chunk_xcomet_delta > 0.0
        )
        promotable = bool(
            scope != "full21_claim"
            and xcomet is not None
            and more_permissive
        )
        clean_latency_probe = bool(
            scope != "full21_claim"
            and xcomet is not None
            and more_permissive
            and clean_policy
        )
        blockers = promotion_blockers(
            xcomet=xcomet,
            same_chunk_cu=same_chunk_cu,
            same_chunk_xcomet_delta=same_chunk_xcomet_delta,
            more_permissive=more_permissive,
            clean_policy=clean_policy,
        )
        candidates.append(
            {
                "scope": scope,
                "relative_dir": row.get("relative_dir") or "",
                "num_inputs": num_inputs,
                "chunk_ms": chunk_ms,
                "longyaal_cu_ms": float(latency),
                "xcometxl": xcomet,
                "bleu": finite_float(row.get("bleu")),
                "chrf": finite_float(row.get("chrf")),
                "same_chunk_public_cu_ms": same_chunk_cu,
                "delta_cu_vs_same_chunk_public_ms": cu_delta,
                "more_permissive_than_same_chunk_public": more_permissive,
                "public_interp_xcometxl": public_interp,
                "delta_xcometxl_vs_public_interp": interp_delta,
                "same_chunk_public_xcometxl": same_chunk_xcomet,
                "delta_xcometxl_vs_same_chunk_public": same_chunk_xcomet_delta,
                "promotable_latency_probe": promotable,
                "clean_alignatt_latency_probe": clean_latency_probe,
                "same_chunk_quality_beat": same_chunk_quality_beat,
                "clean_alignatt_same_chunk_claim_candidate": not blockers,
                "promotion_blockers": ",".join(blockers),
                "clean_alignatt_subfamily": clean_subfamily,
                "alignatt_policy_family": policy_family,
                "alignatt_guard_flags": guard_flags,
                "translation_alignatt_acceptance_variant": row.get(
                    "translation_alignatt_acceptance_variant"
                )
                or "",
                "translation_alignatt_min_source_mass": row.get(
                    "translation_alignatt_min_source_mass"
                ),
                "translation_alignatt_inaccessible_ms": row.get(
                    "translation_alignatt_inaccessible_ms"
                ),
                "translation_alignatt_frontier_min_inaccessible_mass": row.get(
                    "translation_alignatt_frontier_min_inaccessible_mass"
                ),
                "translation_alignatt_source_frontier_action": row.get(
                    "translation_alignatt_source_frontier_action"
                ),
                "translation_alignatt_source_regression_action": row.get(
                    "translation_alignatt_source_regression_action"
                )
                or "",
                "translation_alignatt_max_non_source_prompt_mass": row.get(
                    "translation_alignatt_max_non_source_prompt_mass"
                ),
                "translation_alignatt_min_accepted_accessible_source_mass": row.get(
                    "translation_alignatt_min_accepted_accessible_source_mass"
                ),
                "translation_alignatt_accepted_accessible_source_mass_recent_units": row.get(
                    "translation_alignatt_accepted_accessible_source_mass_recent_units"
                ),
                "translation_alignatt_unit_consensus_min_head_ratio": row.get(
                    "translation_alignatt_unit_consensus_min_head_ratio"
                ),
                "translation_alignatt_source_bearing_min_source_mass": row.get(
                    "translation_alignatt_source_bearing_min_source_mass"
                ),
                "translation_alignatt_source_bearing_hard_inaccessible_cap": row.get(
                    "translation_alignatt_source_bearing_hard_inaccessible_cap"
                ),
                "translation_alignatt_top_k_heads": row.get(
                    "translation_alignatt_top_k_heads"
                ),
                "translation_alignatt_filter_width": row.get(
                    "translation_alignatt_filter_width"
                ),
                "translation_alignatt_online_normalization": row.get(
                    "translation_alignatt_online_normalization"
                )
                or "",
            }
        )
    candidates.sort(key=candidate_sort_key)
    for rank, row in enumerate(candidates, start=1):
        row["rank"] = rank
    return candidates


def candidate_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    scope_priority = {
        "mini3_probe": 0,
        "single_audio_probe": 1,
        "full21_claim": 2,
    }.get(str(row.get("scope")), 9)
    clean_claim_priority = (
        0 if row.get("clean_alignatt_same_chunk_claim_candidate") else 1
    )
    promotable_priority = 0 if row.get("promotable_latency_probe") else 1
    xcomet_gap = row.get("delta_xcometxl_vs_same_chunk_public")
    interp_gap = row.get("delta_xcometxl_vs_public_interp")
    chrf = row.get("chrf")
    latency_margin = row.get("delta_cu_vs_same_chunk_public_ms")
    return (
        clean_claim_priority,
        promotable_priority,
        scope_priority,
        -(float(xcomet_gap) if xcomet_gap is not None else -999.0),
        -(float(interp_gap) if interp_gap is not None else -999.0),
        -(float(chrf) if chrf is not None else -999.0),
        float(latency_margin) if latency_margin is not None else 999999.0,
        str(row.get("relative_dir") or ""),
    )


def summarize_candidates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    promotable = [row for row in rows if row.get("promotable_latency_probe")]
    clean_latency = [row for row in rows if row.get("clean_alignatt_latency_probe")]
    pure_latency = [
        row
        for row in clean_latency
        if row.get("clean_alignatt_subfamily") == "pure_frontier"
    ]
    source_mass_latency = [
        row
        for row in clean_latency
        if row.get("clean_alignatt_subfamily") == "source_mass_floor"
    ]
    unit_frontier_latency = [
        row
        for row in clean_latency
        if row.get("clean_alignatt_subfamily") == "unit_frontier"
    ]
    source_bearing_unit_latency = [
        row
        for row in clean_latency
        if row.get("clean_alignatt_subfamily") == "source_bearing_unit"
    ]
    clean_rows = [
        row
        for row in rows
        if row.get("clean_alignatt_subfamily")
        in {
            "pure_frontier",
            "source_mass_floor",
            "unit_frontier",
            "source_bearing_unit",
        }
    ]
    clean_full21 = [
        row for row in clean_rows if row.get("scope") == "full21_claim"
    ]
    clean_claim = [
        row for row in rows if row.get("clean_alignatt_same_chunk_claim_candidate")
    ]
    full21 = [row for row in rows if row.get("scope") == "full21_claim"]
    return {
        "baseline_source_url": BASELINE_SOURCE_URL,
        "baseline_source_commit": BASELINE_SOURCE_COMMIT,
        "candidate_count": len(rows),
        "promotable_latency_probe_count": len(promotable),
        "clean_alignatt_latency_probe_count": len(clean_latency),
        "pure_frontier_latency_probe_count": len(pure_latency),
        "source_mass_floor_latency_probe_count": len(source_mass_latency),
        "unit_frontier_latency_probe_count": len(unit_frontier_latency),
        "source_bearing_unit_latency_probe_count": len(source_bearing_unit_latency),
        "clean_alignatt_same_chunk_claim_candidate_count": len(clean_claim),
        "best_promotable_latency_probes": promotable[:8],
        "best_clean_alignatt_latency_probes": clean_latency[:8],
        "best_source_mass_floor_latency_probes": source_mass_latency[:8],
        "best_unit_frontier_latency_probes": unit_frontier_latency[:8],
        "best_source_bearing_unit_latency_probes": source_bearing_unit_latency[:8],
        "best_clean_alignatt_same_chunk_claim_candidates": clean_claim[:8],
        "best_full21_claim_points": full21[:4],
        "public_baseline_anchor_coverage": anchor_coverage_summary(
            rows,
            coverage_scope="all_valid_candidates",
        ),
        "full21_public_baseline_anchor_coverage": anchor_coverage_summary(
            full21,
            coverage_scope="full21_claims",
        ),
        "promotable_public_baseline_anchor_coverage": anchor_coverage_summary(
            promotable,
            coverage_scope="promotable_latency_probes",
        ),
        "clean_public_baseline_anchor_coverage": anchor_coverage_summary(
            clean_rows,
            coverage_scope="clean_alignatt_candidates",
        ),
        "clean_full21_public_baseline_anchor_coverage": anchor_coverage_summary(
            clean_full21,
            coverage_scope="clean_full21_claims",
        ),
        "scope_counts": {
            scope: sum(1 for row in rows if row.get("scope") == scope)
            for scope in ("full21_claim", "mini3_probe", "single_audio_probe")
        },
    }


def _best_row_at_or_below_latency(
    rows: list[dict[str, Any]],
    latency_ms: float,
) -> dict[str, Any] | None:
    eligible = [
        row
        for row in rows
        if finite_float(row.get("longyaal_cu_ms")) is not None
        and float(row["longyaal_cu_ms"]) <= float(latency_ms)
        and finite_float(row.get("xcometxl")) is not None
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (
            float(row["xcometxl"]),
            -float(row["longyaal_cu_ms"]),
            str(row.get("relative_dir") or ""),
        ),
    )


def anchor_coverage_rows(
    rows: list[dict[str, Any]],
    *,
    coverage_scope: str,
) -> list[dict[str, Any]]:
    coverage: list[dict[str, Any]] = []
    for anchor in baseline_points("baseline"):
        best = _best_row_at_or_below_latency(rows, anchor.longyaal_cu_ms)
        best_xcomet = None if best is None else finite_float(best.get("xcometxl"))
        delta = None if best_xcomet is None else float(best_xcomet) - anchor.xcometxl
        coverage.append(
            {
                "coverage_scope": coverage_scope,
                "segment_ms": anchor.chunk_ms,
                "baseline_longyaal_cu_ms": anchor.longyaal_cu_ms,
                "baseline_xcometxl": anchor.xcometxl,
                "best_relative_dir": "" if best is None else best["relative_dir"],
                "best_scope": "" if best is None else best["scope"],
                "best_chunk_ms": "" if best is None else best["chunk_ms"],
                "best_longyaal_cu_ms": None
                if best is None
                else best["longyaal_cu_ms"],
                "best_xcometxl": best_xcomet,
                "delta_xcometxl_vs_anchor": delta,
                "dominates_anchor": bool(delta is not None and delta > 0.0),
                "best_clean_alignatt_subfamily": ""
                if best is None
                else best["clean_alignatt_subfamily"],
                "best_promotion_blockers": ""
                if best is None
                else best["promotion_blockers"],
            }
        )
    return coverage


def anchor_coverage_summary(
    rows: list[dict[str, Any]],
    *,
    coverage_scope: str,
) -> dict[str, Any]:
    coverage = anchor_coverage_rows(rows, coverage_scope=coverage_scope)
    return {
        "coverage_scope": coverage_scope,
        "covered_public_baseline_anchor_count": sum(
            1 for row in coverage if row["dominates_anchor"]
        ),
        "total_public_baseline_anchor_count": len(coverage),
        "all_public_baseline_anchors_dominated": all(
            bool(row["dominates_anchor"]) for row in coverage
        ),
        "anchors": coverage,
    }


def write_report(rows: list[dict[str, Any]], *, output_dir: Path, output_stem: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = output_dir / f"{output_stem}.tsv"
    json_path = output_dir / f"{output_stem}.json"
    anchor_path = output_dir / f"{output_stem}_anchor_coverage.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "rank": row["rank"],
                    "scope": row["scope"],
                    "relative_dir": row["relative_dir"],
                    "num_inputs": row["num_inputs"],
                    "chunk_ms": "" if row["chunk_ms"] is None else row["chunk_ms"],
                    "longyaal_cu_ms": _format_float(row["longyaal_cu_ms"], digits=3),
                    "xcometxl": _format_float(row["xcometxl"], digits=3),
                    "bleu": _format_float(row["bleu"], digits=6),
                    "chrf": _format_float(row["chrf"], digits=6),
                    "same_chunk_public_cu_ms": _format_float(
                        row["same_chunk_public_cu_ms"],
                        digits=3,
                    ),
                    "delta_cu_vs_same_chunk_public_ms": _format_float(
                        row["delta_cu_vs_same_chunk_public_ms"],
                        digits=3,
                    ),
                    "more_permissive_than_same_chunk_public": str(
                        row["more_permissive_than_same_chunk_public"]
                    ).lower(),
                    "public_interp_xcometxl": _format_float(
                        row["public_interp_xcometxl"],
                        digits=3,
                    ),
                    "delta_xcometxl_vs_public_interp": _format_float(
                        row["delta_xcometxl_vs_public_interp"],
                        digits=3,
                    ),
                    "same_chunk_public_xcometxl": _format_float(
                        row["same_chunk_public_xcometxl"],
                        digits=3,
                    ),
                    "delta_xcometxl_vs_same_chunk_public": _format_float(
                        row["delta_xcometxl_vs_same_chunk_public"],
                        digits=3,
                    ),
                    "promotable_latency_probe": str(
                        row["promotable_latency_probe"]
                    ).lower(),
                    "clean_alignatt_latency_probe": str(
                        row["clean_alignatt_latency_probe"]
                    ).lower(),
                    "same_chunk_quality_beat": str(
                        row["same_chunk_quality_beat"]
                    ).lower(),
                    "clean_alignatt_same_chunk_claim_candidate": str(
                        row["clean_alignatt_same_chunk_claim_candidate"]
                    ).lower(),
                    "promotion_blockers": row["promotion_blockers"],
                    "clean_alignatt_subfamily": row["clean_alignatt_subfamily"],
                    "alignatt_policy_family": row["alignatt_policy_family"],
                    "alignatt_guard_flags": row["alignatt_guard_flags"],
                    "translation_alignatt_acceptance_variant": row[
                        "translation_alignatt_acceptance_variant"
                    ],
                    "translation_alignatt_min_source_mass": "" if row[
                        "translation_alignatt_min_source_mass"
                    ] is None else row["translation_alignatt_min_source_mass"],
                    "translation_alignatt_inaccessible_ms": "" if row[
                        "translation_alignatt_inaccessible_ms"
                    ] is None else row["translation_alignatt_inaccessible_ms"],
                    "translation_alignatt_frontier_min_inaccessible_mass": "" if row[
                        "translation_alignatt_frontier_min_inaccessible_mass"
                    ] is None else row[
                        "translation_alignatt_frontier_min_inaccessible_mass"
                    ],
                    "translation_alignatt_source_frontier_action": row[
                        "translation_alignatt_source_frontier_action"
                    ],
                    "translation_alignatt_source_regression_action": row[
                        "translation_alignatt_source_regression_action"
                    ],
                    "translation_alignatt_max_non_source_prompt_mass": "" if row[
                        "translation_alignatt_max_non_source_prompt_mass"
                    ] is None else row[
                        "translation_alignatt_max_non_source_prompt_mass"
                    ],
                    "translation_alignatt_min_accepted_accessible_source_mass": "" if row[
                        "translation_alignatt_min_accepted_accessible_source_mass"
                    ] is None else row[
                        "translation_alignatt_min_accepted_accessible_source_mass"
                    ],
                    "translation_alignatt_accepted_accessible_source_mass_recent_units": "" if row[
                        "translation_alignatt_accepted_accessible_source_mass_recent_units"
                    ] is None else row[
                        "translation_alignatt_accepted_accessible_source_mass_recent_units"
                    ],
                    "translation_alignatt_unit_consensus_min_head_ratio": "" if row[
                        "translation_alignatt_unit_consensus_min_head_ratio"
                    ] is None else row[
                        "translation_alignatt_unit_consensus_min_head_ratio"
                    ],
                    "translation_alignatt_source_bearing_min_source_mass": "" if row[
                        "translation_alignatt_source_bearing_min_source_mass"
                    ] is None else row[
                        "translation_alignatt_source_bearing_min_source_mass"
                    ],
                    "translation_alignatt_source_bearing_hard_inaccessible_cap": "" if row[
                        "translation_alignatt_source_bearing_hard_inaccessible_cap"
                    ] is None else row[
                        "translation_alignatt_source_bearing_hard_inaccessible_cap"
                    ],
                    "translation_alignatt_top_k_heads": "" if row[
                        "translation_alignatt_top_k_heads"
                    ] is None else row["translation_alignatt_top_k_heads"],
                    "translation_alignatt_filter_width": "" if row[
                        "translation_alignatt_filter_width"
                    ] is None else row["translation_alignatt_filter_width"],
                    "translation_alignatt_online_normalization": row[
                        "translation_alignatt_online_normalization"
                    ],
                }
            )
    json_path.write_text(
        json.dumps(summarize_candidates(rows), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with anchor_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=ANCHOR_COVERAGE_COLUMNS,
            delimiter="\t",
        )
        writer.writeheader()
        coverage_rows = [
            *anchor_coverage_rows(rows, coverage_scope="all_valid_candidates"),
            *anchor_coverage_rows(
                [
                    row
                    for row in rows
                    if row.get("clean_alignatt_subfamily")
                    in {"pure_frontier", "source_mass_floor", "unit_frontier"}
                ],
                coverage_scope="clean_alignatt_candidates",
            ),
            *anchor_coverage_rows(
                [row for row in rows if row.get("promotable_latency_probe")],
                coverage_scope="promotable_latency_probes",
            ),
            *anchor_coverage_rows(
                [row for row in rows if row.get("scope") == "full21_claim"],
                coverage_scope="full21_claims",
            ),
            *anchor_coverage_rows(
                [
                    row
                    for row in rows
                    if row.get("scope") == "full21_claim"
                    and row.get("clean_alignatt_subfamily")
                    in {"pure_frontier", "source_mass_floor", "unit_frontier"}
                ],
                coverage_scope="clean_full21_claims",
            ),
        ]
        for row in coverage_rows:
            writer.writerow(
                {
                    "coverage_scope": row["coverage_scope"],
                    "segment_ms": row["segment_ms"],
                    "baseline_longyaal_cu_ms": _format_float(
                        row["baseline_longyaal_cu_ms"],
                        digits=3,
                    ),
                    "baseline_xcometxl": _format_float(
                        row["baseline_xcometxl"],
                        digits=3,
                    ),
                    "best_relative_dir": row["best_relative_dir"],
                    "best_scope": row["best_scope"],
                    "best_chunk_ms": row["best_chunk_ms"],
                    "best_longyaal_cu_ms": _format_float(
                        row["best_longyaal_cu_ms"],
                        digits=3,
                    ),
                    "best_xcometxl": _format_float(row["best_xcometxl"], digits=3),
                    "delta_xcometxl_vs_anchor": _format_float(
                        row["delta_xcometxl_vs_anchor"],
                        digits=3,
                    ),
                    "dominates_anchor": str(row["dominates_anchor"]).lower(),
                    "best_clean_alignatt_subfamily": row[
                        "best_clean_alignatt_subfamily"
                    ],
                    "best_promotion_blockers": row["best_promotion_blockers"],
                }
            )
    return {"tsv": tsv_path, "json": json_path, "anchor_coverage_tsv": anchor_path}


def main() -> None:
    args = parse_args()
    rows = candidate_rows(args.index)
    paths = write_report(rows, output_dir=args.output_dir, output_stem=args.output_stem)
    print(paths["tsv"])
    print(paths["anchor_coverage_tsv"])
    print(paths["json"])


if __name__ == "__main__":
    main()
