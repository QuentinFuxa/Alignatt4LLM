#!/usr/bin/env python3
"""Verify whether current EN->ZH evidence is ready to claim baseline dominance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMOTION_SUMMARY = (
    REPO_ROOT / "outputs" / "plots" / "enzh_candidate_promotions_20260607.json"
)
DEFAULT_TRADEOFF_SUMMARY = (
    REPO_ROOT
    / "outputs"
    / "plots"
    / "enzh_quality_latency_tradeoff_diagnostics_20260607.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "outputs" / "plots" / "enzh_baseline_dominance_verdict_20260607.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--promotion-summary",
        type=Path,
        default=DEFAULT_PROMOTION_SUMMARY,
    )
    parser.add_argument(
        "--tradeoff-summary",
        type=Path,
        default=DEFAULT_TRADEOFF_SUMMARY,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-guarded-full21",
        action="store_true",
        help=(
            "Do not require clean_full21 coverage. This is only for explicit "
            "guarded-policy reporting, not the default AlignAtt claim."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _coverage_failures(
    coverage: dict[str, Any],
    *,
    requirement: str,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    anchors = coverage.get("anchors") or []
    for anchor in anchors:
        if bool(anchor.get("dominates_anchor") or anchor.get("dominates")):
            continue
        failures.append(
            {
                "requirement": requirement,
                "segment_ms": anchor.get("segment_ms"),
                "baseline_longyaal_cu_ms": anchor.get("baseline_longyaal_cu_ms"),
                "baseline_xcometxl": anchor.get("baseline_xcometxl"),
                "best_relative_dir": anchor.get("best_relative_dir")
                or anchor.get("best_label_at_or_below_latency")
                or "",
                "best_longyaal_cu_ms": anchor.get("best_longyaal_cu_ms"),
                "best_xcometxl": anchor.get("best_xcometxl"),
                "delta_xcometxl_vs_anchor": anchor.get("delta_xcometxl_vs_anchor")
                if "delta_xcometxl_vs_anchor" in anchor
                else anchor.get("delta_vs_baseline_xcometxl"),
                "blockers": anchor.get("best_promotion_blockers") or "",
            }
        )
    return failures


def _permissiveness_failures(summary: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for check in summary.get("checks") or []:
        if bool(check.get("more_permissive")):
            continue
        failures.append(
            {
                "requirement": "same_chunk_permissiveness",
                "label": check.get("label"),
                "chunk_ms": check.get("chunk_ms"),
                "longyaal_cu_ms": check.get("longyaal_cu_ms"),
                "public_baseline_same_chunk_cu_ms": check.get(
                    "public_baseline_same_chunk_cu_ms"
                ),
                "delta_cu_vs_public_same_chunk_ms": check.get(
                    "delta_cu_vs_public_same_chunk_ms"
                ),
            }
        )
    return failures


def dominance_verdict(
    *,
    promotion_summary: dict[str, Any],
    tradeoff_summary: dict[str, Any],
    require_clean_full21: bool = True,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []

    full21 = promotion_summary.get("full21_public_baseline_anchor_coverage") or {}
    failures.extend(
        _coverage_failures(
            full21,
            requirement="full21_public_baseline_anchor_coverage",
        )
    )

    clean_full21 = (
        promotion_summary.get("clean_full21_public_baseline_anchor_coverage") or {}
    )
    if require_clean_full21:
        failures.extend(
            _coverage_failures(
                clean_full21,
                requirement="clean_full21_public_baseline_anchor_coverage",
            )
        )

    tradeoff_dominance = tradeoff_summary.get("public_baseline_anchor_dominance") or {}
    failures.extend(
        _coverage_failures(
            tradeoff_dominance,
            requirement="tradeoff_full21_public_baseline_anchor_dominance",
        )
    )

    permissiveness = tradeoff_summary.get("alignatt_same_chunk_permissiveness") or {}
    if not bool(permissiveness.get("all_checked_alignatt_points_more_permissive")):
        failures.extend(_permissiveness_failures(permissiveness))

    ready = not failures
    return {
        "ready_for_enzh_baseline_dominance_claim": ready,
        "require_clean_full21": bool(require_clean_full21),
        "baseline_source_url": promotion_summary.get("baseline_source_url")
        or tradeoff_summary.get("baseline_source_url"),
        "baseline_source_commit": promotion_summary.get("baseline_source_commit")
        or tradeoff_summary.get("baseline_source_commit"),
        "full21_covered_anchors": full21.get("covered_public_baseline_anchor_count", 0),
        "full21_total_anchors": full21.get("total_public_baseline_anchor_count", 0),
        "clean_full21_covered_anchors": clean_full21.get(
            "covered_public_baseline_anchor_count",
            0,
        ),
        "clean_full21_total_anchors": clean_full21.get(
            "total_public_baseline_anchor_count",
            0,
        ),
        "same_chunk_permissive_count": permissiveness.get("more_permissive_count", 0),
        "same_chunk_checked_count": permissiveness.get(
            "checked_alignatt_point_count",
            0,
        ),
        "failure_count": len(failures),
        "failures": failures,
    }


def main() -> None:
    args = parse_args()
    verdict = dominance_verdict(
        promotion_summary=load_json(args.promotion_summary),
        tradeoff_summary=load_json(args.tradeoff_summary),
        require_clean_full21=not args.allow_guarded_full21,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    if not verdict["ready_for_enzh_baseline_dominance_claim"]:
        print(
            "EN->ZH dominance claim is not ready: "
            f"{verdict['failure_count']} failing checks.",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
