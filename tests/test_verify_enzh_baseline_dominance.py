from __future__ import annotations

from tools.reports.verify_enzh_baseline_dominance import dominance_verdict


def _coverage(*, dominated: bool) -> dict:
    return {
        "covered_public_baseline_anchor_count": 1 if dominated else 0,
        "total_public_baseline_anchor_count": 1,
        "all_public_baseline_anchors_dominated": dominated,
        "anchors": [
            {
                "segment_ms": 640,
                "baseline_longyaal_cu_ms": 1760.0,
                "baseline_xcometxl": 74.9,
                "best_relative_dir": "run",
                "best_longyaal_cu_ms": 1500.0,
                "best_xcometxl": 75.5 if dominated else 72.0,
                "delta_xcometxl_vs_anchor": 0.6 if dominated else -2.9,
                "dominates_anchor": dominated,
                "best_promotion_blockers": ""
                if dominated
                else "not_higher_quality_than_same_chunk_public",
            }
        ],
    }


def _tradeoff_dominance(*, dominated: bool) -> dict:
    return {
        "all_public_baseline_anchors_dominated": dominated,
        "covered_public_baseline_anchor_count": 1 if dominated else 0,
        "total_public_baseline_anchor_count": 1,
        "anchors": [
            {
                "segment_ms": 640,
                "baseline_longyaal_cu_ms": 1760.0,
                "baseline_xcometxl": 74.9,
                "best_label_at_or_below_latency": "run",
                "best_longyaal_cu_ms": 1500.0,
                "best_xcometxl": 75.5 if dominated else 72.0,
                "delta_vs_baseline_xcometxl": 0.6 if dominated else -2.9,
                "dominates": dominated,
            }
        ],
    }


def test_dominance_verdict_passes_only_when_full21_clean_and_permissive():
    verdict = dominance_verdict(
        promotion_summary={
            "full21_public_baseline_anchor_coverage": _coverage(dominated=True),
            "clean_full21_public_baseline_anchor_coverage": _coverage(
                dominated=True
            ),
        },
        tradeoff_summary={
            "public_baseline_anchor_dominance": _tradeoff_dominance(
                dominated=True
            ),
            "alignatt_same_chunk_permissiveness": {
                "all_checked_alignatt_points_more_permissive": True,
                "checked_alignatt_point_count": 1,
                "more_permissive_count": 1,
                "checks": [
                    {
                        "label": "run",
                        "chunk_ms": 640,
                        "more_permissive": True,
                    }
                ],
            },
        },
    )

    assert verdict["ready_for_enzh_baseline_dominance_claim"] is True
    assert verdict["failure_count"] == 0


def test_dominance_verdict_reports_missing_anchor_and_permissiveness_failures():
    verdict = dominance_verdict(
        promotion_summary={
            "full21_public_baseline_anchor_coverage": _coverage(dominated=False),
            "clean_full21_public_baseline_anchor_coverage": _coverage(
                dominated=False
            ),
        },
        tradeoff_summary={
            "public_baseline_anchor_dominance": _tradeoff_dominance(
                dominated=False
            ),
            "alignatt_same_chunk_permissiveness": {
                "all_checked_alignatt_points_more_permissive": False,
                "checked_alignatt_point_count": 1,
                "more_permissive_count": 0,
                "checks": [
                    {
                        "label": "slow",
                        "chunk_ms": 640,
                        "longyaal_cu_ms": 2296.0,
                        "public_baseline_same_chunk_cu_ms": 1760.0,
                        "delta_cu_vs_public_same_chunk_ms": 536.0,
                        "more_permissive": False,
                    }
                ],
            },
        },
    )

    assert verdict["ready_for_enzh_baseline_dominance_claim"] is False
    assert verdict["failure_count"] == 4
    assert {
        failure["requirement"] for failure in verdict["failures"]
    } == {
        "full21_public_baseline_anchor_coverage",
        "clean_full21_public_baseline_anchor_coverage",
        "tradeoff_full21_public_baseline_anchor_dominance",
        "same_chunk_permissiveness",
    }
