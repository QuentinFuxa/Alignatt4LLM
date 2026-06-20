from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.report_enzh_candidate_promotions import (
    candidate_rows,
    clean_alignatt_subfamily,
    is_clean_alignatt_policy,
    summarize_candidates,
    write_report,
)


def _row(
    relative_dir: str,
    *,
    valid: bool = True,
    num_inputs: int = 3,
    chunk_ms: int = 640,
    longyaal_cu_ms: float = 1600.0,
    xcometxl: float | None = 0.72,
    alignatt_policy_family: str = "guarded_alignatt",
    alignatt_guard_flags: str = "acceptance_variant:unit_mass",
) -> dict:
    return {
        "relative_dir": relative_dir,
        "valid_for_claims": valid,
        "num_inputs": num_inputs,
        "target_language_code": "zh",
        "mt_backend_name": "milmmt_vllm_alignatt",
        "chunk_ms": chunk_ms,
        "longyaal_cu_ms": longyaal_cu_ms,
        "xcometxl": xcometxl,
        "bleu": 36.0,
        "chrf": 31.0,
        "alignatt_policy_family": alignatt_policy_family,
        "alignatt_guard_flags": alignatt_guard_flags,
        "translation_alignatt_acceptance_variant": "token",
        "translation_alignatt_min_source_mass": 0.02,
        "translation_alignatt_inaccessible_ms": 160,
        "translation_alignatt_frontier_min_inaccessible_mass": 0.03,
        "translation_alignatt_source_frontier_action": "stop",
        "translation_alignatt_top_k_heads": 8,
        "translation_alignatt_filter_width": 7,
        "translation_alignatt_online_normalization": "zscore",
        "translation_alignatt_source_bearing_min_source_mass": 0.05,
        "translation_alignatt_source_bearing_hard_inaccessible_cap": 0.75,
    }


def test_recoverable_frontier_is_clean_promotion_subfamily():
    assert is_clean_alignatt_policy("clean_recoverable_soft_frontier", "")
    assert (
        clean_alignatt_subfamily("clean_recoverable_soft_frontier", "")
        == "recoverable_frontier"
    )
    assert (
        clean_alignatt_subfamily(
            "clean_recoverable_argmax_frontier_source_mass_floor",
            "",
        )
        == "source_mass_floor"
    )


def test_candidate_rows_rank_promotable_mini3_before_slow_claims(tmp_path: Path):
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            [
                _row("invalid", valid=False),
                _row("wrong_lang", xcometxl=0.9) | {"target_language_code": "de"},
                _row("mini3_permissive", longyaal_cu_ms=1588.0, xcometxl=0.720),
                _row(
                    "mini3_clean_low_quality",
                    longyaal_cu_ms=1200.0,
                    xcometxl=0.650,
                    alignatt_policy_family="clean_soft_frontier_source_mass_floor",
                    alignatt_guard_flags="",
                ),
                _row(
                    "mini3_clean_claim",
                    longyaal_cu_ms=1500.0,
                    xcometxl=0.760,
                    alignatt_policy_family="clean_argmax_frontier_source_mass_floor",
                    alignatt_guard_flags="",
                ),
                _row(
                    "mini3_unit_mass_clean",
                    longyaal_cu_ms=1550.0,
                    xcometxl=0.755,
                    alignatt_policy_family="clean_unit_source_mass_floor",
                    alignatt_guard_flags="",
                )
                | {
                    "translation_alignatt_acceptance_variant": "unit_mass",
                    "translation_alignatt_min_source_mass": 0.001,
                },
                _row(
                    "mini3_source_bearing_clean",
                    longyaal_cu_ms=1250.0,
                    xcometxl=0.730,
                    alignatt_policy_family="clean_unit_source_bearing",
                    alignatt_guard_flags="",
                )
                | {
                    "translation_alignatt_acceptance_variant": (
                        "unit_mass_source_bearing"
                    ),
                    "translation_alignatt_min_source_mass": 0.0,
                    "translation_alignatt_source_bearing_min_source_mass": 0.04,
                    "translation_alignatt_source_bearing_hard_inaccessible_cap": 0.60,
                },
                _row("mini3_slow_quality", longyaal_cu_ms=2037.0, xcometxl=0.752),
                _row(
                    "full21_slow",
                    num_inputs=21,
                    longyaal_cu_ms=2296.0,
                    xcometxl=0.761,
                ),
            ]
        ),
        encoding="utf-8",
    )

    rows = candidate_rows(index_path)
    by_name = {row["relative_dir"]: row for row in rows}

    assert rows[0]["relative_dir"] == "mini3_clean_claim"
    assert rows[0]["scope"] == "mini3_probe"
    assert rows[0]["promotable_latency_probe"] is True
    assert rows[0]["clean_alignatt_latency_probe"] is True
    assert rows[0]["same_chunk_quality_beat"] is True
    assert rows[0]["clean_alignatt_same_chunk_claim_candidate"] is True
    assert rows[0]["promotion_blockers"] == ""
    assert rows[0]["clean_alignatt_subfamily"] == "source_mass_floor"
    assert rows[0]["translation_alignatt_inaccessible_ms"] == 160
    assert rows[0]["translation_alignatt_min_source_mass"] == 0.02
    assert rows[0]["translation_alignatt_frontier_min_inaccessible_mass"] == 0.03
    assert by_name["mini3_permissive"]["more_permissive_than_same_chunk_public"] is True
    assert by_name["mini3_permissive"]["delta_cu_vs_same_chunk_public_ms"] == -172.0
    assert round(by_name["mini3_permissive"]["delta_xcometxl_vs_same_chunk_public"], 3) == -2.9
    assert by_name["mini3_permissive"]["promotable_latency_probe"] is True
    assert by_name["mini3_permissive"]["clean_alignatt_latency_probe"] is False
    assert by_name["mini3_permissive"]["clean_alignatt_same_chunk_claim_candidate"] is False
    assert "guarded_or_cutoff_policy" in by_name["mini3_permissive"]["promotion_blockers"]
    assert (
        "not_higher_quality_than_same_chunk_public"
        in by_name["mini3_permissive"]["promotion_blockers"]
    )
    assert by_name["mini3_clean_low_quality"]["clean_alignatt_latency_probe"] is True
    assert by_name["mini3_clean_low_quality"]["clean_alignatt_subfamily"] == (
        "source_mass_floor"
    )
    assert (
        by_name["mini3_clean_low_quality"]["clean_alignatt_same_chunk_claim_candidate"]
        is False
    )
    assert by_name["mini3_unit_mass_clean"]["clean_alignatt_latency_probe"] is True
    assert by_name["mini3_unit_mass_clean"]["clean_alignatt_subfamily"] == (
        "source_mass_floor"
    )
    assert by_name["mini3_unit_mass_clean"]["translation_alignatt_acceptance_variant"] == (
        "unit_mass"
    )
    assert by_name["mini3_source_bearing_clean"]["clean_alignatt_latency_probe"] is True
    assert by_name["mini3_source_bearing_clean"]["clean_alignatt_subfamily"] == (
        "source_bearing_unit"
    )
    assert by_name["mini3_source_bearing_clean"][
        "translation_alignatt_acceptance_variant"
    ] == "unit_mass_source_bearing"
    assert by_name["mini3_source_bearing_clean"][
        "translation_alignatt_source_bearing_min_source_mass"
    ] == 0.04
    assert by_name["mini3_slow_quality"]["promotable_latency_probe"] is False
    assert by_name["full21_slow"]["scope"] == "full21_claim"
    assert by_name["full21_slow"]["promotable_latency_probe"] is False

    summary = summarize_candidates(rows)
    assert summary["candidate_count"] == 7
    assert summary["promotable_latency_probe_count"] == 5
    assert summary["clean_alignatt_latency_probe_count"] == 4
    assert summary["pure_frontier_latency_probe_count"] == 0
    assert summary["source_mass_floor_latency_probe_count"] == 3
    assert summary["source_bearing_unit_latency_probe_count"] == 1
    assert summary["clean_alignatt_same_chunk_claim_candidate_count"] == 2
    assert summary["best_promotable_latency_probes"][0]["relative_dir"] == "mini3_clean_claim"
    assert summary["best_clean_alignatt_latency_probes"][0]["relative_dir"] == "mini3_clean_claim"
    assert (
        summary["best_clean_alignatt_same_chunk_claim_candidates"][0]["relative_dir"]
        == "mini3_clean_claim"
    )
    assert (
        summary["best_source_mass_floor_latency_probes"][0]["relative_dir"]
        == "mini3_clean_claim"
    )
    assert (
        summary["best_source_bearing_unit_latency_probes"][0]["relative_dir"]
        == "mini3_source_bearing_clean"
    )
    coverage = summary["public_baseline_anchor_coverage"]
    assert coverage["covered_public_baseline_anchor_count"] == 1
    assert coverage["all_public_baseline_anchors_dominated"] is False
    assert coverage["anchors"][0]["segment_ms"] == 640
    assert coverage["anchors"][0]["best_relative_dir"] == "mini3_clean_claim"
    assert round(coverage["anchors"][0]["delta_xcometxl_vs_anchor"], 3) == 1.1
    clean_coverage = summary["clean_public_baseline_anchor_coverage"]
    assert clean_coverage["covered_public_baseline_anchor_count"] == 1
    assert clean_coverage["anchors"][0]["best_relative_dir"] == "mini3_clean_claim"
    assert summary["full21_public_baseline_anchor_coverage"][
        "covered_public_baseline_anchor_count"
    ] == 0
    assert summary["promotable_public_baseline_anchor_coverage"][
        "covered_public_baseline_anchor_count"
    ] == 1
    assert summary["clean_full21_public_baseline_anchor_coverage"][
        "covered_public_baseline_anchor_count"
    ] == 0


def test_write_report_outputs_tsv_and_summary_json(tmp_path: Path):
    rows = [
        _row("mini3_permissive", longyaal_cu_ms=1588.0, xcometxl=0.720),
    ]
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(rows), encoding="utf-8")
    candidate_report_rows = candidate_rows(index_path)

    paths = write_report(
        candidate_report_rows,
        output_dir=tmp_path,
        output_stem="report",
    )

    with paths["tsv"].open(encoding="utf-8", newline="") as handle:
        report_row = next(csv.DictReader(handle, delimiter="\t"))
    assert report_row["relative_dir"] == "mini3_permissive"
    assert report_row["promotable_latency_probe"] == "true"
    assert report_row["clean_alignatt_latency_probe"] == "false"
    assert report_row["same_chunk_quality_beat"] == "false"
    assert report_row["clean_alignatt_same_chunk_claim_candidate"] == "false"
    assert "guarded_or_cutoff_policy" in report_row["promotion_blockers"]
    assert report_row["clean_alignatt_subfamily"] == "guarded_or_cutoff"
    assert report_row["same_chunk_public_cu_ms"] == "1760.000"
    assert report_row["translation_alignatt_acceptance_variant"] == "token"
    assert report_row["translation_alignatt_min_source_mass"] == "0.02"
    assert report_row["translation_alignatt_inaccessible_ms"] == "160"
    assert report_row["translation_alignatt_frontier_min_inaccessible_mass"] == "0.03"
    assert report_row["translation_alignatt_source_frontier_action"] == "stop"
    assert report_row["translation_alignatt_top_k_heads"] == "8"
    assert report_row["translation_alignatt_filter_width"] == "7"
    assert report_row["translation_alignatt_online_normalization"] == "zscore"
    assert paths["anchor_coverage_tsv"].is_file()
    with paths["anchor_coverage_tsv"].open(encoding="utf-8", newline="") as handle:
        coverage_rows = list(csv.DictReader(handle, delimiter="\t"))
    assert coverage_rows[0]["coverage_scope"] == "all_valid_candidates"
    assert coverage_rows[0]["segment_ms"] == "640"
    assert coverage_rows[0]["best_relative_dir"] == "mini3_permissive"
    assert coverage_rows[0]["dominates_anchor"] == "false"
    assert coverage_rows[5]["coverage_scope"] == "clean_alignatt_candidates"
    assert coverage_rows[10]["coverage_scope"] == "promotable_latency_probes"
    assert coverage_rows[15]["coverage_scope"] == "full21_claims"
    assert coverage_rows[20]["coverage_scope"] == "clean_full21_claims"

    summary = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert summary["candidate_count"] == 1
    assert summary["promotable_latency_probe_count"] == 1
    assert summary["clean_alignatt_latency_probe_count"] == 0
    assert summary["source_mass_floor_latency_probe_count"] == 0
    assert summary["public_baseline_anchor_coverage"]["covered_public_baseline_anchor_count"] == 0
    assert summary["full21_public_baseline_anchor_coverage"][
        "covered_public_baseline_anchor_count"
    ] == 0
    assert summary["clean_full21_public_baseline_anchor_coverage"][
        "covered_public_baseline_anchor_count"
    ] == 0
