from __future__ import annotations

import pytest

from tools.reports.report_enzh_source_mass_thresholds import summarize_threshold


def test_source_mass_threshold_simulation_trims_to_stable_prefix():
    index_row = {
        "relative_dir": "clean_probe",
        "num_inputs": 3,
        "chunk_ms": 640,
        "longyaal_cu_ms": 1000.0,
        "xcometxl": 0.7,
        "alignatt_policy_family": "pure_soft_frontier",
        "alignatt_guard_flags": "",
        "translation_alignatt_min_source_mass": 0.001,
    }
    stream_updates = [
        {
            "alignatt_metadata": {
                "accepted_candidate_token_count": 4,
                "accepted_token_count": 4,
                "target_stability_unit_end_token_indices": [1, 2, 3, 4],
                "draft_target_stability_unit_end_token_indices": [1, 2, 3, 4],
                "provenance_per_draft_token": [
                    {
                        "source_accessible": 0.50,
                        "source_inaccessible": 0.0,
                        "non_source_prompt": 0.20,
                        "suffix": 0.01,
                    },
                    {
                        "source_accessible": 0.04,
                        "source_inaccessible": 0.01,
                        "non_source_prompt": 0.40,
                        "suffix": 0.02,
                    },
                    {
                        "source_accessible": 0.001,
                        "source_inaccessible": 0.0005,
                        "non_source_prompt": 0.95,
                        "suffix": 0.02,
                    },
                    {
                        "source_accessible": 0.30,
                        "source_inaccessible": 0.0,
                        "non_source_prompt": 0.10,
                        "suffix": 0.15,
                    },
                ],
            },
        },
        {
            "alignatt_metadata": {
                "accepted_candidate_token_count": 2,
                "accepted_token_count": 2,
                "target_stability_unit_end_token_indices": [1, 2],
                "draft_target_stability_unit_end_token_indices": [],
                "provenance_per_draft_token": [
                    {
                        "source_accessible": 0.001,
                        "source_inaccessible": 0.0,
                        "non_source_prompt": 0.90,
                        "suffix": 0.01,
                    },
                    {
                        "source_accessible": 0.30,
                        "source_inaccessible": 0.0,
                        "non_source_prompt": 0.20,
                        "suffix": 0.01,
                    },
                ],
            },
        },
    ]

    row = summarize_threshold(index_row, stream_updates, threshold=0.002)

    assert row["updates_with_provenance"] == 2
    assert row["updates_with_draft_unit_boundaries"] == 1
    assert row["updates_without_draft_unit_boundaries"] == 1
    assert row["draft_target_stability_unit_count_total"] == 4
    assert row["original_accepted_token_count"] == 6
    assert row["simulated_accepted_token_count"] == 2
    assert row["accepted_prefix_simulated_token_count"] == 2
    assert row["updates_trimmed_by_threshold"] == 2
    assert row["updates_emptied_by_threshold"] == 1
    assert row["updates_trimmed_by_accepted_prefix"] == 2
    assert row["updates_emptied_by_accepted_prefix"] == 1
    assert row["accepted_tokens_below_threshold"] == 2
    assert row["accepted_tokens_source_mass_below_threshold"] == 2
    assert row["accepted_tokens_non_source_dominant"] == 3
    assert row["accepted_tokens_non_source_ge_0p80"] == 2
    assert row["accepted_tokens_suffix_ge_0p10"] == 1
    assert row["accepted_token_mean_source_accessible"] == pytest.approx(0.1903333333)
    assert row["accepted_token_mean_source_mass"] == pytest.approx(0.1920833333)
    assert row["accepted_token_mean_non_source_prompt"] == pytest.approx(0.4583333333)


def test_source_mass_threshold_keeps_final_source_flush():
    index_row = {"relative_dir": "final_probe"}
    stream_updates = [
        {
            "alignatt_metadata": {
                "final_source_completed_full_accept": True,
                "accepted_candidate_token_count": 2,
                "accepted_token_count": 2,
                "target_stability_unit_end_token_indices": [1, 2],
                "provenance_per_draft_token": [
                    {"source_accessible": 0.0},
                    {"source_accessible": 0.0},
                ],
            },
        },
    ]

    row = summarize_threshold(index_row, stream_updates, threshold=0.002)

    assert row["original_accepted_token_count"] == 2
    assert row["simulated_accepted_token_count"] == 2
    assert row["accepted_prefix_simulated_token_count"] == 2
    assert row["updates_trimmed_by_threshold"] == 0
    assert row["updates_trimmed_by_accepted_prefix"] == 0


def test_accepted_prefix_summary_allows_low_early_token_if_recent_units_pass():
    index_row = {"relative_dir": "accepted_prefix_probe"}
    stream_updates = [
        {
            "alignatt_metadata": {
                "accepted_candidate_token_count": 3,
                "accepted_token_count": 3,
                "target_stability_unit_end_token_indices": [1, 2, 3],
                "provenance_per_draft_token": [
                    {"source_accessible": 0.0005},
                    {"source_accessible": 0.40},
                    {"source_accessible": 0.30},
                ],
            },
        },
    ]

    row = summarize_threshold(index_row, stream_updates, threshold=0.001)

    assert row["simulated_accepted_token_count"] == 0
    assert row["accepted_prefix_simulated_token_count"] == 3
    assert row["updates_trimmed_by_threshold"] == 1
    assert row["updates_trimmed_by_accepted_prefix"] == 0
