from __future__ import annotations

from tools.reports.replay_enzh_source_mass_thresholds import (
    has_unit_replay_draft_boundaries,
    replay_updates_for_threshold,
)


def test_replay_source_mass_threshold_keeps_append_only_prefix():
    stream_updates = [
        {
            "input_name": "a.wav",
            "audio_processed_ms": 1000.0,
            "wallclock_elapsed_ms": 100.0,
            "translation_text": "你好",
            "partial_accepted_target": "你好",
            "partial_draft_target": "你好",
            "alignatt_metadata": {
                "accepted_token_count": 2,
                "accepted_candidate_token_count": 2,
                "target_stability_unit_end_token_indices": [1, 2],
                "provenance_per_draft_token": [
                    {"source_accessible": 0.50},
                    {"source_accessible": 0.40},
                ],
            },
        },
        {
            "input_name": "a.wav",
            "audio_processed_ms": 2000.0,
            "wallclock_elapsed_ms": 200.0,
            "translation_text": "你好世界",
            "partial_accepted_target": "你好世界",
            "partial_draft_target": "你好世界",
            "alignatt_metadata": {
                "accepted_token_count": 2,
                "accepted_candidate_token_count": 2,
                "target_stability_unit_end_token_indices": [1, 2],
                "provenance_per_draft_token": [
                    {"source_accessible": 0.001},
                    {"source_accessible": 0.60},
                ],
            },
        },
    ]
    original_hypothesis = [{"source": ["a.wav"], "source_length": 3000.0}]

    rows, summary = replay_updates_for_threshold(
        stream_updates=stream_updates,
        original_hypothesis=original_hypothesis,
        threshold=0.01,
        target_lang_code="zh",
    )

    assert rows[0]["prediction"] == "你好"
    assert rows[0]["delays"] == [1000.0, 1000.0]
    assert summary["accepted_update_count"] == 1
    assert summary["rejected_update_count"] == 1
    assert summary["trimmed_partial_update_count"] == 1


def test_replay_source_mass_threshold_can_append_partial_continuation():
    stream_updates = [
        {
            "input_name": "a.wav",
            "audio_processed_ms": 1000.0,
            "wallclock_elapsed_ms": 100.0,
            "translation_text": "你好",
            "partial_accepted_target": "你好",
            "partial_draft_target": "你好",
            "alignatt_metadata": {
                "accepted_token_count": 2,
                "target_stability_unit_end_token_indices": [1, 2],
                "provenance_per_draft_token": [
                    {"source_accessible": 0.50},
                    {"source_accessible": 0.40},
                ],
            },
        },
        {
            "input_name": "a.wav",
            "audio_processed_ms": 2000.0,
            "wallclock_elapsed_ms": 200.0,
            "translation_text": "你好世界",
            "partial_accepted_target": "你好世界",
            "partial_draft_target": "你好世界",
            "alignatt_metadata": {
                "accepted_token_count": 2,
                "target_stability_unit_end_token_indices": [1, 2],
                "provenance_per_draft_token": [
                    {"source_accessible": 0.60},
                    {"source_accessible": 0.001},
                ],
            },
        },
    ]
    original_hypothesis = [{"source": ["a.wav"], "source_length": 3000.0}]

    rows, summary = replay_updates_for_threshold(
        stream_updates=stream_updates,
        original_hypothesis=original_hypothesis,
        threshold=0.01,
        target_lang_code="zh",
    )

    assert rows[0]["prediction"] == "你好世"
    assert rows[0]["delays"] == [1000.0, 1000.0, 2000.0]
    assert summary["accepted_update_count"] == 2


def test_replay_accepted_prefix_keeps_low_early_token_when_recent_units_pass():
    stream_updates = [
        {
            "input_name": "a.wav",
            "audio_processed_ms": 1000.0,
            "wallclock_elapsed_ms": 100.0,
            "translation_text": "你好世",
            "partial_accepted_target": "你好世",
            "partial_draft_target": "你好世",
            "alignatt_metadata": {
                "accepted_token_count": 3,
                "accepted_candidate_token_count": 3,
                "target_stability_unit_end_token_indices": [1, 2, 3],
                "provenance_per_draft_token": [
                    {"source_accessible": 0.0005},
                    {"source_accessible": 0.40},
                    {"source_accessible": 0.30},
                ],
            },
        },
    ]
    original_hypothesis = [{"source": ["a.wav"], "source_length": 3000.0}]

    rows, summary = replay_updates_for_threshold(
        stream_updates=stream_updates,
        original_hypothesis=original_hypothesis,
        threshold=0.001,
        target_lang_code="zh",
        replay_variant="accepted_prefix",
    )

    assert rows[0]["prediction"] == "你好世"
    assert rows[0]["delays"] == [1000.0, 1000.0, 1000.0]
    assert summary["accepted_update_count"] == 1
    assert summary["trimmed_partial_update_count"] == 0


def test_replay_accepted_prefix_trims_weak_recent_unit():
    stream_updates = [
        {
            "input_name": "a.wav",
            "audio_processed_ms": 1000.0,
            "wallclock_elapsed_ms": 100.0,
            "translation_text": "你好世",
            "partial_accepted_target": "你好世",
            "partial_draft_target": "你好世",
            "alignatt_metadata": {
                "accepted_token_count": 3,
                "accepted_candidate_token_count": 3,
                "target_stability_unit_end_token_indices": [1, 2, 3],
                "provenance_per_draft_token": [
                    {"source_accessible": 0.009},
                    {"source_accessible": 0.00001},
                    {"source_accessible": 0.003},
                ],
            },
        },
    ]
    original_hypothesis = [{"source": ["a.wav"], "source_length": 3000.0}]

    rows, summary = replay_updates_for_threshold(
        stream_updates=stream_updates,
        original_hypothesis=original_hypothesis,
        threshold=0.001,
        target_lang_code="zh",
        replay_variant="accepted_prefix",
    )

    assert rows[0]["prediction"] == "你"
    assert rows[0]["delays"] == [1000.0]
    assert summary["accepted_update_count"] == 1
    assert summary["trimmed_partial_update_count"] == 1


def test_replay_unit_mass_can_use_full_draft_unit_boundaries():
    stream_updates = [
        {
            "input_name": "a.wav",
            "audio_processed_ms": 1000.0,
            "wallclock_elapsed_ms": 100.0,
            "translation_text": "你",
            "partial_accepted_target": "你",
            "partial_draft_target": "你好",
                "alignatt_metadata": {
                    "accepted_token_count": 1,
                    "accepted_candidate_token_count": 1,
                    "accessible_source_local_end_exclusive": 2,
                    "aligned_source_local_positions": [1, 1],
                    "alignatt_unit_policy_border_margin": 0,
                    "alignatt_frontier_min_inaccessible_mass": 0.0,
                    "target_stability_unit_end_token_indices": [1],
                    "draft_target_stability_unit_end_token_indices": [1, 2],
                    "provenance_per_draft_token": [
                    {"source_accessible": 0.50},
                    {"source_accessible": 0.40},
                ],
            },
        },
    ]
    original_hypothesis = [{"source": ["a.wav"], "source_length": 3000.0}]

    rows, summary = replay_updates_for_threshold(
        stream_updates=stream_updates,
        original_hypothesis=original_hypothesis,
        threshold=0.01,
        target_lang_code="zh",
        replay_variant="unit_mass",
    )

    assert rows[0]["prediction"] == "你好"
    assert rows[0]["delays"] == [1000.0, 1000.0]
    assert summary["replay_variant"] == "unit_mass"
    assert summary["accepted_update_count"] == 1
    assert summary["expanded_partial_update_count"] == 1
    assert summary["trimmed_partial_update_count"] == 0
    assert summary["unit_mass_draft_boundary_update_count"] == 1
    assert summary["unit_mass_fallback_boundary_update_count"] == 0
    assert summary["unit_replay_draft_boundary_update_count"] == 1
    assert summary["unit_replay_fallback_boundary_update_count"] == 0


def test_replay_source_bearing_uses_source_mass_not_accessible_only():
    stream_updates = [
        {
            "input_name": "a.wav",
            "audio_processed_ms": 1000.0,
            "wallclock_elapsed_ms": 100.0,
            "translation_text": "你",
            "partial_accepted_target": "你",
            "partial_draft_target": "你好",
            "alignatt_metadata": {
                "accepted_token_count": 1,
                "accepted_candidate_token_count": 1,
                "target_stability_unit_end_token_indices": [1],
                "draft_target_stability_unit_end_token_indices": [1, 2],
                "accessible_source_local_end_exclusive": 2,
                "aligned_source_local_positions": [1, 1],
                "alignatt_unit_policy_border_margin": 0,
                "alignatt_frontier_min_inaccessible_mass": 0.0,
                "provenance_per_draft_token": [
                    {"source_accessible": 0.001, "source_inaccessible": 0.06},
                    {"source_accessible": 0.001, "source_inaccessible": 0.001},
                ],
            },
        },
    ]
    original_hypothesis = [{"source": ["a.wav"], "source_length": 3000.0}]

    rows, summary = replay_updates_for_threshold(
        stream_updates=stream_updates,
        original_hypothesis=original_hypothesis,
        threshold=0.05,
        target_lang_code="zh",
        replay_variant="unit_mass_source_bearing",
        source_bearing_hard_inaccessible_cap=0.75,
    )

    assert rows[0]["prediction"] == "你"
    assert rows[0]["delays"] == [1000.0]
    assert summary["replay_variant"] == "unit_mass_source_bearing"
    assert summary["unit_replay_draft_boundary_update_count"] == 1
    assert summary["source_bearing_hard_inaccessible_cap"] == 0.75


def test_replay_source_bearing_blocks_future_frontier_token():
    stream_updates = [
        {
            "input_name": "a.wav",
            "audio_processed_ms": 1000.0,
            "wallclock_elapsed_ms": 100.0,
            "translation_text": "你",
            "partial_accepted_target": "你",
            "partial_draft_target": "你",
            "alignatt_metadata": {
                "accepted_token_count": 1,
                "target_stability_unit_end_token_indices": [1],
                "draft_target_stability_unit_end_token_indices": [1],
                "accessible_source_local_end_exclusive": 1,
                "aligned_source_local_positions": [3],
                "alignatt_unit_policy_border_margin": 0,
                "alignatt_frontier_min_inaccessible_mass": 0.0,
                "provenance_per_draft_token": [
                    {"source_accessible": 0.08, "source_inaccessible": 0.01},
                ],
            },
        },
    ]
    original_hypothesis = [{"source": ["a.wav"], "source_length": 3000.0}]

    rows, summary = replay_updates_for_threshold(
        stream_updates=stream_updates,
        original_hypothesis=original_hypothesis,
        threshold=0.05,
        target_lang_code="zh",
        replay_variant="unit_mass_source_bearing",
        source_bearing_hard_inaccessible_cap=0.75,
    )

    assert rows[0]["prediction"] == ""
    assert rows[0]["delays"] == []
    assert summary["accepted_update_count"] == 0


def test_replay_source_bearing_preserves_soft_frontier_permissiveness():
    stream_updates = [
        {
            "input_name": "a.wav",
            "audio_processed_ms": 1000.0,
            "wallclock_elapsed_ms": 100.0,
            "translation_text": "你",
            "partial_accepted_target": "你",
            "partial_draft_target": "你",
            "alignatt_metadata": {
                "accepted_token_count": 1,
                "target_stability_unit_end_token_indices": [1],
                "draft_target_stability_unit_end_token_indices": [1],
                "accessible_source_local_end_exclusive": 1,
                "aligned_source_local_positions": [3],
                "alignatt_unit_policy_border_margin": 0,
                "alignatt_frontier_min_inaccessible_mass": 0.03,
                "provenance_per_draft_token": [
                    {"source_accessible": 0.08, "source_inaccessible": 0.01},
                ],
            },
        },
    ]
    original_hypothesis = [{"source": ["a.wav"], "source_length": 3000.0}]

    rows, summary = replay_updates_for_threshold(
        stream_updates=stream_updates,
        original_hypothesis=original_hypothesis,
        threshold=0.05,
        target_lang_code="zh",
        replay_variant="unit_mass_source_bearing",
        source_bearing_hard_inaccessible_cap=0.75,
    )

    assert rows[0]["prediction"] == "你"
    assert rows[0]["delays"] == [1000.0]
    assert summary["accepted_update_count"] == 1


def test_replay_source_bearing_blocks_hard_future_source_cap():
    stream_updates = [
        {
            "input_name": "a.wav",
            "audio_processed_ms": 1000.0,
            "wallclock_elapsed_ms": 100.0,
            "translation_text": "你",
            "partial_accepted_target": "你",
            "partial_draft_target": "你",
                "alignatt_metadata": {
                    "accepted_token_count": 1,
                    "target_stability_unit_end_token_indices": [1],
                    "draft_target_stability_unit_end_token_indices": [1],
                    "accessible_source_local_end_exclusive": 1,
                    "aligned_source_local_positions": [0],
                    "alignatt_unit_policy_border_margin": 0,
                    "alignatt_frontier_min_inaccessible_mass": 0.0,
                    "provenance_per_draft_token": [
                        {"source_accessible": 0.001, "source_inaccessible": 0.80},
                    ],
            },
        },
    ]
    original_hypothesis = [{"source": ["a.wav"], "source_length": 3000.0}]

    rows, summary = replay_updates_for_threshold(
        stream_updates=stream_updates,
        original_hypothesis=original_hypothesis,
        threshold=0.05,
        target_lang_code="zh",
        replay_variant="unit_mass_source_bearing",
        source_bearing_hard_inaccessible_cap=0.75,
    )

    assert rows[0]["prediction"] == ""
    assert rows[0]["delays"] == []
    assert summary["accepted_update_count"] == 0
    assert summary["rejected_update_count"] == 1
    assert summary["emptied_partial_update_count"] == 1


def test_replay_source_bearing_default_cap_is_uncapped():
    stream_updates = [
        {
            "input_name": "a.wav",
            "audio_processed_ms": 1000.0,
            "wallclock_elapsed_ms": 100.0,
            "translation_text": "你",
            "partial_accepted_target": "你",
            "partial_draft_target": "你",
            "alignatt_metadata": {
                "accepted_token_count": 1,
                "target_stability_unit_end_token_indices": [1],
                "draft_target_stability_unit_end_token_indices": [1],
                "accessible_source_local_end_exclusive": 1,
                "aligned_source_local_positions": [0],
                "alignatt_unit_policy_border_margin": 0,
                "alignatt_frontier_min_inaccessible_mass": 0.0,
                "provenance_per_draft_token": [
                    {"source_accessible": 0.001, "source_inaccessible": 0.80},
                ],
            },
        },
    ]
    original_hypothesis = [{"source": ["a.wav"], "source_length": 3000.0}]

    rows, summary = replay_updates_for_threshold(
        stream_updates=stream_updates,
        original_hypothesis=original_hypothesis,
        threshold=0.05,
        target_lang_code="zh",
        replay_variant="unit_mass_source_bearing",
    )

    assert rows[0]["prediction"] == "你"
    assert summary["accepted_update_count"] == 1
    assert summary["source_bearing_hard_inaccessible_cap"] == 1.0


def test_replay_source_bearing_surface_boundaries_expand_old_trace_diagnostics():
    stream_updates = [
        {
            "input_name": "a.wav",
            "audio_processed_ms": 1000.0,
            "wallclock_elapsed_ms": 100.0,
            "translation_text": "你",
            "partial_accepted_target": "你",
            "partial_draft_target": "你好",
            "alignatt_metadata": {
                "accepted_token_count": 1,
                "target_stability_unit_end_token_indices": [1],
                "accessible_source_local_end_exclusive": 2,
                "aligned_source_local_positions": [0, 1],
                "alignatt_unit_policy_border_margin": 0,
                "alignatt_frontier_min_inaccessible_mass": 0.0,
                "provenance_per_draft_token": [
                    {"source_accessible": 0.08, "source_inaccessible": 0.0},
                    {"source_accessible": 0.09, "source_inaccessible": 0.0},
                ],
            },
        },
    ]
    original_hypothesis = [{"source": ["a.wav"], "source_length": 3000.0}]

    fallback_rows, fallback_summary = replay_updates_for_threshold(
        stream_updates=stream_updates,
        original_hypothesis=original_hypothesis,
        threshold=0.05,
        target_lang_code="zh",
        replay_variant="unit_mass_source_bearing",
    )
    surface_rows, surface_summary = replay_updates_for_threshold(
        stream_updates=stream_updates,
        original_hypothesis=original_hypothesis,
        threshold=0.05,
        target_lang_code="zh",
        replay_variant="unit_mass_source_bearing",
        allow_surface_unit_boundaries=True,
    )

    assert fallback_rows[0]["prediction"] == "你"
    assert fallback_summary["unit_replay_surface_boundary_update_count"] == 0
    assert fallback_summary["unit_replay_fallback_boundary_update_count"] == 1
    assert surface_rows[0]["prediction"] == "你好"
    assert surface_summary["expanded_partial_update_count"] == 1
    assert surface_summary["unit_replay_surface_boundary_update_count"] == 1
    assert surface_summary["unit_replay_fallback_boundary_update_count"] == 0


def test_unit_replay_draft_boundary_detector_covers_source_bearing():
    assert (
        has_unit_replay_draft_boundaries(
            [
                {
                    "alignatt_metadata": {
                        "target_stability_unit_end_token_indices": [1],
                        "draft_target_stability_unit_end_token_indices": [],
                    }
                }
            ]
        )
        is False
    )
    assert (
        has_unit_replay_draft_boundaries(
            [
                {
                    "alignatt_metadata": {
                        "draft_target_stability_unit_end_token_indices": [1, 2],
                    }
                }
            ]
        )
        is True
    )
