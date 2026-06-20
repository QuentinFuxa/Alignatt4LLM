import math

from tools.reports.report_unit_concentration_separability import (
    accepted_unit_feature_spans,
    floor_simulation,
    matched_threshold,
    unit_feature_minima,
)


def test_accepted_unit_feature_spans_filters_word_boundary_trim() -> None:
    metadata = {
        "target_stability_unit_end_token_indices": [2, 3, 5],
        "accepted_token_count": 4,
        "attention_confidence_per_draft_token": [{} for _ in range(5)],
    }
    assert accepted_unit_feature_spans(metadata) == [(0, 2), (2, 3)]

    assert accepted_unit_feature_spans({"target_stability_unit_end_token_indices": []}) == []

    no_count = {
        "target_stability_unit_end_token_indices": [1, 2],
        "attention_confidence_per_draft_token": [{}, {}],
    }
    assert accepted_unit_feature_spans(no_count) == [(0, 1), (1, 2)]


def test_unit_feature_minima_takes_weakest_finite_token() -> None:
    features = [
        {"concentration": 0.4},
        {"concentration": 0.1},
        {"concentration": None},
        {"concentration": float("nan")},
    ]
    minima = unit_feature_minima(features, [(0, 2), (2, 4)], "concentration")
    assert minima[0] == 0.1
    assert minima[1] is None


def test_floor_simulation_cascade_defers_everything_after_first_weak_unit() -> None:
    per_update = [[0.5, 0.01, 0.5], [0.5, 0.5]]
    result = floor_simulation(per_update, 0.05)
    assert result["n_units"] == 5
    assert result["deferred_frac_independent"] == 1 / 5
    assert result["deferred_frac_cascade"] == 2 / 5
    assert result["none_unit_count"] == 0

    with_none = floor_simulation([[None, 0.5]], 0.05)
    assert with_none["none_unit_count"] == 1
    assert with_none["deferred_frac_cascade"] == 1.0


def test_matched_threshold_hits_target_deferral_quantile() -> None:
    minima = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    threshold = matched_threshold(minima, 0.3)
    assert threshold == 0.4
    assert sum(1 for value in minima if value < threshold) == 3
    assert matched_threshold(minima, 0.0) == -math.inf
    assert matched_threshold(minima, 1.0) == math.inf
    assert matched_threshold([], 0.5) is None
