from scripts.report_learned_acceptance_calibration import (
    survival_positions,
    unit_char_spans,
    unit_feature_row,
)


def test_survival_positions_marks_match_blocks() -> None:
    low = ["你", "好", "世", "界", "。"]
    high = ["你", "好", "朋", "界", "。"]
    survived = survival_positions(low, high)
    assert 0 in survived and 1 in survived
    assert 2 not in survived
    assert 3 in survived and 4 in survived
    assert survival_positions([], high) == set()


def test_unit_char_spans_partition_the_tail_exactly() -> None:
    spans = unit_char_spans(3, tail_start=10, tail_length=7)
    assert spans[0][0] == 10
    assert spans[-1][1] == 17
    assert all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1))
    assert sum(end - start for start, end in spans) == 7

    one = unit_char_spans(1, tail_start=0, tail_length=4)
    assert one == [(0, 4)]


def test_unit_feature_row_min_mean_and_missing() -> None:
    features = [
        {"consensus_ratio": 0.8, "concentration": 0.2},
        {"consensus_ratio": 0.4, "concentration": None},
    ]
    row = unit_feature_row(features, (0, 2))
    assert row["n_tokens"] == 2.0
    assert row["min_consensus_ratio"] == 0.4
    assert row["mean_consensus_ratio"] == 0.6000000000000001
    assert row["min_concentration"] == 0.2
    assert row["min_entropy_norm"] is None
