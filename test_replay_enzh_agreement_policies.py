from scripts.replay_enzh_agreement_policies import (
    replay_updates_for_policy,
    snap_back_incomplete_ascii_word,
    unit_longest_common_prefix,
)


def _update(
    idx: int,
    *,
    translation_text: str,
    draft: str,
    accepted: str,
    audio_ms: float,
    metadata: dict | None = None,
) -> dict:
    return {
        "update_idx": idx,
        "input_name": "clip.wav",
        "audio_processed_ms": audio_ms,
        "wallclock_elapsed_ms": audio_ms + 100.0,
        "translation_text": translation_text,
        "partial_draft_target": draft,
        "partial_accepted_target": accepted,
        "alignatt_metadata": metadata or {},
    }


def _hypothesis_row(prediction: str, delays: list[float]) -> dict:
    return {
        "source": ["clip.wav"],
        "source_length": 10_000.0,
        "prediction": prediction,
        "delays": delays,
    }


def test_la2_commits_clipped_unit_lcp_and_flushes_recorded_final() -> None:
    updates = [
        _update(0, translation_text="你", draft="你好世", accepted="你", audio_ms=1000.0),
        _update(1, translation_text="你好", draft="你好世界", accepted="你好", audio_ms=2000.0),
        _update(2, translation_text="你好朋", draft="你好朋友们", accepted="你好朋", audio_ms=3000.0),
    ]
    hypothesis, summary = replay_updates_for_policy(
        stream_updates=updates,
        original_hypothesis=[_hypothesis_row("你好朋", [1000.0, 2000.0, 3000.0])],
        policy="la",
        agreement_n=2,
        confidence_threshold=None,
        target_lang_code="zh",
    )
    assert summary["agreement_history_short_update_count"] == 1
    assert summary["final_flush_update_count"] == 1
    assert summary["rejected_update_count"] == 1
    assert summary["agreement_offtrajectory_unit_count"] == 1
    row = hypothesis[0]
    assert row["prediction"] == "你好朋"
    assert row["delays"] == [2000.0, 2000.0, 3000.0]


def test_hybrid_rescues_deferred_tail_and_stays_append_only() -> None:
    updates = [
        _update(0, translation_text="你好", draft="你好朋友", accepted="你好", audio_ms=1000.0),
        _update(1, translation_text="你好朋", draft="你好朋友们", accepted="你好朋", audio_ms=2000.0),
        _update(2, translation_text="你好朋友们", draft="你好朋友们", accepted="你好朋友们", audio_ms=3000.0),
    ]
    hypothesis, summary = replay_updates_for_policy(
        stream_updates=updates,
        original_hypothesis=[_hypothesis_row("你好朋友们", [1000.0, 1000.0, 2000.0, 3000.0, 3000.0])],
        policy="hybrid_recorded",
        agreement_n=2,
        confidence_threshold=None,
        target_lang_code="zh",
    )
    assert summary["agreement_branch_win_count"] == 1
    assert summary["agreement_rescued_unit_count"] == 1
    row = hypothesis[0]
    assert row["prediction"] == "你好朋友们"
    assert row["delays"] == [1000.0, 1000.0, 2000.0, 2000.0, 3000.0]
    assert row["delays"] == sorted(row["delays"])


def test_unit_longest_common_prefix_and_ascii_snap() -> None:
    assert unit_longest_common_prefix([["你", "好", "J"], ["你", "好", "K"]]) == ["你", "好"]
    assert unit_longest_common_prefix([]) == []

    full = ["你", "J", "e", "n", "n", "y", "。"]
    assert snap_back_incomplete_ascii_word(full[:3], full) == ["你"]
    assert snap_back_incomplete_ascii_word(full[:7], full) == full
    assert snap_back_incomplete_ascii_word(["你"], full) == ["你"]
