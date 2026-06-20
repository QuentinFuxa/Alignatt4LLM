from __future__ import annotations

import json
from pathlib import Path

from scripts.replay_enzh_source_regression_trim import (
    replay_updates,
    source_regression_trim_partial_target,
    write_replay_artifacts,
)


def runtime_config() -> dict:
    return {
        "translation_alignatt_max_source_regression": 0,
        "translation_alignatt_source_regression_recent_tokens": 0,
        "translation_alignatt_source_regression_reference_mode": "max",
        "translation_alignatt_source_regression_patience_tokens": 1,
        "translation_alignatt_border_margin": 1,
        "translation_alignatt_frontier_min_inaccessible_mass": 0.03,
        "translation_alignatt_token_argmax_frontier_gate": False,
        "translation_alignatt_min_source_mass": 0.0,
        "translation_alignatt_max_inaccessible_source_mass": 1.0,
        "translation_alignatt_max_non_source_prompt_mass": 1.0,
        "translation_alignatt_min_accessible_inaccessible_margin": -1.0,
    }


def source_regression_update(*, positions: list[int], draft: str) -> dict:
    return {
        "input_name": "a.wav",
        "wav_name": "a.wav",
        "audio_processed_ms": 1000.0,
        "wallclock_elapsed_ms": 50.0,
        "translation_text": "甲",
        "partial_accepted_target": "甲",
        "partial_draft_target": draft,
        "new_words": ["甲"],
        "alignatt_metadata": {
            "stop_reason": "alignatt:source_regression",
            "accepted_candidate_token_count": 1,
            "accepted_token_count": 1,
            "unsafe_target_token_index": 1,
            "blocked_source_local_position": positions[1],
            "accessible_source_local_end_exclusive": 8,
            "aligned_source_local_positions": positions,
            "provenance_per_draft_token": [
                {"source_accessible": 0.10, "source_inaccessible": 0.0},
                {"source_accessible": 0.10, "source_inaccessible": 0.0},
                {"source_accessible": 0.10, "source_inaccessible": 0.0},
            ],
        },
    }


def test_trim_replay_expands_when_regression_recovers_in_same_draft():
    simulated, decision = source_regression_trim_partial_target(
        source_regression_update(positions=[5, 2, 6], draft="甲乙丙"),
        previous_partial_target="",
        runtime_config=runtime_config(),
        target_lang_code="zh",
        gate_aware=True,
    )

    assert simulated == "甲乙丙"
    assert decision["changed"] is True
    assert decision["simulated_token_prefix"] == 3


def test_trim_replay_keeps_original_when_regression_suffix_never_recovers():
    simulated, decision = source_regression_trim_partial_target(
        source_regression_update(positions=[5, 2, 3], draft="甲乙丙"),
        previous_partial_target="",
        runtime_config=runtime_config(),
        target_lang_code="zh",
        gate_aware=True,
    )

    assert simulated == "甲"
    assert decision["changed"] is False
    assert decision["simulated_token_prefix"] == 1


def test_trim_replay_refuses_non_monotone_surface_rewrite():
    update = source_regression_update(positions=[5, 2, 6], draft="甲丁丙")
    update["partial_accepted_target"] = "甲乙"
    update["translation_text"] = "甲乙"

    simulated, decision = source_regression_trim_partial_target(
        update,
        previous_partial_target="",
        runtime_config=runtime_config(),
        target_lang_code="zh",
        gate_aware=True,
    )

    assert simulated == "甲乙"
    assert decision["changed"] is False
    assert decision["non_monotone_surface"] is True


def test_replay_updates_recomputes_append_only_delays_for_expanded_prefix():
    rows, summary = replay_updates(
        stream_updates=[source_regression_update(positions=[5, 2, 6], draft="甲乙丙")],
        original_hypothesis=[
            {
                "source": ["a.wav"],
                "source_length": 3000.0,
                "prediction": "甲",
                "delays": [1000.0],
            }
        ],
        runtime_config=runtime_config(),
        target_lang_code="zh",
        gate_aware=True,
    )

    assert rows[0]["prediction"] == "甲乙丙"
    assert rows[0]["delays"] == [1000.0, 1000.0, 1000.0]
    assert summary["source_regression_update_count"] == 1
    assert summary["expanded_partial_update_count"] == 1
    assert summary["added_unit_count"] == 2


def test_write_replay_artifacts_marks_diagnostic_only(tmp_path):
    artifact_dir = tmp_path / "source"
    artifact_dir.mkdir()
    (artifact_dir / "manifest.json").write_text(
        json.dumps({"runtime_config": runtime_config()}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    write_replay_artifacts(
        output_dir=output_dir,
        artifact_dir=artifact_dir,
        target_lang_code="zh",
        gate_aware=True,
        hypothesis_rows=[
            {
                "source": ["a.wav"],
                "source_length": 3000.0,
                "prediction": "甲乙丙",
                "delays": [1000.0, 1000.0, 1000.0],
                "elapsed": [50.0, 50.0, 50.0],
                "elapsed_semantics": "ca_compatible_incremental",
            }
        ],
        summary={"kind": "source_regression_trim_unrecovered"},
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["offline_replay"]["diagnostic_only"] is True
    assert (
        manifest["runtime_config"]["translation_alignatt_source_regression_action"]
        == "trim_unrecovered"
    )
    assert (output_dir / "hypothesis.jsonl").is_file()
