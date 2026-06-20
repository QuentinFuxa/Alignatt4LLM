from __future__ import annotations

import sys
from types import SimpleNamespace

from cascade.simulstream_processor import CascadeAlignAttProcessor
from cascade.runtime import CascadeRuntimeConfig, CascadeSession, LoadedModelBundle
from run_simulstream_batch import (
    chunk_decision_record,
    compact_alignatt_metadata,
    canonical_language_code,
    alignatt_metadata_is_current_chunk,
    parse_args,
    summarize_chunk_decisions,
)


def test_canonical_language_code_accepts_names_and_codes():
    assert canonical_language_code("English") == "en"
    assert canonical_language_code("Simplified Chinese") == "zh"
    assert canonical_language_code("zh") == "zh"


def test_batch_parser_default_disables_inaccessible_source_cap(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_simulstream_batch.py",
            "--inputs",
            "data/smoke/alignatt_smoke18.wav",
            "--output-dir",
            "outputs/tmp",
        ],
    )

    args = parse_args()

    assert args.translation_alignatt_min_source_mass == 0.0
    assert args.translation_alignatt_max_inaccessible_source_mass == 1.0
    assert args.translation_alignatt_source_bearing_min_source_mass == 0.005
    assert args.translation_alignatt_source_bearing_hard_inaccessible_cap == 1.0


def test_processor_config_preserves_source_regression_mass_gate():
    runtime_config = CascadeAlignAttProcessor._build_runtime_config(
        SimpleNamespace(
            source_lang_code="en",
            target_lang_code="zh",
            translation_alignatt_source_regression_min_source_mass=0.15,
            translation_alignatt_source_regression_min_inaccessible_mass=0.04,
            translation_alignatt_max_non_source_prompt_mass=0.80,
            translation_alignatt_source_regression_recent_tokens=4,
            translation_alignatt_source_regression_activation_mode="frontier_reached",
            translation_alignatt_source_regression_activation_slack_tokens=3,
            translation_alignatt_source_frontier_action="trim_unrecovered",
            translation_alignatt_token_argmax_frontier_patience_tokens=2,
            translation_alignatt_source_lcp_stability=True,
            translation_alignatt_source_lcp_append_slack_units=2,
            gemma_max_model_len=2048,
            mt_max_model_len=1536,
            asr_gpu_memory_utilization=0.35,
        )
    )

    assert runtime_config.translation_alignatt_source_regression_min_source_mass == 0.15
    assert (
        runtime_config.translation_alignatt_source_regression_min_inaccessible_mass
        == 0.04
    )
    assert runtime_config.translation_alignatt_max_non_source_prompt_mass == 0.80
    assert runtime_config.translation_alignatt_source_regression_recent_tokens == 4
    assert (
        runtime_config.translation_alignatt_source_regression_activation_mode
        == "frontier_reached"
    )
    assert runtime_config.translation_alignatt_source_regression_activation_slack_tokens == 3
    assert runtime_config.translation_alignatt_source_frontier_action == "trim_unrecovered"
    assert runtime_config.translation_alignatt_token_argmax_frontier_patience_tokens == 2
    assert runtime_config.translation_alignatt_source_lcp_stability is True
    assert runtime_config.translation_alignatt_source_lcp_append_slack_units == 2
    assert runtime_config.gemma_max_model_len == 2048
    assert runtime_config.mt_max_model_len == 1536
    assert runtime_config.asr_gpu_memory_utilization == 0.35


def test_processor_records_append_only_rejection_reason():
    processor = object.__new__(CascadeAlignAttProcessor)
    processor._target_lang_code = "zh"
    processor._emitted_units = list("大家好")
    processor._emission_events = []

    output = processor._compute_incremental_output("大家坏")

    assert not output.new_tokens
    assert processor.emission_events()[-1]["reason"] == "candidate_not_append_prefix"
    assert processor.emission_events()[-1]["accepted"] is False
    assert processor.emission_events()[-1]["previous_emitted_text"] == "大家好"
    assert processor.emission_events()[-1]["candidate_translation"] == "大家坏"


def test_chunk_decision_summary_counts_blocked_current_mt_chunks():
    summary = summarize_chunk_decisions(
        [
            {
                "emitted": False,
                "emission_event": {"reason": "no_new_units"},
                "alignatt_metadata_current_chunk": True,
                "alignatt_decision": {
                    "stop_reason": "alignatt:source_frontier",
                    "accepted_token_count": 0,
                    "alignatt_source_context_under_min": True,
                    "alignatt_source_context_blocked": True,
                },
            },
            {
                "emitted": True,
                "emission_event": {"reason": "accepted"},
                "alignatt_metadata_current_chunk": True,
                "alignatt_decision": {
                    "stop_reason": "stop",
                    "accepted_token_count": 3,
                    "alignatt_source_context_under_min": True,
                    "alignatt_source_context_cap_applied": True,
                    "alignatt_source_frontier_bypassed_count": 2,
                    "alignatt_source_frontier_trimmed": True,
                },
            },
            {
                "emitted": False,
                "emission_event": {"reason": "empty_candidate"},
                "alignatt_metadata_current_chunk": False,
                "alignatt_decision": {
                    "stop_reason": "alignatt:source_frontier",
                    "accepted_token_count": 0,
                },
            },
        ]
    )

    assert summary["chunk_count"] == 3
    assert summary["emitted_chunk_count"] == 1
    assert summary["current_mt_decision_count"] == 2
    assert summary["zero_emit_current_mt_decision_count"] == 1
    assert summary["zero_accept_current_mt_decision_count"] == 1
    assert summary["source_context_under_min_count"] == 2
    assert summary["source_context_blocked_count"] == 1
    assert summary["source_context_cap_applied_count"] == 1
    assert summary["zero_emit_source_context_blocked_count"] == 1
    assert summary["source_frontier_bypassed_count"] == 2
    assert summary["source_frontier_trimmed_count"] == 1
    assert summary["stop_reason_counts"] == {
        "alignatt:source_frontier": 1,
        "stop": 1,
    }
    assert summary["emission_reason_counts"] == {
        "accepted": 1,
        "empty_candidate": 1,
        "no_new_units": 1,
    }


def test_scheduler_skip_snapshot_is_current_chunk_metadata():
    session = CascadeSession(LoadedModelBundle(CascadeRuntimeConfig()))
    source_frontier = SimpleNamespace(
        accessible_unit_count=2,
        units=[object(), object(), object()],
        current_source_ms=1280.0,
        current_audio_ms=1280.0,
        inaccessible_ms=320.0,
    )

    result = session.translation_units.snapshot_skipped_partial_result(
        source_frontier=source_frontier,
        scheduler_reason="source_prefix_unchanged",
    )

    assert result.alignatt_metadata["current_source_ms"] == 1280.0
    assert alignatt_metadata_is_current_chunk(
        result.alignatt_metadata,
        audio_processed_ms=1280.0,
    )


def test_chunk_record_counts_scheduler_skip_as_current_decision():
    session = CascadeSession(LoadedModelBundle(CascadeRuntimeConfig()))
    session.state.partial_translation.last_alignatt_metadata = {
        "scheduler_skipped": True,
        "scheduler_reason": "source_prefix_unchanged",
        "stop_reason": "scheduler:source_prefix_unchanged",
        "accepted_token_count": 0,
        "current_source_ms": 960.0,
    }
    processor = SimpleNamespace(
        session=session,
        _emitted_units=[],
        tokens_to_string=lambda tokens: "".join(tokens),
    )

    decision = chunk_decision_record(
        chunk_idx=0,
        input_name="clip.wav",
        audio_processed_ms=960.0,
        wallclock_elapsed_ms=1.0,
        is_eos=False,
        emitted_new_tokens=[],
        emission_event=None,
        processor=processor,
    )
    summary = summarize_chunk_decisions([decision])

    assert decision["alignatt_metadata_current_chunk"] is True
    assert summary["current_mt_decision_count"] == 1
    assert summary["zero_emit_current_mt_decision_count"] == 1
    assert summary["zero_accept_current_mt_decision_count"] == 1
    assert summary["scheduler_skip_current_decision_count"] == 1
    assert summary["stop_reason_counts"] == {"scheduler:source_prefix_unchanged": 1}


def test_compact_alignatt_metadata_keeps_decision_fields_only():
    compact = compact_alignatt_metadata(
        {
            "stop_reason": "alignatt:provenance_weak",
            "accepted_token_count": 2,
            "alignatt_source_context_cap_target_units": 3,
            "draft_target_stability_unit_end_token_indices": [1, 2, 3],
            "provenance_per_draft_token": [{"source_accessible": 0.1}],
        }
    )

    assert compact == {
        "stop_reason": "alignatt:provenance_weak",
        "accepted_token_count": 2,
        "alignatt_source_context_cap_target_units": 3,
        "draft_target_stability_unit_end_token_indices": [1, 2, 3],
    }
