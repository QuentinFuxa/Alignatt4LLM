#!/usr/bin/env python3
"""Batch evaluation runner for the SimulStream CascadeAlignAttProcessor.

Runs multiple media files through the processor in a single process, keeping
models hot across audios to avoid repeated 5-minute load costs.

Usage (from .venv-inference):
    # Sanity set (3 audios):
    alignatt-batch \\
        --inputs data/devset/audio/myfXyntFYL.wav data/devset/audio/DyXpuURBMP.wav data/devset/audio/ccpXHNfaoy.wav \\
        --output-dir outputs/simulstream_batch_ende_2s \\
        --chunk-ms 850 --target de

    # Full set (all supported media files in directory):
    alignatt-batch \\
        --input-dir data/devset/audio/ \\
        --output-dir outputs/simulstream_fullset_ende_2s \\
        --chunk-ms 850 --target de
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np

from alignatt4llm.audio import discover_input_media_paths, load_audio_mono_16khz
from alignatt4llm.simulstream_processor import (
    CascadeAlignAttProcessor,
    LANGUAGE_CODE_TO_NAME,
    LANGUAGE_NAME_TO_CODE,
)
from alignatt4llm.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
    HYPOTHESIS_FILENAME,
    MANIFEST_FILENAME,
    STREAM_UPDATE_ELAPSED_SEMANTICS_WALLCLOCK,
    STREAM_UPDATES_FILENAME,
    ensure_output_dir,
    normalize_computation_aware_timestamps,
    utc_now_isoformat,
    write_json,
    write_jsonl,
)
from alignatt4llm.text_surface import prediction_text_from_target_surface
from alignatt4llm.emission import register_translation_timestamps, register_translation_words
from alignatt4llm.runtime import VALID_MT_BACKEND_NAMES
from simulstream.server.speech_processors import SAMPLE_RATE

EMISSION_EVENTS_FILENAME = "emission_events.jsonl"
CHUNK_DECISIONS_FILENAME = "chunk_decisions.jsonl"

ALIGNATT_DECISION_METADATA_KEYS = (
    "stop_reason",
    "draft_mean_consensus_ratio",
    "draft_mean_entropy_norm",
    "alignatt_min_alignment_confidence",
    "alignatt_unit_policy_last_unit_confidence",
    "scheduler_skipped",
    "scheduler_reason",
    "source_token_count",
    "source_unit_count",
    "accessible_source_unit_count",
    "accessible_source_local_end_exclusive",
    "accepted_candidate_token_count",
    "accepted_token_count",
    "accepted_target_stability_unit_count",
    "draft_operating_token_count",
    "draft_provenance_token_count",
    "draft_target_stability_unit_count",
    "draft_target_stability_unit_end_token_indices",
    "unsafe_reason",
    "unsafe_target_token_index",
    "blocked_source_local_position",
    "blocked_source_unit_index",
    "alignatt_acceptance_variant",
    "alignatt_online_normalization",
    "alignatt_frontier_min_inaccessible_mass",
    "alignatt_source_frontier_action",
    "alignatt_source_frontier_candidate_seen",
    "alignatt_source_frontier_candidate_position",
    "alignatt_source_frontier_bypassed_count",
    "alignatt_source_frontier_trimmed",
    "alignatt_source_frontier_trimmed_token_count",
    "alignatt_source_frontier_trimmed_unit_count",
    "alignatt_source_frontier_trim_reason",
    "alignatt_source_frontier_trim_unit_position",
    "alignatt_source_frontier_trim_bypassed_count",
    "alignatt_source_frontier_trimmed_before_regression",
    "alignatt_max_inaccessible_source_mass",
    "alignatt_max_non_source_prompt_mass",
    "alignatt_min_accessible_inaccessible_margin",
    "alignatt_min_source_mass",
    "alignatt_min_accessible_source_units",
    "alignatt_min_accessible_source_units_mode",
    "alignatt_source_context_under_min",
    "alignatt_source_context_blocked",
    "alignatt_source_context_cap_applied",
    "alignatt_source_context_cap_target_units",
    "alignatt_token_argmax_frontier_gate",
    "alignatt_token_argmax_source_mass",
    "alignatt_token_argmax_frontier_patience_streak",
    "alignatt_token_argmax_frontier_patience_bypassed_count",
    "alignatt_max_source_regression",
    "alignatt_source_regression_recent_tokens",
    "alignatt_source_regression_activation_mode",
    "alignatt_source_regression_activation_slack_tokens",
    "alignatt_source_regression_patience_tokens",
    "alignatt_source_regression_action",
    "alignatt_source_regression_patience_streak",
    "alignatt_source_regression_patience_bypassed_count",
    "alignatt_source_regression_candidate_seen",
    "alignatt_source_regression_trimmed",
    "alignatt_source_regression_trimmed_token_count",
    "alignatt_source_regression_trimmed_unit_count",
    "alignatt_source_regression_trim_reason",
    "alignatt_source_regression_trim_reference_position",
    "alignatt_source_regression_trim_unit_position",
    "alignatt_source_regression_trim_bypassed_count",
    "alignatt_source_lcp_stability_enabled",
    "source_lcp_append_slack_units",
    "alignatt_unit_policy_stop_reason",
    "alignatt_unit_policy_accepted_unit_count",
    "alignatt_unit_policy_complete_unit_count",
    "alignatt_source_lookback_trimmed",
    "alignatt_hold_back_trimmed",
    "alignatt_terminal_punctuation_trimmed",
    "alignatt_min_emit_blocked",
    "accepted_prefix_mean_source_accessible_mass",
    "accepted_prefix_mean_source_inaccessible_mass",
    "draft_mean_source_accessible_mass",
    "draft_mean_source_inaccessible_mass",
    "blocked_token_source_accessible_mass",
    "blocked_token_source_inaccessible_mass",
    "current_source_ms",
    "inaccessible_ms",
    "probe_mode",
    "probe_backend",
)


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def canonical_language_code(language: str) -> str:
    """Accept either a display name or a language code."""

    return LANGUAGE_NAME_TO_CODE.get(language, language)


def compact_alignatt_metadata(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(metadata, dict):
        return None
    return {
        key: metadata.get(key)
        for key in ALIGNATT_DECISION_METADATA_KEYS
        if key in metadata
    }


def alignatt_metadata_is_current_chunk(
    metadata: dict[str, Any] | None,
    *,
    audio_processed_ms: float,
) -> bool:
    if not isinstance(metadata, dict):
        return False
    current_source_ms = metadata.get("current_source_ms")
    if current_source_ms is None:
        return False
    try:
        return abs(float(current_source_ms) - float(audio_processed_ms)) < 1e-3
    except (TypeError, ValueError):
        return False


def chunk_decision_record(
    *,
    chunk_idx: int,
    input_name: str,
    audio_processed_ms: float,
    wallclock_elapsed_ms: float,
    is_eos: bool,
    emitted_new_tokens: list[str],
    emission_event: dict[str, Any] | None,
    processor: CascadeAlignAttProcessor,
) -> dict[str, Any]:
    partial = processor.session.state.partial_translation
    metadata = partial.last_alignatt_metadata
    compact_metadata = compact_alignatt_metadata(metadata)
    current_metadata = alignatt_metadata_is_current_chunk(
        metadata,
        audio_processed_ms=audio_processed_ms,
    )
    return {
        "chunk_idx": int(chunk_idx),
        "input_name": input_name,
        "wav_name": input_name,
        "audio_processed_ms": float(audio_processed_ms),
        "wallclock_elapsed_ms": float(wallclock_elapsed_ms),
        "is_eos": bool(is_eos),
        "emitted": bool(emitted_new_tokens),
        "emitted_new_tokens": emitted_new_tokens,
        "emission_event": emission_event,
        "asr_text": processor.session.render_public_asr_text(),
        "translation_text": processor.tokens_to_string(processor._emitted_units),
        "partial_source_prefix": partial.source_prefix,
        "partial_accepted_target": partial.accepted_target,
        "partial_draft_target": partial.draft_target,
        "partial_accepted_token_count": len(partial.accepted_token_ids),
        "partial_source_accessible_unit_count": partial.source_accessible_unit_count,
        "partial_source_total_unit_count": partial.source_total_unit_count,
        "alignatt_metadata_current_chunk": current_metadata,
        "alignatt_decision": compact_metadata,
    }


def summarize_chunk_decisions(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    stop_reason_counts: dict[str, int] = {}
    emission_reason_counts: dict[str, int] = {}
    current_mt_decision_count = 0
    zero_emit_current_mt_decision_count = 0
    zero_accept_current_mt_decision_count = 0
    scheduler_skip_current_decision_count = 0
    source_context_under_min_count = 0
    source_context_blocked_count = 0
    source_context_cap_applied_count = 0
    zero_emit_source_context_blocked_count = 0
    source_frontier_bypassed_count = 0
    source_frontier_trimmed_count = 0
    for decision in decisions:
        event = decision.get("emission_event") or {}
        emission_reason = str(event.get("reason") or "missing")
        emission_reason_counts[emission_reason] = emission_reason_counts.get(emission_reason, 0) + 1
        if not decision.get("alignatt_metadata_current_chunk"):
            continue
        current_mt_decision_count += 1
        if not decision.get("emitted"):
            zero_emit_current_mt_decision_count += 1
        alignatt_decision = decision.get("alignatt_decision") or {}
        stop_reason = str(
            alignatt_decision.get("stop_reason")
            or alignatt_decision.get("unsafe_reason")
            or "missing"
        )
        stop_reason_counts[stop_reason] = stop_reason_counts.get(stop_reason, 0) + 1
        if alignatt_decision.get("scheduler_skipped"):
            scheduler_skip_current_decision_count += 1
        if alignatt_decision.get("alignatt_source_context_under_min"):
            source_context_under_min_count += 1
        if alignatt_decision.get("alignatt_source_context_blocked"):
            source_context_blocked_count += 1
            if not decision.get("emitted"):
                zero_emit_source_context_blocked_count += 1
        if alignatt_decision.get("alignatt_source_context_cap_applied"):
            source_context_cap_applied_count += 1
        try:
            source_frontier_bypassed_count += int(
                alignatt_decision.get("alignatt_source_frontier_bypassed_count") or 0
            )
        except (TypeError, ValueError):
            pass
        if alignatt_decision.get("alignatt_source_frontier_trimmed"):
            source_frontier_trimmed_count += 1
        accepted = alignatt_decision.get("accepted_token_count")
        try:
            if int(accepted) == 0:
                zero_accept_current_mt_decision_count += 1
        except (TypeError, ValueError):
            pass
    return {
        "chunk_count": len(decisions),
        "emitted_chunk_count": sum(1 for decision in decisions if decision.get("emitted")),
        "current_mt_decision_count": current_mt_decision_count,
        "zero_emit_current_mt_decision_count": zero_emit_current_mt_decision_count,
        "zero_accept_current_mt_decision_count": zero_accept_current_mt_decision_count,
        "scheduler_skip_current_decision_count": scheduler_skip_current_decision_count,
        "source_context_under_min_count": source_context_under_min_count,
        "source_context_blocked_count": source_context_blocked_count,
        "source_context_cap_applied_count": source_context_cap_applied_count,
        "zero_emit_source_context_blocked_count": zero_emit_source_context_blocked_count,
        "source_frontier_bypassed_count": source_frontier_bypassed_count,
        "source_frontier_trimmed_count": source_frontier_trimmed_count,
        "stop_reason_counts": dict(sorted(stop_reason_counts.items())),
        "emission_reason_counts": dict(sorted(emission_reason_counts.items())),
    }


def run_single_audio(
    processor: CascadeAlignAttProcessor,
    input_path: str,
    chunk_ms: int,
    target_lang_code: str,
) -> dict[str, Any]:
    """Run one audio through the processor and return all artifacts data."""
    processor.clear()
    audio = load_audio_mono_16khz(input_path)
    chunk_size = int(SAMPLE_RATE * chunk_ms / 1000)
    audio_duration_ms = len(audio) * 1000.0 / SAMPLE_RATE
    input_name = Path(input_path).name

    word_delays_ms: list[float] = []
    word_elapsed_ms: list[float] = []
    stream_updates: list[dict[str, Any]] = []
    emission_events: list[dict[str, Any]] = []
    chunk_decisions: list[dict[str, Any]] = []
    emission_event_cursor = 0
    last_translation = ""
    last_raw_translation = ""
    start_time = perf_counter()

    for chunk_idx, start_sample in enumerate(range(0, len(audio), chunk_size)):
        chunk = audio[start_sample : start_sample + chunk_size].astype(np.float32)
        output = processor.process_chunk(chunk)
        current_translation = processor.tokens_to_string(processor._emitted_units)
        audio_processed_ms = min((start_sample + chunk_size), len(audio)) * 1000.0 / SAMPLE_RATE
        wallclock_elapsed_ms = (perf_counter() - start_time) * 1000.0
        new_emission_events = processor.emission_events()[emission_event_cursor:]
        emission_event_cursor += len(new_emission_events)
        for event in new_emission_events:
            payload = dict(event)
            payload.update({
                "input_name": input_name,
                "wav_name": input_name,
                "audio_processed_ms": audio_processed_ms,
                "wallclock_elapsed_ms": wallclock_elapsed_ms,
                "is_eos": False,
                "asr_text": processor.session.render_public_asr_text(),
            })
            emission_events.append(payload)
        latest_emission_event = (
            dict(new_emission_events[-1]) if new_emission_events else None
        )
        chunk_decisions.append(
            chunk_decision_record(
                chunk_idx=chunk_idx,
                input_name=input_name,
                audio_processed_ms=audio_processed_ms,
                wallclock_elapsed_ms=wallclock_elapsed_ms,
                is_eos=False,
                emitted_new_tokens=list(output.new_tokens),
                emission_event=latest_emission_event,
                processor=processor,
            )
        )

        if output.new_tokens:
            register_translation_timestamps(
                last_raw_translation, current_translation,
                wallclock_elapsed_ms, word_elapsed_ms,
                target_lang_code=target_lang_code,
            )
            new_words = register_translation_words(
                last_translation, current_translation,
                audio_processed_ms, word_delays_ms,
                target_lang_code=target_lang_code,
            )
            partial = processor.session.state.partial_translation
            stream_updates.append({
                "update_idx": len(stream_updates),
                "input_name": input_name,
                "wav_name": input_name,
                "audio_processed_ms": audio_processed_ms,
                "wallclock_elapsed_ms": wallclock_elapsed_ms,
                "translation_text": current_translation,
                "new_words": new_words,
                # Observer / MT-state fields for offline replay
                # (continuous-confidence branch, emit-policy replay, etc.).
                # Optional — absent on older artifacts; consumers must tolerate.
                "asr_text": processor.session.render_public_asr_text(),
                "partial_accepted_target": partial.accepted_target,
                "partial_draft_target": partial.draft_target,
                "alignatt_metadata": partial.last_alignatt_metadata,
                "translation_prompt_num_tokens": partial.last_prompt_num_tokens,
                "translation_prompt_num_cached_tokens": partial.last_num_cached_tokens,
            })
            last_translation = current_translation
            last_raw_translation = current_translation

    eos_output = processor.end_of_stream()
    final_translation = processor.tokens_to_string(processor._emitted_units)
    final_elapsed_ms = (perf_counter() - start_time) * 1000.0
    new_emission_events = processor.emission_events()[emission_event_cursor:]
    emission_event_cursor += len(new_emission_events)
    for event in new_emission_events:
        payload = dict(event)
        payload.update({
            "input_name": input_name,
            "wav_name": input_name,
            "audio_processed_ms": audio_duration_ms,
            "wallclock_elapsed_ms": final_elapsed_ms,
            "is_eos": True,
            "asr_text": processor.session.render_public_asr_text(),
        })
        emission_events.append(payload)
    latest_eos_emission_event = (
        dict(new_emission_events[-1]) if new_emission_events else None
    )
    chunk_decisions.append(
        chunk_decision_record(
            chunk_idx=len(chunk_decisions),
            input_name=input_name,
            audio_processed_ms=audio_duration_ms,
            wallclock_elapsed_ms=final_elapsed_ms,
            is_eos=True,
            emitted_new_tokens=list(eos_output.new_tokens),
            emission_event=latest_eos_emission_event,
            processor=processor,
        )
    )

    if eos_output.new_tokens or final_translation != last_translation:
        register_translation_timestamps(
            last_raw_translation, final_translation,
            final_elapsed_ms, word_elapsed_ms,
            target_lang_code=target_lang_code,
        )
        eos_new_words = register_translation_words(
            last_translation, final_translation,
            audio_duration_ms, word_delays_ms,
            target_lang_code=target_lang_code,
        )
        partial = processor.session.state.partial_translation
        stream_updates.append({
            "update_idx": len(stream_updates),
            "input_name": input_name,
            "wav_name": input_name,
            "audio_processed_ms": audio_duration_ms,
            "wallclock_elapsed_ms": final_elapsed_ms,
            "translation_text": final_translation,
            "new_words": eos_new_words,
            "is_eos": True,
            "asr_text": processor.session.render_public_asr_text(),
            "partial_accepted_target": partial.accepted_target,
            "partial_draft_target": partial.draft_target,
            "alignatt_metadata": partial.last_alignatt_metadata,
            "translation_prompt_num_tokens": partial.last_prompt_num_tokens,
            "translation_prompt_num_cached_tokens": partial.last_num_cached_tokens,
        })

    final_asr = processor.session.render_public_asr_text()
    total_wallclock_s = perf_counter() - start_time
    rtf = total_wallclock_s / (audio_duration_ms / 1000.0) if audio_duration_ms > 0 else 0.0

    normalized_elapsed_ms = normalize_computation_aware_timestamps(word_delays_ms, word_elapsed_ms)
    prediction = prediction_text_from_target_surface(
        final_translation,
        target_lang_code=target_lang_code,
    )

    return {
        "input_path": input_path,
        "input_name": input_name,
        "wav_path": input_path,
        "wav_name": input_name,
        "audio_duration_ms": audio_duration_ms,
        "total_wallclock_s": total_wallclock_s,
        "rtf": rtf,
        "final_asr": final_asr,
        "final_translation": final_translation,
        "num_updates": len(stream_updates),
        "num_chunks": len(chunk_decisions),
        "chunk_decision_summary": summarize_chunk_decisions(chunk_decisions),
        "hypothesis_record": {
            "source": [input_name],
            "source_length": audio_duration_ms,
            "prediction": prediction,
            "delays": word_delays_ms,
            "elapsed": normalized_elapsed_ms,
            "elapsed_wallclock_ms": word_elapsed_ms,
            "elapsed_semantics": HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
        },
        "stream_updates": stream_updates,
        "emission_events": emission_events,
        "chunk_decisions": chunk_decisions,
    }


def resolve_input_paths(
    *,
    inputs: list[str] | None,
    input_dir: str | None,
) -> list[str]:
    if input_dir is not None:
        discovered = discover_input_media_paths(input_dir)
        filtered = [
            path for path in discovered
            if not Path(path).name.endswith("_short60s.wav")
        ]
        return filtered or discovered
    if not inputs:
        raise ValueError("Either explicit inputs or an input directory must be provided.")
    return [str(Path(path)) for path in inputs]


def resolve_paper_context_path_for_input(
    input_path: str,
    *,
    explicit_paper_context_path: str | None = None,
    paper_context_dir: str | None = None,
) -> str | None:
    if explicit_paper_context_path is not None and paper_context_dir is not None:
        raise ValueError("paper_context_path and paper_context_dir are mutually exclusive.")
    if explicit_paper_context_path is not None:
        return explicit_paper_context_path
    if paper_context_dir is None:
        return None
    candidate = Path(paper_context_dir) / f"{Path(input_path).stem}.json"
    if not candidate.exists():
        print(
            f"  [paper-context] no artifact for {Path(input_path).name} at {candidate}; "
            f"running without extra context for this input."
        )
        return None
    return str(candidate)


def run_batch_inference(
    *,
    processor_config: SimpleNamespace,
    input_paths: list[str],
    output_dir: str,
    source_lang_code: str,
    target_lang_code: str,
    explicit_paper_context_path: str | None = None,
    paper_context_dir: str | None = None,
    attention_trace_level: str | None = None,
) -> dict[str, Any]:
    print(f"Will process {len(input_paths)} media files for {source_lang_code}->{target_lang_code}")

    print("Loading models ...")
    load_start = perf_counter()
    CascadeAlignAttProcessor.load_model(processor_config)
    load_ms = (perf_counter() - load_start) * 1000.0
    print(f"Models loaded in {load_ms:.0f} ms")

    processor = CascadeAlignAttProcessor(processor_config)
    processor.set_source_language(source_lang_code)
    processor.set_target_language(target_lang_code)
    if attention_trace_level:
        from alignatt4llm.alignment.attention_trace import make_stderr_trace_printer

        processor.session.attention_trace_sink = make_stderr_trace_printer(
            attention_trace_level
        )

    all_hypothesis_records: list[dict[str, Any]] = []
    all_stream_updates: list[dict[str, Any]] = []
    all_emission_events: list[dict[str, Any]] = []
    all_chunk_decisions: list[dict[str, Any]] = []
    per_input_results: list[dict[str, Any]] = []
    batch_start = perf_counter()

    for idx, input_path in enumerate(input_paths):
        context_path = resolve_paper_context_path_for_input(
            input_path,
            explicit_paper_context_path=explicit_paper_context_path,
            paper_context_dir=paper_context_dir,
        )
        if hasattr(processor, "set_paper_context_path"):
            processor.set_paper_context_path(context_path)

        print(f"\n[{idx+1}/{len(input_paths)}] {Path(input_path).name} ...", flush=True)
        result = run_single_audio(
            processor,
            input_path,
            int(getattr(processor_config, "chunk_ms", 450)),
            target_lang_code,
        )
        all_hypothesis_records.append(result["hypothesis_record"])
        all_stream_updates.extend(result["stream_updates"])
        all_emission_events.extend(result["emission_events"])
        all_chunk_decisions.extend(result["chunk_decisions"])
        per_input_results.append({
            "input": result["input_name"],
            "audio_s": round(result["audio_duration_ms"] / 1000, 1),
            "rtf": round(result["rtf"], 4),
            "updates": result["num_updates"],
            "chunks": result["num_chunks"],
            "chunk_decision_summary": result["chunk_decision_summary"],
            "paper_context_path": context_path,
        })
        print(
            f"  RTF={result['rtf']:.3f}  updates={result['num_updates']}  "
            f"wallclock={result['total_wallclock_s']:.1f}s"
        )

    batch_wallclock_s = perf_counter() - batch_start
    total_audio_s = sum(entry["audio_s"] for entry in per_input_results)
    batch_rtf = batch_wallclock_s / total_audio_s if total_audio_s > 0 else 0.0

    output_path = ensure_output_dir(output_dir)
    runtime_config: dict[str, Any] = {
        "chunk_ms": getattr(processor_config, "chunk_ms"),
        "alignment_backend_name": getattr(processor_config, "alignment_backend_name"),
        "mt_backend_name": getattr(processor_config, "mt_backend_name"),
        "asr_gpu_memory_utilization": getattr(
            processor_config, "asr_gpu_memory_utilization", 0.2
        ),
        "min_start_seconds": getattr(processor_config, "min_start_seconds"),
        "max_history_utterances": getattr(processor_config, "max_history_utterances"),
        "partial_max_new_tokens": getattr(processor_config, "partial_max_new_tokens"),
        "translation_alignatt_min_source_mass": getattr(
            processor_config, "translation_alignatt_min_source_mass"
        ),
        "translation_alignatt_border_margin": getattr(
            processor_config, "translation_alignatt_border_margin", 0
        ),
        "translation_alignatt_acceptance_variant": getattr(
            processor_config, "translation_alignatt_acceptance_variant", "token"
        ),
        "translation_alignatt_online_normalization": getattr(
            processor_config, "translation_alignatt_online_normalization", "zscore"
        ),
        "translation_alignatt_inaccessible_ms": getattr(
            processor_config, "translation_alignatt_inaccessible_ms"
        ),
        "translation_alignatt_source_lcp_stability": getattr(
            processor_config, "translation_alignatt_source_lcp_stability", False
        ),
        "translation_alignatt_source_lcp_append_slack_units": getattr(
            processor_config,
            "translation_alignatt_source_lcp_append_slack_units",
            0,
        ),
        "translation_alignatt_argmax_mass_threshold": getattr(
            processor_config, "translation_alignatt_argmax_mass_threshold", 0.0
        ),
        "translation_alignatt_frontier_min_inaccessible_mass": getattr(
            processor_config, "translation_alignatt_frontier_min_inaccessible_mass", 0.0
        ),
        "translation_alignatt_source_frontier_action": getattr(
            processor_config,
            "translation_alignatt_source_frontier_action",
            "stop",
        ),
        "translation_alignatt_max_inaccessible_source_mass": getattr(
            processor_config, "translation_alignatt_max_inaccessible_source_mass", 1.0
        ),
        "translation_alignatt_max_non_source_prompt_mass": getattr(
            processor_config, "translation_alignatt_max_non_source_prompt_mass", 1.0
        ),
        "translation_alignatt_min_accessible_inaccessible_margin": getattr(
            processor_config,
            "translation_alignatt_min_accessible_inaccessible_margin",
            -1.0,
        ),
        "translation_alignatt_min_accepted_accessible_source_mass": getattr(
            processor_config, "translation_alignatt_min_accepted_accessible_source_mass", 0.0
        ),
        "translation_alignatt_min_accessible_source_units": getattr(
            processor_config, "translation_alignatt_min_accessible_source_units", 0
        ),
        "translation_alignatt_min_accessible_source_units_mode": getattr(
            processor_config,
            "translation_alignatt_min_accessible_source_units_mode",
            "block",
        ),
        "translation_alignatt_max_source_regression": getattr(
            processor_config, "translation_alignatt_max_source_regression", -1
        ),
        "translation_alignatt_source_regression_min_source_mass": getattr(
            processor_config,
            "translation_alignatt_source_regression_min_source_mass",
            0.0,
        ),
        "translation_alignatt_source_regression_min_inaccessible_mass": getattr(
            processor_config,
            "translation_alignatt_source_regression_min_inaccessible_mass",
            0.0,
        ),
        "translation_alignatt_source_regression_recent_tokens": getattr(
            processor_config,
            "translation_alignatt_source_regression_recent_tokens",
            0,
        ),
        "translation_alignatt_source_regression_reference_mode": getattr(
            processor_config,
            "translation_alignatt_source_regression_reference_mode",
            "max",
        ),
        "translation_alignatt_source_regression_activation_mode": getattr(
            processor_config,
            "translation_alignatt_source_regression_activation_mode",
            "always",
        ),
        "translation_alignatt_source_regression_activation_slack_tokens": getattr(
            processor_config,
            "translation_alignatt_source_regression_activation_slack_tokens",
            0,
        ),
        "translation_alignatt_source_regression_patience_tokens": getattr(
            processor_config,
            "translation_alignatt_source_regression_patience_tokens",
            1,
        ),
        "translation_alignatt_source_regression_action": getattr(
            processor_config,
            "translation_alignatt_source_regression_action",
            "stop",
        ),
        "translation_alignatt_token_argmax_frontier_gate": getattr(
            processor_config, "translation_alignatt_token_argmax_frontier_gate", False
        ),
        "translation_alignatt_token_argmax_min_source_mass": getattr(
            processor_config, "translation_alignatt_token_argmax_min_source_mass", 0.05
        ),
        "translation_alignatt_token_argmax_frontier_margin": getattr(
            processor_config, "translation_alignatt_token_argmax_frontier_margin", 0
        ),
        "translation_alignatt_token_argmax_frontier_patience_tokens": getattr(
            processor_config,
            "translation_alignatt_token_argmax_frontier_patience_tokens",
            1,
        ),
        "hypothesis_elapsed_semantics": HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
        "stream_update_elapsed_semantics": STREAM_UPDATE_ELAPSED_SEMANTICS_WALLCLOCK,
        "chunk_decisions_filename": CHUNK_DECISIONS_FILENAME,
    }
    for key in [
        "translation_alignatt_heads_path", "translation_alignatt_top_k_heads",
        "translation_alignatt_filter_width", "translation_alignatt_probe_mode",
        "translation_alignatt_acceptance_variant",
        "translation_alignatt_online_normalization",
        "translation_alignatt_source_lcp_stability",
        "translation_alignatt_source_lcp_append_slack_units",
        "translation_alignatt_source_frontier_action",
        "translation_alignatt_min_accepted_accessible_source_mass",
        "translation_alignatt_accepted_accessible_source_mass_recent_units",
        "translation_alignatt_min_accessible_source_units",
        "translation_alignatt_min_accessible_source_units_mode",
        "translation_alignatt_hold_back_target_units",
        "translation_alignatt_min_emit_target_units",
        "translation_alignatt_max_source_regression",
        "translation_alignatt_source_regression_min_source_mass",
        "translation_alignatt_source_regression_min_inaccessible_mass",
        "translation_alignatt_source_regression_recent_tokens",
        "translation_alignatt_source_regression_reference_mode",
        "translation_alignatt_source_regression_activation_mode",
        "translation_alignatt_source_regression_activation_slack_tokens",
        "translation_alignatt_source_regression_patience_tokens",
        "translation_alignatt_source_regression_action",
        "translation_alignatt_unit_consensus_min_head_ratio",
        "translation_alignatt_min_alignment_confidence",
        "translation_alignatt_source_bearing_min_source_mass",
        "translation_alignatt_source_bearing_hard_inaccessible_cap",
        "translation_alignatt_max_non_source_prompt_mass",
        "translation_alignatt_token_argmax_frontier_gate",
        "translation_alignatt_token_argmax_min_source_mass",
        "translation_alignatt_token_argmax_frontier_margin",
        "translation_alignatt_token_argmax_frontier_patience_tokens",
        "translation_alignatt_source_lookback_holdback",
        "translation_alignatt_source_lookback_units",
        "translation_alignatt_source_lookback_min_source_mass",
        "translation_alignatt_source_lookback_min_source_position",
        "translation_alignatt_defer_low_source_terminal_punctuation",
        "translation_alignatt_terminal_punctuation_min_source_mass",
        "translation_acceptance_policy", "translation_static_cutoff_units",
        "gemma_audio_alignment_heads_path",
        "translation_emit_policy", "translation_max_tail_rewrite_words",
        "temperature", "repetition_penalty",
        "asr_alignatt_frame_threshold",
        "asr_alignatt_rewind_threshold",
        "asr_punctuation_min_commit_words",
        "asr_context_committed_words",
        "paper_context_path", "paper_context_mode", "paper_context_top_k",
        "paper_context_max_chars", "paper_context_history_window_words",
        "asr_gpu_memory_utilization",
        "mt_vllm_enforce_eager", "mt_vllm_cudagraph_mode",
        "mt_vllm_enable_prefix_caching", "mt_vllm_gpu_memory_utilization",
        "mt_vllm_enable_speculative_decoding",
        "mt_vllm_speculative_assistant_model",
        "mt_vllm_num_speculative_tokens",
        "gemma_max_model_len",
        "mt_max_model_len",
    ]:
        runtime_config[key] = getattr(processor.session.config, key, None)

    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "generated_at_utc": utc_now_isoformat(),
        "kind": "inference_batch",
        "num_inputs": len(input_paths),
        "input_paths": input_paths,
        # Legacy aliases preserved for existing tooling.
        "num_audios": len(input_paths),
        "wav_paths": input_paths,
        "source_language": LANGUAGE_CODE_TO_NAME.get(source_lang_code, source_lang_code),
        "target_language": LANGUAGE_CODE_TO_NAME.get(target_lang_code, target_lang_code),
        "source_language_code": source_lang_code,
        "target_language_code": target_lang_code,
        "runtime_config": runtime_config,
        "run_provenance": {
            "git_sha": git_sha(),
            "framework_mode": "simulstream_processor",
            "script": "alignatt-batch",
        },
        "speed": {
            "batch_wallclock_s": round(batch_wallclock_s, 2),
            "batch_rtf": round(batch_rtf, 4),
            "total_audio_s": round(total_audio_s, 1),
            "per_input": per_input_results,
            "per_audio": per_input_results,
        },
    }

    write_json(output_path / MANIFEST_FILENAME, manifest)
    write_jsonl(output_path / HYPOTHESIS_FILENAME, all_hypothesis_records)
    write_jsonl(output_path / STREAM_UPDATES_FILENAME, all_stream_updates)
    write_jsonl(output_path / EMISSION_EVENTS_FILENAME, all_emission_events)
    write_jsonl(output_path / CHUNK_DECISIONS_FILENAME, all_chunk_decisions)

    print(f"\n{'='*60}")
    print(f"Batch complete: {len(input_paths)} inputs, {total_audio_s:.0f}s total audio")
    print(f"Batch wallclock: {batch_wallclock_s:.1f}s  RTF: {batch_rtf:.4f}")
    print(f"Artifacts: {output_dir}")
    print(f"Evaluate: alignatt-eval --output-dir {output_dir} --skip-comet")
    print(f"{'='*60}")

    return {
        "manifest": manifest,
        "hypothesis_records": all_hypothesis_records,
        "stream_updates": all_stream_updates,
        "emission_events": all_emission_events,
        "chunk_decisions": all_chunk_decisions,
        "output_dir": output_dir,
        "model_load_ms": load_ms,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch SimulStream evaluation runner.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--inputs",
        "--wavs",
        nargs="+",
        dest="inputs",
        help="List of input media paths (.wav, .mp4, ...).",
    )
    group.add_argument(
        "--input-dir",
        "--wav-dir",
        dest="input_dir",
        help="Directory of input media files (all supported files used).",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-ms", default=850, type=int)
    parser.add_argument("--source", default="en")
    parser.add_argument("--target", default="de")
    parser.add_argument(
        "--alignment-backend-name",
        default="qwen_forced",
        choices=("qwen_forced", "gemma_vllm_qk_fast"),
    )
    parser.add_argument(
        "--mt-backend-name",
        default="gemma_vllm_alignatt",
        choices=VALID_MT_BACKEND_NAMES,
        help=(
            "MT backend route. Default is the stable Gemma baseline; "
            "milmmt_vllm_alignatt is the active MiLMMT improvement route."
        ),
    )
    parser.add_argument(
        "--trace-attention",
        action="store_true",
        help=(
            "Print, on stderr, a live per-token MT attention trace as the model "
            "drafts: which source token each draft token attends to (src@N) and "
            "the accessible/inaccessible mass that drives the cut. Artifacts on "
            "stdout/files are unchanged."
        ),
    )
    parser.add_argument(
        "--trace-attention-level",
        choices=("commits", "all"),
        default="all",
        help=(
            "`all` (default) traces committed and held draft tokens; `commits` "
            "traces only committed tokens. Ignored without --trace-attention."
        ),
    )
    parser.add_argument(
        "--asr-alignatt-frame-threshold",
        default=4,
        type=int,
        help=(
            "AlignAtt token-level frontier gate in audio frames (simul_whisper "
            "§4). Lower = more aggressive commit, higher = safer."
        ),
    )
    parser.add_argument(
        "--asr-alignatt-rewind-threshold",
        default=200,
        type=int,
        help=(
            "Attention-collapse guard: abort the chunk if a generated token's "
            "argmax rewinds more than this many frames before the running "
            "reference."
        ),
    )
    parser.add_argument("--min-start-seconds", default=2.0, type=float)
    parser.add_argument("--max-history-utterances", default=0, type=int)
    parser.add_argument("--partial-max-new-tokens", default=16, type=int)
    parser.add_argument("--translation-alignatt-min-source-mass", default=0.0, type=float)
    parser.add_argument(
        "--translation-alignatt-top-k-heads",
        default=4,
        type=int,
        help="Number of retained MT AlignAtt heads to use.",
    )
    parser.add_argument(
        "--translation-alignatt-filter-width",
        default=7,
        type=int,
        help="Source-axis median filter width for MT AlignAtt rows.",
    )
    parser.add_argument(
        "--translation-alignatt-border-margin",
        default=1,
        type=int,
        help=(
            "Source-token safety margin around the accessible frontier. "
            "Negative values are more conservative; 0 keeps the strict AlignAtt frontier; "
            "positive values allow speculative look-ahead beyond the border."
        ),
    )
    parser.add_argument(
        "--translation-alignatt-acceptance-variant",
        default="token",
        choices=("token", "unit_mass", "unit_mass_source_bearing", "unit_argmax", "unit_consensus", "unit_conf"),
    )
    parser.add_argument(
        "--translation-alignatt-online-normalization",
        default="zscore",
        choices=("zscore", "raw"),
    )
    parser.add_argument("--translation-alignatt-inaccessible-ms", default=0.0, type=float)
    parser.add_argument(
        "--translation-alignatt-source-lcp-stability",
        action="store_true",
        help=(
            "Cap the AlignAtt-accessible source frontier to the ASR live-tail "
            "prefix that is stable across consecutive hypotheses. The MT still "
            "sees the full live source text."
        ),
    )
    parser.add_argument(
        "--translation-alignatt-source-lcp-append-slack-units",
        default=0,
        type=int,
        help=(
            "When source LCP stability is enabled, allow this many newly "
            "appended ASR source units beyond the strict LCP cap. Default 0 "
            "preserves strict source-LCP behavior."
        ),
    )
    parser.add_argument(
        "--translation-alignatt-frontier-min-inaccessible-mass",
        default=0.03,
        type=float,
        help=(
            "Soft-frontier gate. A value of 0.0 blocks any source argmax "
            "beyond the accessible frontier. The script default 0.03 allows "
            "that argmax only when the total inaccessible-source mass is below "
            "the threshold."
        ),
    )
    parser.add_argument(
        "--translation-alignatt-source-frontier-action",
        default="stop",
        choices=("stop", "trim_unrecovered"),
        help=(
            "How to handle source-frontier hits. 'stop' preserves the "
            "historical token-level hard stop; 'trim_unrecovered' lets the "
            "draft continue and trims only a target-unit suffix whose "
            "source-frontier violation never recovers inside the draft."
        ),
    )
    parser.add_argument(
        "--translation-alignatt-max-inaccessible-source-mass",
        default=1.0,
        type=float,
        help=(
            "Optional provenance gate: stop when attention mass on inaccessible "
            "source exceeds this value. Default 1.0 disables this gate for "
            "permissive AlignAtt; pass a lower value only for guarded "
            "diagnostics."
        ),
    )
    parser.add_argument(
        "--translation-alignatt-max-non-source-prompt-mass",
        default=1.0,
        type=float,
        help=(
            "Optional provenance diagnostic gate: stop when reconstructed "
            "attention mass on non-source prompt tokens exceeds this value. "
            "Default 1.0 disables the gate."
        ),
    )
    parser.add_argument(
        "--translation-alignatt-min-accessible-inaccessible-margin",
        default=-1.0,
        type=float,
        help=(
            "Optional provenance gate: require accessible-source mass minus "
            "inaccessible-source mass to exceed this margin. Default -1.0 "
            "disables the gate."
        ),
    )
    parser.add_argument(
        "--translation-alignatt-min-accepted-accessible-source-mass",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--translation-alignatt-accepted-accessible-source-mass-recent-units",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--translation-alignatt-min-accessible-source-units",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--translation-alignatt-min-accessible-source-units-mode",
        default="block",
        choices=("block", "target_unit_cap"),
        help=(
            "How to handle partial MT before the minimum accessible source-unit "
            "context is reached. 'block' preserves the historical hard gate; "
            "'target_unit_cap' keeps a small AlignAtt-accepted target prefix "
            "capped by the number of accessible source units."
        ),
    )
    parser.add_argument(
        "--translation-alignatt-hold-back-target-units",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--translation-alignatt-min-emit-target-units",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--translation-alignatt-max-source-regression",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "--translation-alignatt-source-regression-min-source-mass",
        default=0.0,
        type=float,
        help=(
            "Ignore source-regression stops for tokens whose total source "
            "provenance mass is below this threshold. Default 0.0 preserves "
            "the historical regression gate."
        ),
    )
    parser.add_argument(
        "--translation-alignatt-source-regression-min-inaccessible-mass",
        default=0.0,
        type=float,
        help=(
            "Ignore source-regression stops unless the regressing token also "
            "carries at least this much inaccessible-source mass. Default 0.0 "
            "preserves the historical monotonicity gate; positive values make "
            "the guard future-bearing instead of blocking reordering over "
            "already accessible source."
        ),
    )
    parser.add_argument(
        "--translation-alignatt-source-regression-recent-tokens",
        default=0,
        type=int,
        help=(
            "Use the maximum source position among the last N accepted target "
            "tokens as the source-regression reference. Default 0 preserves "
            "the historical global maximum reference."
        ),
    )
    parser.add_argument(
        "--translation-alignatt-source-regression-reference-mode",
        default="max",
        choices=("max", "median_recent"),
        help=(
            "Reference used by the source-regression gate when recent-token "
            "mode is enabled. 'max' preserves the historical conservative "
            "behavior; 'median_recent' is robust to one noisy forward argmax."
        ),
    )
    parser.add_argument(
        "--translation-alignatt-source-regression-activation-mode",
        default="always",
        choices=("always", "frontier_reached"),
        help=(
            "When to apply the source-regression gate. 'always' preserves the "
            "historical token-level monotonicity guard; 'frontier_reached' "
            "uses regression only after accepted source attention has reached "
            "the accessible source frontier."
        ),
    )
    parser.add_argument(
        "--translation-alignatt-source-regression-activation-slack-tokens",
        default=0,
        type=int,
        help=(
            "Extra source-token slack for frontier_reached activation. Larger "
            "values start the source-regression gate when accepted AlignAtt "
            "coverage is near the accessible source frontier instead of only "
            "at the exact frontier."
        ),
    )
    parser.add_argument(
        "--translation-alignatt-source-regression-patience-tokens",
        default=1,
        type=int,
        help=(
            "Consecutive source-regression tokens required before stopping. "
            "Default 1 preserves the historical token-level gate."
        ),
    )
    parser.add_argument(
        "--translation-alignatt-source-regression-action",
        default="stop",
        choices=("stop", "trim_target_unit", "trim_unrecovered"),
        help=(
            "How source-regression evidence is applied. 'stop' preserves the "
            "historical token-level hard stop; 'trim_target_unit' lets the "
            "draft continue and accepts the longest target-unit prefix whose "
            "AlignAtt source progression remains monotone; 'trim_unrecovered' "
            "keeps bounded local regressions that recover later in the same "
            "draft and trims only an unrecovered regressive suffix."
        ),
    )
    parser.add_argument(
        "--translation-alignatt-unit-consensus-min-head-ratio",
        default=0.60,
        type=float,
    )
    parser.add_argument("--asr-context-committed-words", default=0, type=int)
    parser.add_argument(
        "--translation-alignatt-min-alignment-confidence",
        default=0.0,
        type=float,
        help=(
            "Alignment-confidence floor for the unit_conf acceptance variant "
            "(fraction of heads agreeing with the consensus argmax). 0.0 "
            "disables the gate."
        ),
    )
    parser.add_argument(
        "--translation-alignatt-source-bearing-min-source-mass",
        default=0.005,
        type=float,
    )
    parser.add_argument(
        "--translation-alignatt-source-bearing-hard-inaccessible-cap",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--translation-alignatt-token-argmax-frontier-gate",
        action="store_true",
    )
    parser.add_argument(
        "--translation-alignatt-token-argmax-min-source-mass",
        default=0.05,
        type=float,
    )
    parser.add_argument(
        "--translation-alignatt-token-argmax-frontier-margin",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--translation-alignatt-token-argmax-frontier-patience-tokens",
        default=1,
        type=int,
        help=(
            "Consecutive token-argmax frontier hits required before stopping. "
            "A value of 2 ignores isolated frontier spikes without advancing "
            "the source-regression reference."
        ),
    )
    parser.add_argument(
        "--translation-alignatt-source-lookback-holdback",
        action="store_true",
    )
    parser.add_argument(
        "--translation-alignatt-source-lookback-units",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--translation-alignatt-source-lookback-min-source-mass",
        default=0.05,
        type=float,
    )
    parser.add_argument(
        "--translation-alignatt-source-lookback-min-source-position",
        default=3,
        type=int,
    )
    parser.add_argument(
        "--translation-alignatt-defer-low-source-terminal-punctuation",
        action="store_true",
    )
    parser.add_argument(
        "--translation-alignatt-terminal-punctuation-min-source-mass",
        default=0.06,
        type=float,
    )
    parser.add_argument(
        "--translation-acceptance-policy",
        default="alignatt",
        choices=("alignatt", "cut_last_target_units", "cut_last_x"),
        help=(
            "Partial MT acceptance rule. The default AlignAtt path uses the "
            "source frontier; cut_last_target_units accepts the same Gemma draft "
            "after removing a fixed target-side suffix."
        ),
    )
    parser.add_argument(
        "--translation-static-cutoff-units",
        default=0,
        type=int,
        help="Target stability units to drop when --translation-acceptance-policy=cut_last_target_units.",
    )
    parser.add_argument(
        "--translation-alignatt-argmax-mass-threshold",
        default=0.0,
        type=float,
        help=(
            "Confidence-gated acceptance threshold on the reconstructed softmax "
            "mass at the argmax source position (per-head averaged). Default "
            "0.0 disables the gate and preserves argmax-only AlignAtt; raising "
            "it stops acceptance with reason 'alignatt:argmax_mass_weak' when "
            "the attention at the aligned source token is too diffuse."
        ),
    )
    parser.add_argument(
        "--mt-vllm-enable-prefix-caching",
        action="store_true",
        help=(
            "Enable vLLM prefix caching for the MT backend. Caches the stable "
            "prompt prefix (system + task instructions) across partial MT "
            "calls within an utterance. Source tokens live after the stable "
            "prefix, so the observer still captures their K on every prefill."
        ),
    )
    parser.add_argument(
        "--mt-vllm-enable-speculative-decoding",
        action="store_true",
        help=(
            "Enable vLLM MTP speculative decoding for the MT engine. Off by "
            "default; for Gemma-4 E4B the backend defaults to the official "
            "google/gemma-4-E4B-it-assistant model unless an assistant path is supplied."
        ),
    )
    parser.add_argument(
        "--mt-vllm-enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Run the MT vLLM engine eagerly (runtime default: enabled). "
            "Eager execution is required for trustworthy AlignAtt observer "
            "capture: CUDA graph replay NaN-corrupts the captured q/k payload "
            "on the vLLM 0.22.1rc stack and the artifact index quarantines "
            "such runs. Pass --no-mt-vllm-enforce-eager to re-enable CUDA "
            "graph execution (corrupts observer capture; debugging only)."
        ),
    )
    parser.add_argument(
        "--mt-vllm-cudagraph-mode",
        default=None,
        choices=("full", "piecewise", "full_and_piecewise"),
        help=(
            "Override the MT vLLM CUDA graph mode (runtime default: full). "
            "Only takes effect with --no-mt-vllm-enforce-eager; the default "
            "eager engine builds no CUDA graphs. No mode is observer-safe on "
            "the current stack: full and piecewise both corrupt the captured "
            "q/k payload (the capture op sits inside the compiled pieces)."
        ),
    )
    parser.add_argument(
        "--mt-vllm-speculative-assistant-model",
        default=None,
        help=(
            "Assistant model id/path for MT speculative decoding. If omitted "
            "on Gemma-4 E4B, the backend searches CASCADE_GEMMA_ASSISTANT_SNAPSHOT "
            "and the local HF cache before falling back to the official model id."
        ),
    )
    parser.add_argument(
        "--mt-vllm-num-speculative-tokens",
        "--mt-vllm-speculative-num-tokens",
        dest="mt_vllm_num_speculative_tokens",
        type=int,
        default=4,
        help="Number of speculative draft tokens per vLLM cycle; Gemma-4 E4B docs recommend 4.",
    )
    parser.add_argument(
        "--milmmt-prompt-mode",
        default="direct",
        choices=("direct",),
        help="MiLMMT prompt mode. Keep the official direct prompt shape.",
    )
    parser.add_argument(
        "--milmmt-prompt-add-bos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Prepend the tokenizer BOS to the MiLMMT raw completion prompt, "
            "matching the offline screening tokenization (the streaming path "
            "historically omitted it). Source-map positions shift accordingly."
        ),
    )
    parser.add_argument("--milmmt-temperature", type=float, default=0.0)
    parser.add_argument("--milmmt-top-p", type=float, default=1.0)
    parser.add_argument("--milmmt-top-k", type=int, default=1)
    parser.add_argument("--milmmt-repetition-penalty", type=float, default=1.0)
    parser.add_argument(
        "--asr-gpu-memory-utilization",
        type=float,
        default=None,
        help="vLLM gpu_memory_utilization for the ASR engine (default runtime setting).",
    )
    parser.add_argument(
        "--gemma-vllm-gpu-memory-utilization",
        type=float,
        default=None,
        help="vLLM gpu_memory_utilization for the Gemma ASR engine (default 0.5).",
    )
    parser.add_argument(
        "--asr-punctuation-min-commit-words",
        type=int,
        default=0,
        help=(
            "Minimum lexical words required before the Qwen punctuation-LCP "
            "ASR path commits a sentence boundary. Default 0 preserves legacy "
            "punctuation commits."
        ),
    )
    parser.add_argument(
        "--mt-vllm-gpu-memory-utilization",
        type=float,
        default=None,
        help="vLLM gpu_memory_utilization for the MT engine (default 0.5).",
    )
    parser.add_argument(
        "--mt-max-model-len",
        type=int,
        default=None,
        help=(
            "Maximum MT prompt/context length exposed to the vLLM backend. "
            "Defaults to the runtime setting."
        ),
    )
    parser.add_argument(
        "--gemma-max-model-len",
        type=int,
        default=None,
        help=(
            "Maximum context length for Gemma-family vLLM backends. Defaults "
            "to the runtime setting."
        ),
    )
    parser.add_argument(
        "--paper-context-path",
        default=None,
        help=(
            "Path to a PaperArtifact JSON (produced by "
            "cascade.paper_context.paper_artifact) to inject as MT-side [Paper "
            "context]. Default: no context."
        ),
    )
    parser.add_argument(
        "--paper-context-mode",
        default="off",
        choices=("off", "title_abstract", "retrieved_chunks", "title_and_chunks"),
        help=(
            "Context mechanism. 'off' disables injection, 'title_abstract' "
            "renders the paper's title+abstract, 'retrieved_chunks' BM25-"
            "retrieves paragraph chunks from the artifact using the current "
            "ASR prefix + recent source history as the query, and "
            "'title_and_chunks' combines both."
        ),
    )
    parser.add_argument("--paper-context-top-k", type=int, default=3)
    parser.add_argument("--paper-context-max-chars", type=int, default=1200)
    parser.add_argument("--paper-context-history-window-words", type=int, default=60)
    parser.add_argument(
        "--paper-context-dir",
        default=None,
        help=(
            "Directory containing one PaperArtifact JSON per input, matched by "
            "input stem (e.g. talk.mp4 -> talk.json). Useful for multi-talk "
            "extra-context runs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = resolve_input_paths(inputs=args.inputs, input_dir=args.input_dir)
    if args.paper_context_path is not None and args.paper_context_dir is not None:
        raise ValueError("Use either --paper-context-path or --paper-context-dir, not both.")
    source_lang_code = canonical_language_code(args.source)
    target_lang_code = canonical_language_code(args.target)

    processor_config = SimpleNamespace(
        source_lang_code=source_lang_code,
        target_lang_code=target_lang_code,
        chunk_ms=args.chunk_ms,
        speech_chunk_size=args.chunk_ms / 1000.0,
        alignment_backend_name=args.alignment_backend_name,
        mt_backend_name=args.mt_backend_name,
        min_start_seconds=args.min_start_seconds,
        max_history_utterances=args.max_history_utterances,
        partial_max_new_tokens=args.partial_max_new_tokens,
        translation_alignatt_min_source_mass=args.translation_alignatt_min_source_mass,
        translation_alignatt_top_k_heads=args.translation_alignatt_top_k_heads,
        translation_alignatt_filter_width=args.translation_alignatt_filter_width,
        translation_alignatt_acceptance_variant=args.translation_alignatt_acceptance_variant,
        translation_alignatt_online_normalization=args.translation_alignatt_online_normalization,
        translation_alignatt_border_margin=args.translation_alignatt_border_margin,
        translation_alignatt_inaccessible_ms=args.translation_alignatt_inaccessible_ms,
        translation_alignatt_source_lcp_stability=args.translation_alignatt_source_lcp_stability,
        translation_alignatt_source_lcp_append_slack_units=(
            args.translation_alignatt_source_lcp_append_slack_units
        ),
        translation_alignatt_argmax_mass_threshold=args.translation_alignatt_argmax_mass_threshold,
        translation_alignatt_frontier_min_inaccessible_mass=args.translation_alignatt_frontier_min_inaccessible_mass,
        translation_alignatt_source_frontier_action=args.translation_alignatt_source_frontier_action,
        translation_alignatt_max_inaccessible_source_mass=args.translation_alignatt_max_inaccessible_source_mass,
        translation_alignatt_max_non_source_prompt_mass=args.translation_alignatt_max_non_source_prompt_mass,
        translation_alignatt_min_accessible_inaccessible_margin=args.translation_alignatt_min_accessible_inaccessible_margin,
        translation_alignatt_min_accepted_accessible_source_mass=args.translation_alignatt_min_accepted_accessible_source_mass,
        translation_alignatt_accepted_accessible_source_mass_recent_units=args.translation_alignatt_accepted_accessible_source_mass_recent_units,
        translation_alignatt_min_accessible_source_units=args.translation_alignatt_min_accessible_source_units,
        translation_alignatt_min_accessible_source_units_mode=args.translation_alignatt_min_accessible_source_units_mode,
        translation_alignatt_hold_back_target_units=args.translation_alignatt_hold_back_target_units,
        translation_alignatt_min_emit_target_units=args.translation_alignatt_min_emit_target_units,
        translation_alignatt_max_source_regression=args.translation_alignatt_max_source_regression,
        translation_alignatt_source_regression_min_source_mass=args.translation_alignatt_source_regression_min_source_mass,
        translation_alignatt_source_regression_min_inaccessible_mass=args.translation_alignatt_source_regression_min_inaccessible_mass,
        translation_alignatt_source_regression_recent_tokens=args.translation_alignatt_source_regression_recent_tokens,
        translation_alignatt_source_regression_reference_mode=args.translation_alignatt_source_regression_reference_mode,
        translation_alignatt_source_regression_activation_mode=args.translation_alignatt_source_regression_activation_mode,
        translation_alignatt_source_regression_activation_slack_tokens=args.translation_alignatt_source_regression_activation_slack_tokens,
        translation_alignatt_source_regression_patience_tokens=args.translation_alignatt_source_regression_patience_tokens,
        translation_alignatt_source_regression_action=args.translation_alignatt_source_regression_action,
        translation_alignatt_unit_consensus_min_head_ratio=args.translation_alignatt_unit_consensus_min_head_ratio,
        translation_alignatt_min_alignment_confidence=args.translation_alignatt_min_alignment_confidence,
        translation_alignatt_source_bearing_min_source_mass=args.translation_alignatt_source_bearing_min_source_mass,
        translation_alignatt_source_bearing_hard_inaccessible_cap=args.translation_alignatt_source_bearing_hard_inaccessible_cap,
        translation_alignatt_token_argmax_frontier_gate=args.translation_alignatt_token_argmax_frontier_gate,
        translation_alignatt_token_argmax_min_source_mass=args.translation_alignatt_token_argmax_min_source_mass,
        translation_alignatt_token_argmax_frontier_margin=args.translation_alignatt_token_argmax_frontier_margin,
        translation_alignatt_token_argmax_frontier_patience_tokens=args.translation_alignatt_token_argmax_frontier_patience_tokens,
        translation_alignatt_source_lookback_holdback=args.translation_alignatt_source_lookback_holdback,
        translation_alignatt_source_lookback_units=args.translation_alignatt_source_lookback_units,
        translation_alignatt_source_lookback_min_source_mass=args.translation_alignatt_source_lookback_min_source_mass,
        translation_alignatt_source_lookback_min_source_position=args.translation_alignatt_source_lookback_min_source_position,
        translation_alignatt_defer_low_source_terminal_punctuation=args.translation_alignatt_defer_low_source_terminal_punctuation,
        translation_alignatt_terminal_punctuation_min_source_mass=args.translation_alignatt_terminal_punctuation_min_source_mass,
        translation_acceptance_policy=args.translation_acceptance_policy,
        translation_static_cutoff_units=args.translation_static_cutoff_units,
        asr_gpu_memory_utilization=args.asr_gpu_memory_utilization,
        gemma_vllm_gpu_memory_utilization=args.gemma_vllm_gpu_memory_utilization,
        mt_vllm_enable_prefix_caching=args.mt_vllm_enable_prefix_caching,
        mt_vllm_gpu_memory_utilization=args.mt_vllm_gpu_memory_utilization,
        mt_vllm_enforce_eager=args.mt_vllm_enforce_eager,
        mt_vllm_cudagraph_mode=args.mt_vllm_cudagraph_mode,
        mt_vllm_enable_speculative_decoding=args.mt_vllm_enable_speculative_decoding,
        mt_vllm_speculative_assistant_model=args.mt_vllm_speculative_assistant_model,
        mt_vllm_num_speculative_tokens=args.mt_vllm_num_speculative_tokens,
        mt_max_model_len=args.mt_max_model_len,
        gemma_max_model_len=args.gemma_max_model_len,
        milmmt_prompt_mode=args.milmmt_prompt_mode,
        milmmt_prompt_add_bos=bool(args.milmmt_prompt_add_bos),
        milmmt_temperature=args.milmmt_temperature,
        milmmt_top_p=args.milmmt_top_p,
        milmmt_top_k=args.milmmt_top_k,
        milmmt_repetition_penalty=args.milmmt_repetition_penalty,
        asr_alignatt_frame_threshold=args.asr_alignatt_frame_threshold,
        asr_alignatt_rewind_threshold=args.asr_alignatt_rewind_threshold,
        asr_punctuation_min_commit_words=args.asr_punctuation_min_commit_words,
        asr_context_committed_words=args.asr_context_committed_words,
        paper_context_path=args.paper_context_path,
        paper_context_mode=args.paper_context_mode,
        paper_context_top_k=args.paper_context_top_k,
        paper_context_max_chars=args.paper_context_max_chars,
        paper_context_history_window_words=args.paper_context_history_window_words,
    )
    run_batch_inference(
        processor_config=processor_config,
        input_paths=input_paths,
        output_dir=args.output_dir,
        source_lang_code=source_lang_code,
        target_lang_code=target_lang_code,
        explicit_paper_context_path=args.paper_context_path,
        paper_context_dir=args.paper_context_dir,
        attention_trace_level=(
            args.trace_attention_level if args.trace_attention else None
        ),
    )


if __name__ == "__main__":
    main()
