#!/usr/bin/env python3
"""Canonical single-audio SimulStream comparison runner.

Runs the two supported cascades sequentially and in isolated subprocesses:

- ``qwen_forced`` = Qwen3-ASR + Qwen3 Forced Aligner
- ``gemma_vllm_qk_fast`` = Gemma 4 ASR + AlignAtt ``qk_fast`` streaming (vLLM-native)

Each backend writes a standard inference artifact directory, then this script
consolidates transcript quality and latency diagnostics into a comparison
report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from subprocess import run as run_subprocess
import string
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

from cascade.artifacts import (
    MANIFEST_FILENAME,
    STREAM_UPDATES_FILENAME,
    InferenceArtifacts,
    StreamUpdate,
    final_asr_filename,
    final_translation_filename,
    utc_now_isoformat,
    write_inference_artifacts,
    write_jsonl,
)
from cascade.emission import register_translation_timestamps, register_translation_words
from cascade.runtime import (
    STABLE_ALIGNMENT_BACKEND_NAMES,
    VALID_ALIGNATT_ACCEPTANCE_VARIANTS,
    VALID_ALIGNATT_ONLINE_NORMALIZATIONS,
    VALID_ALIGNATT_SOURCE_CONTEXT_MODES,
    VALID_ALIGNATT_SOURCE_FRONTIER_ACTIONS,
    VALID_ALIGNATT_SOURCE_REGRESSION_ACTIONS,
    VALID_ALIGNATT_SOURCE_REGRESSION_ACTIVATION_MODES,
    VALID_ALIGNATT_SOURCE_REGRESSION_REFERENCE_MODES,
    VALID_MT_BACKEND_NAMES,
    VALID_TRANSLATION_ACCEPTANCE_POLICIES,
)
from cascade.paper_context import VALID_CONTEXT_MODES
from cascade.simulstream_processor import CascadeAlignAttProcessor, LANGUAGE_CODE_TO_NAME
from run_simulstream_batch import (
    CHUNK_DECISIONS_FILENAME,
    chunk_decision_record,
    summarize_chunk_decisions,
)
from simulstream.server.speech_processors import SAMPLE_RATE


DEFAULT_WAV = "data/smoke/alignatt_smoke18.wav"
DEFAULT_REFERENCE = "data/smoke/smoke18_reference.txt"
DEFAULT_OUTPUT_DIR = "outputs/simulstream_compare_smoke18"
BACKEND_IDS = tuple(STABLE_ALIGNMENT_BACKEND_NAMES)


def load_wav_raw(path: str) -> np.ndarray:
    import wave as _wave

    with _wave.open(path, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        num_channels = wav_file.getnchannels()
        raw = wav_file.readframes(wav_file.getnframes())
    if sample_width != 2:
        raise ValueError("Only 16-bit PCM WAV supported.")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if num_channels > 1:
        audio = audio.reshape(-1, num_channels).mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        duration = len(audio) / sample_rate
        old_times = np.linspace(0.0, duration, num=len(audio), endpoint=False)
        new_length = int(duration * SAMPLE_RATE)
        new_times = np.linspace(0.0, duration, num=new_length, endpoint=False)
        audio = np.interp(new_times, old_times, audio).astype(np.float32)
    return audio


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _normalize_for_asr_metric(text: str) -> list[str]:
    table = str.maketrans("", "", string.punctuation + "\u201c\u201d\u2018\u2019")
    return text.lower().translate(table).split()


def _levenshtein(ref: Sequence[str], hyp: Sequence[str]) -> int:
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    previous = list(range(len(hyp) + 1))
    for i, ref_item in enumerate(ref, start=1):
        current = [i] + [0] * len(hyp)
        for j, hyp_item in enumerate(hyp, start=1):
            cost = 0 if ref_item == hyp_item else 1
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
        previous = current
    return previous[-1]


def compute_wer(reference: str, hypothesis: str) -> float:
    ref = _normalize_for_asr_metric(reference)
    hyp = _normalize_for_asr_metric(hypothesis)
    return _levenshtein(ref, hyp) / max(1, len(ref))


def compute_cer(reference: str, hypothesis: str) -> float:
    ref = list("".join(_normalize_for_asr_metric(reference)))
    hyp = list("".join(_normalize_for_asr_metric(hypothesis)))
    return _levenshtein(ref, hyp) / max(1, len(ref))


def first_nonempty_emission(
    stream_updates: list[dict[str, Any]],
) -> tuple[float | None, float | None]:
    for payload in stream_updates:
        if str(payload.get("translation_text", "")).strip():
            audio_ms = payload.get("audio_processed_ms")
            wallclock_ms = payload.get("wallclock_elapsed_ms")
            return (
                None if audio_ms is None else float(audio_ms) / 1000.0,
                None if wallclock_ms is None else float(wallclock_ms) / 1000.0,
            )
    return None, None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def selected_backend_ids(args: argparse.Namespace) -> tuple[str, ...]:
    backend_name = getattr(args, "alignment_backend_name", None)
    if backend_name is None:
        return BACKEND_IDS
    return (str(backend_name),)


def build_processor_config(args: argparse.Namespace, *, backend_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        source_lang_code=args.source,
        target_lang_code=args.target,
        chunk_ms=args.chunk_ms,
        speech_chunk_size=args.chunk_ms / 1000.0,
        alignment_backend_name=backend_name,
        mt_backend_name=args.mt_backend_name,
        asr_alignatt_commit_policy=args.asr_alignatt_commit_policy,
        asr_alignatt_frame_threshold=args.asr_alignatt_frame_threshold,
        asr_alignatt_rewind_threshold=args.asr_alignatt_rewind_threshold,
        asr_punctuation_min_commit_words=args.asr_punctuation_min_commit_words,
        asr_context_committed_words=args.asr_context_committed_words,
        min_start_seconds=args.min_start_seconds,
        max_history_utterances=args.max_history_utterances,
        partial_max_new_tokens=args.partial_max_new_tokens,
        translation_alignatt_top_k_heads=args.translation_alignatt_top_k_heads,
        translation_alignatt_filter_width=args.translation_alignatt_filter_width,
        translation_alignatt_acceptance_variant=args.translation_alignatt_acceptance_variant,
        translation_alignatt_online_normalization=args.translation_alignatt_online_normalization,
        translation_alignatt_min_source_mass=args.translation_alignatt_min_source_mass,
        translation_alignatt_border_margin=args.translation_alignatt_border_margin,
        translation_alignatt_inaccessible_ms=args.translation_alignatt_inaccessible_ms,
        translation_alignatt_source_lcp_stability=args.translation_alignatt_source_lcp_stability,
        translation_alignatt_source_lcp_append_slack_units=(
            args.translation_alignatt_source_lcp_append_slack_units
        ),
        translation_alignatt_frontier_min_inaccessible_mass=args.translation_alignatt_frontier_min_inaccessible_mass,
        translation_alignatt_source_frontier_action=args.translation_alignatt_source_frontier_action,
        translation_alignatt_max_inaccessible_source_mass=args.translation_alignatt_max_inaccessible_source_mass,
        translation_alignatt_max_non_source_prompt_mass=args.translation_alignatt_max_non_source_prompt_mass,
        translation_alignatt_min_accessible_inaccessible_margin=args.translation_alignatt_min_accessible_inaccessible_margin,
        translation_alignatt_min_accepted_accessible_source_mass=args.translation_alignatt_min_accepted_accessible_source_mass,
        translation_alignatt_accepted_accessible_source_mass_recent_units=args.translation_alignatt_accepted_accessible_source_mass_recent_units,
        translation_alignatt_argmax_mass_threshold=args.translation_alignatt_argmax_mass_threshold,
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
        translation_alignatt_source_bearing_min_source_mass=args.translation_alignatt_source_bearing_min_source_mass,
        translation_alignatt_source_bearing_hard_inaccessible_cap=args.translation_alignatt_source_bearing_hard_inaccessible_cap,
        translation_alignatt_token_argmax_frontier_gate=args.translation_alignatt_token_argmax_frontier_gate,
        translation_alignatt_token_argmax_min_source_mass=args.translation_alignatt_token_argmax_min_source_mass,
        translation_alignatt_token_argmax_frontier_margin=args.translation_alignatt_token_argmax_frontier_margin,
        translation_alignatt_token_argmax_frontier_patience_tokens=args.translation_alignatt_token_argmax_frontier_patience_tokens,
        translation_alignatt_min_alignment_confidence=args.translation_alignatt_min_alignment_confidence,
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
        milmmt_prompt_mode=args.milmmt_prompt_mode,
        milmmt_prompt_add_bos=bool(args.milmmt_prompt_add_bos),
        milmmt_temperature=args.milmmt_temperature,
        milmmt_top_p=args.milmmt_top_p,
        milmmt_top_k=args.milmmt_top_k,
        milmmt_repetition_penalty=args.milmmt_repetition_penalty,
        paper_context_path=args.paper_context_path,
        paper_context_mode=args.paper_context_mode,
        paper_context_top_k=args.paper_context_top_k,
        paper_context_max_chars=args.paper_context_max_chars,
        paper_context_history_window_words=args.paper_context_history_window_words,
    )


def run_single_backend_to_artifacts(
    processor: CascadeAlignAttProcessor,
    *,
    wav_path: str,
    chunk_ms: int,
    source_lang_code: str,
    target_lang_code: str,
    backend_name: str,
    load_ms: float,
) -> tuple[InferenceArtifacts, dict[str, Any], list[dict[str, Any]]]:
    processor.clear()
    audio = load_wav_raw(wav_path)
    chunk_size = int(SAMPLE_RATE * chunk_ms / 1000)
    audio_duration_ms = len(audio) * 1000.0 / SAMPLE_RATE

    word_delays_ms: list[float] = []
    word_elapsed_ms: list[float] = []
    updates: list[StreamUpdate] = []
    chunk_decisions: list[dict[str, Any]] = []
    emission_event_cursor = 0
    last_translation = ""
    last_raw_translation = ""
    start_time = perf_counter()

    for chunk_idx, start_sample in enumerate(range(0, len(audio), chunk_size)):
        chunk = audio[start_sample : start_sample + chunk_size].astype(np.float32)
        output = processor.process_chunk(chunk)
        current_translation = processor.tokens_to_string(processor._emitted_units)
        current_asr = processor.session.render_public_asr_text()
        audio_processed_ms = min((start_sample + chunk_size), len(audio)) * 1000.0 / SAMPLE_RATE
        wallclock_elapsed_ms = (perf_counter() - start_time) * 1000.0
        new_emission_events = processor.emission_events()[emission_event_cursor:]
        emission_event_cursor += len(new_emission_events)
        latest_emission_event = (
            dict(new_emission_events[-1]) if new_emission_events else None
        )
        chunk_decisions.append(
            chunk_decision_record(
                chunk_idx=chunk_idx,
                input_name=Path(wav_path).name,
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
                last_raw_translation,
                current_translation,
                wallclock_elapsed_ms,
                word_elapsed_ms,
                target_lang_code=target_lang_code,
            )
            new_words = register_translation_words(
                last_translation,
                current_translation,
                audio_processed_ms,
                word_delays_ms,
                target_lang_code=target_lang_code,
            )
            updates.append(
                StreamUpdate(
                    update_idx=len(updates),
                    audio_processed_ms=audio_processed_ms,
                    wallclock_elapsed_ms=wallclock_elapsed_ms,
                    asr_text=current_asr,
                    translation_text=current_translation,
                    new_words=new_words,
                )
            )
            last_translation = current_translation
            last_raw_translation = current_translation

    eos_output = processor.end_of_stream()
    final_translation = processor.tokens_to_string(processor._emitted_units)
    final_asr = processor.session.render_public_asr_text()
    final_elapsed_ms = (perf_counter() - start_time) * 1000.0
    total_wallclock_s = final_elapsed_ms / 1000.0
    rtf = total_wallclock_s / (audio_duration_ms / 1000.0) if audio_duration_ms > 0 else 0.0
    new_emission_events = processor.emission_events()[emission_event_cursor:]
    emission_event_cursor += len(new_emission_events)
    latest_eos_emission_event = (
        dict(new_emission_events[-1]) if new_emission_events else None
    )
    chunk_decisions.append(
        chunk_decision_record(
            chunk_idx=len(chunk_decisions),
            input_name=Path(wav_path).name,
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
            last_raw_translation,
            final_translation,
            final_elapsed_ms,
            word_elapsed_ms,
            target_lang_code=target_lang_code,
        )
        eos_new_words = register_translation_words(
            last_translation,
            final_translation,
            audio_duration_ms,
            word_delays_ms,
            target_lang_code=target_lang_code,
        )
        last_translation = final_translation
    else:
        eos_new_words = []

    updates.append(
        StreamUpdate(
            update_idx=len(updates),
            audio_processed_ms=audio_duration_ms,
            wallclock_elapsed_ms=final_elapsed_ms,
            asr_text=final_asr,
            translation_text=final_translation,
            new_words=eos_new_words,
            is_eos=True,
        )
    )

    runtime_config = dict(processor.session.config.__dict__)
    runtime_config["alignment_backend_name"] = backend_name
    runtime_config["chunk_decisions_filename"] = CHUNK_DECISIONS_FILENAME
    artifacts = InferenceArtifacts(
        wav_path=wav_path,
        chunk_ms=chunk_ms,
        translation_variant=processor.session.config.translation_variant_id,
        source_language=LANGUAGE_CODE_TO_NAME.get(source_lang_code, source_lang_code),
        target_language=LANGUAGE_CODE_TO_NAME.get(target_lang_code, target_lang_code),
        source_language_code=source_lang_code,
        target_language_code=target_lang_code,
        latency_unit=processor.session.config.latency_unit,
        audio_duration_ms=audio_duration_ms,
        final_asr_text=final_asr,
        final_translation_text=final_translation,
        translation_word_delays_ms=word_delays_ms,
        translation_word_elapsed_ms=word_elapsed_ms,
        updates=updates,
        runtime_config=runtime_config,
        run_provenance={
            "generated_at_utc": utc_now_isoformat(),
            "git_sha": git_sha(),
            "framework_mode": "simulstream_processor",
            "script": "run_simulstream_compare.py",
            "alignment_backend_name": backend_name,
            "model_load_ms": round(load_ms, 2),
            "total_wallclock_s": round(total_wallclock_s, 4),
            "rtf": round(rtf, 6),
        },
    )
    return artifacts, {
        "model_load_ms": round(load_ms, 2),
        "total_wallclock_s": round(total_wallclock_s, 4),
        "rtf": round(rtf, 6),
        "num_updates": len(updates),
        "num_chunks": len(chunk_decisions),
        "chunk_decision_summary": summarize_chunk_decisions(chunk_decisions),
    }, chunk_decisions


def summarize_backend_artifacts(
    artifact_dir: str | Path,
    *,
    reference_text: str,
) -> dict[str, Any]:
    artifact_path = Path(artifact_dir)
    manifest = load_json(artifact_path / MANIFEST_FILENAME)
    source_lang_code = str(manifest.get("source_language_code", "en"))
    target_lang_code = str(manifest.get("target_language_code", "de"))
    transcript_path = artifact_path / final_asr_filename(source_lang_code)
    translation_path = artifact_path / final_translation_filename(target_lang_code)
    stream_updates_path = artifact_path / STREAM_UPDATES_FILENAME
    chunk_decisions_path = artifact_path / CHUNK_DECISIONS_FILENAME

    final_asr = transcript_path.read_text(encoding="utf-8").strip()
    final_translation = translation_path.read_text(encoding="utf-8").strip()
    stream_updates = load_jsonl(stream_updates_path)
    first_audio_s, first_wallclock_s = first_nonempty_emission(stream_updates)
    audio_duration_ms = float(manifest.get("audio_duration_ms", 0.0))
    provenance = dict(manifest.get("run_provenance", {}) or {})
    total_wallclock_s = provenance.get("total_wallclock_s")
    if total_wallclock_s is None and audio_duration_ms > 0.0:
        total_wallclock_s = None
    load_ms = provenance.get("model_load_ms")
    backend_id = (
        manifest.get("runtime_config", {}) or {}
    ).get("alignment_backend_name") or provenance.get("alignment_backend_name")

    return {
        "backend_id": backend_id,
        "artifact_dir": str(artifact_path),
        "manifest_path": str(artifact_path / MANIFEST_FILENAME),
        "stream_updates_path": str(stream_updates_path),
        "chunk_decisions_path": (
            str(chunk_decisions_path) if chunk_decisions_path.is_file() else None
        ),
        "transcript_path": str(transcript_path),
        "translation_path": str(translation_path),
        "final_asr": final_asr,
        "final_translation": final_translation,
        "wer": round(compute_wer(reference_text, final_asr), 6),
        "cer": round(compute_cer(reference_text, final_asr), 6),
        "audio_duration_s": round(audio_duration_ms / 1000.0, 4),
        "model_load_ms": load_ms,
        "total_wallclock_s": total_wallclock_s,
        "rtf": provenance.get("rtf"),
        "num_updates": len(stream_updates),
        "first_nonempty_emission_audio_s": first_audio_s,
        "first_nonempty_emission_wallclock_s": first_wallclock_s,
        "has_terminal_eos_update": bool(stream_updates and stream_updates[-1].get("is_eos")),
    }


def build_comparison_report(
    *,
    wav_path: str,
    reference_path: str,
    backend_artifact_dirs: dict[str, str | Path],
) -> dict[str, Any]:
    reference_text = Path(reference_path).read_text(encoding="utf-8").strip()
    backend_reports = [
        summarize_backend_artifacts(path, reference_text=reference_text)
        for _, path in sorted(backend_artifact_dirs.items())
    ]
    return {
        "generated_at_utc": utc_now_isoformat(),
        "kind": "simulstream_compare",
        "wav_path": wav_path,
        "reference_path": reference_path,
        "backend_ids": [report["backend_id"] for report in backend_reports],
        "backends": backend_reports,
    }


def write_comparison_outputs(output_dir: str | Path, report: dict[str, Any]) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "comparison_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"WAV: {report['wav_path']}",
        f"Reference: {report['reference_path']}",
        "",
    ]
    for backend in report["backends"]:
        lines.extend(
            [
                f"[{backend['backend_id']}]",
                f"  WER={backend['wer']:.4f}  CER={backend['cer']:.4f}  "
                f"first_emit_audio_s={backend['first_nonempty_emission_audio_s']}",
                f"  load_ms={backend['model_load_ms']}  wallclock_s={backend['total_wallclock_s']}  "
                f"rtf={backend['rtf']}",
                f"  updates={backend['num_updates']}",
                f"  stream_updates={backend['stream_updates_path']}",
                "",
            ]
        )
    (output_path / "comparison_report.txt").write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )


def run_backend_subprocess(
    *,
    python_executable: str,
    backend_name: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    cmd = [
        python_executable,
        __file__,
        "--internal-run-backend",
        backend_name,
        "--wav",
        args.wav,
        "--reference",
        args.reference,
        "--output-dir",
        str(output_dir),
        "--chunk-ms",
        str(args.chunk_ms),
        "--source",
        args.source,
        "--target",
        args.target,
        "--mt-backend-name",
        args.mt_backend_name,
        "--asr-alignatt-frame-threshold",
        str(args.asr_alignatt_frame_threshold),
        "--asr-alignatt-commit-policy",
        str(args.asr_alignatt_commit_policy),
        "--asr-alignatt-rewind-threshold",
        str(args.asr_alignatt_rewind_threshold),
        "--asr-punctuation-min-commit-words",
        str(args.asr_punctuation_min_commit_words),
        "--min-start-seconds",
        str(args.min_start_seconds),
        "--max-history-utterances",
        str(args.max_history_utterances),
        "--partial-max-new-tokens",
        str(args.partial_max_new_tokens),
        "--translation-alignatt-min-source-mass",
        str(args.translation_alignatt_min_source_mass),
        "--translation-alignatt-top-k-heads",
        str(args.translation_alignatt_top_k_heads),
        "--translation-alignatt-filter-width",
        str(args.translation_alignatt_filter_width),
        "--translation-alignatt-acceptance-variant",
        str(args.translation_alignatt_acceptance_variant),
        "--translation-alignatt-online-normalization",
        str(args.translation_alignatt_online_normalization),
        "--translation-alignatt-border-margin",
        str(args.translation_alignatt_border_margin),
        "--translation-alignatt-inaccessible-ms",
        str(args.translation_alignatt_inaccessible_ms),
        "--translation-alignatt-frontier-min-inaccessible-mass",
        str(args.translation_alignatt_frontier_min_inaccessible_mass),
        "--translation-alignatt-source-frontier-action",
        str(args.translation_alignatt_source_frontier_action),
        "--translation-alignatt-max-inaccessible-source-mass",
        str(args.translation_alignatt_max_inaccessible_source_mass),
        "--translation-alignatt-max-non-source-prompt-mass",
        str(args.translation_alignatt_max_non_source_prompt_mass),
        "--translation-alignatt-min-accessible-inaccessible-margin",
        str(args.translation_alignatt_min_accessible_inaccessible_margin),
        "--translation-alignatt-min-accepted-accessible-source-mass",
        str(args.translation_alignatt_min_accepted_accessible_source_mass),
        "--translation-alignatt-accepted-accessible-source-mass-recent-units",
        str(args.translation_alignatt_accepted_accessible_source_mass_recent_units),
        "--translation-alignatt-argmax-mass-threshold",
        str(args.translation_alignatt_argmax_mass_threshold),
        "--translation-alignatt-min-accessible-source-units",
        str(args.translation_alignatt_min_accessible_source_units),
        "--translation-alignatt-min-accessible-source-units-mode",
        str(args.translation_alignatt_min_accessible_source_units_mode),
        "--translation-alignatt-hold-back-target-units",
        str(args.translation_alignatt_hold_back_target_units),
        "--translation-alignatt-min-emit-target-units",
        str(args.translation_alignatt_min_emit_target_units),
        "--translation-alignatt-max-source-regression",
        str(args.translation_alignatt_max_source_regression),
        "--translation-alignatt-source-regression-min-source-mass",
        str(args.translation_alignatt_source_regression_min_source_mass),
        "--translation-alignatt-source-regression-min-inaccessible-mass",
        str(args.translation_alignatt_source_regression_min_inaccessible_mass),
        "--translation-alignatt-source-regression-recent-tokens",
        str(args.translation_alignatt_source_regression_recent_tokens),
        "--translation-alignatt-source-regression-reference-mode",
        str(args.translation_alignatt_source_regression_reference_mode),
        "--translation-alignatt-source-regression-activation-mode",
        str(args.translation_alignatt_source_regression_activation_mode),
        "--translation-alignatt-source-regression-activation-slack-tokens",
        str(args.translation_alignatt_source_regression_activation_slack_tokens),
        "--translation-alignatt-source-regression-patience-tokens",
        str(args.translation_alignatt_source_regression_patience_tokens),
        "--translation-alignatt-source-regression-action",
        str(args.translation_alignatt_source_regression_action),
        "--translation-alignatt-unit-consensus-min-head-ratio",
        str(args.translation_alignatt_unit_consensus_min_head_ratio),
        "--translation-alignatt-source-bearing-min-source-mass",
        str(args.translation_alignatt_source_bearing_min_source_mass),
        "--translation-alignatt-source-bearing-hard-inaccessible-cap",
        str(args.translation_alignatt_source_bearing_hard_inaccessible_cap),
        "--translation-alignatt-token-argmax-min-source-mass",
        str(args.translation_alignatt_token_argmax_min_source_mass),
        "--translation-alignatt-token-argmax-frontier-margin",
        str(args.translation_alignatt_token_argmax_frontier_margin),
        "--translation-alignatt-token-argmax-frontier-patience-tokens",
        str(args.translation_alignatt_token_argmax_frontier_patience_tokens),
        "--translation-alignatt-source-lookback-units",
        str(args.translation_alignatt_source_lookback_units),
        "--translation-alignatt-source-lookback-min-source-mass",
        str(args.translation_alignatt_source_lookback_min_source_mass),
        "--translation-alignatt-source-lookback-min-source-position",
        str(args.translation_alignatt_source_lookback_min_source_position),
        "--translation-alignatt-terminal-punctuation-min-source-mass",
        str(args.translation_alignatt_terminal_punctuation_min_source_mass),
        "--translation-acceptance-policy",
        str(args.translation_acceptance_policy),
        "--translation-static-cutoff-units",
        str(args.translation_static_cutoff_units),
        "--milmmt-prompt-mode",
        str(args.milmmt_prompt_mode),
        "--milmmt-prompt-add-bos" if args.milmmt_prompt_add_bos else "--no-milmmt-prompt-add-bos",
        "--milmmt-temperature",
        str(args.milmmt_temperature),
        "--milmmt-top-p",
        str(args.milmmt_top_p),
        "--milmmt-top-k",
        str(args.milmmt_top_k),
        "--milmmt-repetition-penalty",
        str(args.milmmt_repetition_penalty),
        "--paper-context-mode",
        str(args.paper_context_mode),
        "--paper-context-top-k",
        str(args.paper_context_top_k),
        "--paper-context-max-chars",
        str(args.paper_context_max_chars),
        "--paper-context-history-window-words",
        str(args.paper_context_history_window_words),
    ]
    if args.asr_gpu_memory_utilization is not None:
        cmd.extend(
            [
                "--asr-gpu-memory-utilization",
                str(args.asr_gpu_memory_utilization),
            ]
        )
    if args.gemma_vllm_gpu_memory_utilization is not None:
        cmd.extend(
            [
                "--gemma-vllm-gpu-memory-utilization",
                str(args.gemma_vllm_gpu_memory_utilization),
            ]
        )
    if args.mt_vllm_gpu_memory_utilization is not None:
        cmd.extend(
            [
                "--mt-vllm-gpu-memory-utilization",
                str(args.mt_vllm_gpu_memory_utilization),
            ]
        )
    if args.mt_vllm_enable_prefix_caching:
        cmd.append("--mt-vllm-enable-prefix-caching")
    cmd.extend(
        [
            "--translation-alignatt-min-alignment-confidence",
            str(args.translation_alignatt_min_alignment_confidence),
        ]
    )
    cmd.extend(
        [
            "--asr-context-committed-words",
            str(args.asr_context_committed_words),
        ]
    )
    if args.mt_vllm_enforce_eager is not None:
        cmd.append(
            "--mt-vllm-enforce-eager"
            if args.mt_vllm_enforce_eager
            else "--no-mt-vllm-enforce-eager"
        )
    if args.mt_vllm_cudagraph_mode is not None:
        cmd.extend(["--mt-vllm-cudagraph-mode", str(args.mt_vllm_cudagraph_mode)])
    if args.paper_context_path is not None:
        cmd.extend(["--paper-context-path", str(args.paper_context_path)])
    if args.translation_alignatt_source_lcp_stability:
        cmd.append("--translation-alignatt-source-lcp-stability")
    if args.translation_alignatt_token_argmax_frontier_gate:
        cmd.append("--translation-alignatt-token-argmax-frontier-gate")
    if args.translation_alignatt_source_lookback_holdback:
        cmd.append("--translation-alignatt-source-lookback-holdback")
    if args.translation_alignatt_defer_low_source_terminal_punctuation:
        cmd.append("--translation-alignatt-defer-low-source-terminal-punctuation")
    cmd.extend(
        [
            "--translation-alignatt-source-lcp-append-slack-units",
            str(args.translation_alignatt_source_lcp_append_slack_units),
        ]
    )
    if args.mt_vllm_enable_speculative_decoding:
        cmd.append("--mt-vllm-enable-speculative-decoding")
    if args.mt_vllm_speculative_assistant_model:
        cmd.extend(
            [
                "--mt-vllm-speculative-assistant-model",
                str(args.mt_vllm_speculative_assistant_model),
            ]
        )
    cmd.extend(
        [
            "--mt-vllm-num-speculative-tokens",
            str(args.mt_vllm_num_speculative_tokens),
        ]
    )
    completed = run_subprocess(cmd, check=False, cwd=str(Path(__file__).resolve().parent))
    if completed.returncode != 0:
        raise SystemExit(
            f"Backend subprocess failed for {backend_name}: exit code {completed.returncode}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wav", default=DEFAULT_WAV)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-ms", default=850, type=int)
    parser.add_argument("--source", default="en")
    parser.add_argument("--target", default="de")
    parser.add_argument(
        "--alignment-backend-name",
        default=None,
        choices=BACKEND_IDS,
        help=(
            "Run only one alignment backend. Omit to preserve the historical "
            "two-backend comparison."
        ),
    )
    parser.add_argument(
        "--mt-backend-name",
        default="gemma_vllm_alignatt",
        choices=VALID_MT_BACKEND_NAMES,
        help=(
            "MT backend route. Use milmmt_vllm_alignatt for the active "
            "EN->ZH MiLMMT improvement path."
        ),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--asr-alignatt-commit-policy",
        default="frontier_flush",
        choices=("frontier_flush", "rewind_abort"),
        help=(
            "ASR AlignAtt commit rule. `frontier_flush` commits the maximal "
            "monotone prefix on every chunk and only keeps the final frontier "
            "band; `rewind_abort` preserves the legacy whole-chunk abort path."
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
            "Legacy `rewind_abort` guard in audio frames. Ignored by the "
            "default `frontier_flush` policy."
        ),
    )
    parser.add_argument(
        "--asr-punctuation-min-commit-words",
        default=0,
        type=int,
        help=(
            "Minimum lexical words required before the Qwen punctuation-LCP "
            "ASR path commits a sentence boundary. Default 0 preserves legacy "
            "punctuation commits."
        ),
    )
    parser.add_argument("--min-start-seconds", default=2.0, type=float)
    parser.add_argument("--max-history-utterances", default=0, type=int)
    parser.add_argument("--partial-max-new-tokens", default=16, type=int)
    parser.add_argument("--translation-alignatt-min-source-mass", default=0.0, type=float)
    parser.add_argument("--translation-alignatt-top-k-heads", default=4, type=int)
    parser.add_argument("--translation-alignatt-filter-width", default=7, type=int)
    parser.add_argument(
        "--translation-alignatt-acceptance-variant",
        default="token",
        choices=VALID_ALIGNATT_ACCEPTANCE_VARIANTS,
    )
    parser.add_argument(
        "--translation-alignatt-online-normalization",
        default="zscore",
        choices=VALID_ALIGNATT_ONLINE_NORMALIZATIONS,
    )
    parser.add_argument("--translation-alignatt-border-margin", default=1, type=int)
    parser.add_argument("--translation-alignatt-inaccessible-ms", default=0.0, type=float)
    parser.add_argument("--translation-alignatt-source-lcp-stability", action="store_true")
    parser.add_argument(
        "--translation-alignatt-source-lcp-append-slack-units",
        default=0,
        type=int,
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
        choices=VALID_ALIGNATT_SOURCE_FRONTIER_ACTIONS,
        help=(
            "How to handle source-frontier hits. 'stop' preserves the "
            "historical token-level hard stop; 'trim_unrecovered' lets the "
            "draft continue and trims only an unrecovered target-unit suffix."
        ),
    )
    parser.add_argument(
        "--translation-alignatt-max-inaccessible-source-mass",
        default=1.0,
        type=float,
        help=(
            "Optional provenance gate. Default 1.0 disables this gate for "
            "permissive AlignAtt; pass a lower value only for guarded "
            "diagnostics."
        ),
    )
    parser.add_argument(
        "--translation-alignatt-max-non-source-prompt-mass",
        default=1.0,
        type=float,
        help=(
            "Optional provenance diagnostic gate on non-source prompt mass. "
            "Default 1.0 disables the gate."
        ),
    )
    parser.add_argument("--translation-alignatt-min-accessible-inaccessible-margin", default=-1.0, type=float)
    parser.add_argument("--translation-alignatt-min-accepted-accessible-source-mass", default=0.0, type=float)
    parser.add_argument("--translation-alignatt-accepted-accessible-source-mass-recent-units", default=2, type=int)
    parser.add_argument("--translation-alignatt-argmax-mass-threshold", default=0.0, type=float)
    parser.add_argument("--translation-alignatt-min-accessible-source-units", default=0, type=int)
    parser.add_argument(
        "--translation-alignatt-min-accessible-source-units-mode",
        default="block",
        choices=VALID_ALIGNATT_SOURCE_CONTEXT_MODES,
    )
    parser.add_argument("--translation-alignatt-hold-back-target-units", default=0, type=int)
    parser.add_argument("--translation-alignatt-min-emit-target-units", default=0, type=int)
    parser.add_argument("--translation-alignatt-max-source-regression", default=-1, type=int)
    parser.add_argument("--translation-alignatt-source-regression-min-source-mass", default=0.0, type=float)
    parser.add_argument("--translation-alignatt-source-regression-min-inaccessible-mass", default=0.0, type=float)
    parser.add_argument("--translation-alignatt-source-regression-recent-tokens", default=0, type=int)
    parser.add_argument(
        "--translation-alignatt-source-regression-reference-mode",
        default="max",
        choices=VALID_ALIGNATT_SOURCE_REGRESSION_REFERENCE_MODES,
    )
    parser.add_argument(
        "--translation-alignatt-source-regression-activation-mode",
        default="always",
        choices=VALID_ALIGNATT_SOURCE_REGRESSION_ACTIVATION_MODES,
    )
    parser.add_argument("--translation-alignatt-source-regression-activation-slack-tokens", default=0, type=int)
    parser.add_argument("--translation-alignatt-source-regression-patience-tokens", default=1, type=int)
    parser.add_argument(
        "--translation-alignatt-source-regression-action",
        default="stop",
        choices=VALID_ALIGNATT_SOURCE_REGRESSION_ACTIONS,
    )
    parser.add_argument("--translation-alignatt-unit-consensus-min-head-ratio", default=0.60, type=float)
    parser.add_argument("--translation-alignatt-min-alignment-confidence", default=0.0, type=float)
    parser.add_argument("--asr-context-committed-words", default=0, type=int)
    parser.add_argument("--translation-alignatt-source-bearing-min-source-mass", default=0.005, type=float)
    parser.add_argument("--translation-alignatt-source-bearing-hard-inaccessible-cap", default=1.0, type=float)
    parser.add_argument("--translation-alignatt-token-argmax-frontier-gate", action="store_true")
    parser.add_argument("--translation-alignatt-token-argmax-min-source-mass", default=0.05, type=float)
    parser.add_argument("--translation-alignatt-token-argmax-frontier-margin", default=0, type=int)
    parser.add_argument("--translation-alignatt-token-argmax-frontier-patience-tokens", default=1, type=int)
    parser.add_argument("--translation-alignatt-source-lookback-holdback", action="store_true")
    parser.add_argument("--translation-alignatt-source-lookback-units", default=2, type=int)
    parser.add_argument("--translation-alignatt-source-lookback-min-source-mass", default=0.05, type=float)
    parser.add_argument("--translation-alignatt-source-lookback-min-source-position", default=3, type=int)
    parser.add_argument("--translation-alignatt-defer-low-source-terminal-punctuation", action="store_true")
    parser.add_argument("--translation-alignatt-terminal-punctuation-min-source-mass", default=0.06, type=float)
    parser.add_argument(
        "--translation-acceptance-policy",
        default="alignatt",
        choices=VALID_TRANSLATION_ACCEPTANCE_POLICIES,
    )
    parser.add_argument("--translation-static-cutoff-units", default=0, type=int)
    parser.add_argument("--mt-vllm-enable-speculative-decoding", action="store_true")
    parser.add_argument("--mt-vllm-speculative-assistant-model", default=None)
    parser.add_argument(
        "--mt-vllm-num-speculative-tokens",
        "--mt-vllm-speculative-num-tokens",
        dest="mt_vllm_num_speculative_tokens",
        default=4,
        type=int,
    )
    parser.add_argument("--mt-vllm-enable-prefix-caching", action="store_true")
    parser.add_argument("--mt-vllm-gpu-memory-utilization", default=None, type=float)
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
    parser.add_argument("--asr-gpu-memory-utilization", default=None, type=float)
    parser.add_argument("--gemma-vllm-gpu-memory-utilization", default=None, type=float)
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
    parser.add_argument("--milmmt-temperature", default=0.0, type=float)
    parser.add_argument("--milmmt-top-p", default=1.0, type=float)
    parser.add_argument("--milmmt-top-k", default=1, type=int)
    parser.add_argument("--milmmt-repetition-penalty", default=1.0, type=float)
    parser.add_argument("--paper-context-path", default=None)
    parser.add_argument(
        "--paper-context-mode",
        default="off",
        choices=VALID_CONTEXT_MODES,
    )
    parser.add_argument("--paper-context-top-k", default=3, type=int)
    parser.add_argument("--paper-context-max-chars", default=1200, type=int)
    parser.add_argument("--paper-context-history-window-words", default=60, type=int)
    parser.add_argument(
        "--internal-run-backend",
        choices=BACKEND_IDS,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def internal_run_backend(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor_config = build_processor_config(
        args,
        backend_name=str(args.internal_run_backend),
    )
    load_start = perf_counter()
    CascadeAlignAttProcessor.load_model(processor_config)
    load_ms = (perf_counter() - load_start) * 1000.0

    processor = CascadeAlignAttProcessor(processor_config)
    processor.set_source_language(args.source)
    processor.set_target_language(args.target)
    artifacts, summary, chunk_decisions = run_single_backend_to_artifacts(
        processor,
        wav_path=args.wav,
        chunk_ms=args.chunk_ms,
        source_lang_code=args.source,
        target_lang_code=args.target,
        backend_name=str(args.internal_run_backend),
        load_ms=load_ms,
    )
    written = write_inference_artifacts(artifacts, output_dir)
    chunk_decisions_path = output_dir / CHUNK_DECISIONS_FILENAME
    write_jsonl(chunk_decisions_path, chunk_decisions)
    written["chunk_decisions"] = str(chunk_decisions_path)
    backend_summary = {
        "backend_id": args.internal_run_backend,
        "files": written,
        **summary,
    }
    (output_dir / "backend_summary.json").write_text(
        json.dumps(backend_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(backend_summary, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()

    if args.internal_run_backend is not None:
        internal_run_backend(args)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    backend_dirs: dict[str, Path] = {}
    for backend_name in selected_backend_ids(args):
        backend_output_dir = output_dir / backend_name
        backend_dirs[backend_name] = backend_output_dir
        print(f"\n[{backend_name}] running isolated SimulStream comparison pass")
        run_backend_subprocess(
            python_executable=args.python,
            backend_name=backend_name,
            args=args,
            output_dir=backend_output_dir,
        )

    report = build_comparison_report(
        wav_path=args.wav,
        reference_path=args.reference,
        backend_artifact_dirs=backend_dirs,
    )
    write_comparison_outputs(output_dir, report)

    print("\nComparison summary")
    print("=" * 60)
    for backend in report["backends"]:
        print(
            f"{backend['backend_id']}: "
            f"WER={backend['wer']:.4f}  CER={backend['cer']:.4f}  "
            f"first_emit={backend['first_nonempty_emission_audio_s']}s  "
            f"load={backend['model_load_ms']}ms  "
            f"wallclock={backend['total_wallclock_s']}s  "
            f"RTF={backend['rtf']}"
        )
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
