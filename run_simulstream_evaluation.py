#!/usr/bin/env python3
"""Run audio through the SimulStream CascadeAlignAttProcessor and produce
evaluation-compatible artifacts (hypothesis.jsonl, manifest.json, etc.).

This bridges the real SimulStream delivery path to the existing
evaluate_cascade_outputs.py pipeline so we can measure quality and latency
through the canonical processor, not only the research harness.

Usage (from .venv-inference):
    python run_simulstream_evaluation.py \\
        --wav test-set/audio/ccpXHNfaoy.wav \\
        --output-dir outputs/simulstream_ende_2s \\
        --chunk-ms 450 --target de
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np

from cascade_simulstream_processor import CascadeAlignAttProcessor, LANGUAGE_CODE_TO_NAME
from cascade_artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    DEFAULT_WAV_PATH,
    HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
    HYPOTHESIS_FILENAME,
    MANIFEST_FILENAME,
    STREAM_UPDATE_ELAPSED_SEMANTICS_WALLCLOCK,
    STREAM_UPDATES_FILENAME,
    ensure_output_dir,
    final_asr_filename,
    final_translation_filename,
    normalize_computation_aware_timestamps,
    utc_now_isoformat,
    write_json,
    write_jsonl,
    write_text,
)
from cascade_text_surface import (
    is_char_level_target_lang,
    split_target_emission_units,
)
from cascade_emission import register_translation_timestamps, register_translation_words
from simulstream.server.speech_processors import SAMPLE_RATE


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
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run audio through SimulStream processor and produce evaluation artifacts.",
    )
    parser.add_argument("--wav", default=DEFAULT_WAV_PATH, help="Input WAV file.")
    parser.add_argument("--output-dir", required=True, help="Artifact output directory.")
    parser.add_argument("--chunk-ms", default=450, type=int)
    parser.add_argument("--source", default="en", help="Source language code.")
    parser.add_argument("--target", default="de", help="Target language code.")
    parser.add_argument("--min-start-seconds", default=2.0, type=float)
    parser.add_argument("--max-history-utterances", default=1, type=int)
    parser.add_argument("--partial-max-new-tokens", default=16, type=int)
    parser.add_argument("--partial-followup-max-new-tokens", default=8, type=int)
    parser.add_argument("--translation-alignatt-min-source-mass", default=0.0, type=float)
    parser.add_argument("--translation-alignatt-rewind-threshold", default=8, type=int)
    parser.add_argument("--translation-alignatt-inaccessible-ms", default=0.0, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_lang_code = args.target
    source_lang_code = args.source

    processor_config = SimpleNamespace(
        source_lang_code=source_lang_code,
        target_lang_code=target_lang_code,
        chunk_ms=args.chunk_ms,
        speech_chunk_size=args.chunk_ms / 1000.0,
        min_start_seconds=args.min_start_seconds,
        max_history_utterances=args.max_history_utterances,
        partial_max_new_tokens=args.partial_max_new_tokens,
        partial_followup_max_new_tokens=args.partial_followup_max_new_tokens,
        translation_alignatt_min_source_mass=args.translation_alignatt_min_source_mass,
        translation_alignatt_rewind_threshold=args.translation_alignatt_rewind_threshold,
        translation_alignatt_inaccessible_ms=args.translation_alignatt_inaccessible_ms,
    )

    print(f"Loading models for {source_lang_code}->{target_lang_code} ...")
    CascadeAlignAttProcessor.load_model(processor_config)
    processor = CascadeAlignAttProcessor(processor_config)
    processor.set_source_language(source_lang_code)
    processor.set_target_language(target_lang_code)
    processor.clear()

    audio = load_wav_raw(args.wav)
    chunk_size = int(SAMPLE_RATE * args.chunk_ms / 1000)
    audio_duration_ms = len(audio) * 1000.0 / SAMPLE_RATE

    word_delays_ms: list[float] = []
    word_elapsed_ms: list[float] = []
    stream_updates: list[dict[str, Any]] = []
    last_translation = ""
    last_raw_translation = ""
    start_time = perf_counter()

    print(f"Streaming {audio_duration_ms / 1000:.1f}s audio at {args.chunk_ms}ms chunks ...")

    for chunk_idx, start_sample in enumerate(range(0, len(audio), chunk_size)):
        chunk = audio[start_sample : start_sample + chunk_size].astype(np.float32)
        output = processor.process_chunk(chunk)

        current_translation = processor.tokens_to_string(processor._emitted_units)
        audio_processed_ms = min((start_sample + chunk_size), len(audio)) * 1000.0 / SAMPLE_RATE
        wallclock_elapsed_ms = (perf_counter() - start_time) * 1000.0

        if output.new_tokens or output.deleted_tokens:
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
            stream_updates.append({
                "update_idx": len(stream_updates),
                "audio_processed_ms": audio_processed_ms,
                "wallclock_elapsed_ms": wallclock_elapsed_ms,
                "translation_text": current_translation,
                "new_words": new_words,
                "new_string": output.new_string,
                "deleted_string": output.deleted_string,
            })
            current_time = audio_processed_ms / 1000.0
            print(f"[{current_time:6.2f}s] {target_lang_code.upper()}: {current_translation}")
            last_translation = current_translation
            last_raw_translation = current_translation

    eos_output = processor.end_of_stream()
    final_translation = processor.tokens_to_string(processor._emitted_units)
    final_elapsed_ms = (perf_counter() - start_time) * 1000.0

    if eos_output.new_tokens or eos_output.deleted_tokens or final_translation != last_translation:
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
        stream_updates.append({
            "update_idx": len(stream_updates),
            "audio_processed_ms": audio_duration_ms,
            "wallclock_elapsed_ms": final_elapsed_ms,
            "translation_text": final_translation,
            "new_words": eos_new_words,
            "new_string": eos_output.new_string,
            "deleted_string": eos_output.deleted_string,
            "is_eos": True,
        })

    # Get final ASR from core state
    core = CascadeAlignAttProcessor._get_core()
    final_asr = core.render_public_asr_text()

    total_wallclock_s = perf_counter() - start_time
    rtf = total_wallclock_s / (audio_duration_ms / 1000.0) if audio_duration_ms > 0 else 0.0

    print(f"\nFinal ASR: {final_asr}")
    print(f"Final translation: {final_translation}")
    print(f"RTF: {rtf:.4f}")

    # Build hypothesis record
    normalized_elapsed_ms = normalize_computation_aware_timestamps(
        word_delays_ms, word_elapsed_ms
    )
    units = split_target_emission_units(final_translation, target_lang_code=target_lang_code)
    prediction = (
        "".join(units) if is_char_level_target_lang(target_lang_code) else " ".join(units)
    )
    hypothesis_record = {
        "source": [Path(args.wav).name],
        "source_length": audio_duration_ms,
        "prediction": prediction,
        "delays": word_delays_ms,
        "elapsed": normalized_elapsed_ms,
        "elapsed_wallclock_ms": word_elapsed_ms,
        "elapsed_semantics": HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
    }

    # Build runtime config for manifest
    runtime_config: dict[str, Any] = {
        "chunk_ms": args.chunk_ms,
        "min_start_seconds": args.min_start_seconds,
        "max_history_utterances": args.max_history_utterances,
        "partial_max_new_tokens": args.partial_max_new_tokens,
        "partial_followup_max_new_tokens": args.partial_followup_max_new_tokens,
        "translation_alignatt_min_source_mass": args.translation_alignatt_min_source_mass,
        "translation_alignatt_rewind_threshold": args.translation_alignatt_rewind_threshold,
        "translation_alignatt_inaccessible_ms": args.translation_alignatt_inaccessible_ms,
        "hypothesis_elapsed_semantics": HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
        "stream_update_elapsed_semantics": STREAM_UPDATE_ELAPSED_SEMANTICS_WALLCLOCK,
    }
    # Pull additional config from core
    for key in [
        "translation_alignatt_heads_path",
        "translation_alignatt_top_k_heads",
        "translation_alignatt_filter_width",
        "translation_alignatt_probe_mode",
        "translation_emit_policy",
        "translation_max_tail_rewrite_words",
        "temperature",
        "repetition_penalty",
    ]:
        runtime_config[key] = getattr(core.config, key, None)

    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "generated_at_utc": utc_now_isoformat(),
        "kind": "inference",
        "wav_path": args.wav,
        "chunk_ms": args.chunk_ms,
        "source_language": LANGUAGE_CODE_TO_NAME.get(source_lang_code, source_lang_code),
        "target_language": LANGUAGE_CODE_TO_NAME.get(target_lang_code, target_lang_code),
        "source_language_code": source_lang_code,
        "target_language_code": target_lang_code,
        "audio_duration_ms": audio_duration_ms,
        "files": {
            "hypothesis_jsonl": HYPOTHESIS_FILENAME,
            "stream_updates_jsonl": STREAM_UPDATES_FILENAME,
            "transcript_txt": final_asr_filename(source_lang_code),
            "translation_txt": final_translation_filename(target_lang_code),
        },
        "runtime_config": runtime_config,
        "run_provenance": {
            "git_sha": git_sha(),
            "framework_mode": "simulstream_processor",
            "source_lang": source_lang_code,
            "target_lang": target_lang_code,
            "script": "run_simulstream_evaluation.py",
        },
        "speed": {
            "total_wallclock_s": round(total_wallclock_s, 2),
            "rtf": round(rtf, 4),
            "num_updates": len(stream_updates),
        },
    }

    # Write artifacts
    output_path = ensure_output_dir(args.output_dir)
    write_json(output_path / MANIFEST_FILENAME, manifest)
    write_jsonl(output_path / HYPOTHESIS_FILENAME, [hypothesis_record])
    write_jsonl(output_path / STREAM_UPDATES_FILENAME, stream_updates)
    write_text(output_path / final_asr_filename(source_lang_code), final_asr)
    write_text(output_path / final_translation_filename(target_lang_code), final_translation)

    print(f"\nWrote artifacts to {args.output_dir}")
    print(f"  manifest: {output_path / MANIFEST_FILENAME}")
    print(f"  hypothesis: {output_path / HYPOTHESIS_FILENAME}")
    print(f"  stream_updates: {output_path / STREAM_UPDATES_FILENAME}")
    print(f"\nTo evaluate: python evaluate_cascade_outputs.py --output-dir {args.output_dir} --skip-comet")


if __name__ == "__main__":
    main()
