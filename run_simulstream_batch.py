#!/usr/bin/env python3
"""Batch evaluation runner for the SimulStream CascadeAlignAttProcessor.

Runs multiple WAV files through the processor in a single process, keeping
models hot across audios to avoid repeated 5-minute load costs.

Usage (from .venv-inference):
    # Sanity set (3 audios):
    python run_simulstream_batch.py \\
        --wavs test-set/audio/myfXyntFYL.wav test-set/audio/DyXpuURBMP.wav test-set/audio/ccpXHNfaoy.wav \\
        --output-dir outputs/simulstream_batch_ende_2s \\
        --chunk-ms 450 --target de

    # Full set (all audios in directory):
    python run_simulstream_batch.py \\
        --wav-dir test-set/audio/ \\
        --output-dir outputs/simulstream_fullset_ende_2s \\
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
from cascade_text_surface import is_char_level_target_lang, split_target_emission_units
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


def run_single_audio(
    processor: CascadeAlignAttProcessor,
    wav_path: str,
    chunk_ms: int,
    target_lang_code: str,
    source_lang_code: str,
) -> dict[str, Any]:
    """Run one audio through the processor and return all artifacts data."""
    processor.clear()
    audio = load_wav_raw(wav_path)
    chunk_size = int(SAMPLE_RATE * chunk_ms / 1000)
    audio_duration_ms = len(audio) * 1000.0 / SAMPLE_RATE

    word_delays_ms: list[float] = []
    word_elapsed_ms: list[float] = []
    stream_updates: list[dict[str, Any]] = []
    last_translation = ""
    last_raw_translation = ""
    start_time = perf_counter()

    for start_sample in range(0, len(audio), chunk_size):
        chunk = audio[start_sample : start_sample + chunk_size].astype(np.float32)
        output = processor.process_chunk(chunk)
        current_translation = processor.tokens_to_string(processor._emitted_units)
        audio_processed_ms = min((start_sample + chunk_size), len(audio)) * 1000.0 / SAMPLE_RATE
        wallclock_elapsed_ms = (perf_counter() - start_time) * 1000.0

        if output.new_tokens or output.deleted_tokens:
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
            stream_updates.append({
                "update_idx": len(stream_updates),
                "wav_name": Path(wav_path).name,
                "audio_processed_ms": audio_processed_ms,
                "wallclock_elapsed_ms": wallclock_elapsed_ms,
                "translation_text": current_translation,
                "new_words": new_words,
            })
            last_translation = current_translation
            last_raw_translation = current_translation

    eos_output = processor.end_of_stream()
    final_translation = processor.tokens_to_string(processor._emitted_units)
    final_elapsed_ms = (perf_counter() - start_time) * 1000.0

    if eos_output.new_tokens or eos_output.deleted_tokens or final_translation != last_translation:
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
        stream_updates.append({
            "update_idx": len(stream_updates),
            "wav_name": Path(wav_path).name,
            "audio_processed_ms": audio_duration_ms,
            "wallclock_elapsed_ms": final_elapsed_ms,
            "translation_text": final_translation,
            "new_words": eos_new_words,
            "is_eos": True,
        })

    core = CascadeAlignAttProcessor._get_core()
    final_asr = core.render_public_asr_text()
    total_wallclock_s = perf_counter() - start_time
    rtf = total_wallclock_s / (audio_duration_ms / 1000.0) if audio_duration_ms > 0 else 0.0

    normalized_elapsed_ms = normalize_computation_aware_timestamps(word_delays_ms, word_elapsed_ms)
    units = split_target_emission_units(final_translation, target_lang_code=target_lang_code)
    prediction = "".join(units) if is_char_level_target_lang(target_lang_code) else " ".join(units)

    return {
        "wav_path": wav_path,
        "wav_name": Path(wav_path).name,
        "audio_duration_ms": audio_duration_ms,
        "total_wallclock_s": total_wallclock_s,
        "rtf": rtf,
        "final_asr": final_asr,
        "final_translation": final_translation,
        "num_updates": len(stream_updates),
        "hypothesis_record": {
            "source": [Path(wav_path).name],
            "source_length": audio_duration_ms,
            "prediction": prediction,
            "delays": word_delays_ms,
            "elapsed": normalized_elapsed_ms,
            "elapsed_wallclock_ms": word_elapsed_ms,
            "elapsed_semantics": HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
        },
        "stream_updates": stream_updates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch SimulStream evaluation runner.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--wavs", nargs="+", help="List of WAV file paths.")
    group.add_argument("--wav-dir", help="Directory of WAV files (all .wav files used).")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-ms", default=450, type=int)
    parser.add_argument("--source", default="en")
    parser.add_argument("--target", default="de")
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

    if args.wav_dir:
        wav_paths = sorted(str(p) for p in Path(args.wav_dir).glob("*.wav")
                           if not p.name.endswith("_short60s.wav"))
    else:
        wav_paths = args.wavs

    print(f"Will process {len(wav_paths)} audio files for {args.source}->{args.target}")

    processor_config = SimpleNamespace(
        source_lang_code=args.source,
        target_lang_code=args.target,
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

    print("Loading models ...")
    load_start = perf_counter()
    CascadeAlignAttProcessor.load_model(processor_config)
    load_ms = (perf_counter() - load_start) * 1000.0
    print(f"Models loaded in {load_ms:.0f} ms")

    processor = CascadeAlignAttProcessor(processor_config)
    processor.set_source_language(args.source)
    processor.set_target_language(args.target)

    all_hypothesis_records: list[dict] = []
    all_stream_updates: list[dict] = []
    per_audio_results: list[dict] = []
    batch_start = perf_counter()

    for idx, wav_path in enumerate(wav_paths):
        print(f"\n[{idx+1}/{len(wav_paths)}] {Path(wav_path).name} ...", flush=True)
        result = run_single_audio(
            processor, wav_path, args.chunk_ms,
            args.target, args.source,
        )
        all_hypothesis_records.append(result["hypothesis_record"])
        all_stream_updates.extend(result["stream_updates"])
        per_audio_results.append({
            "wav": result["wav_name"],
            "audio_s": round(result["audio_duration_ms"] / 1000, 1),
            "rtf": round(result["rtf"], 4),
            "updates": result["num_updates"],
        })
        print(f"  RTF={result['rtf']:.3f}  updates={result['num_updates']}  "
              f"wallclock={result['total_wallclock_s']:.1f}s")

    batch_wallclock_s = perf_counter() - batch_start
    total_audio_s = sum(r["audio_s"] for r in per_audio_results)
    batch_rtf = batch_wallclock_s / total_audio_s if total_audio_s > 0 else 0.0

    # Write artifacts
    output_path = ensure_output_dir(args.output_dir)
    core = CascadeAlignAttProcessor._get_core()

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
    for key in [
        "translation_alignatt_heads_path", "translation_alignatt_top_k_heads",
        "translation_alignatt_filter_width", "translation_alignatt_probe_mode",
        "translation_emit_policy", "translation_max_tail_rewrite_words",
        "temperature", "repetition_penalty",
    ]:
        runtime_config[key] = getattr(core.config, key, None)

    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "generated_at_utc": utc_now_isoformat(),
        "kind": "inference_batch",
        "num_audios": len(wav_paths),
        "wav_paths": wav_paths,
        "source_language": LANGUAGE_CODE_TO_NAME.get(args.source, args.source),
        "target_language": LANGUAGE_CODE_TO_NAME.get(args.target, args.target),
        "source_language_code": args.source,
        "target_language_code": args.target,
        "runtime_config": runtime_config,
        "run_provenance": {
            "git_sha": git_sha(),
            "framework_mode": "simulstream_processor",
            "script": "run_simulstream_batch.py",
        },
        "speed": {
            "batch_wallclock_s": round(batch_wallclock_s, 2),
            "batch_rtf": round(batch_rtf, 4),
            "total_audio_s": round(total_audio_s, 1),
            "per_audio": per_audio_results,
        },
    }

    write_json(output_path / MANIFEST_FILENAME, manifest)
    write_jsonl(output_path / HYPOTHESIS_FILENAME, all_hypothesis_records)
    write_jsonl(output_path / STREAM_UPDATES_FILENAME, all_stream_updates)

    print(f"\n{'='*60}")
    print(f"Batch complete: {len(wav_paths)} audios, {total_audio_s:.0f}s total audio")
    print(f"Batch wallclock: {batch_wallclock_s:.1f}s  RTF: {batch_rtf:.4f}")
    print(f"Artifacts: {args.output_dir}")
    print(f"Evaluate: python evaluate_cascade_outputs.py --output-dir {args.output_dir} --skip-comet")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
