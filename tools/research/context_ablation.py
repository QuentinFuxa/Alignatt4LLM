#!/usr/bin/env python3
"""Single-audio three-condition ablation for MT-side paper context injection.

Loads ASR + MT once, then runs the same audio three times through the
canonical SimulStream processor while flipping only the paper-context mode:

    1. no context (paper_context_mode='off')
    2. static title + abstract
    3. BM25-retrieved paper chunks

The only knob that changes between conditions is ``paper_context_mode``; the
paper artifact, ``max_history_utterances`` and every other runtime knob are
held constant. The bundle is hot-reused between conditions (the
PaperContextSelector is cheap to rebuild — PLAN.md Step 4 explicitly asks for
a minimal three-condition comparison).

Usage (from ``.venv-inference``):

    python tools/research/context_ablation.py \
        --wav data/smoke/ccpXHNfaoy_first75.wav \
        --paper-context-path data/context_artifacts/ccpXHNfaoy.json \
        --output-dir outputs/context_ablation_ccp75 \
        --source en --target de --chunk-ms 800
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np

from alignatt4llm.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    ensure_output_dir,
    final_asr_filename,
    final_translation_filename,
    normalize_computation_aware_timestamps,
    utc_now_isoformat,
    write_json,
    write_jsonl,
    write_text,
)
from alignatt4llm.emission import register_translation_timestamps, register_translation_words
from alignatt4llm.runtime import VALID_MT_BACKEND_NAMES
from alignatt4llm.simulstream_processor import CascadeAlignAttProcessor, LANGUAGE_CODE_TO_NAME
from alignatt4llm.text_surface import prediction_text_from_target_surface
from simulstream.server.speech_processors import SAMPLE_RATE


CONDITIONS = ("off", "title_abstract", "retrieved_chunks")
DEFAULT_MAX_HISTORY_UTTERANCES = 0


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
        old = np.linspace(0.0, duration, num=len(audio), endpoint=False)
        new_length = int(duration * SAMPLE_RATE)
        new = np.linspace(0.0, duration, num=new_length, endpoint=False)
        audio = np.interp(new, old, audio).astype(np.float32)
    return audio


@dataclass
class ConditionResult:
    mode: str
    final_translation: str
    final_asr: str
    num_updates: int
    rtf: float
    wallclock_s: float
    first_emit_audio_s: float | None
    first_emit_wallclock_s: float | None
    mean_word_delay_ms: float | None
    stream_updates: list[dict[str, Any]]
    hypothesis_record: dict[str, Any]


def build_processor_config(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        source_lang_code=args.source,
        target_lang_code=args.target,
        chunk_ms=args.chunk_ms,
        speech_chunk_size=args.chunk_ms / 1000.0,
        alignment_backend_name=args.alignment_backend_name,
        mt_backend_name=args.mt_backend_name,
        min_start_seconds=args.min_start_seconds,
        max_history_utterances=args.max_history_utterances,
        paper_context_path=args.paper_context_path,
        paper_context_mode="off",
        paper_context_top_k=args.paper_context_top_k,
        paper_context_max_chars=args.paper_context_max_chars,
        paper_context_history_window_words=args.paper_context_history_window_words,
        translation_alignatt_min_source_mass=args.translation_alignatt_min_source_mass,
    )


def build_summary(
    *,
    args: argparse.Namespace,
    load_ms: float,
    per_mode_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "generated_at_utc": utc_now_isoformat(),
        "kind": "context_ablation_single_audio",
        "wav": args.wav,
        "paper_context_path": args.paper_context_path,
        "source_language_code": args.source,
        "target_language_code": args.target,
        "alignment_backend_name": args.alignment_backend_name,
        "mt_backend_name": args.mt_backend_name,
        "chunk_ms": args.chunk_ms,
        "max_history_utterances": args.max_history_utterances,
        "paper_context_top_k": args.paper_context_top_k,
        "paper_context_max_chars": args.paper_context_max_chars,
        "paper_context_history_window_words": args.paper_context_history_window_words,
        "translation_alignatt_min_source_mass": args.translation_alignatt_min_source_mass,
        "model_load_ms": round(load_ms, 1),
        "conditions": per_mode_summaries,
    }


def run_one_condition(
    processor: CascadeAlignAttProcessor,
    *,
    wav_path: str,
    mode: str,
    chunk_ms: int,
    source_lang_code: str,
    target_lang_code: str,
) -> ConditionResult:
    processor.clear()
    audio = load_wav_raw(wav_path)
    chunk_size = int(SAMPLE_RATE * chunk_ms / 1000)
    audio_duration_ms = len(audio) * 1000.0 / SAMPLE_RATE

    word_delays_ms: list[float] = []
    word_elapsed_ms: list[float] = []
    stream_updates: list[dict[str, Any]] = []
    last_translation = ""
    last_raw = ""
    first_emit_audio_s: float | None = None
    first_emit_wallclock_s: float | None = None
    start = perf_counter()

    for start_sample in range(0, len(audio), chunk_size):
        chunk = audio[start_sample : start_sample + chunk_size].astype(np.float32)
        output = processor.process_chunk(chunk)
        current = processor.tokens_to_string(processor._emitted_units)
        audio_processed_ms = min(start_sample + chunk_size, len(audio)) * 1000.0 / SAMPLE_RATE
        wallclock_elapsed_ms = (perf_counter() - start) * 1000.0
        if output.new_tokens:
            register_translation_timestamps(
                last_raw, current, wallclock_elapsed_ms, word_elapsed_ms,
                target_lang_code=target_lang_code,
            )
            new_words = register_translation_words(
                last_translation, current, audio_processed_ms, word_delays_ms,
                target_lang_code=target_lang_code,
            )
            if first_emit_audio_s is None and current.strip():
                first_emit_audio_s = audio_processed_ms / 1000.0
                first_emit_wallclock_s = wallclock_elapsed_ms / 1000.0
            stream_updates.append({
                "update_idx": len(stream_updates),
                "audio_processed_ms": audio_processed_ms,
                "wallclock_elapsed_ms": wallclock_elapsed_ms,
                "translation_text": current,
                "new_words": new_words,
            })
            last_translation = current
            last_raw = current

    eos = processor.end_of_stream()
    final_translation = processor.tokens_to_string(processor._emitted_units)
    final_elapsed_ms = (perf_counter() - start) * 1000.0
    if eos.new_tokens or final_translation != last_translation:
        register_translation_timestamps(
            last_raw, final_translation, final_elapsed_ms, word_elapsed_ms,
            target_lang_code=target_lang_code,
        )
        eos_new_words = register_translation_words(
            last_translation, final_translation, audio_duration_ms, word_delays_ms,
            target_lang_code=target_lang_code,
        )
        stream_updates.append({
            "update_idx": len(stream_updates),
            "audio_processed_ms": audio_duration_ms,
            "wallclock_elapsed_ms": final_elapsed_ms,
            "translation_text": final_translation,
            "new_words": eos_new_words,
            "is_eos": True,
        })

    wallclock_s = perf_counter() - start
    rtf = wallclock_s / (audio_duration_ms / 1000.0) if audio_duration_ms > 0 else 0.0
    final_asr = processor.session.render_public_asr_text()
    mean_delay = (sum(word_delays_ms) / len(word_delays_ms)) if word_delays_ms else None

    normalized_elapsed = normalize_computation_aware_timestamps(word_delays_ms, word_elapsed_ms)
    prediction = prediction_text_from_target_surface(
        final_translation,
        target_lang_code=target_lang_code,
    )
    hypothesis_record = {
        "source": [Path(wav_path).name],
        "source_length": audio_duration_ms,
        "prediction": prediction,
        "delays": word_delays_ms,
        "elapsed": normalized_elapsed,
        "elapsed_wallclock_ms": word_elapsed_ms,
    }

    return ConditionResult(
        mode=mode,
        final_translation=final_translation,
        final_asr=final_asr,
        num_updates=len(stream_updates),
        rtf=rtf,
        wallclock_s=wallclock_s,
        first_emit_audio_s=first_emit_audio_s,
        first_emit_wallclock_s=first_emit_wallclock_s,
        mean_word_delay_ms=mean_delay,
        stream_updates=stream_updates,
        hypothesis_record=hypothesis_record,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wav", required=True, help="Path to the validation WAV.")
    parser.add_argument(
        "--paper-context-path",
        required=True,
        help="Path to the PaperArtifact JSON matching --wav.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-ms", type=int, default=800)
    parser.add_argument("--source", default="en")
    parser.add_argument("--target", default="de")
    parser.add_argument(
        "--max-history-utterances",
        type=int,
        default=DEFAULT_MAX_HISTORY_UTTERANCES,
        help=(
            "Number of previously committed source utterances to include in the "
            "retrieval query. Default 0, matching the simplified canonical batch runner so "
            "single-audio ablations remain comparable."
        ),
    )
    parser.add_argument(
        "--alignment-backend-name",
        default="qwen_forced",
        choices=("qwen_forced", "gemma_vllm_qk_fast"),
    )
    parser.add_argument(
        "--mt-backend-name",
        default="gemma_vllm_alignatt",
        choices=VALID_MT_BACKEND_NAMES,
        help="Default is the stable Gemma baseline; MiLMMT is the active improvement route.",
    )
    parser.add_argument("--paper-context-top-k", type=int, default=3)
    parser.add_argument("--paper-context-max-chars", type=int, default=1200)
    parser.add_argument("--paper-context-history-window-words", type=int, default=60)
    parser.add_argument("--min-start-seconds", type=float, default=2.0)
    parser.add_argument(
        "--translation-alignatt-min-source-mass",
        type=float,
        default=0.0,
        help=(
            "Provenance guard: veto any drafted target token whose "
            "source_accessible attention mass falls below this threshold. "
            "With paper_context_mode != off, tokens attending mostly to the "
            "[Paper context] region get rejected, so paper phrasing can no "
            "longer slip into the MT output. Default 0.0 (guard off)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_output_dir(args.output_dir)

    processor_config = build_processor_config(args)

    load_start = perf_counter()
    CascadeAlignAttProcessor.load_model(processor_config)
    load_ms = (perf_counter() - load_start) * 1000.0
    processor = CascadeAlignAttProcessor(processor_config)
    processor.set_source_language(args.source)
    processor.set_target_language(args.target)
    print(f"Models loaded in {load_ms:.0f} ms")

    results: list[ConditionResult] = []
    for mode in CONDITIONS:
        # Flip the mode on the live config; the bundle stays hot because
        # paper_context_path is not in _bundle_key.
        processor._runtime_config.paper_context_mode = mode
        processor.session.bundle.config = processor._runtime_config
        processor.session.config = processor._runtime_config
        processor.session.bundle.ensure_paper_context_selector()
        print(f"\n=== condition: {mode} ===")
        result = run_one_condition(
            processor,
            wav_path=args.wav,
            mode=mode,
            chunk_ms=args.chunk_ms,
            source_lang_code=args.source,
            target_lang_code=args.target,
        )
        results.append(result)
        print(
            f"  RTF={result.rtf:.3f}  wallclock={result.wallclock_s:.1f}s  "
            f"updates={result.num_updates}  first_emit_audio_s={result.first_emit_audio_s}  "
            f"mean_delay_ms={result.mean_word_delay_ms}"
        )
        print(f"  final_translation: {result.final_translation}")

    # Write artefacts per condition.
    per_mode_summaries = []
    for result in results:
        mode_dir = out_dir / result.mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(mode_dir / "stream_updates.jsonl", result.stream_updates)
        write_jsonl(mode_dir / "hypothesis.jsonl", [result.hypothesis_record])
        write_text(mode_dir / final_translation_filename(args.target), result.final_translation + "\n")
        write_text(mode_dir / final_asr_filename(args.source), result.final_asr + "\n")
        per_mode_summaries.append({
            "mode": result.mode,
            "rtf": round(result.rtf, 4),
            "wallclock_s": round(result.wallclock_s, 3),
            "num_updates": result.num_updates,
            "first_emit_audio_s": result.first_emit_audio_s,
            "first_emit_wallclock_s": result.first_emit_wallclock_s,
            "mean_word_delay_ms": (
                None if result.mean_word_delay_ms is None
                else round(result.mean_word_delay_ms, 2)
            ),
            "final_translation": result.final_translation,
            "final_asr": result.final_asr,
        })

    summary = build_summary(args=args, load_ms=load_ms, per_mode_summaries=per_mode_summaries)
    write_json(out_dir / "ablation_summary.json", summary)
    print(f"\nSummary written to {out_dir / 'ablation_summary.json'}")


if __name__ == "__main__":
    main()
