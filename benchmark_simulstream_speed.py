#!/usr/bin/env python3
"""SimulStream speed harness.

Runs audio through the real CascadeAlignAttProcessor and reports:
  - total wallclock
  - real-time factor (RTF)
  - per-chunk processing mean / p95 / max
  - number of emitted updates
  - peak GPU memory (if available)

Usage (from .venv-inference):
    python benchmark_simulstream_speed.py --wav test-set/audio/ccpXHNfaoy.wav
    python benchmark_simulstream_speed.py --wav test-set/audio/ccpXHNfaoy.wav --chunk-ms 800 --target de
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from cascade_simulstream_processor import CascadeAlignAttProcessor
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SimulStream speed benchmark.")
    parser.add_argument("--wav", required=True, help="Input WAV file path.")
    parser.add_argument("--chunk-ms", default=450, type=int, help="Chunk size in ms.")
    parser.add_argument("--source", default="en", help="Source language code.")
    parser.add_argument("--target", default="de", help="Target language code.")
    parser.add_argument("--min-start-seconds", default=2.0, type=float)
    parser.add_argument("--max-history-utterances", default=1, type=int)
    parser.add_argument("--partial-max-new-tokens", default=16, type=int)
    parser.add_argument("--partial-followup-max-new-tokens", default=8, type=int)
    parser.add_argument("--translation-alignatt-min-source-mass", default=0.0, type=float)
    parser.add_argument("--output-json", default=None, help="Write results to JSON file.")
    return parser.parse_args()


def gpu_peak_memory_mb() -> float | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return None


def main() -> None:
    args = parse_args()

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
    )

    print(f"Loading models for {args.source}->{args.target} ...")
    load_start = perf_counter()
    CascadeAlignAttProcessor.load_model(processor_config)
    load_ms = (perf_counter() - load_start) * 1000.0
    print(f"Models loaded in {load_ms:.0f} ms")

    processor = CascadeAlignAttProcessor(processor_config)
    processor.set_source_language(args.source)
    processor.set_target_language(args.target)
    processor.clear()

    audio = load_wav_raw(args.wav)
    chunk_size = int(SAMPLE_RATE * args.chunk_ms / 1000)
    audio_duration_s = len(audio) / SAMPLE_RATE
    print(f"Audio: {audio_duration_s:.1f}s, chunk: {args.chunk_ms}ms ({chunk_size} samples)")

    chunk_times_ms: list[float] = []
    num_updates_with_output = 0
    total_new_tokens = 0
    total_deleted_tokens = 0

    run_start = perf_counter()
    for start_sample in range(0, len(audio), chunk_size):
        chunk = audio[start_sample : start_sample + chunk_size].astype(np.float32)
        chunk_start = perf_counter()
        output = processor.process_chunk(chunk)
        chunk_ms = (perf_counter() - chunk_start) * 1000.0
        chunk_times_ms.append(chunk_ms)
        if output.new_tokens:
            num_updates_with_output += 1
            total_new_tokens += len(output.new_tokens)
        total_deleted_tokens += len(output.deleted_tokens)

    eos_start = perf_counter()
    eos_output = processor.end_of_stream()
    eos_ms = (perf_counter() - eos_start) * 1000.0
    total_wallclock_s = perf_counter() - run_start

    if eos_output.new_tokens:
        num_updates_with_output += 1
        total_new_tokens += len(eos_output.new_tokens)
    total_deleted_tokens += len(eos_output.deleted_tokens)

    sorted_times = sorted(chunk_times_ms)
    n_chunks = len(chunk_times_ms)
    mean_chunk_ms = sum(chunk_times_ms) / n_chunks if n_chunks else 0.0
    p95_idx = min(n_chunks - 1, int(n_chunks * 0.95))
    p95_chunk_ms = sorted_times[p95_idx] if n_chunks else 0.0
    max_chunk_ms = sorted_times[-1] if n_chunks else 0.0
    rtf = total_wallclock_s / audio_duration_s if audio_duration_s > 0 else float("inf")

    peak_gpu_mb = gpu_peak_memory_mb()

    results = {
        "wav": args.wav,
        "source": args.source,
        "target": args.target,
        "chunk_ms": args.chunk_ms,
        "audio_duration_s": round(audio_duration_s, 2),
        "total_wallclock_s": round(total_wallclock_s, 2),
        "rtf": round(rtf, 4),
        "num_chunks": n_chunks,
        "mean_chunk_ms": round(mean_chunk_ms, 2),
        "p95_chunk_ms": round(p95_chunk_ms, 2),
        "max_chunk_ms": round(max_chunk_ms, 2),
        "eos_ms": round(eos_ms, 2),
        "num_updates_with_output": num_updates_with_output,
        "total_new_tokens": total_new_tokens,
        "total_deleted_tokens": total_deleted_tokens,
        "peak_gpu_mb": round(peak_gpu_mb, 1) if peak_gpu_mb is not None else None,
    }

    print()
    print("=" * 60)
    print("SimulStream Speed Benchmark Results")
    print("=" * 60)
    print(f"  Audio duration:     {audio_duration_s:.1f} s")
    print(f"  Total wallclock:    {total_wallclock_s:.2f} s")
    print(f"  RTF:                {rtf:.4f}")
    print(f"  Chunks processed:   {n_chunks}")
    print(f"  Mean chunk time:    {mean_chunk_ms:.1f} ms")
    print(f"  P95 chunk time:     {p95_chunk_ms:.1f} ms")
    print(f"  Max chunk time:     {max_chunk_ms:.1f} ms")
    print(f"  EOS time:           {eos_ms:.1f} ms")
    print(f"  Updates w/ output:  {num_updates_with_output}")
    print(f"  Total new tokens:   {total_new_tokens}")
    print(f"  Total deleted:      {total_deleted_tokens}")
    if peak_gpu_mb is not None:
        print(f"  Peak GPU memory:    {peak_gpu_mb:.0f} MB")
    print("=" * 60)

    chunk_interval_ms = float(args.chunk_ms)
    if p95_chunk_ms > chunk_interval_ms:
        print(f"  WARNING: p95 ({p95_chunk_ms:.0f}ms) exceeds chunk interval ({chunk_interval_ms:.0f}ms)")
    if rtf >= 1.0:
        print(f"  WARNING: RTF >= 1.0 — cannot run in real time")
    elif rtf > 0.6:
        print(f"  NOTE: RTF > 0.6 — near real-time boundary")

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nResults written to {args.output_json}")


if __name__ == "__main__":
    main()
