#!/usr/bin/env python3
"""Per-audio ASR comparison on the 21-clip dev-set.

Loads each alignment backend once and streams every audio through it,
producing per-audio WER/CER/boundary-lag payloads. Mirrors the single-
audio harness in ``compare_asr_full_audio.py`` but amortizes model-load
cost across the full dev-set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from cascade.audio import load_audio_mono_16khz
from cascade.runtime import CascadeRuntimeConfig, LoadedModelBundle
from scripts.compare_asr_full_audio import (
    SAMPLE_RATE,
    build_run_id,
    build_run_label,
    compare_backend_to_reference,
    label_for_backend,
    load_reference_segments,
    maybe_warmup_backend,
)


def stream_audio_through_session(
    *,
    bundle: LoadedModelBundle,
    wav_path: str,
    chunk_ms: int,
    min_start_seconds: float,
) -> dict:
    session = bundle.new_session()
    audio = load_audio_mono_16khz(wav_path)
    chunk_size = int(SAMPLE_RATE * chunk_ms / 1000)
    audio_duration_s = len(audio) / SAMPLE_RATE

    commit_events: list[dict] = []
    stream_trace: list[dict] = []
    processing_start = perf_counter()
    last_commit_count = len(session.state.utt_sources) - 1
    last_trace_len = 0
    update_count = 0
    last_chunk_idx = 0

    for chunk_idx, stop_sample in enumerate(
        range(chunk_size, len(audio) + chunk_size, chunk_size),
        start=1,
    ):
        stop_sample = min(stop_sample, len(audio))
        last_chunk_idx = chunk_idx
        session.state.source = np.asarray(audio[:stop_sample], dtype=np.float32)
        if session.current_audio_seconds() < min_start_seconds:
            continue

        current_asr = session.transcribe_audio()
        trace_snapshot = session.asr_stream_trace()
        for row in trace_snapshot[last_trace_len:]:
            enriched = dict(row)
            enriched["chunk_idx"] = int(chunk_idx)
            enriched["wallclock_s"] = perf_counter() - processing_start
            stream_trace.append(enriched)
        last_trace_len = len(trace_snapshot)
        if current_asr:
            update_count += 1

        commit_count = len(session.state.utt_sources) - 1
        if commit_count > last_commit_count:
            wallclock_s = perf_counter() - processing_start
            for segment_idx in range(last_commit_count + 1, commit_count + 1):
                commit_events.append(
                    {
                        "segment_index": int(segment_idx),
                        "text": session.state.utt_sources[segment_idx].strip(),
                        "end_time_s": session.state.utt_timestamps[segment_idx] / SAMPLE_RATE,
                        "audio_processed_s": stop_sample / SAMPLE_RATE,
                        "wallclock_s": wallclock_s,
                        "is_eos_flush": False,
                    }
                )
            last_commit_count = commit_count

    session.state.source = np.asarray(audio, dtype=np.float32)
    final_asr = session.transcribe_audio(is_final_chunk=True) or session.render_public_asr_text()
    final_wallclock_s = perf_counter() - processing_start
    trace_snapshot = session.asr_stream_trace()
    for row in trace_snapshot[last_trace_len:]:
        enriched = dict(row)
        enriched["chunk_idx"] = int(max(1, last_chunk_idx))
        enriched["wallclock_s"] = final_wallclock_s
        stream_trace.append(enriched)
    final_commit_count = len(session.state.utt_sources) - 1
    if final_commit_count > last_commit_count:
        for segment_idx in range(last_commit_count + 1, final_commit_count + 1):
            commit_events.append(
                {
                    "segment_index": int(segment_idx),
                    "text": session.state.utt_sources[segment_idx].strip(),
                    "end_time_s": session.state.utt_timestamps[segment_idx] / SAMPLE_RATE,
                    "audio_processed_s": audio_duration_s,
                    "wallclock_s": final_wallclock_s,
                    "is_eos_flush": True,
                }
            )

    committed_texts = [
        segment.strip()
        for segment in session.state.utt_sources[1:]
        if segment.strip()
    ]
    return {
        "wav_path": wav_path,
        "audio_duration_s": audio_duration_s,
        "processing_s": final_wallclock_s,
        "rtf_wallclock": final_wallclock_s / max(audio_duration_s, 1e-9),
        "update_count": int(update_count),
        "committed_segment_count": len(committed_texts),
        "final_asr_text": final_asr,
        "committed_texts": committed_texts,
        "commit_events": commit_events,
        "stream_trace": stream_trace,
    }


def run_backend_over_dataset(
    *,
    backend_name: str,
    wavs: list[str],
    segments_path: str,
    source_ref_path: str,
    chunk_ms: int,
    min_start_seconds: float,
    asr_alignatt_frame_threshold: int,
    asr_alignatt_rewind_threshold: int,
    gemma_audio_alignment_top_k_heads: int | None,
    warmup_seconds: float,
    match_tolerance_words: int,
    output_dir: Path,
) -> list[dict]:
    config = CascadeRuntimeConfig(
        source_lang="English",
        target_lang="German",
        alignment_backend_name=backend_name,
    )
    config.min_start_seconds = float(min_start_seconds)
    config.asr_alignatt_frame_threshold = int(asr_alignatt_frame_threshold)
    config.asr_alignatt_rewind_threshold = int(asr_alignatt_rewind_threshold)
    if gemma_audio_alignment_top_k_heads is not None:
        config.gemma_audio_alignment_top_k_heads = int(gemma_audio_alignment_top_k_heads)

    bundle = LoadedModelBundle(config)
    load_start = perf_counter()
    alignment_backend = bundle.ensure_alignment_backend()
    load_s = perf_counter() - load_start
    warmup_s = maybe_warmup_backend(
        backend_name=backend_name,
        alignment_backend=alignment_backend,
        warmup_seconds=warmup_seconds,
    )
    print(f"[{backend_name}] loaded in {load_s:.1f}s; warmup {warmup_s:.1f}s")

    per_audio_dir = output_dir / backend_name
    per_audio_dir.mkdir(parents=True, exist_ok=True)
    run_id = build_run_id(
        backend_name=backend_name,
        eval_mode="streaming_full",
        gemma_vllm_path_mode="shipping",
        gemma_sampling_mode="shipping",
        asr_alignatt_frame_threshold=asr_alignatt_frame_threshold,
        asr_alignatt_rewind_threshold=asr_alignatt_rewind_threshold,
    )
    run_label = build_run_label(
        backend_name=backend_name,
        run_id=run_id,
    )

    rows: list[dict] = []
    for idx, wav_path in enumerate(wavs, start=1):
        wav_name = Path(wav_path).name
        print(f"[{backend_name}] ({idx}/{len(wavs)}) {wav_name}", flush=True)
        per_wav_start = perf_counter()
        reference_payload = load_reference_segments(
            wav_path=wav_path,
            segments_path=segments_path,
            source_ref_path=source_ref_path,
        )
        payload = stream_audio_through_session(
            bundle=bundle,
            wav_path=wav_path,
            chunk_ms=chunk_ms,
            min_start_seconds=min_start_seconds,
        )
        payload.update(
            {
                "run_id": run_id,
                "run_label": run_label,
                "backend_name": backend_name,
                "eval_mode": "streaming_full",
                "reference_wav_name": reference_payload["reference_wav_name"],
                "chunk_ms": int(chunk_ms),
                "min_start_seconds": float(min_start_seconds),
                "asr_alignatt_frame_threshold": int(asr_alignatt_frame_threshold),
                "asr_alignatt_rewind_threshold": int(asr_alignatt_rewind_threshold),
                "load_s": load_s,
                "warmup_s": warmup_s,
            }
        )
        payload["metrics"] = compare_backend_to_reference(
            backend_payload=payload,
            reference_payload=reference_payload,
            tolerance_words=match_tolerance_words,
        )
        (per_audio_dir / f"{Path(wav_name).stem}.json").write_text(
            json.dumps(payload, indent=2)
        )
        wall = perf_counter() - per_wav_start
        m = payload["metrics"]
        matched = len(m.get("lag_points", []) or [])
        mean_lag = (m.get("lag_summary") or {}).get("mean_s") or 0.0
        rows.append(
            {
                "wav_name": wav_name,
                "backend": backend_name,
                "audio_duration_s": payload["audio_duration_s"],
                "processing_s": payload["processing_s"],
                "rtf_wallclock": payload["rtf_wallclock"],
                "wer": m["wer"],
                "cer": m["cer"],
                "matched_lag_count": matched,
                "matched_mean_lag_s": mean_lag,
                "per_wav_wallclock_s": wall,
            }
        )
        print(
            f"    wer={m['wer']:.3f} cer={m['cer']:.3f} "
            f"matched_lag_n={matched} mean_lag={mean_lag:.2f}s "
            f"wall={wall:.1f}s",
            flush=True,
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["qwen_forced", "gemma_vllm_qk_fast"])
    parser.add_argument("--wavs", nargs="+", required=True)
    parser.add_argument("--segments", default="dev-set/audio-segments.yaml")
    parser.add_argument("--source-ref", default="dev-set/ref/en.txt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-ms", type=int, default=800)
    parser.add_argument("--min-start-seconds", type=float, default=2.0)
    parser.add_argument("--asr-alignatt-frame-threshold", type=int, default=4)
    parser.add_argument("--asr-alignatt-rewind-threshold", type=int, default=200)
    parser.add_argument("--gemma-audio-alignment-top-k-heads", type=int, default=None)
    parser.add_argument("--gemma-warmup-seconds", type=float, default=18.0)
    parser.add_argument("--match-tolerance-words", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = run_backend_over_dataset(
        backend_name=args.backend,
        wavs=list(args.wavs),
        segments_path=args.segments,
        source_ref_path=args.source_ref,
        chunk_ms=args.chunk_ms,
        min_start_seconds=args.min_start_seconds,
        asr_alignatt_frame_threshold=args.asr_alignatt_frame_threshold,
        asr_alignatt_rewind_threshold=args.asr_alignatt_rewind_threshold,
        gemma_audio_alignment_top_k_heads=args.gemma_audio_alignment_top_k_heads,
        warmup_seconds=args.gemma_warmup_seconds,
        match_tolerance_words=args.match_tolerance_words,
        output_dir=output_dir,
    )

    summary_path = output_dir / f"{args.backend}__summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "backend_name": args.backend,
                "backend_label": label_for_backend(args.backend),
                "run_args": {k: v for k, v in vars(args).items() if k != "wavs"},
                "wavs": list(args.wavs),
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
