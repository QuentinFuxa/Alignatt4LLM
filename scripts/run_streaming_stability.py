#!/usr/bin/env python
"""Streaming-stability harness for the two supported alignment backends.

Simulates the cascade's per-chunk streaming pattern on one audio: at each
tick point ``t`` we slice ``audio[0:t]`` and re-run the alignment backend.
The script records the predicted word-end time for every word across
every tick, then reports:

- drift stdev per word: how much the predicted end time of the same word
  moves as more audio arrives
- max backward jump: the worst backwards revision seen for any word
- time-to-stable-word: how many ticks past a word's first emission before
  its end-time stops moving by more than a threshold
- transcript prefix stability: whether earlier words keep their identity

This harness compares the public backends that remain in the runtime:

- ``qwen_forced``
- ``gemma_onepass_qk_fast``
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cascade.alignment.base import AlignmentResult, WordAlignment
from run_alignment_single_audio import (
    build_qwen_backend,
    build_gemma_backend,
    build_runtime_config,
    load_wav,
    serialize_alignment,
)


@dataclass
class StreamingTick:
    tick_s: float
    text: str
    words: list[dict]
    diagnostics: dict | None = None


def run_streaming_ticks(
    *,
    backend,
    audio: np.ndarray,
    sample_rate: int,
    tick_starts_s: list[float],
    language: str,
    transcript_source: AlignmentResult | None = None,
    align_transcript_mode: bool = False,
) -> list[StreamingTick]:
    ticks: list[StreamingTick] = []
    for tick_s in tick_starts_s:
        sample_end = int(tick_s * sample_rate)
        if sample_end <= 0 or sample_end > len(audio):
            continue
        audio_slice = np.asarray(audio[:sample_end], dtype=np.float32)
        if align_transcript_mode:
            if transcript_source is None:
                raise ValueError("align_transcript_mode needs a transcript_source")
            full_text = transcript_source.text
            result = backend.align_transcript(
                audio_slice,
                sample_rate=sample_rate,
                language=language,
                transcript=full_text,
            )
        else:
            result = backend.transcribe_and_align(
                audio_slice, sample_rate=sample_rate, language=language
            )
        if result is None:
            continue
        diagnostics = dict(getattr(result, "diagnostics", {}) or {})
        # Drop verbose per-token arrays before serializing — they bloat the
        # tick log and the per-tick aggregate metrics don't read them.
        diagnostics.pop("aligned_audio_positions", None)
        ticks.append(
            StreamingTick(
                tick_s=float(tick_s),
                text=result.text,
                words=[asdict(w) for w in result.words],
                diagnostics=diagnostics,
            )
        )
        gemma_used = diagnostics.get("gemma_alignment_used")
        fallback_reason = diagnostics.get("fallback_reason")
        usage_tag = ""
        if gemma_used is not None:
            usage_tag = (
                f"  gemma_used={gemma_used}"
                + (f" reason={fallback_reason}" if fallback_reason else "")
            )
        print(
            f"[{backend.name}] tick={tick_s:5.2f}s  "
            f"words={len(result.words):2d}  text='{result.text[:60]}...'"
            f"{usage_tag}"
        )
    return ticks


def compute_streaming_metrics(
    ticks: list[StreamingTick],
    *,
    stability_threshold_s: float = 0.05,
) -> dict:
    """Report per-word drift + transcript prefix stability across ticks.

    A word is identified by its position within the emitted transcript; we
    track the predicted end-time of position ``i`` across all ticks that
    emit at least ``i+1`` words. This is the same notion of "word
    identity" the cascade uses (the ASR hypothesis is monotone enough
    that position is a stable key within the committed prefix).
    """
    if len(ticks) < 2:
        return {"num_ticks": len(ticks), "note": "need at least 2 ticks for drift"}

    # Per-word-position series of (tick_s, end_time)
    per_position: dict[int, list[tuple[float, float, str]]] = {}
    for tick in ticks:
        for pos, word in enumerate(tick.words):
            per_position.setdefault(pos, []).append(
                (tick.tick_s, float(word["end_time"]), str(word["text"]))
            )

    per_word_stats: list[dict] = []
    backward_jumps: list[float] = []
    time_to_stable: list[float] = []
    identity_changes = 0

    for pos, series in sorted(per_position.items()):
        if len(series) < 2:
            continue
        end_times = [e for _, e, _ in series]
        tick_times = [t for t, _, _ in series]
        surfaces = [s for _, _, s in series]

        stdev = statistics.stdev(end_times) if len(end_times) >= 2 else 0.0
        drift_range = max(end_times) - min(end_times)
        max_backward = max(
            (end_times[i] - end_times[i + 1] for i in range(len(end_times) - 1)),
            default=0.0,
        )
        if max_backward > 0:
            backward_jumps.append(max_backward)

        # Time to stable: smallest k such that all end_times[k:] stay within
        # stability_threshold of end_times[-1].
        final_end = end_times[-1]
        stable_from_idx = len(end_times) - 1
        for idx in range(len(end_times)):
            if all(
                abs(end_times[j] - final_end) <= stability_threshold_s
                for j in range(idx, len(end_times))
            ):
                stable_from_idx = idx
                break
        time_to_stable.append(tick_times[stable_from_idx] - tick_times[0])

        # Identity change: position keeps the same surface?
        unique_surfaces = set(surfaces)
        if len(unique_surfaces) > 1:
            identity_changes += 1

        per_word_stats.append(
            {
                "position": pos,
                "surface_modes": sorted(unique_surfaces),
                "stdev_end_s": stdev,
                "drift_range_s": drift_range,
                "max_backward_jump_s": max_backward,
                "time_to_stable_s": tick_times[stable_from_idx] - tick_times[0],
                "end_time_series": end_times,
                "tick_series": tick_times,
            }
        )

    stdevs = [w["stdev_end_s"] for w in per_word_stats if w["stdev_end_s"] is not None]
    drift_ranges = [w["drift_range_s"] for w in per_word_stats]

    # Some backends may emit extra per-tick diagnostics describing how the
    # alignment path was obtained; aggregate them when present.
    fallback_total = 0
    gemma_used_total = 0
    fallback_reasons: dict[str, int] = {}
    fallback_aware_ticks = 0
    for tick in ticks:
        diag = tick.diagnostics or {}
        if "gemma_alignment_used" not in diag:
            continue
        fallback_aware_ticks += 1
        if diag.get("gemma_alignment_used"):
            gemma_used_total += 1
        else:
            fallback_total += 1
            reason = str(diag.get("fallback_reason") or "unknown")
            fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
    fallback_rate = (
        fallback_total / fallback_aware_ticks if fallback_aware_ticks else None
    )

    return {
        "num_ticks": len(ticks),
        "num_words_tracked": len(per_word_stats),
        "identity_changes": identity_changes,
        "mean_stdev_end_s": float(statistics.mean(stdevs)) if stdevs else 0.0,
        "median_stdev_end_s": float(statistics.median(stdevs)) if stdevs else 0.0,
        "mean_drift_range_s": float(statistics.mean(drift_ranges)) if drift_ranges else 0.0,
        "max_drift_range_s": float(max(drift_ranges)) if drift_ranges else 0.0,
        "num_backward_jumps": len(backward_jumps),
        "max_backward_jump_s": float(max(backward_jumps)) if backward_jumps else 0.0,
        "mean_time_to_stable_s": (
            float(statistics.mean(time_to_stable)) if time_to_stable else 0.0
        ),
        "stability_threshold_s": stability_threshold_s,
        "fallback_aware_ticks": fallback_aware_ticks,
        "gemma_used_ticks": gemma_used_total,
        "fallback_ticks": fallback_total,
        "fallback_rate": fallback_rate,
        "fallback_reasons": fallback_reasons,
        "per_word_stats": per_word_stats,
    }


def cmd_run(args: argparse.Namespace) -> None:
    audio, sr = load_wav(args.wav)
    duration_s = len(audio) / sr
    tick_starts_s = [
        float(t)
        for t in np.arange(args.tick_start_s, min(duration_s, args.tick_stop_s) + 0.001, args.tick_step_s)
    ]
    print(
        f"[info] {args.wav}: duration={duration_s:.2f}s  "
        f"ticks={len(tick_starts_s)} from {tick_starts_s[0]:.1f}s to {tick_starts_s[-1]:.1f}s"
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.qwen_baseline:
        print("\n[info] loading Qwen baseline")
        qwen = build_qwen_backend()
        qwen_ticks = run_streaming_ticks(
            backend=qwen,
            audio=audio,
            sample_rate=sr,
            tick_starts_s=tick_starts_s,
            language=args.language,
        )
        qwen_metrics = compute_streaming_metrics(qwen_ticks)
        _write_report(out_dir / f"{args.tag}_qwen_ticks.json", qwen_ticks, qwen_metrics)

    if args.gemma_onepass:
        print("\n[info] loading Gemma one-pass qk_fast frontend")
        gemma = build_gemma_backend(heads_path=args.heads_path, top_k=args.top_k)
        gemma_ticks = run_streaming_ticks(
            backend=gemma,
            audio=audio,
            sample_rate=sr,
            tick_starts_s=tick_starts_s,
            language=args.language,
        )
        gemma_metrics = compute_streaming_metrics(gemma_ticks)
        _write_report(
            out_dir / f"{args.tag}_gemma_onepass_ticks.json",
            gemma_ticks,
            gemma_metrics,
        )


def _write_report(path: Path, ticks, metrics) -> None:
    payload = {
        "metrics": metrics,
        "ticks": [asdict(t) if hasattr(t, "__dict__") else t for t in ticks],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    short = {
        k: v for k, v in metrics.items()
        if k not in {"per_word_stats"}
    }
    print(f"\n[metrics @ {path.name}] {json.dumps(short, indent=2)}")


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wav", required=True)
    parser.add_argument("--output-dir", default="tmp/alignment_research")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--language", default="English")
    parser.add_argument("--tick-start-s", type=float, default=5.0)
    parser.add_argument("--tick-stop-s", type=float, default=25.0)
    parser.add_argument("--tick-step-s", type=float, default=2.0)
    parser.add_argument("--heads-path", default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--qwen-baseline", action="store_true")
    parser.add_argument("--gemma-onepass", action="store_true")
    parser.set_defaults(func=cmd_run)
    return parser


def main(argv=None) -> None:
    args = build_cli().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
