#!/usr/bin/env python
"""Unified audit harness for PLAN.md Phases 2–4.

Runs the three decisive experiments that determine whether the hybrid
front-end (Qwen3-ASR text + Gemma attention timings) should become the
research baseline:

  Phase 2 — Hybrid fallback audit on one real talk
  Phase 3 — Small robustness check across 3–5 clips
  Phase 4 — Cascade-level comparison: qwen vs hybrid

Each phase can be run independently or all together. Results are written
to ``tmp/hybrid_audit/`` and a summary note is printed at the end.

Requires the ``.venv-inference`` kernel with models hot in memory.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import wave
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np


HEADS_PATH = "assets/attention_heads/audio_alignment_heads_google_gemma-4-E4B-it_en_forced.json"
OUTPUT_DIR = "tmp/hybrid_audit"
SAMPLE_RATE = 16000


def load_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wav:
        sr = wav.getframerate()
        width = wav.getsampwidth()
        ch = wav.getnchannels()
        raw = wav.readframes(wav.getnframes())
    if width != 2:
        raise ValueError("Only 16-bit PCM WAV is supported.")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    if sr != SAMPLE_RATE:
        duration = len(audio) / sr
        new_length = int(duration * SAMPLE_RATE)
        old_times = np.linspace(0.0, duration, num=len(audio), endpoint=False)
        new_times = np.linspace(0.0, duration, num=new_length, endpoint=False)
        audio = np.interp(new_times, old_times, audio).astype(np.float32)
        sr = SAMPLE_RATE
    return audio, sr


def extract_clip(audio: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    start_sample = int(start_s * sr)
    end_sample = min(int(end_s * sr), len(audio))
    return audio[start_sample:end_sample]


# ---------------------------------------------------------------------------
# Phase 2: Hybrid Fallback Audit
# ---------------------------------------------------------------------------

def run_fallback_audit(
    wav_path: str,
    *,
    heads_path: str,
    tick_start_s: float = 5.0,
    tick_stop_s: float = 60.0,
    tick_step_s: float = 2.0,
    output_dir: str = OUTPUT_DIR,
) -> dict:
    from run_streaming_stability import (
        run_streaming_ticks,
        compute_streaming_metrics,
    )
    from run_alignment_single_audio import build_qwen_backend, build_gemma_backend
    from hybrid_alignment_backend import HybridQwenAsrGemmaAlignerBackend

    audio, sr = load_wav(wav_path)
    duration_s = len(audio) / sr
    tick_stop_s = min(tick_stop_s, duration_s)
    tick_starts_s = [
        float(t)
        for t in np.arange(tick_start_s, tick_stop_s + 0.001, tick_step_s)
    ]
    print(f"\n{'='*60}")
    print(f"PHASE 2: Hybrid Fallback Audit")
    print(f"  wav: {wav_path} ({duration_s:.1f}s)")
    print(f"  ticks: {len(tick_starts_s)} from {tick_starts_s[0]:.1f}s to {tick_starts_s[-1]:.1f}s")
    print(f"{'='*60}")

    qwen = build_qwen_backend()
    gemma = build_gemma_backend(heads_path=heads_path, top_k=8)
    hybrid = HybridQwenAsrGemmaAlignerBackend(
        asr_backend=qwen, gemma_backend=gemma
    )

    hybrid_ticks = run_streaming_ticks(
        backend=hybrid,
        audio=audio,
        sample_rate=sr,
        tick_starts_s=tick_starts_s,
        language="English",
    )
    metrics = compute_streaming_metrics(hybrid_ticks)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "phase2_fallback_audit.json"
    payload = {
        "wav_path": wav_path,
        "heads_path": heads_path,
        "metrics": {k: v for k, v in metrics.items() if k != "per_word_stats"},
        "ticks": [asdict(t) if hasattr(t, "__dataclass_fields__") else t for t in hybrid_ticks],
    }
    report_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"\n[Phase 2 Summary]")
    print(f"  fallback_aware_ticks: {metrics.get('fallback_aware_ticks')}")
    print(f"  gemma_used_ticks:     {metrics.get('gemma_used_ticks')}")
    print(f"  fallback_ticks:       {metrics.get('fallback_ticks')}")
    print(f"  fallback_rate:        {metrics.get('fallback_rate')}")
    print(f"  fallback_reasons:     {metrics.get('fallback_reasons')}")
    print(f"  mean_stdev_end_s:     {metrics.get('mean_stdev_end_s', 0):.4f}")
    print(f"  mean_time_to_stable:  {metrics.get('mean_time_to_stable_s', 0):.2f}s")
    print(f"  Report: {report_path}")
    return metrics


# ---------------------------------------------------------------------------
# Phase 3: Small Robustness Check
# ---------------------------------------------------------------------------

ROBUSTNESS_CLIPS = [
    {
        "tag": "smoke18",
        "wav": "tmp/alignatt_smoke18.wav",
        "start_s": 0.0,
        "end_s": 18.0,
        "description": "Calibration clip (Siyu Yuan, Fudan, Chinese accent)",
    },
    {
        "tag": "ccp_30_48",
        "wav": "test-set/audio/ccpXHNfaoy.wav",
        "start_s": 30.0,
        "end_s": 48.0,
        "description": "Same talk, different content (30-48s)",
    },
    {
        "tag": "ccp_60_78",
        "wav": "test-set/audio/ccpXHNfaoy.wav",
        "start_s": 60.0,
        "end_s": 78.0,
        "description": "Same talk, later section (60-78s)",
    },
    {
        "tag": "talk2_5_23",
        "wav": "test-set/audio/DyXpuURBMP.wav",
        "start_s": 5.0,
        "end_s": 23.0,
        "description": "Different talk/speaker (DyXpuURBMP 5-23s)",
    },
    {
        "tag": "talk3_5_23",
        "wav": "test-set/audio/ERmKpJPPDc.wav",
        "start_s": 5.0,
        "end_s": 23.0,
        "description": "Different talk/speaker (ERmKpJPPDc 5-23s)",
    },
]


def run_robustness_check(
    *,
    heads_path: str,
    output_dir: str = OUTPUT_DIR,
    clips: list[dict] | None = None,
) -> list[dict]:
    from run_alignment_single_audio import (
        build_qwen_backend,
        build_gemma_backend,
        compare_alignments,
        serialize_alignment,
    )

    clips = clips or ROBUSTNESS_CLIPS
    print(f"\n{'='*60}")
    print(f"PHASE 3: Small Robustness Check ({len(clips)} clips)")
    print(f"{'='*60}")

    qwen = build_qwen_backend()
    gemma = build_gemma_backend(heads_path=heads_path, top_k=8)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []

    for clip in clips:
        tag = clip["tag"]
        print(f"\n--- {tag}: {clip['description']} ---")

        if clip["wav"] == "tmp/alignatt_smoke18.wav" and clip["start_s"] == 0.0:
            audio, sr = load_wav(clip["wav"])
        else:
            full_audio, sr = load_wav(clip["wav"])
            audio = extract_clip(full_audio, sr, clip["start_s"], clip["end_s"])

        duration_s = len(audio) / sr
        print(f"  audio: {duration_s:.1f}s")

        qwen_result = qwen.transcribe_and_align(audio, sample_rate=sr, language="English")
        if qwen_result is None:
            print(f"  SKIP: Qwen returned no result")
            continue

        gemma_result = gemma.align_transcript(
            audio,
            sample_rate=sr,
            language="English",
            transcript=qwen_result.text,
        )
        if gemma_result is None:
            print(f"  SKIP: Gemma returned no result")
            continue

        report = compare_alignments(qwen_result, gemma_result)
        mono = gemma_result.diagnostics.get("monotonicity", None)
        report["monotonicity"] = mono
        report["tag"] = tag
        report["description"] = clip["description"]
        report["duration_s"] = duration_s
        results.append(report)

        mae = report.get("word_end_mae_seconds", None)
        median = report.get("word_end_median_error_seconds", None)
        p90 = report.get("word_end_p90_error_seconds", None)
        print(f"  words: {report['paired_words']}")
        print(f"  MAE:   {mae*1000:.0f} ms" if mae else "  MAE:   N/A")
        print(f"  Med:   {median*1000:.0f} ms" if median else "  Med:   N/A")
        print(f"  P90:   {p90*1000:.0f} ms" if p90 else "  P90:   N/A")
        print(f"  Mono:  {mono:.3f}" if mono else "  Mono:  N/A")

        bundle_path = out / f"phase3_{tag}.json"
        bundle_path.write_text(json.dumps({
            "tag": tag,
            "clip": clip,
            "qwen": serialize_alignment(qwen_result),
            "gemma": serialize_alignment(gemma_result),
            "comparison": {k: v for k, v in report.items() if k != "per_word_end_errors_seconds"},
        }, indent=2, default=str), encoding="utf-8")

    summary_path = out / "phase3_robustness_summary.json"
    summary = [{k: v for k, v in r.items() if k != "per_word_end_errors_seconds"} for r in results]
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n[Phase 3 Summary Table]")
    print(f"{'Tag':<15} {'Words':>5} {'MAE ms':>7} {'Med ms':>7} {'P90 ms':>7} {'Mono':>6}")
    print("-" * 55)
    for r in results:
        mae = r.get("word_end_mae_seconds")
        med = r.get("word_end_median_error_seconds")
        p90 = r.get("word_end_p90_error_seconds")
        mono = r.get("monotonicity")
        print(
            f"{r['tag']:<15} {r['paired_words']:>5} "
            f"{mae*1000:>7.0f} {med*1000:>7.0f} {p90*1000:>7.0f} "
            f"{mono:>6.3f}" if all(x is not None for x in [mae, med, p90, mono])
            else f"{r['tag']:<15} {r['paired_words']:>5}     N/A"
        )
    print(f"  Summary: {summary_path}")
    return results


# ---------------------------------------------------------------------------
# Phase 4: Cascade-Level Comparison
# ---------------------------------------------------------------------------

def run_cascade_comparison(
    wav_path: str,
    *,
    output_dir: str = OUTPUT_DIR,
    chunk_ms: int = 960,
) -> dict:
    print(f"\n{'='*60}")
    print(f"PHASE 4: Cascade-Level Comparison")
    print(f"  wav: {wav_path}")
    print(f"{'='*60}")

    from qwen3asr_gemma_cascade_core import (
        run_stream_to_artifacts, config, load_models, clear_state,
        build_alignment_backend,
    )
    import qwen3asr_gemma_cascade_core as core

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Run each backend in a subprocess to avoid OOM from accumulating
    # models in the same GPU. The cascade loads Qwen ASR (vLLM ~7GB) +
    # Gemma MT (~10GB); the hybrid path adds Gemma aligner (~10GB) on top.
    # Running sequentially in subprocesses ensures only one set is live.
    import subprocess

    python = sys.executable
    qwen_dir = str(out / "cascade_qwen")
    hybrid_dir = str(out / "cascade_hybrid")

    print("\n--- Running Qwen baseline cascade ---")
    _run_cascade_subprocess(python, wav_path, "qwen", qwen_dir, chunk_ms)

    print("\n--- Running Hybrid cascade ---")
    _run_cascade_subprocess(python, wav_path, "hybrid_qwen_asr_gemma_aligner", hybrid_dir, chunk_ms)

    comparison = _build_cascade_comparison(wav_path, qwen_dir, hybrid_dir, out)
    return comparison


def _run_cascade_subprocess(python: str, wav_path: str, backend: str, output_dir: str, chunk_ms: int) -> None:
    import subprocess

    script = (
        "import sys; sys.path.insert(0, '.');"
        "from qwen3asr_gemma_cascade_core import config, run_stream;"
        f"config.alignment_backend_name = {backend!r};"
        f"run_stream({wav_path!r}, chunk_ms={chunk_ms}, output_dir={output_dir!r})"
    )
    result = subprocess.run(
        [python, "-c", script],
        capture_output=False,
        timeout=1800,
    )
    if result.returncode != 0:
        print(f"  WARNING: {backend} cascade exited with code {result.returncode}")


def _build_cascade_comparison(wav_path: str, qwen_dir: str, hybrid_dir: str, out: Path) -> dict:
    qwen_manifest = _load_manifest(qwen_dir)
    hybrid_manifest = _load_manifest(hybrid_dir)

    comparison = {"wav_path": wav_path}
    for label, manifest in [("qwen", qwen_manifest), ("hybrid", hybrid_manifest)]:
        if manifest is None:
            comparison[label] = {"error": "cascade run failed or produced no artifacts"}
            continue
        comparison[label] = {
            "final_asr": manifest.get("final_asr_text", ""),
            "final_translation": manifest.get("final_translation_text", ""),
            "num_updates": manifest.get("num_updates", 0),
            "mean_word_delay_ms": _safe_mean(manifest.get("translation_word_delays_ms")),
            "median_word_delay_ms": _safe_median(manifest.get("translation_word_delays_ms")),
        }

    comparison_path = out / "phase4_cascade_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")

    print(f"\n[Phase 4 Summary]")
    for label in ["qwen", "hybrid"]:
        c = comparison.get(label, {})
        if "error" in c:
            print(f"  {label}: {c['error']}")
            continue
        print(f"  {label}:")
        print(f"    updates:          {c.get('num_updates', 'N/A')}")
        md = c.get("mean_word_delay_ms")
        print(f"    mean delay:       {md:.0f} ms" if md else "    mean delay:       N/A")
        md2 = c.get("median_word_delay_ms")
        print(f"    median delay:     {md2:.0f} ms" if md2 else "    median delay:     N/A")
    print(f"  Comparison: {comparison_path}")
    print(f"  Qwen artifacts:   {qwen_dir}")
    print(f"  Hybrid artifacts: {hybrid_dir}")
    return comparison


def _load_manifest(output_dir: str) -> dict | None:
    manifest_path = Path(output_dir) / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _safe_mean(values) -> float | None:
    if not values:
        return None
    return statistics.mean(values)


def _safe_median(values) -> float | None:
    if not values:
        return None
    return statistics.median(values)


# ---------------------------------------------------------------------------
# Phase 5: Architecture Decision Summary
# ---------------------------------------------------------------------------

def write_recommendation(
    fallback_metrics: dict | None,
    robustness_results: list[dict] | None,
    cascade_comparison: dict | None,
    *,
    output_dir: str = OUTPUT_DIR,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lines = ["# Hybrid Architecture Audit — Recommendation Note\n"]

    if fallback_metrics:
        rate = fallback_metrics.get("fallback_rate")
        gemma = fallback_metrics.get("gemma_used_ticks", 0)
        fallback = fallback_metrics.get("fallback_ticks", 0)
        total = fallback_metrics.get("fallback_aware_ticks", 0)
        lines.append("## Fallback Audit (Phase 2)\n")
        lines.append(f"- Ticks with Gemma timings: {gemma}/{total}")
        lines.append(f"- Fallback rate: {rate:.1%}" if rate is not None else "- Fallback rate: N/A")
        lines.append(f"- Reasons: {fallback_metrics.get('fallback_reasons', {})}")
        if rate is not None and rate < 0.2:
            lines.append("- Verdict: Gemma alignment is dominant — fallback is rare.\n")
        elif rate is not None and rate < 0.5:
            lines.append("- Verdict: Mixed — Gemma contributes but fallback is common.\n")
        else:
            lines.append("- Verdict: Fallback-dominant — hybrid is mostly Qwen timings.\n")

    if robustness_results:
        lines.append("## Robustness Check (Phase 3)\n")
        lines.append(f"| Tag | Words | MAE ms | Med ms | P90 ms | Mono |")
        lines.append(f"|---|---:|---:|---:|---:|---:|")
        maes = []
        for r in robustness_results:
            mae = r.get("word_end_mae_seconds")
            med = r.get("word_end_median_error_seconds")
            p90 = r.get("word_end_p90_error_seconds")
            mono = r.get("monotonicity")
            if mae is not None:
                maes.append(mae)
                lines.append(
                    f"| {r['tag']} | {r['paired_words']} | "
                    f"{mae*1000:.0f} | {med*1000:.0f} | {p90*1000:.0f} | "
                    f"{mono:.3f} |"
                )
        if maes:
            mean_mae = statistics.mean(maes)
            std_mae = statistics.stdev(maes) if len(maes) > 1 else 0
            lines.append(f"\nMean MAE: {mean_mae*1000:.0f} ms (std: {std_mae*1000:.0f} ms)")
            if std_mae < 0.05:
                lines.append("- Verdict: Robust — MAE is stable across clips.\n")
            else:
                lines.append("- Verdict: Variable — performance depends on clip.\n")

    if cascade_comparison:
        lines.append("## Cascade Comparison (Phase 4)\n")
        for label in ["qwen", "hybrid"]:
            c = cascade_comparison.get(label, {})
            lines.append(f"### {label}")
            lines.append(f"- Mean word delay: {c.get('mean_word_delay_ms', 'N/A')} ms")
            lines.append(f"- Median word delay: {c.get('median_word_delay_ms', 'N/A')} ms")
            lines.append(f"- Updates: {c.get('num_updates', 'N/A')}")
            lines.append("")

    lines.append("## Final Recommendation\n")
    lines.append("_To be filled after reviewing the numbers above._\n")

    note_path = out / "phase5_recommendation.md"
    note_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[Phase 5] Recommendation note: {note_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument(
        "--heads-path", default=HEADS_PATH,
        help="Calibrated heads bundle for Gemma alignment",
    )
    parser.add_argument(
        "--wav", default="test-set/audio/ccpXHNfaoy.wav",
        help="Primary talk WAV for fallback audit and cascade comparison",
    )

    parser.add_argument("--phase2", action="store_true", help="Run fallback audit")
    parser.add_argument("--phase3", action="store_true", help="Run robustness check")
    parser.add_argument("--phase4", action="store_true", help="Run cascade comparison")
    parser.add_argument("--all", action="store_true", help="Run all phases")

    parser.add_argument("--tick-start", type=float, default=5.0)
    parser.add_argument("--tick-stop", type=float, default=60.0)
    parser.add_argument("--tick-step", type=float, default=2.0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_cli().parse_args(argv)
    run_all = args.all or not (args.phase2 or args.phase3 or args.phase4)

    fallback_metrics = None
    robustness_results = None
    cascade_comparison = None

    if run_all or args.phase2:
        fallback_metrics = run_fallback_audit(
            args.wav,
            heads_path=args.heads_path,
            tick_start_s=args.tick_start,
            tick_stop_s=args.tick_stop,
            tick_step_s=args.tick_step,
            output_dir=args.output_dir,
        )

    if run_all or args.phase3:
        robustness_results = run_robustness_check(
            heads_path=args.heads_path,
            output_dir=args.output_dir,
        )

    if run_all or args.phase4:
        cascade_comparison = run_cascade_comparison(
            args.wav,
            output_dir=args.output_dir,
        )

    write_recommendation(
        fallback_metrics, robustness_results, cascade_comparison,
        output_dir=args.output_dir,
    )

    print(f"\n{'='*60}")
    print("AUDIT COMPLETE")
    print(f"All results in: {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
