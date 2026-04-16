#!/usr/bin/env python3
"""Compare en->de cascade: qwen baseline vs hybrid+qk_fast alignment."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from qwen3asr_gemma_cascade_core import (
    config,
    run_stream_to_artifacts,
    temporary_runtime_config,
)
from cascade_artifacts import write_inference_artifacts

WAV = "test-set/audio/ccpXHNfaoy.wav"
CHUNK_MS = 450
OUT_BASE = "outputs/qk_fast_cascade_eval"

OVERRIDES = dict(
    target_lang="German",
    source_lang="English",
    min_start_seconds=2.0,
    max_history_utterances=1,
    partial_max_new_tokens=16,
    partial_followup_max_new_tokens=8,
    translation_alignatt_inaccessible_ms=0.0,
    translation_alignatt_rewind_threshold=8,
)


def run_cascade(backend_name: str, tag: str) -> dict:
    output_dir = f"{OUT_BASE}/{tag}"
    with temporary_runtime_config(alignment_backend_name=backend_name, **OVERRIDES):
        t0 = perf_counter()
        artifacts = run_stream_to_artifacts(
            WAV, chunk_ms=CHUNK_MS,
            run_provenance={"test": "qk_fast_eval", "backend": backend_name},
        )
        elapsed = perf_counter() - t0

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    write_inference_artifacts(artifacts, output_dir=output_dir)
    n = len(artifacts.translation_word_delays_ms)
    return {
        "tag": tag,
        "output_dir": output_dir,
        "elapsed_s": elapsed,
        "words": n,
        "mean_delay_ms": sum(artifacts.translation_word_delays_ms) / max(n, 1),
        "p90_delay_ms": sorted(artifacts.translation_word_delays_ms)[int(n * 0.9)] if n else 0,
        "translation": (artifacts.final_translation or "")[:200],
    }


def evaluate(output_dir: str) -> dict:
    """Run evaluation via .venv-evaluation."""
    venv = Path(".venv-evaluation").resolve()
    result = subprocess.run(
        [str(venv / "bin" / "python"), "evaluate_cascade_outputs.py",
         "--output-dir", output_dir, "--target", "de"],
        capture_output=True, text=True, timeout=120,
    )
    report = Path(output_dir) / "evaluation_report.json"
    if report.exists():
        return json.loads(report.read_text())
    # Try to parse from stdout
    print(f"  eval exit={result.returncode}")
    if result.stdout:
        print(f"  stdout (last 300): {result.stdout[-300:]}")
    if result.stderr:
        print(f"  stderr (last 300): {result.stderr[-300:]}")
    return {}


def main():
    # --- Baseline: qwen alignment ---
    print("=" * 60)
    print("1/2  Qwen alignment (baseline)")
    print("=" * 60)
    r1 = run_cascade("qwen", "qwen_baseline")
    print(f"  {r1['elapsed_s']:.1f}s, {r1['words']} words, "
          f"mean_delay={r1['mean_delay_ms']:.0f}ms, p90={r1['p90_delay_ms']:.0f}ms")

    # --- Hybrid: qwen ASR + gemma qk_fast alignment ---
    print(f"\n{'=' * 60}")
    print("2/2  Hybrid alignment (qwen ASR + gemma qk_fast)")
    print("=" * 60)
    r2 = run_cascade("hybrid_qwen_asr_gemma_aligner", "hybrid_qk_fast")
    print(f"  {r2['elapsed_s']:.1f}s, {r2['words']} words, "
          f"mean_delay={r2['mean_delay_ms']:.0f}ms, p90={r2['p90_delay_ms']:.0f}ms")

    # --- Evaluate both ---
    print(f"\n{'=' * 60}")
    print("Evaluation")
    print("=" * 60)
    metrics = {}
    for r in [r1, r2]:
        print(f"\n--- {r['tag']} ---")
        m = evaluate(r["output_dir"])
        metrics[r["tag"]] = m
        for k in ["BLEU", "chrF", "LongYAAL_CU", "LongYAAL_CA"]:
            if k in m:
                print(f"  {k}: {m[k]:.2f}")

    # --- Summary ---
    m1 = metrics.get("qwen_baseline", {})
    m2 = metrics.get("hybrid_qk_fast", {})
    print(f"\n{'=' * 60}")
    print(f"{'Metric':<20} {'Qwen':>10} {'Hybrid+qk_fast':>15} {'Delta':>10}")
    print("-" * 58)
    print(f"{'Elapsed (s)':<20} {r1['elapsed_s']:>10.1f} {r2['elapsed_s']:>15.1f} "
          f"{r2['elapsed_s'] - r1['elapsed_s']:>+10.1f}")
    print(f"{'Words':<20} {r1['words']:>10} {r2['words']:>15} "
          f"{r2['words'] - r1['words']:>+10}")
    print(f"{'Mean delay (ms)':<20} {r1['mean_delay_ms']:>10.0f} {r2['mean_delay_ms']:>15.0f} "
          f"{r2['mean_delay_ms'] - r1['mean_delay_ms']:>+10.0f}")
    for k in ["BLEU", "chrF", "LongYAAL_CU"]:
        v1 = m1.get(k)
        v2 = m2.get(k)
        if v1 is not None and v2 is not None:
            print(f"{k:<20} {v1:>10.2f} {v2:>15.2f} {v2 - v1:>+10.2f}")


if __name__ == "__main__":
    main()
