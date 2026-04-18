#!/usr/bin/env python3
"""Scalar threshold sweep on Transformers MT with routing fix.

Baseline (discrete): BLEU 28.22, 40 src_frontier firings.
Scalar @ 0.015:      BLEU 27.46, 26 src_frontier firings (-35%).

Sweep thresholds to characterise the scalar mechanism's
latency-quality curve properly.
"""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cascade.simulstream_processor import CascadeAlignAttProcessor
from run_simulstream_batch import run_single_audio
from tmp.reanchor_baseline import write_artifacts, build_processor_config


def run_at_threshold(threshold: float, label: str, reset: bool) -> None:
    cfg = build_processor_config(450)
    cfg.target_lang_code = "de"
    cfg.mt_backend_name = "gemma_transformers_alignatt"
    cfg.asr_commit_mode = "punctuation_lcp"
    cfg.translation_source_frontier_mode = "scalar"
    cfg.translation_source_frontier_scalar_threshold = threshold

    if reset:
        print(f"Loading models (threshold={threshold}) ...", flush=True)
        load_start = perf_counter()
        CascadeAlignAttProcessor.load_model(cfg)
        print(f"Models loaded in {(perf_counter()-load_start):.1f}s", flush=True)

    processor = CascadeAlignAttProcessor(cfg)
    processor.set_source_language("en")
    processor.set_target_language("de")

    inner_mode = getattr(
        processor.session.config, "translation_source_frontier_mode", "<unset>"
    )
    inner_thr = getattr(
        processor.session.config,
        "translation_source_frontier_scalar_threshold",
        "<unset>",
    )
    print(f"[verify] mode={inner_mode!r} threshold={inner_thr!r}", flush=True)
    assert inner_mode == "scalar" and abs(float(inner_thr) - threshold) < 1e-9

    print(f"\n==== scalar_thr_{label} ====", flush=True)
    batch_start = perf_counter()
    result = run_single_audio(
        processor, "test-set/audio/ccpXHNfaoy.wav", 450, "de", "en",
    )
    batch_wallclock_s = perf_counter() - batch_start
    print(
        f"  thr={threshold} RTF={result['rtf']:.3f}  updates={result['num_updates']}  "
        f"wallclock={batch_wallclock_s:.1f}s",
        flush=True,
    )
    output_dir = f"outputs/night1_ende_scalar_thr_{label}_REAL"
    write_artifacts(result, output_dir, cfg, processor, batch_wallclock_s,
                    "test-set/audio/ccpXHNfaoy.wav")
    print(f"  Artifacts: {output_dir}", flush=True)


def main():
    # Three thresholds bracketing 0.015: tighter, same, looser
    for i, (thr, label) in enumerate([
        (0.005, "0p005"),
        (0.050, "0p050"),
    ]):
        run_at_threshold(thr, label, reset=(i == 0))
    print("\nDONE_THRESHOLD_SWEEP", flush=True)


if __name__ == "__main__":
    main()
