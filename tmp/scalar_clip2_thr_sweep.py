#!/usr/bin/env python3
"""Threshold sweep on clip 2 to verify scalar invariance is clip-robust.

On clip 1 (ccpXHNfaoy), scalar @ 0.005 / 0.015 / 0.050 produced
bit-identical 5569-char outputs. Test whether the same
invariance property holds on clip 2 (OiqEWDVtWk).

Baseline scalar @ 0.015 on clip 2: BLEU 28.11, 323 updates, 40 src_fr.
Comparing scalar @ 0.005 and scalar @ 0.050 to that.
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
        print(f"Loading models (clip2 thr={threshold})...", flush=True)
        load_start = perf_counter()
        CascadeAlignAttProcessor.load_model(cfg)
        print(f"Models loaded in {(perf_counter()-load_start):.1f}s", flush=True)

    processor = CascadeAlignAttProcessor(cfg)
    processor.set_source_language("en")
    processor.set_target_language("de")

    inner = getattr(processor.session.config, "translation_source_frontier_mode", None)
    inner_thr = getattr(processor.session.config, "translation_source_frontier_scalar_threshold", None)
    assert inner == "scalar" and abs(float(inner_thr) - threshold) < 1e-9
    print(f"[verify clip2] mode={inner!r} threshold={inner_thr!r}", flush=True)

    print(f"\n==== scalar_clip2_thr_{label} ====", flush=True)
    batch_start = perf_counter()
    result = run_single_audio(
        processor, "test-set/audio/OiqEWDVtWk.wav", 450, "de", "en",
    )
    batch_wallclock_s = perf_counter() - batch_start
    print(
        f"  thr={threshold} RTF={result['rtf']:.3f}  updates={result['num_updates']}  "
        f"wallclock={batch_wallclock_s:.1f}s",
        flush=True,
    )
    output_dir = f"outputs/night1_ende_scalar_clip2_thr_{label}_REAL"
    write_artifacts(result, output_dir, cfg, processor, batch_wallclock_s,
                    "test-set/audio/OiqEWDVtWk.wav")
    print(f"  Artifacts: {output_dir}", flush=True)


def main():
    for i, (thr, label) in enumerate([(0.005, "0p005"), (0.050, "0p050")]):
        run_at_threshold(thr, label, reset=(i == 0))
    print("\nDONE_CLIP2_SWEEP", flush=True)


if __name__ == "__main__":
    main()
