#!/usr/bin/env python3
"""Canonical baseline on a second clip (OiqEWDVtWk.wav) to validate
the surprising per-gate finding from ccpXHNfaoy:

  canonical submission path ->
    source_frontier F1 0.99 (1-feature scalar)
    rewind          F1 0.91 (1-feature scalar)

If this reproduces on OiqEWDVtWk, the finding generalises. If not,
it's clip-specific and the paper needs to qualify accordingly.

Uses Transformers MT fallback (vLLM MT compile-cache fragility
blocks this config on repeat runs; see DECISIONS.md).
"""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cascade.simulstream_processor import CascadeAlignAttProcessor
from run_simulstream_batch import run_single_audio
from tmp.reanchor_baseline import write_artifacts, build_processor_config


def main():
    cfg = build_processor_config(450)
    cfg.target_lang_code = "de"
    cfg.mt_backend_name = "gemma_transformers_alignatt"
    cfg.asr_commit_mode = "punctuation_lcp"

    print("Loading models (Transformers MT) ...", flush=True)
    load_start = perf_counter()
    CascadeAlignAttProcessor.load_model(cfg)
    print(f"Models loaded in {(perf_counter()-load_start):.1f}s", flush=True)

    processor = CascadeAlignAttProcessor(cfg)
    processor.set_source_language("en")
    processor.set_target_language("de")

    print(f"\n==== ende_punct_chunk450_OiqEWDVtWk_instrumented ====", flush=True)
    batch_start = perf_counter()
    result = run_single_audio(
        processor, "test-set/audio/OiqEWDVtWk.wav", 450, "de", "en",
    )
    batch_wallclock_s = perf_counter() - batch_start
    print(
        f"  RTF={result['rtf']:.3f}  updates={result['num_updates']}  "
        f"wallclock={batch_wallclock_s:.1f}s",
        flush=True,
    )
    output_dir = "outputs/night1_ende_punct_chunk450_OiqEWDVtWk_instrumented"
    write_artifacts(result, output_dir, cfg, processor, batch_wallclock_s,
                    "test-set/audio/OiqEWDVtWk.wav")
    print(f"  Artifacts: {output_dir}", flush=True)
    print("\nDONE_CLIP2", flush=True)


if __name__ == "__main__":
    main()
