#!/usr/bin/env python3
"""Online A/B for the scalar source_frontier substitution.

Runs the canonical config on ccpXHNfaoy.wav twice:
  - discrete source_frontier (default): pre-existing path
  - scalar source_frontier at threshold 0.015 (drift-optimal)

The discrete reference already exists as
`outputs/night1_ende_punct_chunk450_instrumented`. This run produces
the scalar counterpart so they can be compared by BLEU / chrF /
COMET / LongYAAL CA.

Uses Transformers MT to sidestep the vLLM-MT compile-cache
fragility documented in DECISIONS.
"""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cascade_simulstream_processor import CascadeAlignAttProcessor
from run_simulstream_batch import run_single_audio
from tmp.reanchor_baseline import write_artifacts, build_processor_config


def main():
    cfg = build_processor_config(450)
    cfg.target_lang_code = "de"
    cfg.mt_backend_name = "gemma_transformers_alignatt"
    cfg.asr_commit_mode = "punctuation_lcp"
    cfg.translation_source_frontier_mode = "scalar"
    cfg.translation_source_frontier_scalar_threshold = 0.015

    print("Loading models (scalar source_frontier mode) ...", flush=True)
    load_start = perf_counter()
    CascadeAlignAttProcessor.load_model(cfg)
    print(f"Models loaded in {(perf_counter()-load_start):.1f}s", flush=True)

    processor = CascadeAlignAttProcessor(cfg)
    processor.set_source_language("en")
    processor.set_target_language("de")

    print(f"\n==== ende_punct_chunk450_scalar_instrumented ====", flush=True)
    batch_start = perf_counter()
    result = run_single_audio(
        processor, "test-set/audio/ccpXHNfaoy.wav", 450, "de", "en",
    )
    batch_wallclock_s = perf_counter() - batch_start
    print(
        f"  RTF={result['rtf']:.3f}  updates={result['num_updates']}  "
        f"wallclock={batch_wallclock_s:.1f}s",
        flush=True,
    )
    output_dir = "outputs/night1_ende_punct_chunk450_scalar_instrumented"
    write_artifacts(result, output_dir, cfg, processor, batch_wallclock_s,
                    "test-set/audio/ccpXHNfaoy.wav")
    print(f"  Artifacts: {output_dir}", flush=True)
    print("\nDONE_SCALAR_AB", flush=True)


if __name__ == "__main__":
    main()
