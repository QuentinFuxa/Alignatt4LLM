#!/usr/bin/env python3
"""Real scalar-mode Transformers MT A/B (after fix cascade_simulstream_processor routing).

Previously cfg.translation_source_frontier_mode = "scalar" was
silently ignored because _build_runtime_config didn't route it
through. Every "scalar" run this session actually ran as
discrete. With the routing fixed in cascade_simulstream_processor
override_keys, run scalar on Transformers MT (observer works there)
and compare to the discrete reanchor baseline.
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
    cfg.mt_backend_name = "gemma_transformers_alignatt"  # Observer works here
    cfg.asr_commit_mode = "punctuation_lcp"
    cfg.translation_source_frontier_mode = "scalar"
    cfg.translation_source_frontier_scalar_threshold = 0.015

    print(
        "Loading models (scalar + Transformers MT, real fix)...",
        flush=True,
    )
    load_start = perf_counter()
    CascadeAlignAttProcessor.load_model(cfg)
    print(f"Models loaded in {(perf_counter()-load_start):.1f}s", flush=True)

    processor = CascadeAlignAttProcessor(cfg)
    processor.set_source_language("en")
    processor.set_target_language("de")

    # Verify the mode actually took effect
    inner_mode = getattr(
        processor.session.config, "translation_source_frontier_mode", "<unset>"
    )
    inner_thr = getattr(
        processor.session.config,
        "translation_source_frontier_scalar_threshold",
        "<unset>",
    )
    print(
        f"[verify] runtime translation_source_frontier_mode={inner_mode!r} "
        f"threshold={inner_thr!r}",
        flush=True,
    )
    assert inner_mode == "scalar", (
        f"Expected scalar, got {inner_mode!r} — config routing still broken"
    )

    print(f"\n==== scalar_transformers_mt_real ====", flush=True)
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
    output_dir = "outputs/night1_ende_scalar_transformers_mt_REAL"
    write_artifacts(result, output_dir, cfg, processor, batch_wallclock_s,
                    "test-set/audio/ccpXHNfaoy.wav")
    print(f"  Artifacts: {output_dir}", flush=True)
    print("\nDONE_SCALAR_TRANSFORMERS_REAL", flush=True)


if __name__ == "__main__":
    main()
