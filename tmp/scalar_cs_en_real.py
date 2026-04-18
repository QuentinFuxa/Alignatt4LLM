#!/usr/bin/env python3
"""Real scalar cs->en with routing fix.

Prior cs->en scalar vs discrete were byte-identical (5556/5556
chars) under the routing bug. Post-fix, expect scalar to diverge
(as on en->de clip 1 and clip 2).
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
    cfg.source_lang_code = "cs"
    cfg.target_lang_code = "en"
    cfg.mt_backend_name = "gemma_transformers_alignatt"
    cfg.asr_commit_mode = "punctuation_lcp"
    cfg.translation_source_frontier_mode = "scalar"
    cfg.translation_source_frontier_scalar_threshold = 0.015

    print("Loading (scalar cs->en real)...", flush=True)
    load_start = perf_counter()
    CascadeAlignAttProcessor.load_model(cfg)
    print(f"Models loaded in {(perf_counter()-load_start):.1f}s", flush=True)

    processor = CascadeAlignAttProcessor(cfg)
    processor.set_source_language("cs")
    processor.set_target_language("en")

    inner = getattr(processor.session.config, "translation_source_frontier_mode", None)
    assert inner == "scalar", f"mode={inner}"
    print(f"[verify] mode=scalar threshold=0.015", flush=True)

    print(f"\n==== scalar_cs_en ====", flush=True)
    batch_start = perf_counter()
    result = run_single_audio(
        processor, "test-set/audio/csJIsDTYMW.wav", 450, "en", "cs",
    )
    batch_wallclock_s = perf_counter() - batch_start
    print(
        f"  RTF={result['rtf']:.3f}  updates={result['num_updates']}  "
        f"wallclock={batch_wallclock_s:.1f}s",
        flush=True,
    )
    output_dir = "outputs/night1_cs_en_scalar_REAL"
    write_artifacts(result, output_dir, cfg, processor, batch_wallclock_s,
                    "test-set/audio/csJIsDTYMW.wav")
    print(f"  Artifacts: {output_dir}", flush=True)
    print("\nDONE_SCALAR_CS_EN", flush=True)


if __name__ == "__main__":
    main()
