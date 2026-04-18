#!/usr/bin/env python3
"""Real scalar on clip 2 (OiqEWDVtWk.wav) with routing fix.

Confirm the 0.76 BLEU delta vs discrete is a clip-invariant
property, not specific to ccpXHNfaoy.wav.

Existing discrete baseline: night1_ende_punct_chunk450_OiqEWDVtWk_instrumented
BLEU 27.60 / COMET 0.832 / 323 updates.
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
    cfg.translation_source_frontier_mode = "scalar"
    cfg.translation_source_frontier_scalar_threshold = 0.015

    print("Loading (scalar real, clip 2)...", flush=True)
    load_start = perf_counter()
    CascadeAlignAttProcessor.load_model(cfg)
    print(f"Models loaded in {(perf_counter()-load_start):.1f}s", flush=True)

    processor = CascadeAlignAttProcessor(cfg)
    processor.set_source_language("en")
    processor.set_target_language("de")

    inner = getattr(processor.session.config, "translation_source_frontier_mode", None)
    assert inner == "scalar", f"mode={inner}"
    print(f"[verify] mode=scalar threshold=0.015", flush=True)

    print(f"\n==== scalar_clip2 ====", flush=True)
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
    output_dir = "outputs/night1_ende_scalar_clip2_REAL"
    write_artifacts(result, output_dir, cfg, processor, batch_wallclock_s,
                    "test-set/audio/OiqEWDVtWk.wav")
    print(f"  Artifacts: {output_dir}", flush=True)
    print("\nDONE_SCALAR_CLIP2", flush=True)


if __name__ == "__main__":
    main()
