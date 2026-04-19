#!/usr/bin/env python3
"""Stress-test scalar substitution on cs->en.

Offline drift on cs->en was 47-55% update agreement and -41% to
-24% token drift — much worse than en->de's 83-87% / +/-3%. If
online quality stays close to the discrete reference despite the
worse offline drift, the paper's "MT absorbs drift" finding is
strengthened. If quality degrades materially, the en->de
bit-identical result is direction-specific.

Discrete reference is the existing
`outputs/night1_cs_en_chunk450/` (Transformers MT baseline).
This run produces the scalar counterpart on the same clip.
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

    print("Loading models (scalar, cs->en) ...", flush=True)
    load_start = perf_counter()
    CascadeAlignAttProcessor.load_model(cfg)
    print(f"Models loaded in {(perf_counter()-load_start):.1f}s", flush=True)

    processor = CascadeAlignAttProcessor(cfg)
    processor.set_source_language("cs")
    processor.set_target_language("en")

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
    output_dir = "outputs/night1_cs_en_scalar_chunk450_instrumented"
    write_artifacts(result, output_dir, cfg, processor, batch_wallclock_s,
                    "test-set/audio/csJIsDTYMW.wav")
    print(f"  Artifacts: {output_dir}", flush=True)
    print("\nDONE_SCALAR_CS_EN", flush=True)


if __name__ == "__main__":
    main()
