#!/usr/bin/env python3
"""Discrete vLLM MT after mutates_args=unknown fix for observer DCE.

Previous discrete_vllm_mt run had forward_call_count=0 on the
observer — the custom op was registered with mutates_args=()
which tells the compiler the op is pure, and inductor DCE elided
it from the AOT-compiled graph. Fix: mutates_args="unknown"
prevents DCE.

Forces a cold AOT compile (cache was cleared) to validate that
the observer now captures.
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
    cfg.mt_backend_name = "gemma_vllm_alignatt"
    cfg.asr_commit_mode = "punctuation_lcp"
    cfg.translation_source_frontier_mode = "discrete"

    print("Loading models (discrete + vLLM MT, FIXED observer) ...", flush=True)
    load_start = perf_counter()
    CascadeAlignAttProcessor.load_model(cfg)
    print(f"Models loaded in {(perf_counter()-load_start):.1f}s", flush=True)

    processor = CascadeAlignAttProcessor(cfg)
    processor.set_source_language("en")
    processor.set_target_language("de")

    print(f"\n==== discrete_vllm_mt_fixed ====", flush=True)
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
    output_dir = "outputs/night1_ende_discrete_vllm_mt_customop_FIXED"
    write_artifacts(result, output_dir, cfg, processor, batch_wallclock_s,
                    "test-set/audio/ccpXHNfaoy.wav")
    print(f"  Artifacts: {output_dir}", flush=True)
    print("\nDONE_DISCRETE_VLLM_FIXED", flush=True)


if __name__ == "__main__":
    main()
