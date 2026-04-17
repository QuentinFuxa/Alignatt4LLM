#!/usr/bin/env python3
"""Discrete vLLM MT on current SHA, same config as scalar vLLM MT.

The scalar vLLM MT A/B run tonight (night1_ende_scalar_vllm_mt_instrumented)
produced BLEU 28.83 / 102 updates. The previous discrete vLLM MT
reanchor reference (reanchor_chunk450) produced BLEU 27.51 /
438 updates, but it was on an earlier SHA (pre-a0edcc6, pre-f1cfafa).

To separate "scalar vs discrete effect" from "custom-op-era
scheduler effects", run discrete vLLM MT on the CURRENT SHA with
the custom-op code path in place, then compare to scalar vLLM MT.
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
    cfg.mt_backend_name = "gemma_vllm_alignatt"
    cfg.asr_commit_mode = "punctuation_lcp"
    cfg.translation_source_frontier_mode = "discrete"

    print("Loading models (discrete + vLLM MT, current SHA) ...", flush=True)
    load_start = perf_counter()
    CascadeAlignAttProcessor.load_model(cfg)
    print(f"Models loaded in {(perf_counter()-load_start):.1f}s", flush=True)

    processor = CascadeAlignAttProcessor(cfg)
    processor.set_source_language("en")
    processor.set_target_language("de")

    print(f"\n==== discrete_vllm_mt ====", flush=True)
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
    output_dir = "outputs/night1_ende_discrete_vllm_mt_customop_instrumented"
    write_artifacts(result, output_dir, cfg, processor, batch_wallclock_s,
                    "test-set/audio/ccpXHNfaoy.wav")
    print(f"  Artifacts: {output_dir}", flush=True)
    print("\nDONE_DISCRETE_VLLM", flush=True)


if __name__ == "__main__":
    main()
