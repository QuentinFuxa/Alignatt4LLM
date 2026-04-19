#!/usr/bin/env python3
"""Real scalar vLLM MT in eager mode, with observer-working fix.

Post commit d153be2: vLLM MT observer captures correctly in
eager mode (fwd_count=90, 37 src_fr + 22 rewind firings vs
cudagraph=full fwd_count=0 broken state).

This runs the scalar substitution at threshold 0.015 on the
same config, completing the cleanest possible scalar-vs-discrete
A/B at the vLLM MT level.
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
    cfg.translation_source_frontier_mode = "scalar"
    cfg.translation_source_frontier_scalar_threshold = 0.015
    cfg.mt_vllm_enforce_eager = True
    cfg.mt_vllm_cudagraph_mode = None

    print("Loading (scalar vLLM MT, eager with fix)...", flush=True)
    load_start = perf_counter()
    CascadeAlignAttProcessor.load_model(cfg)
    print(f"Models loaded in {(perf_counter()-load_start):.1f}s", flush=True)

    processor = CascadeAlignAttProcessor(cfg)
    processor.set_source_language("en")
    processor.set_target_language("de")

    inner = getattr(processor.session.config, "translation_source_frontier_mode", None)
    inner_eager = getattr(processor.session.config, "mt_vllm_enforce_eager", None)
    assert inner == "scalar" and inner_eager == True
    print(f"[verify] mode={inner!r} enforce_eager={inner_eager!r}", flush=True)

    print(f"\n==== scalar_vllm_eager_real ====", flush=True)
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
    output_dir = "outputs/night1_ende_scalar_vllm_eager_REAL"
    write_artifacts(result, output_dir, cfg, processor, batch_wallclock_s,
                    "test-set/audio/ccpXHNfaoy.wav")
    print(f"  Artifacts: {output_dir}", flush=True)
    print("\nDONE_SCALAR_VLLM_EAGER", flush=True)


if __name__ == "__main__":
    main()
