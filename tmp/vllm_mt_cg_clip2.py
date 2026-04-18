#!/usr/bin/env python3
"""vLLM MT cg=full on clip 2 (OiqEWDVtWk): both discrete + scalar.

Replicates the bit-identity finding on a second clip. Reuses hot
models across both runs.
"""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cascade.simulstream_processor import CascadeAlignAttProcessor
from run_simulstream_batch import run_single_audio
from tmp.reanchor_baseline import write_artifacts, build_processor_config


def run_at_mode(mode: str, label: str, reset: bool) -> None:
    cfg = build_processor_config(450)
    cfg.target_lang_code = "de"
    cfg.mt_backend_name = "gemma_vllm_alignatt"
    cfg.asr_commit_mode = "punctuation_lcp"
    cfg.translation_source_frontier_mode = mode
    if mode == "scalar":
        cfg.translation_source_frontier_scalar_threshold = 0.015

    if reset:
        print(f"Loading models (clip 2, mode={mode})...", flush=True)
        load_start = perf_counter()
        CascadeAlignAttProcessor.load_model(cfg)
        print(f"Models loaded in {(perf_counter()-load_start):.1f}s", flush=True)

    processor = CascadeAlignAttProcessor(cfg)
    processor.set_source_language("en")
    processor.set_target_language("de")

    inner = getattr(processor.session.config, "translation_source_frontier_mode", None)
    assert inner == mode, f"mode expected {mode}, got {inner}"
    print(f"[verify] mode={inner!r}", flush=True)

    print(f"\n==== vllm_mt_cg_clip2_{label} ====", flush=True)
    batch_start = perf_counter()
    result = run_single_audio(
        processor, "test-set/audio/OiqEWDVtWk.wav", 450, "de", "en",
    )
    batch_wallclock_s = perf_counter() - batch_start
    print(
        f"  mode={mode} RTF={result['rtf']:.3f}  updates={result['num_updates']}  "
        f"wallclock={batch_wallclock_s:.1f}s",
        flush=True,
    )
    output_dir = f"outputs/night1_ende_{label}_vllm_cg_clip2_FIXED"
    write_artifacts(result, output_dir, cfg, processor, batch_wallclock_s,
                    "test-set/audio/OiqEWDVtWk.wav")
    print(f"  Artifacts: {output_dir}", flush=True)


def main():
    for i, (mode, label) in enumerate([("discrete", "disc"), ("scalar", "scal")]):
        run_at_mode(mode, label, reset=(i == 0))
    print("\nDONE_CLIP2_VLLM_CG", flush=True)


if __name__ == "__main__":
    main()
