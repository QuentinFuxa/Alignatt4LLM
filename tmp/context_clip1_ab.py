#!/usr/bin/env python3
"""3-way context A/B on clip 1 ccpXHNfaoy (for parallel A40 run).

Mirrors the clip 2 experiment. Reuses hot models across modes.
"""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cascade.simulstream_processor import CascadeAlignAttProcessor
from run_simulstream_batch import run_single_audio
from tmp.reanchor_baseline import write_artifacts, build_processor_config


def run_at_mode(mode: str, reset: bool) -> None:
    cfg = build_processor_config(450)
    cfg.target_lang_code = "de"
    cfg.mt_backend_name = "gemma_vllm_alignatt"
    cfg.asr_commit_mode = "punctuation_lcp"
    if mode != "off":
        cfg.paper_context_path = "data/paper_artifacts/ccpXHNfaoy.json"
        cfg.paper_context_mode = mode
        cfg.paper_context_top_k = 3
        cfg.paper_context_max_chars = 1200

    if reset:
        print(f"Loading models (clip1 context mode={mode})...", flush=True)
        load_start = perf_counter()
        CascadeAlignAttProcessor.load_model(cfg)
        print(f"Models loaded in {(perf_counter()-load_start):.1f}s", flush=True)

    processor = CascadeAlignAttProcessor(cfg)
    processor.set_source_language("en")
    processor.set_target_language("de")

    inner = getattr(processor.session.config, "paper_context_mode", "off")
    print(f"[verify] paper_context_mode={inner!r}", flush=True)

    print(f"\n==== context_clip1_{mode} ====", flush=True)
    batch_start = perf_counter()
    result = run_single_audio(
        processor, "test-set/audio/ccpXHNfaoy.wav", 450, "de", "en",
    )
    batch_wallclock_s = perf_counter() - batch_start
    print(
        f"  mode={mode} RTF={result['rtf']:.3f}  updates={result['num_updates']}  "
        f"wallclock={batch_wallclock_s:.1f}s",
        flush=True,
    )
    output_dir = f"outputs/night2_context_clip1_{mode}"
    write_artifacts(result, output_dir, cfg, processor, batch_wallclock_s,
                    "test-set/audio/ccpXHNfaoy.wav")
    print(f"  Artifacts: {output_dir}", flush=True)


def main():
    for i, mode in enumerate(["off", "title_abstract", "retrieved_chunks", "title_and_chunks"]):
        run_at_mode(mode, reset=(i == 0))
    print("\nDONE_CLIP1_CONTEXT", flush=True)


if __name__ == "__main__":
    main()
