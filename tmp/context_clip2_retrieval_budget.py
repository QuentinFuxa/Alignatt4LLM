#!/usr/bin/env python3
"""Budget sweep for retrieved_chunks on clip 2.

max_chars in {400, 600, 2400} to check if budget size changes the
outcome on the clip where retrieval cratered BLEU (-3.28).
"""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cascade.simulstream_processor import CascadeAlignAttProcessor
from run_simulstream_batch import run_single_audio
from tmp.reanchor_baseline import write_artifacts, build_processor_config


def run_at_budget(max_chars: int, reset: bool) -> None:
    cfg = build_processor_config(450)
    cfg.target_lang_code = "de"
    cfg.mt_backend_name = "gemma_vllm_alignatt"
    cfg.asr_commit_mode = "punctuation_lcp"
    cfg.paper_context_path = "data/paper_artifacts/OiqEWDVtWk.json"
    cfg.paper_context_mode = "retrieved_chunks"
    cfg.paper_context_top_k = 3
    cfg.paper_context_max_chars = max_chars

    if reset:
        print(f"Loading (retrieved_chunks max_chars={max_chars})...", flush=True)
        load_start = perf_counter()
        CascadeAlignAttProcessor.load_model(cfg)
        print(f"Models loaded in {(perf_counter()-load_start):.1f}s", flush=True)

    processor = CascadeAlignAttProcessor(cfg)
    processor.set_source_language("en")
    processor.set_target_language("de")

    print(f"\n==== rc_budget_{max_chars} ====", flush=True)
    batch_start = perf_counter()
    result = run_single_audio(
        processor, "test-set/audio/OiqEWDVtWk.wav", 450, "de", "en",
    )
    batch_wallclock_s = perf_counter() - batch_start
    print(f"  max_chars={max_chars} RTF={result['rtf']:.3f} updates={result['num_updates']} wallclock={batch_wallclock_s:.1f}s", flush=True)
    output_dir = f"outputs/night2_context_clip2_rc_budget_{max_chars}"
    write_artifacts(result, output_dir, cfg, processor, batch_wallclock_s,
                    "test-set/audio/OiqEWDVtWk.wav")
    print(f"  Artifacts: {output_dir}", flush=True)


def main():
    for i, mc in enumerate([400, 600, 2400]):
        run_at_budget(mc, reset=(i == 0))
    print("\nDONE_RC_BUDGET", flush=True)


if __name__ == "__main__":
    main()
