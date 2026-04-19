#!/usr/bin/env python3
"""vLLM MT cudagraph=full at chunk_ms=700 (second PLAN operating point).

Observer fix stack (54e8b94 + d153be2 + 50ad207) lets this run
cleanly on the production speed path. Matches discrete and scalar.
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
    cfg = build_processor_config(700)
    cfg.target_lang_code = "de"
    cfg.mt_backend_name = "gemma_vllm_alignatt"
    cfg.asr_commit_mode = "punctuation_lcp"

    print("Loading (vLLM MT cg=full chunk700, discrete)...", flush=True)
    load_start = perf_counter()
    CascadeAlignAttProcessor.load_model(cfg)
    print(f"Models loaded in {(perf_counter()-load_start):.1f}s", flush=True)

    processor = CascadeAlignAttProcessor(cfg)
    processor.set_source_language("en")
    processor.set_target_language("de")

    print(f"\n==== vllm_mt_chunk700_discrete ====", flush=True)
    batch_start = perf_counter()
    result = run_single_audio(
        processor, "test-set/audio/ccpXHNfaoy.wav", 700, "de", "en",
    )
    batch_wallclock_s = perf_counter() - batch_start
    print(
        f"  RTF={result['rtf']:.3f}  updates={result['num_updates']}  "
        f"wallclock={batch_wallclock_s:.1f}s",
        flush=True,
    )
    output_dir = "outputs/night1_ende_disc_vllm_cg_chunk700_FIXED"
    write_artifacts(result, output_dir, cfg, processor, batch_wallclock_s,
                    "test-set/audio/ccpXHNfaoy.wav")
    print(f"  Artifacts: {output_dir}", flush=True)
    print("\nDONE_CHUNK700", flush=True)


if __name__ == "__main__":
    main()
