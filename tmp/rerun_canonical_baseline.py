#!/usr/bin/env python3
"""Regenerate the canonical en->de submission baseline with the new
instrumented stream_updates schema (observer metadata per update).

The existing reanchor_chunk450 artifact was produced before commit
a0edcc6, so its stream_updates.jsonl carries only translation_text
+ new_words. This rerun produces the same numerical result but with
alignatt_metadata per update, usable for loop-replay validation on
the primary submission path.
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
    # Default cfg already uses qwen_forced ASR + gemma_vllm_alignatt MT
    # + punctuation_lcp. Confirm explicitly for artefact provenance.
    cfg.target_lang_code = "de"
    cfg.mt_backend_name = "gemma_vllm_alignatt"
    cfg.asr_commit_mode = "punctuation_lcp"

    print("Loading models (cold, instrumented schema) ...", flush=True)
    load_start = perf_counter()
    CascadeAlignAttProcessor.load_model(cfg)
    print(f"Models loaded in {(perf_counter()-load_start):.1f}s", flush=True)

    processor = CascadeAlignAttProcessor(cfg)
    processor.set_source_language("en")
    processor.set_target_language("de")

    print(f"\n==== ende_punct_chunk450_instrumented ====", flush=True)
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
    output_dir = "outputs/night1_ende_punct_chunk450_instrumented"
    write_artifacts(result, output_dir, cfg, processor, batch_wallclock_s,
                    "test-set/audio/ccpXHNfaoy.wav")
    print(f"  Artifacts: {output_dir}", flush=True)
    print("\nDONE_CANONICAL_INSTRUMENTED", flush=True)


if __name__ == "__main__":
    main()
