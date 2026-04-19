#!/usr/bin/env python3
"""Cross-latency ablation: stable_and_accessible K=3 at chunk_ms=700.

Tests whether the frontier-family rule's MT-context-fragmentation
penalty is milder at longer chunks (where each chunk already delivers
more source context to MT). Also exercises the newly instrumented
stream_updates.jsonl schema (alignatt_metadata per update) so the
artifacts are usable for future offline continuous-confidence replay.
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
    cfg.asr_commit_mode = "stable_and_accessible"
    cfg.asr_stability_k = 3

    print("Loading models (cold) ...", flush=True)
    load_start = perf_counter()
    CascadeAlignAttProcessor.load_model(cfg)
    print(f"Models loaded in {(perf_counter()-load_start):.1f}s", flush=True)

    processor = CascadeAlignAttProcessor(cfg)
    processor.set_source_language("en")
    processor.set_target_language("de")

    print(f"\n==== ende_stable_k3_chunk700 ====", flush=True)
    processor.session.config.apply_overrides(
        asr_commit_mode="stable_and_accessible",
        asr_stability_k=3,
    )
    batch_start = perf_counter()
    result = run_single_audio(
        processor, "test-set/audio/ccpXHNfaoy.wav", 700, "de", "en",
    )
    batch_wallclock_s = perf_counter() - batch_start
    print(
        f"  RTF={result['rtf']:.3f}  updates={result['num_updates']}  "
        f"wallclock={batch_wallclock_s:.1f}s  K=3 chunk_ms=700",
        flush=True,
    )
    output_dir = "outputs/night1_ende_stable_k3_chunk700"
    write_artifacts(result, output_dir, cfg, processor, batch_wallclock_s,
                    "test-set/audio/ccpXHNfaoy.wav")
    print(f"  Artifacts: {output_dir}", flush=True)
    print("\nDONE_CROSS_LATENCY", flush=True)


if __name__ == "__main__":
    main()
