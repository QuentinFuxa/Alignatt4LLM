#!/usr/bin/env python3
"""Extend the stable_and_accessible K-sweep to K=5 and K=6 on ccpXHNfaoy.wav
at chunk_ms=450. Pins the saturation curve so the paper can report a
full ablation rather than a truncated one.
"""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cascade_simulstream_processor import CascadeAlignAttProcessor
from run_simulstream_batch import run_single_audio
from tmp.reanchor_baseline import write_artifacts, build_processor_config


EXPERIMENTS = [
    dict(tag="ende_stable_k5_chunk450", asr_stability_k=5),
    dict(tag="ende_stable_k6_chunk450", asr_stability_k=6),
]


def main():
    initial_cfg = build_processor_config(450)
    initial_cfg.target_lang_code = "de"
    initial_cfg.asr_commit_mode = "stable_and_accessible"
    initial_cfg.asr_stability_k = EXPERIMENTS[0]["asr_stability_k"]

    print("Loading models (cold) ...", flush=True)
    load_start = perf_counter()
    CascadeAlignAttProcessor.load_model(initial_cfg)
    print(f"Models loaded in {(perf_counter()-load_start):.1f}s", flush=True)

    processor = CascadeAlignAttProcessor(initial_cfg)
    processor.set_source_language("en")
    processor.set_target_language("de")

    for exp in EXPERIMENTS:
        print(f"\n==== {exp['tag']} ====", flush=True)
        processor.session.config.apply_overrides(
            asr_commit_mode="stable_and_accessible",
            asr_stability_k=exp["asr_stability_k"],
        )
        cfg = build_processor_config(450)
        cfg.target_lang_code = "de"
        cfg.asr_commit_mode = "stable_and_accessible"
        cfg.asr_stability_k = exp["asr_stability_k"]

        batch_start = perf_counter()
        result = run_single_audio(
            processor, "test-set/audio/ccpXHNfaoy.wav", 450, "de", "en",
        )
        batch_wallclock_s = perf_counter() - batch_start
        print(
            f"  RTF={result['rtf']:.3f}  updates={result['num_updates']}  "
            f"wallclock={batch_wallclock_s:.1f}s  K={exp['asr_stability_k']}",
            flush=True,
        )
        output_dir = f"outputs/night1_{exp['tag']}"
        write_artifacts(result, output_dir, cfg, processor, batch_wallclock_s,
                        "test-set/audio/ccpXHNfaoy.wav")
        print(f"  Artifacts: {output_dir}", flush=True)

    print("\nDONE_K_SATURATION", flush=True)


if __name__ == "__main__":
    main()
