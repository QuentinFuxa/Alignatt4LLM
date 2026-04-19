#!/usr/bin/env python3
"""Step 6 follow-ups: min_source_mass sweep + emit_policy A/B.

All on ccpXHNfaoy.wav at chunk_ms=450 with the canonical pair
(qwen_forced + gemma_vllm_alignatt + punctuation_lcp). Models load
once and stay hot across all experiments.
"""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cascade.simulstream_processor import CascadeAlignAttProcessor
from run_simulstream_batch import run_single_audio
from tmp.reanchor_baseline import write_artifacts, build_processor_config


EXPERIMENTS = [
    # min_source_mass sweep (Step 6a). ms=0.0 is already measured in the
    # chunk_ms=450 re-anchor baseline (outputs/reanchor_chunk450), so we
    # pick two non-zero points to bracket the curve.
    dict(tag="ms10_punct", min_source_mass=0.1, emit_policy="raw_passthrough"),
    dict(tag="ms20_punct", min_source_mass=0.2, emit_policy="raw_passthrough"),
    # emit-policy A/B (Step 6b) — min_source_mass held at 0 (baseline).
    dict(tag="ms00_freeze", min_source_mass=0.0, emit_policy="freeze_nonexpanding_major_rewrites"),
]


def main():
    first = EXPERIMENTS[0]
    initial_cfg = build_processor_config(450)
    initial_cfg.target_lang_code = "de"
    initial_cfg.translation_alignatt_min_source_mass = first["min_source_mass"]

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
            translation_alignatt_min_source_mass=exp["min_source_mass"],
            translation_emit_policy=exp["emit_policy"],
        )
        cfg = build_processor_config(450)
        cfg.target_lang_code = "de"
        cfg.translation_alignatt_min_source_mass = exp["min_source_mass"]
        cfg.translation_emit_policy = exp["emit_policy"]

        batch_start = perf_counter()
        result = run_single_audio(
            processor, "test-set/audio/ccpXHNfaoy.wav", 450, "de", "en",
        )
        batch_wallclock_s = perf_counter() - batch_start
        print(
            f"  RTF={result['rtf']:.3f}  updates={result['num_updates']}  "
            f"wallclock={batch_wallclock_s:.1f}s  "
            f"min_mass={exp['min_source_mass']} policy={exp['emit_policy']}",
            flush=True,
        )
        output_dir = f"outputs/night1_step6_{exp['tag']}"
        write_artifacts(result, output_dir, cfg, processor, batch_wallclock_s,
                        "test-set/audio/ccpXHNfaoy.wav")
        print(f"  Artifacts: {output_dir}", flush=True)

    print("\nDONE_STEP6", flush=True)


if __name__ == "__main__":
    main()
