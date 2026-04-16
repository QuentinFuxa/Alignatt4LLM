#!/usr/bin/env python3
"""Run GPU experiments for Phase 4/5 of PLAN.md.

Loads models once, then runs:
1. Phase 5: en->de with min_source_mass thresholds (0.1, 0.2, 0.3)
2. Phase 4: en->it with shared_kernel heads

All runs use the Phase 0 operating point on the control audio (ccpXHNfaoy.wav).
"""
from __future__ import annotations

from qwen3asr_gemma_cascade_core import load_models, run_baseline

PHASE0_OVERRIDES = {
    "min_start_seconds": 2.0,
    "partial_max_new_tokens": 16,
    "partial_followup_max_new_tokens": 8,
    "max_history_utterances": 1,
    "translation_alignatt_inaccessible_ms": 0.0,
    "translation_alignatt_rewind_threshold": 8,
}

WAV = "test-set/audio/ccpXHNfaoy.wav"
CHUNK_MS = 450


def run_experiment(output_dir: str, extra_overrides: dict | None = None):
    overrides = dict(PHASE0_OVERRIDES)
    if extra_overrides:
        overrides.update(extra_overrides)
    run_baseline(
        wav_path=WAV,
        output_dir=output_dir,
        chunk_ms=CHUNK_MS,
        runtime_overrides=overrides,
    )


def main():
    print("Loading models (ASR + Gemma)...")
    load_models()
    print("Models loaded.\n")

    # Phase 5: provenance-aware acceptance with various thresholds
    for mass_threshold in [0.1, 0.2, 0.3]:
        tag = f"phase5_v1_ende_minmass{int(mass_threshold * 100)}"
        print(f"\n{'='*60}")
        print(f"Running {tag} (min_source_mass={mass_threshold})")
        print(f"{'='*60}")
        run_experiment(
            f"outputs/{tag}",
            {"translation_alignatt_min_source_mass": mass_threshold},
        )

    # Phase 4: en->it with shared_kernel heads
    print(f"\n{'='*60}")
    print("Running phase4_v4_enit_shared_kernel")
    print(f"{'='*60}")
    run_experiment(
        "outputs/phase4_v4_enit_shared_kernel",
        {
            "target_lang": "Italian",
            "translation_alignatt_heads_path": "assets/attention_heads/translation_heads_shared_kernel_top8.json",
        },
    )

    print("\n\nAll experiments complete. Run evaluate_cascade_outputs.py on each output dir.")


if __name__ == "__main__":
    main()
