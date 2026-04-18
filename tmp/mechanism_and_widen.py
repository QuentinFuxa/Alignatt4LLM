#!/usr/bin/env python3
"""Step 3 + Step 4 driver: run the stable_and_accessible mechanism branch
and widen the canonical pair to en→it and en→zh.

All experiments share one model load. Order minimises heads-path swaps:

    1. en→de stable_and_accessible K=3 @ chunk_ms=450
    2. en→de stable_and_accessible K=4 @ chunk_ms=450  (K-ablation)
    3. en→it punctuation_lcp           @ chunk_ms=450  (widen)
    4. en→zh punctuation_lcp           @ chunk_ms=450  (widen)

Output directories land under outputs/night1_* so they can be evaluated
after the driver exits.
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


def build_cfg(**overrides):
    cfg = build_processor_config(overrides.pop("chunk_ms", 450))
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


EXPERIMENTS = [
    dict(
        tag="ende_stable_k3_chunk450",
        wav="test-set/audio/ccpXHNfaoy.wav",
        target="de",
        chunk_ms=450,
        asr_commit_mode="stable_and_accessible",
        asr_stability_k=3,
    ),
    dict(
        tag="ende_stable_k4_chunk450",
        wav="test-set/audio/ccpXHNfaoy.wav",
        target="de",
        chunk_ms=450,
        asr_commit_mode="stable_and_accessible",
        asr_stability_k=4,
    ),
    dict(
        tag="enit_punct_chunk450",
        wav="test-set/audio/ccpXHNfaoy.wav",
        target="it",
        chunk_ms=450,
        asr_commit_mode="punctuation_lcp",
    ),
    dict(
        tag="enzh_punct_chunk450",
        wav="test-set/audio/ccpXHNfaoy.wav",
        target="zh",
        chunk_ms=450,
        asr_commit_mode="punctuation_lcp",
    ),
]


def main():
    first = EXPERIMENTS[0]
    initial_cfg = build_cfg(
        chunk_ms=first["chunk_ms"],
        asr_commit_mode=first["asr_commit_mode"],
        asr_stability_k=first.get("asr_stability_k", 3),
    )
    initial_cfg.target_lang_code = first["target"]

    print(f"Loading models (cold) ...", flush=True)
    load_start = perf_counter()
    CascadeAlignAttProcessor.load_model(initial_cfg)
    print(f"Models loaded in {(perf_counter()-load_start):.1f}s", flush=True)

    processor = CascadeAlignAttProcessor(initial_cfg)
    processor.set_source_language("en")
    processor.set_target_language(first["target"])

    for exp in EXPERIMENTS:
        print(f"\n==== {exp['tag']} ====", flush=True)
        processor.set_target_language(exp["target"])
        # Apply live overrides: commit mode, K, chunk (chunk only matters per-call)
        overrides = {
            "asr_commit_mode": exp["asr_commit_mode"],
        }
        if "asr_stability_k" in exp:
            overrides["asr_stability_k"] = exp["asr_stability_k"]
        processor.session.config.apply_overrides(**overrides)

        cfg = build_cfg(
            chunk_ms=exp["chunk_ms"],
            asr_commit_mode=exp["asr_commit_mode"],
            asr_stability_k=exp.get("asr_stability_k", 3),
        )
        cfg.target_lang_code = exp["target"]

        batch_start = perf_counter()
        result = run_single_audio(
            processor, exp["wav"], exp["chunk_ms"],
            exp["target"], "en",
        )
        batch_wallclock_s = perf_counter() - batch_start
        print(
            f"  RTF={result['rtf']:.3f}  updates={result['num_updates']}  "
            f"wallclock={batch_wallclock_s:.1f}s  "
            f"mode={exp['asr_commit_mode']} K={exp.get('asr_stability_k','-')}",
            flush=True,
        )
        output_dir = f"outputs/night1_{exp['tag']}"
        write_artifacts(result, output_dir, cfg, processor, batch_wallclock_s, exp["wav"])
        print(f"  Artifacts: {output_dir}", flush=True)

    print("\nDONE_ALL_EXPERIMENTS", flush=True)


if __name__ == "__main__":
    main()
