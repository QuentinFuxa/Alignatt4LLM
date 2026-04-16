#!/usr/bin/env python3
"""Retry cs->en with the canonical gemma_vllm_alignatt MT backend.

The earlier attempt crashed with KeyError:
'_alignatt_mt_qk_tensor_observer' during vLLM's memory profiling
dummy_run. Commit c04356b+ installs a None-observer stub on every
Gemma4Attention layer at load_model time so the AOT-compiled
forward's dict lookup always finds the attribute. This run verifies
the fix.
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
    cfg.source_lang_code = "cs"
    cfg.target_lang_code = "en"
    cfg.mt_backend_name = "gemma_vllm_alignatt"

    print("Loading models (cold, with canonical vLLM MT) ...", flush=True)
    load_start = perf_counter()
    CascadeAlignAttProcessor.load_model(cfg)
    print(f"Models loaded in {(perf_counter()-load_start):.1f}s", flush=True)

    processor = CascadeAlignAttProcessor(cfg)
    processor.set_source_language("cs")
    processor.set_target_language("en")

    print(f"\n==== cs_en_vllm_mt_chunk450 ====", flush=True)
    batch_start = perf_counter()
    result = run_single_audio(
        processor, "test-set/audio/csJIsDTYMW.wav", 450, "en", "cs",
    )
    batch_wallclock_s = perf_counter() - batch_start
    print(
        f"  RTF={result['rtf']:.3f}  updates={result['num_updates']}  "
        f"wallclock={batch_wallclock_s:.1f}s",
        flush=True,
    )
    pred_head = result["final_translation"][:300]
    print(f"  Prediction head: {pred_head!r}", flush=True)
    output_dir = "outputs/night1_cs_en_vllm_mt_chunk450"
    write_artifacts(result, output_dir, cfg, processor, batch_wallclock_s,
                    "test-set/audio/csJIsDTYMW.wav")
    print(f"  Artifacts: {output_dir}", flush=True)
    print("\nDONE_CS_EN_VLLM", flush=True)


if __name__ == "__main__":
    main()
