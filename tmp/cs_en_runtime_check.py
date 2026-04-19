#!/usr/bin/env python3
"""cs→en runtime validation. No reference file locally, so we do NOT
evaluate — goal is proving the full end-to-end pipeline handles
Czech source + English target without crashing, under the Step 1
language-map fix.
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
    cfg = build_processor_config(450)
    cfg.source_lang_code = "cs"
    cfg.target_lang_code = "en"
    # Transformers MT here sidesteps a torch.compile-cache interaction
    # with the MT vLLM observer init that surfaced on the first cs->en
    # run of the night. The point of this check is Step 1's language-map
    # + heads-path fixes, not the MT engine choice. Captured in DECISIONS.
    cfg.mt_backend_name = "gemma_transformers_alignatt"

    print("Loading models (cold) ...", flush=True)
    load_start = perf_counter()
    CascadeAlignAttProcessor.load_model(cfg)
    print(f"Models loaded in {(perf_counter()-load_start):.1f}s", flush=True)

    processor = CascadeAlignAttProcessor(cfg)
    processor.set_source_language("cs")
    processor.set_target_language("en")

    print(f"\n==== cs_en_cschunk450 ====", flush=True)
    batch_start = perf_counter()
    result = run_single_audio(
        processor, "test-set/audio/csJIsDTYMW.wav", 450, "en", "cs",
    )
    batch_wallclock_s = perf_counter() - batch_start
    print(
        f"  RTF={result['rtf']:.3f}  updates={result['num_updates']}  "
        f"wallclock={batch_wallclock_s:.1f}s  cs->en chunk_ms=450",
        flush=True,
    )
    pred_head = result["final_translation"][:300]
    print(f"  Prediction head: {pred_head!r}", flush=True)
    output_dir = "outputs/night1_cs_en_chunk450"
    write_artifacts(result, output_dir, cfg, processor, batch_wallclock_s,
                    "test-set/audio/csJIsDTYMW.wav")
    print(f"  Artifacts: {output_dir}", flush=True)
    print("\nDONE_CS_EN", flush=True)


if __name__ == "__main__":
    main()
