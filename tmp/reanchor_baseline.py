#!/usr/bin/env python3
"""Step 2 re-anchor driver: run ccpXHNfaoy.wav at chunk_ms=450 and 700
with qwen_forced + gemma_vllm_alignatt, keeping models hot across both
runs. Writes outputs under outputs/reanchor_chunk{450,700}/.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cascade_simulstream_processor import CascadeAlignAttProcessor, LANGUAGE_CODE_TO_NAME
from cascade_artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
    HYPOTHESIS_FILENAME,
    MANIFEST_FILENAME,
    STREAM_UPDATE_ELAPSED_SEMANTICS_WALLCLOCK,
    STREAM_UPDATES_FILENAME,
    ensure_output_dir,
    utc_now_isoformat,
    write_json,
    write_jsonl,
)
from run_simulstream_batch import git_sha, run_single_audio


def build_processor_config(chunk_ms: int) -> SimpleNamespace:
    return SimpleNamespace(
        source_lang_code="en",
        target_lang_code="de",
        chunk_ms=chunk_ms,
        speech_chunk_size=chunk_ms / 1000.0,
        alignment_backend_name="qwen_forced",
        mt_backend_name="gemma_vllm_alignatt",
        min_start_seconds=2.0,
        max_history_utterances=1,
        partial_max_new_tokens=16,
        partial_followup_max_new_tokens=8,
        translation_alignatt_min_source_mass=0.0,
        translation_alignatt_rewind_threshold=8,
        translation_alignatt_inaccessible_ms=0.0,
        asr_streaming_prefix_enabled=False,
        asr_streaming_rollback_words=2,
        asr_streaming_unfixed_chunks=2,
        gemma_vllm_force_generate_api=False,
        asr_commit_mode="punctuation_lcp",
        asr_alignatt_frontier_margin_ms=500.0,
        asr_stability_k=3,
    )


def write_artifacts(result: dict, output_dir: str, cfg: SimpleNamespace,
                    processor: CascadeAlignAttProcessor,
                    batch_wallclock_s: float, wav_path: str):
    output_path = ensure_output_dir(output_dir)
    runtime_config: dict[str, Any] = {
        "chunk_ms": cfg.chunk_ms,
        "alignment_backend_name": cfg.alignment_backend_name,
        "mt_backend_name": cfg.mt_backend_name,
        "min_start_seconds": cfg.min_start_seconds,
        "max_history_utterances": cfg.max_history_utterances,
        "partial_max_new_tokens": cfg.partial_max_new_tokens,
        "partial_followup_max_new_tokens": cfg.partial_followup_max_new_tokens,
        "translation_alignatt_min_source_mass": cfg.translation_alignatt_min_source_mass,
        "translation_alignatt_rewind_threshold": cfg.translation_alignatt_rewind_threshold,
        "translation_alignatt_inaccessible_ms": cfg.translation_alignatt_inaccessible_ms,
        "hypothesis_elapsed_semantics": HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
        "stream_update_elapsed_semantics": STREAM_UPDATE_ELAPSED_SEMANTICS_WALLCLOCK,
    }
    for key in [
        "translation_alignatt_heads_path", "translation_alignatt_top_k_heads",
        "translation_alignatt_filter_width", "translation_alignatt_probe_mode",
        "gemma_audio_alignment_heads_path", "gemma_audio_align_probe_mode",
        "translation_emit_policy", "translation_max_tail_rewrite_words",
        "temperature", "repetition_penalty",
        "asr_streaming_prefix_enabled", "asr_streaming_rollback_words",
        "asr_streaming_unfixed_chunks",
        "gemma_vllm_force_generate_api",
        "asr_commit_mode", "asr_alignatt_frontier_margin_ms",
    ]:
        runtime_config[key] = getattr(processor.session.config, key, None)

    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "generated_at_utc": utc_now_isoformat(),
        "kind": "inference_batch",
        "num_audios": 1,
        "wav_paths": [wav_path],
        "source_language": LANGUAGE_CODE_TO_NAME["en"],
        "target_language": LANGUAGE_CODE_TO_NAME["de"],
        "source_language_code": "en",
        "target_language_code": "de",
        "runtime_config": runtime_config,
        "run_provenance": {
            "git_sha": git_sha(),
            "framework_mode": "simulstream_processor",
            "script": "tmp/reanchor_baseline.py",
        },
        "speed": {
            "batch_wallclock_s": round(batch_wallclock_s, 2),
            "batch_rtf": round(result["rtf"], 4),
            "total_audio_s": round(result["audio_duration_ms"] / 1000, 1),
            "per_audio": [{
                "wav": result["wav_name"],
                "audio_s": round(result["audio_duration_ms"] / 1000, 1),
                "rtf": round(result["rtf"], 4),
                "updates": result["num_updates"],
            }],
        },
    }

    write_json(output_path / MANIFEST_FILENAME, manifest)
    write_jsonl(output_path / HYPOTHESIS_FILENAME, [result["hypothesis_record"]])
    write_jsonl(output_path / STREAM_UPDATES_FILENAME, result["stream_updates"])


def main():
    wav_path = "test-set/audio/ccpXHNfaoy.wav"
    chunks = [450, 700]

    print(f"Loading models (cold) with chunk_ms={chunks[0]} ...", flush=True)
    load_start = perf_counter()
    initial_cfg = build_processor_config(chunks[0])
    CascadeAlignAttProcessor.load_model(initial_cfg)
    print(f"Models loaded in {(perf_counter()-load_start):.1f}s", flush=True)

    processor = CascadeAlignAttProcessor(initial_cfg)
    processor.set_source_language("en")
    processor.set_target_language("de")

    for chunk_ms in chunks:
        print(f"\n==== chunk_ms={chunk_ms} ====", flush=True)
        cfg = build_processor_config(chunk_ms)
        processor._processor_config = cfg
        # Update runtime config live for chunk-size-dependent runtime knobs.
        processor.session.config.apply_overrides()
        batch_start = perf_counter()
        result = run_single_audio(
            processor, wav_path, chunk_ms, "de", "en",
        )
        batch_wallclock_s = perf_counter() - batch_start
        print(f"  RTF={result['rtf']:.3f}  updates={result['num_updates']}  "
              f"wallclock={batch_wallclock_s:.1f}s", flush=True)
        output_dir = f"outputs/reanchor_chunk{chunk_ms}"
        write_artifacts(result, output_dir, cfg, processor, batch_wallclock_s, wav_path)
        print(f"  Artifacts: {output_dir}", flush=True)

    print("\nDONE_ALL_CHUNKS", flush=True)


if __name__ == "__main__":
    main()
