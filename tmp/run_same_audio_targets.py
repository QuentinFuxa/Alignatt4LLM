#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from cascade.artifacts import (
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
from cascade.simulstream_processor import CascadeAlignAttProcessor, LANGUAGE_CODE_TO_NAME
from run_simulstream_batch import git_sha, run_single_audio


RUNTIME_KEYS = [
    "chunk_ms",
    "alignment_backend_name",
    "mt_backend_name",
    "min_start_seconds",
    "max_history_utterances",
    "partial_max_new_tokens",
    "translation_alignatt_min_source_mass",
    "translation_alignatt_rewind_threshold",
    "translation_alignatt_border_margin",
    "translation_alignatt_inaccessible_ms",
    "translation_alignatt_argmax_mass_threshold",
    "hypothesis_elapsed_semantics",
    "stream_update_elapsed_semantics",
    "translation_alignatt_heads_path",
    "translation_alignatt_top_k_heads",
    "translation_alignatt_filter_width",
    "translation_alignatt_probe_mode",
    "gemma_audio_alignment_heads_path",
    "gemma_audio_align_probe_mode",
    "translation_emit_policy",
    "translation_max_tail_rewrite_words",
    "temperature",
    "repetition_penalty",
    "asr_streaming_prefix_enabled",
    "asr_streaming_rollback_words",
    "asr_streaming_unfixed_chunks",
    "gemma_vllm_force_generate_api",
    "paper_context_path",
    "paper_context_mode",
    "paper_context_top_k",
    "paper_context_max_chars",
    "paper_context_history_window_words",
    "mt_vllm_enforce_eager",
    "mt_vllm_cudagraph_mode",
    "mt_vllm_enable_prefix_caching",
    "mt_vllm_gpu_memory_utilization",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one audio sequentially for multiple target languages in one hot process."
    )
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument(
        "--input-path",
        default=None,
        help="Override the input audio path instead of reusing the base manifest input.",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=None,
        help="Override the chunk size instead of reusing the base manifest value.",
    )
    parser.add_argument(
        "--translation-alignatt-border-margin",
        type=int,
        default=None,
        help="Override the AlignAtt source-frontier border margin.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_manifest_path = Path(args.base_manifest)
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    base_runtime = dict(base_manifest["runtime_config"])
    source_lang_code = str(base_manifest["source_language_code"])
    input_path = str(args.input_path or base_manifest["input_paths"][0])
    chunk_ms = int(args.chunk_ms or base_runtime["chunk_ms"])

    config_kwargs = dict(base_runtime)
    config_kwargs.update(
        {
            "source_lang_code": source_lang_code,
            "target_lang_code": str(args.targets[0]),
            "chunk_ms": chunk_ms,
        }
    )
    if args.translation_alignatt_border_margin is not None:
        config_kwargs["translation_alignatt_border_margin"] = int(
            args.translation_alignatt_border_margin
        )
    processor_config = SimpleNamespace(**config_kwargs)

    print(f"Loading models once from base manifest: {base_manifest_path}", flush=True)
    CascadeAlignAttProcessor.load_model(processor_config)
    processor = CascadeAlignAttProcessor(processor_config)
    processor.set_source_language(source_lang_code)

    stem = Path(input_path).stem
    base_prefix = args.output_prefix
    if base_prefix is None:
        base_prefix = f"outputs/real_audio_{stem}_chunk{chunk_ms}_qwenforced_gemma_alignatt_charsurface"

    for target in args.targets:
        print(f"\n=== Running {source_lang_code}->{target} on {input_path} ===", flush=True)
        processor.set_target_language(target)
        result = run_single_audio(processor, input_path, chunk_ms, target)
        output_dir = f"{base_prefix}_en{target}"
        output_path = ensure_output_dir(output_dir)

        session_cfg = processor.session.config
        runtime_config = {key: getattr(session_cfg, key, None) for key in RUNTIME_KEYS}
        runtime_config["chunk_ms"] = chunk_ms
        runtime_config["hypothesis_elapsed_semantics"] = HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE
        runtime_config["stream_update_elapsed_semantics"] = STREAM_UPDATE_ELAPSED_SEMANTICS_WALLCLOCK

        per_input = {
            "input": result["input_name"],
            "audio_s": round(result["audio_duration_ms"] / 1000.0, 1),
            "rtf": round(result["rtf"], 4),
            "updates": result["num_updates"],
            "paper_context_path": getattr(session_cfg, "paper_context_path", None),
        }

        manifest = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "generated_at_utc": utc_now_isoformat(),
            "kind": "inference_batch",
            "num_inputs": 1,
            "input_paths": [input_path],
            "num_audios": 1,
            "wav_paths": [input_path],
            "source_language": LANGUAGE_CODE_TO_NAME.get(source_lang_code, source_lang_code),
            "target_language": LANGUAGE_CODE_TO_NAME.get(target, target),
            "source_language_code": source_lang_code,
            "target_language_code": target,
            "runtime_config": runtime_config,
            "run_provenance": {
                "git_sha": git_sha(),
                "framework_mode": "simulstream_processor",
                "script": "tmp/run_same_audio_targets.py",
                "base_manifest": str(base_manifest_path),
            },
            "speed": {
                "batch_wallclock_s": round(result["total_wallclock_s"], 2),
                "batch_rtf": round(result["rtf"], 4),
                "total_audio_s": round(result["audio_duration_ms"] / 1000.0, 1),
                "per_input": [per_input],
                "per_audio": [per_input],
            },
        }

        write_json(output_path / MANIFEST_FILENAME, manifest)
        write_jsonl(output_path / HYPOTHESIS_FILENAME, [result["hypothesis_record"]])
        write_jsonl(output_path / STREAM_UPDATES_FILENAME, result["stream_updates"])
        print(
            f"DONE {source_lang_code}->{target}: updates={result['num_updates']} "
            f"rtf={result['rtf']:.4f} out={output_dir}",
            flush=True,
        )

    print("\nAll target runs completed.", flush=True)


if __name__ == "__main__":
    main()
