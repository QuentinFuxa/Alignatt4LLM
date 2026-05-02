#!/usr/bin/env python3
"""Batch evaluation runner for the SimulStream CascadeAlignAttProcessor.

Runs multiple media files through the processor in a single process, keeping
models hot across audios to avoid repeated 5-minute load costs.

Usage (from .venv-inference):
    # Sanity set (3 audios):
    python run_simulstream_batch.py \\
        --inputs data/devset/audio/myfXyntFYL.wav data/devset/audio/DyXpuURBMP.wav data/devset/audio/ccpXHNfaoy.wav \\
        --output-dir outputs/simulstream_batch_ende_2s \\
        --chunk-ms 450 --target de

    # Full set (all supported media files in directory):
    python run_simulstream_batch.py \\
        --input-dir data/devset/audio/ \\
        --output-dir outputs/simulstream_fullset_ende_2s \\
        --chunk-ms 450 --target de
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np

from cascade.audio import discover_input_media_paths, load_audio_mono_16khz
from cascade.simulstream_processor import CascadeAlignAttProcessor, LANGUAGE_CODE_TO_NAME
from cascade.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
    HYPOTHESIS_FILENAME,
    MANIFEST_FILENAME,
    STREAM_UPDATE_ELAPSED_SEMANTICS_WALLCLOCK,
    STREAM_UPDATES_FILENAME,
    ensure_output_dir,
    normalize_computation_aware_timestamps,
    utc_now_isoformat,
    write_json,
    write_jsonl,
)
from cascade.text_surface import prediction_text_from_target_surface
from cascade.emission import register_translation_timestamps, register_translation_words
from simulstream.server.speech_processors import SAMPLE_RATE


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def run_single_audio(
    processor: CascadeAlignAttProcessor,
    input_path: str,
    chunk_ms: int,
    target_lang_code: str,
) -> dict[str, Any]:
    """Run one audio through the processor and return all artifacts data."""
    processor.clear()
    audio = load_audio_mono_16khz(input_path)
    chunk_size = int(SAMPLE_RATE * chunk_ms / 1000)
    audio_duration_ms = len(audio) * 1000.0 / SAMPLE_RATE
    input_name = Path(input_path).name

    word_delays_ms: list[float] = []
    word_elapsed_ms: list[float] = []
    stream_updates: list[dict[str, Any]] = []
    last_translation = ""
    last_raw_translation = ""
    start_time = perf_counter()

    for start_sample in range(0, len(audio), chunk_size):
        chunk = audio[start_sample : start_sample + chunk_size].astype(np.float32)
        output = processor.process_chunk(chunk)
        current_translation = processor.tokens_to_string(processor._emitted_units)
        audio_processed_ms = min((start_sample + chunk_size), len(audio)) * 1000.0 / SAMPLE_RATE
        wallclock_elapsed_ms = (perf_counter() - start_time) * 1000.0

        if output.new_tokens:
            register_translation_timestamps(
                last_raw_translation, current_translation,
                wallclock_elapsed_ms, word_elapsed_ms,
                target_lang_code=target_lang_code,
            )
            new_words = register_translation_words(
                last_translation, current_translation,
                audio_processed_ms, word_delays_ms,
                target_lang_code=target_lang_code,
            )
            partial = processor.session.state.partial_translation
            stream_updates.append({
                "update_idx": len(stream_updates),
                "input_name": input_name,
                "wav_name": input_name,
                "audio_processed_ms": audio_processed_ms,
                "wallclock_elapsed_ms": wallclock_elapsed_ms,
                "translation_text": current_translation,
                "new_words": new_words,
                # Observer / MT-state fields for offline replay
                # (continuous-confidence branch, emit-policy replay, etc.).
                # Optional — absent on older artifacts; consumers must tolerate.
                "asr_text": processor.session.render_public_asr_text(),
                "partial_accepted_target": partial.accepted_target,
                "partial_draft_target": partial.draft_target,
                "alignatt_metadata": partial.last_alignatt_metadata,
                "translation_prompt_num_tokens": partial.last_prompt_num_tokens,
                "translation_prompt_num_cached_tokens": partial.last_num_cached_tokens,
            })
            last_translation = current_translation
            last_raw_translation = current_translation

    eos_output = processor.end_of_stream()
    final_translation = processor.tokens_to_string(processor._emitted_units)
    final_elapsed_ms = (perf_counter() - start_time) * 1000.0

    if eos_output.new_tokens or final_translation != last_translation:
        register_translation_timestamps(
            last_raw_translation, final_translation,
            final_elapsed_ms, word_elapsed_ms,
            target_lang_code=target_lang_code,
        )
        eos_new_words = register_translation_words(
            last_translation, final_translation,
            audio_duration_ms, word_delays_ms,
            target_lang_code=target_lang_code,
        )
        partial = processor.session.state.partial_translation
        stream_updates.append({
            "update_idx": len(stream_updates),
            "input_name": input_name,
            "wav_name": input_name,
            "audio_processed_ms": audio_duration_ms,
            "wallclock_elapsed_ms": final_elapsed_ms,
            "translation_text": final_translation,
            "new_words": eos_new_words,
            "is_eos": True,
            "asr_text": processor.session.render_public_asr_text(),
            "partial_accepted_target": partial.accepted_target,
            "partial_draft_target": partial.draft_target,
            "alignatt_metadata": partial.last_alignatt_metadata,
            "translation_prompt_num_tokens": partial.last_prompt_num_tokens,
            "translation_prompt_num_cached_tokens": partial.last_num_cached_tokens,
        })

    final_asr = processor.session.render_public_asr_text()
    total_wallclock_s = perf_counter() - start_time
    rtf = total_wallclock_s / (audio_duration_ms / 1000.0) if audio_duration_ms > 0 else 0.0

    normalized_elapsed_ms = normalize_computation_aware_timestamps(word_delays_ms, word_elapsed_ms)
    prediction = prediction_text_from_target_surface(
        final_translation,
        target_lang_code=target_lang_code,
    )

    return {
        "input_path": input_path,
        "input_name": input_name,
        "wav_path": input_path,
        "wav_name": input_name,
        "audio_duration_ms": audio_duration_ms,
        "total_wallclock_s": total_wallclock_s,
        "rtf": rtf,
        "final_asr": final_asr,
        "final_translation": final_translation,
        "num_updates": len(stream_updates),
        "hypothesis_record": {
            "source": [input_name],
            "source_length": audio_duration_ms,
            "prediction": prediction,
            "delays": word_delays_ms,
            "elapsed": normalized_elapsed_ms,
            "elapsed_wallclock_ms": word_elapsed_ms,
            "elapsed_semantics": HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
        },
        "stream_updates": stream_updates,
    }


def resolve_input_paths(
    *,
    inputs: list[str] | None,
    input_dir: str | None,
) -> list[str]:
    if input_dir is not None:
        discovered = discover_input_media_paths(input_dir)
        filtered = [
            path for path in discovered
            if not Path(path).name.endswith("_short60s.wav")
        ]
        return filtered or discovered
    if not inputs:
        raise ValueError("Either explicit inputs or an input directory must be provided.")
    return [str(Path(path)) for path in inputs]


def resolve_paper_context_path_for_input(
    input_path: str,
    *,
    explicit_paper_context_path: str | None = None,
    paper_context_dir: str | None = None,
) -> str | None:
    if explicit_paper_context_path is not None and paper_context_dir is not None:
        raise ValueError("paper_context_path and paper_context_dir are mutually exclusive.")
    if explicit_paper_context_path is not None:
        return explicit_paper_context_path
    if paper_context_dir is None:
        return None
    candidate = Path(paper_context_dir) / f"{Path(input_path).stem}.json"
    if not candidate.exists():
        print(
            f"  [paper-context] no artifact for {Path(input_path).name} at {candidate}; "
            f"running without extra context for this input."
        )
        return None
    return str(candidate)


def run_batch_inference(
    *,
    processor_config: SimpleNamespace,
    input_paths: list[str],
    output_dir: str,
    source_lang_code: str,
    target_lang_code: str,
    explicit_paper_context_path: str | None = None,
    paper_context_dir: str | None = None,
) -> dict[str, Any]:
    print(f"Will process {len(input_paths)} media files for {source_lang_code}->{target_lang_code}")

    print("Loading models ...")
    load_start = perf_counter()
    CascadeAlignAttProcessor.load_model(processor_config)
    load_ms = (perf_counter() - load_start) * 1000.0
    print(f"Models loaded in {load_ms:.0f} ms")

    processor = CascadeAlignAttProcessor(processor_config)
    processor.set_source_language(source_lang_code)
    processor.set_target_language(target_lang_code)

    all_hypothesis_records: list[dict[str, Any]] = []
    all_stream_updates: list[dict[str, Any]] = []
    per_input_results: list[dict[str, Any]] = []
    batch_start = perf_counter()

    for idx, input_path in enumerate(input_paths):
        context_path = resolve_paper_context_path_for_input(
            input_path,
            explicit_paper_context_path=explicit_paper_context_path,
            paper_context_dir=paper_context_dir,
        )
        if hasattr(processor, "set_paper_context_path"):
            processor.set_paper_context_path(context_path)

        print(f"\n[{idx+1}/{len(input_paths)}] {Path(input_path).name} ...", flush=True)
        result = run_single_audio(
            processor,
            input_path,
            int(getattr(processor_config, "chunk_ms", 450)),
            target_lang_code,
        )
        all_hypothesis_records.append(result["hypothesis_record"])
        all_stream_updates.extend(result["stream_updates"])
        per_input_results.append({
            "input": result["input_name"],
            "audio_s": round(result["audio_duration_ms"] / 1000, 1),
            "rtf": round(result["rtf"], 4),
            "updates": result["num_updates"],
            "paper_context_path": context_path,
        })
        print(
            f"  RTF={result['rtf']:.3f}  updates={result['num_updates']}  "
            f"wallclock={result['total_wallclock_s']:.1f}s"
        )

    batch_wallclock_s = perf_counter() - batch_start
    total_audio_s = sum(entry["audio_s"] for entry in per_input_results)
    batch_rtf = batch_wallclock_s / total_audio_s if total_audio_s > 0 else 0.0

    output_path = ensure_output_dir(output_dir)
    runtime_config: dict[str, Any] = {
        "chunk_ms": getattr(processor_config, "chunk_ms"),
        "alignment_backend_name": getattr(processor_config, "alignment_backend_name"),
        "mt_backend_name": getattr(processor_config, "mt_backend_name"),
        "min_start_seconds": getattr(processor_config, "min_start_seconds"),
        "max_history_utterances": getattr(processor_config, "max_history_utterances"),
        "partial_max_new_tokens": getattr(processor_config, "partial_max_new_tokens"),
        "translation_alignatt_min_source_mass": getattr(
            processor_config, "translation_alignatt_min_source_mass"
        ),
        "translation_alignatt_border_margin": getattr(
            processor_config, "translation_alignatt_border_margin", 0
        ),
        "translation_alignatt_inaccessible_ms": getattr(
            processor_config, "translation_alignatt_inaccessible_ms"
        ),
        "translation_alignatt_argmax_mass_threshold": getattr(
            processor_config, "translation_alignatt_argmax_mass_threshold", 0.0
        ),
        "hypothesis_elapsed_semantics": HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
        "stream_update_elapsed_semantics": STREAM_UPDATE_ELAPSED_SEMANTICS_WALLCLOCK,
    }
    for key in [
        "translation_alignatt_heads_path", "translation_alignatt_top_k_heads",
        "translation_alignatt_filter_width", "translation_alignatt_probe_mode",
        "gemma_audio_alignment_heads_path",
        "translation_emit_policy", "translation_max_tail_rewrite_words",
        "temperature", "repetition_penalty",
        "asr_alignatt_frame_threshold",
        "asr_alignatt_rewind_threshold",
        "paper_context_path", "paper_context_mode", "paper_context_top_k",
        "paper_context_max_chars", "paper_context_history_window_words",
        "mt_vllm_enforce_eager", "mt_vllm_cudagraph_mode",
        "mt_vllm_enable_prefix_caching", "mt_vllm_gpu_memory_utilization",
    ]:
        runtime_config[key] = getattr(processor.session.config, key, None)

    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "generated_at_utc": utc_now_isoformat(),
        "kind": "inference_batch",
        "num_inputs": len(input_paths),
        "input_paths": input_paths,
        # Legacy aliases preserved for existing tooling.
        "num_audios": len(input_paths),
        "wav_paths": input_paths,
        "source_language": LANGUAGE_CODE_TO_NAME.get(source_lang_code, source_lang_code),
        "target_language": LANGUAGE_CODE_TO_NAME.get(target_lang_code, target_lang_code),
        "source_language_code": source_lang_code,
        "target_language_code": target_lang_code,
        "runtime_config": runtime_config,
        "run_provenance": {
            "git_sha": git_sha(),
            "framework_mode": "simulstream_processor",
            "script": "run_simulstream_batch.py",
        },
        "speed": {
            "batch_wallclock_s": round(batch_wallclock_s, 2),
            "batch_rtf": round(batch_rtf, 4),
            "total_audio_s": round(total_audio_s, 1),
            "per_input": per_input_results,
            "per_audio": per_input_results,
        },
    }

    write_json(output_path / MANIFEST_FILENAME, manifest)
    write_jsonl(output_path / HYPOTHESIS_FILENAME, all_hypothesis_records)
    write_jsonl(output_path / STREAM_UPDATES_FILENAME, all_stream_updates)

    print(f"\n{'='*60}")
    print(f"Batch complete: {len(input_paths)} inputs, {total_audio_s:.0f}s total audio")
    print(f"Batch wallclock: {batch_wallclock_s:.1f}s  RTF: {batch_rtf:.4f}")
    print(f"Artifacts: {output_dir}")
    print(f"Evaluate: python evaluate_cascade_outputs.py --output-dir {output_dir} --skip-comet")
    print(f"{'='*60}")

    return {
        "manifest": manifest,
        "hypothesis_records": all_hypothesis_records,
        "stream_updates": all_stream_updates,
        "output_dir": output_dir,
        "model_load_ms": load_ms,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch SimulStream evaluation runner.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--inputs",
        "--wavs",
        nargs="+",
        dest="inputs",
        help="List of input media paths (.wav, .mp4, ...).",
    )
    group.add_argument(
        "--input-dir",
        "--wav-dir",
        dest="input_dir",
        help="Directory of input media files (all supported files used).",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-ms", default=800, type=int)
    parser.add_argument("--source", default="en")
    parser.add_argument("--target", default="de")
    parser.add_argument(
        "--alignment-backend-name",
        default="qwen_forced",
        choices=("qwen_forced", "gemma_vllm_qk_fast"),
    )
    parser.add_argument(
        "--asr-alignatt-frame-threshold",
        default=4,
        type=int,
        help=(
            "AlignAtt token-level frontier gate in audio frames (simul_whisper "
            "§4). Lower = more aggressive commit, higher = safer."
        ),
    )
    parser.add_argument(
        "--asr-alignatt-rewind-threshold",
        default=200,
        type=int,
        help=(
            "Attention-collapse guard: abort the chunk if a generated token's "
            "argmax rewinds more than this many frames before the running "
            "reference."
        ),
    )
    parser.add_argument("--min-start-seconds", default=2.0, type=float)
    parser.add_argument("--max-history-utterances", default=0, type=int)
    parser.add_argument("--partial-max-new-tokens", default=16, type=int)
    parser.add_argument("--translation-alignatt-min-source-mass", default=0.0, type=float)
    parser.add_argument(
        "--translation-alignatt-border-margin",
        default=0,
        type=int,
        help=(
            "Source-token safety margin around the accessible frontier. "
            "Negative values are more conservative; 0 keeps the strict AlignAtt frontier; "
            "positive values allow speculative look-ahead beyond the border."
        ),
    )
    parser.add_argument("--translation-alignatt-inaccessible-ms", default=0.0, type=float)
    parser.add_argument(
        "--translation-alignatt-argmax-mass-threshold",
        default=0.0,
        type=float,
        help=(
            "Confidence-gated acceptance threshold on the reconstructed softmax "
            "mass at the argmax source position (per-head averaged). Default "
            "0.0 disables the gate and preserves argmax-only AlignAtt; raising "
            "it stops acceptance with reason 'alignatt:argmax_mass_weak' when "
            "the attention at the aligned source token is too diffuse."
        ),
    )
    parser.add_argument(
        "--mt-vllm-enable-prefix-caching",
        action="store_true",
        help=(
            "Enable vLLM prefix caching for the MT backend. Caches the stable "
            "prompt prefix (system + task instructions) across partial MT "
            "calls within an utterance. Source tokens live after the stable "
            "prefix, so the observer still captures their K on every prefill."
        ),
    )
    parser.add_argument(
        "--gemma-vllm-gpu-memory-utilization",
        type=float,
        default=None,
        help="vLLM gpu_memory_utilization for the Gemma ASR engine (default 0.5).",
    )
    parser.add_argument(
        "--mt-vllm-gpu-memory-utilization",
        type=float,
        default=None,
        help="vLLM gpu_memory_utilization for the MT engine (default 0.5).",
    )
    parser.add_argument(
        "--paper-context-path",
        default=None,
        help=(
            "Path to a PaperArtifact JSON (produced by "
            "cascade.paper_context.paper_artifact) to inject as MT-side [Paper "
            "context]. Default: no context."
        ),
    )
    parser.add_argument(
        "--paper-context-mode",
        default="off",
        choices=("off", "title_abstract", "retrieved_chunks", "title_and_chunks"),
        help=(
            "Context mechanism. 'off' disables injection, 'title_abstract' "
            "renders the paper's title+abstract, 'retrieved_chunks' BM25-"
            "retrieves paragraph chunks from the artifact using the current "
            "ASR prefix + recent source history as the query, and "
            "'title_and_chunks' combines both."
        ),
    )
    parser.add_argument("--paper-context-top-k", type=int, default=3)
    parser.add_argument("--paper-context-max-chars", type=int, default=1200)
    parser.add_argument("--paper-context-history-window-words", type=int, default=60)
    parser.add_argument(
        "--paper-context-dir",
        default=None,
        help=(
            "Directory containing one PaperArtifact JSON per input, matched by "
            "input stem (e.g. talk.mp4 -> talk.json). Useful for multi-talk "
            "extra-context runs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = resolve_input_paths(inputs=args.inputs, input_dir=args.input_dir)
    if args.paper_context_path is not None and args.paper_context_dir is not None:
        raise ValueError("Use either --paper-context-path or --paper-context-dir, not both.")

    processor_config = SimpleNamespace(
        source_lang_code=args.source,
        target_lang_code=args.target,
        chunk_ms=args.chunk_ms,
        speech_chunk_size=args.chunk_ms / 1000.0,
        alignment_backend_name=args.alignment_backend_name,
        mt_backend_name="gemma_vllm_alignatt",
        min_start_seconds=args.min_start_seconds,
        max_history_utterances=args.max_history_utterances,
        partial_max_new_tokens=args.partial_max_new_tokens,
        translation_alignatt_min_source_mass=args.translation_alignatt_min_source_mass,
        translation_alignatt_border_margin=args.translation_alignatt_border_margin,
        translation_alignatt_inaccessible_ms=args.translation_alignatt_inaccessible_ms,
        translation_alignatt_argmax_mass_threshold=args.translation_alignatt_argmax_mass_threshold,
        gemma_vllm_gpu_memory_utilization=args.gemma_vllm_gpu_memory_utilization,
        mt_vllm_enable_prefix_caching=args.mt_vllm_enable_prefix_caching,
        mt_vllm_gpu_memory_utilization=args.mt_vllm_gpu_memory_utilization,
        asr_alignatt_frame_threshold=args.asr_alignatt_frame_threshold,
        asr_alignatt_rewind_threshold=args.asr_alignatt_rewind_threshold,
        paper_context_path=args.paper_context_path,
        paper_context_mode=args.paper_context_mode,
        paper_context_top_k=args.paper_context_top_k,
        paper_context_max_chars=args.paper_context_max_chars,
        paper_context_history_window_words=args.paper_context_history_window_words,
    )
    run_batch_inference(
        processor_config=processor_config,
        input_paths=input_paths,
        output_dir=args.output_dir,
        source_lang_code=args.source,
        target_lang_code=args.target,
        explicit_paper_context_path=args.paper_context_path,
        paper_context_dir=args.paper_context_dir,
    )


if __name__ == "__main__":
    main()
