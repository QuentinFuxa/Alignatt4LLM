#!/usr/bin/env python
"""Single-audio diagnostic harness for the Gemma-only aligner research path.

This is the single-audio diagnostic tool retained from the historical plan
notes now archived under ``docs/archive/``: collect, on one carefully chosen
clip, what each
alignment backend produces so that head selection, quality, and streaming
behavior can be diagnosed manually before any broader run.

Modes:

- ``baseline`` produces the Qwen teacher bundle for a clip and writes it
  as a JSON artifact.
- ``gemma_inspect`` runs the Gemma attention aligner on the same clip,
  using previously calibrated heads if available, and writes its
  transcript + word timestamps + per-token audio positions.
- ``gemma_forced_align`` runs teacher-forced Gemma alignment on a known
  transcript; ``--probe-mode eager|qk_fast`` selects the attention
  extraction backend explicitly.
- ``gemma_calibrate_heads`` runs Gemma once with full-attention capture
  and ranks every (layer, head) against the Qwen teacher, writing the
  top-K into ``assets/attention_heads/audio_alignment_heads_*.json``.
- ``compare`` loads both bundles and prints a simple word-end-MAE table.

Run with the already-hot ``.venv-inference`` kernel — model reloads are
expensive, so prefer reusing a warm process.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from cascade.alignment.base import (
    AlignAttObserverToken,
    AlignAttProvenanceBreakdown,
    AlignmentBackend,
    AlignmentResult,
    WordAlignment,
)


def load_wav(path: str) -> tuple[np.ndarray, int]:
    import wave

    with wave.open(path, "rb") as wav:
        sr = wav.getframerate()
        width = wav.getsampwidth()
        ch = wav.getnchannels()
        raw = wav.readframes(wav.getnframes())
    if width != 2:
        raise ValueError("Only 16-bit PCM WAV is supported in this harness.")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        duration = len(audio) / sr
        new_length = int(duration * 16000)
        old_times = np.linspace(0.0, duration, num=len(audio), endpoint=False)
        new_times = np.linspace(0.0, duration, num=new_length, endpoint=False)
        audio = np.interp(new_times, old_times, audio).astype(np.float32)
        sr = 16000
    return audio, sr


def serialize_alignment(result: AlignmentResult) -> dict:
    return {
        "text": result.text,
        "audio_duration_s": result.audio_duration_s,
        "words": [asdict(w) for w in result.words],
        "observer_tokens": [asdict(token) for token in result.observer_tokens],
        "diagnostics": {
            k: v for k, v in result.diagnostics.items() if _is_jsonable(v)
        },
    }


def _is_jsonable(value) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def deserialize_alignment(payload: dict) -> AlignmentResult:
    words = tuple(
        WordAlignment(
            text=entry["text"],
            start_time=float(entry["start_time"]),
            end_time=float(entry["end_time"]),
        )
        for entry in payload.get("words", [])
    )
    observer_tokens = []
    for entry in payload.get("observer_tokens", []) or []:
        provenance_payload = entry.get("provenance")
        provenance = None
        if provenance_payload is not None:
            provenance = AlignAttProvenanceBreakdown(
                source_accessible=float(provenance_payload["source_accessible"]),
                source_inaccessible=float(provenance_payload["source_inaccessible"]),
                non_source_prompt=float(provenance_payload["non_source_prompt"]),
                suffix=float(provenance_payload["suffix"]),
            )
        observer_tokens.append(
            AlignAttObserverToken(
                token_id=int(entry["token_id"]),
                token_str=str(entry["token_str"]),
                aligned_source_position=(
                    None
                    if entry.get("aligned_source_position") is None
                    else int(entry["aligned_source_position"])
                ),
                source_accessible_mass=(
                    None
                    if entry.get("source_accessible_mass") is None
                    else float(entry["source_accessible_mass"])
                ),
                blocked_source_local_position=(
                    None
                    if entry.get("blocked_source_local_position") is None
                    else int(entry["blocked_source_local_position"])
                ),
                blocked_source_unit_index=(
                    None
                    if entry.get("blocked_source_unit_index") is None
                    else int(entry["blocked_source_unit_index"])
                ),
                provenance=provenance,
            )
        )
    return AlignmentResult(
        text=str(payload.get("text", "")),
        words=words,
        audio_duration_s=float(payload.get("audio_duration_s", 0.0)),
        observer_tokens=tuple(observer_tokens),
        diagnostics=dict(payload.get("diagnostics", {})),
    )


def build_runtime_config():
    from cascade.runtime import CascadeRuntimeConfig

    return CascadeRuntimeConfig()


def build_qwen_backend() -> AlignmentBackend:
    from cascade.alignment.qwen_forced_backend import QwenAlignmentBackend
    from cascade.runtime import (
        asr_model_name,
        forced_aligner_model_name,
    )

    backend = QwenAlignmentBackend(
        asr_model_path=asr_model_name,
        forced_aligner_model_path=forced_aligner_model_name,
        runtime_config=build_runtime_config(),
    )
    backend.load()
    return backend


def build_gemma_backend(
    *,
    heads_path: str | None,
    top_k: int,
    probe_mode: str | None = None,
):
    from cascade.alignment.gemma_transformers_asr_backend import GemmaTransformersASRBackend
    from cascade.runtime import gemma_model_name

    runtime_config = build_runtime_config()
    if probe_mode is not None:
        runtime_config.gemma_audio_align_probe_mode = str(probe_mode)
    backend = GemmaTransformersASRBackend(
        model_name=gemma_model_name,
        runtime_config=runtime_config,
        audio_heads_path=heads_path,
        audio_heads_top_k=top_k,
        filter_width=int(getattr(runtime_config, "gemma_audio_alignment_filter_width", 7)),
        max_new_tokens=int(getattr(runtime_config, "gemma_audio_alignment_max_new_tokens", 256)),
    )
    backend.load()
    return backend


def build_gemma_vllm_backend(
    *,
    heads_path: str | None,
    top_k: int,
    executor_backend: str | None = None,
    patch_mode: str | None = None,
    enforce_eager: bool | None = None,
    enable_prefix_caching: bool | None = None,
    compilation_mode: str | None = None,
    cudagraph_mode: str | None = None,
    compile_cache_dir: str | None = None,
    disable_compile_cache: bool | None = None,
):
    from cascade.alignment.gemma_vllm_asr_backend import GemmaVLLMASRBackend
    from cascade.runtime import gemma_model_name

    runtime_config = build_runtime_config()
    if executor_backend is not None:
        runtime_config.gemma_vllm_executor_backend = str(executor_backend)
    if patch_mode is not None:
        runtime_config.gemma_vllm_patch_mode = str(patch_mode)
    if enforce_eager is not None:
        runtime_config.gemma_vllm_enforce_eager = bool(enforce_eager)
    if enable_prefix_caching is not None:
        runtime_config.gemma_vllm_enable_prefix_caching = bool(enable_prefix_caching)
    if compilation_mode is not None:
        runtime_config.gemma_vllm_compilation_mode = str(compilation_mode)
    if cudagraph_mode is not None:
        runtime_config.gemma_vllm_cudagraph_mode = str(cudagraph_mode)
    if compile_cache_dir is not None:
        runtime_config.gemma_vllm_compile_cache_dir = str(compile_cache_dir)
    if disable_compile_cache is not None:
        runtime_config.gemma_vllm_disable_compile_cache = bool(disable_compile_cache)
    backend = GemmaVLLMASRBackend(
        model_name=gemma_model_name,
        runtime_config=runtime_config,
        audio_heads_path=heads_path,
        audio_heads_top_k=top_k,
        filter_width=int(
            getattr(runtime_config, "gemma_audio_alignment_filter_width", 7)
        ),
        max_new_tokens=int(
            getattr(runtime_config, "gemma_audio_alignment_max_new_tokens", 256)
        ),
    )
    backend.load()
    return backend


def cmd_baseline(args: argparse.Namespace) -> None:
    audio, sr = load_wav(args.wav)
    backend = build_qwen_backend()
    result = backend.transcribe_and_align(audio, sample_rate=sr, language=args.language)
    if result is None:
        raise RuntimeError("Qwen transcription produced no valid result.")
    _write_bundle(args.output, result, tag="qwen_forced", wav_path=args.wav)


def cmd_gemma_inspect(args: argparse.Namespace) -> None:
    audio, sr = load_wav(args.wav)
    backend = build_gemma_backend(
        heads_path=args.heads_path or None,
        top_k=int(args.top_k),
    )
    result = backend.transcribe_and_align(audio, sample_rate=sr, language=args.language)
    if result is None:
        raise RuntimeError("Gemma alignment produced no valid result.")
    _write_bundle(args.output, result, tag="gemma_onepass_qk_fast", wav_path=args.wav)


def cmd_gemma_vllm_inspect(args: argparse.Namespace) -> None:
    audio, sr = load_wav(args.wav)
    backend = build_gemma_vllm_backend(
        heads_path=args.heads_path or None,
        top_k=int(args.top_k),
        executor_backend=args.vllm_executor_backend,
        patch_mode=args.vllm_patch_mode,
        enforce_eager=args.vllm_enforce_eager,
        enable_prefix_caching=args.vllm_enable_prefix_caching,
        compilation_mode=args.vllm_compilation_mode,
        cudagraph_mode=args.vllm_cudagraph_mode,
        compile_cache_dir=args.vllm_compile_cache_dir,
        disable_compile_cache=args.vllm_disable_compile_cache,
    )
    warmup_seconds = float(getattr(args, "warmup_seconds", 0.0) or 0.0)
    if warmup_seconds > 0:
        print(f"[warmup] running {warmup_seconds:.1f}s noise pass to pre-capture cudagraphs")
        backend.warmup(duration_seconds=warmup_seconds)
        print("[warmup] done")
    repeat = int(getattr(args, "repeat", 1) or 1)
    output_base = Path(args.output)
    results: list[AlignmentResult] = []
    for run_index in range(repeat):
        result = backend.transcribe_and_align(audio, sample_rate=sr, language=args.language)
        if result is None:
            raise RuntimeError(
                f"Gemma vLLM alignment produced no valid result (run {run_index})."
            )
        results.append(result)
        if repeat == 1:
            _write_bundle(args.output, result, tag="gemma_vllm_qk_fast", wav_path=args.wav)
        else:
            run_path = output_base.with_suffix(f".run{run_index}{output_base.suffix}")
            _write_bundle(
                str(run_path),
                result,
                tag=f"gemma_vllm_qk_fast_run{run_index}",
                wav_path=args.wav,
            )
    if repeat > 1:
        _write_repeat_stability_summary(output_base, results, wav_path=args.wav)


def cmd_seam_comparison(args: argparse.Namespace) -> None:
    """Run the three-seam comparison on one audio: eager, cudagraph=full, compile.

    This is the minimum system-level validation set from PLAN.md section 3:
    eager mono-audio baseline, worker_cls + cudagraph=full, worker_cls +
    vllm_compile + cudagraph=none.
    """
    audio, sr = load_wav(args.wav)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seams = [
        {
            "name": "eager_baseline",
            "enforce_eager": True,
            "compilation_mode": None,
            "cudagraph_mode": None,
            "worker_mode": "postload_instance",
        },
        {
            "name": "cudagraph_full",
            "enforce_eager": False,
            "compilation_mode": None,
            "cudagraph_mode": "full",
            "worker_mode": "custom_tensor",
        },
        {
            "name": "vllm_compile",
            "enforce_eager": False,
            "compilation_mode": "vllm_compile",
            "cudagraph_mode": "none",
            "worker_mode": "custom_tensor",
        },
    ]

    seam_results: list[dict] = []
    for seam in seams:
        print(f"\n--- Running seam: {seam['name']} ---")
        backend = build_gemma_vllm_backend(
            heads_path=args.heads_path or None,
            top_k=int(args.top_k),
            enforce_eager=seam["enforce_eager"],
            enable_prefix_caching=False,
            compilation_mode=seam["compilation_mode"],
            cudagraph_mode=seam["cudagraph_mode"],
        )
        # Override worker mode if not custom_tensor (eager baseline uses
        # the default postload path, not the custom worker).
        if seam["worker_mode"] != "custom_tensor":
            backend.worker_mode = seam["worker_mode"]

        result = backend.transcribe_and_align(
            audio, sample_rate=sr, language=args.language
        )
        if result is None:
            print(f"  WARNING: seam {seam['name']} produced no result")
            seam_results.append({"name": seam["name"], "result": None})
            continue

        bundle_path = out_dir / f"{seam['name']}.json"
        _write_bundle(str(bundle_path), result, tag=seam["name"], wav_path=args.wav)
        seam_results.append({
            "name": seam["name"],
            "text": result.text,
            "word_count": len(result.words),
            "generated_token_count": result.diagnostics.get("generated_token_count"),
            "monotonicity": result.diagnostics.get("monotonicity"),
            "total_backend_ms": (
                result.diagnostics.get("timings_ms", {}).get("total_backend")
            ),
            "observer_effective_heads": (
                result.diagnostics.get("capture", {}).get("effective_head_count")
            ),
        })

        # Clean up the engine between seams to avoid OOM.
        del backend

    # Write comparison summary.
    texts = [s["text"] for s in seam_results if s.get("text")]
    summary = {
        "tag": "seam_comparison",
        "wav_path": args.wav,
        "seam_count": len(seam_results),
        "text_agreement": len(set(texts)) <= 1 if texts else False,
        "seams": seam_results,
    }
    summary_path = out_dir / "seam_comparison.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote seam comparison: {summary_path}")
    if summary["text_agreement"]:
        print("  All seams produced identical text.")
    else:
        print(f"  Text divergence: {len(set(texts))} unique texts across {len(texts)} seams.")


def cmd_gemma_forced_align(args: argparse.Namespace) -> None:
    audio, sr = load_wav(args.wav)
    teacher = deserialize_alignment(
        json.loads(Path(args.teacher).read_text(encoding="utf-8"))
    )
    backend = build_gemma_backend(
        heads_path=args.heads_path or None,
        top_k=int(args.top_k),
        probe_mode=args.probe_mode,
    )
    result = backend.align_transcript(
        audio,
        sample_rate=sr,
        language=args.language,
        transcript=teacher.text,
    )
    if result is None:
        raise RuntimeError("Gemma forced alignment produced no valid result.")
    _write_bundle(args.output, result, tag="gemma_forced_align", wav_path=args.wav)


def cmd_gemma_calibrate_heads_forced(args: argparse.Namespace) -> None:
    audio, sr = load_wav(args.wav)
    teacher = deserialize_alignment(
        json.loads(Path(args.teacher).read_text(encoding="utf-8"))
    )
    backend = build_gemma_backend(heads_path=None, top_k=int(args.top_k))
    scored = backend.calibrate_alignment_heads_forced(
        audio,
        sample_rate=sr,
        language=args.language,
        teacher=teacher,
        top_k=None,
    )
    top_heads = scored[: int(args.top_k)]

    # Fit a single systematic word-end offset on the selected heads: a
    # causal LLM's attention peak lags the acoustic boundary. Subtract the
    # median signed error on this one teacher so the heads file ships with
    # its own calibration and downstream consumers don't need to re-fit.
    offset_s = _estimate_word_end_offset(
        backend=backend,
        audio=audio,
        sample_rate=sr,
        language=args.language,
        teacher=teacher,
        top_heads=top_heads,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from cascade.alignment.gemma_transformers_asr_backend import save_audio_alignment_heads

    save_audio_alignment_heads(
        str(out_path),
        scored_heads=top_heads,
        model_name=backend.model_name,
        language=args.language,
        word_end_offset_seconds=offset_s,
        scoring_notes={
            "teacher": str(Path(args.teacher)),
            "teacher_audio_duration_s": teacher.audio_duration_s,
            "num_heads_scored": len(scored),
            "filter_width": backend.filter_width,
            "mode": "forced_alignment",
            "notes": (
                "Teacher-forced per-token attention; MAE is on aligned words (same "
                "sequence), so it is a true alignment-quality signal rather than a "
                "transcription-quality artifact. The word_end_offset_seconds value "
                "is the median signed lag between the selected-heads prediction and "
                "the teacher on this clip."
            ),
        },
    )
    ranking_path = out_path.with_suffix(".full_ranking.json")
    ranking_path.write_text(json.dumps(scored, indent=2), encoding="utf-8")
    print(f"Wrote heads file: {out_path}  (offset={offset_s:.3f}s)")
    print(f"Wrote full scoring ranking: {ranking_path}")


def _estimate_word_end_offset(
    *,
    backend,
    audio,
    sample_rate: int,
    language: str,
    teacher: AlignmentResult,
    top_heads: list[dict],
) -> float:
    import statistics

    from cascade.alignment.gemma_transformers_asr_backend import AlignAttHead

    # Temporarily install the selected heads on the backend and run one
    # forced alignment with offset=0, then fit the offset to the teacher.
    prev_heads = list(backend.alignatt_heads)
    prev_offset = backend.word_end_offset_s
    prev_recorder = backend.alignatt_recorder
    prev_layer_input_recorder = getattr(backend, "alignatt_layer_input_recorder", None)
    try:
        backend.alignatt_heads = [
            AlignAttHead(layer=int(h["layer"]), head=int(h["head"]), ts=float(h["ts"]))
            for h in top_heads
        ]
        backend.word_end_offset_s = 0.0
        backend.alignatt_recorder = None  # let align_transcript rebuild
        backend.alignatt_layer_input_recorder = None  # idem for qk_fast replay
        result = backend.align_transcript(
            audio,
            sample_rate=sample_rate,
            language=language,
            transcript=teacher.text,
        )
        if result is None or not result.words:
            return 0.0
        n = min(len(result.words), len(teacher.words))
        diffs = [
            result.words[i].end_time - teacher.words[i].end_time for i in range(n)
        ]
        if not diffs:
            return 0.0
        return float(statistics.median(diffs))
    finally:
        backend.alignatt_heads = prev_heads
        backend.word_end_offset_s = prev_offset
        backend.alignatt_recorder = prev_recorder
        backend.alignatt_layer_input_recorder = prev_layer_input_recorder


def cmd_gemma_calibrate_heads(args: argparse.Namespace) -> None:
    audio, sr = load_wav(args.wav)
    teacher_path = Path(args.teacher)
    teacher = deserialize_alignment(json.loads(teacher_path.read_text(encoding="utf-8")))

    backend = build_gemma_backend(heads_path=None, top_k=int(args.top_k))
    scored = backend.calibrate_alignment_heads(
        audio,
        sample_rate=sr,
        language=args.language,
        teacher=teacher,
        top_k=None,  # keep the full ranking in the diagnostic output
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from cascade.alignment.gemma_transformers_asr_backend import save_audio_alignment_heads

    save_audio_alignment_heads(
        str(out_path),
        scored_heads=scored[: int(args.top_k)],
        model_name=backend.model_name,
        language=args.language,
        scoring_notes={
            "teacher": str(teacher_path),
            "teacher_audio_duration_s": teacher.audio_duration_s,
            "num_heads_scored": len(scored),
            "filter_width": backend.filter_width,
            "max_new_tokens": backend.max_new_tokens,
            "notes": (
                "Monotonicity-dominant ranking with Qwen-teacher MAE as a soft "
                "quality term. See cascade.alignment.gemma_transformers_asr_backend._combine_head_score."
            ),
        },
    )
    # Also dump the full ranking (including low-score heads) for later
    # ablation / analysis — this is cheap and avoids rerunning the scoring.
    ranking_path = out_path.with_suffix(".full_ranking.json")
    ranking_path.write_text(json.dumps(scored, indent=2), encoding="utf-8")
    print(f"Wrote heads file: {out_path}")
    print(f"Wrote full scoring ranking: {ranking_path}")


def cmd_compare(args: argparse.Namespace) -> None:
    qwen = deserialize_alignment(
        json.loads(Path(args.qwen).read_text(encoding="utf-8"))
    )
    gemma = deserialize_alignment(
        json.loads(Path(args.gemma).read_text(encoding="utf-8"))
    )
    report = compare_alignments(qwen, gemma)
    print(json.dumps(report, indent=2))


def compare_alignments(qwen: AlignmentResult, gemma: AlignmentResult) -> dict:
    n = min(len(qwen.words), len(gemma.words))
    word_errors = [
        abs(qwen.words[i].end_time - gemma.words[i].end_time) for i in range(n)
    ]
    report = {
        "qwen_text": qwen.text,
        "gemma_text": gemma.text,
        "qwen_word_count": len(qwen.words),
        "gemma_word_count": len(gemma.words),
        "paired_words": n,
    }
    if word_errors:
        sorted_errors = sorted(word_errors)
        report.update(
            {
                "word_end_mae_seconds": float(sum(word_errors) / len(word_errors)),
                "word_end_median_error_seconds": float(
                    sorted_errors[len(sorted_errors) // 2]
                ),
                "word_end_p90_error_seconds": float(
                    sorted_errors[min(len(sorted_errors) - 1, int(len(sorted_errors) * 0.9))]
                ),
                "per_word_end_errors_seconds": word_errors,
            }
        )
    return report


def _write_repeat_stability_summary(
    output_base: Path,
    results: list[AlignmentResult],
    *,
    wav_path: str,
) -> None:
    """Write a summary comparing repeated runs on the same engine.

    Reports text stability, token-count drift, timing variance, and
    per-run decode_drift diagnostics. This is the reproducible one-clip
    comparison the PLAN calls for.
    """
    texts = [r.text for r in results]
    token_counts = [
        r.diagnostics.get("generated_token_count", 0) for r in results
    ]
    timings = [
        r.diagnostics.get("timings_ms", {}).get("total_backend", 0.0)
        for r in results
    ]
    decode_drifts = [r.diagnostics.get("decode_drift") for r in results]
    summary = {
        "tag": "repeat_stability",
        "wav_path": wav_path,
        "run_count": len(results),
        "text_stable": len(set(texts)) == 1,
        "unique_texts": list(set(texts)),
        "token_counts": token_counts,
        "backend_ms_per_run": [round(t, 1) for t in timings],
        "decode_drifts": decode_drifts,
        "per_run_diagnostics": [
            {
                k: v
                for k, v in r.diagnostics.items()
                if k
                in (
                    "generated_token_count",
                    "monotonicity",
                    "finish_reason",
                    "decode_drift",
                    "timings_ms",
                )
            }
            for r in results
        ],
    }
    summary_path = output_base.with_suffix(".stability.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote repeat stability summary: {summary_path}")
    if summary["text_stable"]:
        print(f"  text_stable=True across {len(results)} runs")
    else:
        print(f"  text_stable=False — {len(set(texts))} unique texts across {len(results)} runs")


def _write_bundle(
    output_path: str | Path,
    result: AlignmentResult,
    *,
    tag: str,
    wav_path: str,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tag": tag,
        "wav_path": wav_path,
        **serialize_alignment(result),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {tag} bundle: {path}")


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(required=True, dest="cmd")

    baseline = subparsers.add_parser("baseline", help="Run Qwen baseline")
    baseline.add_argument("--wav", required=True)
    baseline.add_argument("--language", default="English")
    baseline.add_argument("--output", required=True)
    baseline.set_defaults(func=cmd_baseline)

    inspect = subparsers.add_parser("gemma_inspect", help="Run Gemma aligner with heads")
    inspect.add_argument("--wav", required=True)
    inspect.add_argument("--language", default="English")
    inspect.add_argument("--heads-path", default=None)
    inspect.add_argument("--top-k", type=int, default=8)
    inspect.add_argument("--output", required=True)
    inspect.set_defaults(func=cmd_gemma_inspect)

    vllm_inspect = subparsers.add_parser(
        "gemma_vllm_inspect",
        help="Experimental: run Gemma ASR + AlignAtt through vLLM on one clip",
    )
    vllm_inspect.add_argument("--wav", required=True)
    vllm_inspect.add_argument("--language", default="English")
    vllm_inspect.add_argument("--heads-path", default=None)
    vllm_inspect.add_argument("--top-k", type=int, default=8)
    vllm_inspect.add_argument(
        "--vllm-executor-backend",
        choices=("mp", "uni"),
        default=None,
        help="Experimental vLLM executor backend override.",
    )
    vllm_inspect.add_argument(
        "--vllm-patch-mode",
        choices=("postload_instance", "preload_class"),
        default=None,
        help="Experimental observer installation strategy.",
    )
    vllm_inspect.add_argument(
        "--vllm-enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override vLLM eager mode for the experimental backend.",
    )
    vllm_inspect.add_argument(
        "--vllm-enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override prefix caching for the experimental backend. The current "
            "observer path defaults to disabled because prompt-side K capture "
            "still depends on real prompt forwards."
        ),
    )
    vllm_inspect.add_argument(
        "--vllm-compilation-mode",
        choices=("none", "stock_torch_compile", "dynamo_trace_once", "vllm_compile"),
        default=None,
        help="Override vLLM compilation mode for diagnostic runs.",
    )
    vllm_inspect.add_argument(
        "--vllm-cudagraph-mode",
        choices=("none", "piecewise", "full", "full_decode_only", "full_and_piecewise"),
        default=None,
        help="Override vLLM cudagraph mode for diagnostic runs.",
    )
    vllm_inspect.add_argument(
        "--vllm-compile-cache-dir",
        default=None,
        help="Explicit torch.compile cache directory for this diagnostic run.",
    )
    vllm_inspect.add_argument(
        "--vllm-disable-compile-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Disable torch.compile cache reuse for this diagnostic run.",
    )
    vllm_inspect.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "Run the same request N times on the same engine. Writes per-run "
            "bundles and a stability summary for decode-drift investigation."
        ),
    )
    vllm_inspect.add_argument(
        "--warmup-seconds",
        type=float,
        default=0.0,
        help=(
            "If > 0, run one throwaway transcribe_and_align on synthetic noise of "
            "this duration before the first --repeat run. Used to test whether the "
            "cold-run text degradation under cudagraph is a first-capture artifact."
        ),
    )
    vllm_inspect.add_argument("--output", required=True)
    vllm_inspect.set_defaults(func=cmd_gemma_vllm_inspect)

    cal = subparsers.add_parser(
        "gemma_calibrate_heads",
        help="Score every (layer, head) against a Qwen teacher bundle and save top-K",
    )
    cal.add_argument("--wav", required=True)
    cal.add_argument("--language", default="English")
    cal.add_argument("--teacher", required=True)
    cal.add_argument("--output", required=True)
    cal.add_argument("--top-k", type=int, default=8)
    cal.set_defaults(func=cmd_gemma_calibrate_heads)

    cal_forced = subparsers.add_parser(
        "gemma_calibrate_heads_forced",
        help="Forced-alignment head ranking: teacher transcript is teacher-forced into Gemma, MAE is on identical word sequences.",
    )
    cal_forced.add_argument("--wav", required=True)
    cal_forced.add_argument("--language", default="English")
    cal_forced.add_argument("--teacher", required=True)
    cal_forced.add_argument("--output", required=True)
    cal_forced.add_argument("--top-k", type=int, default=8)
    cal_forced.set_defaults(func=cmd_gemma_calibrate_heads_forced)

    forced = subparsers.add_parser(
        "gemma_forced_align",
        help="Forced alignment: take the teacher's transcript and produce Gemma word timestamps via attention.",
    )
    forced.add_argument("--wav", required=True)
    forced.add_argument("--language", default="English")
    forced.add_argument("--teacher", required=True)
    forced.add_argument("--heads-path", default=None)
    forced.add_argument("--top-k", type=int, default=8)
    forced.add_argument(
        "--probe-mode",
        choices=("eager", "qk_fast"),
        default="eager",
        help="Forced-alignment attention extraction backend.",
    )
    forced.add_argument("--output", required=True)
    forced.set_defaults(func=cmd_gemma_forced_align)

    seam_cmp = subparsers.add_parser(
        "seam_comparison",
        help=(
            "Run the three-seam vLLM comparison on one audio: eager baseline, "
            "cudagraph=full, vllm_compile+cudagraph=none."
        ),
    )
    seam_cmp.add_argument("--wav", required=True)
    seam_cmp.add_argument("--language", default="English")
    seam_cmp.add_argument("--heads-path", default=None)
    seam_cmp.add_argument("--top-k", type=int, default=8)
    seam_cmp.add_argument("--output-dir", required=True)
    seam_cmp.set_defaults(func=cmd_seam_comparison)

    compare = subparsers.add_parser("compare", help="Compare two alignment bundles")
    compare.add_argument("--qwen", required=True)
    compare.add_argument("--gemma", required=True)
    compare.set_defaults(func=cmd_compare)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_cli()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
