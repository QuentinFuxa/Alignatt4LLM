"""Validate the two-pass full-Gemma frontend and produce saved artifacts.

Phases (matching PLAN.md):
  Phase 1-2: Backend-level validation on one clip, save JSON artifact
  Phase 3:   Cascade-level comparison (two-pass vs hybrid), save artifact
  Phase 4:   Probe a harder clip

Usage:
  python run_gemma_two_pass_validation.py                  # Phase 1-2: smoke18 backend validation
  python run_gemma_two_pass_validation.py --compare        # Phase 1-2: + backend comparison vs hybrid
  python run_gemma_two_pass_validation.py --cascade        # Phase 3: cascade-level comparison
  python run_gemma_two_pass_validation.py --harder         # Phase 4: probe rxrToXvRyM_first18
  python run_gemma_two_pass_validation.py --all            # All phases
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf


WAV_SMOKE18 = "tmp/alignatt_smoke18.wav"
WAV_HARDER = "tmp/rxrToXvRyM_first18.wav"
HEADS_PATH = "assets/attention_heads/audio_alignment_heads_google_gemma-4-E4B-it_en_forced.json"
OUTPUT_DIR = "tmp/two_pass_validation"


def _resolve_hf_snapshot(suffix: str) -> str:
    import os
    candidates = [
        os.path.join(os.path.expanduser("~"), ".cache/huggingface/hub", suffix),
        os.path.join("/root/.cache/huggingface/hub", suffix),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


GEMMA_SNAPSHOT = "models--google--gemma-4-E4B-it/snapshots/83df0a889143b1dbfc61b591bbc639540fd9ce4c"
QWEN_ASR_SNAPSHOT = "models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5"
QWEN_ALIGNER_SNAPSHOT = "models--Qwen--Qwen3-ForcedAligner-0.6B/snapshots/c7cbfc2048c462b0d63a45797104fc9db3ad62b7"


def _result_to_dict(result, *, backend_name: str, runtime_s: float, wav_path: str, audio_duration_s: float) -> dict:
    return {
        "backend": backend_name,
        "wav_path": wav_path,
        "audio_duration_s": audio_duration_s,
        "runtime_s": round(runtime_s, 3),
        "transcript": result.text,
        "word_count": len(result.words),
        "words": [
            {"text": w.text, "start_time": round(w.start_time, 4), "end_time": round(w.end_time, 4)}
            for w in result.words
        ],
        "diagnostics": result.diagnostics,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _save_artifact(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"Artifact saved: {path}")


def build_gemma_two_pass():
    from gemma_alignment_probe import GemmaAttentionAlignmentBackend
    from gemma_two_pass_frontend import GemmaTwoPassAlignmentBackend

    gemma_model = _resolve_hf_snapshot(GEMMA_SNAPSHOT)
    runtime_config = SimpleNamespace(
        gemma_transformers_device="cuda:0",
        gemma_transformers_dtype="bfloat16",
    )
    gemma_backend = GemmaAttentionAlignmentBackend(
        model_name=gemma_model,
        runtime_config=runtime_config,
        audio_heads_path=HEADS_PATH if Path(HEADS_PATH).exists() else None,
        audio_heads_top_k=8,
        filter_width=7,
        max_new_tokens=256,
    )
    return GemmaTwoPassAlignmentBackend(gemma_backend=gemma_backend)


def build_hybrid():
    from qwen_alignment_backend import QwenAlignmentBackend
    from gemma_alignment_probe import GemmaAttentionAlignmentBackend
    from hybrid_alignment_backend import HybridQwenAsrGemmaAlignerBackend

    gemma_model = _resolve_hf_snapshot(GEMMA_SNAPSHOT)
    asr_model = _resolve_hf_snapshot(QWEN_ASR_SNAPSHOT)
    aligner_model = _resolve_hf_snapshot(QWEN_ALIGNER_SNAPSHOT)
    runtime_config = SimpleNamespace(
        gemma_transformers_device="cuda:0",
        gemma_transformers_dtype="bfloat16",
        asr_gpu_memory_utilization=0.2,
    )
    asr_backend = QwenAlignmentBackend(
        asr_model_path=asr_model,
        forced_aligner_model_path=aligner_model,
        runtime_config=runtime_config,
    )
    gemma_backend = GemmaAttentionAlignmentBackend(
        model_name=gemma_model,
        runtime_config=runtime_config,
        audio_heads_path=HEADS_PATH if Path(HEADS_PATH).exists() else None,
        audio_heads_top_k=8,
        filter_width=7,
        max_new_tokens=256,
    )
    return HybridQwenAsrGemmaAlignerBackend(
        asr_backend=asr_backend,
        gemma_backend=gemma_backend,
        strict=False,
    )


def run_backend(backend, wav_path: str) -> tuple[object, float]:
    audio, sr = sf.read(wav_path, dtype="float32")
    t0 = time.perf_counter()
    result = backend.transcribe_and_align(audio, sample_rate=sr, language="English")
    elapsed = time.perf_counter() - t0
    return result, elapsed


def run_two_pass_validation(wav_path: str, *, tag: str, out_dir: Path) -> dict | None:
    audio, sr = sf.read(wav_path, dtype="float32")
    audio_duration_s = len(audio) / sr
    print(f"\nAudio: {wav_path} ({audio_duration_s:.1f}s, {sr}Hz)")

    backend = build_gemma_two_pass()
    print("Loading Gemma model...")
    t0 = time.perf_counter()
    backend.load()
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.1f}s")

    print(f"\n--- Two-pass full-Gemma frontend ({tag}) ---")
    result, elapsed = run_backend(backend, wav_path)
    print(f"Two-pass completed in {elapsed:.1f}s")

    if result is None:
        print("ERROR: backend returned None")
        return None

    print(f"Transcript: {result.text}")
    print(f"Words: {len(result.words)}")
    if result.words:
        for w in result.words:
            print(f"  [{w.start_time:.3f} - {w.end_time:.3f}] {w.text}")

    artifact = _result_to_dict(
        result,
        backend_name="gemma_two_pass",
        runtime_s=elapsed,
        wav_path=wav_path,
        audio_duration_s=audio_duration_s,
    )
    artifact["model_load_time_s"] = round(load_time, 3)
    _save_artifact(artifact, out_dir / f"two_pass_{tag}.json")
    return artifact


def run_backend_comparison(wav_path: str, *, tag: str, out_dir: Path, two_pass_artifact: dict) -> dict | None:
    audio, sr = sf.read(wav_path, dtype="float32")
    audio_duration_s = len(audio) / sr

    print(f"\nLoading hybrid backend (Qwen ASR + Gemma aligner)...")
    hybrid = build_hybrid()
    t0 = time.perf_counter()
    hybrid.load()
    print(f"Hybrid loaded in {time.perf_counter() - t0:.1f}s")

    result, elapsed = run_backend(hybrid, wav_path)
    print(f"Hybrid completed in {elapsed:.1f}s")

    if result is None:
        print("ERROR: hybrid returned None")
        return None

    hybrid_artifact = _result_to_dict(
        result,
        backend_name="hybrid_qwen_asr_gemma_aligner",
        runtime_s=elapsed,
        wav_path=wav_path,
        audio_duration_s=audio_duration_s,
    )
    _save_artifact(hybrid_artifact, out_dir / f"hybrid_{tag}.json")

    tp_words = two_pass_artifact["words"]
    hy_words = hybrid_artifact["words"]
    n = min(len(tp_words), len(hy_words))
    timing_diffs = [
        abs(tp_words[i]["end_time"] - hy_words[i]["end_time"])
        for i in range(n)
    ] if n > 0 else []

    comparison = {
        "tag": tag,
        "wav_path": wav_path,
        "audio_duration_s": audio_duration_s,
        "two_pass": {
            "transcript": two_pass_artifact["transcript"],
            "word_count": two_pass_artifact["word_count"],
            "runtime_s": two_pass_artifact["runtime_s"],
            "diagnostics": two_pass_artifact["diagnostics"],
        },
        "hybrid": {
            "transcript": hybrid_artifact["transcript"],
            "word_count": hybrid_artifact["word_count"],
            "runtime_s": hybrid_artifact["runtime_s"],
            "diagnostics": hybrid_artifact["diagnostics"],
        },
        "timing_comparison": {
            "paired_words": n,
            "mean_end_time_diff_s": round(sum(timing_diffs) / len(timing_diffs), 4) if timing_diffs else None,
            "max_end_time_diff_s": round(max(timing_diffs), 4) if timing_diffs else None,
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    _save_artifact(comparison, out_dir / f"comparison_backend_{tag}.json")

    print(f"\n--- Backend Comparison ({tag}) ---")
    print(f"{'':20s} {'Two-Pass':>15s}  {'Hybrid':>15s}")
    print(f"{'Word count':20s} {two_pass_artifact['word_count']:>15d}  {hybrid_artifact['word_count']:>15d}")
    print(f"{'Runtime (s)':20s} {two_pass_artifact['runtime_s']:>15.1f}  {hybrid_artifact['runtime_s']:>15.1f}")
    if timing_diffs:
        print(f"{'Mean timing diff':20s} {comparison['timing_comparison']['mean_end_time_diff_s']:>15.3f}s")
        print(f"{'Max timing diff':20s} {comparison['timing_comparison']['max_end_time_diff_s']:>15.3f}s")
    print(f"\nTwo-pass: {two_pass_artifact['transcript']}")
    print(f"Hybrid:   {hybrid_artifact['transcript']}")

    return comparison


def run_cascade_comparison(wav_path: str, *, tag: str, out_dir: Path, chunk_ms: int = 960) -> dict | None:
    print(f"\n{'='*60}")
    print(f"PHASE 3: Cascade-Level Comparison ({tag})")
    print(f"{'='*60}")

    python = sys.executable
    tp_dir = str(out_dir / f"cascade_two_pass_{tag}")
    hy_dir = str(out_dir / f"cascade_hybrid_{tag}")

    print("\n--- Running two-pass Gemma cascade ---")
    _run_cascade_subprocess(python, wav_path, "gemma_two_pass", tp_dir, chunk_ms)

    print("\n--- Running hybrid cascade ---")
    _run_cascade_subprocess(python, wav_path, "hybrid_qwen_asr_gemma_aligner", hy_dir, chunk_ms)

    tp_manifest = _load_manifest(tp_dir)
    hy_manifest = _load_manifest(hy_dir)

    comparison = {
        "tag": tag,
        "wav_path": wav_path,
        "chunk_ms": chunk_ms,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    for label, manifest, odir in [("two_pass", tp_manifest, tp_dir), ("hybrid", hy_manifest, hy_dir)]:
        if manifest is None:
            comparison[label] = {"error": f"cascade run failed or produced no artifacts in {odir}"}
            continue
        odir_path = Path(odir)
        transcript_file = odir_path / "transcript.en.txt"
        translation_file = odir_path / "translation.de.txt"
        updates_file = odir_path / "stream_updates.jsonl"
        num_updates = 0
        if updates_file.exists():
            num_updates = sum(1 for _ in updates_file.open())
        comparison[label] = {
            "final_asr": transcript_file.read_text().strip() if transcript_file.exists() else "",
            "final_translation": translation_file.read_text().strip() if translation_file.exists() else "",
            "num_updates": num_updates,
            "artifacts_dir": odir,
        }

    _save_artifact(comparison, out_dir / f"comparison_cascade_{tag}.json")

    print(f"\n--- Cascade Comparison ({tag}) ---")
    for label in ["two_pass", "hybrid"]:
        c = comparison.get(label, {})
        if "error" in c:
            print(f"  {label}: {c['error']}")
        else:
            print(f"  {label}: {c.get('num_updates', 'N/A')} updates")
            print(f"    ASR:  {c.get('final_asr', '')[:100]}")
            print(f"    MT:   {c.get('final_translation', '')[:100]}")

    return comparison


def _run_cascade_subprocess(python: str, wav_path: str, backend: str, output_dir: str, chunk_ms: int) -> None:
    script = (
        "import sys; sys.path.insert(0, '.');"
        "from qwen3asr_gemma_cascade_core import config, run_stream;"
        f"config.alignment_backend_name = {backend!r};"
        f"run_stream({wav_path!r}, chunk_ms={chunk_ms}, output_dir={output_dir!r})"
    )
    result = subprocess.run(
        [python, "-c", script],
        capture_output=False,
        timeout=1800,
    )
    if result.returncode != 0:
        print(f"  WARNING: {backend} cascade exited with code {result.returncode}")


def _load_manifest(output_dir: str) -> dict | None:
    manifest_path = Path(output_dir) / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def main() -> None:
    args = set(sys.argv[1:])
    run_all = "--all" in args
    do_compare = "--compare" in args or run_all
    do_cascade = "--cascade" in args or run_all
    do_harder = "--harder" in args or run_all
    cascade_only = "--cascade" in args and not run_all and not do_compare and not do_harder

    out_dir = Path(OUTPUT_DIR)

    if not cascade_only:
        # Phase 1-2: Two-pass validation on smoke18
        print(f"\n{'='*60}")
        print("PHASE 1-2: Two-pass validation on smoke18")
        print(f"{'='*60}")
        tp_artifact = run_two_pass_validation(WAV_SMOKE18, tag="smoke18", out_dir=out_dir)

        # Phase 1-2 (optional): Backend comparison vs hybrid
        if do_compare and tp_artifact:
            run_backend_comparison(WAV_SMOKE18, tag="smoke18", out_dir=out_dir, two_pass_artifact=tp_artifact)

    # Phase 3: Cascade-level comparison (runs in subprocesses — needs full GPU)
    if do_cascade:
        run_cascade_comparison(WAV_SMOKE18, tag="smoke18", out_dir=out_dir)

    if not cascade_only:
        # Phase 4: Harder clip
        if do_harder:
            print(f"\n{'='*60}")
            print("PHASE 4: Harder clip (rxrToXvRyM_first18)")
            print(f"{'='*60}")
            harder_artifact = run_two_pass_validation(WAV_HARDER, tag="rxrToXvRyM_first18", out_dir=out_dir)
            if do_compare and harder_artifact:
                run_backend_comparison(WAV_HARDER, tag="rxrToXvRyM_first18", out_dir=out_dir, two_pass_artifact=harder_artifact)

    print(f"\n{'='*60}")
    print(f"All artifacts in: {out_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
