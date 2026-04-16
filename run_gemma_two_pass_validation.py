"""Validate the two-pass full-Gemma frontend on one audio clip.

Runs the gemma_two_pass backend on tmp/alignatt_smoke18.wav, prints:
1. Transcript from pass 1 (default attention ASR)
2. Word-level timings from pass 2 (eager attention forced alignment)
3. Diagnostics

Optionally compares against the hybrid backend if --compare is passed.
"""

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf


WAV_PATH = "tmp/alignatt_smoke18.wav"
HEADS_PATH = "assets/attention_heads/audio_alignment_heads_google_gemma-4-E4B-it_en_forced.json"


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


def build_gemma_two_pass():
    from gemma_alignment_probe import GemmaAttentionAlignmentBackend
    from gemma_two_pass_frontend import GemmaTwoPassAlignmentBackend

    gemma_model = _resolve_hf_snapshot(
        "models--google--gemma-4-E4B-it/snapshots/83df0a889143b1dbfc61b591bbc639540fd9ce4c"
    )
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


def run_two_pass(wav_path: str):
    audio, sr = sf.read(wav_path, dtype="float32")
    print(f"Audio: {wav_path} ({len(audio)/sr:.1f}s, {sr}Hz)")

    backend = build_gemma_two_pass()
    print("Loading Gemma model...")
    t0 = time.perf_counter()
    backend.load()
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    print("\n--- Two-pass full-Gemma frontend ---")
    t0 = time.perf_counter()
    result = backend.transcribe_and_align(audio, sample_rate=sr, language="English")
    elapsed = time.perf_counter() - t0
    print(f"Two-pass completed in {elapsed:.1f}s")

    if result is None:
        print("ERROR: backend returned None")
        return None

    print(f"\nTranscript: {result.text}")
    print(f"Words: {len(result.words)}")
    print(f"Audio duration: {result.audio_duration_s:.2f}s")
    print(f"\nDiagnostics: {json.dumps(result.diagnostics, indent=2, default=str)}")

    if result.words:
        print("\nWord timings:")
        for w in result.words:
            print(f"  [{w.start_time:.3f} - {w.end_time:.3f}] {w.text}")

    return result


def run_comparison(wav_path: str, two_pass_result):
    """Compare against hybrid backend."""
    from qwen_alignment_backend import QwenAlignmentBackend
    from gemma_alignment_probe import GemmaAttentionAlignmentBackend
    from hybrid_alignment_backend import HybridQwenAsrGemmaAlignerBackend

    audio, sr = sf.read(wav_path, dtype="float32")

    gemma_model = _resolve_hf_snapshot(
        "models--google--gemma-4-E4B-it/snapshots/83df0a889143b1dbfc61b591bbc639540fd9ce4c"
    )
    asr_model = _resolve_hf_snapshot(
        "models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5"
    )
    aligner_model = _resolve_hf_snapshot(
        "models--Qwen--Qwen3-ForcedAligner-0.6B/snapshots/c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
    )
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
    hybrid = HybridQwenAsrGemmaAlignerBackend(
        asr_backend=asr_backend,
        gemma_backend=gemma_backend,
        strict=False,
    )

    print("\nLoading hybrid backend (Qwen ASR + Gemma aligner)...")
    t0 = time.perf_counter()
    hybrid.load()
    print(f"Hybrid loaded in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    hybrid_result = hybrid.transcribe_and_align(audio, sample_rate=sr, language="English")
    elapsed = time.perf_counter() - t0
    print(f"Hybrid completed in {elapsed:.1f}s")

    if hybrid_result is None:
        print("ERROR: hybrid returned None")
        return

    print(f"\n--- Comparison ---")
    print(f"{'':20s} {'Two-Pass Gemma':>20s}  {'Hybrid (Qwen+Gemma)':>20s}")
    print(f"{'Transcript length':20s} {len(two_pass_result.text):>20d}  {len(hybrid_result.text):>20d}")
    print(f"{'Word count':20s} {len(two_pass_result.words):>20d}  {len(hybrid_result.words):>20d}")

    if two_pass_result.words and hybrid_result.words:
        n = min(len(two_pass_result.words), len(hybrid_result.words))
        timing_diffs = [
            abs(two_pass_result.words[i].end_time - hybrid_result.words[i].end_time)
            for i in range(n)
        ]
        print(f"{'Mean timing diff (s)':20s} {sum(timing_diffs)/len(timing_diffs):>20.3f}")
        print(f"{'Max timing diff (s)':20s} {max(timing_diffs):>20.3f}")

    print(f"\nTwo-pass transcript: {two_pass_result.text}")
    print(f"Hybrid transcript:   {hybrid_result.text}")


if __name__ == "__main__":
    result = run_two_pass(WAV_PATH)
    if result and "--compare" in sys.argv:
        run_comparison(WAV_PATH, result)
