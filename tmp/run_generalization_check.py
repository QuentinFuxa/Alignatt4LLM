"""Generalization check: calibrated heads (from smoke18) vs a DIFFERENT 18 s
slice cut from ``tmp/ccpXHNfaoy_first75.wav``.

Loads Qwen once + Gemma once in the same Python process, runs the
pipeline end-to-end, and prints the comparison. Useful for checking
whether Layer 23 + 480 ms offset generalize across speakers / cuts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_alignment_single_audio import (
    build_qwen_backend,
    build_gemma_backend,
    compare_alignments,
    load_wav,
    serialize_alignment,
)


def main() -> None:
    source_wav = "tmp/ccpXHNfaoy_first75.wav"
    heads_path = "assets/attention_heads/audio_alignment_heads_google_gemma-4-E4B-it_en_forced.json"
    slice_seconds = 18.0
    slice_start_s = 30.0

    audio, sr = load_wav(source_wav)
    start = int(slice_start_s * sr)
    end = start + int(slice_seconds * sr)
    if end > len(audio):
        raise RuntimeError(
            f"Source clip too short for {slice_seconds}s slice starting at {slice_start_s}s"
        )
    audio_slice = np.asarray(audio[start:end], dtype=np.float32)
    print(
        f"[info] sliced {slice_seconds}s from {source_wav} starting at {slice_start_s}s"
    )

    print("[info] loading Qwen for baseline")
    qwen_backend = build_qwen_backend()
    qwen_result = qwen_backend.transcribe_and_align(
        audio_slice, sample_rate=sr, language="English"
    )
    if qwen_result is None:
        raise RuntimeError("Qwen baseline failed")
    print(f"[qwen] text: {qwen_result.text}")
    print(f"[qwen] words: {len(qwen_result.words)}")

    print(f"[info] loading Gemma for forced alignment (heads: {heads_path})")
    gemma_backend = build_gemma_backend(heads_path=heads_path, top_k=8)
    gemma_result = gemma_backend.align_transcript(
        audio_slice,
        sample_rate=sr,
        language="English",
        transcript=qwen_result.text,
    )
    if gemma_result is None:
        raise RuntimeError("Gemma forced alignment failed")
    print(f"[gemma] monotonicity: {gemma_result.diagnostics.get('monotonicity')}")
    print(f"[gemma] word count: {len(gemma_result.words)}")
    print(f"[gemma] offset_s: {gemma_result.diagnostics.get('word_end_offset_s')}")

    report = compare_alignments(qwen_result, gemma_result)
    print("\n--- generalization report ---")
    for key in (
        "qwen_word_count",
        "gemma_word_count",
        "paired_words",
        "word_end_mae_seconds",
        "word_end_median_error_seconds",
        "word_end_p90_error_seconds",
    ):
        if key in report:
            print(f"  {key}: {report[key]}")

    out_dir = Path("tmp/alignment_research")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ccpXHNfaoy_{int(slice_start_s):02d}s_{int(slice_start_s + slice_seconds):02d}s"
    (out_dir / f"{stem}_qwen_teacher.json").write_text(
        json.dumps(
            {
                "tag": "qwen_baseline",
                "source_wav": source_wav,
                "slice_start_s": slice_start_s,
                "slice_seconds": slice_seconds,
                **serialize_alignment(qwen_result),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / f"{stem}_gemma_forced.json").write_text(
        json.dumps(
            {
                "tag": "gemma_forced_align",
                "source_wav": source_wav,
                "slice_start_s": slice_start_s,
                "slice_seconds": slice_seconds,
                **serialize_alignment(gemma_result),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / f"{stem}_generalization_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    print(f"\nArtifacts written to tmp/alignment_research/{stem}_*.json")


if __name__ == "__main__":
    main()
