"""Generate Qwen teacher timestamps for all clips in a split manifest.

Loads Qwen3-ASR once and processes all clips that don't already have
teacher artifacts. Saves one JSON per clip.

Usage:
    .venv-inference/bin/python run_generate_split_teachers.py [--manifest PATH]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_MANIFEST = Path("tmp/feature_aligner/split_manifest_full.json")
TEACHER_DIR = Path("tmp/feature_aligner/teachers")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    TEACHER_DIR.mkdir(parents=True, exist_ok=True)

    with open(args.manifest) as f:
        manifest = json.load(f)

    all_clips = []
    for split_name, clips in manifest["splits"].items():
        for clip in clips:
            clip["split"] = split_name
            all_clips.append(clip)

    missing = []
    for clip in all_clips:
        teacher_path = TEACHER_DIR / f"{clip['tag']}_qwen_teacher.json"
        if teacher_path.exists():
            pass
        else:
            missing.append(clip)

    existing = len(all_clips) - len(missing)
    print(f"Total clips: {len(all_clips)}, existing teachers: {existing}, to generate: {len(missing)}")

    if not missing:
        print("All teachers already generated.")
        return

    print(f"\nLoading Qwen...")
    from run_alignment_single_audio import build_qwen_backend
    backend = build_qwen_backend()
    print("Qwen loaded.\n")

    for i, clip in enumerate(missing):
        tag = clip["tag"]
        audio, sr = sf.read(clip["audio"])
        audio = audio.astype(np.float32)
        dur = len(audio) / sr

        print(f"  [{i+1}/{len(missing)}] {tag} ({dur:.1f}s)...", end=" ", flush=True)
        t0 = time.time()
        result = backend.transcribe_and_align(audio, sample_rate=sr, language="English")
        dt = time.time() - t0

        if result is None:
            print(f"FAILED")
            continue

        teacher = {
            "tag": "qwen_baseline",
            "source_wav": clip["audio"],
            "text": result.text,
            "audio_duration_s": result.audio_duration_s,
            "words": [asdict(w) for w in result.words],
            "diagnostics": result.diagnostics,
        }
        out_path = TEACHER_DIR / f"{tag}_qwen_teacher.json"
        with open(out_path, "w") as f:
            json.dump(teacher, f, indent=2)
        print(f"{len(result.words)} words, {dt:.1f}s")

    print(f"\nDone. Teachers saved to {TEACHER_DIR}")


if __name__ == "__main__":
    main()
