#!/usr/bin/env python3
from __future__ import annotations

import argparse

from cascade_artifacts import DEFAULT_OUTPUT_DIR, DEFAULT_WAV_PATH
from qwen3asr_gemma_cascade_core import run_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the En->DE cascade baseline and persist outputs/cascade_v1 artifacts.",
    )
    parser.add_argument("--wav-path", default=DEFAULT_WAV_PATH, help="Input WAV file.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Artifact directory.")
    parser.add_argument("--chunk-ms", default=960, type=int, help="Streaming chunk size in milliseconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_baseline(
        wav_path=args.wav_path,
        output_dir=args.output_dir,
        chunk_ms=args.chunk_ms,
    )


if __name__ == "__main__":
    main()
