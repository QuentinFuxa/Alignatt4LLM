#!/usr/bin/env python3
from __future__ import annotations

import argparse

from cascade_artifacts import DEFAULT_OUTPUT_DIR, DEFAULT_WAV_PATH
from cascade_translation_variants import DEFAULT_TRANSLATION_VARIANT_ID, TRANSLATION_VARIANTS
from qwen3asr_gemma_cascade_core import run_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the En->DE cascade baseline and persist outputs/cascade_v1 artifacts.",
    )
    parser.add_argument("--wav-path", default=DEFAULT_WAV_PATH, help="Input WAV file.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Artifact directory.")
    parser.add_argument("--chunk-ms", default=960, type=int, help="Streaming chunk size in milliseconds.")
    parser.add_argument(
        "--translation-variant",
        default=DEFAULT_TRANSLATION_VARIANT_ID,
        choices=sorted(TRANSLATION_VARIANTS),
        help="Named prompt/context variant to run.",
    )
    args = parser.parse_args()
    if (
        args.translation_variant != DEFAULT_TRANSLATION_VARIANT_ID
        and args.output_dir == DEFAULT_OUTPUT_DIR
    ):
        parser.error(
            "Non-baseline translation variants must write to a non-default output dir so outputs/cascade_v1 stays canonical."
        )
    return args


def main() -> None:
    args = parse_args()
    run_baseline(
        wav_path=args.wav_path,
        output_dir=args.output_dir,
        chunk_ms=args.chunk_ms,
        translation_variant=args.translation_variant,
    )


if __name__ == "__main__":
    main()
