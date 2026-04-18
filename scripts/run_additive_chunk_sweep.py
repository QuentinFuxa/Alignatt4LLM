#!/usr/bin/env python3
"""Run an additive chunk_ms calibration sweep.

For a given `--chunk-ms`, load the cascade models once and then run every
combination of (dataset in {test-set, dev-set}) x (direction in {de, it, zh})
inside a single Python process. Keeping the ASR + MT bundle resident across
the six sub-jobs avoids paying the ~5 minute reload cost per direction and
matches the AGENTS.md "keep models hot" rule.

Each sub-job materialises the same artifacts as ``run_simulstream_batch.py``:
`manifest.json`, `hypothesis.jsonl`, `stream_updates.jsonl`. The scripted
preset is equivalent to `main_low_latency` / `main_high_latency` except for
`chunk_ms`.

Usage (from `.venv-inference`):

    python scripts/run_additive_chunk_sweep.py \
        --chunk-ms 850 \
        --output-tag chunk850_borderp1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cascade_runtime import LoadedModelBundle
from cascade_submission import get_submission_preset
from run_simulstream_batch import resolve_input_paths, run_batch_inference
from cascade_simulstream_processor import CascadeAlignAttProcessor


DIRECTIONS = ("de", "it", "zh")
DATASETS = (
    ("devset", "dev-set/audio"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunk-ms",
        type=int,
        required=True,
        help="Cascade streaming chunk in milliseconds (e.g. 850 or 1900).",
    )
    parser.add_argument(
        "--output-tag",
        required=True,
        help=(
            "Tag used to form the output directory name: "
            "outputs/iwslt26_<dataset>_<tag>_en<target>."
        ),
    )
    parser.add_argument(
        "--base-preset",
        default="main_low_latency",
        help=(
            "Submission preset whose knobs seed the runtime config (only "
            "chunk_ms is overridden). Default: main_low_latency."
        ),
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        default=list(DIRECTIONS),
        choices=DIRECTIONS,
        help="Target-language codes to run (default: de it zh).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[name for name, _ in DATASETS],
        choices=[name for name, _ in DATASETS],
        help="Dataset tags to run (default: testset devset).",
    )
    return parser.parse_args()


def build_processor_config(
    *, base_preset_name: str, chunk_ms: int, target_lang_code: str
) -> SimpleNamespace:
    preset = get_submission_preset(base_preset_name)
    processor_config = preset.build_speech_processor_config(
        source_lang_code="en",
        target_lang_code=target_lang_code,
        paper_context_path=None,
    )
    processor_config.chunk_ms = chunk_ms
    processor_config.speech_chunk_size = chunk_ms / 1000.0
    return processor_config


def main() -> None:
    args = parse_args()
    datasets_by_name = dict(DATASETS)
    for name in args.datasets:
        if name not in datasets_by_name:
            raise ValueError(f"Unknown dataset tag: {name}")

    print(
        f"Additive sweep chunk_ms={args.chunk_ms} "
        f"directions={args.directions} datasets={args.datasets}",
        flush=True,
    )

    first_target = args.directions[0]
    load_config = build_processor_config(
        base_preset_name=args.base_preset,
        chunk_ms=args.chunk_ms,
        target_lang_code=first_target,
    )
    print(f"Loading models once (initial target={first_target})", flush=True)
    CascadeAlignAttProcessor.load_model(load_config)

    # Keep the ASR + MT bundle hot across direction changes. The default
    # `_ensure_bundle` path recreates `LoadedModelBundle` whenever any element
    # of the bundle key changes (target_lang, heads path, ...), which forces a
    # ~3-5 minute ASR + MT reload per direction. We instead reuse the already
    # loaded bundle and rely on `set_target_language` to swap MT alignatt heads,
    # which is a cheap observer refresh. This matches the `set_target_language`
    # hot-swap pattern already relied on by `tmp/run_same_audio_targets.py`.
    def _ensure_hot_bundle(cls, runtime_config):
        if cls._bundle is None:
            cls._bundle = LoadedModelBundle(runtime_config)
            cls._bundle.load()
            cls._bundle_signature = cls._bundle_key(runtime_config)
        else:
            cls._bundle.config = runtime_config
        return cls._bundle

    CascadeAlignAttProcessor._ensure_bundle = classmethod(_ensure_hot_bundle)

    summary: list[tuple[str, str, str]] = []
    for target in args.directions:
        for dataset_name in args.datasets:
            input_dir = datasets_by_name[dataset_name]
            output_dir = (
                f"outputs/iwslt26_{dataset_name}_{args.output_tag}_en{target}"
            )
            input_paths = resolve_input_paths(inputs=None, input_dir=input_dir)
            processor_config = build_processor_config(
                base_preset_name=args.base_preset,
                chunk_ms=args.chunk_ms,
                target_lang_code=target,
            )
            print(
                f"\n>>> Running {dataset_name} en->{target} "
                f"({len(input_paths)} inputs) -> {output_dir}",
                flush=True,
            )
            run_batch_inference(
                processor_config=processor_config,
                input_paths=input_paths,
                output_dir=output_dir,
                source_lang_code="en",
                target_lang_code=target,
                explicit_paper_context_path=None,
                paper_context_dir=None,
            )
            summary.append((dataset_name, target, output_dir))

    print("\n" + "=" * 60)
    print(f"Additive sweep chunk_ms={args.chunk_ms} done")
    for dataset_name, target, output_dir in summary:
        print(f"  {dataset_name:>7s}  en->{target}  {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
