#!/usr/bin/env python3
"""Run IWSLT 2026 ACL Talks test-set logs for main and context tracks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alignatt4llm.presets import VALID_RUNTIME_PRESET_NAMES, get_runtime_preset
from alignatt4llm.cli.batch import resolve_input_paths, run_batch_inference



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--paper-context-dir", required=True)
    parser.add_argument(
        "--output-root",
        default="outputs/iwslt26_acl_testset",
    )
    parser.add_argument("--source", default="en")
    parser.add_argument("--targets", nargs="+", default=["de", "it", "zh"])
    parser.add_argument(
        "--presets",
        nargs="+",
        default=["gemma_low_latency", "gemma_high_latency"],
        choices=VALID_RUNTIME_PRESET_NAMES,
    )
    parser.add_argument(
        "--tracks",
        nargs="+",
        default=["main", "context"],
        choices=("main", "context"),
    )
    parser.add_argument(
        "--paper-context-mode",
        default="title_and_chunks",
        choices=("title_abstract", "retrieved_chunks", "title_and_chunks"),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a track/preset/target output directory when manifest.json exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = resolve_input_paths(inputs=None, input_dir=args.input_dir)
    output_root = Path(args.output_root)

    for target in args.targets:
        for preset_name in args.presets:
            preset = get_runtime_preset(preset_name)
            for track in args.tracks:
                cfg = preset.build_speech_processor_config(
                    source_lang_code=args.source,
                    target_lang_code=target,
                    repo_root=REPO_ROOT,
                )
                paper_context_dir = None
                if track == "context":
                    cfg.paper_context_mode = args.paper_context_mode
                    paper_context_dir = args.paper_context_dir

                output_dir = (
                    output_root
                    / track
                    / preset_name
                    / f"{args.source}-{target}"
                )
                if args.skip_existing and (output_dir / "manifest.json").is_file():
                    print(f"Skipping existing output={output_dir}", flush=True)
                    continue
                print(
                    f"\n=== track={track} preset={preset_name} "
                    f"{args.source}->{target} output={output_dir} ===",
                    flush=True,
                )
                run_batch_inference(
                    processor_config=cfg,
                    input_paths=input_paths,
                    output_dir=str(output_dir),
                    source_lang_code=args.source,
                    target_lang_code=target,
                    paper_context_dir=paper_context_dir,
                )


if __name__ == "__main__":
    main()
