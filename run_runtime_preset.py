#!/usr/bin/env python3
"""Run maintained Cascade runtime presets from one CLI."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from cascade.server import serve_cascade_processor
from cascade.presets import VALID_RUNTIME_PRESET_NAMES, get_runtime_preset
from run_simulstream_batch import resolve_input_paths, run_batch_inference


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
LOGGER = logging.getLogger("run_runtime_preset")
REPO_ROOT = Path(__file__).resolve().parent


def _add_shared_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--preset",
        required=True,
        choices=VALID_RUNTIME_PRESET_NAMES,
        help="Named runtime preset.",
    )
    parser.add_argument("--source", default="en", help="Source language code.")
    parser.add_argument("--target", default="de", help="Target language code.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    batch = subparsers.add_parser(
        "batch",
        help="Run a runtime preset offline and write SimulStream-style logs.",
    )
    _add_shared_runtime_args(batch)
    batch_group = batch.add_mutually_exclusive_group(required=True)
    batch_group.add_argument(
        "--inputs",
        nargs="+",
        help="Input media paths (.wav, .mp4, ...).",
    )
    batch_group.add_argument(
        "--input-dir",
        help="Directory containing supported input media files.",
    )
    batch.add_argument("--output-dir", required=True)

    server = subparsers.add_parser(
        "server",
        help="Start the websocket server with a runtime preset.",
    )
    _add_shared_runtime_args(server)
    server.add_argument("--host", default="0.0.0.0")
    server.add_argument("--port", type=int, default=8765)
    server.add_argument("--pool-size", type=int, default=1)
    server.add_argument("--acquire-timeout", type=int, default=600)
    server.add_argument("--metrics-log-file", default="metrics.jsonl")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = get_runtime_preset(args.preset)
    LOGGER.info("Using preset %s: %s", preset.name, preset.description)

    if args.command == "batch":
        processor_config = preset.build_speech_processor_config(
            source_lang_code=args.source.lower(),
            target_lang_code=args.target.lower(),
            repo_root=REPO_ROOT,
        )
        input_paths = resolve_input_paths(inputs=args.inputs, input_dir=args.input_dir)
        run_batch_inference(
            processor_config=processor_config,
            input_paths=input_paths,
            output_dir=args.output_dir,
            source_lang_code=args.source.lower(),
            target_lang_code=args.target.lower(),
        )
        return

    if args.command == "server":
        processor_config = preset.build_speech_processor_config(
            source_lang_code=args.source.lower(),
            target_lang_code=args.target.lower(),
            repo_root=REPO_ROOT,
        )
        asyncio.run(
            serve_cascade_processor(
                speech_processor_config=processor_config,
                hostname=args.host,
                port=args.port,
                pool_size=args.pool_size,
                acquire_timeout=args.acquire_timeout,
                metrics_log_file=args.metrics_log_file,
            )
        )
        return

    raise AssertionError(f"Unhandled subcommand: {args.command}")


if __name__ == "__main__":
    main()
