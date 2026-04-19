#!/usr/bin/env python3
"""Submission-oriented entry points for IWSLT 2026 Simultaneous."""

from __future__ import annotations

import argparse
import asyncio
import logging

from cascade.server import serve_submission_processor
from cascade.submission import VALID_SUBMISSION_PRESET_NAMES, get_submission_preset
from run_simulstream_batch import resolve_input_paths, run_batch_inference


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
LOGGER = logging.getLogger("run_iwslt_submission")


def _add_shared_submission_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--preset",
        required=True,
        choices=VALID_SUBMISSION_PRESET_NAMES,
        help="Named submission preset that freezes the validated runtime knobs.",
    )
    parser.add_argument("--source", default="en", help="Source language code.")
    parser.add_argument("--target", default="de", help="Target language code.")
    parser.add_argument(
        "--paper-context-path",
        default=None,
        help="Single PaperArtifact JSON for one-stream extra-context runs.",
    )
    parser.add_argument(
        "--paper-context-dir",
        default=None,
        help=(
            "Directory of PaperArtifact JSON files matched by input stem. "
            "Only used by the batch subcommand."
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    batch = subparsers.add_parser(
        "batch",
        help="Run the frozen submission preset offline and write SimulStream-style logs.",
    )
    _add_shared_submission_args(batch)
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
        help="Start a websocket server with a frozen submission preset for Docker evaluation.",
    )
    _add_shared_submission_args(server)
    server.add_argument("--host", default="0.0.0.0")
    server.add_argument("--port", type=int, default=8765)
    server.add_argument("--pool-size", type=int, default=1)
    server.add_argument("--acquire-timeout", type=int, default=600)
    server.add_argument("--metrics-log-file", default="metrics.jsonl")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = get_submission_preset(args.preset)
    LOGGER.info("Using preset %s: %s", preset.name, preset.description)

    if args.command == "batch":
        if args.paper_context_path is not None and args.paper_context_dir is not None:
            raise ValueError("Use either --paper-context-path or --paper-context-dir, not both.")
        if (
            preset.track == "extra_context"
            and args.paper_context_path is None
            and args.paper_context_dir is None
        ):
            raise ValueError(
                "Extra-context presets require --paper-context-path or --paper-context-dir "
                "for offline batch generation."
            )
        processor_config = preset.build_speech_processor_config(
            source_lang_code=args.source,
            target_lang_code=args.target,
            paper_context_path=args.paper_context_path,
        )
        input_paths = resolve_input_paths(inputs=args.inputs, input_dir=args.input_dir)
        run_batch_inference(
            processor_config=processor_config,
            input_paths=input_paths,
            output_dir=args.output_dir,
            source_lang_code=args.source,
            target_lang_code=args.target,
            explicit_paper_context_path=args.paper_context_path,
            paper_context_dir=args.paper_context_dir,
        )
        return

    if args.command == "server":
        if args.paper_context_dir is not None:
            raise ValueError("--paper-context-dir is only supported by the batch subcommand.")
        processor_config = preset.build_speech_processor_config(
            source_lang_code=args.source,
            target_lang_code=args.target,
            paper_context_path=args.paper_context_path,
        )
        asyncio.run(
            serve_submission_processor(
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
