#!/usr/bin/env python3
"""Render a SUBMISSION_PRESET to a simulstream speech_processor.yaml.

Invoked from submission/docker-entrypoint.sh to hand off a validated,
frozen configuration to the official `simulstream_inference` CLI.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from cascade.submission import VALID_SUBMISSION_PRESET_NAMES, get_submission_preset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True, choices=VALID_SUBMISSION_PRESET_NAMES)
    parser.add_argument("--source-lang-code", required=True)
    parser.add_argument("--target-lang-code", required=True)
    parser.add_argument("--paper-context-path", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    preset = get_submission_preset(args.preset)
    cfg = preset.build_speech_processor_config(
        source_lang_code=args.source_lang_code,
        target_lang_code=args.target_lang_code,
        paper_context_path=args.paper_context_path,
    )

    Path(args.output).write_text(yaml.safe_dump(vars(cfg), sort_keys=False))


if __name__ == "__main__":
    main()
