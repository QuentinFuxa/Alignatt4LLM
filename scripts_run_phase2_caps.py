#!/usr/bin/env python3
"""Phase 2: probe the partial-cap lever at fixed operating point.

Keeps ``chunk_ms=450`` and ``min_start_seconds=2.0`` from Phase 0 so the
latency contract stays intact, but widens ``partial_max_new_tokens`` and
``partial_followup_max_new_tokens`` so a single partial can emit a longer
German phrase.
"""
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone

from qwen3asr_gemma_cascade_core import run_baseline


def _git(*cmd: str) -> str:
    try:
        out = subprocess.run(
            ["git", *cmd], cwd="/home/cascade_simultaneous",
            check=True, capture_output=True, text=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _git_provenance() -> dict:
    dirty = _git("status", "--porcelain")
    return {
        "git_commit_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(dirty),
        "started_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wav-path", default="test-set/audio/ccpXHNfaoy.wav")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-lang", default="German")
    parser.add_argument("--partial-max-new-tokens", type=int, default=24)
    parser.add_argument("--partial-followup-max-new-tokens", type=int, default=12)
    parser.add_argument("--max-history-utterances", type=int, default=1)
    parser.add_argument("--min-start-seconds", type=float, default=2.0)
    args = parser.parse_args()

    overrides = {
        "min_start_seconds": args.min_start_seconds,
        "max_history_utterances": args.max_history_utterances,
        "partial_max_new_tokens": args.partial_max_new_tokens,
        "partial_followup_max_new_tokens": args.partial_followup_max_new_tokens,
        "translation_alignatt_inaccessible_ms": 0.0,
        "translation_alignatt_rewind_threshold": 8,
        "target_lang": args.target_lang,
    }

    run_baseline(
        wav_path=args.wav_path,
        output_dir=args.output_dir,
        chunk_ms=450,
        runtime_overrides=overrides,
        run_provenance={
            "tag": "phase2_caps_probe",
            "caps": {
                "partial_max_new_tokens": args.partial_max_new_tokens,
                "partial_followup_max_new_tokens": args.partial_followup_max_new_tokens,
                "max_history_utterances": args.max_history_utterances,
            },
            **_git_provenance(),
        },
    )


if __name__ == "__main__":
    main()
