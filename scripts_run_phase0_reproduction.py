#!/usr/bin/env python3
"""One-shot Phase 0 reproduction: run the PLAN operating point on the control audio.

Writes outputs under ``--output-dir`` so we can compare BLEU / chrF / LongYAAL CU
against ``outputs/revalidate_phaseA_v2`` (BLEU 28.22, chrF 63.53, CU 1747.19).
"""
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone

from qwen3asr_gemma_cascade_core import run_baseline


def _git(*cmd: str) -> str:
    try:
        out = subprocess.run(
            ["git", *cmd],
            cwd="/home/cascade_simultaneous",
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _git_provenance() -> dict:
    dirty = _git("status", "--porcelain")
    return {
        "git_commit_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(dirty),
        "git_dirty_files": [line for line in dirty.splitlines() if line.strip()],
        "started_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    }


OPERATING_POINT = {
    "min_start_seconds": 2.0,
    "max_history_utterances": 1,
    "partial_max_new_tokens": 16,
    "partial_followup_max_new_tokens": 8,
    "translation_alignatt_inaccessible_ms": 0.0,
    "translation_alignatt_rewind_threshold": 8,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wav-path", default="test-set/audio/ccpXHNfaoy.wav")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-ms", default=450, type=int)
    parser.add_argument(
        "--target-lang",
        default="German",
        choices=["German", "Italian", "Chinese"],
    )
    parser.add_argument(
        "--heads-path",
        default=None,
        help="Override translation_alignatt_heads_path (e.g. shared kernel file).",
    )
    args = parser.parse_args()

    overrides = dict(OPERATING_POINT)
    if args.target_lang is not None:
        overrides["target_lang"] = args.target_lang
    if args.heads_path is not None:
        overrides["translation_alignatt_heads_path"] = args.heads_path

    provenance = {"tag": "phase0_reproduction_v3", **_git_provenance()}
    run_baseline(
        wav_path=args.wav_path,
        output_dir=args.output_dir,
        chunk_ms=args.chunk_ms,
        runtime_overrides=overrides,
        run_provenance=provenance,
    )


if __name__ == "__main__":
    main()
