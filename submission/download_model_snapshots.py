#!/usr/bin/env python3
"""Download the exact model snapshots bundled into the DockerHub image."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


SNAPSHOTS = (
    {
        "repo_id": "Qwen/Qwen3-ASR-1.7B",
        "revision": "7278e1e70fe206f11671096ffdd38061171dd6e5",
        "local_name": "qwen3-asr-1.7b",
    },
    {
        "repo_id": "Qwen/Qwen3-ForcedAligner-0.6B",
        "revision": "c7cbfc2048c462b0d63a45797104fc9db3ad62b7",
        "local_name": "qwen3-forced-aligner-0.6b",
    },
    {
        "repo_id": "google/gemma-4-E4B-it",
        "revision": "83df0a889143b1dbfc61b591bbc639540fd9ce4c",
        "local_name": "gemma-4-e4b-it",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--token-file", default=None)
    return parser.parse_args()


def read_token(path: str | None) -> str | None:
    if path:
        token_path = Path(path)
        if token_path.is_file():
            token = token_path.read_text(encoding="utf-8").strip()
            if token:
                return token
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token = read_token(args.token_file)

    for snapshot in SNAPSHOTS:
        destination = output_dir / snapshot["local_name"]
        print(
            "Downloading "
            f"{snapshot['repo_id']}@{snapshot['revision']} -> {destination}",
            flush=True,
        )
        snapshot_download(
            repo_id=snapshot["repo_id"],
            revision=snapshot["revision"],
            local_dir=destination,
            token=token,
        )


if __name__ == "__main__":
    main()
