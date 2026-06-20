#!/usr/bin/env python3
"""Score offline backbone-screening arms with XCOMET-XL.

Runs in .venv-evaluation on a GPU box (XCOMET-XL cached on the campaign
instance). Inputs are line-aligned: --source (EN segments), --reference (ZH
reference), and a directory of <tag>.zh.txt arm outputs from
``screen_offline_backbones.py``. Emits one JSON with the system score per arm.
Arms whose line count does not match the source are recorded as errors and
skipped (e.g. the 20-line smoke file).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--mt-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Unbabel/XCOMET-XL")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    from comet import download_model, load_from_checkpoint

    source = args.source.read_text(encoding="utf-8").splitlines()
    reference = args.reference.read_text(encoding="utf-8").splitlines()
    checkpoint = download_model(args.model)
    model = load_from_checkpoint(checkpoint)

    results: dict[str, dict] = {}
    for mt_path in sorted(args.mt_dir.glob("*.zh.txt")):
        translations = mt_path.read_text(encoding="utf-8").splitlines()
        tag = mt_path.name.removesuffix(".zh.txt")
        if len(translations) != len(source):
            results[tag] = {
                "error": f"line count {len(translations)} != {len(source)}"
            }
            continue
        data = [
            {"src": src, "mt": mt, "ref": ref}
            for src, mt, ref in zip(source, translations, reference)
        ]
        prediction = model.predict(data, batch_size=args.batch_size, gpus=1)
        results[tag] = {
            "system_score": float(prediction.system_score),
            "n_segments": len(data),
        }
        print(tag, results[tag])

    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
