#!/usr/bin/env python3
"""Parse official IWSLT 2026 baseline scores from the supplied outputs zip."""

from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
import urllib.request
from pathlib import Path
from typing import Any
from zipfile import ZipFile


OFFICIAL_BASELINE_REPO = "https://github.com/owaski/iwslt-2026-baselines"
OFFICIAL_OUTPUTS_ZIP_URL = (
    "https://github.com/user-attachments/files/26411361/outputs.zip"
)
SCORES_RE = re.compile(
    r"^outputs/(?P<direction>en-(?:de|it|zh))/baseline/"
    r"seg(?P<segment_ms>\d+)_mss(?P<mss>[^/]+)_h(?P<history>\d+)/"
    r"segmentation_output/scores\.tsv$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--zip-path",
        type=Path,
        help="Local official outputs.zip. If omitted, --url is downloaded.",
    )
    source.add_argument(
        "--url",
        default=OFFICIAL_OUTPUTS_ZIP_URL,
        help="Official outputs.zip URL to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/official_baseline"),
        help="Directory for compact parsed JSON/TSV summaries.",
    )
    return parser.parse_args()


def materialize_zip(args: argparse.Namespace, tmp_dir: Path) -> Path:
    if args.zip_path is not None:
        if not args.zip_path.is_file():
            raise FileNotFoundError(args.zip_path)
        return args.zip_path

    destination = tmp_dir / "outputs.zip"
    urllib.request.urlretrieve(args.url, destination)
    return destination


def parse_scores_tsv(raw: str) -> dict[str, float]:
    rows = csv.DictReader(raw.splitlines(), delimiter="\t")
    scores: dict[str, float] = {}
    for row in rows:
        metric = str(row["metric"]).strip()
        value = str(row["value"]).strip()
        if metric:
            scores[metric] = float(value)
    return scores


def read_score_rows(zip_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with ZipFile(zip_path) as archive:
        for name in sorted(archive.namelist()):
            match = SCORES_RE.match(name)
            if match is None:
                continue
            scores = parse_scores_tsv(archive.read(name).decode("utf-8"))
            rows.append(
                {
                    "source": "official_iwslt_2026_baseline_outputs",
                    "baseline_repo": OFFICIAL_BASELINE_REPO,
                    "outputs_zip_url": OFFICIAL_OUTPUTS_ZIP_URL,
                    "direction": match.group("direction"),
                    "segment_ms": int(match.group("segment_ms")),
                    "mss": match.group("mss"),
                    "history": int(match.group("history")),
                    "scores_path": name,
                    "scores": scores,
                }
            )
    if not rows:
        raise RuntimeError(f"No official baseline scores.tsv files found in {zip_path}")
    return rows


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "official_baseline_scores.json"
    tsv_path = output_dir / "official_baseline_scores.tsv"

    json_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    metrics = sorted({metric for row in rows for metric in row["scores"]})
    columns = [
        "direction",
        "segment_ms",
        "mss",
        "history",
        *metrics,
        "scores_path",
    ]
    with tsv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "direction": row["direction"],
                    "segment_ms": row["segment_ms"],
                    "mss": row["mss"],
                    "history": row["history"],
                    **row["scores"],
                    "scores_path": row["scores_path"],
                }
            )

    print(f"wrote {len(rows)} rows")
    print(f"json={json_path}")
    print(f"tsv={tsv_path}")


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = materialize_zip(args, Path(tmp))
        rows = read_score_rows(zip_path)
    write_outputs(rows, args.output_dir)


if __name__ == "__main__":
    main()
