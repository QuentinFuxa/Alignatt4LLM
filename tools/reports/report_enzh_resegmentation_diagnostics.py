#!/usr/bin/env python3
"""Summarize EN->ZH resegmented hypothesis error shapes from recovered runs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import difflib
import json
from pathlib import Path
import statistics
import string
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX = (
    REPO_ROOT
    / "outputs"
    / "diagnostics_jarvislab_20260606"
    / "diagnostics_artifact_index.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "plots"

PUNCTUATION_CHARS = set(string.punctuation) | set(
    "，。！？；：、“”‘’（）《》〈〉【】—…·"
)

SUMMARY_COLUMNS = (
    "relative_dir",
    "num_inputs",
    "chunk_ms",
    "longyaal_cu_ms",
    "xcometxl",
    "alignatt_policy_family",
    "alignatt_guard_flags",
    "instance_count",
    "empty_prediction_count",
    "punctuation_only_prediction_count",
    "mean_prediction_chars",
    "mean_reference_chars",
    "mean_prediction_reference_char_ratio",
    "overlong_prediction_count",
    "underlong_prediction_count",
    "mean_char_similarity",
    "median_char_similarity",
    "low_similarity_count",
)


@dataclass(frozen=True)
class SegmentDiagnostic:
    index: int
    docid: int | str | None
    segid: int | str | None
    prediction: str
    reference: str
    prediction_chars: int
    reference_chars: int
    prediction_reference_char_ratio: float | None
    char_similarity: float
    punctuation_only_prediction: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-stem",
        default="enzh_resegmentation_diagnostics",
    )
    parser.add_argument(
        "--relative-dir",
        action="append",
        default=[],
        help="Restrict to one recovered relative_dir; may be repeated.",
    )
    parser.add_argument("--worst-segments-per-run", type=int, default=5)
    return parser.parse_args()


def finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        return None
    return numeric


def char_similarity(prediction: str, reference: str) -> float:
    return difflib.SequenceMatcher(a=prediction, b=reference).ratio()


def is_punctuation_only(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and all(char in PUNCTUATION_CHARS for char in stripped)


def load_instances(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def segment_diagnostics(instances: list[dict[str, Any]]) -> list[SegmentDiagnostic]:
    diagnostics: list[SegmentDiagnostic] = []
    for fallback_index, row in enumerate(instances):
        prediction = str(row.get("prediction") or "")
        reference = str(row.get("reference") or "")
        prediction_chars = len(prediction)
        reference_chars = len(reference)
        ratio = (
            None
            if reference_chars <= 0
            else float(prediction_chars) / float(reference_chars)
        )
        diagnostics.append(
            SegmentDiagnostic(
                index=int(row.get("index", fallback_index)),
                docid=row.get("docid"),
                segid=row.get("segid"),
                prediction=prediction,
                reference=reference,
                prediction_chars=prediction_chars,
                reference_chars=reference_chars,
                prediction_reference_char_ratio=ratio,
                char_similarity=char_similarity(prediction, reference),
                punctuation_only_prediction=is_punctuation_only(prediction),
            )
        )
    return diagnostics


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_run(
    index_row: dict[str, Any],
    diagnostics: list[SegmentDiagnostic],
) -> dict[str, Any]:
    prediction_chars = [float(item.prediction_chars) for item in diagnostics]
    reference_chars = [float(item.reference_chars) for item in diagnostics]
    ratios = [
        float(item.prediction_reference_char_ratio)
        for item in diagnostics
        if item.prediction_reference_char_ratio is not None
    ]
    similarities = [float(item.char_similarity) for item in diagnostics]
    return {
        "relative_dir": index_row.get("relative_dir") or "",
        "num_inputs": index_row.get("num_inputs"),
        "chunk_ms": index_row.get("chunk_ms"),
        "longyaal_cu_ms": finite_float(index_row.get("longyaal_cu_ms")),
        "xcometxl": finite_float(index_row.get("xcometxl")),
        "alignatt_policy_family": index_row.get("alignatt_policy_family") or "",
        "alignatt_guard_flags": index_row.get("alignatt_guard_flags") or "",
        "instance_count": len(diagnostics),
        "empty_prediction_count": sum(
            1 for item in diagnostics if not item.prediction.strip()
        ),
        "punctuation_only_prediction_count": sum(
            1 for item in diagnostics if item.punctuation_only_prediction
        ),
        "mean_prediction_chars": mean(prediction_chars),
        "mean_reference_chars": mean(reference_chars),
        "mean_prediction_reference_char_ratio": mean(ratios),
        "overlong_prediction_count": sum(
            1
            for item in diagnostics
            if item.prediction_reference_char_ratio is not None
            and item.prediction_reference_char_ratio > 1.5
        ),
        "underlong_prediction_count": sum(
            1
            for item in diagnostics
            if item.prediction_reference_char_ratio is not None
            and item.prediction_reference_char_ratio < 0.5
        ),
        "mean_char_similarity": mean(similarities),
        "median_char_similarity": (
            statistics.median(similarities) if similarities else None
        ),
        "low_similarity_count": sum(1 for value in similarities if value < 0.35),
    }


def worst_segments(
    diagnostics: list[SegmentDiagnostic],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    ordered = sorted(
        diagnostics,
        key=lambda item: (
            item.char_similarity,
            -(item.prediction_reference_char_ratio or 0.0),
            item.index,
        ),
    )
    return [
        {
            "index": item.index,
            "docid": item.docid,
            "segid": item.segid,
            "char_similarity": item.char_similarity,
            "prediction_reference_char_ratio": item.prediction_reference_char_ratio,
            "punctuation_only_prediction": item.punctuation_only_prediction,
            "prediction": item.prediction,
            "reference": item.reference,
        }
        for item in ordered[: max(0, int(limit))]
    ]


def candidate_index_rows(
    index_path: Path,
    *,
    relative_dirs: set[str],
) -> list[dict[str, Any]]:
    rows = json.loads(index_path.read_text(encoding="utf-8"))
    selected: list[dict[str, Any]] = []
    for row in rows:
        rel = str(row.get("relative_dir") or "")
        if relative_dirs and rel not in relative_dirs:
            continue
        if not row.get("valid_for_claims"):
            continue
        if row.get("target_language_code") != "zh":
            continue
        if row.get("mt_backend_name") != "milmmt_vllm_alignatt":
            continue
        copied_dir = Path(str(row.get("copied_dir") or ""))
        if not (copied_dir / "instances.resegmented.jsonl").is_file():
            continue
        selected.append(row)
    selected.sort(
        key=lambda row: (
            int(row.get("num_inputs") or 0) < 3,
            finite_float(row.get("longyaal_cu_ms")) or 999999.0,
            str(row.get("relative_dir") or ""),
        )
    )
    return selected


def format_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def write_outputs(
    rows: list[dict[str, Any]],
    worst: dict[str, list[dict[str, Any]]],
    *,
    output_dir: Path,
    output_stem: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = output_dir / f"{output_stem}.tsv"
    json_path = output_dir / f"{output_stem}.json"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: format_value(row.get(column)) for column in SUMMARY_COLUMNS}
            )
    json_path.write_text(
        json.dumps(
            {
                "summary": rows,
                "worst_segments_by_run": worst,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"tsv": tsv_path, "json": json_path}


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    worst: dict[str, list[dict[str, Any]]] = {}
    for index_row in candidate_index_rows(
        args.index,
        relative_dirs=set(args.relative_dir),
    ):
        rel = str(index_row.get("relative_dir") or "")
        instances_path = Path(str(index_row["copied_dir"])) / "instances.resegmented.jsonl"
        diagnostics = segment_diagnostics(load_instances(instances_path))
        rows.append(summarize_run(index_row, diagnostics))
        worst[rel] = worst_segments(
            diagnostics,
            limit=int(args.worst_segments_per_run),
        )
    if not rows:
        raise SystemExit("No matching EN->ZH MiLMMT runs with instances.resegmented.jsonl.")
    paths = write_outputs(rows, worst, output_dir=args.output_dir, output_stem=args.output_stem)
    print(paths["tsv"])
    print(paths["json"])


if __name__ == "__main__":
    main()
