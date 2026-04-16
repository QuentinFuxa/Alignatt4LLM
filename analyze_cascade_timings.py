#!/usr/bin/env python3
"""Aggregate AlignAtt timing breakdown from ``stream_updates.jsonl``.

The cascade runtime records per-update ``translation_timings_ms`` dicts with
keys such as ``prompt_render``, ``prompt_cache_restore``, ``draft_decode``,
``alignment_probe``, ``alignment_filter`` and ``total`` (see
``cascade_mt_backend.TransformersAlignAttGemmaMTBackend``). Phase 3 of
``PLAN.md`` asks for an explicit measurement of how much of each translation
step is spent in each phase, so we can distinguish "lower end-to-end latency"
from "better AlignAtt observer". This script produces that breakdown from an
existing output bundle, without re-running inference.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from cascade_artifacts import DEFAULT_OUTPUT_DIR, STREAM_UPDATES_FILENAME


DEFAULT_PHASE_KEYS = (
    "prompt_render",
    "prompt_cache_restore",
    "draft_decode",
    "alignment_probe",
    "alignment_filter",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help="Directory holding the cascade bundle to analyse.",
    )
    parser.add_argument(
        "--phase-keys",
        nargs="+",
        default=list(DEFAULT_PHASE_KEYS),
        help="timings_ms keys to aggregate.",
    )
    return parser.parse_args()


def iter_stream_updates(stream_updates_path: Path):
    with stream_updates_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def collect_timings(
    stream_updates_path: Path,
    phase_keys: list[str],
) -> tuple[dict[str, list[float]], list[float], int, int]:
    per_phase: dict[str, list[float]] = {key: [] for key in phase_keys}
    totals: list[float] = []
    total_updates = 0
    updates_with_timings = 0

    for payload in iter_stream_updates(stream_updates_path):
        total_updates += 1
        timings = payload.get("translation_timings_ms")
        if not isinstance(timings, dict):
            continue
        updates_with_timings += 1
        for key in phase_keys:
            value = timings.get(key)
            if isinstance(value, (int, float)):
                per_phase[key].append(float(value))
        total_value = timings.get("total")
        if isinstance(total_value, (int, float)):
            totals.append(float(total_value))

    return per_phase, totals, total_updates, updates_with_timings


def summarise(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "sum_ms": 0.0}
    sorted_values = sorted(values)
    p95_index = max(0, int(round(0.95 * (len(sorted_values) - 1))))
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": sorted_values[p95_index],
        "sum_ms": sum(values),
    }


def render_report(
    per_phase: dict[str, list[float]],
    totals: list[float],
    total_updates: int,
    updates_with_timings: int,
) -> str:
    total_sum_ms = sum(totals) or 1.0
    lines: list[str] = []
    lines.append("AlignAtt timing breakdown")
    lines.append("-" * 64)
    lines.append(f"updates total          : {total_updates}")
    lines.append(f"updates with timings   : {updates_with_timings}")
    lines.append(f"total translate ms sum : {sum(totals):.1f}")
    lines.append("")
    header = f"{'phase':<22} {'count':>6} {'mean_ms':>10} {'median_ms':>11} {'p95_ms':>10} {'sum_ms':>12} {'share':>7}"
    lines.append(header)
    lines.append("-" * len(header))
    for phase, values in per_phase.items():
        stats = summarise(values)
        share = stats["sum_ms"] / total_sum_ms
        lines.append(
            f"{phase:<22} {int(stats['count']):>6d} "
            f"{stats['mean_ms']:>10.2f} {stats['median_ms']:>11.2f} "
            f"{stats['p95_ms']:>10.2f} {stats['sum_ms']:>12.1f} "
            f"{share * 100.0:>6.1f}%"
        )
    if totals:
        stats = summarise(totals)
        lines.append("")
        lines.append(
            f"{'total':<22} {int(stats['count']):>6d} "
            f"{stats['mean_ms']:>10.2f} {stats['median_ms']:>11.2f} "
            f"{stats['p95_ms']:>10.2f} {stats['sum_ms']:>12.1f} "
            f"{100.0:>6.1f}%"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    stream_updates_path = args.output_dir / STREAM_UPDATES_FILENAME
    if not stream_updates_path.exists():
        raise SystemExit(f"Missing {stream_updates_path}; run a baseline first.")
    per_phase, totals, total_updates, updates_with_timings = collect_timings(
        stream_updates_path, args.phase_keys
    )
    print(render_report(per_phase, totals, total_updates, updates_with_timings))


if __name__ == "__main__":
    main()
