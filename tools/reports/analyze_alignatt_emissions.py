#!/usr/bin/env python3
"""Inspect MT AlignAtt decisions emission by emission for one audio."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--wav-name", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--label", default=None)
    parser.add_argument("--tail", type=int, default=8)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def unit_text(units: list[Any]) -> str:
    return "".join(str(unit) for unit in units)


def text_tail(text: str | None, *, max_chars: int = 90) -> str:
    if not text:
        return ""
    text = " ".join(str(text).split())
    return text if len(text) <= max_chars else text[-max_chars:]


def provenance_stats(rows: list[dict[str, float]]) -> dict[str, float | None]:
    if not rows:
        return {
            "src_acc_mean": None,
            "src_inacc_mean": None,
            "non_source_mean": None,
            "suffix_mean": None,
        }
    return {
        "src_acc_mean": mean(float(row.get("source_accessible", 0.0)) for row in rows),
        "src_inacc_mean": mean(float(row.get("source_inaccessible", 0.0)) for row in rows),
        "non_source_mean": mean(float(row.get("non_source_prompt", 0.0)) for row in rows),
        "suffix_mean": mean(float(row.get("suffix", 0.0)) for row in rows),
    }


def token_provenance(
    rows: list[dict[str, float]],
    index: int | None,
) -> dict[str, float | None]:
    if index is None or index < 0 or index >= len(rows):
        return {
            "token_src_acc": None,
            "token_src_inacc": None,
            "token_non_source": None,
            "token_suffix": None,
        }
    row = rows[index]
    return {
        "token_src_acc": float(row.get("source_accessible", 0.0)),
        "token_src_inacc": float(row.get("source_inaccessible", 0.0)),
        "token_non_source": float(row.get("non_source_prompt", 0.0)),
        "token_suffix": float(row.get("suffix", 0.0)),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metadata_rows = [row for row in rows if row.get("alignatt_metadata")]
    stop_counter = Counter(
        (row.get("alignatt_metadata") or {}).get("stop_reason")
        for row in metadata_rows
    )
    unsafe_counter = Counter(
        (row.get("alignatt_metadata") or {}).get("unsafe_reason")
        for row in metadata_rows
    )
    trimmed_count = sum(
        1
        for row in metadata_rows
        if (row.get("alignatt_metadata") or {}).get("word_boundary_trimmed")
    )
    emitted_units = sum(len(row.get("new_words") or []) for row in rows)
    source_frontier_blocks = [
        row
        for row in metadata_rows
        if (row.get("alignatt_metadata") or {}).get("stop_reason")
        == "alignatt:source_frontier"
    ]
    return {
        "updates": len(rows),
        "updates_with_metadata": len(metadata_rows),
        "emitted_units": emitted_units,
        "stop_reasons": dict(stop_counter),
        "unsafe_reasons": dict(unsafe_counter),
        "word_boundary_trimmed_count": trimmed_count,
        "source_frontier_block_count": len(source_frontier_blocks),
    }


def flatten_update(row: dict[str, Any], *, tail: int) -> dict[str, Any]:
    md = row.get("alignatt_metadata") or {}
    provenance = md.get("provenance_per_draft_token") or []
    unsafe_index = md.get("unsafe_target_token_index")
    accepted_count = md.get("accepted_token_count")
    blocked_token_stats = token_provenance(provenance, unsafe_index)
    last_accepted_stats = token_provenance(
        provenance,
        None if accepted_count is None else int(accepted_count) - 1,
    )
    prov_means = provenance_stats(provenance)
    aligned_positions = md.get("aligned_source_local_positions") or []
    observer = md.get("observer_diagnostics") or {}
    return {
        "update_idx": row.get("update_idx"),
        "audio_ms": row.get("audio_processed_ms"),
        "new_units": unit_text(row.get("new_words") or []),
        "new_unit_count": len(row.get("new_words") or []),
        "stop_reason": md.get("stop_reason"),
        "unsafe_reason": md.get("unsafe_reason"),
        "word_boundary_trimmed": md.get("word_boundary_trimmed"),
        "accepted_tokens": md.get("accepted_token_count"),
        "candidate_tokens": md.get("accepted_candidate_token_count"),
        "unsafe_target_token_index": unsafe_index,
        "unsafe_token_starts_new_unit": md.get("unsafe_token_starts_new_unit"),
        "blocked_source_local_position": md.get("blocked_source_local_position"),
        "blocked_source_unit_index": md.get("blocked_source_unit_index"),
        "accessible_source_units": md.get("accessible_source_unit_count"),
        "source_units": md.get("source_unit_count"),
        "accessible_source_token_end": md.get("accessible_source_local_end_exclusive"),
        "source_token_count": md.get("source_token_count"),
        "aligned_positions_tail": ",".join(str(x) for x in aligned_positions[-tail:]),
        "observer_effective_heads": observer.get("effective_head_count"),
        "observer_missing_heads": ",".join(str(x) for x in observer.get("missing_heads") or []),
        **prov_means,
        **{f"blocked_{k}": v for k, v in blocked_token_stats.items()},
        **{f"last_accepted_{k}": v for k, v in last_accepted_stats.items()},
        "asr_tail": text_tail(row.get("asr_text")),
        "accepted_target_tail": text_tail(row.get("partial_accepted_target")),
        "draft_target_tail": text_tail(row.get("partial_draft_target")),
    }


def main() -> None:
    args = parse_args()
    stream_path = args.run_dir / "stream_updates.jsonl"
    rows = [
        row
        for row in load_jsonl(stream_path)
        if row.get("wav_name") == args.wav_name or row.get("input_name") == args.wav_name
    ]
    if not rows:
        raise SystemExit(f"No updates found for {args.wav_name} in {stream_path}")

    label = args.label or args.run_dir.name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    flat_rows = [flatten_update(row, tail=args.tail) for row in rows]
    summary = {
        "label": label,
        "run_dir": str(args.run_dir),
        "wav_name": args.wav_name,
        **summarize_rows(rows),
    }

    prefix = args.output_dir / f"{label}_{Path(args.wav_name).stem}"
    (prefix.with_suffix(".summary.json")).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    columns = list(flat_rows[0])
    with prefix.with_suffix(".emissions.tsv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(flat_rows)

    interesting = [
        row
        for row in flat_rows
        if row.get("stop_reason") == "alignatt:source_frontier"
        or row.get("new_unit_count", 0) >= 8
        or row.get("word_boundary_trimmed")
    ]
    interesting = interesting[:120]
    with prefix.with_suffix(".interesting.tsv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(interesting)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"emissions={prefix.with_suffix('.emissions.tsv')}")
    print(f"interesting={prefix.with_suffix('.interesting.tsv')}")


if __name__ == "__main__":
    main()
