#!/usr/bin/env python3
"""Loop-replay predictor for the three discrete MT gates.

The per-gate feature analyses capped at F1 ≤ 0.98 on
``alignatt:source_frontier`` and F1 ≤ 0.75 on ``alignatt:rewind``
even with the closest-to-definition feature
``max_drop_vs_prev_non_none``. The reason is structural: the online
policy is a ``for``-loop that ``break``s at the first unsafe event —
whether that event is rewind or source_frontier depends on the
ordering of tokens, their aligned positions, and the running
``last_aligned`` state.

A single-scalar classifier asking "does this update have any large
backward drop" cannot distinguish "the large drop came first" from
"a source_frontier breach came first and we never reached the
drop". Both answer yes to the scalar question.

This script replays the online loop offline using just
``aligned_source_local_positions`` and
``accessible_source_local_end_exclusive`` from alignatt_metadata,
with the (constant) rewind threshold. It produces a predicted
stop_reason per update and compares against the ground-truth
stop_reason.

Expected result: near-1.0 F1 for both ``alignatt:rewind`` and
``alignatt:source_frontier`` — confirming that the metadata
contains the full information to recover the discrete gates, just
not via a single scalar threshold. This is the cleanest paper
answer to the continuous-confidence question: the observer payload
does not map to discrete gates through scalar features, only
through loop replay.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _is_finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and x == x and abs(x) != float("inf")


def replay_gate(
    aligned: list,
    accessible_end_exclusive: int,
    rewind_threshold: int,
) -> str:
    """Mirror cascade_mt_backend.py's should_stop_in_loop."""
    last_aligned = None
    for current in aligned:
        if current is not None and last_aligned is not None:
            if last_aligned - current > rewind_threshold:
                return "alignatt:rewind"
        if current is not None and current >= max(0, int(accessible_end_exclusive)):
            return "alignatt:source_frontier"
        if current is not None:
            last_aligned = current
    return "stop"


def score(input_path: Path, rewind_threshold: int) -> str:
    predicted_counts: Counter = Counter()
    actual_counts: Counter = Counter()
    confusion: dict = defaultdict(lambda: Counter())
    total = 0
    with input_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            update = json.loads(line)
            md = update.get("alignatt_metadata") or {}
            if not md:
                continue
            aligned = md.get("aligned_source_local_positions") or []
            if not aligned:
                continue
            accessible = md.get("accessible_source_local_end_exclusive")
            if accessible is None:
                continue
            actual = md.get("stop_reason") or "unknown"
            predicted = replay_gate(aligned, int(accessible), rewind_threshold)
            total += 1
            actual_counts[actual] += 1
            predicted_counts[predicted] += 1
            confusion[actual][predicted] += 1

    if total == 0:
        return f"# {input_path}: no updates with provenance"

    # Per-class P/R/F1 against the binary "this gate vs rest" target.
    def f1_binary(actual_counts, confusion_for_actual, target) -> tuple[float, float, float]:
        tp = confusion_for_actual.get(target, 0)
        fn = sum(v for k, v in confusion_for_actual.items() if k != target)
        # false positive = predicted target when actual was not target
        fp = sum(confusion[other].get(target, 0) for other in confusion
                 if other != target)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return prec, rec, f1

    lines = []
    lines.append(f"# Loop-replay gate prediction on {input_path}")
    lines.append(f"# rewind_threshold = {rewind_threshold}")
    lines.append(f"# total updates = {total}")
    lines.append(f"# actual_counts = {dict(actual_counts)}")
    lines.append(f"# predicted_counts = {dict(predicted_counts)}")
    lines.append("")
    for gate in ("alignatt:rewind", "alignatt:source_frontier", "stop"):
        if actual_counts.get(gate, 0) == 0:
            continue
        prec, rec, f1 = f1_binary(actual_counts, confusion[gate], gate)
        lines.append(
            f"{gate:<30}  n={actual_counts[gate]}"
            f"  precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}"
        )

    lines.append("")
    lines.append("# Confusion matrix (rows=actual, cols=predicted):")
    all_labels = sorted(set(list(actual_counts.keys()) + list(predicted_counts.keys())))
    header = "actual \\ predicted".ljust(30) + " ".join(f"{l[:18]:>18}" for l in all_labels)
    lines.append(header)
    for actual in all_labels:
        row = str(actual)[:28].ljust(30)
        row += " ".join(
            f"{confusion[actual].get(pred, 0):>18}" for pred in all_labels
        )
        lines.append(row)
    report = "\n".join(lines)
    print(report)
    out = input_path.parent / "loop_replay_gate_prediction.txt"
    out.write_text(report + "\n")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--rewind-threshold", type=int, default=8,
                   help="Must match translation_alignatt_rewind_threshold "
                        "for the run. Default 8 = our runtime config.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    score(Path(args.input), int(args.rewind_threshold))


if __name__ == "__main__":
    main()
