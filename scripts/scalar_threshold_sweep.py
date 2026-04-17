#!/usr/bin/env python3
"""Threshold sweep for the scalar source_frontier substitution.

The v6 drift analysis used a fixed threshold 0.002. Sweep thresholds
to find a value that minimises agreement loss with the exact discrete
gate. Provides the per-artifact best threshold and its drift, so the
paper can report the best achievable approximation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and x == x and abs(x) != float("inf")


def replay_exact(aligned, accessible_end, rewind_threshold):
    last_aligned = None
    for i, current in enumerate(aligned):
        if current is not None and last_aligned is not None:
            if last_aligned - current > rewind_threshold:
                return i, "rewind"
        if current is not None and current >= max(0, int(accessible_end)):
            return i, "source_frontier"
        if current is not None:
            last_aligned = current
    return len(aligned), "stop"


def replay_scalar(aligned, accessible_end, rewind_threshold, provenance, thr):
    last_aligned = None
    for i, current in enumerate(aligned):
        if current is not None and last_aligned is not None:
            if last_aligned - current > rewind_threshold:
                return i, "rewind"
        pv = provenance[i] if i < len(provenance) else None
        if pv is not None:
            src_inacc = float(pv.get("source_inaccessible") or 0.0)
            if src_inacc >= thr:
                return i, "source_frontier_scalar"
        if current is not None:
            last_aligned = current
    return len(aligned), "stop"


def sweep(input_path, rewind_threshold, thresholds):
    updates_data = []
    with input_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            update = json.loads(line)
            md = update.get("alignatt_metadata") or {}
            aligned = md.get("aligned_source_local_positions") or []
            accessible = md.get("accessible_source_local_end_exclusive")
            provenance = md.get("provenance_per_draft_token") or []
            if not aligned or accessible is None:
                continue
            if provenance and not all(
                all(_is_finite(p.get(k))
                    for k in ("source_accessible", "source_inaccessible",
                              "non_source_prompt", "suffix"))
                for p in provenance):
                continue
            updates_data.append((aligned, int(accessible), provenance))

    rows = []
    rows.append(("threshold", "agree%", "agree_n", "total", "token_delta%",
                 "abs_token_drift"))
    for thr in thresholds:
        agree = 0
        total_exact = 0
        total_scalar = 0
        total_drift = 0
        for aligned, accessible, provenance in updates_data:
            ei, _ = replay_exact(aligned, accessible, rewind_threshold)
            si, _ = replay_scalar(aligned, accessible, rewind_threshold,
                                  provenance, thr)
            if ei == si:
                agree += 1
            total_drift += abs(ei - si)
            total_exact += ei
            total_scalar += si
        n = len(updates_data)
        if total_exact == 0:
            token_delta_pct = 0.0
        else:
            token_delta_pct = 100 * (total_scalar - total_exact) / total_exact
        rows.append((
            f"{thr:.4f}",
            f"{100*agree/max(1,n):.1f}",
            str(agree),
            str(n),
            f"{token_delta_pct:+.2f}",
            str(total_drift),
        ))
    return rows


def analyse(input_path, rewind_threshold, thresholds):
    rows = sweep(input_path, rewind_threshold, thresholds)
    lines = [f"# Threshold sweep on {input_path}",
             f"# rewind_threshold = {rewind_threshold}",
             ""]
    widths = [max(len(r[i]) for r in rows) for i in range(6)]
    for row in rows:
        lines.append("  " + "  ".join(
            cell.rjust(widths[i]) for i, cell in enumerate(row)
        ))
    report = "\n".join(lines)
    print(report)
    out = input_path.parent / "scalar_threshold_sweep.txt"
    out.write_text(report + "\n")
    return report


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--rewind-threshold", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    analyse(
        Path(args.input),
        int(args.rewind_threshold),
        thresholds=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
    )


if __name__ == "__main__":
    main()
