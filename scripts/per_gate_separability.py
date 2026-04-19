#!/usr/bin/env python3
"""Per-gate separability analysis for the MT observer's discrete gates.

Given a ``confidence_replay.csv`` produced by
``continuous_confidence_replay.py``, find — for each discrete stop
reason — the best single-scalar predictor (feature + threshold) that
distinguishes updates-triggering-that-gate from updates-not-triggering
it. Operates at the **update** level (one decision per update), not
at the per-token level, because the discrete gate fires once per
update.

This sharpens the earlier aggregate-F1 result. The aggregate analysis
pools all accept/reject tokens together, which dilutes per-gate
signal. A gate-by-gate search asks the right question: "for a
continuous scalar to replace gate X, how separable is gate-X-triggered
from the rest under that scalar?"

Paper motivation: if any one gate is cleanly separable under a simple
provenance-derived scalar, that gate is a natural candidate to
collapse into the continuous confidence mechanism first. Gates that
are not separable under such a scalar need richer features
(positional, learned, per-head).

Usage::

    PYTHONPATH=. .venv-inference/bin/python \
      scripts/per_gate_separability.py \
      --csv outputs/night1_ende_stable_k3_chunk700/confidence_replay.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


FLOAT_COLUMNS = (
    "source_accessible",
    "source_inaccessible",
    "non_source_prompt",
    "suffix",
    "entropy_nats",
    "confidence",
)


def _parse_float(value: str):
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except ValueError:
        return None
    if x != x or abs(x) == float("inf"):
        return None
    return x


def load_rows(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            parsed = dict(row)
            for k in FLOAT_COLUMNS:
                parsed[k] = _parse_float(parsed.get(k))
            parsed["token_idx"] = int(row["token_idx"])
            parsed["update_idx"] = int(row["update_idx"])
            parsed["accepted"] = row["accepted"] == "True"
            parsed["is_unsafe"] = row["is_unsafe"] == "True"
            rows.append(parsed)
    return rows


def aggregate_update_features(rows: list[dict]) -> list[dict]:
    """Collapse per-token rows into one record per update."""
    by_update: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_update[row["update_idx"]].append(row)
    aggregated: list[dict] = []
    for update_idx, token_rows in sorted(by_update.items()):
        accepted = [r for r in token_rows if r["accepted"]]
        unsafe = [r for r in token_rows if r["is_unsafe"]]
        stop_reason = token_rows[0]["stop_reason"]
        record: dict = {
            "update_idx": update_idx,
            "stop_reason": stop_reason,
            "n_tokens": len(token_rows),
            "n_accepted": len(accepted),
        }
        # Per-token feature aggregates — we keep min/mean over the
        # accepted prefix (the window the policy actually commits from)
        # and the single value at the first rejected ("unsafe") token.
        for feat in ("source_accessible", "source_inaccessible",
                     "non_source_prompt", "suffix",
                     "entropy_nats", "confidence"):
            accepted_vals = [r[feat] for r in accepted if r[feat] is not None]
            all_vals = [r[feat] for r in token_rows if r[feat] is not None]
            record[f"accepted_mean__{feat}"] = (
                sum(accepted_vals) / len(accepted_vals) if accepted_vals else None
            )
            record[f"accepted_min__{feat}"] = (
                min(accepted_vals) if accepted_vals else None
            )
            record[f"all_mean__{feat}"] = (
                sum(all_vals) / len(all_vals) if all_vals else None
            )
            unsafe_val = unsafe[0][feat] if unsafe else None
            record[f"unsafe_token__{feat}"] = unsafe_val
        aggregated.append(record)
    return aggregated


def best_threshold(values_pos, values_neg, direction: str) -> tuple[float, float, float, float]:
    """Find the threshold that maximises F1 for separating ``values_pos``
    from ``values_neg`` under a given direction.

    Returns (threshold, precision, recall, f1).
    ``direction='greater'`` classifies as positive when value >= thr;
    ``direction='less'`` classifies as positive when value <= thr.
    """
    values = sorted(set(v for v in values_pos + values_neg if v is not None))
    if not values:
        return (float("nan"), 0.0, 0.0, 0.0)
    best = (float("nan"), 0.0, 0.0, 0.0)
    for thr in values:
        if direction == "greater":
            tp = sum(1 for v in values_pos if v is not None and v >= thr)
            fp = sum(1 for v in values_neg if v is not None and v >= thr)
            fn = sum(1 for v in values_pos if v is not None and v < thr)
        else:
            tp = sum(1 for v in values_pos if v is not None and v <= thr)
            fp = sum(1 for v in values_neg if v is not None and v <= thr)
            fn = sum(1 for v in values_pos if v is not None and v > thr)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best[3]:
            best = (thr, prec, rec, f1)
    return best


def analyse(csv_path: Path) -> str:
    rows = load_rows(csv_path)
    updates = aggregate_update_features(rows)
    if not updates:
        return f"# {csv_path}: empty"

    stop_reasons = sorted({u["stop_reason"] for u in updates})
    lines: list[str] = []
    lines.append(f"# Per-gate separability on {csv_path}")
    lines.append(f"# {len(updates)} updates, {len(rows)} per-token rows")
    lines.append(f"# Stop reasons: {stop_reasons}")
    lines.append("")

    # Feature list: the aggregate stats across the update.
    feature_names = [
        "accepted_mean__confidence",
        "accepted_min__confidence",
        "accepted_mean__source_accessible",
        "accepted_min__source_accessible",
        "accepted_mean__source_inaccessible",
        "accepted_mean__entropy_nats",
        "all_mean__confidence",
        "unsafe_token__confidence",
        "unsafe_token__source_accessible",
        "unsafe_token__source_inaccessible",
        "unsafe_token__entropy_nats",
    ]

    for gate in stop_reasons:
        pos_updates = [u for u in updates if u["stop_reason"] == gate]
        neg_updates = [u for u in updates if u["stop_reason"] != gate]
        if not pos_updates or not neg_updates:
            continue
        lines.append(f"## gate = {gate}  (n_pos={len(pos_updates)}, n_neg={len(neg_updates)})")
        gate_rows = []
        for feat in feature_names:
            pos_vals = [u[feat] for u in pos_updates if u.get(feat) is not None]
            neg_vals = [u[feat] for u in neg_updates if u.get(feat) is not None]
            if not pos_vals or not neg_vals:
                continue
            mean_pos = statistics.mean(pos_vals)
            mean_neg = statistics.mean(neg_vals)
            # Try both directions; pick the one with higher F1.
            g_thr, g_prec, g_rec, g_f1 = best_threshold(pos_vals, neg_vals, "greater")
            l_thr, l_prec, l_rec, l_f1 = best_threshold(pos_vals, neg_vals, "less")
            if g_f1 >= l_f1:
                dir_, thr, prec, rec, f1 = "≥", g_thr, g_prec, g_rec, g_f1
            else:
                dir_, thr, prec, rec, f1 = "≤", l_thr, l_prec, l_rec, l_f1
            gate_rows.append({
                "feature": feat,
                "mean_pos": mean_pos,
                "mean_neg": mean_neg,
                "delta": mean_pos - mean_neg,
                "dir": dir_,
                "thr": thr,
                "prec": prec,
                "rec": rec,
                "f1": f1,
            })
        # Sort by F1 desc
        gate_rows.sort(key=lambda r: -r["f1"])
        lines.append(
            f"{'feature':<40} {'mean_pos':>9} {'mean_neg':>9} {'delta':>8} "
            f"{'dir':>3} {'thr':>8} {'prec':>6} {'rec':>6} {'F1':>6}"
        )
        for r in gate_rows:
            lines.append(
                f"{r['feature']:<40} {r['mean_pos']:>9.3f} {r['mean_neg']:>9.3f} "
                f"{r['delta']:>+8.3f} {r['dir']:>3} {r['thr']:>8.3f} "
                f"{r['prec']:>6.3f} {r['rec']:>6.3f} {r['f1']:>6.3f}"
            )
        lines.append("")
    report = "\n".join(lines)
    print(report)
    out_path = csv_path.parent / "per_gate_separability.txt"
    out_path.write_text(report + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True,
                        help="confidence_replay.csv produced by "
                             "scripts/continuous_confidence_replay.py")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyse(Path(args.csv))


if __name__ == "__main__":
    main()
