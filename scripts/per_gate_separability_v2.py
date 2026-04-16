#!/usr/bin/env python3
"""Per-gate separability with positional + monotonicity features.

Reads a stream_updates.jsonl directly (rather than the per-token CSV)
so it has access to ``aligned_source_local_positions`` and the
per-update integer indices that don't survive the CSV flattening.

Motivation: the v1 analysis found that
``alignatt:source_frontier`` collapses cleanly under a
``unsafe_token.source_inaccessible`` threshold but
``alignatt:rewind`` caps at F1 ≈ 0.72-0.75 under provenance-only
features. Rewind is defined by a backward jump in
``aligned_source_local_positions``, so this script adds:

  - ``position_drift``       last - first aligned source position
  - ``max_backward_jump``    max(aligned[i] - aligned[i+1]) (0 if monotone)
  - ``n_backward_pairs``     count of adjacent non-monotone pairs
  - ``monotonicity_ratio``   fraction of adjacent pairs that are non-decreasing
  - ``unsafe_idx_ratio``     unsafe_target_token_index / n_tokens
  - ``accepted_ratio``       accepted_token_count / n_tokens
  - ``accessibility_ratio``  accessible_source_unit_count / source_unit_count

Each run searches (feature x threshold x direction) for the best
single-scalar predictor of each gate at the update level.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _finite(x):
    return x is not None and isinstance(x, (int, float)) and x == x and abs(x) != float("inf")


def monotonicity_features(aligned: list) -> dict:
    """Summary features of aligned_source_local_positions.

    ``max_backward_jump`` is the max drop across *adjacent* pairs,
    which is NOT what the online rewind gate checks. The online gate
    checks ``last_non_none_aligned - current > rewind_threshold`` —
    a drop vs. the *most recent non-None* aligned position, not just
    the immediate neighbour. ``max_drop_vs_prev_non_none`` below
    reproduces the quantity the gate actually uses.
    """
    if not aligned or len(aligned) < 2:
        return {
            "position_drift": 0.0,
            "max_backward_jump": 0.0,
            "n_backward_pairs": 0.0,
            "monotonicity_ratio": 1.0,
            "align_first": float(aligned[0]) if aligned else 0.0,
            "align_last": float(aligned[-1]) if aligned else 0.0,
            "max_drop_vs_prev_non_none": 0.0,
        }
    max_back = 0
    n_back = 0
    for a, b in zip(aligned, aligned[1:]):
        if a is None or b is None:
            continue
        if a > b:
            n_back += 1
            max_back = max(max_back, int(a) - int(b))

    max_drop_prev = 0
    prev_non_none = None
    for current in aligned:
        if current is None:
            continue
        if prev_non_none is not None and prev_non_none > current:
            drop = prev_non_none - current
            if drop > max_drop_prev:
                max_drop_prev = drop
        prev_non_none = current

    first = next((x for x in aligned if x is not None), 0)
    last = next((x for x in reversed(aligned) if x is not None), 0)
    return {
        "position_drift": float(last - first),
        "max_backward_jump": float(max_back),
        "n_backward_pairs": float(n_back),
        "monotonicity_ratio": 1.0 - n_back / max(1, len(aligned) - 1),
        "align_first": float(first),
        "align_last": float(last),
        "max_drop_vs_prev_non_none": float(max_drop_prev),
    }


def build_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            update = json.loads(line)
            md = update.get("alignatt_metadata") or {}
            if not md:
                continue
            provenance = md.get("provenance_per_draft_token") or []
            aligned = md.get("aligned_source_local_positions") or []
            if not provenance:
                continue
            stop_reason = md.get("stop_reason") or "unknown"
            source_unit_count = md.get("source_unit_count")
            accessible_count = md.get("accessible_source_unit_count")
            accepted = md.get("accepted_token_count") or 0
            n_tokens = len(provenance)
            unsafe_idx = md.get("unsafe_target_token_index")

            record: dict = {
                "update_idx": update["update_idx"],
                "stop_reason": stop_reason,
                "n_tokens": float(n_tokens),
                "accepted_token_count": float(accepted),
                "accepted_ratio": (
                    float(accepted) / n_tokens if n_tokens > 0 else 0.0
                ),
                "accessibility_ratio": (
                    float(accessible_count) / source_unit_count
                    if source_unit_count
                    else 0.0
                ),
                "unsafe_idx_ratio": (
                    float(unsafe_idx) / n_tokens
                    if (unsafe_idx is not None and n_tokens > 0)
                    else -1.0
                ),
                "source_token_count": float(md.get("source_token_count") or 0),
                "source_unit_count": float(md.get("source_unit_count") or 0),
            }
            record.update(monotonicity_features(aligned))

            # Provenance-based features at the unsafe token and averaged
            # across the accepted window. Mirror the v1 feature set so
            # the new positional features can be compared on the same
            # footing.
            def _all_finite(pv_list: list[dict]) -> bool:
                for pv in pv_list:
                    for k in ("source_accessible", "source_inaccessible",
                              "non_source_prompt", "suffix"):
                        if not _finite(pv.get(k)):
                            return False
                return True

            if not _all_finite(provenance):
                continue

            unsafe_pv = None
            if unsafe_idx is not None and 0 <= int(unsafe_idx) < len(provenance):
                unsafe_pv = provenance[int(unsafe_idx)]
            for k in ("source_accessible", "source_inaccessible",
                      "non_source_prompt", "suffix"):
                record[f"unsafe__{k}"] = (
                    float(unsafe_pv.get(k, 0.0)) if unsafe_pv else 0.0
                )

            acc_window = provenance[:int(accepted)] if accepted else []
            for k in ("source_accessible", "source_inaccessible",
                      "non_source_prompt", "suffix"):
                if acc_window:
                    vals = [float(p.get(k, 0.0)) for p in acc_window]
                    record[f"accepted_mean__{k}"] = sum(vals) / len(vals)
                    record[f"accepted_min__{k}"] = min(vals)
                else:
                    record[f"accepted_mean__{k}"] = 0.0
                    record[f"accepted_min__{k}"] = 0.0

            records.append(record)
    return records


def best_threshold(pos, neg, direction: str):
    vals = sorted(set(v for v in pos + neg if v is not None))
    if not vals:
        return (float("nan"), 0.0, 0.0, 0.0)
    best = (float("nan"), 0.0, 0.0, 0.0)
    for thr in vals:
        if direction == "greater":
            tp = sum(1 for v in pos if v is not None and v >= thr)
            fp = sum(1 for v in neg if v is not None and v >= thr)
            fn = sum(1 for v in pos if v is not None and v < thr)
        else:
            tp = sum(1 for v in pos if v is not None and v <= thr)
            fp = sum(1 for v in neg if v is not None and v <= thr)
            fn = sum(1 for v in pos if v is not None and v > thr)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best[3]:
            best = (thr, prec, rec, f1)
    return best


FEATURES = [
    "max_drop_vs_prev_non_none",
    "max_backward_jump",
    "n_backward_pairs",
    "monotonicity_ratio",
    "position_drift",
    "align_first",
    "align_last",
    "unsafe_idx_ratio",
    "accepted_ratio",
    "accessibility_ratio",
    "n_tokens",
    "source_unit_count",
    "unsafe__source_accessible",
    "unsafe__source_inaccessible",
    "unsafe__non_source_prompt",
    "unsafe__suffix",
    "accepted_mean__source_accessible",
    "accepted_mean__source_inaccessible",
    "accepted_min__source_accessible",
]


def analyse(path: Path) -> str:
    records = build_records(path)
    if not records:
        return f"# {path}: empty"
    stop_reasons = sorted({r["stop_reason"] for r in records})
    lines = []
    lines.append(f"# Per-gate separability v2 on {path}")
    lines.append(f"# {len(records)} updates, stop_reasons={stop_reasons}")
    lines.append("")

    for gate in stop_reasons:
        pos = [r for r in records if r["stop_reason"] == gate]
        neg = [r for r in records if r["stop_reason"] != gate]
        if not pos or not neg:
            continue
        lines.append(f"## gate={gate}  (n_pos={len(pos)}, n_neg={len(neg)})")
        rows = []
        for feat in FEATURES:
            pv = [r[feat] for r in pos if r.get(feat) is not None]
            nv = [r[feat] for r in neg if r.get(feat) is not None]
            if not pv or not nv:
                continue
            g = best_threshold(pv, nv, "greater")
            l = best_threshold(pv, nv, "less")
            if g[3] >= l[3]:
                dir_, thr, prec, rec, f1 = "≥", *g
            else:
                dir_, thr, prec, rec, f1 = "≤", *l
            rows.append({
                "feature": feat,
                "mean_pos": statistics.mean(pv),
                "mean_neg": statistics.mean(nv),
                "dir": dir_,
                "thr": thr,
                "prec": prec,
                "rec": rec,
                "f1": f1,
            })
        rows.sort(key=lambda r: -r["f1"])
        lines.append(
            f"{'feature':<40} {'mean_pos':>10} {'mean_neg':>10} "
            f"{'dir':>3} {'thr':>10} {'prec':>6} {'rec':>6} {'F1':>6}"
        )
        for r in rows[:8]:  # top 8
            lines.append(
                f"{r['feature']:<40} {r['mean_pos']:>10.3f} {r['mean_neg']:>10.3f} "
                f"{r['dir']:>3} {r['thr']:>10.3f} "
                f"{r['prec']:>6.3f} {r['rec']:>6.3f} {r['f1']:>6.3f}"
            )
        lines.append("")
    report = "\n".join(lines)
    print(report)
    out = path.parent / "per_gate_separability_v2.txt"
    out.write_text(report + "\n")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True,
                   help="stream_updates.jsonl with observer metadata")
    return p.parse_args()


def main():
    args = parse_args()
    analyse(Path(args.input))


if __name__ == "__main__":
    main()
