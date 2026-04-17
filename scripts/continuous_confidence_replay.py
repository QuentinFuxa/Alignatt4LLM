#!/usr/bin/env python3
"""Offline replay analysis: derive a continuous confidence scalar from the
MT observer's per-token 4-way provenance, and compare it against the
discrete stop-reason gates already emitted by the online runtime.

Motivation (PLAN.md Step 7, continuous-confidence branch):

    The current MT policy ships three discrete gates
    (`alignatt:source_frontier`, `alignatt:rewind`, `alignatt:provenance_weak`)
    and three loosely-related knobs
    (`translation_alignatt_rewind_threshold`,
     `translation_alignatt_min_source_mass`,
     `translation_alignatt_filter_width`).
    Whether these could collapse into a single confidence scalar that
    monotonically tracks "is this token safe to commit?" is the open
    paper-side question. Answering it on live artefacts without a
    second GPU run is what this replay provides.

Run::

    PYTHONPATH=. .venv-inference/bin/python \
      scripts/continuous_confidence_replay.py \
      --input outputs/night1_ende_stable_k3_chunk700/stream_updates.jsonl

Outputs a summary of per-update confidence statistics plus a CSV of the
per-token provenance / confidence rows to the artifact directory
(``confidence_replay.csv``).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def _is_finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and x == x and abs(x) != float("inf")


def _finite_provenance(pv: dict) -> bool:
    for k in ("source_accessible", "source_inaccessible", "non_source_prompt", "suffix"):
        if not _is_finite(pv.get(k)):
            return False
    return True


def _safe_log(p: float) -> float:
    return math.log(p) if p > 0.0 else 0.0


def provenance_entropy(pv: dict) -> float:
    """Shannon entropy in nats over the 4-way provenance distribution.

    Low entropy = attention mass concentrated on one partition; high
    entropy = diffuse attention across all four. In the absence of a
    trained classifier, entropy alone is a principled scalar derived
    from already-captured quantities.
    """
    total = 0.0
    components = [
        float(pv.get("source_accessible", 0.0)),
        float(pv.get("source_inaccessible", 0.0)),
        float(pv.get("non_source_prompt", 0.0)),
        float(pv.get("suffix", 0.0)),
    ]
    s = sum(components)
    if s <= 0.0:
        return 0.0
    components = [c / s for c in components]
    for p in components:
        total -= p * _safe_log(p)
    return total


def per_token_confidence(pv: dict) -> float:
    """Single per-token confidence scalar.

    Combines three quantities that already live in the observer:
      - source_accessible_mass: fraction of attention on safe source
      - 1 - source_inaccessible_mass: penalty for attention spillover
      - -entropy: concentration bonus (negative Shannon entropy)

    Higher = the draft token's attention is concentrated on safe,
    accessible source positions. Lower = attention drifted to
    inaccessible source, the non-source prompt, or became diffuse.

    The scalar is unit-free in [0, 1] under a simple logistic-style
    squash; the squash is a presentational convenience, the underlying
    quantity is a weighted linear combination.
    """
    src_acc = float(pv.get("source_accessible", 0.0))
    src_inacc = float(pv.get("source_inaccessible", 0.0))
    entropy = provenance_entropy(pv)
    # Normalise entropy to [0, 1] against log(4), the uniform-over-4
    # upper bound (four provenance bins).
    entropy_norm = entropy / math.log(4.0)
    raw = 0.5 * src_acc - 0.2 * src_inacc - 0.3 * entropy_norm + 0.3
    # Squash into [0, 1] for plotting convenience; monotone in raw.
    return 1.0 / (1.0 + math.exp(-6.0 * (raw - 0.5)))


def summarise(rows: list[dict], label: str) -> str:
    if not rows:
        return f"{label}: <empty>"
    vals = [r["confidence"] for r in rows]
    n = len(vals)
    mean = sum(vals) / n
    sorted_vals = sorted(vals)
    p50 = sorted_vals[n // 2]
    p05 = sorted_vals[max(0, n * 5 // 100)]
    p95 = sorted_vals[min(n - 1, n * 95 // 100)]
    return (
        f"{label} (n={n}): mean={mean:.3f} p5={p05:.3f} "
        f"p50={p50:.3f} p95={p95:.3f}"
    )


def process_stream(updates_path: Path, out_csv: Path) -> dict:
    per_token_rows: list[dict] = []
    per_update_rows: list[dict] = []
    stop_reason_counts: Counter[str] = Counter()
    with updates_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            update = json.loads(line)
            md = update.get("alignatt_metadata") or {}
            if not md:
                continue
            provenance_tokens = md.get("provenance_per_draft_token") or []
            if not provenance_tokens:
                continue
            # Some observer captures emit NaN provenance (e.g., when the
            # source slice is empty during a rewind). Drop those updates
            # rather than let NaNs poison the scalar.
            if not all(_finite_provenance(pv) for pv in provenance_tokens):
                continue

            stop_reason = md.get("stop_reason") or "unknown"
            stop_reason_counts[stop_reason] += 1

            unsafe_idx = md.get("unsafe_target_token_index")
            accepted = int(md.get("accepted_token_count") or 0)

            per_update_conf: list[float] = []
            for token_idx, pv in enumerate(provenance_tokens):
                conf = per_token_confidence(pv)
                per_update_conf.append(conf)
                is_accepted = token_idx < accepted
                is_unsafe = (
                    unsafe_idx is not None and token_idx == int(unsafe_idx)
                )
                per_token_rows.append({
                    "update_idx": update["update_idx"],
                    "token_idx": token_idx,
                    "accepted": is_accepted,
                    "is_unsafe": is_unsafe,
                    "stop_reason": stop_reason,
                    "source_accessible": pv.get("source_accessible"),
                    "source_inaccessible": pv.get("source_inaccessible"),
                    "non_source_prompt": pv.get("non_source_prompt"),
                    "suffix": pv.get("suffix"),
                    "entropy_nats": provenance_entropy(pv),
                    "confidence": conf,
                })
            if per_update_conf:
                mean_conf = sum(per_update_conf) / len(per_update_conf)
                per_update_rows.append({
                    "update_idx": update["update_idx"],
                    "stop_reason": stop_reason,
                    "n_tokens": len(per_update_conf),
                    "mean_confidence": mean_conf,
                    "min_confidence": min(per_update_conf),
                    "first_token_confidence": per_update_conf[0],
                    "unsafe_confidence": (
                        per_update_conf[int(unsafe_idx)]
                        if unsafe_idx is not None
                        and 0 <= int(unsafe_idx) < len(per_update_conf)
                        else None
                    ),
                })

    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "update_idx",
                "token_idx",
                "accepted",
                "is_unsafe",
                "stop_reason",
                "source_accessible",
                "source_inaccessible",
                "non_source_prompt",
                "suffix",
                "entropy_nats",
                "confidence",
            ],
        )
        writer.writeheader()
        for row in per_token_rows:
            writer.writerow(row)

    # Summaries
    accepted_rows = [r for r in per_token_rows if r["accepted"]]
    rejected_rows = [r for r in per_token_rows if not r["accepted"]]
    unsafe_rows = [r for r in per_token_rows if r["is_unsafe"]]

    by_stop: dict[str, list[dict]] = defaultdict(list)
    for row in per_token_rows:
        by_stop[row["stop_reason"]].append(row)

    lines = []
    lines.append(f"# Continuous-confidence replay on {updates_path}")
    lines.append(f"# Artifact dir: {updates_path.parent}")
    lines.append(f"# Total updates with provenance: {len(per_update_rows)}")
    lines.append(f"# Total draft tokens: {len(per_token_rows)}")
    lines.append(f"# Stop-reason counts: {dict(stop_reason_counts)}")
    lines.append("")
    lines.append(summarise(accepted_rows, "accepted   "))
    lines.append(summarise(rejected_rows, "rejected   "))
    lines.append(summarise(unsafe_rows, "unsafe-flag"))
    lines.append("")
    lines.append("# Per-stop-reason confidence statistics (aggregated over all tokens in those updates):")
    for stop_reason in sorted(by_stop.keys()):
        lines.append(summarise(by_stop[stop_reason], f"{stop_reason:<32}"))
    lines.append("")
    if accepted_rows and rejected_rows:
        mean_accepted = sum(r["confidence"] for r in accepted_rows) / len(accepted_rows)
        mean_rejected = sum(r["confidence"] for r in rejected_rows) / len(rejected_rows)
        delta = mean_accepted - mean_rejected
        lines.append(
            f"# Mean confidence delta (accepted - rejected): {delta:+.3f}"
        )
        # Threshold sweep — how well does a single confidence threshold
        # replicate the discrete accept/reject decision?
        best_thresh = None
        best_f1 = -1.0
        for thr in [i / 100.0 for i in range(5, 96, 5)]:
            tp = sum(1 for r in accepted_rows if r["confidence"] >= thr)
            fp = sum(1 for r in rejected_rows if r["confidence"] >= thr)
            fn = len(accepted_rows) - tp
            tn = len(rejected_rows) - fp
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thr
                best_stats = (prec, rec, tp, fp, fn, tn)
        prec, rec, tp, fp, fn, tn = best_stats
        lines.append(
            f"# Best single-threshold replication of accept/reject:"
            f" thr={best_thresh:.2f}  precision={prec:.3f} recall={rec:.3f}"
            f"  F1={best_f1:.3f}  (tp={tp} fp={fp} fn={fn} tn={tn})"
        )
    report = "\n".join(lines)
    print(report)
    (updates_path.parent / "confidence_replay_report.txt").write_text(report + "\n")
    return {"csv": out_csv, "report": report}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True,
        help="stream_updates.jsonl produced by run_simulstream_batch at a "
             "SHA that emits observer metadata (commit a0edcc6 or later).",
    )
    parser.add_argument(
        "--out-csv", default=None,
        help="CSV output path. Defaults to <artifact-dir>/confidence_replay.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    updates_path = Path(args.input)
    out_csv = Path(args.out_csv) if args.out_csv else (
        updates_path.parent / "confidence_replay.csv"
    )
    process_stream(updates_path, out_csv)


if __name__ == "__main__":
    main()
