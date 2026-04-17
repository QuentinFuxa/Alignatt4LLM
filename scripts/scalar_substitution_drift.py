#!/usr/bin/env python3
"""Offline "what-if" analysis: how many commit decisions would change
if the ``source_frontier`` discrete gate were replaced by a scalar
threshold on ``unsafe.source_inaccessible``?

The per-gate separability analyses found ``source_frontier`` F1 0.91-
0.99 with ``unsafe.source_inaccessible ≥ 0.002`` as a single-feature
classifier. That's a gate-level F1; the paper-grade question is
whether the downstream commit decisions would change at the
update or token level if we swapped the online gate.

This script replays the MT policy loop twice on each update:

  - Exact: the current online policy
    (``current >= accessible_end`` for source_frontier,
     ``last_aligned - current > rewind_threshold`` for rewind)
  - Scalar: source_frontier is replaced by
    ``provenance[i].source_inaccessible >= scalar_threshold``
    (rewind left unchanged; it's not the scalar-reducible gate)

For each update we record which token each loop would have broken at
(or "stop" if the loop ran to completion). Then we count:

  - Updates where the two loops produce the SAME accepted token count
  - Updates where they differ, by how many tokens
  - Tokens that would have been committed under discrete but not scalar
  - Tokens that would have been committed under scalar but not discrete

This is the clean paper number for "how invasive would the scalar
substitution be in practice": if drift is < a few percent of tokens,
the substitution is a defensible drop-in for this gate on this path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def replay_exact(aligned, accessible_end, rewind_threshold):
    """Mirror cascade_mt_backend.should_stop_in_loop exactly."""
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


def replay_scalar_source_frontier(aligned, accessible_end, rewind_threshold,
                                  provenance, scalar_threshold):
    """Rewind unchanged; source_frontier replaced by scalar threshold."""
    last_aligned = None
    for i, current in enumerate(aligned):
        if current is not None and last_aligned is not None:
            if last_aligned - current > rewind_threshold:
                return i, "rewind"
        pv = provenance[i] if i < len(provenance) else None
        if pv is not None:
            src_inacc = float(pv.get("source_inaccessible") or 0.0)
            if src_inacc >= scalar_threshold:
                return i, "source_frontier_scalar"
        if current is not None:
            last_aligned = current
    return len(aligned), "stop"


def analyse(input_path: Path, rewind_threshold: int, scalar_threshold: float):
    lines = []
    lines.append(f"# Scalar-substitution drift analysis on {input_path}")
    lines.append(f"# rewind_threshold = {rewind_threshold}")
    lines.append(f"# scalar_threshold (source_inaccessible) = {scalar_threshold}")

    total_updates = 0
    agree_updates = 0
    disagree_updates = 0
    exact_accept_total = 0
    scalar_accept_total = 0
    drift_total_tokens = 0
    by_reason = {}

    skipped_missing_data = 0

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
                    all(p.get(k) is not None
                        and isinstance(p.get(k), (int, float))
                        and p.get(k) == p.get(k)
                        for k in ("source_accessible", "source_inaccessible",
                                  "non_source_prompt", "suffix"))
                    for p in provenance):
                skipped_missing_data += 1
                continue

            total_updates += 1
            exact_idx, exact_reason = replay_exact(
                aligned, int(accessible), rewind_threshold,
            )
            scalar_idx, scalar_reason = replay_scalar_source_frontier(
                aligned, int(accessible), rewind_threshold,
                provenance, scalar_threshold,
            )

            exact_accept = exact_idx
            scalar_accept = scalar_idx
            exact_accept_total += exact_accept
            scalar_accept_total += scalar_accept

            pair = (exact_reason, scalar_reason)
            by_reason[pair] = by_reason.get(pair, 0) + 1

            if exact_accept == scalar_accept:
                agree_updates += 1
            else:
                disagree_updates += 1
                drift_total_tokens += abs(exact_accept - scalar_accept)

    lines.append(f"# total_updates = {total_updates}")
    lines.append(f"# skipped (NaN / missing provenance) = {skipped_missing_data}")
    lines.append("")
    lines.append(
        f"# Updates where exact and scalar agree on accepted-token count:"
        f" {agree_updates}/{total_updates} "
        f"({100 * agree_updates / total_updates:.1f}%)"
    )
    lines.append(
        f"# Updates where they disagree:"
        f" {disagree_updates}/{total_updates} "
        f"({100 * disagree_updates / total_updates:.1f}%)"
    )
    lines.append(
        f"# Exact accepted tokens (sum over updates): {exact_accept_total}"
    )
    lines.append(
        f"# Scalar accepted tokens (sum over updates): {scalar_accept_total}"
    )
    delta = scalar_accept_total - exact_accept_total
    lines.append(
        f"# Aggregate drift (scalar - exact): {delta:+d} "
        f"({100 * delta / max(1, exact_accept_total):+.2f}% vs exact)"
    )
    lines.append(
        f"# Total absolute per-update drift: {drift_total_tokens} tokens"
    )
    lines.append("")
    lines.append("# Stop-reason pairs (exact → scalar) with counts:")
    for pair in sorted(by_reason.keys()):
        lines.append(f"  {str(pair):<60} {by_reason[pair]}")

    report = "\n".join(lines)
    print(report)
    out = input_path.parent / "scalar_substitution_drift.txt"
    out.write_text(report + "\n")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--rewind-threshold", type=int, default=8)
    p.add_argument("--scalar-threshold", type=float, default=0.002)
    return p.parse_args()


def main():
    args = parse_args()
    analyse(Path(args.input), int(args.rewind_threshold),
            float(args.scalar_threshold))


if __name__ == "__main__":
    main()
