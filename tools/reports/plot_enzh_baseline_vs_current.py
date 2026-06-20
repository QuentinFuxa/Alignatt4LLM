#!/usr/bin/env python3
"""Focused EN->ZH comparison: public no-context baseline vs current full21 points.

Reads the regenerated tradeoff diagnostics JSON (single source of truth for
baseline anchors and recovered full21 points) and renders one readable chart:
the public no-context curve, our clean points (with their Pareto front), and
the guarded points marked as a separate non-claimable class.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRADEOFF_JSON = (
    REPO_ROOT / "outputs" / "plots" / "enzh_quality_latency_tradeoff_diagnostics_20260607.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "plots"

# Display names only; scores/latencies always come from the tradeoff JSON.
PRETTY_LABELS = {
    "enzh_clean_token_sftrim_apm001_eager_full21_20260609": "sftrim+apm @640",
    "enzh_clean_sb005cap1_sb05_eager_full21_20260610": "source-bearing @640",
    "gemma_zh_clean_eager_chunk960_full21_20260610": "θ0 @960",
    "gemma_zh_clean_eager_chunk1280_full21_20260610": "θ0 @1280",
    "gemma_zh_unitconf_top16_final_full21_20260611": "unit_conf 10/16 @1280",
    "gemma_zh_unitconf_top16_conf06875_full21_20260611": "unit_conf 11/16 @1280",
    "gemma_zh_unitconf_chunk1920_conf04375_full21_20260611": "unit_conf 7/16 @1920",
    "enzh_milmmt_chunk640_asrmin6_eager_full21_20260610": "asrmin6 eager @640",
    "enzh_milmmt_chunk640_maxreg1_recent1_full21_20260606": "maxreg1 @640",
    "enzh_milmmt_chunk640_asrmin6_full21_20260606": "asrmin6 @640 (old stack)",
}

LABEL_OFFSETS = {
    "sftrim+apm @640": (8, -3),
    "source-bearing @640": (8, -3),
    "θ0 @960": (8, -3),
    "θ0 @1280": (10, 2),
    "unit_conf 10/16 @1280": (-30, -18),
    "unit_conf 11/16 @1280": (-150, -30),
    "unit_conf 7/16 @1920": (8, -12),
    "asrmin6 eager @640": (-40, 9),
    "maxreg1 @640": (8, -10),
    "asrmin6 @640 (old stack)": (-120, 24),
}

# Dense regions: draw a thin leader line from label to marker.
ARROW_LABELS = {"unit_conf 11/16 @1280", "asrmin6 @640 (old stack)"}


def pareto_front(points: list[dict]) -> list[dict]:
    """Points with strictly increasing quality when sorted by latency."""
    front: list[dict] = []
    for point in sorted(points, key=lambda p: p["longyaal_cu_ms"]):
        if not front or point["xcometxl"] > front[-1]["xcometxl"]:
            front.append(point)
    return front


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tradeoff-json", type=Path, default=DEFAULT_TRADEOFF_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default="enzh_baseline_vs_current_20260612")
    args = parser.parse_args()

    data = json.loads(args.tradeoff_json.read_text())
    baseline = data["baseline_points"]["baseline"]
    ours = data["our_points"]
    clean = [p for p in ours if not p.get("alignatt_guard_flags")]
    guarded = [p for p in ours if p.get("alignatt_guard_flags")]
    clean_front = pareto_front(clean)

    fig, ax = plt.subplots(figsize=(12.5, 7.5))

    bx = [p["longyaal_cu_ms"] for p in baseline]
    by = [p["xcometxl"] for p in baseline]
    ax.plot(bx, by, "-o", color="#1f77b4", linewidth=2.2, markersize=9, zorder=4,
            label="Public baseline, no context (LA chunks 640→1920)")
    for p in baseline:
        ax.annotate(f"{p['segment_ms']}", (p["longyaal_cu_ms"], p["xcometxl"]),
                    textcoords="offset points", xytext=(0, 10), ha="center",
                    fontsize=9, color="#1f77b4", fontweight="bold")

    fx = [p["longyaal_cu_ms"] for p in clean_front]
    fy = [p["xcometxl"] for p in clean_front]
    ax.plot(fx, fy, "--", color="#2ca02c", linewidth=1.6, zorder=3, alpha=0.8,
            label="Ours — clean Pareto front (full21)")
    gemma_clean = [p for p in clean if p["label"].startswith("gemma_")]
    milmmt_clean = [p for p in clean if not p["label"].startswith("gemma_")]
    ax.scatter([p["longyaal_cu_ms"] for p in gemma_clean],
               [p["xcometxl"] for p in gemma_clean],
               marker="D", s=90, color="#2ca02c", zorder=5,
               label="Clean — Gemma AlignAtt (claimable)")
    ax.scatter([p["longyaal_cu_ms"] for p in milmmt_clean],
               [p["xcometxl"] for p in milmmt_clean],
               marker="^", s=70, color="#17becf", zorder=5,
               label="Clean — MiLMMT variants")
    ax.scatter([p["longyaal_cu_ms"] for p in guarded],
               [p["xcometxl"] for p in guarded],
               marker="s", s=80, facecolors="none", edgecolors="#ff7f0e",
               linewidths=2.0, zorder=5, label="Guarded — MiLMMT (not claimable)")

    for p in clean + guarded:
        name = PRETTY_LABELS.get(p["label"], p["label"])
        dx, dy = LABEL_OFFSETS.get(name, (8, -3))
        arrow = (dict(arrowstyle="-", color="#999999", lw=0.8)
                 if name in ARROW_LABELS else None)
        ax.annotate(name, (p["longyaal_cu_ms"], p["xcometxl"]),
                    textcoords="offset points", xytext=(dx, dy), fontsize=8,
                    color="#444444", arrowprops=arrow)

    # Per-anchor gap: best full21 point (any class) at or below the anchor CU.
    print(f"{'anchor':>6}  {'baseline':>14}  {'best ≤ CU':>14}  {'ΔX':>6}  class")
    for anchor in baseline:
        eligible = [p for p in ours if p["longyaal_cu_ms"] <= anchor["longyaal_cu_ms"]]
        best = max(eligible, key=lambda p: p["xcometxl"])
        delta = best["xcometxl"] - anchor["xcometxl"]
        is_clean = not best.get("alignatt_guard_flags")
        klass = "clean" if is_clean else "guarded"
        ax.annotate(f"{delta:+.2f}", (anchor["longyaal_cu_ms"], anchor["xcometxl"]),
                    textcoords="offset points", xytext=(0, 22), ha="center",
                    fontsize=10, color="#d62728", fontweight="bold")
        print(f"{anchor['segment_ms']:>6}  "
              f"{anchor['xcometxl']:>6.2f} @ {anchor['longyaal_cu_ms']:>5.0f}  "
              f"{best['xcometxl']:>6.2f} @ {best['longyaal_cu_ms']:>5.0f}  "
              f"{delta:>+6.2f}  {klass}: {PRETTY_LABELS.get(best['label'], best['label'])}")

    ax.set_xlabel("LongYAAL CU (ms)", fontsize=12)
    ax.set_ylabel("XCOMET-XL × 100", fontsize=12)
    ax.set_title("EN→ZH full21 (21 MCIF dev audios), no context — "
                 "public baseline vs ours, 2026-06-12", fontsize=13)
    ax.grid(alpha=0.3)
    ax.set_xlim(700, 5700)
    ax.set_ylim(58.5, 83.0)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.output_dir / f"{args.output_stem}.png"
    fig.savefig(out_png, dpi=150)
    print(out_png)


if __name__ == "__main__":
    main()
