#!/usr/bin/env python3
"""Comparison table and decision verdicts for the agreement-policy replays.

Offline diagnostic companion to ``replay_enzh_agreement_policies.py``. Scans
the four full21 base runs and their ``offline_replay_agreement_*`` roots,
joins ``evaluation.json`` contract scores with the replay fidelity summaries,
and applies the pre-registered decision gates:

- Gate 0 (validity): every ``control`` replay reproduced the original
  predictions and delays exactly.
- D1 (theta-0 runs): pure local agreement (la2) is an OPTIMISTIC latency bound
  (prefill bias); if even that bound costs >= +400 ms CU vs the original,
  attention-only dominates output-agreement at this chunk. Quality deltas are
  structurally ~0 (final surfaces coincide) — latency is the measured axis.
- D2 (unit_conf runs): hyb_rec_la2 is promotable to a real GPU probe if it
  saves >= 150 ms CU with chrF within -0.3 of the original.
- D3 (simulator calibration): confsim0p625 replayed from the theta-0
  chunk-1280 run should reproduce >= ~50% of the real CU shift between
  eager1280 and unitconf 10/16 (+2853 ms); otherwise conf_sim/hybrid_conf
  rows are indicative only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS_ROOT = REPO_ROOT / "outputs" / "diagnostics_jarvislab_20260606"
BASE_RUNS = (
    "gemma_zh_clean_eager_chunk960_full21_20260610",
    "gemma_zh_clean_eager_chunk1280_full21_20260610",
    "gemma_zh_unitconf_top16_final_full21_20260611",
    "gemma_zh_unitconf_top16_conf06875_full21_20260611",
)
THETA0_RUNS = BASE_RUNS[:2]
UNITCONF_RUNS = BASE_RUNS[2:]
EAGER_1280 = "gemma_zh_clean_eager_chunk1280_full21_20260610"
UNITCONF_10_16 = "gemma_zh_unitconf_top16_final_full21_20260611"

SCORE_KEYS = {
    "bleu": "BLEU",
    "chrf": "CHRF",
    "cu": "LongYAAL CU",
    "ca": "LongYAAL CA",
}


def contract_scores(run_dir: Path) -> dict[str, float] | None:
    evaluation_path = run_dir / "evaluation.json"
    if not evaluation_path.exists():
        return None
    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    contract = payload.get("contract_scores") or {}
    return {short: float(contract[key]) for short, key in SCORE_KEYS.items()}


def collect_rows(replay_suffix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base_run in BASE_RUNS:
        base_scores = contract_scores(DIAGNOSTICS_ROOT / base_run)
        if base_scores is None:
            continue
        manifest = json.loads(
            (DIAGNOSTICS_ROOT / base_run / "manifest.json").read_text(encoding="utf-8")
        )
        chunk_ms = (manifest.get("runtime_config") or {}).get("chunk_ms")
        rows.append(
            {
                "base_run": base_run,
                "chunk_ms": chunk_ms,
                "policy": "original",
                **base_scores,
                "delta_cu": 0.0,
                "delta_chrf": 0.0,
            }
        )
        replay_root = DIAGNOSTICS_ROOT / f"offline_replay_agreement_{base_run}_{replay_suffix}"
        if not replay_root.is_dir():
            continue
        for policy_dir in sorted(replay_root.iterdir()):
            if not policy_dir.is_dir():
                continue
            scores = contract_scores(policy_dir)
            if scores is None:
                continue
            summary_path = policy_dir / "offline_replay_summary.json"
            summary = (
                json.loads(summary_path.read_text(encoding="utf-8"))
                if summary_path.exists()
                else {}
            )
            rows.append(
                {
                    "base_run": base_run,
                    "chunk_ms": chunk_ms,
                    "policy": policy_dir.name,
                    **scores,
                    "delta_cu": scores["cu"] - base_scores["cu"],
                    "delta_chrf": scores["chrf"] - base_scores["chrf"],
                    "rejected_updates": summary.get("rejected_update_count"),
                    "changed_candidates": summary.get("changed_candidate_update_count"),
                    "agreement_rescued_units": summary.get("agreement_rescued_unit_count"),
                    "control_pred_mismatch": summary.get(
                        "control_prediction_mismatch_count"
                    ),
                    "control_delays_mismatch": summary.get(
                        "control_delays_mismatch_count"
                    ),
                }
            )
    return rows


def row_for(rows: list[dict[str, Any]], base_run: str, policy: str) -> dict[str, Any] | None:
    return next(
        (row for row in rows if row["base_run"] == base_run and row["policy"] == policy),
        None,
    )


def print_verdicts(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    print("\n=== Decision gates ===")
    gate0 = True
    for base_run in BASE_RUNS:
        control = row_for(rows, base_run, "control")
        ok = (
            control is not None
            and control.get("control_pred_mismatch") == 0
            and control.get("control_delays_mismatch") == 0
        )
        gate0 = gate0 and ok
        print(f"  [Gate0] {base_run}: control exact -> {'PASS' if ok else 'FAIL'}")
    if not gate0:
        print("verdict: INVALID (Gate 0 failed; all replay numbers untrusted)")
        return

    print("  [D1] theta-0 runs - pure LA latency (optimistic bound) vs original:")
    d1_attention_dominates = True
    for base_run in THETA0_RUNS:
        la2 = row_for(rows, base_run, "la2")
        if la2 is None:
            d1_attention_dominates = False
            print(f"    {base_run}: la2 missing")
            continue
        dominates = la2["delta_cu"] >= args.d1_min_cu_penalty_ms
        d1_attention_dominates = d1_attention_dominates and dominates
        print(
            f"    {base_run}: delta_cu={la2['delta_cu']:+.0f} ms "
            f"delta_chrf={la2['delta_chrf']:+.2f} -> "
            f"{'attention dominates' if dominates else 'LA competitive'}"
        )
    print(
        f"    D1 verdict: "
        f"{'attention-only dominates output-agreement (optimistic LA bound is slower)' if d1_attention_dominates else 'LA competitive at theta-0 latency; grounded-to-meaning deserves GPU'}"
    )

    print("  [D2] unit_conf runs - hybrid rescue (recorded OR agreement):")
    d2_candidates = []
    for base_run in UNITCONF_RUNS:
        hybrid = row_for(rows, base_run, "hyb_rec_la2")
        if hybrid is None:
            print(f"    {base_run}: hyb_rec_la2 missing")
            continue
        promotable = (
            hybrid["delta_cu"] <= -args.d2_min_cu_gain_ms
            and hybrid["delta_chrf"] >= -args.d2_max_chrf_drop
        )
        if promotable:
            d2_candidates.append(base_run)
        print(
            f"    {base_run}: delta_cu={hybrid['delta_cu']:+.0f} ms "
            f"delta_chrf={hybrid['delta_chrf']:+.2f} "
            f"rescued_units={hybrid.get('agreement_rescued_units')} -> "
            f"{'PROMOTABLE' if promotable else 'not promotable'}"
        )
    print(f"    D2 verdict: promotable on {d2_candidates or 'none'}")

    print("  [D3] simulator calibration - confsim0p625 from eager1280 vs real shift:")
    confsim = row_for(rows, EAGER_1280, "confsim0p625")
    original_1280 = row_for(rows, EAGER_1280, "original")
    real_10_16 = row_for(rows, UNITCONF_10_16, "original")
    if confsim and original_1280 and real_10_16:
        real_shift = real_10_16["cu"] - original_1280["cu"]
        simulated_shift = confsim["delta_cu"]
        ratio = simulated_shift / real_shift if real_shift else None
        calibrated = ratio is not None and ratio >= args.d3_min_shift_ratio
        print(
            f"    real CU shift eager1280->unitconf10/16: {real_shift:+.0f} ms; "
            f"simulated: {simulated_shift:+.0f} ms "
            f"(ratio {'n/a' if ratio is None else f'{ratio:.2f}'}) -> "
            f"{'calibrated' if calibrated else 'conf_sim/hybrid_conf rows are INDICATIVE ONLY'}"
        )
        print(
            "    note: quality axis is structurally pinned in replays "
            "(final surfaces coincide); only the latency axis is simulated."
        )
    else:
        print("    missing rows; D3 not evaluable")


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "base_run",
        "chunk_ms",
        "policy",
        "bleu",
        "chrf",
        "cu",
        "ca",
        "delta_cu",
        "delta_chrf",
        "rejected_updates",
        "changed_candidates",
        "agreement_rescued_units",
    ]
    tsv_path = output_dir / "agreement_policy_tradeoff.tsv"
    with tsv_path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write(
                "\t".join(
                    ""
                    if row.get(col) is None
                    else f"{row[col]:.4f}"
                    if isinstance(row.get(col), float)
                    else str(row[col])
                    for col in columns
                )
                + "\n"
            )
    md_path = output_dir / "agreement_policy_tradeoff.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "# EN->ZH agreement-policy replay tradeoff (offline diagnostic, "
            "quality axis pinned)\n\n"
        )
        handle.write("| base run | policy | chrF | CU ms | dCU ms | dchrF |\n")
        handle.write("|---|---|---|---|---|---|\n")
        for row in rows:
            handle.write(
                f"| {row['base_run']} | {row['policy']} | {row['chrf']:.2f} | "
                f"{row['cu']:.0f} | {row['delta_cu']:+.0f} | "
                f"{row['delta_chrf']:+.2f} |\n"
            )
    print(tsv_path)
    print(md_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-suffix", default="20260612")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DIAGNOSTICS_ROOT / "enzh_agreement_policy_tradeoff_20260612",
    )
    parser.add_argument("--d1-min-cu-penalty-ms", type=float, default=400.0)
    parser.add_argument("--d2-min-cu-gain-ms", type=float, default=150.0)
    parser.add_argument("--d2-max-chrf-drop", type=float, default=0.3)
    parser.add_argument("--d3-min-shift-ratio", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.replay_suffix)
    for row in rows:
        print(
            f"{row['base_run']:55s} {row['policy']:18s} "
            f"chrf={row['chrf']:.2f} cu={row['cu']:.0f} "
            f"dcu={row['delta_cu']:+.0f} dchrf={row['delta_chrf']:+.2f}"
        )
    write_outputs(rows, args.output_dir)
    print_verdicts(rows, args)


if __name__ == "__main__":
    main()
