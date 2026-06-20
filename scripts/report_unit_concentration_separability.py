#!/usr/bin/env python3
"""Unit-level separability of attention concentration vs head-agreement consensus.

Offline diagnostic (zero GPU): reads recorded full21 ``stream_updates.jsonl``
traces, modifies no runtime behavior. Groups the per-draft-token
``attention_confidence_per_draft_token`` features into accepted target
stability units (``target_stability_unit_end_token_indices`` filtered to
``accepted_token_count``), takes the per-unit MIN (mirroring how
``_unit_conf_token_safe`` gates the weakest token of a unit), and asks whether
a per-unit concentration floor (top1-top2 attention mass) would partition
accepted units by downstream segment quality better than the consensus ratio
already gated by ``unit_conf``.

Pre-registered GO/NO-GO criterion for implementing a runtime
``translation_alignatt_min_alignment_concentration`` floor — GO iff, on BOTH
unit_conf full21 runs (top16_final 10/16 and conf06875 11/16):
  1. spearman(min_unit_concentration, segment chrF) >= +0.10 AND
     >= 2x |spearman(min_unit_consensus, segment chrF)|;
  2. some floor in the grid has cascade deferral <= 15% on the 11/16 run and,
     on both unit_conf runs, a segment partition delta chrF >= +0.02 with
     n_below >= 15 where the deferral-matched consensus threshold achieves
     less than half that delta;
  3. spearman(min_unit_concentration, chrF) is positive on both theta-0 runs.
NO-GO otherwise -> do not implement the floor; fall back to learned acceptance.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.report_attention_confidence_separability import (  # noqa: E402
    chrf_score,
    docid_to_wav_stem,
    emission_window,
    spearman,
)

DIAGNOSTICS_ROOT = REPO_ROOT / "outputs" / "diagnostics_jarvislab_20260606"
DEFAULT_RUN_DIRS = (
    DIAGNOSTICS_ROOT / "gemma_zh_clean_eager_chunk960_full21_20260610",
    DIAGNOSTICS_ROOT / "gemma_zh_clean_eager_chunk1280_full21_20260610",
    DIAGNOSTICS_ROOT / "gemma_zh_unitconf_top16_final_full21_20260611",
    DIAGNOSTICS_ROOT / "gemma_zh_unitconf_top16_conf06875_full21_20260611",
)
DEFAULT_FLOORS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2)
UNITCONF_RUNS = (
    "gemma_zh_unitconf_top16_final_full21_20260611",
    "gemma_zh_unitconf_top16_conf06875_full21_20260611",
)
RUN_11_16 = "gemma_zh_unitconf_top16_conf06875_full21_20260611"
THETA0_RUNS = (
    "gemma_zh_clean_eager_chunk960_full21_20260610",
    "gemma_zh_clean_eager_chunk1280_full21_20260610",
)
MIN_SEGMENTS_BELOW = 15
MIN_SPEARMAN_CONCENTRATION = 0.10
MAX_CASCADE_DEFERRAL = 0.15
MIN_PARTITION_DELTA_CHRF = 0.02


def accepted_unit_feature_spans(metadata: dict[str, Any]) -> list[tuple[int, int]]:
    """Token index spans [start, end) of fully accepted target stability units.

    ``target_stability_unit_end_token_indices`` is 1-indexed over the accepted
    candidate ids; units whose end exceeds ``accepted_token_count`` (e.g. a
    word-boundary trim) are not accepted and are dropped.
    """
    ends = metadata.get("target_stability_unit_end_token_indices") or []
    features = metadata.get("attention_confidence_per_draft_token") or []
    accepted = metadata.get("accepted_token_count")
    if accepted is None:
        accepted = len(features)
    accepted = min(int(accepted), len(features))
    spans: list[tuple[int, int]] = []
    prev = 0
    for raw_end in ends:
        end = int(raw_end)
        if end <= prev:
            continue
        if end > accepted:
            break
        spans.append((prev, end))
        prev = end
    return spans


def unit_feature_minima(
    features: list[dict[str, Any]], spans: list[tuple[int, int]], name: str
) -> list[float | None]:
    """Per-unit min of one confidence feature; None when no finite value."""
    minima: list[float | None] = []
    for start, end in spans:
        values = [
            float(row.get(name))
            for row in features[start:end]
            if row.get(name) is not None and math.isfinite(float(row.get(name)))
        ]
        minima.append(min(values) if values else None)
    return minima


def floor_simulation(
    per_update_unit_minima: list[list[float | None]], floor: float
) -> dict[str, Any]:
    """Deferral cost of a min-feature floor over recorded accepted units.

    independent: every unit below the floor counts deferred. cascade: within an
    update, the first unit below the floor defers itself and every later unit
    (faithful to prefix acceptance, the honest CU proxy). None minima count as
    below the floor (no finite evidence -> not safe).
    """
    n_units = 0
    deferred_independent = 0
    deferred_cascade = 0
    none_units = 0
    for minima in per_update_unit_minima:
        n_units += len(minima)
        below_seen = False
        for value in minima:
            below = value is None or value < floor
            if value is None:
                none_units += 1
            if below:
                deferred_independent += 1
            if below_seen or below:
                below_seen = True
                deferred_cascade += 1
    return {
        "n_units": n_units,
        "none_unit_count": none_units,
        "deferred_frac_independent": deferred_independent / n_units if n_units else None,
        "deferred_frac_cascade": deferred_cascade / n_units if n_units else None,
    }


def matched_threshold(minima: list[float], target_deferral: float) -> float | None:
    """Threshold on another feature deferring ~the same fraction of units."""
    finite = sorted(minima)
    if not finite:
        return None
    index = round(target_deferral * len(finite))
    if index <= 0:
        return -math.inf
    if index >= len(finite):
        return math.inf
    return finite[index]


def partition_delta_chrf(
    rows: list[dict[str, Any]], floor: float, key: str
) -> dict[str, Any]:
    """Mean segment chrF above vs below the floor on a per-segment min feature."""
    below = [row["chrf"] for row in rows if row.get(key) is not None and row[key] < floor]
    above = [row["chrf"] for row in rows if row.get(key) is not None and row[key] >= floor]
    mean_below = sum(below) / len(below) if below else None
    mean_above = sum(above) / len(above) if above else None
    delta = (
        mean_above - mean_below
        if mean_below is not None and mean_above is not None
        else None
    )
    return {
        "n_below": len(below),
        "n_above": len(above),
        "mean_chrf_below": mean_below,
        "mean_chrf_above": mean_above,
        "delta_chrf": delta,
    }


def write_tsv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write(
                "\t".join(
                    ""
                    if row.get(col) is None
                    else f"{row[col]:.6f}"
                    if isinstance(row.get(col), float)
                    else str(row[col])
                    for col in columns
                )
                + "\n"
            )


def analyze_run(
    run_dir: Path, floors: list[float], *, chrf_char_order: int, chrf_beta: float
) -> dict[str, Any]:
    doc_map = docid_to_wav_stem(run_dir)
    entries_by_wav: dict[str, list[dict[str, Any]]] = {}
    all_concentration_minima: list[list[float | None]] = []
    all_consensus_minima: list[float] = []
    for line in (run_dir / "stream_updates.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        update = json.loads(line)
        metadata = update.get("alignatt_metadata") or {}
        spans = accepted_unit_feature_spans(metadata)
        if not spans:
            continue
        features = metadata.get("attention_confidence_per_draft_token") or []
        concentration = unit_feature_minima(features, spans, "concentration")
        consensus = unit_feature_minima(features, spans, "consensus_ratio")
        all_concentration_minima.append(concentration)
        all_consensus_minima.extend(v for v in consensus if v is not None)
        wav = str(update.get("wav_name") or update.get("input_name") or "")
        entries_by_wav.setdefault(wav.removesuffix(".wav"), []).append(
            {
                "audio_processed_ms": update.get("audio_processed_ms"),
                "concentration_minima": concentration,
                "consensus_minima": consensus,
            }
        )

    segment_rows: list[dict[str, Any]] = []
    for line in (
        (run_dir / "instances.resegmented.jsonl").read_text(encoding="utf-8").splitlines()
    ):
        if not line.strip():
            continue
        instance = json.loads(line)
        window = emission_window(instance)
        if window is None:
            continue
        raw_docid = str(instance.get("docid", ""))
        docid = doc_map.get(raw_docid, raw_docid.removesuffix(".wav"))
        start_ms, end_ms = window
        concentration_minima: list[float] = []
        consensus_minima: list[float] = []
        n_units = 0
        for entry in entries_by_wav.get(docid, []):
            at_ms = entry["audio_processed_ms"]
            if at_ms is None or not (start_ms <= float(at_ms) <= end_ms):
                continue
            n_units += len(entry["concentration_minima"])
            concentration_minima.extend(
                v for v in entry["concentration_minima"] if v is not None
            )
            consensus_minima.extend(v for v in entry["consensus_minima"] if v is not None)
        if n_units == 0:
            continue
        segment_rows.append(
            {
                "docid": docid,
                "segid": instance.get("segid"),
                "chrf": chrf_score(
                    str(instance.get("prediction", "")),
                    str(instance.get("reference", "")),
                    char_order=chrf_char_order,
                    beta=chrf_beta,
                ),
                "n_units": n_units,
                "min_unit_concentration": min(concentration_minima)
                if concentration_minima
                else None,
                "mean_unit_min_concentration": sum(concentration_minima)
                / len(concentration_minima)
                if concentration_minima
                else None,
                "min_unit_consensus": min(consensus_minima) if consensus_minima else None,
                "mean_unit_min_consensus": sum(consensus_minima) / len(consensus_minima)
                if consensus_minima
                else None,
            }
        )

    spearmans: dict[str, float | None] = {}
    for key in (
        "min_unit_concentration",
        "mean_unit_min_concentration",
        "min_unit_consensus",
        "mean_unit_min_consensus",
    ):
        paired = [
            (row[key], row["chrf"]) for row in segment_rows if row.get(key) is not None
        ]
        spearmans[key] = (
            spearman([p[0] for p in paired], [p[1] for p in paired])
            if len(paired) >= 3
            else None
        )

    floor_rows: list[dict[str, Any]] = []
    for floor in floors:
        simulation = floor_simulation(all_concentration_minima, floor)
        partition = partition_delta_chrf(segment_rows, floor, "min_unit_concentration")
        target = simulation["deferred_frac_independent"] or 0.0
        consensus_threshold = matched_threshold(all_consensus_minima, target)
        if consensus_threshold is None:
            consensus_partition = {
                "n_below": None,
                "n_above": None,
                "delta_chrf": None,
            }
        else:
            consensus_partition = partition_delta_chrf(
                segment_rows, consensus_threshold, "min_unit_consensus"
            )
        floor_rows.append(
            {
                "floor": floor,
                "n_units": simulation["n_units"],
                "none_unit_count": simulation["none_unit_count"],
                "deferred_frac_independent": simulation["deferred_frac_independent"],
                "deferred_frac_cascade": simulation["deferred_frac_cascade"],
                "n_seg_below": partition["n_below"],
                "n_seg_above": partition["n_above"],
                "mean_chrf_below": partition["mean_chrf_below"],
                "mean_chrf_above": partition["mean_chrf_above"],
                "delta_chrf": partition["delta_chrf"],
                "matched_consensus_threshold": consensus_threshold
                if consensus_threshold not in (None, -math.inf, math.inf)
                else None,
                "consensus_n_seg_below": consensus_partition["n_below"],
                "delta_chrf_consensus_matched": consensus_partition["delta_chrf"],
            }
        )

    return {
        "run": run_dir.name,
        "segment_rows": segment_rows,
        "spearmans": spearmans,
        "floor_rows": floor_rows,
        "n_units": sum(len(m) for m in all_concentration_minima),
    }


def floor_qualifies(
    floor: float, results_by_run: dict[str, dict[str, Any]]
) -> tuple[bool, str]:
    row_11_16 = next(
        row
        for row in results_by_run[RUN_11_16]["floor_rows"]
        if row["floor"] == floor
    )
    cascade = row_11_16["deferred_frac_cascade"]
    if cascade is None or cascade > MAX_CASCADE_DEFERRAL:
        return False, f"cascade deferral {cascade} > {MAX_CASCADE_DEFERRAL} on 11/16"
    for run in UNITCONF_RUNS:
        row = next(
            r for r in results_by_run[run]["floor_rows"] if r["floor"] == floor
        )
        delta = row["delta_chrf"]
        if delta is None or delta < MIN_PARTITION_DELTA_CHRF:
            return False, f"delta_chrf {delta} < {MIN_PARTITION_DELTA_CHRF} on {run}"
        if row["n_seg_below"] is None or row["n_seg_below"] < MIN_SEGMENTS_BELOW:
            return False, f"n_seg_below {row['n_seg_below']} < {MIN_SEGMENTS_BELOW} on {run}"
        consensus_delta = row["delta_chrf_consensus_matched"]
        if consensus_delta is not None and consensus_delta >= delta / 2.0:
            return False, (
                f"consensus-matched delta {consensus_delta:.4f} >= half of "
                f"{delta:.4f} on {run}"
            )
    return True, "qualifies"


def evaluate_verdict(
    results_by_run: dict[str, dict[str, Any]], floors: list[float]
) -> None:
    required = set(UNITCONF_RUNS) | set(THETA0_RUNS)
    if not required.issubset(results_by_run):
        print("verdict: skipped (default 4-run set not fully analyzed)")
        return

    print("\n=== GO/NO-GO verdict (pre-registered criteria) ===")
    criterion1 = True
    for run in UNITCONF_RUNS:
        sp = results_by_run[run]["spearmans"]
        conc = sp["min_unit_concentration"]
        cons = sp["min_unit_consensus"]
        ok = (
            conc is not None
            and conc >= MIN_SPEARMAN_CONCENTRATION
            and (cons is None or conc >= 2.0 * abs(cons))
        )
        criterion1 = criterion1 and ok
        print(
            f"  [1] {run}: spearman(min_unit_concentration)="
            f"{'n/a' if conc is None else f'{conc:+.3f}'} "
            f"vs consensus={'n/a' if cons is None else f'{cons:+.3f}'} -> "
            f"{'PASS' if ok else 'FAIL'}"
        )

    qualifying = []
    for floor in floors:
        ok, reason = floor_qualifies(floor, results_by_run)
        if ok:
            qualifying.append(floor)
        print(f"  [2] floor {floor}: {'PASS' if ok else f'FAIL ({reason})'}")
    criterion2 = bool(qualifying)

    criterion3 = True
    for run in THETA0_RUNS:
        conc = results_by_run[run]["spearmans"]["min_unit_concentration"]
        ok = conc is not None and conc > 0.0
        criterion3 = criterion3 and ok
        print(
            f"  [3] {run}: spearman(min_unit_concentration)="
            f"{'n/a' if conc is None else f'{conc:+.3f}'} -> "
            f"{'PASS' if ok else 'FAIL'}"
        )

    verdict = "GO" if (criterion1 and criterion2 and criterion3) else "NO-GO"
    print(
        f"verdict: {verdict} "
        f"(criterion1={criterion1}, criterion2={criterion2} "
        f"qualifying_floors={qualifying}, criterion3={criterion3})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=[],
        help="Repeatable; defaults to the four gemma_zh full21 trace runs.",
    )
    parser.add_argument(
        "--floors",
        default=",".join(str(f) for f in DEFAULT_FLOORS),
        help="Comma-separated concentration floors to simulate.",
    )
    parser.add_argument("--chrf-char-order", type=int, default=6)
    parser.add_argument("--chrf-beta", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = args.run_dir or [Path(p) for p in DEFAULT_RUN_DIRS]
    floors = [float(value) for value in args.floors.split(",") if value.strip()]

    segment_columns = [
        "docid",
        "segid",
        "chrf",
        "n_units",
        "min_unit_concentration",
        "mean_unit_min_concentration",
        "min_unit_consensus",
        "mean_unit_min_consensus",
    ]
    floor_columns = [
        "floor",
        "n_units",
        "none_unit_count",
        "deferred_frac_independent",
        "deferred_frac_cascade",
        "n_seg_below",
        "n_seg_above",
        "mean_chrf_below",
        "mean_chrf_above",
        "delta_chrf",
        "matched_consensus_threshold",
        "consensus_n_seg_below",
        "delta_chrf_consensus_matched",
    ]

    results_by_run: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        result = analyze_run(
            run_dir,
            floors,
            chrf_char_order=args.chrf_char_order,
            chrf_beta=args.chrf_beta,
        )
        results_by_run[result["run"]] = result
        write_tsv(
            args.output_dir / f"unit_concentration_separability_{result['run']}.tsv",
            segment_columns,
            result["segment_rows"],
        )
        write_tsv(
            args.output_dir / f"unit_concentration_floor_simulation_{result['run']}.tsv",
            floor_columns,
            result["floor_rows"],
        )
        print(
            f"{result['run']}: {len(result['segment_rows'])} segments, "
            f"{result['n_units']} accepted units"
        )
        for key, value in result["spearmans"].items():
            print(
                f"  spearman({key}, chrf) = "
                f"{'n/a' if value is None else f'{value:+.3f}'}"
            )
        for row in result["floor_rows"]:
            delta = row["delta_chrf"]
            consensus_delta = row["delta_chrf_consensus_matched"]
            print(
                f"  floor {row['floor']}: defer_indep="
                f"{row['deferred_frac_independent']:.3f} "
                f"defer_cascade={row['deferred_frac_cascade']:.3f} "
                f"delta_chrf={'n/a' if delta is None else f'{delta:+.4f}'} "
                f"(n_below={row['n_seg_below']}) consensus_matched_delta="
                f"{'n/a' if consensus_delta is None else f'{consensus_delta:+.4f}'}"
            )
        summary = {
            "run": result["run"],
            "n_segments": len(result["segment_rows"]),
            "n_units": result["n_units"],
        }
        for key, value in result["spearmans"].items():
            summary[f"spearman_{key}"] = value
        eligible = [
            row
            for row in result["floor_rows"]
            if row["delta_chrf"] is not None
            and row["n_seg_below"] is not None
            and row["n_seg_below"] >= MIN_SEGMENTS_BELOW
        ]
        best = max(eligible, key=lambda row: row["delta_chrf"], default=None)
        summary["best_floor"] = best["floor"] if best else None
        summary["best_floor_delta_chrf"] = best["delta_chrf"] if best else None
        summary["best_floor_deferred_frac_cascade"] = (
            best["deferred_frac_cascade"] if best else None
        )
        summary["best_floor_delta_chrf_consensus_matched"] = (
            best["delta_chrf_consensus_matched"] if best else None
        )
        summary_rows.append(summary)

    summary_columns = list(summary_rows[0].keys()) if summary_rows else []
    write_tsv(
        args.output_dir / "unit_concentration_separability_summary.tsv",
        summary_columns,
        summary_rows,
    )
    evaluate_verdict(results_by_run, floors)


if __name__ == "__main__":
    main()
