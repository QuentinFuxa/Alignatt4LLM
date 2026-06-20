#!/usr/bin/env python3
"""Offline calibration of a learned acceptance gate (handoff §5.2), zero GPU.

Labels accepted target stability units of a permissive (theta-0) run by
cross-run survival: a unit survives when its position in the run's final
surface lies inside a difflib matching block against the final surface of a
much stronger high-theta run of the same audio (the silver target). The
unit_conf statistic itself validates the label (min consensus ranks survival
well above chance), and the question is whether a learned combination of ALL
recorded per-token confidence features ({min,mean} x {consensus_ratio,
entropy_norm, concentration, argmax_mass} + token count) separates surviving
from non-surviving accepted units enough better than the single min-consensus
threshold to justify implementing a learned runtime gate.

Granularity: one sample per accepted stability unit (the gate's decision
grain). Unit features are exact (token spans); the survival label maps the
update's newly emitted surface units onto its accepted token-units
proportionally (Gemma tokens are not zh characters), binary at >= 0.5 span
survival. Cross-validation is grouped by input (no within-talk leakage);
the baseline ranker (min consensus) is evaluated on identical rows.

Pre-registered GO/NO-GO for implementing a learned acceptance gate — GO iff:
  1. on the primary pair (theta-0 chunk1280 -> unit_conf 11/16 silver), the
     grouped-CV pooled AUC of the logistic model is >= 0.65 AND >= the
     min-consensus baseline AUC + 0.04;
  2. at matched independent deferral budgets of 10/15/20% of accepted units,
     the learned ranking captures >= 1.25x as many non-surviving units as the
     min-consensus ranking on at least two of the three budgets;
  3. the learned-minus-baseline AUC gap is positive on BOTH robustness pairs
     (theta-0 chunk1280 -> 10/16 silver, theta-0 chunk960 -> 11/16 silver).
NO-GO otherwise -> the recorded attention-confidence features are exhausted
(single threshold suffices); remaining frontier-movers are backbone/ASR.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alignatt4llm.text_surface import split_public_emission_units  # noqa: E402
from tools.reports.report_unit_concentration_separability import (  # noqa: E402
    accepted_unit_feature_spans,
)

DIAGNOSTICS_ROOT = REPO_ROOT / "outputs" / "diagnostics_jarvislab_20260606"
PRIMARY_PAIR = (
    "gemma_zh_clean_eager_chunk1280_full21_20260610",
    "gemma_zh_unitconf_top16_conf06875_full21_20260611",
)
ROBUSTNESS_PAIRS = (
    (
        "gemma_zh_clean_eager_chunk1280_full21_20260610",
        "gemma_zh_unitconf_top16_final_full21_20260611",
    ),
    (
        "gemma_zh_clean_eager_chunk960_full21_20260610",
        "gemma_zh_unitconf_top16_conf06875_full21_20260611",
    ),
)
FEATURE_NAMES = ("consensus_ratio", "entropy_norm", "concentration", "argmax_mass")
BASELINE_FEATURE = "min_consensus_ratio"
DEFERRAL_BUDGETS = (0.10, 0.15, 0.20)
MIN_PRIMARY_AUC = 0.65
MIN_AUC_GAP = 0.04
MIN_CAPTURE_RATIO = 1.25
N_FOLDS = 7


def survival_positions(low_units: list[str], high_units: list[str]) -> set[int]:
    """Positions of low-run final-surface units inside difflib match blocks."""
    matcher = difflib.SequenceMatcher(None, low_units, high_units, autojunk=False)
    survived: set[int] = set()
    for block in matcher.get_matching_blocks():
        survived.update(range(block.a, block.a + block.size))
    return survived


def unit_char_spans(
    unit_count: int, tail_start: int, tail_length: int
) -> list[tuple[int, int]]:
    """Proportional [start, end) final-surface spans for the update's units."""
    spans: list[tuple[int, int]] = []
    for index in range(unit_count):
        start = tail_start + round(tail_length * index / unit_count)
        end = tail_start + round(tail_length * (index + 1) / unit_count)
        spans.append((start, end))
    return spans


def unit_feature_row(
    features: list[dict[str, Any]], span: tuple[int, int]
) -> dict[str, float | None]:
    row: dict[str, float | None] = {"n_tokens": float(span[1] - span[0])}
    for name in FEATURE_NAMES:
        values = [
            float(feature.get(name))
            for feature in features[span[0] : span[1]]
            if feature.get(name) is not None
            and math.isfinite(float(feature.get(name)))
        ]
        row[f"min_{name}"] = min(values) if values else None
        row[f"mean_{name}"] = sum(values) / len(values) if values else None
    return row


def collect_samples(low_run: Path, high_run: Path) -> list[dict[str, Any]]:
    high_final: dict[str, list[str]] = {}
    for line in (high_run / "hypothesis.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        high_final[str(record["source"][0])] = split_public_emission_units(
            str(record.get("prediction") or ""), target_lang_code="zh"
        )

    updates_by_input: dict[str, list[dict[str, Any]]] = {}
    for line in (low_run / "stream_updates.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        update = json.loads(line)
        updates_by_input.setdefault(str(update.get("input_name") or ""), []).append(
            update
        )

    samples: list[dict[str, Any]] = []
    for input_name, updates in updates_by_input.items():
        low_final = split_public_emission_units(
            str(updates[-1].get("translation_text") or ""), target_lang_code="zh"
        )
        survived = survival_positions(low_final, high_final.get(input_name, []))
        previous_length = 0
        previous_partial_units: list[str] = []
        for update in updates:
            surface_units = split_public_emission_units(
                str(update.get("translation_text") or ""), target_lang_code="zh"
            )
            tail_length = len(surface_units) - previous_length
            metadata = update.get("alignatt_metadata") or {}
            spans = accepted_unit_feature_spans(metadata)
            partial_units = split_public_emission_units(
                str(update.get("partial_accepted_target") or ""), target_lang_code="zh"
            )
            if tail_length <= 0 or not spans:
                previous_length = len(surface_units)
                previous_partial_units = partial_units
                continue
            # On hypothesis-restart updates the surface tail also contains the
            # flushed completion of the OLD hypothesis; the recorded unit spans
            # describe only the newly accepted text, which lands as the SUFFIX
            # of the tail. Map spans onto that accepted suffix window only.
            if (
                previous_partial_units
                and partial_units[: len(previous_partial_units)] == previous_partial_units
            ):
                accepted_window = len(partial_units) - len(previous_partial_units)
            else:
                accepted_window = len(partial_units)
            accepted_window = min(accepted_window, tail_length)
            if accepted_window <= 0:
                previous_length = len(surface_units)
                previous_partial_units = partial_units
                continue
            features = metadata.get("attention_confidence_per_draft_token") or []
            char_spans = unit_char_spans(
                len(spans),
                previous_length + tail_length - accepted_window,
                accepted_window,
            )
            for unit_index, (token_span, char_span) in enumerate(
                zip(spans, char_spans)
            ):
                span_size = char_span[1] - char_span[0]
                if span_size <= 0:
                    continue
                survived_count = sum(
                    1 for position in range(char_span[0], char_span[1]) if position in survived
                )
                row = unit_feature_row(features, token_span)
                row.update(
                    {
                        "input_name": input_name,
                        "update_idx": update.get("update_idx"),
                        "unit_index": unit_index,
                        "survival_frac": survived_count / span_size,
                        "label": int(survived_count / span_size >= 0.5),
                    }
                )
                samples.append(row)
            previous_length = len(surface_units)
            previous_partial_units = partial_units
    return samples


def cross_validated_scores(samples: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feature_columns = ["n_tokens"] + [
        f"{stat}_{name}" for name in FEATURE_NAMES for stat in ("min", "mean")
    ]
    rows = [s for s in samples if s.get(BASELINE_FEATURE) is not None]
    matrix = np.array(
        [[np.nan if r.get(c) is None else float(r[c]) for c in feature_columns] for r in rows]
    )
    labels = np.array([int(r["label"]) for r in rows])
    groups = np.array([r["input_name"] for r in rows])
    baseline_values = np.array([float(r[BASELINE_FEATURE]) for r in rows])

    out_of_fold = np.full(len(rows), np.nan)
    splitter = GroupKFold(n_splits=N_FOLDS)
    for train_index, test_index in splitter.split(matrix, labels, groups):
        model = make_pipeline(
            SimpleImputer(strategy="mean"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0),
        )
        model.fit(matrix[train_index], labels[train_index])
        out_of_fold[test_index] = model.predict_proba(matrix[test_index])[:, 1]

    learned_auc = float(roc_auc_score(labels, out_of_fold))
    baseline_auc = float(roc_auc_score(labels, baseline_values))
    per_feature_auc: dict[str, float] = {}
    for column in feature_columns:
        values = np.array(
            [np.nan if r.get(column) is None else float(r[column]) for r in rows]
        )
        mask = ~np.isnan(values)
        if mask.sum() >= 100 and 0 < labels[mask].mean() < 1:
            per_feature_auc[column] = float(roc_auc_score(labels[mask], values[mask]))

    final_model = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0),
    )
    final_model.fit(matrix, labels)
    coefficients = dict(
        zip(feature_columns, final_model.named_steps["logisticregression"].coef_[0])
    )

    captures: dict[str, dict[str, float]] = {}
    bad_total = int((labels == 0).sum())
    learned_risk_order = np.argsort(out_of_fold)
    baseline_risk_order = np.argsort(baseline_values)
    for budget in DEFERRAL_BUDGETS:
        flagged = int(round(budget * len(rows)))
        learned_capture = float(
            (labels[learned_risk_order[:flagged]] == 0).sum() / max(1, bad_total)
        )
        baseline_capture = float(
            (labels[baseline_risk_order[:flagged]] == 0).sum() / max(1, bad_total)
        )
        captures[f"{budget:.2f}"] = {
            "learned_capture": learned_capture,
            "baseline_capture": baseline_capture,
            "ratio": learned_capture / baseline_capture if baseline_capture else None,
        }

    return {
        "n_samples": len(rows),
        "n_dropped_missing_baseline": len(samples) - len(rows),
        "survival_rate": float(labels.mean()),
        "learned_auc": learned_auc,
        "baseline_auc": baseline_auc,
        "auc_gap": learned_auc - baseline_auc,
        "per_feature_auc": per_feature_auc,
        "coefficients": coefficients,
        "captures": captures,
    }


def pair_report(low_run: Path, high_run: Path, role: str) -> dict[str, Any]:
    samples = collect_samples(low_run, high_run)
    scores = cross_validated_scores(samples)
    report = {"low_run": low_run.name, "high_run": high_run.name, "role": role, **scores}
    print(
        f"\n[{role}] {low_run.name} -> {high_run.name}\n"
        f"  units={scores['n_samples']} survival_rate={scores['survival_rate']:.3f}\n"
        f"  AUC learned={scores['learned_auc']:.3f} "
        f"baseline(min_consensus)={scores['baseline_auc']:.3f} "
        f"gap={scores['auc_gap']:+.3f}"
    )
    for column, value in sorted(
        scores["per_feature_auc"].items(), key=lambda item: -abs(item[1] - 0.5)
    ):
        print(f"    auc({column}) = {value:.3f}")
    for budget, capture in scores["captures"].items():
        ratio = capture["ratio"]
        print(
            f"    capture@{budget}: learned={capture['learned_capture']:.3f} "
            f"baseline={capture['baseline_capture']:.3f} "
            f"ratio={'n/a' if ratio is None else f'{ratio:.2f}'}"
        )
    return report


def evaluate_verdict(reports: list[dict[str, Any]]) -> None:
    primary = next(r for r in reports if r["role"] == "primary")
    criterion1 = (
        primary["learned_auc"] >= MIN_PRIMARY_AUC
        and primary["auc_gap"] >= MIN_AUC_GAP
    )
    budget_passes = sum(
        1
        for capture in primary["captures"].values()
        if capture["ratio"] is not None and capture["ratio"] >= MIN_CAPTURE_RATIO
    )
    criterion2 = budget_passes >= 2
    robustness = [r for r in reports if r["role"] == "robustness"]
    criterion3 = bool(robustness) and all(r["auc_gap"] > 0.0 for r in robustness)
    verdict = "GO" if (criterion1 and criterion2 and criterion3) else "NO-GO"
    print("\n=== GO/NO-GO verdict (pre-registered criteria) ===")
    print(
        f"  [1] primary learned AUC {primary['learned_auc']:.3f} >= {MIN_PRIMARY_AUC} "
        f"and gap {primary['auc_gap']:+.3f} >= +{MIN_AUC_GAP} -> "
        f"{'PASS' if criterion1 else 'FAIL'}"
    )
    print(
        f"  [2] capture ratio >= {MIN_CAPTURE_RATIO} on >=2/3 budgets "
        f"({budget_passes}/3) -> {'PASS' if criterion2 else 'FAIL'}"
    )
    for r in robustness:
        print(
            f"  [3] {r['low_run']} -> {r['high_run']}: gap {r['auc_gap']:+.3f} -> "
            f"{'PASS' if r['auc_gap'] > 0 else 'FAIL'}"
        )
    print(f"verdict: {verdict}")


def write_summary(reports: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n{output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-run", type=Path, default=None)
    parser.add_argument("--high-run", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "plots" / "learned_acceptance_calibration_20260612.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.low_run and args.high_run:
        reports = [pair_report(args.low_run, args.high_run, "custom")]
    else:
        reports = [
            pair_report(
                DIAGNOSTICS_ROOT / PRIMARY_PAIR[0],
                DIAGNOSTICS_ROOT / PRIMARY_PAIR[1],
                "primary",
            )
        ]
        for low, high in ROBUSTNESS_PAIRS:
            reports.append(
                pair_report(DIAGNOSTICS_ROOT / low, DIAGNOSTICS_ROOT / high, "robustness")
            )
        evaluate_verdict(reports)
    write_summary(reports, args.output)


if __name__ == "__main__":
    main()
