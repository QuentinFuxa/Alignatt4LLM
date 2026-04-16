#!/usr/bin/env python3
"""Multi-feature logistic regression for the rewind gate.

Closes the single-feature → 2-feature → multi-feature → loop-replay
spectrum. Fits an L2-regularised logistic regression on the
per-update feature set from ``per_gate_separability_v2.build_records``
(provenance averages + positional features + monotonicity features),
using each artifact's rewind gate as the target.

Evaluated via stratified 5-fold cross-validation so the reported F1
is not an optimistic in-sample fit. Also prints the top feature
weights so the relative importance is readable.

Usage::

    PYTHONPATH=scripts:. .venv-inference/bin/python \
      scripts/multi_feature_rewind_classifier.py \
      --input outputs/night1_cs_en_chunk450/stream_updates.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from per_gate_separability_v2 import FEATURES, build_records


def make_design_matrix(records, gate: str):
    xs, ys = [], []
    for r in records:
        row = [r.get(feat, 0.0) for feat in FEATURES]
        if any(v is None for v in row):
            continue
        xs.append([float(v) for v in row])
        ys.append(1 if r["stop_reason"] == gate else 0)
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.int64)


def report(input_path: Path, gate: str) -> str:
    records = build_records(input_path)
    X, y = make_design_matrix(records, gate)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    lines = []
    lines.append(f"# Multi-feature classifier on {input_path}")
    lines.append(f"# gate = {gate}")
    lines.append(f"# n_records = {len(y)}  n_pos = {n_pos}  n_neg = {n_neg}")

    if n_pos < 5:
        lines.append(
            f"# SKIP: n_pos={n_pos} too small for meaningful 5-fold CV"
        )
        return "\n".join(lines)

    # Stratified 5-fold CV. Reports out-of-fold F1 so we're not
    # reporting optimistic in-sample fit.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    y_pred_all = np.zeros_like(y)
    y_score_all = np.zeros(len(y), dtype=np.float64)
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_te = scaler.transform(X[te])
        clf = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs",
            max_iter=2000, random_state=0,
            class_weight="balanced",
        )
        clf.fit(X_tr, y[tr])
        y_pred_all[te] = clf.predict(X_te)
        y_score_all[te] = clf.predict_proba(X_te)[:, 1]

    f1 = f1_score(y, y_pred_all)
    prec = precision_score(y, y_pred_all, zero_division=0.0)
    rec = recall_score(y, y_pred_all)
    lines.append("")
    lines.append(
        f"# 5-fold CV default-threshold F1 = {f1:.3f}  "
        f"(precision={prec:.3f}  recall={rec:.3f})"
    )

    # Threshold sweep on pooled out-of-fold scores to find the best
    # decision threshold for this classifier.
    best_f1, best_thr, best_prec, best_rec = 0.0, 0.5, 0.0, 0.0
    for thr in np.linspace(0.05, 0.95, 19):
        pred = (y_score_all >= thr).astype(np.int64)
        cur_f1 = f1_score(y, pred)
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_thr = thr
            best_prec = precision_score(y, pred, zero_division=0.0)
            best_rec = recall_score(y, pred)
    lines.append(
        f"# 5-fold CV best-threshold F1 = {best_f1:.3f}  "
        f"(thr={best_thr:.2f}  precision={best_prec:.3f}  recall={best_rec:.3f})"
    )

    # Fit a final model on all data for coefficient reporting. Not a
    # generalisation claim; this exposes which features carry the
    # signal.
    scaler_all = StandardScaler()
    X_scaled = scaler_all.fit_transform(X)
    clf_all = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs",
        max_iter=2000, random_state=0,
        class_weight="balanced",
    )
    clf_all.fit(X_scaled, y)
    coefs = list(zip(FEATURES, clf_all.coef_[0].tolist()))
    coefs.sort(key=lambda kv: -abs(kv[1]))
    lines.append("")
    lines.append("# Top weighted features (standardised inputs, all-data fit):")
    for feat, w in coefs[:8]:
        lines.append(f"  {feat:<40} {w:+.3f}")
    report_text = "\n".join(lines)
    print(report_text)
    out = input_path.parent / f"multi_feature_classifier_{gate.replace(':', '_')}.txt"
    out.write_text(report_text + "\n")
    return report_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--gate", default="alignatt:rewind")
    return p.parse_args()


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    args = parse_args()
    report(Path(args.input), args.gate)


if __name__ == "__main__":
    main()
