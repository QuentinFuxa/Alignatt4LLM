#!/usr/bin/env python3
"""2-feature search for the hardest MT observer gate (``alignatt:rewind``).

The per-gate v1/v2 analyses found that ``alignatt:source_frontier``
is cleanly single-feature separable (F1 0.91-0.98) but
``alignatt:rewind`` caps at F1 ≤ 0.75 under every single feature we
tried, including the closest-to-definition feature
``max_backward_jump``.

This script grid-searches pairs (feat_A, feat_B) with thresholds and
an AND / OR combination rule, asking whether any 2-feature
conjunction or disjunction lifts rewind above the single-feature
cap. If nothing clears F1 ~0.85, the "rewind is irreducible to
observed features alone" conclusion holds stronger.

The search operates at the update level (one decision per update).
Thresholds for each feature are drawn from the empirical quantiles
in the data (11 values over [5th, 95th] percentile) to keep the
Cartesian product tractable.

Usage::

    PYTHONPATH=. .venv-inference/bin/python \
      scripts/two_feature_gate_search.py \
      --input outputs/night1_cs_en_vllm_mt_chunk450/stream_updates.jsonl \
      --gate alignatt:rewind
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

# Reuse the feature-extraction code from per_gate_separability_v2
from per_gate_separability_v2 import FEATURES, build_records


def quantiles(values: list[float], n: int = 11) -> list[float]:
    if not values:
        return []
    values = sorted(v for v in values if v is not None)
    if not values:
        return []
    if len(values) == 1:
        return list(values)
    lo = values[len(values) * 5 // 100]
    hi = values[min(len(values) - 1, len(values) * 95 // 100)]
    if hi == lo:
        return [lo]
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]


def eval_rule(pos_records: list[dict], neg_records: list[dict],
              feat_a: str, thr_a: float, dir_a: str,
              feat_b: str, thr_b: float, dir_b: str,
              combine: str) -> tuple[float, float, float]:
    def predict(r: dict) -> bool:
        va, vb = r.get(feat_a), r.get(feat_b)
        if va is None or vb is None:
            return False
        ok_a = (va >= thr_a) if dir_a == "≥" else (va <= thr_a)
        ok_b = (vb >= thr_b) if dir_b == "≥" else (vb <= thr_b)
        return (ok_a and ok_b) if combine == "AND" else (ok_a or ok_b)

    tp = sum(1 for r in pos_records if predict(r))
    fp = sum(1 for r in neg_records if predict(r))
    fn = len(pos_records) - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def search(records: list[dict], gate: str,
           focus_features: list[str] | None = None) -> list[dict]:
    pos = [r for r in records if r["stop_reason"] == gate]
    neg = [r for r in records if r["stop_reason"] != gate]
    if not pos or not neg:
        return []
    feature_names = focus_features or FEATURES
    thresholds_by_feature: dict[str, list[float]] = {}
    for feat in feature_names:
        vals = [r.get(feat) for r in records if r.get(feat) is not None]
        thresholds_by_feature[feat] = quantiles([float(v) for v in vals])

    results: list[dict] = []
    pairs = list(itertools.combinations(feature_names, 2))
    for feat_a, feat_b in pairs:
        thr_a_list = thresholds_by_feature.get(feat_a) or []
        thr_b_list = thresholds_by_feature.get(feat_b) or []
        if not thr_a_list or not thr_b_list:
            continue
        for thr_a in thr_a_list:
            for thr_b in thr_b_list:
                for dir_a, dir_b, combine in itertools.product(
                    ("≥", "≤"), ("≥", "≤"), ("AND", "OR")
                ):
                    prec, rec, f1 = eval_rule(
                        pos, neg, feat_a, thr_a, dir_a,
                        feat_b, thr_b, dir_b, combine,
                    )
                    if f1 > 0.0:
                        results.append({
                            "feat_a": feat_a, "thr_a": thr_a, "dir_a": dir_a,
                            "feat_b": feat_b, "thr_b": thr_b, "dir_b": dir_b,
                            "combine": combine,
                            "prec": prec, "rec": rec, "f1": f1,
                        })
    results.sort(key=lambda r: -r["f1"])
    return results


def summarise(records: list[dict], gate: str) -> str:
    pos = [r for r in records if r["stop_reason"] == gate]
    neg = [r for r in records if r["stop_reason"] != gate]
    top_results = search(records, gate)
    if not top_results:
        return f"# {gate}: no results (n_pos={len(pos)}, n_neg={len(neg)})"

    lines = []
    lines.append(f"# gate={gate}  n_pos={len(pos)}  n_neg={len(neg)}")
    lines.append(f"# search size = {len(top_results)} valid rules")
    lines.append("")
    lines.append(f"{'combine':<8} {'feat_a':<35} {'dir':>3} {'thr_a':>10} "
                 f"{'feat_b':<35} {'dir':>3} {'thr_b':>10} "
                 f"{'prec':>6} {'rec':>6} {'F1':>6}")
    for r in top_results[:10]:
        lines.append(
            f"{r['combine']:<8} {r['feat_a']:<35} {r['dir_a']:>3} "
            f"{r['thr_a']:>10.3f} {r['feat_b']:<35} {r['dir_b']:>3} "
            f"{r['thr_b']:>10.3f} {r['prec']:>6.3f} {r['rec']:>6.3f} "
            f"{r['f1']:>6.3f}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--gate", default="alignatt:rewind")
    return p.parse_args()


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    args = parse_args()
    records = build_records(Path(args.input))
    report = summarise(records, args.gate)
    print(report)
    out = Path(args.input).parent / f"two_feature_search_{args.gate.replace(':', '_')}.txt"
    out.write_text(report + "\n")


if __name__ == "__main__":
    main()
