#!/usr/bin/env python3
"""Segment-level separability of attention-confidence features.

Reads a batch run directory containing ``stream_updates.jsonl`` (with the
``attention_confidence_per_draft_token`` lists recorded by the MT backend) and
``instances.resegmented.jsonl``. For every resegmented instance it aggregates
the confidence features of the tokens accepted inside the segment's emission
window and correlates them (Spearman) with the segment's character n-gram
F-score against the reference.

This is diagnostic evidence for calibrating the ``unit_conf`` acceptance
variant: if low-confidence acceptances concentrate in low-quality segments,
the alignment-confidence floor has signal to spend latency on.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

FEATURES = ("consensus_ratio", "entropy_norm", "concentration", "argmax_mass")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output TSV path (default: <run-dir name> under outputs/plots).",
    )
    parser.add_argument("--chrf-char-order", type=int, default=6)
    parser.add_argument("--chrf-beta", type=float, default=2.0)
    return parser.parse_args()


def chrf_score(prediction: str, reference: str, *, char_order: int, beta: float) -> float:
    """Plain chrF on characters (no word n-grams), enough for ranking zh segments."""
    prediction = "".join(prediction.split())
    reference = "".join(reference.split())
    if not reference:
        return 0.0
    total_precision = 0.0
    total_recall = 0.0
    orders = 0
    for order in range(1, char_order + 1):
        pred_grams = Counter(
            prediction[i : i + order] for i in range(max(0, len(prediction) - order + 1))
        )
        ref_grams = Counter(
            reference[i : i + order] for i in range(max(0, len(reference) - order + 1))
        )
        if not pred_grams and not ref_grams:
            continue
        overlap = sum((pred_grams & ref_grams).values())
        precision = overlap / max(1, sum(pred_grams.values()))
        recall = overlap / max(1, sum(ref_grams.values()))
        total_precision += precision
        total_recall += recall
        orders += 1
    if orders == 0:
        return 0.0
    precision = total_precision / orders
    recall = total_recall / orders
    if precision + recall == 0.0:
        return 0.0
    beta_sq = beta * beta
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        result = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            mean_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                result[order[k]] = mean_rank
            i = j + 1
        return result

    rx = ranks(xs)
    ry = ranks(ys)
    mean_x = sum(rx) / len(rx)
    mean_y = sum(ry) / len(ry)
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    var_x = math.sqrt(sum((a - mean_x) ** 2 for a in rx))
    var_y = math.sqrt(sum((b - mean_y) ** 2 for b in ry))
    if var_x == 0.0 or var_y == 0.0:
        return None
    return cov / (var_x * var_y)


def accepted_features_per_update(update: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = update.get("alignatt_metadata") or {}
    features = metadata.get("attention_confidence_per_draft_token") or []
    accepted = metadata.get("accepted_token_count")
    if accepted is None:
        accepted = len(features)
    return features[: int(accepted)]


def emission_window(instance: dict[str, Any]) -> tuple[float, float] | None:
    times = instance.get("emission_cu") or []
    numeric = [float(t) for t in times if t is not None]
    if not numeric:
        return None
    return min(numeric), max(numeric)


def docid_to_wav_stem(run_dir: Path) -> dict[str, str]:
    """Resegmented instances index documents by position in the batch order."""
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    input_paths = manifest.get("input_paths") or manifest.get("wav_paths") or []
    return {
        str(index): Path(str(path)).stem for index, path in enumerate(input_paths)
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    doc_map = docid_to_wav_stem(run_dir)
    updates_by_wav: dict[str, list[dict[str, Any]]] = {}
    for line in (run_dir / "stream_updates.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        update = json.loads(line)
        wav = str(update.get("wav_name") or update.get("input_name") or "")
        wav = wav.removesuffix(".wav")
        updates_by_wav.setdefault(wav, []).append(update)

    rows: list[dict[str, Any]] = []
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
        collected: dict[str, list[float]] = {name: [] for name in FEATURES}
        token_count = 0
        for update in updates_by_wav.get(docid, []):
            at_ms = update.get("audio_processed_ms")
            if at_ms is None or not (start_ms <= float(at_ms) <= end_ms):
                continue
            for feature_row in accepted_features_per_update(update):
                token_count += 1
                for name in FEATURES:
                    value = feature_row.get(name)
                    if value is not None and math.isfinite(float(value)):
                        collected[name].append(float(value))
        if token_count == 0:
            continue
        row: dict[str, Any] = {
            "docid": docid,
            "segid": instance.get("segid"),
            "chrf": chrf_score(
                str(instance.get("prediction", "")),
                str(instance.get("reference", "")),
                char_order=args.chrf_char_order,
                beta=args.chrf_beta,
            ),
            "n_tokens": token_count,
        }
        for name in FEATURES:
            values = collected[name]
            row[f"mean_{name}"] = sum(values) / len(values) if values else None
            row[f"min_{name}"] = min(values) if values else None
        rows.append(row)

    output = args.output
    if output is None:
        output = Path("outputs/plots") / (
            f"attention_confidence_separability_{run_dir.name}.tsv"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["docid", "segid", "chrf", "n_tokens"] + [
        f"{stat}_{name}" for name in FEATURES for stat in ("mean", "min")
    ]
    with output.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write(
                "\t".join(
                    "" if row.get(col) is None else f"{row[col]:.6f}"
                    if isinstance(row.get(col), float)
                    else str(row[col])
                    for col in columns
                )
                + "\n"
            )

    print(f"segments with features: {len(rows)} -> {output}")
    chrfs = [row["chrf"] for row in rows]
    for name in FEATURES:
        for stat in ("mean", "min"):
            key = f"{stat}_{name}"
            paired = [
                (row[key], row["chrf"]) for row in rows if row.get(key) is not None
            ]
            if len(paired) < 3:
                continue
            corr = spearman([p[0] for p in paired], [p[1] for p in paired])
            print(
                f"spearman({key}, chrf) = "
                f"{'n/a' if corr is None else f'{corr:+.3f}'}  (n={len(paired)})"
            )
    if chrfs:
        print(f"chrf mean={sum(chrfs)/len(chrfs):.3f}")


if __name__ == "__main__":
    main()
