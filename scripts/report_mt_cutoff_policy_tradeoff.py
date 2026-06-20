#!/usr/bin/env python3
"""Render quality-latency metrics for MT acceptance-policy sweep outputs."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


DEFAULT_REPORT_DIR = Path("outputs/reports")
DEFAULT_BOOTSTRAP_SAMPLES = 200
BOOTSTRAP_SEED = 1729


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--paper-generated-dir",
        type=Path,
        default=None,
        help="Deprecated alias for --report-dir.",
    )
    parser.add_argument("--tex-name", default="mt_cutoff_policy_quality_latency.tex")
    parser.add_argument("--frontier-tex-name", default="mt_cutoff_policy_frontier_gap.tex")
    parser.add_argument("--json-name", default="mt_cutoff_policy_quality_latency.json")
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument(
        "--cu-budget-ms",
        type=float,
        default=2000.0,
        help="CU-LongYAAL budget used for fixed-cutoff budget diagnostics.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_rows(output_root: Path) -> list[dict[str, Any]]:
    policy_points_path = output_root / "policy_points.json"
    if policy_points_path.exists():
        points = load_json(policy_points_path)
        seen_tags = {str(point.get("tag")) for point in points}
        for path in sorted(output_root.iterdir()):
            if not path.is_dir() or path.name in seen_tags:
                continue
            manifest_path = path / "manifest.json"
            runtime_config = load_json(manifest_path).get("runtime_config", {}) if manifest_path.exists() else {}
            points.append(
                {
                    "tag": path.name,
                    "output_dir": str(path),
                    "policy": runtime_config.get(
                        "translation_acceptance_policy",
                        "alignatt" if path.name.startswith("alignatt") else "cut_last_target_units",
                    ),
                    "cutoff_units": runtime_config.get("translation_static_cutoff_units", 0),
                    "alignatt_border_margin": runtime_config.get(
                        "translation_alignatt_border_margin"
                    ),
                    "alignatt_top_k_heads": runtime_config.get(
                        "translation_alignatt_top_k_heads"
                    ),
                    "alignatt_min_source_mass": runtime_config.get(
                        "translation_alignatt_min_source_mass"
                    ),
                    "alignatt_frontier_min_inaccessible_mass": runtime_config.get(
                        "translation_alignatt_frontier_min_inaccessible_mass"
                    ),
                    "alignatt_max_inaccessible_source_mass": runtime_config.get(
                        "translation_alignatt_max_inaccessible_source_mass"
                    ),
                    "alignatt_min_accessible_inaccessible_margin": runtime_config.get(
                        "translation_alignatt_min_accessible_inaccessible_margin"
                    ),
                }
            )
    else:
        points = []
        for path in sorted(output_root.iterdir()):
            if not path.is_dir():
                continue
            manifest_path = path / "manifest.json"
            runtime_config = load_json(manifest_path).get("runtime_config", {}) if manifest_path.exists() else {}
            inferred_policy = (
                "alignatt"
                if path.name.startswith("alignatt")
                else "cut_last_target_units"
            )
            points.append(
                {
                    "tag": path.name,
                    "output_dir": str(path),
                    "policy": runtime_config.get(
                        "translation_acceptance_policy",
                        inferred_policy,
                    ),
                    "cutoff_units": runtime_config.get(
                        "translation_static_cutoff_units",
                        0,
                    ),
                }
            )

    rows: list[dict[str, Any]] = []
    for point in points:
        output_dir = Path(point["output_dir"])
        evaluation_path = output_dir / "evaluation.json"
        manifest_path = output_dir / "manifest.json"
        if not evaluation_path.exists():
            continue
        evaluation = load_json(evaluation_path)
        manifest = load_json(manifest_path) if manifest_path.exists() else {}
        scores = evaluation.get("contract_scores", {}) or {}
        runtime_config = manifest.get("runtime_config", {}) or {}
        rows.append(
            {
                "tag": point.get("tag", output_dir.name),
                "policy": point.get("policy"),
                "cutoff_units": point.get("cutoff_units"),
                "alignatt_border_margin": point.get(
                    "alignatt_border_margin",
                    runtime_config.get("translation_alignatt_border_margin"),
                ),
                "alignatt_top_k_heads": point.get(
                    "alignatt_top_k_heads",
                    runtime_config.get("translation_alignatt_top_k_heads"),
                ),
                "alignatt_min_source_mass": point.get(
                    "alignatt_min_source_mass",
                    runtime_config.get("translation_alignatt_min_source_mass"),
                ),
                "alignatt_frontier_min_inaccessible_mass": point.get(
                    "alignatt_frontier_min_inaccessible_mass",
                    runtime_config.get(
                        "translation_alignatt_frontier_min_inaccessible_mass"
                    ),
                ),
                "alignatt_max_inaccessible_source_mass": point.get(
                    "alignatt_max_inaccessible_source_mass",
                    runtime_config.get("translation_alignatt_max_inaccessible_source_mass"),
                ),
                "alignatt_min_accessible_inaccessible_margin": point.get(
                    "alignatt_min_accessible_inaccessible_margin",
                    runtime_config.get(
                        "translation_alignatt_min_accessible_inaccessible_margin"
                    ),
                ),
                "output_dir": str(output_dir),
                "num_inputs": manifest.get("num_inputs"),
                "chunk_ms": runtime_config.get("chunk_ms"),
                "bleu": scores.get("BLEU"),
                "chrf": scores.get("CHRF"),
                "xcometxl": scores.get("XCOMETXL"),
                "longyaal_cu_ms": scores.get("LongYAAL CU"),
                "longyaal_ca_ms": scores.get("LongYAAL CA"),
            }
        )
    return rows


def policy_label(row: dict[str, Any]) -> str:
    if row.get("policy") == "alignatt":
        tag = str(row.get("tag") or "")
        if tag == "alignatt" or tag == "alignatt_b1_top8":
            return "\\textsc{AlignAtt} baseline"
        border = row.get("alignatt_border_margin")
        top_k = row.get("alignatt_top_k_heads")
        min_mass = row.get("alignatt_min_source_mass")
        if min_mass is not None:
            frontier_min = row.get("alignatt_frontier_min_inaccessible_mass")
            max_inaccessible = row.get("alignatt_max_inaccessible_source_mass")
            margin = row.get("alignatt_min_accessible_inaccessible_margin")
            extras = []
            if frontier_min not in (None, 0, 0.0):
                extras.append(f"f={float(frontier_min):.2f}")
            if max_inaccessible not in (None, 1, 1.0):
                extras.append(f"u={float(max_inaccessible):.2f}")
            if margin not in (None, -1, -1.0):
                extras.append(f"d={float(margin):.2f}")
            extra_label = "," + ",".join(extras) if extras else ""
            return (
                "\\textsc{AlignAtt} tuned "
                f"($b={int(border)},k={int(top_k)},m={fmt_compact_float(float(min_mass))}{extra_label}$)"
            )
        return "\\textsc{AlignAtt} tuned"
    cutoff = row.get("cutoff_units")
    return f"cut-last-{int(cutoff)}"


def fmt(value: Any, precision: int = 2) -> str:
    if value is None:
        return "--"
    return f"{float(value):.{precision}f}"


def fmt_compact_float(value: float) -> str:
    return f"{float(value):.4f}".rstrip("0").rstrip(".")


def fmt_seconds(value_ms: Any) -> str:
    if value_ms is None:
        return "--"
    return f"{float(value_ms) / 1000.0:.2f}"


def sort_key(row: dict[str, Any]) -> tuple[int, int]:
    if row.get("tag") == "alignatt" or row.get("policy") == "alignatt":
        return (0, 0)
    return (1, int(row.get("cutoff_units") or 0))


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _delta(value: Any, baseline: Any) -> float | None:
    current = _as_float(value)
    base = _as_float(baseline)
    if current is None or base is None:
        return None
    return current - base


def _best_row(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    higher_is_better: bool,
) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get(metric) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row[metric])) if higher_is_better else min(
        candidates,
        key=lambda row: float(row[metric]),
    )


def _dominates(candidate: dict[str, Any], target: dict[str, Any]) -> bool:
    required = ("bleu", "chrf", "longyaal_cu_ms", "longyaal_ca_ms")
    if any(candidate.get(metric) is None or target.get(metric) is None for metric in required):
        return False
    quality_not_worse = (
        float(candidate["bleu"]) >= float(target["bleu"])
        and float(candidate["chrf"]) >= float(target["chrf"])
    )
    latency_not_worse = (
        float(candidate["longyaal_cu_ms"]) <= float(target["longyaal_cu_ms"])
        and float(candidate["longyaal_ca_ms"]) <= float(target["longyaal_ca_ms"])
    )
    strict = (
        float(candidate["bleu"]) > float(target["bleu"])
        or float(candidate["chrf"]) > float(target["chrf"])
        or float(candidate["longyaal_cu_ms"]) < float(target["longyaal_cu_ms"])
        or float(candidate["longyaal_ca_ms"]) < float(target["longyaal_ca_ms"])
    )
    return quality_not_worse and latency_not_worse and strict


def _pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if not any(other is not row and _dominates(other, row) for other in rows)
    ]


def _load_resegmented_instances(row: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(str(row["output_dir"])) / "instances.resegmented.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _word_len(text: str) -> int:
    return len(text.split(" ")) if text else 0


def _yaal_for_instance(instance: dict[str, Any], *, timestamp_key: str) -> float | None:
    delays = instance.get(timestamp_key)
    source_length = instance.get("source_length")
    if not delays or source_length is None or float(source_length) <= 0.0:
        return None

    recording_end = instance.get("time_to_recording_end")
    recording_end = float(recording_end) if recording_end is not None else math.inf
    if float(delays[0]) >= recording_end:
        return None

    target_length = _word_len(str(instance.get("reference") or ""))
    gamma = max(len(delays), target_length) / float(source_length)
    if gamma <= 0.0:
        return None

    total = 0.0
    tau = 0
    for token_index, delay in enumerate(delays):
        delay = float(delay)
        if delay >= recording_end:
            break
        total += delay - token_index / gamma
        tau = token_index + 1
    return total / tau if tau > 0 else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _corpus_quality(instances: list[dict[str, Any]], metric: str) -> float | None:
    if not instances:
        return None
    try:
        from sacrebleu.metrics import BLEU, CHRF
    except Exception:
        return None

    predictions = [str(instance.get("prediction") or "") for instance in instances]
    references = [[str(instance.get("reference") or "") for instance in instances]]
    if metric == "bleu":
        return float(BLEU(tokenize="13a").corpus_score(predictions, references).score)
    if metric == "chrf":
        return float(CHRF().corpus_score(predictions, references).score)
    raise ValueError(f"Unknown corpus metric: {metric}")


def _sentence_chrf(instance: dict[str, Any]) -> float | None:
    try:
        from sacrebleu.metrics import CHRF
    except Exception:
        return None
    return float(
        CHRF().sentence_score(
            str(instance.get("prediction") or ""),
            [str(instance.get("reference") or "")],
        ).score
    )


def _latency_mean(instances: list[dict[str, Any]], *, timestamp_key: str) -> float | None:
    values = [
        value
        for value in (
            _yaal_for_instance(instance, timestamp_key=timestamp_key)
            for instance in instances
        )
        if value is not None
    ]
    return _mean(values)


def _quantile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _ci95(values: list[float]) -> list[float | None]:
    ordered = sorted(values)
    return [_quantile(ordered, 0.025), _quantile(ordered, 0.975)]


def _paired_segment_counts(
    alignatt_instances: list[dict[str, Any]],
    cutoff_instances: list[dict[str, Any]],
) -> dict[str, Any]:
    n = min(len(alignatt_instances), len(cutoff_instances))
    counts = {
        "segments": n,
        "alignatt_higher_sentence_chrf": 0,
        "alignatt_lower_cu_yaal": 0,
        "alignatt_lower_ca_yaal": 0,
        "alignatt_higher_chrf_and_lower_cu": 0,
        "alignatt_higher_chrf_and_lower_ca": 0,
    }
    chrf_deltas: list[float] = []
    cu_deltas: list[float] = []
    ca_deltas: list[float] = []
    for alignatt_instance, cutoff_instance in zip(
        alignatt_instances[:n],
        cutoff_instances[:n],
    ):
        alignatt_chrf = _sentence_chrf(alignatt_instance)
        cutoff_chrf = _sentence_chrf(cutoff_instance)
        alignatt_cu = _yaal_for_instance(alignatt_instance, timestamp_key="emission_cu")
        cutoff_cu = _yaal_for_instance(cutoff_instance, timestamp_key="emission_cu")
        alignatt_ca = _yaal_for_instance(alignatt_instance, timestamp_key="emission_ca")
        cutoff_ca = _yaal_for_instance(cutoff_instance, timestamp_key="emission_ca")

        chrf_win = (
            alignatt_chrf is not None
            and cutoff_chrf is not None
            and alignatt_chrf > cutoff_chrf
        )
        cu_win = alignatt_cu is not None and cutoff_cu is not None and alignatt_cu < cutoff_cu
        ca_win = alignatt_ca is not None and cutoff_ca is not None and alignatt_ca < cutoff_ca

        counts["alignatt_higher_sentence_chrf"] += int(chrf_win)
        counts["alignatt_lower_cu_yaal"] += int(cu_win)
        counts["alignatt_lower_ca_yaal"] += int(ca_win)
        counts["alignatt_higher_chrf_and_lower_cu"] += int(chrf_win and cu_win)
        counts["alignatt_higher_chrf_and_lower_ca"] += int(chrf_win and ca_win)
        if alignatt_chrf is not None and cutoff_chrf is not None:
            chrf_deltas.append(alignatt_chrf - cutoff_chrf)
        if alignatt_cu is not None and cutoff_cu is not None:
            cu_deltas.append(alignatt_cu - cutoff_cu)
        if alignatt_ca is not None and cutoff_ca is not None:
            ca_deltas.append(alignatt_ca - cutoff_ca)

    counts["mean_sentence_chrf_delta"] = _mean(chrf_deltas)
    counts["mean_cu_yaal_delta_ms"] = _mean(cu_deltas)
    counts["mean_ca_yaal_delta_ms"] = _mean(ca_deltas)
    return counts


def _sample_instances(
    instances: list[dict[str, Any]],
    indices: list[int],
) -> list[dict[str, Any]]:
    return [instances[index] for index in indices]


def _paired_bootstrap(
    alignatt_instances: list[dict[str, Any]],
    cutoff_instances: list[dict[str, Any]],
    *,
    samples: int,
) -> dict[str, Any]:
    n = min(len(alignatt_instances), len(cutoff_instances))
    if n == 0 or samples <= 0:
        return {}

    alignatt_instances = alignatt_instances[:n]
    cutoff_instances = cutoff_instances[:n]
    rng = random.Random(BOOTSTRAP_SEED)
    deltas: dict[str, list[float]] = {
        "bleu": [],
        "chrf": [],
        "cu_yaal_ms": [],
        "ca_yaal_ms": [],
    }
    bleu_cu_wins = 0
    chrf_cu_wins = 0
    bleu_ca_wins = 0
    chrf_ca_wins = 0
    not_dominated = 0
    valid_samples = 0

    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        alignatt_sample = _sample_instances(alignatt_instances, indices)
        cutoff_sample = _sample_instances(cutoff_instances, indices)
        values = {
            "bleu": (
                _corpus_quality(alignatt_sample, "bleu"),
                _corpus_quality(cutoff_sample, "bleu"),
            ),
            "chrf": (
                _corpus_quality(alignatt_sample, "chrf"),
                _corpus_quality(cutoff_sample, "chrf"),
            ),
            "cu_yaal_ms": (
                _latency_mean(alignatt_sample, timestamp_key="emission_cu"),
                _latency_mean(cutoff_sample, timestamp_key="emission_cu"),
            ),
            "ca_yaal_ms": (
                _latency_mean(alignatt_sample, timestamp_key="emission_ca"),
                _latency_mean(cutoff_sample, timestamp_key="emission_ca"),
            ),
        }
        if any(left is None or right is None for left, right in values.values()):
            continue

        valid_samples += 1
        sample_deltas = {
            metric: float(left) - float(right)
            for metric, (left, right) in values.items()
        }
        for metric, delta in sample_deltas.items():
            deltas[metric].append(delta)

        bleu_cu_wins += int(
            sample_deltas["bleu"] > 0.0 and sample_deltas["cu_yaal_ms"] < 0.0
        )
        chrf_cu_wins += int(
            sample_deltas["chrf"] > 0.0 and sample_deltas["cu_yaal_ms"] < 0.0
        )
        bleu_ca_wins += int(
            sample_deltas["bleu"] > 0.0 and sample_deltas["ca_yaal_ms"] < 0.0
        )
        chrf_ca_wins += int(
            sample_deltas["chrf"] > 0.0 and sample_deltas["ca_yaal_ms"] < 0.0
        )
        cutoff_dominates = (
            sample_deltas["bleu"] <= 0.0
            and sample_deltas["chrf"] <= 0.0
            and sample_deltas["cu_yaal_ms"] >= 0.0
            and sample_deltas["ca_yaal_ms"] >= 0.0
            and any(delta != 0.0 for delta in sample_deltas.values())
        )
        not_dominated += int(not cutoff_dominates)

    if valid_samples == 0:
        return {}

    return {
        "samples": valid_samples,
        "delta_ci95": {
            metric: _ci95(metric_deltas)
            for metric, metric_deltas in deltas.items()
        },
        "prob_alignatt_bleu_and_cu_win": bleu_cu_wins / valid_samples,
        "prob_alignatt_chrf_and_cu_win": chrf_cu_wins / valid_samples,
        "prob_alignatt_bleu_and_ca_win": bleu_ca_wins / valid_samples,
        "prob_alignatt_chrf_and_ca_win": chrf_ca_wins / valid_samples,
        "prob_alignatt_not_dominated": not_dominated / valid_samples,
    }


def _primary_alignatt_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    front = _pareto_front(rows)
    pareto_alignatt = [row for row in front if row.get("policy") == "alignatt"]
    if pareto_alignatt:
        return max(pareto_alignatt, key=lambda row: float(row.get("bleu") or -1.0))
    alignatt_rows = [row for row in rows if row.get("policy") == "alignatt"]
    if not alignatt_rows:
        return None
    return max(alignatt_rows, key=lambda row: float(row.get("bleu") or -1.0))


def _metric_frontier(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    latency_metric: str,
) -> list[dict[str, Any]]:
    points = sorted(
        [
            row
            for row in rows
            if row.get("policy") == "cut_last_target_units"
            and row.get(metric) is not None
            and row.get(latency_metric) is not None
        ],
        key=lambda row: float(row[latency_metric]),
    )
    frontier: list[dict[str, Any]] = []
    best_quality = -math.inf
    for row in points:
        quality = float(row[metric])
        if quality > best_quality:
            frontier.append(row)
            best_quality = quality
    return frontier


def _interpolate_quality(
    lower: dict[str, Any],
    upper: dict[str, Any],
    *,
    metric: str,
    latency_metric: str,
    target_latency: float,
) -> float:
    lower_latency = float(lower[latency_metric])
    upper_latency = float(upper[latency_metric])
    lower_quality = float(lower[metric])
    upper_quality = float(upper[metric])
    if upper_latency == lower_latency:
        return max(lower_quality, upper_quality)
    ratio = (target_latency - lower_latency) / (upper_latency - lower_latency)
    return lower_quality + ratio * (upper_quality - lower_quality)


def _static_frontier_gap(
    rows: list[dict[str, Any]],
    alignatt: dict[str, Any],
    *,
    metric: str,
    latency_metric: str,
) -> dict[str, Any] | None:
    target_latency = alignatt.get(latency_metric)
    target_quality = alignatt.get(metric)
    if target_latency is None or target_quality is None:
        return None
    frontier = _metric_frontier(rows, metric=metric, latency_metric=latency_metric)
    if len(frontier) < 2:
        return None
    target_latency = float(target_latency)
    for lower, upper in zip(frontier, frontier[1:]):
        if float(lower[latency_metric]) <= target_latency <= float(upper[latency_metric]):
            interpolated = _interpolate_quality(
                lower,
                upper,
                metric=metric,
                latency_metric=latency_metric,
                target_latency=target_latency,
            )
            return {
                "metric": metric,
                "latency_metric": latency_metric,
                "alignatt_tag": alignatt.get("tag"),
                "target_latency_ms": target_latency,
                "alignatt_quality": float(target_quality),
                "interpolated_static_quality": interpolated,
                "gap": float(target_quality) - interpolated,
                "bracket_tags": [lower.get("tag"), upper.get("tag")],
            }
    return None


def _robustness_analysis(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    ordered = sorted(rows, key=sort_key)
    alignatt = _primary_alignatt_row(ordered)
    if alignatt is None:
        return {}
    alignatt_instances = _load_resegmented_instances(alignatt)
    if not alignatt_instances:
        return {"primary_alignatt_tag": alignatt.get("tag")}

    cutoffs = [
        row
        for row in ordered
        if row.get("policy") == "cut_last_target_units"
    ]
    paired: dict[str, Any] = {}
    for cutoff in cutoffs:
        cutoff_instances = _load_resegmented_instances(cutoff)
        if not cutoff_instances:
            continue
        tag = str(cutoff.get("tag"))
        paired[tag] = {
            "cutoff_units": cutoff.get("cutoff_units"),
            "delta_vs_cutoff": {
                "bleu": _delta(alignatt.get("bleu"), cutoff.get("bleu")),
                "chrf": _delta(alignatt.get("chrf"), cutoff.get("chrf")),
                "longyaal_cu_ms": _delta(
                    alignatt.get("longyaal_cu_ms"),
                    cutoff.get("longyaal_cu_ms"),
                ),
                "longyaal_ca_ms": _delta(
                    alignatt.get("longyaal_ca_ms"),
                    cutoff.get("longyaal_ca_ms"),
                ),
            },
            "paired_segment_counts": _paired_segment_counts(
                alignatt_instances,
                cutoff_instances,
            ),
            "bootstrap": _paired_bootstrap(
                alignatt_instances,
                cutoff_instances,
                samples=bootstrap_samples,
            ),
        }

    frontier_gaps = [
        gap
        for metric in ("bleu", "chrf")
        for latency_metric in ("longyaal_cu_ms", "longyaal_ca_ms")
        for gap in [_static_frontier_gap(ordered, alignatt, metric=metric, latency_metric=latency_metric)]
        if gap is not None
    ]
    not_dominated_probs = [
        diagnostic.get("bootstrap", {}).get("prob_alignatt_not_dominated")
        for diagnostic in paired.values()
        if diagnostic.get("bootstrap", {}).get("prob_alignatt_not_dominated") is not None
    ]
    return {
        "primary_alignatt_tag": alignatt.get("tag"),
        "primary_alignatt_label": policy_label(alignatt),
        "num_resegmented_instances": len(alignatt_instances),
        "min_bootstrap_prob_alignatt_not_dominated_by_any_cutoff": min(
            not_dominated_probs
        )
        if not_dominated_probs
        else None,
        "frontier_gaps": frontier_gaps,
        "paired_cutoff_diagnostics": paired,
    }


def _budget_analysis(
    rows: list[dict[str, Any]],
    alignatt: dict[str, Any] | None,
    *,
    cu_budget_ms: float,
) -> dict[str, Any]:
    if alignatt is None:
        return {"cu_budget_ms": cu_budget_ms}

    cutoffs = [
        row
        for row in rows
        if row.get("policy") == "cut_last_target_units"
        and row.get("longyaal_cu_ms") is not None
    ]
    cutoffs_under_budget = [
        row for row in cutoffs if float(row["longyaal_cu_ms"]) <= cu_budget_ms
    ]
    cutoffs_over_budget = [
        row for row in cutoffs if float(row["longyaal_cu_ms"]) > cu_budget_ms
    ]
    best_bleu_under_budget = _best_row(
        cutoffs_under_budget,
        "bleu",
        higher_is_better=True,
    )
    best_chrf_under_budget = _best_row(
        cutoffs_under_budget,
        "chrf",
        higher_is_better=True,
    )
    closest_over_budget = (
        min(cutoffs_over_budget, key=lambda row: float(row["longyaal_cu_ms"]))
        if cutoffs_over_budget
        else None
    )
    alignatt_cu = alignatt.get("longyaal_cu_ms")

    def _beats_all_budget_cutoffs(metric: str) -> bool | None:
        value = alignatt.get(metric)
        if value is None or not cutoffs_under_budget:
            return None
        cutoff_values = [row.get(metric) for row in cutoffs_under_budget]
        if any(cutoff_value is None for cutoff_value in cutoff_values):
            return None
        return all(float(value) > float(cutoff_value) for cutoff_value in cutoff_values)

    return {
        "cu_budget_ms": cu_budget_ms,
        "alignatt_tag": alignatt.get("tag"),
        "alignatt_under_budget": (
            float(alignatt_cu) <= cu_budget_ms if alignatt_cu is not None else None
        ),
        "cutoff_tags_under_budget": [row.get("tag") for row in cutoffs_under_budget],
        "cutoff_tags_over_budget": [row.get("tag") for row in cutoffs_over_budget],
        "best_cutoff_under_budget_by_bleu": best_bleu_under_budget,
        "best_cutoff_under_budget_by_chrf": best_chrf_under_budget,
        "closest_cutoff_over_budget": closest_over_budget,
        "alignatt_beats_all_cutoffs_under_budget_bleu": _beats_all_budget_cutoffs(
            "bleu"
        ),
        "alignatt_beats_all_cutoffs_under_budget_chrf": _beats_all_budget_cutoffs(
            "chrf"
        ),
        "delta_vs_best_budget_bleu": _delta(
            alignatt.get("bleu"),
            best_bleu_under_budget.get("bleu") if best_bleu_under_budget else None,
        ),
        "delta_vs_best_budget_chrf": _delta(
            alignatt.get("chrf"),
            best_chrf_under_budget.get("chrf") if best_chrf_under_budget else None,
        ),
        "delta_cu_vs_best_budget_bleu_ms": _delta(
            alignatt_cu,
            best_bleu_under_budget.get("longyaal_cu_ms") if best_bleu_under_budget else None,
        ),
    }


def _baseline_alignatt_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for tag in ("alignatt_b1_top8", "alignatt"):
        for row in rows:
            if row.get("tag") == tag:
                return row
    return next((row for row in rows if row.get("policy") == "alignatt"), None)


def _selected_table_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=sort_key)
    selected: list[dict[str, Any]] = []
    baseline = _baseline_alignatt_row(ordered)
    if baseline is not None:
        selected.append(baseline)
    pareto_alignatt = [
        row for row in _pareto_front(ordered) if row.get("policy") == "alignatt"
    ]
    if pareto_alignatt:
        tuned = max(pareto_alignatt, key=lambda row: float(row.get("bleu") or -1.0))
        if baseline is None or tuned.get("tag") != baseline.get("tag"):
            selected.append(tuned)
    selected.extend(row for row in ordered if row.get("policy") == "cut_last_target_units")
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in selected:
        tag = str(row.get("tag"))
        if tag in seen:
            continue
        seen.add(tag)
        deduped.append(row)
    return deduped


def build_json_payload(
    rows: list[dict[str, Any]],
    *,
    output_root: Path,
    bootstrap_samples: int,
    cu_budget_ms: float,
) -> dict[str, Any]:
    ordered = sorted(rows, key=sort_key)
    alignatt = _baseline_alignatt_row(ordered)
    front = _pareto_front(ordered)
    enriched_rows: list[dict[str, Any]] = []
    for row in ordered:
        enriched = dict(row)
        enriched["pareto_front_bleu_chrf_latency"] = any(
            row.get("tag") == front_row.get("tag") for front_row in front
        )
        if alignatt is not None:
            enriched["delta_vs_alignatt"] = {
                "bleu": _delta(row.get("bleu"), alignatt.get("bleu")),
                "chrf": _delta(row.get("chrf"), alignatt.get("chrf")),
                "xcometxl": _delta(row.get("xcometxl"), alignatt.get("xcometxl")),
                "longyaal_cu_ms": _delta(
                    row.get("longyaal_cu_ms"),
                    alignatt.get("longyaal_cu_ms"),
                ),
                "longyaal_ca_ms": _delta(
                    row.get("longyaal_ca_ms"),
                    alignatt.get("longyaal_ca_ms"),
                ),
            }
            quality_not_worse = (
                row.get("bleu") is not None
                and alignatt.get("bleu") is not None
                and float(row["bleu"]) >= float(alignatt["bleu"])
                and row.get("chrf") is not None
                and alignatt.get("chrf") is not None
                and float(row["chrf"]) >= float(alignatt["chrf"])
            )
            latency_not_worse = (
                row.get("longyaal_cu_ms") is not None
                and alignatt.get("longyaal_cu_ms") is not None
                and float(row["longyaal_cu_ms"]) <= float(alignatt["longyaal_cu_ms"])
                and row.get("longyaal_ca_ms") is not None
                and alignatt.get("longyaal_ca_ms") is not None
                and float(row["longyaal_ca_ms"]) <= float(alignatt["longyaal_ca_ms"])
            )
            enriched["dominates_alignatt_bleu_chrf_latency"] = bool(
                quality_not_worse
                and latency_not_worse
                and row.get("tag") != alignatt.get("tag")
            )
        enriched_rows.append(enriched)

    analysis = {
        "best_bleu": _best_row(ordered, "bleu", higher_is_better=True),
        "best_chrf": _best_row(ordered, "chrf", higher_is_better=True),
        "lowest_longyaal_cu": _best_row(
            ordered,
            "longyaal_cu_ms",
            higher_is_better=False,
        ),
        "lowest_longyaal_ca": _best_row(
            ordered,
            "longyaal_ca_ms",
            higher_is_better=False,
        ),
        "rows_dominating_alignatt_bleu_chrf_latency": [
            row["tag"]
            for row in enriched_rows
            if row.get("dominates_alignatt_bleu_chrf_latency")
        ],
        "pareto_front_tags": [row["tag"] for row in front],
        "pareto_alignatt_tags": [
            row["tag"] for row in front if row.get("policy") == "alignatt"
        ],
        "table_tags": [row["tag"] for row in _selected_table_rows(ordered)],
        "robustness": _robustness_analysis(
            ordered,
            bootstrap_samples=bootstrap_samples,
        ),
        "cu_budget": _budget_analysis(
            ordered,
            alignatt,
            cu_budget_ms=cu_budget_ms,
        ),
    }
    return {"output_root": str(output_root), "rows": enriched_rows, "analysis": analysis}


def render_tex(rows: list[dict[str, Any]], *, output_root: Path) -> str:
    ordered = _selected_table_rows(rows)
    has_xcomet = any(row.get("xcometxl") is not None for row in ordered)
    num_inputs = next((row.get("num_inputs") for row in ordered if row.get("num_inputs")), None)
    chunk_ms = next((row.get("chunk_ms") for row in ordered if row.get("chunk_ms")), None)
    if num_inputs == 1:
        caption_title = "Single-clip quality--latency diagnostic for MT acceptance policies."
    else:
        caption_title = "Quality--latency trade-off for MT acceptance policies."
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{l r r r r r}" if has_xcomet else "\\begin{tabular}{l r r r r}",
        "\\toprule",
        (
            "Policy & BLEU $\\uparrow$ & chrF $\\uparrow$ & XCOMET-XL $\\uparrow$ & CU (s) $\\downarrow$ & CA (s) $\\downarrow$ \\\\"
            if has_xcomet
            else "Policy & BLEU $\\uparrow$ & chrF $\\uparrow$ & CU (s) $\\downarrow$ & CA (s) $\\downarrow$ \\\\"
        ),
        "\\midrule",
    ]
    for row in ordered:
        if has_xcomet:
            lines.append(
                f"{policy_label(row)} & {fmt(row.get('bleu'))} & {fmt(row.get('chrf'))} & "
                f"{fmt(row.get('xcometxl'), precision=3)} & {fmt_seconds(row.get('longyaal_cu_ms'))} & "
                f"{fmt_seconds(row.get('longyaal_ca_ms'))} \\\\"
            )
        else:
            lines.append(
                f"{policy_label(row)} & {fmt(row.get('bleu'))} & {fmt(row.get('chrf'))} & "
                f"{fmt_seconds(row.get('longyaal_cu_ms'))} & "
                f"{fmt_seconds(row.get('longyaal_ca_ms'))} \\\\"
            )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            f"\\caption{{\\textbf{{{caption_title}}} Each row is a real streaming run with the same ASR, Gemma draft model, chunk size, and evaluation protocol; only the partial MT acceptance policy changes. Metrics are computed by OmniSTEval after long-form resegmentation.}}",
            "\\label{tab:mt-cutoff-quality-latency}",
            "\\end{table}",
            "",
        ]
    )
    if num_inputs is not None or chunk_ms is not None:
        lines.insert(
            -4,
            f"% source={output_root}; num_inputs={num_inputs}; chunk_ms={chunk_ms}",
        )
    return "\n".join(lines)


def _frontier_gap_lookup(robustness: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(gap.get("metric")), str(gap.get("latency_metric"))): gap
        for gap in robustness.get("frontier_gaps", [])
    }


def render_frontier_gap_tex(payload: dict[str, Any]) -> str:
    robustness = payload.get("analysis", {}).get("robustness", {}) or {}
    gap_by_key = _frontier_gap_lookup(robustness)
    primary_label = robustness.get("primary_alignatt_label", "\\textsc{AlignAtt}")
    num_instances = robustness.get("num_resegmented_instances")
    min_prob = robustness.get("min_bootstrap_prob_alignatt_not_dominated_by_any_cutoff")
    rows = [
        ("BLEU at matched CU", gap_by_key.get(("bleu", "longyaal_cu_ms"))),
        ("BLEU at matched CA", gap_by_key.get(("bleu", "longyaal_ca_ms"))),
        ("chrF at matched CU", gap_by_key.get(("chrf", "longyaal_cu_ms"))),
        ("chrF at matched CA", gap_by_key.get(("chrf", "longyaal_ca_ms"))),
    ]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{l r r r}",
        "\\toprule",
        "Diagnostic & AlignAtt & Static frontier & Gap \\\\",
        "\\midrule",
    ]
    for label, gap in rows:
        if not gap:
            continue
        lines.append(
            f"{label} & {fmt(gap.get('alignatt_quality'))} & "
            f"{fmt(gap.get('interpolated_static_quality'))} & "
            f"{fmt(gap.get('gap'))} \\\\"
        )
    if min_prob is not None:
        lines.extend(
            [
                "\\midrule",
                (
                    "Min. bootstrap $P$(not dominated) & "
                    f"\\multicolumn{{3}}{{r}}{{{100.0 * float(min_prob):.1f}\\%}} \\\\"
                ),
            ]
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            (
                "\\caption{\\textbf{Robustness diagnostic against the static cutoff frontier.} "
                f"The primary attention point is {primary_label}. "
                "Static-frontier values linearly interpolate the best fixed-cutoff "
                "quality--latency curve at the AlignAtt latency; positive gaps mean "
                "the attention policy sits above that curve. "
                f"The bootstrap diagnostic resamples the {num_instances} resegmented "
                "segments and reports the weakest probability, over all fixed cutoffs, "
                "that no cutoff dominates the attention point in BLEU, chrF, CU, and CA.}"
            ),
            "\\label{tab:mt-cutoff-frontier-gap}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = discover_rows(args.output_root)
    if not rows:
        raise FileNotFoundError(
            f"No evaluated policy outputs found under {args.output_root}"
        )
    report_dir = args.paper_generated_dir or args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    tex_path = report_dir / args.tex_name
    frontier_tex_path = report_dir / args.frontier_tex_name
    json_path = report_dir / args.json_name
    payload = build_json_payload(
        rows,
        output_root=args.output_root,
        bootstrap_samples=args.bootstrap_samples,
        cu_budget_ms=args.cu_budget_ms,
    )
    tex_path.write_text(render_tex(rows, output_root=args.output_root), encoding="utf-8")
    frontier_tex_path.write_text(render_frontier_gap_tex(payload), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "tex": str(tex_path),
                "frontier_tex": str(frontier_tex_path),
                "json": str(json_path),
                **payload,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
