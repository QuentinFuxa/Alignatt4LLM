#!/usr/bin/env python3
"""Compare ASR backends as a quality/runtime trade-off.

The figure is intentionally system-facing: final ASR WER and wall-clock
real-time factor (RTF) are shown in aligned panels.  The RTF=1 line marks
whether an ASR backend can keep up with real-time audio.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np


DEFAULT_QWEN_SUMMARY = Path("outputs/asr_compare_enen_21audio_20260421/qwen_forced__summary.json")
DEFAULT_QWEN_DIR = Path("outputs/asr_compare_enen_21audio_20260421/qwen_forced")
DEFAULT_QWEN_EVAL = Path("outputs/asr_compare_enen_21audio_20260421/qwen_forced_eval/evaluation.json")
DEFAULT_VOXTRAL_RESULTS = Path(
    "outputs/precomputed_asr_voxtral/voxtral_wer_cer_results.json"
)
DEFAULT_VOXTRAL_DELAY480_EVAL = Path("outputs/comparaison_asr/voxtral_delay480_eval/evaluation.json")
DEFAULT_GEMMA_MANIFEST = Path("outputs/gemma_e4b_asr_mcif_la_full_20260424/manifest.json")
DEFAULT_GEMMA_EVAL = Path("outputs/gemma_e4b_asr_mcif_la_full_20260424/eval/evaluation.json")


@dataclass(frozen=True)
class TalkMetric:
    talk_id: str
    wer: float
    cer: float
    ref_words: float
    rtf: float


@dataclass
class MethodSummary:
    key: str
    label: str
    short_label: str
    color: str
    marker: str
    talk_metrics: list[TalkMetric]
    weighted_wer: float
    mean_wer: float
    mean_cer: float
    mean_rtf: float
    median_rtf: float
    ci_wer: tuple[float, float]
    ci_rtf: tuple[float, float]
    wer_delta_vs_qwen: float | None = None
    rtf_ratio_vs_qwen: float | None = None
    wins_vs_qwen: int | None = None
    losses_vs_qwen: int | None = None
    long_yaal_cu_ms: float | None = None
    long_yaal_ca_ms: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-summary", type=Path, default=DEFAULT_QWEN_SUMMARY)
    parser.add_argument("--qwen-dir", type=Path, default=DEFAULT_QWEN_DIR)
    parser.add_argument("--qwen-eval", type=Path, default=DEFAULT_QWEN_EVAL)
    parser.add_argument("--voxtral-results", type=Path, default=DEFAULT_VOXTRAL_RESULTS)
    parser.add_argument("--voxtral-delay480-eval", type=Path, default=DEFAULT_VOXTRAL_DELAY480_EVAL)
    parser.add_argument("--gemma-manifest", type=Path, default=DEFAULT_GEMMA_MANIFEST)
    parser.add_argument("--gemma-eval", type=Path, default=DEFAULT_GEMMA_EVAL)
    parser.add_argument("--output-stem", default="comparaison_asr")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/comparaison_asr"))
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    parser.add_argument("--ci", type=float, default=90.0)
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_longyaal_scores(path: Path) -> tuple[float | None, float | None]:
    if not path.exists():
        return None, None
    payload = load_json(path)
    scores = dict(payload.get("contract_scores") or {})
    cu = scores.get("LongYAAL CU")
    ca = scores.get("LongYAAL CA")
    return (None if cu is None else float(cu), None if ca is None else float(ca))


def stem_from_wav_name(wav_name: str) -> str:
    return Path(str(wav_name)).stem


def load_qwen_metrics(summary_path: Path, qwen_dir: Path) -> dict[str, TalkMetric]:
    summary = load_json(summary_path)
    rows = summary.get("rows") or []
    metrics: dict[str, TalkMetric] = {}
    for row in rows:
        talk_id = stem_from_wav_name(str(row["wav_name"]))
        detail_path = qwen_dir / f"{talk_id}.json"
        detail = load_json(detail_path)
        detail_metrics = dict(detail.get("metrics") or {})
        metrics[talk_id] = TalkMetric(
            talk_id=talk_id,
            wer=float(row["wer"]),
            cer=float(row["cer"]),
            ref_words=float(detail_metrics["reference_word_count"]),
            rtf=float(row["rtf_wallclock"]),
        )
    return metrics


def load_voxtral_metrics(results_path: Path, delay_key: str) -> dict[str, TalkMetric]:
    payload = load_json(results_path)
    per_talk = dict(payload.get("per_talk") or {})
    metrics: dict[str, TalkMetric] = {}
    for talk_id, delay_payloads in per_talk.items():
        if delay_key not in delay_payloads:
            continue
        row = dict(delay_payloads[delay_key])
        metrics[str(talk_id)] = TalkMetric(
            talk_id=str(talk_id),
            wer=float(row["wer"]),
            cer=float(row["cer"]),
            ref_words=float(row["ref_words"]),
            rtf=float(row["rtf"]),
        )
    return metrics


def load_gemma_metrics(manifest_path: Path) -> dict[str, TalkMetric]:
    payload = load_json(manifest_path)
    metrics: dict[str, TalkMetric] = {}
    for row in payload.get("rows") or []:
        talk_id = stem_from_wav_name(str(row["wav_name"]))
        row_metrics = dict(row.get("metrics") or {})
        metrics[talk_id] = TalkMetric(
            talk_id=talk_id,
            wer=float(row_metrics["wer"]),
            cer=float(row_metrics["cer"]),
            ref_words=float(row_metrics["reference_word_count"]),
            rtf=float(row["rtf_wallclock"]),
        )
    return metrics


def compute_clip0_longyaal_cu_ms(resegmented_path: Path) -> float | None:
    """YAAL with impossible negative emissions clipped to 0 ms.

    Gemma E4B's raw OmniSTEval LongYAAL is polluted by prompt-leakage tokens
    resegmented into late utterances with negative emission times. This helper
    preserves OmniSTEval's long-form YAAL formula but clamps negative CU
    timestamps before averaging. Empty resegmented instances remain skipped,
    matching OmniSTEval's raw scorer.
    """
    if not resegmented_path.exists():
        return None

    instance_scores: list[float] = []
    with resegmented_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            delays = [max(0.0, float(value)) for value in (row.get("emission_cu") or [])]
            if not delays:
                continue
            source_length = float(row.get("source_length") or 0.0)
            if source_length <= 0.0:
                continue
            recording_end = row.get("time_to_recording_end")
            recording_end = float(recording_end) if recording_end is not None else float("inf")
            if delays[0] >= recording_end:
                continue

            target_length = len(str(row.get("reference") or "").split())
            gamma = max(len(delays), target_length) / source_length
            values: list[float] = []
            for idx, delay in enumerate(delays):
                if delay >= recording_end:
                    break
                values.append(delay - float(idx) / gamma)
            if values:
                instance_scores.append(float(np.mean(values)))

    return None if not instance_scores else float(np.mean(instance_scores))


def weighted_wer(metrics: list[TalkMetric]) -> float:
    numerator = sum(metric.wer * metric.ref_words for metric in metrics)
    denominator = sum(metric.ref_words for metric in metrics)
    if denominator <= 0:
        raise ValueError("Cannot compute weighted WER without reference words.")
    return numerator / denominator


def summarize_values(metrics: list[TalkMetric]) -> tuple[float, float, float, float]:
    wers = np.array([metric.wer for metric in metrics], dtype=np.float64)
    cers = np.array([metric.cer for metric in metrics], dtype=np.float64)
    rtfs = np.array([metric.rtf for metric in metrics], dtype=np.float64)
    return weighted_wer(metrics), float(wers.mean()), float(cers.mean()), float(rtfs.mean())


def bootstrap_ci(
    metrics: list[TalkMetric],
    *,
    samples: int,
    rng: np.random.Generator,
    ci: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    n_items = len(metrics)
    weights = np.array([metric.ref_words for metric in metrics], dtype=np.float64)
    wers = np.array([metric.wer for metric in metrics], dtype=np.float64)
    rtfs = np.array([metric.rtf for metric in metrics], dtype=np.float64)

    boot_wer = np.empty(samples, dtype=np.float64)
    boot_rtf = np.empty(samples, dtype=np.float64)
    for sample_idx in range(samples):
        indices = rng.integers(0, n_items, size=n_items)
        sample_weights = weights[indices]
        boot_wer[sample_idx] = float(np.sum(wers[indices] * sample_weights) / np.sum(sample_weights))
        boot_rtf[sample_idx] = float(np.mean(rtfs[indices]))

    alpha = (100.0 - float(ci)) / 2.0
    return (
        (float(np.percentile(boot_wer, alpha)), float(np.percentile(boot_wer, 100.0 - alpha))),
        (float(np.percentile(boot_rtf, alpha)), float(np.percentile(boot_rtf, 100.0 - alpha))),
    )


def build_method_summary(
    *,
    key: str,
    label: str,
    short_label: str,
    color: str,
    marker: str,
    metrics_by_talk: dict[str, TalkMetric],
    common_talks: list[str],
    bootstrap_samples: int,
    rng: np.random.Generator,
    ci: float,
) -> MethodSummary:
    metrics = [metrics_by_talk[talk_id] for talk_id in common_talks]
    corpus_wer, mean_wer, mean_cer, mean_rtf = summarize_values(metrics)
    ci_wer, ci_rtf = bootstrap_ci(metrics, samples=bootstrap_samples, rng=rng, ci=ci)
    return MethodSummary(
        key=key,
        label=label,
        short_label=short_label,
        color=color,
        marker=marker,
        talk_metrics=metrics,
        weighted_wer=corpus_wer,
        mean_wer=mean_wer,
        mean_cer=mean_cer,
        mean_rtf=mean_rtf,
        median_rtf=float(np.median([metric.rtf for metric in metrics])),
        ci_wer=ci_wer,
        ci_rtf=ci_rtf,
    )


def attach_qwen_relative_stats(
    method: MethodSummary,
    *,
    qwen: MethodSummary,
) -> MethodSummary:
    qwen_by_talk = {metric.talk_id: metric for metric in qwen.talk_metrics}
    deltas = [
        metric.wer - qwen_by_talk[metric.talk_id].wer
        for metric in method.talk_metrics
        if metric.talk_id in qwen_by_talk
    ]
    wins = sum(delta < 0.0 for delta in deltas)
    losses = sum(delta > 0.0 for delta in deltas)
    return MethodSummary(
        key=method.key,
        label=method.label,
        short_label=method.short_label,
        color=method.color,
        marker=method.marker,
        talk_metrics=method.talk_metrics,
        weighted_wer=method.weighted_wer,
        mean_wer=method.mean_wer,
        mean_cer=method.mean_cer,
        mean_rtf=method.mean_rtf,
        median_rtf=method.median_rtf,
        ci_wer=method.ci_wer,
        ci_rtf=method.ci_rtf,
        wer_delta_vs_qwen=method.weighted_wer - qwen.weighted_wer,
        rtf_ratio_vs_qwen=method.mean_rtf / qwen.mean_rtf,
        wins_vs_qwen=wins,
        losses_vs_qwen=losses,
        long_yaal_cu_ms=method.long_yaal_cu_ms,
        long_yaal_ca_ms=method.long_yaal_ca_ms,
    )


def write_tsv(path: Path, summaries: list[MethodSummary]) -> None:
    lines = [
        "\t".join(
            [
                "method",
                "weighted_wer",
                "weighted_wer_pct",
                "mean_wer",
                "mean_wer_pct",
                "mean_cer",
                "mean_rtf",
                "median_rtf",
                "ci90_wer_low",
                "ci90_wer_high",
                "ci90_wer_low_pct",
                "ci90_wer_high_pct",
                "ci90_rtf_low",
                "ci90_rtf_high",
                "wer_delta_vs_qwen",
                "wer_delta_vs_qwen_pts",
                "rtf_ratio_vs_qwen",
                "wins_vs_qwen",
                "losses_vs_qwen",
                "long_yaal_cu_ms",
                "long_yaal_ca_ms",
            ]
        )
    ]
    for summary in summaries:
        values = [
            summary.label,
            f"{summary.weighted_wer:.6f}",
            f"{summary.weighted_wer * 100.0:.2f}",
            f"{summary.mean_wer:.6f}",
            f"{summary.mean_wer * 100.0:.2f}",
            f"{summary.mean_cer:.6f}",
            f"{summary.mean_rtf:.6f}",
            f"{summary.median_rtf:.6f}",
            f"{summary.ci_wer[0]:.6f}",
            f"{summary.ci_wer[1]:.6f}",
            f"{summary.ci_wer[0] * 100.0:.2f}",
            f"{summary.ci_wer[1] * 100.0:.2f}",
            f"{summary.ci_rtf[0]:.6f}",
            f"{summary.ci_rtf[1]:.6f}",
            "" if summary.wer_delta_vs_qwen is None else f"{summary.wer_delta_vs_qwen:.6f}",
            "" if summary.wer_delta_vs_qwen is None else f"{summary.wer_delta_vs_qwen * 100.0:.2f}",
            "" if summary.rtf_ratio_vs_qwen is None else f"{summary.rtf_ratio_vs_qwen:.6f}",
            "" if summary.wins_vs_qwen is None else str(summary.wins_vs_qwen),
            "" if summary.losses_vs_qwen is None else str(summary.losses_vs_qwen),
            "" if summary.long_yaal_cu_ms is None else f"{summary.long_yaal_cu_ms:.3f}",
            "" if summary.long_yaal_ca_ms is None else f"{summary.long_yaal_ca_ms:.3f}",
        ]
        lines.append("\t".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summary_to_json(
    *,
    qwen_summary_path: Path,
    qwen_dir: Path,
    voxtral_results_path: Path,
    gemma_manifest_path: Path,
    gemma_eval_path: Path,
    common_talks: list[str],
    summaries: list[MethodSummary],
) -> dict[str, Any]:
    return {
        "inputs": {
            "qwen_summary": str(qwen_summary_path),
            "qwen_dir": str(qwen_dir),
            "voxtral_results": str(voxtral_results_path),
            "gemma_manifest": str(gemma_manifest_path),
            "gemma_eval": str(gemma_eval_path),
        },
        "common_talk_count": len(common_talks),
        "common_talks": common_talks,
        "methods": {
            summary.key: {
                "label": summary.label,
                "weighted_wer": summary.weighted_wer,
                "mean_wer": summary.mean_wer,
                "mean_cer": summary.mean_cer,
                "mean_rtf": summary.mean_rtf,
                "median_rtf": summary.median_rtf,
                "ci90_wer": list(summary.ci_wer),
                "ci90_rtf": list(summary.ci_rtf),
                "wer_delta_vs_qwen": summary.wer_delta_vs_qwen,
                "rtf_ratio_vs_qwen": summary.rtf_ratio_vs_qwen,
                "wins_vs_qwen": summary.wins_vs_qwen,
                "losses_vs_qwen": summary.losses_vs_qwen,
                "long_yaal_cu_ms": summary.long_yaal_cu_ms,
                "long_yaal_ca_ms": summary.long_yaal_ca_ms,
                "per_talk": [
                    {
                        "talk_id": metric.talk_id,
                        "wer": metric.wer,
                        "cer": metric.cer,
                        "ref_words": metric.ref_words,
                        "rtf": metric.rtf,
                    }
                    for metric in summary.talk_metrics
                ],
            }
            for summary in summaries
        },
    }


def plot_pareto(path_base: Path, summaries: list[MethodSummary], *, dpi: int) -> None:
    display_summaries = [
        summary
        for summary in summaries
        if summary.key in {"qwen_forced", "voxtral_delay480ms", "gemma_e4b_la"}
    ]
    if len(display_summaries) != 3:
        raise ValueError("The decision figure expects Qwen, Voxtral delay480, and Gemma summaries.")

    fig, (wer_ax, latency_ax, rtf_ax) = plt.subplots(
        ncols=3,
        sharey=True,
        figsize=(8.1, 3.0),
        gridspec_kw={"width_ratios": [1.0, 0.95, 0.95], "wspace": 0.08},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")

    row_labels = {
        "qwen_forced": "Qwen3 ASR + aligner",
        "voxtral_delay480ms": "Voxtral Realtime 4B",
        "gemma_e4b_la": "Gemma E4B ASR LA",
    }
    y_positions = np.array([1.1, 0.55, 0.0], dtype=np.float64)

    for ax in (wer_ax, latency_ax, rtf_ax):
        ax.set_facecolor("white")
        ax.grid(True, which="major", axis="x", color="#d4d4d4", linewidth=0.75)
        ax.grid(False, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        for y in y_positions:
            ax.axhline(y, color="#ededed", linewidth=0.8, zorder=0)

    rtf_ax.axvspan(1.0, 1.52, color="#f4d7d7", alpha=0.48, zorder=0)
    rtf_ax.axvline(1.0, color="#8a1f1f", linestyle=(0, (4, 4)), linewidth=1.15, zorder=1)
    rtf_ax.text(
        1.015,
        y_positions[0] - 0.22,
        "RTF > 1",
        ha="left",
        va="center",
        fontsize=8.5,
        color="#7f1d1d",
    )

    for y, summary in zip(y_positions, display_summaries):
        wer_x = summary.weighted_wer * 100.0
        wer_low = summary.ci_wer[0] * 100.0
        wer_high = summary.ci_wer[1] * 100.0
        rtf_x = summary.mean_rtf
        latency_x = None if summary.long_yaal_cu_ms is None else summary.long_yaal_cu_ms / 1000.0

        wer_ax.errorbar(
            wer_x,
            y,
            xerr=np.array([[wer_x - wer_low], [wer_high - wer_x]], dtype=np.float64),
            fmt=summary.marker,
            markersize=8.0,
            markeredgewidth=1.2,
            markeredgecolor="white",
            color=summary.color,
            ecolor=summary.color,
            elinewidth=1.45,
            capsize=3,
            zorder=3,
        )
        wer_ax.text(
            wer_high + 0.35,
            y,
            f"{wer_x:.1f}%",
            ha="left",
            va="center",
            fontsize=8.5,
            color="#262626",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.25, "alpha": 0.78},
        )

        if latency_x is not None:
            latency_ax.plot(
                latency_x,
                y,
                summary.marker,
                markersize=8.0,
                markeredgewidth=1.2,
                markeredgecolor="white",
                color=summary.color,
                zorder=3,
            )
            latency_ax.text(
                latency_x + 0.035,
                y,
                f"{latency_x:.2f}s",
                ha="left",
                va="center",
                fontsize=8.5,
                color="#262626",
            )

        rtf_ax.errorbar(
            rtf_x,
            y,
            xerr=np.array([[rtf_x - summary.ci_rtf[0]], [summary.ci_rtf[1] - rtf_x]], dtype=np.float64),
            fmt=summary.marker,
            markersize=8.0,
            markeredgewidth=1.2,
            markeredgecolor="white",
            color=summary.color,
            ecolor=summary.color,
            elinewidth=1.45,
            capsize=3,
            zorder=3,
        )
        rtf_ax.text(
            rtf_x + 0.035,
            y,
            f"{rtf_x:.2f}",
            ha="left",
            va="center",
            fontsize=8.5,
            color="#262626",
        )

    wer_ax.set_yticks(y_positions)
    wer_ax.set_yticklabels([row_labels[summary.key] for summary in display_summaries], fontsize=9)
    latency_ax.tick_params(axis="y", labelleft=False)
    rtf_ax.tick_params(axis="y", labelleft=False)

    wer_ax.set_title("Final ASR error", fontsize=10.2, pad=8)
    latency_ax.set_title("ASR latency", fontsize=10.2, pad=8)
    rtf_ax.set_title("Streaming cost", fontsize=10.2, pad=8)
    wer_ax.set_xlabel("Corpus WER (%) ↓")
    latency_ax.set_xlabel("LongYAAL CU (s) ↓")
    rtf_ax.set_xlabel("Wall-clock RTF ↓")
    wer_values = np.array([summary.weighted_wer * 100.0 for summary in display_summaries])
    wer_highs = np.array([summary.ci_wer[1] * 100.0 for summary in display_summaries])
    wer_ax.set_xlim(
        max(0.0, float(np.min(wer_values)) - 1.0),
        float(np.max(wer_highs)) + 1.0,
    )
    latency_values = [
        summary.long_yaal_cu_ms / 1000.0
        for summary in display_summaries
        if summary.long_yaal_cu_ms is not None
    ]
    latency_ax.set_xlim(
        max(0.0, min(latency_values) - 0.25),
        max(latency_values) + 0.3,
    )
    rtf_ax.set_xlim(0.22, 1.52)
    wer_ax.xaxis.set_major_locator(MultipleLocator(2.5))
    latency_ax.xaxis.set_major_locator(MultipleLocator(0.25))
    rtf_ax.xaxis.set_major_locator(MultipleLocator(0.5))
    wer_ax.set_ylim(-0.18, 1.25)

    fig.suptitle("Comparaison ASR", x=0.02, y=1.04, ha="left", fontsize=13.2)

    fig.text(
        0.5,
        -0.015,
        "21 common MCIF dev talks. WER/RTF bars are 90% audio bootstrap CIs. Gemma latency uses CU-LongYAAL with negative resegmented emissions clipped to 0 ms.",
        ha="center",
        va="top",
        fontsize=7.5,
        color="#525252",
    )

    fig.savefig(path_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    qwen_metrics = load_qwen_metrics(args.qwen_summary, args.qwen_dir)
    voxtral_by_delay = {
        "delay240ms": load_voxtral_metrics(args.voxtral_results, "delay240ms"),
        "delay480ms": load_voxtral_metrics(args.voxtral_results, "delay480ms"),
        "delay960ms": load_voxtral_metrics(args.voxtral_results, "delay960ms"),
    }
    gemma_metrics = load_gemma_metrics(args.gemma_manifest)
    common_talks = sorted(
        set(qwen_metrics).intersection(
            set(gemma_metrics),
            *(set(rows) for rows in voxtral_by_delay.values()),
        )
    )
    if not common_talks:
        raise SystemExit("No common talks between Qwen, Voxtral, and Gemma metrics.")

    rng = np.random.default_rng(args.bootstrap_seed)
    qwen = build_method_summary(
        key="qwen_forced",
        label="Qwen3 ASR + Forced Aligner",
        short_label="Qwen3 ASR + aligner",
        color="#2563eb",
        marker="o",
        metrics_by_talk=qwen_metrics,
        common_talks=common_talks,
        bootstrap_samples=args.bootstrap_samples,
        rng=rng,
        ci=args.ci,
    )
    qwen.long_yaal_cu_ms, qwen.long_yaal_ca_ms = load_longyaal_scores(args.qwen_eval)

    voxtral_summaries: list[MethodSummary] = []
    for delay_key, short_delay in [("delay240ms", "240 ms"), ("delay480ms", "480 ms"), ("delay960ms", "960 ms")]:
        raw_summary = build_method_summary(
            key=f"voxtral_{delay_key}",
            label=f"Voxtral Mini 4B Realtime ({short_delay} delay)",
            short_label=f"Voxtral {short_delay}",
            color="#dc2626",
            marker="^",
            metrics_by_talk=voxtral_by_delay[delay_key],
            common_talks=common_talks,
            bootstrap_samples=args.bootstrap_samples,
            rng=rng,
            ci=args.ci,
        )
        voxtral_summaries.append(attach_qwen_relative_stats(raw_summary, qwen=qwen))

    voxtral_delay480 = next(
        summary for summary in voxtral_summaries if summary.key == "voxtral_delay480ms"
    )
    voxtral_delay480.long_yaal_cu_ms, voxtral_delay480.long_yaal_ca_ms = load_longyaal_scores(
        args.voxtral_delay480_eval
    )

    gemma = build_method_summary(
        key="gemma_e4b_la",
        label="Gemma E4B ASR local agreement",
        short_label="Gemma E4B LA",
        color="#0f766e",
        marker="s",
        metrics_by_talk=gemma_metrics,
        common_talks=common_talks,
        bootstrap_samples=args.bootstrap_samples,
        rng=rng,
        ci=args.ci,
    )
    gemma = attach_qwen_relative_stats(gemma, qwen=qwen)
    gemma_raw_cu, gemma_raw_ca = load_longyaal_scores(args.gemma_eval)
    gemma.long_yaal_cu_ms = (
        compute_clip0_longyaal_cu_ms(args.gemma_eval.parent / "instances.resegmented.jsonl")
        or gemma_raw_cu
    )
    gemma.long_yaal_ca_ms = gemma_raw_ca

    summaries = [qwen, *voxtral_summaries, gemma]
    summary_stem = str(args.output_stem)
    write_tsv(args.output_dir / f"{summary_stem}_summary.tsv", summaries)
    (args.output_dir / f"{summary_stem}_summary.json").write_text(
        json.dumps(
            summary_to_json(
                qwen_summary_path=args.qwen_summary,
                qwen_dir=args.qwen_dir,
                voxtral_results_path=args.voxtral_results,
                gemma_manifest_path=args.gemma_manifest,
                gemma_eval_path=args.gemma_eval,
                common_talks=common_talks,
                summaries=summaries,
            ),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_pareto(args.output_dir / summary_stem, summaries, dpi=args.dpi)

    for summary in summaries:
        delta = ""
        if summary.wer_delta_vs_qwen is not None:
            delta = f", ΔWER={summary.wer_delta_vs_qwen * 100.0:+.2f} pts, RTFx={summary.rtf_ratio_vs_qwen:.2f}"
        print(
            f"{summary.label}: WER={summary.weighted_wer * 100.0:.2f}%, "
            f"mean RTF={summary.mean_rtf:.2f}{delta}"
        )
    if gemma_raw_cu is not None and gemma.long_yaal_cu_ms is not None:
        print(
            "Gemma raw LongYAAL CU="
            f"{gemma_raw_cu / 1000.0:.2f}s; plotted clipped CU="
            f"{gemma.long_yaal_cu_ms / 1000.0:.2f}s"
        )
    print(f"Wrote {args.output_dir / (summary_stem + '.png')}")


if __name__ == "__main__":
    main()
