#!/usr/bin/env python3
"""Paper-style smoothed risk curve for ASR live-tail words.

For each captured ASR chunk, we compare live-tail source words against the
human English source reference and estimate:

    P(word is wrong vs. reference | distance from current ASR tail end)

The curve is a Gaussian-kernel estimate over exposure-weighted word counts.
Uncertainty bands are audio-level bootstrap intervals.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from difflib import SequenceMatcher
import json
import math
from pathlib import Path
import re
import string
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np
import yaml


PUNCT_TABLE = str.maketrans("", "", string.punctuation + "\u201c\u201d\u2018\u2019")
NONSPACE_RE = re.compile(r"\S+")
DEFAULT_VOXTRAL_ROOT = Path("/home/fuxa/iwslt-2026-baselines/precomputed_asr_voxtral")
DEFAULT_VOXTRAL_LATENCY_OFFSET_MS = 290.0
DEFAULT_GEMMA_CAPTURE_ROOT = Path("outputs/gemma_e4b_asr_mcif_la_full_20260424")
DEFAULT_QWEN_EVAL = Path("outputs/asr_compare_enen_21audio_20260421/qwen_forced_eval/evaluation.json")
DEFAULT_GEMMA_EVAL = Path("outputs/gemma_e4b_asr_mcif_la_full_20260424/eval/evaluation.json")


@dataclass(frozen=True)
class AudioBinnedCounts:
    wav_name: str
    total: np.ndarray
    wrong: np.ndarray


def normalize_token(surface: str) -> str:
    return surface.lower().translate(PUNCT_TABLE).strip()


def tokenize_text(text: str) -> list[str]:
    tokens: list[str] = []
    for match in NONSPACE_RE.finditer(text):
        token = normalize_token(match.group(0))
        if token:
            tokens.append(token)
    return tokens


def word_surface(item: dict[str, Any]) -> str:
    return str(item.get("text") or item.get("surface") or "").strip()


def normalized_word_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        token = normalize_token(word_surface(item))
        if not token:
            continue
        rows.append({"token": token, "end_time": item.get("end_time")})
    return rows


def resolve_capture_dir(capture_root: Path) -> Path:
    nested = capture_root / "captures"
    if nested.exists():
        return nested
    return capture_root


def load_captures(capture_root: Path) -> list[dict[str, Any]]:
    capture_dir = resolve_capture_dir(capture_root)
    paths = sorted(path for path in capture_dir.glob("*.json") if path.name != "manifest.json")
    if not paths:
        raise SystemExit(f"No capture JSON files under {capture_dir}")
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def load_references_by_wav(
    *,
    segmentation_path: Path,
    reference_path: Path,
) -> dict[str, list[str]]:
    segments = yaml.safe_load(segmentation_path.read_text(encoding="utf-8")) or []
    references = reference_path.read_text(encoding="utf-8").splitlines()
    if len(segments) != len(references):
        raise ValueError("Segmentation and reference files must have matching lengths.")

    grouped: dict[str, list[str]] = {}
    for segment, reference in zip(segments, references):
        wav_name = Path(str(segment["wav"])).name
        grouped.setdefault(wav_name, []).append(reference.strip())
    return {
        wav_name: tokenize_text(" ".join(part for part in parts if part))
        for wav_name, parts in grouped.items()
    }


def alignment_count_match(chunk: dict[str, Any], word_count: int) -> bool:
    if "alignment_count_match" in chunk:
        return bool(chunk.get("alignment_count_match"))
    diagnostics = dict(chunk.get("diagnostics") or {})
    if "global_alignment_count_match" in diagnostics:
        return bool(diagnostics["global_alignment_count_match"])
    return int(chunk.get("alignment_word_count", word_count)) == int(word_count)


def reference_match_flags(
    *,
    reference_tokens: list[str],
    hypothesis_tokens: list[str],
) -> list[bool]:
    flags = [False] * len(hypothesis_tokens)
    matcher = SequenceMatcher(None, reference_tokens, hypothesis_tokens, autojunk=False)
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            continue
        for idx in range(j1, j2):
            flags[idx] = True
    return flags


def live_tail_start_index(
    *,
    chunk: dict[str, Any],
    public_committed_count_before: int,
) -> tuple[int, int]:
    public_committed_count = public_committed_count_before
    if bool(chunk.get("did_commit_segment", False)):
        public_committed_count += len(list(chunk.get("new_committed_words") or []))
    return public_committed_count, public_committed_count


def lcp_count(left: list[str], right: list[str]) -> int:
    count = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        count += 1
    return count


def distance_bin_index(
    *,
    distance_ms: float,
    bin_ms: int,
    max_distance_ms: int,
) -> int | None:
    if float(distance_ms) >= float(max_distance_ms):
        return None
    return min(int(max_distance_ms // bin_ms), int(math.floor(float(distance_ms) / float(bin_ms))))


def collect_binned_counts(
    *,
    captures: list[dict[str, Any]],
    references_by_wav: dict[str, list[str]],
    bin_ms: int,
    max_distance_ms: int,
) -> list[AudioBinnedCounts]:
    n_bins = int(max_distance_ms // bin_ms) + 1
    audio_counts: list[AudioBinnedCounts] = []

    for capture in captures:
        wav_name = str(capture.get("wav_name") or Path(str(capture.get("wav_path", ""))).name)
        reference_tokens = references_by_wav.get(wav_name)
        if reference_tokens is None:
            raise ValueError(f"No reference found for {wav_name}.")

        total = np.zeros(n_bins, dtype=np.float64)
        wrong = np.zeros(n_bins, dtype=np.float64)
        public_committed_count = 0

        for chunk in list(capture.get("chunks") or []):
            word_rows = normalized_word_rows(list(chunk.get("words") or []))
            if not word_rows or not alignment_count_match(chunk, len(word_rows)):
                continue

            tail_start, public_committed_count = live_tail_start_index(
                chunk=chunk,
                public_committed_count_before=public_committed_count,
            )
            if tail_start >= len(word_rows):
                continue

            tail_end_times_ms = [
                float(row["end_time"]) * 1000.0
                for row in word_rows[tail_start:]
                if row.get("end_time") is not None
            ]
            if not tail_end_times_ms:
                continue
            tail_end_ms = max(tail_end_times_ms)

            hypothesis_tokens = [row["token"] for row in word_rows]
            match_flags = reference_match_flags(
                reference_tokens=reference_tokens,
                hypothesis_tokens=hypothesis_tokens,
            )

            for word_idx in range(tail_start, len(word_rows)):
                row = word_rows[word_idx]
                if row.get("end_time") is None:
                    continue
                word_end_ms = float(row["end_time"]) * 1000.0
                distance_ms = max(0.0, tail_end_ms - word_end_ms)
                bin_idx = distance_bin_index(
                    distance_ms=distance_ms,
                    bin_ms=int(bin_ms),
                    max_distance_ms=int(max_distance_ms),
                )
                if bin_idx is None:
                    continue
                total[bin_idx] += 1.0
                wrong[bin_idx] += 0.0 if bool(match_flags[word_idx]) else 1.0

        audio_counts.append(AudioBinnedCounts(wav_name=wav_name, total=total, wrong=wrong))

    return audio_counts


def collect_voxtral_binned_counts(
    *,
    voxtral_root: Path,
    delay_dir: str,
    references_by_wav: dict[str, list[str]],
    bin_ms: int,
    max_distance_ms: int,
) -> list[AudioBinnedCounts]:
    n_bins = int(max_distance_ms // bin_ms) + 1
    audio_counts: list[AudioBinnedCounts] = []

    for trace_path in sorted(voxtral_root.glob(f"*/{delay_dir}/asr_chunks.jsonl")):
        wav_name = f"{trace_path.parents[1].name}.wav"
        reference_tokens = references_by_wav.get(wav_name)
        if reference_tokens is None:
            continue

        total = np.zeros(n_bins, dtype=np.float64)
        wrong = np.zeros(n_bins, dtype=np.float64)
        previous_tokens: list[str] = []

        for row in load_jsonl(trace_path):
            hypothesis_tokens = tokenize_text(str(row.get("full_text") or ""))
            if not hypothesis_tokens:
                previous_tokens = hypothesis_tokens
                continue

            tail_start = lcp_count(previous_tokens, hypothesis_tokens)
            previous_tokens = hypothesis_tokens
            if tail_start >= len(hypothesis_tokens):
                continue

            audio_seconds = float(row.get("audio_seconds") or 0.0)
            if audio_seconds <= 0.0:
                continue

            match_flags = reference_match_flags(
                reference_tokens=reference_tokens,
                hypothesis_tokens=hypothesis_tokens,
            )
            word_count = len(hypothesis_tokens)
            tail_end_ms = audio_seconds * 1000.0
            for word_idx in range(tail_start, word_count):
                word_end_ms = tail_end_ms * float(word_idx + 1) / float(word_count)
                distance_ms = max(0.0, tail_end_ms - word_end_ms)
                bin_idx = distance_bin_index(
                    distance_ms=distance_ms,
                    bin_ms=int(bin_ms),
                    max_distance_ms=int(max_distance_ms),
                )
                if bin_idx is None:
                    continue
                total[bin_idx] += 1.0
                wrong[bin_idx] += 0.0 if bool(match_flags[word_idx]) else 1.0

        audio_counts.append(AudioBinnedCounts(wav_name=wav_name, total=total, wrong=wrong))

    return audio_counts


def collect_text_tail_binned_counts(
    *,
    captures: list[dict[str, Any]],
    references_by_wav: dict[str, list[str]],
    bin_ms: int,
    max_distance_ms: int,
) -> list[AudioBinnedCounts]:
    """Collect tail-risk counts when only chunk text is available.

    Gemma E4B local-agreement captures deliberately do not use AlignAtt or a
    forced aligner, so they have no word timestamps. For comparison with the
    Voxtral trace path, we estimate each current hypothesis word's acoustic
    position proportionally on the source-time axis up to the chunk boundary.
    The live tail is the uncommitted suffix after ``committed_text``.
    """
    n_bins = int(max_distance_ms // bin_ms) + 1
    audio_counts: list[AudioBinnedCounts] = []

    for capture in captures:
        wav_name = str(capture.get("wav_name") or Path(str(capture.get("wav_path", ""))).name)
        reference_tokens = references_by_wav.get(wav_name)
        if reference_tokens is None:
            raise ValueError(f"No reference found for {wav_name}.")

        total = np.zeros(n_bins, dtype=np.float64)
        wrong = np.zeros(n_bins, dtype=np.float64)

        for chunk in list(capture.get("chunks") or []):
            hypothesis_tokens = tokenize_text(str(chunk.get("hypothesis_text") or ""))
            if not hypothesis_tokens:
                continue
            committed_tokens = tokenize_text(str(chunk.get("committed_text") or ""))
            tail_start = min(len(committed_tokens), len(hypothesis_tokens))
            if tail_start >= len(hypothesis_tokens):
                continue

            audio_seconds = float(chunk.get("audio_processed_s") or 0.0)
            if audio_seconds <= 0.0:
                continue

            match_flags = reference_match_flags(
                reference_tokens=reference_tokens,
                hypothesis_tokens=hypothesis_tokens,
            )
            word_count = len(hypothesis_tokens)
            tail_end_ms = audio_seconds * 1000.0
            for word_idx in range(tail_start, word_count):
                word_end_ms = tail_end_ms * float(word_idx + 1) / float(word_count)
                distance_ms = max(0.0, tail_end_ms - word_end_ms)
                bin_idx = distance_bin_index(
                    distance_ms=distance_ms,
                    bin_ms=int(bin_ms),
                    max_distance_ms=int(max_distance_ms),
                )
                if bin_idx is None:
                    continue
                total[bin_idx] += 1.0
                wrong[bin_idx] += 0.0 if bool(match_flags[word_idx]) else 1.0

        audio_counts.append(AudioBinnedCounts(wav_name=wav_name, total=total, wrong=wrong))

    return audio_counts


def load_longyaal_cu_ms(path: Path) -> float | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    scores = dict(payload.get("contract_scores") or {})
    value = scores.get("LongYAAL CU")
    return None if value is None else float(value)


def resolve_gemma_latency_offset_ms(
    *,
    qwen_eval: Path,
    gemma_eval: Path,
    explicit_offset_ms: float | None,
) -> float:
    if explicit_offset_ms is not None:
        return float(explicit_offset_ms)
    qwen_longyaal = load_longyaal_cu_ms(qwen_eval)
    gemma_longyaal = load_longyaal_cu_ms(gemma_eval)
    if qwen_longyaal is None or gemma_longyaal is None:
        return 0.0
    return float(gemma_longyaal - qwen_longyaal)


def gaussian_weight_matrix(
    *,
    bin_centers: np.ndarray,
    grid_ms: np.ndarray,
    bandwidth_ms: float,
) -> np.ndarray:
    deltas = grid_ms[:, None] - bin_centers[None, :]
    weights = np.exp(-0.5 * (deltas / float(bandwidth_ms)) ** 2)
    return weights


def smooth_rate(
    *,
    total: np.ndarray,
    wrong: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    numerator = weights @ wrong
    denominator = weights @ total
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0.0)
    return rate


def bootstrap_intervals(
    *,
    audio_counts: list[AudioBinnedCounts],
    weights: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if samples <= 0 or not audio_counts:
        n_grid = weights.shape[0]
        return np.zeros(n_grid), np.zeros(n_grid)

    rng = np.random.default_rng(seed)
    totals = np.stack([item.total for item in audio_counts], axis=0)
    wrongs = np.stack([item.wrong for item in audio_counts], axis=0)
    n_audio = totals.shape[0]
    curves = np.zeros((samples, weights.shape[0]), dtype=np.float64)

    for sample_idx in range(samples):
        indices = rng.integers(0, n_audio, size=n_audio)
        sample_total = totals[indices].sum(axis=0)
        sample_wrong = wrongs[indices].sum(axis=0)
        curves[sample_idx] = smooth_rate(total=sample_total, wrong=sample_wrong, weights=weights)

    return np.quantile(curves, 0.05, axis=0), np.quantile(curves, 0.95, axis=0)


def summarize_series(
    *,
    audio_counts: list[AudioBinnedCounts],
    bin_starts: np.ndarray,
    weights: np.ndarray,
    bin_ms: int,
    max_distance_ms: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    total_counts = np.stack([item.total for item in audio_counts], axis=0).sum(axis=0)
    wrong_counts = np.stack([item.wrong for item in audio_counts], axis=0).sum(axis=0)
    rate = smooth_rate(total=total_counts, wrong=wrong_counts, weights=weights)
    lo, hi = bootstrap_intervals(
        audio_counts=audio_counts,
        weights=weights,
        samples=int(bootstrap_samples),
        seed=int(bootstrap_seed),
    )

    exposure_total = float(total_counts.sum())
    histogram = []
    for start, total, wrong in zip(bin_starts, total_counts, wrong_counts):
        histogram.append(
            {
                "bucket_start_ms": int(start),
                "bucket_end_ms": None if int(start) == int(max_distance_ms) else int(start + bin_ms),
                "exposure_count": int(total),
                "wrong_count": int(wrong),
                "wrong_rate": 0.0 if total <= 0.0 else float(wrong / total),
                "exposure_share": 0.0 if exposure_total <= 0.0 else float(total / exposure_total),
            }
        )

    return {
        "aggregate": {
            "wav_count": len(audio_counts),
            "exposed_word_count": int(exposure_total),
            "wrong_word_count": int(wrong_counts.sum()),
            "wrong_word_rate": 0.0 if exposure_total <= 0.0 else float(wrong_counts.sum() / exposure_total),
        },
        "rate": rate,
        "lo": lo,
        "hi": hi,
        "histogram": histogram,
        "per_audio": [
            {
                "wav_name": item.wav_name,
                "exposed_word_count": int(item.total.sum()),
                "wrong_word_count": int(item.wrong.sum()),
                "wrong_word_rate": (
                    0.0 if item.total.sum() <= 0.0 else float(item.wrong.sum() / item.total.sum())
                ),
            }
            for item in audio_counts
        ],
    }


def build_payload(
    *,
    audio_counts: list[AudioBinnedCounts],
    voxtral_audio_counts: list[AudioBinnedCounts] | None,
    gemma_audio_counts: list[AudioBinnedCounts] | None,
    bin_ms: int,
    max_distance_ms: int,
    grid_step_ms: int,
    bandwidth_ms: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
    capture_root: Path,
    segmentation: Path,
    reference: Path,
    voxtral_root: Path | None,
    voxtral_delay_dir: str,
    voxtral_latency_offset_ms: float,
    gemma_capture_root: Path | None,
    gemma_latency_offset_ms: float,
    qwen_eval: Path,
    gemma_eval: Path,
) -> dict[str, Any]:
    bin_starts = np.arange(0, int(max_distance_ms) + int(bin_ms), int(bin_ms), dtype=np.float64)
    bin_centers = bin_starts + float(bin_ms) / 2.0
    grid_ms = np.arange(0, int(max_distance_ms) + int(grid_step_ms), int(grid_step_ms), dtype=np.float64)
    weights = gaussian_weight_matrix(
        bin_centers=bin_centers,
        grid_ms=grid_ms,
        bandwidth_ms=float(bandwidth_ms),
    )

    qwen_series = summarize_series(
        audio_counts=audio_counts,
        bin_starts=bin_starts,
        weights=weights,
        bin_ms=int(bin_ms),
        max_distance_ms=int(max_distance_ms),
        bootstrap_samples=int(bootstrap_samples),
        bootstrap_seed=int(bootstrap_seed),
    )

    voxtral_series = None
    if voxtral_audio_counts:
        voxtral_series = summarize_series(
            audio_counts=voxtral_audio_counts,
            bin_starts=bin_starts,
            weights=weights,
            bin_ms=int(bin_ms),
            max_distance_ms=int(max_distance_ms),
            bootstrap_samples=int(bootstrap_samples),
            bootstrap_seed=int(bootstrap_seed) + 101,
        )

    gemma_series = None
    if gemma_audio_counts:
        gemma_series = summarize_series(
            audio_counts=gemma_audio_counts,
            bin_starts=bin_starts,
            weights=weights,
            bin_ms=int(bin_ms),
            max_distance_ms=int(max_distance_ms),
            bootstrap_samples=int(bootstrap_samples),
            bootstrap_seed=int(bootstrap_seed) + 202,
        )

    rows = [
        {
            "distance_ms": int(x),
            "wrong_rate": float(y),
            "wrong_rate_p05": float(low),
            "wrong_rate_p95": float(high),
        }
        for x, y, low, high in zip(
            grid_ms,
            qwen_series["rate"],
            qwen_series["lo"],
            qwen_series["hi"],
        )
    ]
    if voxtral_series is not None:
        for row, y, low, high in zip(
            rows,
            voxtral_series["rate"],
            voxtral_series["lo"],
            voxtral_series["hi"],
        ):
            row["voxtral_wrong_rate"] = float(y)
            row["voxtral_wrong_rate_p05"] = float(low)
            row["voxtral_wrong_rate_p95"] = float(high)
            row["voxtral_display_distance_ms"] = int(row["distance_ms"] + float(voxtral_latency_offset_ms))
    if gemma_series is not None:
        for row, y, low, high in zip(
            rows,
            gemma_series["rate"],
            gemma_series["lo"],
            gemma_series["hi"],
        ):
            row["gemma_wrong_rate"] = float(y)
            row["gemma_wrong_rate_p05"] = float(low)
            row["gemma_wrong_rate_p95"] = float(high)
            row["gemma_display_distance_ms"] = int(row["distance_ms"] + float(gemma_latency_offset_ms))

    return {
        "schema_version": "asr_reference_tail_risk_curve_v4",
        "config": {
            "capture_root": str(capture_root),
            "segmentation": str(segmentation),
            "reference": str(reference),
            "voxtral_root": None if voxtral_root is None else str(voxtral_root),
            "voxtral_delay_dir": str(voxtral_delay_dir),
            "voxtral_latency_offset_ms": float(voxtral_latency_offset_ms),
            "gemma_capture_root": None if gemma_capture_root is None else str(gemma_capture_root),
            "gemma_latency_offset_ms": float(gemma_latency_offset_ms),
            "qwen_eval": str(qwen_eval),
            "gemma_eval": str(gemma_eval),
            "bin_ms": int(bin_ms),
            "max_distance_ms": int(max_distance_ms),
            "grid_step_ms": int(grid_step_ms),
            "bandwidth_ms": float(bandwidth_ms),
            "bootstrap_samples": int(bootstrap_samples),
            "bootstrap_seed": int(bootstrap_seed),
            "metric": "wrong_live_tail_asr_words_vs_source_reference",
        },
        "series": {
            "qwen": {
                "label": "Qwen3 local word risk at distance d",
                "aggregate": qwen_series["aggregate"],
                "per_audio": qwen_series["per_audio"],
            },
            **(
                {}
                if voxtral_series is None
                else {
                    "voxtral": {
                        "label": "Voxtral local word risk at distance d",
                        "aggregate": voxtral_series["aggregate"],
                        "per_audio": voxtral_series["per_audio"],
                        "display_latency_offset_ms": float(voxtral_latency_offset_ms),
                        "timing_note": "Voxtral logs do not include word timestamps; word end times are estimated proportionally within each current hypothesis.",
                    }
                }
            ),
            **(
                {}
                if gemma_series is None
                else {
                    "gemma": {
                        "label": "Gemma E4B local word risk at distance d",
                        "aggregate": gemma_series["aggregate"],
                        "per_audio": gemma_series["per_audio"],
                        "display_latency_offset_ms": float(gemma_latency_offset_ms),
                        "timing_note": "Gemma E4B LA captures do not include word timestamps; word end times are estimated proportionally within each current hypothesis.",
                    }
                }
            ),
        },
        "aggregate": qwen_series["aggregate"],
        "rows": rows,
        "histogram": qwen_series["histogram"],
        "per_audio": qwen_series["per_audio"],
    }


def write_tsv(path: Path, rows: list[dict[str, Any]], header: list[str]) -> None:
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join("" if row.get(key) is None else str(row.get(key, "")) for key in header))
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def plot_payload(payload: dict[str, Any], output_dir: Path) -> None:
    rows = list(payload["rows"])
    x = np.array([float(row["distance_ms"]) for row in rows], dtype=np.float64)
    voxtral_latency_offset_ms = float(payload["config"].get("voxtral_latency_offset_ms") or 0.0)
    gemma_latency_offset_ms = float(payload["config"].get("gemma_latency_offset_ms") or 0.0)
    y = np.array([100.0 * float(row["wrong_rate"]) for row in rows], dtype=np.float64)
    lo = np.array([100.0 * float(row["wrong_rate_p05"]) for row in rows], dtype=np.float64)
    hi = np.array([100.0 * float(row["wrong_rate_p95"]) for row in rows], dtype=np.float64)
    has_voxtral = any("voxtral_wrong_rate" in row for row in rows)
    has_gemma = any("gemma_wrong_rate" in row for row in rows)
    voxtral_y = np.array(
        [100.0 * float(row.get("voxtral_wrong_rate", 0.0)) for row in rows],
        dtype=np.float64,
    )
    voxtral_lo = np.array(
        [100.0 * float(row.get("voxtral_wrong_rate_p05", 0.0)) for row in rows],
        dtype=np.float64,
    )
    voxtral_hi = np.array(
        [100.0 * float(row.get("voxtral_wrong_rate_p95", 0.0)) for row in rows],
        dtype=np.float64,
    )
    gemma_y = np.array(
        [100.0 * float(row.get("gemma_wrong_rate", 0.0)) for row in rows],
        dtype=np.float64,
    )
    gemma_lo = np.array(
        [100.0 * float(row.get("gemma_wrong_rate_p05", 0.0)) for row in rows],
        dtype=np.float64,
    )
    gemma_hi = np.array(
        [100.0 * float(row.get("gemma_wrong_rate_p95", 0.0)) for row in rows],
        dtype=np.float64,
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.6,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    fig.subplots_adjust(left=0.165, right=0.945, top=0.965, bottom=0.205)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#e7e7e7", linewidth=0.8)

    qwen_color = "#b3312c"
    voxtral_color = "#2563eb"
    gemma_color = "#0f766e"
    ax.fill_between(x, lo, hi, color=qwen_color, alpha=0.15, linewidth=0)
    ax.plot(
        x,
        y,
        color=qwen_color,
        linewidth=1.8,
        solid_capstyle="round",
    )
    if has_voxtral:
        voxtral_x = x + voxtral_latency_offset_ms
        ax.fill_between(
            voxtral_x,
            voxtral_lo,
            voxtral_hi,
            color=voxtral_color,
            alpha=0.13,
            linewidth=0,
        )
        ax.plot(
            voxtral_x,
            voxtral_y,
            color=voxtral_color,
            linewidth=1.7,
            solid_capstyle="round",
        )
    if has_gemma:
        gemma_x = x + gemma_latency_offset_ms
        ax.fill_between(
            gemma_x,
            gemma_lo,
            gemma_hi,
            color=gemma_color,
            alpha=0.12,
            linewidth=0,
        )
        ax.plot(
            gemma_x,
            gemma_y,
            color=gemma_color,
            linewidth=1.7,
            solid_capstyle="round",
        )
    ax.set_ylabel("Ref. error (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))
    ax.yaxis.set_major_locator(MultipleLocator(5.0))
    ax.yaxis.set_minor_locator(MultipleLocator(2.5))
    ax.set_xlabel("Distance to ASR tail (ms)")
    ax.xaxis.set_major_locator(MultipleLocator(250.0))
    ax.xaxis.set_minor_locator(MultipleLocator(50.0))
    display_offsets = [0.0]
    if has_voxtral:
        display_offsets.append(float(voxtral_latency_offset_ms))
    if has_gemma:
        display_offsets.append(float(gemma_latency_offset_ms))
    plot_xmax_ms = payload["config"].get("plot_xmax_ms")
    xmax = (
        float(plot_xmax_ms)
        if plot_xmax_ms is not None
        else float(payload["config"]["max_distance_ms"]) + max(0.0, max(display_offsets))
    )
    ax.set_xlim(
        min(0.0, min(display_offsets)),
        xmax,
    )
    max_error = float(np.max(hi))
    if has_voxtral:
        max_error = max(max_error, float(np.max(voxtral_hi)))
    if has_gemma:
        max_error = max(max_error, float(np.max(gemma_hi)))
    ax.set_ylim(0.0, max(15.0, math.ceil(max_error * 1.10 / 5.0) * 5.0))

    def label_curve(
        x_values: np.ndarray,
        y_values: np.ndarray,
        label: str,
        color: str,
    ) -> None:
        visible = x_values <= ax.get_xlim()[1]
        if not np.any(visible):
            return
        idx = int(np.flatnonzero(visible)[-1])
        ax.annotate(
            label,
            xy=(float(x_values[idx]), float(y_values[idx])),
            xytext=(-4, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=6.6,
            color=color,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
                "pad": 0.8,
            },
        )

    label_curve(x, y, "Qwen3", qwen_color)
    if has_voxtral:
        label_curve(
            x + voxtral_latency_offset_ms,
            voxtral_y,
            f"Voxtral +{voxtral_latency_offset_ms:.0f} ms",
            voxtral_color,
        )
    if has_gemma:
        gemma_label = (
            "Gemma proxy"
            if abs(gemma_latency_offset_ms) < 1e-6
            else f"Gemma {gemma_latency_offset_ms:+.0f} ms"
        )
        label_curve(x + gemma_latency_offset_ms, gemma_y, gemma_label, gemma_color)

    fig.savefig(output_dir / "asr_reference_tail_risk_curve.png", dpi=220)
    fig.savefig(output_dir / "asr_reference_tail_risk_curve.svg")
    fig.savefig(output_dir / "asr_reference_tail_risk_curve.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--segmentation", default="data/devset/audio-segments.yaml", type=Path)
    parser.add_argument("--reference", default="data/devset/ref/en.txt", type=Path)
    parser.add_argument("--voxtral-root", default=DEFAULT_VOXTRAL_ROOT, type=Path)
    parser.add_argument("--voxtral-delay-dir", default="delay480ms")
    parser.add_argument("--voxtral-latency-offset-ms", type=float, default=DEFAULT_VOXTRAL_LATENCY_OFFSET_MS)
    parser.add_argument("--no-voxtral", action="store_true")
    parser.add_argument("--gemma-capture-root", default=DEFAULT_GEMMA_CAPTURE_ROOT, type=Path)
    parser.add_argument("--qwen-eval", default=DEFAULT_QWEN_EVAL, type=Path)
    parser.add_argument("--gemma-eval", default=DEFAULT_GEMMA_EVAL, type=Path)
    parser.add_argument(
        "--gemma-latency-offset-ms",
        type=float,
        default=0.0,
        help=(
            "Display offset for Gemma E4B LA in ms. Defaults to 0 because the "
            "raw Gemma LongYAAL score is contaminated by prompt-leak "
            "resegmentation artifacts."
        ),
    )
    parser.add_argument("--no-gemma", action="store_true")
    parser.add_argument("--bin-ms", type=int, default=25)
    parser.add_argument("--max-distance-ms", type=int, default=1200)
    parser.add_argument("--grid-step-ms", type=int, default=10)
    parser.add_argument("--bandwidth-ms", type=float, default=75.0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    parser.add_argument("--plot-xmax-ms", type=float, default=None)
    args = parser.parse_args()

    captures = load_captures(Path(args.capture_root))
    references_by_wav = load_references_by_wav(
        segmentation_path=Path(args.segmentation),
        reference_path=Path(args.reference),
    )
    audio_counts = collect_binned_counts(
        captures=captures,
        references_by_wav=references_by_wav,
        bin_ms=int(args.bin_ms),
        max_distance_ms=int(args.max_distance_ms),
    )
    voxtral_audio_counts = None
    if not bool(args.no_voxtral) and Path(args.voxtral_root).exists():
        voxtral_audio_counts = collect_voxtral_binned_counts(
            voxtral_root=Path(args.voxtral_root),
            delay_dir=str(args.voxtral_delay_dir),
            references_by_wav=references_by_wav,
            bin_ms=int(args.bin_ms),
            max_distance_ms=int(args.max_distance_ms),
        )
    gemma_audio_counts = None
    if not bool(args.no_gemma) and Path(args.gemma_capture_root).exists():
        gemma_audio_counts = collect_text_tail_binned_counts(
            captures=load_captures(Path(args.gemma_capture_root)),
            references_by_wav=references_by_wav,
            bin_ms=int(args.bin_ms),
            max_distance_ms=int(args.max_distance_ms),
        )
    gemma_latency_offset_ms = resolve_gemma_latency_offset_ms(
        qwen_eval=Path(args.qwen_eval),
        gemma_eval=Path(args.gemma_eval),
        explicit_offset_ms=args.gemma_latency_offset_ms,
    )
    payload = build_payload(
        audio_counts=audio_counts,
        voxtral_audio_counts=voxtral_audio_counts,
        gemma_audio_counts=gemma_audio_counts,
        bin_ms=int(args.bin_ms),
        max_distance_ms=int(args.max_distance_ms),
        grid_step_ms=int(args.grid_step_ms),
        bandwidth_ms=float(args.bandwidth_ms),
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
        capture_root=Path(args.capture_root),
        segmentation=Path(args.segmentation),
        reference=Path(args.reference),
        voxtral_root=None if bool(args.no_voxtral) else Path(args.voxtral_root),
        voxtral_delay_dir=str(args.voxtral_delay_dir),
        voxtral_latency_offset_ms=float(args.voxtral_latency_offset_ms),
        gemma_capture_root=None if bool(args.no_gemma) else Path(args.gemma_capture_root),
        gemma_latency_offset_ms=float(gemma_latency_offset_ms),
        qwen_eval=Path(args.qwen_eval),
        gemma_eval=Path(args.gemma_eval),
    )
    payload["config"]["plot_xmax_ms"] = args.plot_xmax_ms

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "asr_reference_tail_risk_curve_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_tsv(
        output_dir / "asr_reference_tail_risk_curve.tsv",
        list(payload["rows"]),
        [
            "distance_ms",
            "wrong_rate",
            "wrong_rate_p05",
            "wrong_rate_p95",
            "voxtral_wrong_rate",
            "voxtral_wrong_rate_p05",
            "voxtral_wrong_rate_p95",
            "voxtral_display_distance_ms",
            "gemma_wrong_rate",
            "gemma_wrong_rate_p05",
            "gemma_wrong_rate_p95",
            "gemma_display_distance_ms",
        ],
    )
    write_tsv(
        output_dir / "asr_reference_tail_risk_curve_histogram.tsv",
        list(payload["histogram"]),
        [
            "bucket_start_ms",
            "bucket_end_ms",
            "exposure_count",
            "wrong_count",
            "wrong_rate",
            "exposure_share",
        ],
    )
    plot_payload(payload, output_dir)
    print(f"[done] wrote {output_dir / 'asr_reference_tail_risk_curve.png'}", flush=True)


if __name__ == "__main__":
    main()
