#!/usr/bin/env python3
"""Evaluate a calibrated Gemma ASR word-end proxy as a hold-back cut rule.

This script consumes alignment bundles already materialized by
``compare_asr_word_end_bias.py``:

  <bundle-root>/qwen_forced/<wav>.json
  <bundle-root>/gemma_vllm_qk_fast/<wav>.json

It answers the question that matters for a streaming cut policy:

  if we treat Gemma's word-end proxy as the boundary estimator and hold back
  the last ``holdback_ms`` of audio, how often would we commit too early or
  too late relative to the Qwen forced-align teacher?

The Gemma bundles already include the shipping per-head-set timestamp
correction from ``word_end_offset_seconds`` in the AlignAtt head payload.
This script evaluates one maintained policy on top of that: a single
frontier-calibration scalar supplied explicitly by the caller. We keep that
scalar out of this script's fitting logic on purpose: fitting on the same set
being evaluated is convenient for exploration, but it is not the maintained
paper-grade path.

Two diagnostics are reported:

  1. timestamp residuals after an optional global scalar calibration offset
  2. frontier-decision fidelity under a chunked hold-back rule

The latter is the real policy metric. For each matched word piece we compare
the teacher commit chunk

    ceil(max(min_start_s, teacher_end_s + holdback_s) / chunk_s) * chunk_s

against the proxy commit chunk computed from the calibrated Gemma word end.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


_WORD_PIECE_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
_EPS = 1e-9


@dataclass(frozen=True)
class ExpandedWordPiece:
    raw_text: str
    normalized_piece: str
    end_time_s: float
    raw_word_index: int
    piece_index: int


def _quantile(sorted_values: Sequence[float], q: float) -> float | None:
    if not sorted_values:
        return None
    idx = min(len(sorted_values) - 1, max(0, int(q * (len(sorted_values) - 1))))
    return float(sorted_values[idx])


def _safe_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _safe_median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _safe_pstdev(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(statistics.pstdev(values))


def _expand_alignment_words(words: Sequence[dict[str, Any]]) -> list[ExpandedWordPiece]:
    expanded: list[ExpandedWordPiece] = []
    for raw_word_index, word in enumerate(words):
        raw_text = str(word["text"])
        end_time_s = float(word["end_time"])
        pieces = _WORD_PIECE_RE.findall(raw_text.lower())
        for piece_index, normalized_piece in enumerate(pieces):
            expanded.append(
                ExpandedWordPiece(
                    raw_text=raw_text,
                    normalized_piece=normalized_piece,
                    end_time_s=end_time_s,
                    raw_word_index=int(raw_word_index),
                    piece_index=int(piece_index),
                )
            )
    return expanded


def _match_word_pieces(
    *,
    teacher_words: Sequence[dict[str, Any]],
    proxy_words: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    teacher_expanded = _expand_alignment_words(teacher_words)
    proxy_expanded = _expand_alignment_words(proxy_words)
    matcher = SequenceMatcher(
        a=[piece.normalized_piece for piece in teacher_expanded],
        b=[piece.normalized_piece for piece in proxy_expanded],
        autojunk=False,
    )

    matches: list[dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            continue
        span_len = min(i2 - i1, j2 - j1)
        for offset in range(span_len):
            teacher_piece = teacher_expanded[i1 + offset]
            proxy_piece = proxy_expanded[j1 + offset]
            matches.append(
                {
                    "normalized_piece": teacher_piece.normalized_piece,
                    "teacher_raw_text": teacher_piece.raw_text,
                    "proxy_raw_text": proxy_piece.raw_text,
                    "teacher_raw_word_index": int(teacher_piece.raw_word_index),
                    "proxy_raw_word_index": int(proxy_piece.raw_word_index),
                    "teacher_end_time_s": float(teacher_piece.end_time_s),
                    "proxy_end_time_s": float(proxy_piece.end_time_s),
                }
            )
    return matches


def _load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _select_stems(args: argparse.Namespace, teacher_dir: Path) -> list[str]:
    if args.wavs:
        stems = [Path(path).stem for path in args.wavs]
    elif args.wav_dir:
        wavs = sorted(Path(args.wav_dir).glob("*.wav"))
        if args.limit is not None:
            wavs = wavs[: int(args.limit)]
        stems = [wav.stem for wav in wavs]
    else:
        stems = sorted(path.stem for path in teacher_dir.glob("*.json"))
        if args.limit is not None:
            stems = stems[: int(args.limit)]
    return stems


def _frontier_commit_time_s(
    *,
    end_time_s: float,
    chunk_ms: int,
    holdback_ms: int,
    min_start_s: float,
) -> float:
    chunk_s = float(chunk_ms) / 1000.0
    holdback_s = float(holdback_ms) / 1000.0
    threshold_time_s = max(float(min_start_s), float(end_time_s) + holdback_s)
    chunk_index = max(1, int(math.ceil((threshold_time_s - _EPS) / chunk_s)))
    return float(chunk_index) * chunk_s


def _summarize_timestamp_residuals(
    matches: Sequence[dict[str, Any]],
    *,
    offset_s: float,
) -> dict[str, Any]:
    residuals = [
        float(match["proxy_end_time_s"]) + float(offset_s) - float(match["teacher_end_time_s"])
        for match in matches
    ]
    absolute = [abs(value) for value in residuals]
    absolute_sorted = sorted(absolute)
    return {
        "matched_piece_count": int(len(matches)),
        "offset_s": float(offset_s),
        "signed_mean_s": _safe_mean(residuals),
        "signed_median_s": _safe_median(residuals),
        "mae_s": _safe_mean(absolute),
        "abs_p50_s": _quantile(absolute_sorted, 0.50),
        "abs_p90_s": _quantile(absolute_sorted, 0.90),
        "abs_max_s": (None if not absolute else float(max(absolute))),
        "signed_std_s": _safe_pstdev(residuals),
    }


def _summarize_frontier_decisions(
    matches: Sequence[dict[str, Any]],
    *,
    offset_s: float,
    chunk_ms: int,
    holdback_ms: int,
    min_start_s: float,
) -> dict[str, Any]:
    chunk_s = float(chunk_ms) / 1000.0
    holdback_s = float(holdback_ms) / 1000.0

    rows: list[dict[str, Any]] = []
    commit_errors_s: list[float] = []
    commit_errors_chunks: list[int] = []
    early_leaks_s: list[float] = []
    extra_holds_s: list[float] = []

    for match in matches:
        teacher_end_s = float(match["teacher_end_time_s"])
        proxy_end_s = float(match["proxy_end_time_s"]) + float(offset_s)
        teacher_commit_s = _frontier_commit_time_s(
            end_time_s=teacher_end_s,
            chunk_ms=chunk_ms,
            holdback_ms=holdback_ms,
            min_start_s=min_start_s,
        )
        proxy_commit_s = _frontier_commit_time_s(
            end_time_s=proxy_end_s,
            chunk_ms=chunk_ms,
            holdback_ms=holdback_ms,
            min_start_s=min_start_s,
        )
        commit_error_s = float(proxy_commit_s - teacher_commit_s)
        chunk_error = int(round(commit_error_s / chunk_s))
        # Positive => the proxy committed before the teacher frontier would
        # allow, i.e. leaked content that was still inside the protected band.
        unsafe_early_leak_s = max(
            0.0,
            float(teacher_end_s) - (float(proxy_commit_s) - holdback_s),
        )
        conservative_extra_hold_s = max(
            0.0,
            (float(proxy_commit_s) - holdback_s) - float(teacher_end_s),
        )

        commit_errors_s.append(commit_error_s)
        commit_errors_chunks.append(chunk_error)
        if unsafe_early_leak_s > 0.0:
            early_leaks_s.append(unsafe_early_leak_s)
        if conservative_extra_hold_s > 0.0:
            extra_holds_s.append(conservative_extra_hold_s)

        rows.append(
            {
                **match,
                "proxy_end_time_calibrated_s": float(proxy_end_s),
                "teacher_commit_s": float(teacher_commit_s),
                "proxy_commit_s": float(proxy_commit_s),
                "commit_error_s": float(commit_error_s),
                "commit_error_chunks": int(chunk_error),
                "unsafe_early_leak_s": float(unsafe_early_leak_s),
                "conservative_extra_hold_s": float(conservative_extra_hold_s),
            }
        )

    absolute_errors_s = [abs(value) for value in commit_errors_s]
    absolute_errors_sorted = sorted(absolute_errors_s)
    early_sorted = sorted(early_leaks_s)
    late_sorted = sorted(extra_holds_s)

    chunk_histogram: dict[str, int] = {}
    for error in commit_errors_chunks:
        key = str(int(error))
        chunk_histogram[key] = int(chunk_histogram.get(key, 0) + 1)

    worst_early = sorted(rows, key=lambda row: row["unsafe_early_leak_s"], reverse=True)[:12]
    worst_late = sorted(rows, key=lambda row: row["conservative_extra_hold_s"], reverse=True)[:12]
    exact = sum(abs(value) < _EPS for value in commit_errors_s)
    early = sum(value < -_EPS for value in commit_errors_s)
    late = sum(value > _EPS for value in commit_errors_s)

    return {
        "matched_piece_count": int(len(matches)),
        "exact_commit_ratio": (
            None if not rows else float(exact / len(rows))
        ),
        "early_commit_ratio": (
            None if not rows else float(early / len(rows))
        ),
        "late_commit_ratio": (
            None if not rows else float(late / len(rows))
        ),
        "commit_mae_s": _safe_mean(absolute_errors_s),
        "commit_abs_p90_s": _quantile(absolute_errors_sorted, 0.90),
        "commit_abs_max_s": (None if not absolute_errors_s else float(max(absolute_errors_s))),
        "commit_chunk_error_histogram": chunk_histogram,
        "unsafe_early_mean_s": _safe_mean(early_leaks_s),
        "unsafe_early_p90_s": _quantile(early_sorted, 0.90),
        "unsafe_early_max_s": (None if not early_sorted else float(max(early_sorted))),
        "conservative_hold_mean_s": _safe_mean(extra_holds_s),
        "conservative_hold_p90_s": _quantile(late_sorted, 0.90),
        "conservative_hold_max_s": (None if not late_sorted else float(max(late_sorted))),
        "worst_unsafe_early_examples": worst_early,
        "worst_conservative_hold_examples": worst_late,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-root",
        required=True,
        type=Path,
        help="Root directory produced by compare_asr_word_end_bias.py.",
    )
    parser.add_argument("--teacher-tag", default="qwen_forced")
    parser.add_argument("--proxy-tag", default="gemma_vllm_qk_fast")
    parser.add_argument(
        "--wav-dir",
        default=None,
        help="Optional wav directory to define the evaluation subset/order.",
    )
    parser.add_argument(
        "--wavs",
        nargs="*",
        default=None,
        help="Optional explicit wav paths; overrides --wav-dir.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-ms", type=int, default=800)
    parser.add_argument("--holdback-ms", type=int, default=250)
    parser.add_argument("--min-start-s", type=float, default=2.0)
    parser.add_argument(
        "--offset-s",
        type=float,
        required=True,
        help=(
            "Scalar added to the already-shipping-calibrated Gemma word-end "
            "times before frontier evaluation. Pass the fixed value chosen on "
            "a held-out calibration split; do not fit on the same set here."
        ),
    )
    parser.add_argument("--output-path", required=True, type=Path)
    args = parser.parse_args()

    teacher_dir = args.bundle_root / args.teacher_tag
    proxy_dir = args.bundle_root / args.proxy_tag
    if not teacher_dir.exists():
        raise SystemExit(f"Missing teacher bundle directory: {teacher_dir}")
    if not proxy_dir.exists():
        raise SystemExit(f"Missing proxy bundle directory: {proxy_dir}")

    stems = _select_stems(args, teacher_dir)
    if not stems:
        raise SystemExit("No wavs selected.")

    per_wav_matches: list[dict[str, Any]] = []
    all_matches: list[dict[str, Any]] = []
    for stem in stems:
        teacher_path = teacher_dir / f"{stem}.json"
        proxy_path = proxy_dir / f"{stem}.json"
        if not teacher_path.exists():
            raise SystemExit(f"Missing teacher bundle for {stem}: {teacher_path}")
        if not proxy_path.exists():
            raise SystemExit(f"Missing proxy bundle for {stem}: {proxy_path}")
        teacher_payload = _load_payload(teacher_path)
        proxy_payload = _load_payload(proxy_path)
        matches = _match_word_pieces(
            teacher_words=teacher_payload.get("words") or [],
            proxy_words=proxy_payload.get("words") or [],
        )
        all_matches.extend(matches)
        per_wav_matches.append(
            {
                "wav_name": teacher_path.name,
                "wav_path": str(teacher_payload.get("wav_path") or proxy_payload.get("wav_path") or stem),
                "matched_pieces": matches,
                "teacher_text": str(teacher_payload.get("text", "")),
                "proxy_text": str(proxy_payload.get("text", "")),
            }
        )

    total_offset_s = float(args.offset_s)

    per_wav_reports: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for report in per_wav_matches:
        timestamp_summary = _summarize_timestamp_residuals(
            report["matched_pieces"],
            offset_s=total_offset_s,
        )
        frontier_summary = _summarize_frontier_decisions(
            report["matched_pieces"],
            offset_s=total_offset_s,
            chunk_ms=int(args.chunk_ms),
            holdback_ms=int(args.holdback_ms),
            min_start_s=float(args.min_start_s),
        )
        per_wav_reports.append(
            {
                "wav_name": report["wav_name"],
                "wav_path": report["wav_path"],
                "matched_piece_count": int(len(report["matched_pieces"])),
                "timestamp": timestamp_summary,
                "frontier": frontier_summary,
            }
        )
        all_rows.extend(report["matched_pieces"])
        print(
            f"[report] {report['wav_name']} matched={len(report['matched_pieces'])} "
            f"ts_mae={timestamp_summary['mae_s']:.3f}s "
            f"frontier_exact={frontier_summary['exact_commit_ratio']:.3f} "
            f"early={frontier_summary['early_commit_ratio']:.3f} "
            f"late={frontier_summary['late_commit_ratio']:.3f}",
            flush=True,
        )

    aggregate = {
        "timestamp": _summarize_timestamp_residuals(
            all_rows,
            offset_s=total_offset_s,
        ),
        "frontier": _summarize_frontier_decisions(
            all_rows,
            offset_s=total_offset_s,
            chunk_ms=int(args.chunk_ms),
            holdback_ms=int(args.holdback_ms),
            min_start_s=float(args.min_start_s),
        ),
    }

    payload = {
        "bundle_root": str(args.bundle_root),
        "teacher_tag": args.teacher_tag,
        "proxy_tag": args.proxy_tag,
        "wav_count": int(len(stems)),
        "chunk_ms": int(args.chunk_ms),
        "holdback_ms": int(args.holdback_ms),
        "min_start_s": float(args.min_start_s),
        "calibration": {
            "offset_s": float(total_offset_s),
            "note": (
                "Applied on top of the shipping word_end_offset_seconds already "
                "baked into the Gemma proxy bundles."
            ),
        },
        "aggregate": aggregate,
        "per_wav": per_wav_reports,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[done] wrote {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
