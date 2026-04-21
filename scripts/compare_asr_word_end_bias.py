#!/usr/bin/env python3
"""Compare Gemma AlignAtt ASR word-end times against the Qwen forced-align teacher.

This script is deliberately batch-oriented and hot-backend friendly:

- load the Qwen teacher once and emit one alignment bundle per wav
- release it
- load the Gemma AlignAtt backend once and emit one alignment bundle per wav
- compare the resulting word-end times with a sequence matcher over
  normalized word pieces

The main diagnostic is the signed word-end bias

    signed_error_s = gemma_end_time_s - qwen_end_time_s

Negative values mean Gemma AlignAtt anchored the word too far to the
left (earlier than the teacher), which artificially inflates
`commit_time - acoustic_end_time` latency estimates.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import statistics
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from run_alignment_single_audio import (  # noqa: E402
    build_gemma_vllm_backend,
    build_qwen_backend,
    build_runtime_config,
    load_wav,
    serialize_alignment,
)


_WORD_PIECE_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
DEFAULT_GEMMA_AUDIO_HEADS_PATH = build_runtime_config().gemma_audio_alignment_heads_path


@dataclass(frozen=True)
class ExpandedWordPiece:
    raw_text: str
    normalized_piece: str
    end_time_s: float
    raw_word_index: int
    piece_index: int


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute a quantile over an empty sequence.")
    idx = min(len(sorted_values) - 1, max(0, int(q * (len(sorted_values) - 1))))
    return float(sorted_values[idx])


def _expand_alignment_words(words: Sequence[dict[str, Any]]) -> list[ExpandedWordPiece]:
    """Expand one aligned word into one or more normalized comparison pieces.

    Hyphenated compounds such as `step-by-step` are split into separate
    comparison pieces that all share the same end time. This keeps the
    sequence matcher usable even when one backend emits compounds and
    the other emits separate surface words.
    """
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
    qwen_words: Sequence[dict[str, Any]],
    gemma_words: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    qwen_expanded = _expand_alignment_words(qwen_words)
    gemma_expanded = _expand_alignment_words(gemma_words)
    matcher = SequenceMatcher(
        a=[piece.normalized_piece for piece in qwen_expanded],
        b=[piece.normalized_piece for piece in gemma_expanded],
        autojunk=False,
    )

    matches: list[dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            continue
        span_len = min(i2 - i1, j2 - j1)
        for offset in range(span_len):
            qwen_piece = qwen_expanded[i1 + offset]
            gemma_piece = gemma_expanded[j1 + offset]
            signed_error_s = float(gemma_piece.end_time_s) - float(qwen_piece.end_time_s)
            matches.append(
                {
                    "normalized_piece": qwen_piece.normalized_piece,
                    "qwen_raw_text": qwen_piece.raw_text,
                    "gemma_raw_text": gemma_piece.raw_text,
                    "qwen_raw_word_index": int(qwen_piece.raw_word_index),
                    "gemma_raw_word_index": int(gemma_piece.raw_word_index),
                    "qwen_end_time_s": float(qwen_piece.end_time_s),
                    "gemma_end_time_s": float(gemma_piece.end_time_s),
                    "signed_error_s": float(signed_error_s),
                    "abs_error_s": abs(float(signed_error_s)),
                }
            )
    return matches


def _summarize_matches(matches: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not matches:
        return {
            "matched_piece_count": 0,
            "signed_mean_s": None,
            "signed_median_s": None,
            "signed_p10_s": None,
            "signed_p90_s": None,
            "mae_s": None,
            "abs_p90_s": None,
            "early_bias_ratio": None,
            "late_bias_ratio": None,
            "early_over_250ms_ratio": None,
            "early_over_500ms_ratio": None,
            "early_over_1000ms_ratio": None,
            "worst_early_examples": [],
            "worst_late_examples": [],
        }

    signed = [float(match["signed_error_s"]) for match in matches]
    absolute = [abs(value) for value in signed]
    signed_sorted = sorted(signed)
    abs_sorted = sorted(absolute)

    early = sum(value < 0.0 for value in signed)
    late = sum(value > 0.0 for value in signed)
    worst_early = sorted(matches, key=lambda row: row["signed_error_s"])[:12]
    worst_late = sorted(matches, key=lambda row: row["signed_error_s"], reverse=True)[:12]

    return {
        "matched_piece_count": int(len(matches)),
        "signed_mean_s": float(sum(signed) / len(signed)),
        "signed_median_s": float(statistics.median(signed)),
        "signed_p10_s": _quantile(signed_sorted, 0.10),
        "signed_p90_s": _quantile(signed_sorted, 0.90),
        "mae_s": float(sum(absolute) / len(absolute)),
        "abs_p90_s": _quantile(abs_sorted, 0.90),
        "early_bias_ratio": float(early / len(signed)),
        "late_bias_ratio": float(late / len(signed)),
        "early_over_250ms_ratio": float(sum(value <= -0.25 for value in signed) / len(signed)),
        "early_over_500ms_ratio": float(sum(value <= -0.50 for value in signed) / len(signed)),
        "early_over_1000ms_ratio": float(sum(value <= -1.00 for value in signed) / len(signed)),
        "worst_early_examples": worst_early,
        "worst_late_examples": worst_late,
    }


def compare_alignment_payloads(
    *,
    qwen_payload: dict[str, Any],
    gemma_payload: dict[str, Any],
) -> dict[str, Any]:
    qwen_words = list(qwen_payload.get("words") or [])
    gemma_words = list(gemma_payload.get("words") or [])
    matches = _match_word_pieces(
        qwen_words=qwen_words,
        gemma_words=gemma_words,
    )
    summary = _summarize_matches(matches)
    summary.update(
        {
            "qwen_text": str(qwen_payload.get("text", "")),
            "gemma_text": str(gemma_payload.get("text", "")),
            "qwen_word_count": int(len(qwen_words)),
            "gemma_word_count": int(len(gemma_words)),
            "qwen_expanded_piece_count": int(len(_expand_alignment_words(qwen_words))),
            "gemma_expanded_piece_count": int(len(_expand_alignment_words(gemma_words))),
        }
    )
    return summary


def _write_bundle(
    output_path: Path,
    *,
    tag: str,
    wav_path: Path,
    result_payload: dict[str, Any],
) -> None:
    payload = {
        "tag": str(tag),
        "wav_path": str(wav_path),
        **result_payload,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _maybe_release_backend(backend: Any) -> None:
    try:
        llm = getattr(backend, "llm", None)
        if llm is not None and hasattr(llm, "shutdown"):
            llm.shutdown()
    except Exception:
        pass
    del backend
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _iter_wavs(args: argparse.Namespace) -> list[Path]:
    if args.wavs:
        wavs = [Path(path) for path in args.wavs]
    else:
        wavs = sorted(Path(args.wav_dir).glob("*.wav"))
    if args.limit is not None:
        wavs = wavs[: int(args.limit)]
    return wavs


def _run_backend_over_wavs(
    *,
    wavs: Sequence[Path],
    output_dir: Path,
    backend_name: str,
    language: str,
    top_k: int,
    heads_path: str | None,
) -> None:
    bundle_dir = output_dir / backend_name
    bundle_dir.mkdir(parents=True, exist_ok=True)
    pending = [wav for wav in wavs if not (bundle_dir / f"{wav.stem}.json").exists()]
    if not pending:
        print(f"[{backend_name}] reusing {len(wavs)} existing bundles", flush=True)
        return

    if backend_name == "qwen_forced":
        backend = build_qwen_backend()
    elif backend_name == "gemma_vllm_qk_fast":
        backend = build_gemma_vllm_backend(
            heads_path=heads_path,
            top_k=int(top_k),
        )
    else:
        raise ValueError(f"Unknown backend_name {backend_name!r}")

    try:
        for wav in pending:
            print(f"[{backend_name}] {wav.name}", flush=True)
            audio, sample_rate = load_wav(str(wav))
            result = backend.transcribe_and_align(
                audio,
                sample_rate=sample_rate,
                language=language,
            )
            if result is None:
                raise RuntimeError(f"{backend_name} produced no result for {wav}")
            _write_bundle(
                bundle_dir / f"{wav.stem}.json",
                tag=backend_name,
                wav_path=wav,
                result_payload=serialize_alignment(result),
            )
    finally:
        _maybe_release_backend(backend)


def _load_payloads(bundle_dir: Path, wavs: Sequence[Path]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for wav in wavs:
        path = bundle_dir / f"{wav.stem}.json"
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    return payloads


def _aggregate_reports(per_wav_reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    all_matches: list[dict[str, Any]] = []
    for report in per_wav_reports:
        all_matches.extend(report.get("matched_pieces", []))
    summary = _summarize_matches(all_matches)
    summary["wav_count"] = int(len(per_wav_reports))
    summary["matched_piece_total"] = int(len(all_matches))
    summary["wavs_with_matches"] = int(
        sum(int(report["summary"]["matched_piece_count"]) > 0 for report in per_wav_reports)
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wav-dir",
        default=str(REPO_ROOT / "data/devset/audio"),
        help="Directory of wavs to compare when --wavs is omitted.",
    )
    parser.add_argument(
        "--wavs",
        nargs="*",
        default=None,
        help="Explicit wav paths to compare. Overrides --wav-dir.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--language", default="English")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument(
        "--heads-path",
        default=DEFAULT_GEMMA_AUDIO_HEADS_PATH,
        help=(
            "Gemma ASR AlignAtt heads JSON. Defaults to the canonical runtime "
            "heads path."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    wavs = _iter_wavs(args)
    if not wavs:
        raise SystemExit("No wavs selected.")

    print(f"[plan] comparing {len(wavs)} wav(s)", flush=True)
    _run_backend_over_wavs(
        wavs=wavs,
        output_dir=args.output_dir,
        backend_name="qwen_forced",
        language=args.language,
        top_k=int(args.top_k),
        heads_path=args.heads_path,
    )
    _run_backend_over_wavs(
        wavs=wavs,
        output_dir=args.output_dir,
        backend_name="gemma_vllm_qk_fast",
        language=args.language,
        top_k=int(args.top_k),
        heads_path=args.heads_path,
    )

    qwen_payloads = _load_payloads(args.output_dir / "qwen_forced", wavs)
    gemma_payloads = _load_payloads(args.output_dir / "gemma_vllm_qk_fast", wavs)

    per_wav_reports: list[dict[str, Any]] = []
    for wav, qwen_payload, gemma_payload in zip(wavs, qwen_payloads, gemma_payloads):
        matches = _match_word_pieces(
            qwen_words=qwen_payload.get("words") or [],
            gemma_words=gemma_payload.get("words") or [],
        )
        summary = compare_alignment_payloads(
            qwen_payload=qwen_payload,
            gemma_payload=gemma_payload,
        )
        per_wav_reports.append(
            {
                "wav_name": wav.name,
                "summary": summary,
                "matched_pieces": matches,
            }
        )
        print(
            f"[report] {wav.name} matched={summary['matched_piece_count']} "
            f"mae={summary['mae_s']:.3f}s "
            f"signed_mean={summary['signed_mean_s']:.3f}s "
            f"early>500ms={summary['early_over_500ms_ratio']:.3f}",
            flush=True,
        )

    aggregate = _aggregate_reports(per_wav_reports)
    summary_path = args.output_dir / "word_end_bias_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "wav_count": len(wavs),
                "language": args.language,
                "top_k": int(args.top_k),
                "aggregate": aggregate,
                "per_wav": per_wav_reports,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[done] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
