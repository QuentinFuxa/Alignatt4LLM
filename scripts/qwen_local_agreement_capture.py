#!/usr/bin/env python3
"""Capture Qwen local-agreement traces with reinjected stable prefix.

The strategy is intentionally explicit:

1. decode every chunk on the cumulative audio prefix seen so far;
2. turn the char-level LCP between consecutive full hypotheses into a
   whole-word stable prefix;
3. keep that prefix append-only and re-inject it into the next Qwen prompt;
4. force-align the current full hypothesis so each source word carries a
   timestamp we can later use for holdback-x simulation.

This produces the "base" local-agreement run we want to analyse, rather than
the simpler cumulative-from-scratch probe that ignores prefix reinjection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade.alignment.qwen_local_agreement import QwenLocalAgreementStream  # noqa: E402
from run_alignment_single_audio import build_qwen_backend, load_wav  # noqa: E402


def _load_unique_wav_names(segmentation_path: Path) -> list[str]:
    payload = yaml.safe_load(segmentation_path.read_text(encoding="utf-8")) or []
    seen: set[str] = set()
    ordered: list[str] = []
    for row in payload:
        wav_name = Path(str(row["wav"])).name
        if wav_name in seen:
            continue
        seen.add(wav_name)
        ordered.append(wav_name)
    return ordered


def iter_wavs(args: argparse.Namespace) -> list[Path]:
    if args.wavs:
        wavs = [Path(path) for path in args.wavs]
    else:
        wav_root = Path(args.wav_dir)
        wavs = [
            wav_root / wav_name
            for wav_name in _load_unique_wav_names(Path(args.segmentation))
        ]
    if args.limit is not None:
        wavs = wavs[: int(args.limit)]
    missing = [str(path) for path in wavs if not path.exists()]
    if missing:
        raise SystemExit("Missing wavs for capture:\n- " + "\n- ".join(missing))
    return wavs


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_capture_payload(
    *,
    wav_path: Path,
    chunk_ms: int,
    min_start_seconds: float,
    language: str,
    audio_duration_s: float,
    processing_s: float,
    effective_audio_processed_s: float,
    rows: list[dict[str, Any]],
    stream: QwenLocalAgreementStream,
) -> dict[str, Any]:
    final_row = rows[-1]
    return {
        "schema_version": "qwen_local_agreement_capture_v2",
        "strategy": {
            "decode_mode": "assistant_prefix_reinjection",
            "stable_prefix_rule": "whole_word_lcp_between_consecutive_hypotheses",
            "commit_rule": "append_only_local_agreement",
            "timestamp_rule": "forced_aligner_on_current_full_hypothesis",
        },
        "wav_path": str(wav_path),
        "wav_name": wav_path.name,
        "language": str(language),
        "chunk_ms": int(chunk_ms),
        "min_start_seconds": float(min_start_seconds),
        "audio_duration_s": float(audio_duration_s),
        "effective_audio_processed_s": float(effective_audio_processed_s),
        "processing_s": float(processing_s),
        "rtf_wallclock": (
            0.0 if audio_duration_s <= 0.0 else float(processing_s / audio_duration_s)
        ),
        "rtf_effective_prefix_audio": (
            0.0
            if effective_audio_processed_s <= 0.0
            else float(processing_s / effective_audio_processed_s)
        ),
        "final_text": str(final_row.get("hypothesis_text", "")).strip(),
        "final_words": list(final_row.get("words") or []),
        "local_agreement_final_text": str(final_row.get("local_agreement_candidate_text", "")).strip(),
        "local_agreement_committed_words": list(stream.public_committed_words),
        "local_agreement_stream_trace": list(stream.stream_trace),
        "chunks": rows,
    }


def capture_wav(
    *,
    backend: Any,
    wav_path: Path,
    language: str,
    chunk_ms: int,
    min_start_seconds: float,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    audio, sample_rate = load_wav(str(wav_path))
    chunk_size = max(1, int(sample_rate * float(chunk_ms) / 1000.0))
    audio_duration_s = float(len(audio)) / float(sample_rate)

    stream = QwenLocalAgreementStream(asr=backend.asr, language=language)
    rows: list[dict[str, Any]] = []
    effective_audio_processed_s = 0.0
    processing_start = perf_counter()

    for chunk_idx, stop_sample in enumerate(
        range(chunk_size, len(audio) + chunk_size, chunk_size),
        start=1,
    ):
        stop_sample = min(stop_sample, len(audio))
        audio_processed_s = float(stop_sample) / float(sample_rate)
        is_final_chunk = stop_sample >= len(audio)
        if audio_processed_s < float(min_start_seconds) and not is_final_chunk:
            continue

        effective_audio_processed_s += float(audio_processed_s)
        tail_start_sample = max(
            0,
            min(
                stop_sample,
                int(round(float(stream.tail_start_time_s) * float(sample_rate))),
            ),
        )
        audio_tail = audio[tail_start_sample:stop_sample]
        wallclock_s = perf_counter() - processing_start
        try:
            row = stream.step(
                audio_tail=audio_tail,
                sample_rate=sample_rate,
                chunk_idx=int(chunk_idx),
                audio_processed_s=float(audio_processed_s),
                wallclock_s=float(wallclock_s),
                is_final_chunk=bool(is_final_chunk),
            )
        except Exception as exc:
            raise type(exc)(
                f"{exc} | capture_context="
                f"wav={wav_path.name} chunk_idx={chunk_idx} "
                f"audio_processed_s={audio_processed_s:.3f}"
            ) from exc

        rows.append(row)
        if checkpoint_path is not None:
            write_json(
                checkpoint_path,
                build_capture_payload(
                    wav_path=wav_path,
                    chunk_ms=chunk_ms,
                    min_start_seconds=min_start_seconds,
                    language=language,
                    audio_duration_s=audio_duration_s,
                    processing_s=float(row["wallclock_s"]),
                    effective_audio_processed_s=effective_audio_processed_s,
                    rows=rows,
                    stream=stream,
                ),
            )

    if not rows:
        raise RuntimeError(f"Capture produced no valid chunk for {wav_path}.")

    processing_s = float(rows[-1]["wallclock_s"])
    return build_capture_payload(
        wav_path=wav_path,
        chunk_ms=chunk_ms,
        min_start_seconds=min_start_seconds,
        language=language,
        audio_duration_s=audio_duration_s,
        processing_s=processing_s,
        effective_audio_processed_s=effective_audio_processed_s,
        rows=rows,
        stream=stream,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wav-dir",
        default=str(REPO_ROOT / "data/devset/audio"),
        help="Directory containing the MCIF wav files when --wavs is omitted.",
    )
    parser.add_argument(
        "--wavs",
        nargs="*",
        default=None,
        help="Explicit wav paths to capture. Overrides --wav-dir and --segmentation.",
    )
    parser.add_argument(
        "--segmentation",
        default=str(REPO_ROOT / "mcif-long-trans" / "audio-segments.yaml"),
        help="Segmentation file used to enumerate the MCIF wav set.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunk-ms", type=int, default=800)
    parser.add_argument("--min-start-seconds", type=float, default=2.0)
    parser.add_argument("--language", default="English")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    wavs = iter_wavs(args)
    captures_dir = Path(args.output_dir) / "captures"
    captures_dir.mkdir(parents=True, exist_ok=True)

    backend = build_qwen_backend()
    if getattr(backend, "asr", None) is None:
        raise RuntimeError("Qwen backend did not expose a loaded ASR model.")

    rows: list[dict[str, Any]] = []
    for idx, wav_path in enumerate(wavs, start=1):
        print(f"[capture] ({idx}/{len(wavs)}) {wav_path.name}", flush=True)
        partial_path = captures_dir / f"{wav_path.stem}.partial"
        capture = capture_wav(
            backend=backend,
            wav_path=wav_path,
            language=args.language,
            chunk_ms=int(args.chunk_ms),
            min_start_seconds=float(args.min_start_seconds),
            checkpoint_path=partial_path,
        )
        out_path = captures_dir / f"{wav_path.stem}.json"
        write_json(out_path, capture)
        if partial_path.exists():
            partial_path.unlink()

        rows.append(
            {
                "wav_name": wav_path.name,
                "wav_path": str(wav_path),
                "chunk_row_count": len(capture["chunks"]),
                "append_only_conflict_chunk_count": int(
                    sum(
                        bool(row.get("append_only_conflict", False))
                        for row in capture["chunks"]
                    )
                ),
                "alignment_mismatch_chunk_count": int(
                    sum(
                        not bool(row.get("alignment_count_match", False))
                        for row in capture["chunks"]
                    )
                ),
                "audio_duration_s": float(capture["audio_duration_s"]),
                "effective_audio_processed_s": float(
                    capture["effective_audio_processed_s"]
                ),
                "processing_s": float(capture["processing_s"]),
                "rtf_wallclock": float(capture["rtf_wallclock"]),
                "rtf_effective_prefix_audio": float(
                    capture["rtf_effective_prefix_audio"]
                ),
                "final_word_count": len(capture["final_words"]),
                "local_agreement_final_word_count": len(
                    capture["local_agreement_committed_words"]
                ),
            }
        )
        print(
            f"    chunks={rows[-1]['chunk_row_count']} "
            f"align_mismatch={rows[-1]['alignment_mismatch_chunk_count']} "
            f"conflicts={rows[-1]['append_only_conflict_chunk_count']} "
            f"rtf_wall={rows[-1]['rtf_wallclock']:.2f} "
            f"rtf_eff={rows[-1]['rtf_effective_prefix_audio']:.4f}",
            flush=True,
        )

    manifest = {
        "schema_version": "qwen_local_agreement_capture_manifest_v2",
        "wav_count": len(wavs),
        "chunk_ms": int(args.chunk_ms),
        "min_start_seconds": float(args.min_start_seconds),
        "segmentation": str(args.segmentation),
        "wav_dir": str(args.wav_dir),
        "captures_dir": str(captures_dir),
        "rows": rows,
    }
    write_json(Path(args.output_dir) / "manifest.json", manifest)
    print(f"[done] wrote {Path(args.output_dir) / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
