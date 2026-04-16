#!/usr/bin/env python
"""Single-audio diagnostic harness for the Gemma-only aligner research path.

This is the tool the PLAN.md calls for in Phase 2 (baseline bundle) and
Phase 9 (ablations): collect, on one carefully chosen clip, what each
alignment backend produces so that head selection, quality, and streaming
behavior can be diagnosed manually before any broader run.

Modes:

- ``baseline`` produces the Qwen teacher bundle for a clip and writes it
  as a JSON artifact.
- ``gemma_inspect`` runs the Gemma attention aligner on the same clip,
  using previously calibrated heads if available, and writes its
  transcript + word timestamps + per-token audio positions.
- ``gemma_forced_align`` runs teacher-forced Gemma alignment on a known
  transcript; ``--probe-mode eager|qk_fast`` selects the attention
  extraction backend explicitly.
- ``gemma_calibrate_heads`` runs Gemma once with full-attention capture
  and ranks every (layer, head) against the Qwen teacher, writing the
  top-K into ``assets/attention_heads/audio_alignment_heads_*.json``.
- ``compare`` loads both bundles and prints a simple word-end-MAE table.

Run with the already-hot ``.venv-inference`` kernel — model reloads are
expensive, so prefer reusing a warm process.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np

from alignment_backend import AlignmentBackend, AlignmentResult, WordAlignment


def load_wav(path: str) -> tuple[np.ndarray, int]:
    import wave

    with wave.open(path, "rb") as wav:
        sr = wav.getframerate()
        width = wav.getsampwidth()
        ch = wav.getnchannels()
        raw = wav.readframes(wav.getnframes())
    if width != 2:
        raise ValueError("Only 16-bit PCM WAV is supported in this harness.")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        duration = len(audio) / sr
        new_length = int(duration * 16000)
        old_times = np.linspace(0.0, duration, num=len(audio), endpoint=False)
        new_times = np.linspace(0.0, duration, num=new_length, endpoint=False)
        audio = np.interp(new_times, old_times, audio).astype(np.float32)
        sr = 16000
    return audio, sr


def serialize_alignment(result: AlignmentResult) -> dict:
    return {
        "text": result.text,
        "audio_duration_s": result.audio_duration_s,
        "words": [asdict(w) for w in result.words],
        "diagnostics": {
            k: v for k, v in result.diagnostics.items() if _is_jsonable(v)
        },
    }


def _is_jsonable(value) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def deserialize_alignment(payload: dict) -> AlignmentResult:
    words = tuple(
        WordAlignment(
            text=entry["text"],
            start_time=float(entry["start_time"]),
            end_time=float(entry["end_time"]),
        )
        for entry in payload.get("words", [])
    )
    return AlignmentResult(
        text=str(payload.get("text", "")),
        words=words,
        audio_duration_s=float(payload.get("audio_duration_s", 0.0)),
        diagnostics=dict(payload.get("diagnostics", {})),
    )


def build_runtime_config() -> SimpleNamespace:
    from qwen3asr_gemma_cascade_core import config

    return SimpleNamespace(**vars(config))


def build_qwen_backend() -> AlignmentBackend:
    from qwen_alignment_backend import QwenAlignmentBackend
    from qwen3asr_gemma_cascade_core import (
        asr_model_name,
        forced_aligner_model_name,
    )

    backend = QwenAlignmentBackend(
        asr_model_path=asr_model_name,
        forced_aligner_model_path=forced_aligner_model_name,
        runtime_config=build_runtime_config(),
    )
    backend.load()
    return backend


def build_gemma_backend(
    *,
    heads_path: str | None,
    top_k: int,
    probe_mode: str | None = None,
):
    from gemma_alignment_probe import GemmaAttentionAlignmentBackend
    from qwen3asr_gemma_cascade_core import gemma_model_name

    runtime_config = build_runtime_config()
    if probe_mode is not None:
        runtime_config.gemma_audio_align_probe_mode = str(probe_mode)
    backend = GemmaAttentionAlignmentBackend(
        model_name=gemma_model_name,
        runtime_config=runtime_config,
        audio_heads_path=heads_path,
        audio_heads_top_k=top_k,
        filter_width=int(getattr(runtime_config, "gemma_audio_alignment_filter_width", 7)),
        max_new_tokens=int(getattr(runtime_config, "gemma_audio_alignment_max_new_tokens", 256)),
    )
    backend.load()
    return backend


def cmd_baseline(args: argparse.Namespace) -> None:
    audio, sr = load_wav(args.wav)
    backend = build_qwen_backend()
    result = backend.transcribe_and_align(audio, sample_rate=sr, language=args.language)
    if result is None:
        raise RuntimeError("Qwen transcription produced no valid result.")
    _write_bundle(args.output, result, tag="qwen_baseline", wav_path=args.wav)


def cmd_gemma_inspect(args: argparse.Namespace) -> None:
    audio, sr = load_wav(args.wav)
    backend = build_gemma_backend(
        heads_path=args.heads_path or None,
        top_k=int(args.top_k),
    )
    result = backend.transcribe_and_align(audio, sample_rate=sr, language=args.language)
    if result is None:
        raise RuntimeError("Gemma alignment produced no valid result.")
    _write_bundle(args.output, result, tag="gemma_attention", wav_path=args.wav)


def cmd_gemma_forced_align(args: argparse.Namespace) -> None:
    audio, sr = load_wav(args.wav)
    teacher = deserialize_alignment(
        json.loads(Path(args.teacher).read_text(encoding="utf-8"))
    )
    backend = build_gemma_backend(
        heads_path=args.heads_path or None,
        top_k=int(args.top_k),
        probe_mode=args.probe_mode,
    )
    result = backend.align_transcript(
        audio,
        sample_rate=sr,
        language=args.language,
        transcript=teacher.text,
    )
    if result is None:
        raise RuntimeError("Gemma forced alignment produced no valid result.")
    _write_bundle(args.output, result, tag="gemma_forced_align", wav_path=args.wav)


def cmd_gemma_calibrate_heads_forced(args: argparse.Namespace) -> None:
    audio, sr = load_wav(args.wav)
    teacher = deserialize_alignment(
        json.loads(Path(args.teacher).read_text(encoding="utf-8"))
    )
    backend = build_gemma_backend(heads_path=None, top_k=int(args.top_k))
    scored = backend.calibrate_alignment_heads_forced(
        audio,
        sample_rate=sr,
        language=args.language,
        teacher=teacher,
        top_k=None,
    )
    top_heads = scored[: int(args.top_k)]

    # Fit a single systematic word-end offset on the selected heads: a
    # causal LLM's attention peak lags the acoustic boundary. Subtract the
    # median signed error on this one teacher so the heads file ships with
    # its own calibration and downstream consumers don't need to re-fit.
    offset_s = _estimate_word_end_offset(
        backend=backend,
        audio=audio,
        sample_rate=sr,
        language=args.language,
        teacher=teacher,
        top_heads=top_heads,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from gemma_alignment_probe import save_audio_alignment_heads

    save_audio_alignment_heads(
        str(out_path),
        scored_heads=top_heads,
        model_name=backend.model_name,
        language=args.language,
        word_end_offset_seconds=offset_s,
        scoring_notes={
            "teacher": str(Path(args.teacher)),
            "teacher_audio_duration_s": teacher.audio_duration_s,
            "num_heads_scored": len(scored),
            "filter_width": backend.filter_width,
            "mode": "forced_alignment",
            "notes": (
                "Teacher-forced per-token attention; MAE is on aligned words (same "
                "sequence), so it is a true alignment-quality signal rather than a "
                "transcription-quality artifact. The word_end_offset_seconds value "
                "is the median signed lag between the selected-heads prediction and "
                "the teacher on this clip."
            ),
        },
    )
    ranking_path = out_path.with_suffix(".full_ranking.json")
    ranking_path.write_text(json.dumps(scored, indent=2), encoding="utf-8")
    print(f"Wrote heads file: {out_path}  (offset={offset_s:.3f}s)")
    print(f"Wrote full scoring ranking: {ranking_path}")


def _estimate_word_end_offset(
    *,
    backend,
    audio,
    sample_rate: int,
    language: str,
    teacher: AlignmentResult,
    top_heads: list[dict],
) -> float:
    import statistics

    from gemma_alignment_probe import AlignAttHead

    # Temporarily install the selected heads on the backend and run one
    # forced alignment with offset=0, then fit the offset to the teacher.
    prev_heads = list(backend.alignatt_heads)
    prev_offset = backend.word_end_offset_s
    prev_recorder = backend.alignatt_recorder
    prev_layer_input_recorder = getattr(backend, "alignatt_layer_input_recorder", None)
    try:
        backend.alignatt_heads = [
            AlignAttHead(layer=int(h["layer"]), head=int(h["head"]), ts=float(h["ts"]))
            for h in top_heads
        ]
        backend.word_end_offset_s = 0.0
        backend.alignatt_recorder = None  # let align_transcript rebuild
        backend.alignatt_layer_input_recorder = None  # idem for qk_fast replay
        result = backend.align_transcript(
            audio,
            sample_rate=sample_rate,
            language=language,
            transcript=teacher.text,
        )
        if result is None or not result.words:
            return 0.0
        n = min(len(result.words), len(teacher.words))
        diffs = [
            result.words[i].end_time - teacher.words[i].end_time for i in range(n)
        ]
        if not diffs:
            return 0.0
        return float(statistics.median(diffs))
    finally:
        backend.alignatt_heads = prev_heads
        backend.word_end_offset_s = prev_offset
        backend.alignatt_recorder = prev_recorder
        backend.alignatt_layer_input_recorder = prev_layer_input_recorder


def cmd_gemma_calibrate_heads(args: argparse.Namespace) -> None:
    audio, sr = load_wav(args.wav)
    teacher_path = Path(args.teacher)
    teacher = deserialize_alignment(json.loads(teacher_path.read_text(encoding="utf-8")))

    backend = build_gemma_backend(heads_path=None, top_k=int(args.top_k))
    scored = backend.calibrate_alignment_heads(
        audio,
        sample_rate=sr,
        language=args.language,
        teacher=teacher,
        top_k=None,  # keep the full ranking in the diagnostic output
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from gemma_alignment_probe import save_audio_alignment_heads

    save_audio_alignment_heads(
        str(out_path),
        scored_heads=scored[: int(args.top_k)],
        model_name=backend.model_name,
        language=args.language,
        scoring_notes={
            "teacher": str(teacher_path),
            "teacher_audio_duration_s": teacher.audio_duration_s,
            "num_heads_scored": len(scored),
            "filter_width": backend.filter_width,
            "max_new_tokens": backend.max_new_tokens,
            "notes": (
                "Monotonicity-dominant ranking with Qwen-teacher MAE as a soft "
                "quality term. See gemma_alignment_probe._combine_head_score."
            ),
        },
    )
    # Also dump the full ranking (including low-score heads) for later
    # ablation / analysis — this is cheap and avoids rerunning the scoring.
    ranking_path = out_path.with_suffix(".full_ranking.json")
    ranking_path.write_text(json.dumps(scored, indent=2), encoding="utf-8")
    print(f"Wrote heads file: {out_path}")
    print(f"Wrote full scoring ranking: {ranking_path}")


def cmd_compare(args: argparse.Namespace) -> None:
    qwen = deserialize_alignment(
        json.loads(Path(args.qwen).read_text(encoding="utf-8"))
    )
    gemma = deserialize_alignment(
        json.loads(Path(args.gemma).read_text(encoding="utf-8"))
    )
    report = compare_alignments(qwen, gemma)
    print(json.dumps(report, indent=2))


def compare_alignments(qwen: AlignmentResult, gemma: AlignmentResult) -> dict:
    n = min(len(qwen.words), len(gemma.words))
    word_errors = [
        abs(qwen.words[i].end_time - gemma.words[i].end_time) for i in range(n)
    ]
    report = {
        "qwen_text": qwen.text,
        "gemma_text": gemma.text,
        "qwen_word_count": len(qwen.words),
        "gemma_word_count": len(gemma.words),
        "paired_words": n,
    }
    if word_errors:
        sorted_errors = sorted(word_errors)
        report.update(
            {
                "word_end_mae_seconds": float(sum(word_errors) / len(word_errors)),
                "word_end_median_error_seconds": float(
                    sorted_errors[len(sorted_errors) // 2]
                ),
                "word_end_p90_error_seconds": float(
                    sorted_errors[min(len(sorted_errors) - 1, int(len(sorted_errors) * 0.9))]
                ),
                "per_word_end_errors_seconds": word_errors,
            }
        )
    return report


def _write_bundle(
    output_path: str | Path,
    result: AlignmentResult,
    *,
    tag: str,
    wav_path: str,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tag": tag,
        "wav_path": wav_path,
        **serialize_alignment(result),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {tag} bundle: {path}")


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(required=True, dest="cmd")

    baseline = subparsers.add_parser("baseline", help="Run Qwen baseline")
    baseline.add_argument("--wav", required=True)
    baseline.add_argument("--language", default="English")
    baseline.add_argument("--output", required=True)
    baseline.set_defaults(func=cmd_baseline)

    inspect = subparsers.add_parser("gemma_inspect", help="Run Gemma aligner with heads")
    inspect.add_argument("--wav", required=True)
    inspect.add_argument("--language", default="English")
    inspect.add_argument("--heads-path", default=None)
    inspect.add_argument("--top-k", type=int, default=8)
    inspect.add_argument("--output", required=True)
    inspect.set_defaults(func=cmd_gemma_inspect)

    cal = subparsers.add_parser(
        "gemma_calibrate_heads",
        help="Score every (layer, head) against a Qwen teacher bundle and save top-K",
    )
    cal.add_argument("--wav", required=True)
    cal.add_argument("--language", default="English")
    cal.add_argument("--teacher", required=True)
    cal.add_argument("--output", required=True)
    cal.add_argument("--top-k", type=int, default=8)
    cal.set_defaults(func=cmd_gemma_calibrate_heads)

    cal_forced = subparsers.add_parser(
        "gemma_calibrate_heads_forced",
        help="Forced-alignment head ranking: teacher transcript is teacher-forced into Gemma, MAE is on identical word sequences.",
    )
    cal_forced.add_argument("--wav", required=True)
    cal_forced.add_argument("--language", default="English")
    cal_forced.add_argument("--teacher", required=True)
    cal_forced.add_argument("--output", required=True)
    cal_forced.add_argument("--top-k", type=int, default=8)
    cal_forced.set_defaults(func=cmd_gemma_calibrate_heads_forced)

    forced = subparsers.add_parser(
        "gemma_forced_align",
        help="Forced alignment: take the teacher's transcript and produce Gemma word timestamps via attention.",
    )
    forced.add_argument("--wav", required=True)
    forced.add_argument("--language", default="English")
    forced.add_argument("--teacher", required=True)
    forced.add_argument("--heads-path", default=None)
    forced.add_argument("--top-k", type=int, default=8)
    forced.add_argument(
        "--probe-mode",
        choices=("eager", "qk_fast"),
        default="eager",
        help="Forced-alignment attention extraction backend.",
    )
    forced.add_argument("--output", required=True)
    forced.set_defaults(func=cmd_gemma_forced_align)

    compare = subparsers.add_parser("compare", help="Compare two alignment bundles")
    compare.add_argument("--qwen", required=True)
    compare.add_argument("--gemma", required=True)
    compare.set_defaults(func=cmd_compare)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_cli()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
