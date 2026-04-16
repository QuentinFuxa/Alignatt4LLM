#!/usr/bin/env python3
"""Canonical single-audio SimulStream comparison runner.

Runs the two supported cascades sequentially and in isolated subprocesses:

- ``qwen_forced`` = Qwen3-ASR + Qwen3 Forced Aligner
- ``gemma_onepass_qk_fast`` = Gemma 4 ASR + AlignAtt ``qk_fast`` one pass

Each backend writes a standard inference artifact directory, then this script
consolidates transcript quality and latency diagnostics into a comparison
report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from subprocess import run as run_subprocess
import string
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

from cascade_artifacts import (
    MANIFEST_FILENAME,
    STREAM_UPDATES_FILENAME,
    InferenceArtifacts,
    StreamUpdate,
    final_asr_filename,
    final_translation_filename,
    utc_now_isoformat,
    write_inference_artifacts,
)
from cascade_emission import register_translation_timestamps, register_translation_words
from cascade_runtime import VALID_ALIGNMENT_BACKEND_NAMES
from cascade_simulstream_processor import CascadeAlignAttProcessor, LANGUAGE_CODE_TO_NAME
from cascade_text_surface import split_target_emission_units
from simulstream.server.speech_processors import SAMPLE_RATE


DEFAULT_WAV = "tmp/alignatt_smoke18.wav"
DEFAULT_REFERENCE = "tmp/alignment_research/smoke18_reference.txt"
DEFAULT_OUTPUT_DIR = "outputs/simulstream_compare_smoke18"
BACKEND_IDS = tuple(VALID_ALIGNMENT_BACKEND_NAMES)


def load_wav_raw(path: str) -> np.ndarray:
    import wave as _wave

    with _wave.open(path, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        num_channels = wav_file.getnchannels()
        raw = wav_file.readframes(wav_file.getnframes())
    if sample_width != 2:
        raise ValueError("Only 16-bit PCM WAV supported.")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if num_channels > 1:
        audio = audio.reshape(-1, num_channels).mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        duration = len(audio) / sample_rate
        old_times = np.linspace(0.0, duration, num=len(audio), endpoint=False)
        new_length = int(duration * SAMPLE_RATE)
        new_times = np.linspace(0.0, duration, num=new_length, endpoint=False)
        audio = np.interp(new_times, old_times, audio).astype(np.float32)
    return audio


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _normalize_for_asr_metric(text: str) -> list[str]:
    table = str.maketrans("", "", string.punctuation + "\u201c\u201d\u2018\u2019")
    return text.lower().translate(table).split()


def _levenshtein(ref: Sequence[str], hyp: Sequence[str]) -> int:
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    previous = list(range(len(hyp) + 1))
    for i, ref_item in enumerate(ref, start=1):
        current = [i] + [0] * len(hyp)
        for j, hyp_item in enumerate(hyp, start=1):
            cost = 0 if ref_item == hyp_item else 1
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
        previous = current
    return previous[-1]


def compute_wer(reference: str, hypothesis: str) -> float:
    ref = _normalize_for_asr_metric(reference)
    hyp = _normalize_for_asr_metric(hypothesis)
    return _levenshtein(ref, hyp) / max(1, len(ref))


def compute_cer(reference: str, hypothesis: str) -> float:
    ref = list("".join(_normalize_for_asr_metric(reference)))
    hyp = list("".join(_normalize_for_asr_metric(hypothesis)))
    return _levenshtein(ref, hyp) / max(1, len(ref))


def translation_revision_stats(
    stream_updates: list[dict[str, Any]],
    *,
    target_lang_code: str,
) -> tuple[int, int]:
    revision_updates = 0
    suppressed_units = 0
    previous_units: list[str] = []

    for payload in stream_updates:
        current_units = split_target_emission_units(
            str(payload.get("translation_text", "")),
            target_lang_code=target_lang_code,
        )
        common_prefix_len = 0
        for prev, cur in zip(previous_units, current_units):
            if prev != cur:
                break
            common_prefix_len += 1
        deleted = len(previous_units) - common_prefix_len
        if deleted > 0:
            revision_updates += 1
            suppressed_units += deleted
        previous_units = current_units

    return revision_updates, suppressed_units


def first_nonempty_emission(
    stream_updates: list[dict[str, Any]],
) -> tuple[float | None, float | None]:
    for payload in stream_updates:
        if str(payload.get("translation_text", "")).strip():
            audio_ms = payload.get("audio_processed_ms")
            wallclock_ms = payload.get("wallclock_elapsed_ms")
            return (
                None if audio_ms is None else float(audio_ms) / 1000.0,
                None if wallclock_ms is None else float(wallclock_ms) / 1000.0,
            )
    return None, None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_processor_config(args: argparse.Namespace, *, backend_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        source_lang_code=args.source,
        target_lang_code=args.target,
        chunk_ms=args.chunk_ms,
        speech_chunk_size=args.chunk_ms / 1000.0,
        alignment_backend_name=backend_name,
        min_start_seconds=args.min_start_seconds,
        max_history_utterances=args.max_history_utterances,
        partial_max_new_tokens=args.partial_max_new_tokens,
        partial_followup_max_new_tokens=args.partial_followup_max_new_tokens,
        translation_alignatt_min_source_mass=args.translation_alignatt_min_source_mass,
        translation_alignatt_rewind_threshold=args.translation_alignatt_rewind_threshold,
        translation_alignatt_inaccessible_ms=args.translation_alignatt_inaccessible_ms,
        gemma_audio_align_probe_mode=args.gemma_audio_align_probe_mode,
    )


def run_single_backend_to_artifacts(
    processor: CascadeAlignAttProcessor,
    *,
    wav_path: str,
    chunk_ms: int,
    source_lang_code: str,
    target_lang_code: str,
    backend_name: str,
    load_ms: float,
) -> tuple[InferenceArtifacts, dict[str, Any]]:
    processor.clear()
    audio = load_wav_raw(wav_path)
    chunk_size = int(SAMPLE_RATE * chunk_ms / 1000)
    audio_duration_ms = len(audio) * 1000.0 / SAMPLE_RATE

    word_delays_ms: list[float] = []
    word_elapsed_ms: list[float] = []
    updates: list[StreamUpdate] = []
    revision_updates = 0
    suppressed_units = 0
    last_translation = ""
    last_raw_translation = ""
    start_time = perf_counter()

    for start_sample in range(0, len(audio), chunk_size):
        chunk = audio[start_sample : start_sample + chunk_size].astype(np.float32)
        output = processor.process_chunk(chunk)
        current_translation = processor.tokens_to_string(processor._emitted_units)
        current_asr = processor.session.render_public_asr_text()
        audio_processed_ms = min((start_sample + chunk_size), len(audio)) * 1000.0 / SAMPLE_RATE
        wallclock_elapsed_ms = (perf_counter() - start_time) * 1000.0

        if output.deleted_tokens:
            revision_updates += 1
            suppressed_units += len(output.deleted_tokens)

        if output.new_tokens or output.deleted_tokens:
            register_translation_timestamps(
                last_raw_translation,
                current_translation,
                wallclock_elapsed_ms,
                word_elapsed_ms,
                target_lang_code=target_lang_code,
            )
            new_words = register_translation_words(
                last_translation,
                current_translation,
                audio_processed_ms,
                word_delays_ms,
                target_lang_code=target_lang_code,
            )
            updates.append(
                StreamUpdate(
                    update_idx=len(updates),
                    audio_processed_ms=audio_processed_ms,
                    wallclock_elapsed_ms=wallclock_elapsed_ms,
                    asr_text=current_asr,
                    translation_text=current_translation,
                    new_words=new_words,
                )
            )
            last_translation = current_translation
            last_raw_translation = current_translation

    eos_output = processor.end_of_stream()
    final_translation = processor.tokens_to_string(processor._emitted_units)
    final_asr = processor.session.render_public_asr_text()
    final_elapsed_ms = (perf_counter() - start_time) * 1000.0
    total_wallclock_s = final_elapsed_ms / 1000.0
    rtf = total_wallclock_s / (audio_duration_ms / 1000.0) if audio_duration_ms > 0 else 0.0

    if eos_output.deleted_tokens:
        revision_updates += 1
        suppressed_units += len(eos_output.deleted_tokens)

    if eos_output.new_tokens or eos_output.deleted_tokens or final_translation != last_translation:
        register_translation_timestamps(
            last_raw_translation,
            final_translation,
            final_elapsed_ms,
            word_elapsed_ms,
            target_lang_code=target_lang_code,
        )
        eos_new_words = register_translation_words(
            last_translation,
            final_translation,
            audio_duration_ms,
            word_delays_ms,
            target_lang_code=target_lang_code,
        )
        last_translation = final_translation
    else:
        eos_new_words = []

    updates.append(
        StreamUpdate(
            update_idx=len(updates),
            audio_processed_ms=audio_duration_ms,
            wallclock_elapsed_ms=final_elapsed_ms,
            asr_text=final_asr,
            translation_text=final_translation,
            new_words=eos_new_words,
            is_eos=True,
        )
    )

    runtime_config = dict(processor.session.config.__dict__)
    runtime_config["alignment_backend_name"] = backend_name
    artifacts = InferenceArtifacts(
        wav_path=wav_path,
        chunk_ms=chunk_ms,
        translation_variant=processor.session.config.translation_variant_id,
        source_language=LANGUAGE_CODE_TO_NAME.get(source_lang_code, source_lang_code),
        target_language=LANGUAGE_CODE_TO_NAME.get(target_lang_code, target_lang_code),
        source_language_code=source_lang_code,
        target_language_code=target_lang_code,
        latency_unit=processor.session.config.latency_unit,
        audio_duration_ms=audio_duration_ms,
        final_asr_text=final_asr,
        final_translation_text=final_translation,
        translation_word_delays_ms=word_delays_ms,
        translation_word_elapsed_ms=word_elapsed_ms,
        updates=updates,
        runtime_config=runtime_config,
        run_provenance={
            "generated_at_utc": utc_now_isoformat(),
            "git_sha": git_sha(),
            "framework_mode": "simulstream_processor",
            "script": "run_simulstream_compare.py",
            "alignment_backend_name": backend_name,
            "model_load_ms": round(load_ms, 2),
            "total_wallclock_s": round(total_wallclock_s, 4),
            "rtf": round(rtf, 6),
            "revision_updates": revision_updates,
            "suppressed_units": suppressed_units,
        },
    )
    return artifacts, {
        "model_load_ms": round(load_ms, 2),
        "total_wallclock_s": round(total_wallclock_s, 4),
        "rtf": round(rtf, 6),
        "num_updates": len(updates),
        "revision_updates": revision_updates,
        "suppressed_units": suppressed_units,
    }


def summarize_backend_artifacts(
    artifact_dir: str | Path,
    *,
    reference_text: str,
) -> dict[str, Any]:
    artifact_path = Path(artifact_dir)
    manifest = load_json(artifact_path / MANIFEST_FILENAME)
    source_lang_code = str(manifest.get("source_language_code", "en"))
    target_lang_code = str(manifest.get("target_language_code", "de"))
    transcript_path = artifact_path / final_asr_filename(source_lang_code)
    translation_path = artifact_path / final_translation_filename(target_lang_code)
    stream_updates_path = artifact_path / STREAM_UPDATES_FILENAME

    final_asr = transcript_path.read_text(encoding="utf-8").strip()
    final_translation = translation_path.read_text(encoding="utf-8").strip()
    stream_updates = load_jsonl(stream_updates_path)
    revision_updates, suppressed_units = translation_revision_stats(
        stream_updates,
        target_lang_code=target_lang_code,
    )
    first_audio_s, first_wallclock_s = first_nonempty_emission(stream_updates)
    audio_duration_ms = float(manifest.get("audio_duration_ms", 0.0))
    provenance = dict(manifest.get("run_provenance", {}) or {})
    total_wallclock_s = provenance.get("total_wallclock_s")
    if total_wallclock_s is None and audio_duration_ms > 0.0:
        total_wallclock_s = None
    load_ms = provenance.get("model_load_ms")
    backend_id = (
        manifest.get("runtime_config", {}) or {}
    ).get("alignment_backend_name") or provenance.get("alignment_backend_name")

    return {
        "backend_id": backend_id,
        "artifact_dir": str(artifact_path),
        "manifest_path": str(artifact_path / MANIFEST_FILENAME),
        "stream_updates_path": str(stream_updates_path),
        "transcript_path": str(transcript_path),
        "translation_path": str(translation_path),
        "final_asr": final_asr,
        "final_translation": final_translation,
        "wer": round(compute_wer(reference_text, final_asr), 6),
        "cer": round(compute_cer(reference_text, final_asr), 6),
        "audio_duration_s": round(audio_duration_ms / 1000.0, 4),
        "model_load_ms": load_ms,
        "total_wallclock_s": total_wallclock_s,
        "rtf": provenance.get("rtf"),
        "num_updates": len(stream_updates),
        "revision_updates": revision_updates,
        "suppressed_units": suppressed_units,
        "first_nonempty_emission_audio_s": first_audio_s,
        "first_nonempty_emission_wallclock_s": first_wallclock_s,
        "has_terminal_eos_update": bool(stream_updates and stream_updates[-1].get("is_eos")),
    }


def build_comparison_report(
    *,
    wav_path: str,
    reference_path: str,
    backend_artifact_dirs: dict[str, str | Path],
) -> dict[str, Any]:
    reference_text = Path(reference_path).read_text(encoding="utf-8").strip()
    backend_reports = [
        summarize_backend_artifacts(path, reference_text=reference_text)
        for _, path in sorted(backend_artifact_dirs.items())
    ]
    return {
        "generated_at_utc": utc_now_isoformat(),
        "kind": "simulstream_compare",
        "wav_path": wav_path,
        "reference_path": reference_path,
        "backend_ids": [report["backend_id"] for report in backend_reports],
        "backends": backend_reports,
    }


def write_comparison_outputs(output_dir: str | Path, report: dict[str, Any]) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "comparison_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"WAV: {report['wav_path']}",
        f"Reference: {report['reference_path']}",
        "",
    ]
    for backend in report["backends"]:
        lines.extend(
            [
                f"[{backend['backend_id']}]",
                f"  WER={backend['wer']:.4f}  CER={backend['cer']:.4f}  "
                f"first_emit_audio_s={backend['first_nonempty_emission_audio_s']}",
                f"  load_ms={backend['model_load_ms']}  wallclock_s={backend['total_wallclock_s']}  "
                f"rtf={backend['rtf']}",
                f"  updates={backend['num_updates']}  revisions={backend['revision_updates']}  "
                f"suppressed_units={backend['suppressed_units']}",
                f"  stream_updates={backend['stream_updates_path']}",
                "",
            ]
        )
    (output_path / "comparison_report.txt").write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )


def run_backend_subprocess(
    *,
    python_executable: str,
    backend_name: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    cmd = [
        python_executable,
        __file__,
        "--internal-run-backend",
        backend_name,
        "--wav",
        args.wav,
        "--reference",
        args.reference,
        "--output-dir",
        str(output_dir),
        "--chunk-ms",
        str(args.chunk_ms),
        "--source",
        args.source,
        "--target",
        args.target,
        "--min-start-seconds",
        str(args.min_start_seconds),
        "--max-history-utterances",
        str(args.max_history_utterances),
        "--partial-max-new-tokens",
        str(args.partial_max_new_tokens),
        "--partial-followup-max-new-tokens",
        str(args.partial_followup_max_new_tokens),
        "--translation-alignatt-min-source-mass",
        str(args.translation_alignatt_min_source_mass),
        "--translation-alignatt-rewind-threshold",
        str(args.translation_alignatt_rewind_threshold),
        "--translation-alignatt-inaccessible-ms",
        str(args.translation_alignatt_inaccessible_ms),
    ]
    if args.gemma_audio_align_probe_mode is not None:
        cmd.extend(
            [
                "--gemma-audio-align-probe-mode",
                args.gemma_audio_align_probe_mode,
            ]
        )
    completed = run_subprocess(cmd, check=False, cwd=str(Path(__file__).resolve().parent))
    if completed.returncode != 0:
        raise SystemExit(
            f"Backend subprocess failed for {backend_name}: exit code {completed.returncode}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wav", default=DEFAULT_WAV)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-ms", default=450, type=int)
    parser.add_argument("--source", default="en")
    parser.add_argument("--target", default="de")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--min-start-seconds", default=2.0, type=float)
    parser.add_argument("--max-history-utterances", default=1, type=int)
    parser.add_argument("--partial-max-new-tokens", default=16, type=int)
    parser.add_argument("--partial-followup-max-new-tokens", default=8, type=int)
    parser.add_argument("--translation-alignatt-min-source-mass", default=0.0, type=float)
    parser.add_argument("--translation-alignatt-rewind-threshold", default=8, type=int)
    parser.add_argument("--translation-alignatt-inaccessible-ms", default=0.0, type=float)
    parser.add_argument("--gemma-audio-align-probe-mode", default=None)
    parser.add_argument(
        "--internal-run-backend",
        choices=BACKEND_IDS,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def internal_run_backend(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor_config = build_processor_config(
        args,
        backend_name=str(args.internal_run_backend),
    )
    load_start = perf_counter()
    CascadeAlignAttProcessor.load_model(processor_config)
    load_ms = (perf_counter() - load_start) * 1000.0

    processor = CascadeAlignAttProcessor(processor_config)
    processor.set_source_language(args.source)
    processor.set_target_language(args.target)
    artifacts, summary = run_single_backend_to_artifacts(
        processor,
        wav_path=args.wav,
        chunk_ms=args.chunk_ms,
        source_lang_code=args.source,
        target_lang_code=args.target,
        backend_name=str(args.internal_run_backend),
        load_ms=load_ms,
    )
    written = write_inference_artifacts(artifacts, output_dir)
    backend_summary = {
        "backend_id": args.internal_run_backend,
        "files": written,
        **summary,
    }
    (output_dir / "backend_summary.json").write_text(
        json.dumps(backend_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(backend_summary, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()

    if args.internal_run_backend is not None:
        internal_run_backend(args)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    backend_dirs: dict[str, Path] = {}
    for backend_name in BACKEND_IDS:
        backend_output_dir = output_dir / backend_name
        backend_dirs[backend_name] = backend_output_dir
        print(f"\n[{backend_name}] running isolated SimulStream comparison pass")
        run_backend_subprocess(
            python_executable=args.python,
            backend_name=backend_name,
            args=args,
            output_dir=backend_output_dir,
        )

    report = build_comparison_report(
        wav_path=args.wav,
        reference_path=args.reference,
        backend_artifact_dirs=backend_dirs,
    )
    write_comparison_outputs(output_dir, report)

    print("\nComparison summary")
    print("=" * 60)
    for backend in report["backends"]:
        print(
            f"{backend['backend_id']}: "
            f"WER={backend['wer']:.4f}  CER={backend['cer']:.4f}  "
            f"first_emit={backend['first_nonempty_emission_audio_s']}s  "
            f"load={backend['model_load_ms']}ms  "
            f"wallclock={backend['total_wallclock_s']}s  "
            f"RTF={backend['rtf']}"
        )
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
