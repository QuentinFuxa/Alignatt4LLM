#!/usr/bin/env python3
"""Build and evaluate ASR LongYAAL hypotheses from Gemma E4B LA captures."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from alignatt4llm.artifacts import (  # noqa: E402
    HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
    normalize_computation_aware_timestamps,
)


DEFAULT_CAPTURE_ROOT = Path("outputs/gemma_e4b_asr_mcif_la_full_20260424")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/gemma_e4b_asr_mcif_la_full_20260424/eval"),
    )
    parser.add_argument("--chunk-ms", type=int, default=800)
    parser.add_argument("--source-lang-code", default="en")
    parser.add_argument("--target-lang-code", default="en")
    parser.add_argument(
        "--segmentation",
        type=Path,
        default=REPO_ROOT / "data/devset/audio-segments.yaml",
    )
    parser.add_argument(
        "--target-reference",
        type=Path,
        default=REPO_ROOT / "data/devset/ref/en.txt",
    )
    parser.add_argument(
        "--source-reference",
        type=Path,
        default=REPO_ROOT / "data/devset/ref/en.txt",
    )
    parser.add_argument(
        "--eval-venv-python",
        type=Path,
        default=REPO_ROOT / ".venv-evaluation/bin/python",
    )
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_capture_dir(capture_root: Path) -> Path:
    nested = capture_root / "captures"
    if nested.exists():
        return nested
    return capture_root


def capture_paths(capture_root: Path) -> list[Path]:
    capture_dir = resolve_capture_dir(capture_root)
    paths = sorted(path for path in capture_dir.glob("*.json") if path.name != "manifest.json")
    if not paths:
        raise SystemExit(f"No Gemma capture JSON files under {capture_dir}")
    return paths


def build_record_from_capture(capture: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    final_text = str(capture.get("final_text") or "").strip()
    final_words = final_text.split()
    audio_duration_s = float(capture["audio_duration_s"])
    audio_duration_ms = audio_duration_s * 1000.0
    processing_s = float(capture.get("processing_s") or 0.0)
    final_wallclock_ms = processing_s * 1000.0

    emitted_words: list[str] = []
    delays_ms: list[float] = []
    wallclock_ms: list[float] = []
    for chunk in capture.get("chunks") or []:
        audio_processed_ms = float(chunk["audio_processed_s"]) * 1000.0
        chunk_wallclock_ms = float(chunk.get("wallclock_s") or 0.0) * 1000.0
        for word in chunk.get("new_committed_words") or []:
            emitted_words.append(str(word.get("text") or ""))
            delays_ms.append(audio_processed_ms)
            wallclock_ms.append(chunk_wallclock_ms)

    if len(emitted_words) != len(final_words):
        # Keep the evaluation record usable and conservative: missing suffix
        # words are emitted at EOS. Prefix mismatches are recorded as diagnostics
        # but not repaired lexically.
        start = min(len(emitted_words), len(final_words))
        for word in final_words[start:]:
            emitted_words.append(word)
            delays_ms.append(audio_duration_ms)
            wallclock_ms.append(final_wallclock_ms)

    if len(delays_ms) > len(final_words):
        delays_ms = delays_ms[: len(final_words)]
        wallclock_ms = wallclock_ms[: len(final_words)]
        emitted_words = emitted_words[: len(final_words)]

    elapsed_ms = normalize_computation_aware_timestamps(delays_ms, wallclock_ms)
    record = {
        "source": [str(capture["wav_name"])],
        "source_length": audio_duration_ms,
        "prediction": final_text,
        "delays": delays_ms,
        "elapsed": elapsed_ms,
        "elapsed_wallclock_ms": wallclock_ms,
        "elapsed_semantics": HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
    }
    diagnostics = {
        "wav_name": str(capture["wav_name"]),
        "word_count": len(final_words),
        "emitted_word_count": len(emitted_words),
        "emitted_matches_final_split": emitted_words == final_words,
        "audio_duration_s": audio_duration_s,
        "processing_s": processing_s,
        "rtf_wallclock": capture.get("rtf_wallclock"),
        "wer": (capture.get("metrics") or {}).get("wer"),
        "cer": (capture.get("metrics") or {}).get("cer"),
    }
    return record, diagnostics


def write_hypothesis_and_manifest(
    *,
    records: list[dict[str, Any]],
    output_dir: Path,
    chunk_ms: int,
    source_lang_code: str,
    target_lang_code: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "hypothesis.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": "asr_enen_gemma_e4b_local_agreement_v1",
        "kind": "inference",
        "chunk_ms": int(chunk_ms),
        "source_language_code": source_lang_code,
        "target_language_code": target_lang_code,
        "files": {"hypothesis_jsonl": "hypothesis.jsonl"},
        "runtime_config": {
            "hypothesis_elapsed_semantics": HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
            "asr_backend": "gemma_e4b_vllm_local_agreement",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_omnisteval(
    *,
    eval_venv_python: Path,
    output_dir: Path,
    segmentation: Path,
    target_reference: Path,
    source_reference: Path,
    target_lang_code: str,
) -> dict[str, Any]:
    cmd = [
        str(eval_venv_python),
        "-m",
        "alignatt4llm.cli.evaluate",
        "--output-dir",
        str(output_dir),
        "--speech-segmentation",
        str(segmentation),
        "--target-reference",
        str(target_reference),
        "--source-reference",
        str(source_reference),
        "--target-lang-code",
        target_lang_code,
        "--skip-comet",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return load_json(output_dir / "evaluation.json")


def main() -> None:
    args = parse_args()
    records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for path in capture_paths(args.capture_root):
        record, diagnostic = build_record_from_capture(load_json(path))
        records.append(record)
        diagnostics.append(diagnostic)

    write_hypothesis_and_manifest(
        records=records,
        output_dir=args.output_dir,
        chunk_ms=args.chunk_ms,
        source_lang_code=args.source_lang_code,
        target_lang_code=args.target_lang_code,
    )
    (args.output_dir / "gemma_capture_diagnostics.json").write_text(
        json.dumps(
            {
                "capture_root": str(args.capture_root),
                "capture_count": len(records),
                "diagnostics": diagnostics,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output_dir / 'hypothesis.jsonl'} ({len(records)} talks)")
    if args.skip_eval:
        return

    evaluation = run_omnisteval(
        eval_venv_python=args.eval_venv_python,
        output_dir=args.output_dir,
        segmentation=args.segmentation,
        target_reference=args.target_reference,
        source_reference=args.source_reference,
        target_lang_code=args.target_lang_code,
    )
    print(json.dumps(evaluation.get("contract_scores", {}), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
