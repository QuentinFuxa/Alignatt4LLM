#!/usr/bin/env python3
"""Build and evaluate ASR LongYAAL hypotheses from Voxtral realtime traces."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cascade.artifacts import (  # noqa: E402
    HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
    normalize_computation_aware_timestamps,
)


DEFAULT_VOXTRAL_ROOT = Path("/home/fuxa/iwslt-2026-baselines/precomputed_asr_voxtral")
DEFAULT_QWEN_SUMMARY = Path("outputs/asr_compare_enen_21audio_20260421/qwen_forced__summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voxtral-root", type=Path, default=DEFAULT_VOXTRAL_ROOT)
    parser.add_argument("--delay-dir", default="delay480ms")
    parser.add_argument("--qwen-summary", type=Path, default=DEFAULT_QWEN_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/comparaison_asr/voxtral_delay480_eval"))
    parser.add_argument("--chunk-ms", type=int, default=480)
    parser.add_argument("--source-lang-code", default="en")
    parser.add_argument("--target-lang-code", default="en")
    parser.add_argument("--segmentation", type=Path, default=REPO_ROOT / "data/devset/audio-segments.yaml")
    parser.add_argument("--target-reference", type=Path, default=REPO_ROOT / "data/devset/ref/en.txt")
    parser.add_argument("--source-reference", type=Path, default=REPO_ROOT / "data/devset/ref/en.txt")
    parser.add_argument("--eval-venv-python", type=Path, default=REPO_ROOT / ".venv-evaluation/bin/python")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def common_talk_ids(qwen_summary_path: Path, voxtral_root: Path, delay_dir: str) -> list[str]:
    qwen_summary = load_json(qwen_summary_path)
    qwen_ids = {Path(str(row["wav_name"])).stem for row in qwen_summary.get("rows") or []}
    voxtral_ids = {
        path.parent.parent.name
        for path in voxtral_root.glob(f"*/{delay_dir}/asr_chunks.jsonl")
    }
    return sorted(qwen_ids & voxtral_ids)


def final_text_from_trace(rows: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    for row in reversed(rows):
        text = str(row.get("full_text") or "").strip()
        if text:
            return text
    return str(meta.get("final_text") or "").strip()


def build_voxtral_record(*, talk_id: str, trace_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    meta = load_json(trace_dir / "meta.json")
    rows = load_jsonl(trace_dir / "asr_chunks.jsonl")
    if not rows:
        raise ValueError(f"No rows in {trace_dir / 'asr_chunks.jsonl'}")

    final_text = final_text_from_trace(rows, meta)
    final_words = final_text.split()
    audio_duration_s = float(meta.get("audio_duration_s") or rows[-1]["audio_seconds"])
    audio_duration_ms = audio_duration_s * 1000.0

    delays_ms: list[float | None] = [None] * len(final_words)
    elapsed_wallclock_ms: list[float | None] = [None] * len(final_words)
    emitted_count = 0
    cumulative_wallclock_s = 0.0
    mismatch_count = 0

    for row in rows:
        cumulative_wallclock_s += float(row.get("elapsed_s") or 0.0)
        full_words = str(row.get("full_text") or "").split()
        visible_count = min(len(full_words), len(final_words))
        while emitted_count < visible_count:
            if full_words[emitted_count] != final_words[emitted_count]:
                mismatch_count += 1
                break
            delays_ms[emitted_count] = float(row["audio_seconds"]) * 1000.0
            elapsed_wallclock_ms[emitted_count] = cumulative_wallclock_s * 1000.0
            emitted_count += 1

    final_wallclock_ms = cumulative_wallclock_s * 1000.0
    for idx in range(len(final_words)):
        if delays_ms[idx] is None:
            delays_ms[idx] = audio_duration_ms
        if elapsed_wallclock_ms[idx] is None:
            elapsed_wallclock_ms[idx] = final_wallclock_ms

    delay_values = [float(value) for value in delays_ms]
    wallclock_values = [float(value) for value in elapsed_wallclock_ms]
    elapsed_ms = normalize_computation_aware_timestamps(delay_values, wallclock_values)
    record = {
        "source": [f"{talk_id}.wav"],
        "source_length": audio_duration_ms,
        "prediction": final_text,
        "delays": delay_values,
        "elapsed": elapsed_ms,
        "elapsed_wallclock_ms": wallclock_values,
        "elapsed_semantics": HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
    }
    diagnostics = {
        "talk_id": talk_id,
        "word_count": len(final_words),
        "assigned_word_count": sum(value is not None for value in delays_ms),
        "mismatch_count": mismatch_count,
        "audio_duration_s": audio_duration_s,
        "sum_elapsed_s": cumulative_wallclock_s,
        "meta_rtf": meta.get("rtf"),
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
    hyp_path = output_dir / "hypothesis.jsonl"
    with hyp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": "asr_enen_voxtral_realtime_v1",
        "kind": "inference",
        "chunk_ms": int(chunk_ms),
        "source_language_code": source_lang_code,
        "target_language_code": target_lang_code,
        "files": {"hypothesis_jsonl": "hypothesis.jsonl"},
        "runtime_config": {
            "hypothesis_elapsed_semantics": HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
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
        str(REPO_ROOT / "evaluate_cascade_outputs.py"),
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
    talks = common_talk_ids(args.qwen_summary, args.voxtral_root, args.delay_dir)
    if not talks:
        raise SystemExit("No common Voxtral/Qwen talk IDs found.")

    records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for talk_id in talks:
        trace_dir = args.voxtral_root / talk_id / args.delay_dir
        record, diag = build_voxtral_record(talk_id=talk_id, trace_dir=trace_dir)
        records.append(record)
        diagnostics.append(diag)

    write_hypothesis_and_manifest(
        records=records,
        output_dir=args.output_dir,
        chunk_ms=args.chunk_ms,
        source_lang_code=args.source_lang_code,
        target_lang_code=args.target_lang_code,
    )
    (args.output_dir / "voxtral_trace_diagnostics.json").write_text(
        json.dumps(
            {
                "delay_dir": args.delay_dir,
                "common_talk_count": len(talks),
                "common_talks": talks,
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
