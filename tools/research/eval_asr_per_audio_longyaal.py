#!/usr/bin/env python3
"""Compute aggregate + per-audio LongYAAL from the per-audio JSONs
produced by ``compare_asr_per_audio_batch.py``.

Re-uses the existing `asr_trace_to_hypothesis_jsonl` conversion logic to
turn each stream trace into an OmniSTEval hypothesis line, then:

  * assembles the 21 lines into a single ``hypothesis.jsonl`` and runs
    ``alignatt-eval --skip-comet`` to get dataset-level
    BLEU / chrF / LongYAAL CU / LongYAAL CA on the en--en reference.
  * optionally loops per-audio (``--per-audio``) and runs a 1-line
    evaluation per wav, emitting a JSON with per-audio LongYAAL etc.

The script only needs ``.venv-evaluation`` (the evaluation venv).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from alignatt4llm.artifacts import (  # noqa: E402
    HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
    build_asr_hypothesis_record,
    build_asr_hypothesis_record_from_trace_first_appearance,
)


def build_hypothesis_record_from_run(run: dict) -> dict:
    """Build an ASR hypothesis record with SimulEval emission semantics.

    Prefers the per-token commit log (exact chunk-boundary emission time
    per word) and falls back to first-appearance matching on
    ``public_asr_text`` when a run predates the commit log.
    """
    trace = run.get("stream_trace") or []
    audio_duration_s = float(run["audio_duration_s"])
    wav_name = Path(run["wav_path"]).name
    per_token = run.get("per_token_commits") or []
    processing_s = (
        float(run.get("processing_s", 0.0))
        or (float(trace[-1]["wallclock_s"]) if trace else 0.0)
    )
    if per_token:
        return build_asr_hypothesis_record(
            per_token_commits=per_token,
            stream_trace=trace,
            wav_name=wav_name,
            audio_duration_s=audio_duration_s,
            processing_s=processing_s,
        )
    return build_asr_hypothesis_record_from_trace_first_appearance(
        stream_trace=trace,
        wav_name=wav_name,
        audio_duration_s=audio_duration_s,
    )


def write_hypothesis_and_manifest(
    *,
    records: list[dict],
    output_dir: Path,
    chunk_ms: int,
    source_lang_code: str,
    target_lang_code: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    hyp_path = output_dir / "hypothesis.jsonl"
    with hyp_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": "asr_enen_batch_v1",
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
) -> dict:
    cmd = [
        str(eval_venv_python),
        "-m",
        "alignatt4llm.cli.evaluate",
        "--output-dir", str(output_dir),
        "--speech-segmentation", str(segmentation),
        "--target-reference", str(target_reference),
        "--source-reference", str(source_reference),
        "--target-lang-code", target_lang_code,
        "--skip-comet",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    eval_json = output_dir / "evaluation.json"
    return json.loads(eval_json.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-audio-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunk-ms", type=int, default=800)
    parser.add_argument("--source-lang-code", default="en")
    parser.add_argument("--target-lang-code", default="en")
    parser.add_argument(
        "--segmentation",
        default=str(REPO_ROOT / "data/devset/audio-segments.yaml"),
    )
    parser.add_argument(
        "--target-reference",
        default=str(REPO_ROOT / "data/devset/ref/en.txt"),
    )
    parser.add_argument(
        "--source-reference",
        default=str(REPO_ROOT / "data/devset/ref/en.txt"),
    )
    parser.add_argument(
        "--eval-venv-python",
        default=str(REPO_ROOT / ".venv-evaluation/bin/python"),
    )
    parser.add_argument(
        "--per-audio",
        action="store_true",
        help="Also run a single-audio evaluation for every wav (gives per-audio LongYAAL).",
    )
    args = parser.parse_args()

    per_audio_dir: Path = args.per_audio_dir
    output_dir: Path = args.output_dir

    run_paths = sorted(p for p in per_audio_dir.glob("*.json") if not p.stem.endswith("__summary"))
    if not run_paths:
        raise SystemExit(f"No per-audio JSONs under {per_audio_dir}")

    runs = [(p, json.loads(p.read_text(encoding="utf-8"))) for p in run_paths]
    records = [build_hypothesis_record_from_run(run) for _, run in runs]

    print(f"[aggregate] {len(records)} hypotheses -> {output_dir}/hypothesis.jsonl", flush=True)
    write_hypothesis_and_manifest(
        records=records,
        output_dir=output_dir,
        chunk_ms=args.chunk_ms,
        source_lang_code=args.source_lang_code,
        target_lang_code=args.target_lang_code,
    )

    print("[aggregate] running OmniSTEval ...", flush=True)
    aggregate_eval = run_omnisteval(
        eval_venv_python=Path(args.eval_venv_python),
        output_dir=output_dir,
        segmentation=Path(args.segmentation),
        target_reference=Path(args.target_reference),
        source_reference=Path(args.source_reference),
        target_lang_code=args.target_lang_code,
    )
    agg_scores = aggregate_eval.get("contract_scores", {})
    print("[aggregate] scores:", json.dumps(agg_scores, indent=2), flush=True)

    per_audio_results: list[dict] = []
    if args.per_audio:
        per_audio_root = output_dir / "per_audio"
        per_audio_root.mkdir(parents=True, exist_ok=True)
        for (run_path, run), record in zip(runs, records):
            wav_name = record["source"][0]
            stem = Path(wav_name).stem
            audio_out = per_audio_root / stem
            print(f"[per-audio] {stem}", flush=True)
            write_hypothesis_and_manifest(
                records=[record],
                output_dir=audio_out,
                chunk_ms=args.chunk_ms,
                source_lang_code=args.source_lang_code,
                target_lang_code=args.target_lang_code,
            )
            eval_payload = run_omnisteval(
                eval_venv_python=Path(args.eval_venv_python),
                output_dir=audio_out,
                segmentation=Path(args.segmentation),
                target_reference=Path(args.target_reference),
                source_reference=Path(args.source_reference),
                target_lang_code=args.target_lang_code,
            )
            scores = eval_payload.get("contract_scores", {})
            backend_metrics = (run.get("metrics") or {})
            per_audio_results.append(
                {
                    "wav_name": wav_name,
                    "stem": stem,
                    "audio_duration_s": float(run["audio_duration_s"]),
                    "wer": float(backend_metrics.get("wer") or 0.0),
                    "cer": float(backend_metrics.get("cer") or 0.0),
                    "long_yaal_cu_ms": scores.get("LongYAAL CU"),
                    "long_yaal_ca_ms": scores.get("LongYAAL CA"),
                    "bleu": scores.get("BLEU"),
                    "chrf": scores.get("CHRF"),
                }
            )

    summary_payload = {
        "per_audio_dir": str(per_audio_dir),
        "aggregate": agg_scores,
        "per_audio": per_audio_results,
    }
    summary_path = output_dir / "longyaal_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
