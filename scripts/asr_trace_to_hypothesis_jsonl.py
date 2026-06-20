"""Convert a compare_asr_full_audio stream_trace JSON into hypothesis.jsonl.

SimulEval / LongYAAL semantics: each word's ``delay`` is the
audio-processed time (ms) at which the chunk that committed it
finished — the moment the word became visible to the downstream
consumer. Each word's ``elapsed`` is the real wallclock at that same
chunk boundary. Heavy lifting is done by :mod:`cascade.artifacts`'s
shared ``build_asr_hypothesis_record`` so this script and
``gemma_asr_low_latency.py`` cannot drift.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cascade.artifacts import (  # noqa: E402
    HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
    build_asr_hypothesis_record,
    build_asr_hypothesis_record_from_trace_first_appearance,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-json", required=True, help="Path to a compare_asr_full_audio run JSON.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target-lang-code", default="en")
    p.add_argument("--source-lang-code", default="en")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run = json.loads(Path(args.run_json).read_text(encoding="utf-8"))
    trace = run["stream_trace"]
    audio_duration_s = float(run["audio_duration_s"])
    audio_duration_ms = audio_duration_s * 1000.0
    wav_name = Path(run["wav_path"]).name

    per_token = run.get("per_token_commits") or []
    processing_s = (
        float(run.get("processing_s", 0.0))
        or (float(trace[-1]["wallclock_s"]) if trace else 0.0)
    )

    if per_token:
        hypothesis = build_asr_hypothesis_record(
            per_token_commits=per_token,
            stream_trace=trace,
            wav_name=wav_name,
            audio_duration_s=audio_duration_s,
            processing_s=processing_s,
        )
    else:
        hypothesis = build_asr_hypothesis_record_from_trace_first_appearance(
            stream_trace=trace,
            wav_name=wav_name,
            audio_duration_s=audio_duration_s,
        )
    n_final = len(hypothesis["delays"])

    manifest = {
        "schema_version": "asr_enen_trace_v1",
        "kind": "inference",
        "wav_path": run["wav_path"],
        "chunk_ms": int(run.get("chunk_ms", 800)),
        "source_language_code": args.source_lang_code,
        "target_language_code": args.target_lang_code,
        "audio_duration_ms": audio_duration_ms,
        "files": {
            "hypothesis_jsonl": "hypothesis.jsonl",
        },
        "runtime_config": {
            "hypothesis_elapsed_semantics": HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hypothesis.jsonl").write_text(
        json.dumps(hypothesis, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {output_dir}/hypothesis.jsonl ({n_final} words)")
    print(f"      {output_dir}/manifest.json")


if __name__ == "__main__":
    main()
