"""Convert compare_asr_full_audio stream_trace into a hypothesis.jsonl + manifest.

For pure simultaneous transcription (source == target), treat the ASR emission
timeline as the "translation" emission timeline. Each word's delay is the
audio-processed time (ms) when the word first appears in the stream's
committed text; each word's elapsed is the wallclock (ms) at the same point.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cascade.artifacts import (  # noqa: E402
    HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
    normalize_computation_aware_timestamps,
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
    audio_duration_ms = float(run["audio_duration_s"]) * 1000.0
    wav_name = Path(run["wav_path"]).name

    # ``public_asr_text`` is the cumulative, externally-visible committed
    # transcript across all utterances in the streaming run. Per-utterance
    # fields like ``alignatt_stream_committed_text`` reset at segment
    # boundaries and would truncate the hypothesis to the last utterance.
    final_committed = trace[-1].get("public_asr_text", "") or ""
    final_words = final_committed.split()
    n_final = len(final_words)

    delays_ms: list[float] = [0.0] * n_final
    elapsed_ms: list[float] = [0.0] * n_final
    marked = [False] * n_final

    for row in trace:
        committed = (row.get("public_asr_text", "") or "").split()
        audio_processed_ms = float(row["audio_processed_s"]) * 1000.0
        wallclock_ms = float(row["wallclock_s"]) * 1000.0
        k = min(len(committed), n_final)
        for i in range(k):
            if not marked[i] and committed[i] == final_words[i]:
                delays_ms[i] = audio_processed_ms
                elapsed_ms[i] = wallclock_ms
                marked[i] = True

    for i in range(n_final):
        if not marked[i]:
            delays_ms[i] = audio_duration_ms
            elapsed_ms[i] = float(trace[-1]["wallclock_s"]) * 1000.0

    normalized_elapsed = normalize_computation_aware_timestamps(delays_ms, elapsed_ms)

    hypothesis = {
        "source": [wav_name],
        "source_length": audio_duration_ms,
        "prediction": final_committed,
        "delays": delays_ms,
        "elapsed": normalized_elapsed,
        "elapsed_wallclock_ms": elapsed_ms,
        "elapsed_semantics": HYPOTHESIS_ELAPSED_SEMANTICS_CA_COMPATIBLE,
    }

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
