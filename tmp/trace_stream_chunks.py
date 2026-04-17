#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from cascade_audio import load_audio_mono_16khz
from cascade_simulstream_processor import CascadeAlignAttProcessor
from cascade_text_surface import split_public_emission_units
from simulstream.server.speech_processors import SAMPLE_RATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace every streaming chunk for one audio into JSONL."
    )
    parser.add_argument("--input", required=True, help="Path to the audio file.")
    parser.add_argument("--output", required=True, help="Destination JSONL path.")
    parser.add_argument("--chunk-ms", type=int, default=800)
    parser.add_argument("--limit-seconds", type=float, default=None)
    parser.add_argument("--source", default="en")
    parser.add_argument("--target", default="de")
    parser.add_argument("--alignment-backend-name", default="qwen_forced")
    parser.add_argument("--mt-backend-name", default="gemma_vllm_alignatt")
    parser.add_argument("--min-start-seconds", type=float, default=2.0)
    parser.add_argument("--max-history-utterances", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimpleNamespace(
        chunk_ms=args.chunk_ms,
        source_lang_code=args.source,
        target_lang_code=args.target,
        alignment_backend_name=args.alignment_backend_name,
        mt_backend_name=args.mt_backend_name,
        min_start_seconds=args.min_start_seconds,
        max_history_utterances=args.max_history_utterances,
    )
    processor = CascadeAlignAttProcessor(config)
    processor.clear()
    audio = load_audio_mono_16khz(args.input)
    if args.limit_seconds is not None:
        audio = audio[: int(args.limit_seconds * SAMPLE_RATE)]
    chunk_size = int(SAMPLE_RATE * args.chunk_ms / 1000)

    rows: list[dict[str, object]] = []
    for start in range(0, len(audio), chunk_size):
        chunk = audio[start : start + chunk_size]
        session_result = processor.session.process_audio_chunk(chunk)
        audio_ms = min(start + chunk_size, len(audio)) * 1000.0 / SAMPLE_RATE
        if session_result is None:
            rows.append({"audio_ms": audio_ms, "skipped": "min_start"})
            continue

        previous_public = processor._current_emitted_text()
        translation, emission_policy_action = processor.session.apply_translation_emit_policy(
            previous_public,
            session_result.raw_translation_text,
            is_final=False,
        )
        new_units = split_public_emission_units(translation, target_lang_code=args.target)
        previous_units = list(processor._emitted_units)
        extends_public = (
            len(new_units) >= len(previous_units)
            and new_units[: len(previous_units)] == previous_units
        )
        added_units = (
            new_units[len(previous_units) :]
            if extends_public and len(new_units) >= len(previous_units)
            else []
        )
        partial = processor.session.state.partial_translation
        rows.append(
            {
                "audio_ms": audio_ms,
                "asr_text": session_result.asr_text,
                "raw_translation_text": session_result.raw_translation_text,
                "emitted_translation_candidate": translation,
                "current_public_translation": previous_public,
                "extends_public": extends_public,
                "new_unit_count": len(added_units),
                "new_units": added_units,
                "partial_source_prefix": partial.source_prefix,
                "partial_accepted_target": partial.accepted_target,
                "partial_draft_target": partial.draft_target,
                "stop_reason": (
                    None
                    if session_result.translation_result is None
                    else session_result.translation_result.stop_reason
                ),
                "emission_policy_action": emission_policy_action,
            }
        )
        if extends_public and added_units:
            processor._emitted_units = list(new_units)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    )
    print(output_path)
    print(f"rows={len(rows)}")
    print(f"last_public={processor._current_emitted_text()}")


if __name__ == "__main__":
    main()
