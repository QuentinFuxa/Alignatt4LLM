#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cascade_artifacts import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE_LANG_CODE,
    DEFAULT_TARGET_LANG_CODE,
    HYPOTHESIS_FILENAME,
    FINAL_ASR_FILENAME,
    InferenceArtifacts,
    MANIFEST_FILENAME,
    StreamUpdate,
    STREAM_UPDATES_FILENAME,
    final_asr_filename,
    write_inference_artifacts,
)
from cascade_emission import (
    FREEZE_MAJOR_TAIL_REWRITES,
    FREEZE_NONEXPANDING_MAJOR_REWRITES,
    RAW_PASSTHROUGH,
    replay_stream_updates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-emit cascade artifacts with a deterministic translation emission policy.",
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing an existing inference bundle.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the rewritten inference bundle should be written.",
    )
    parser.add_argument(
        "--max-tail-rewrite-words",
        default=14,
        type=int,
        help="Allow rewrites only within this trailing word window on non-final updates.",
    )
    parser.add_argument(
        "--emit-policy",
        default=FREEZE_MAJOR_TAIL_REWRITES,
        choices=[
            RAW_PASSTHROUGH,
            FREEZE_MAJOR_TAIL_REWRITES,
            FREEZE_NONEXPANDING_MAJOR_REWRITES,
        ],
        help="Emission policy to apply while replaying the stored raw translation stream.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_stream_updates(path: Path) -> list[StreamUpdate]:
    updates: list[StreamUpdate] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        updates.append(
            StreamUpdate(
                update_idx=payload["update_idx"],
                audio_processed_ms=payload["audio_processed_ms"],
                wallclock_elapsed_ms=payload["wallclock_elapsed_ms"],
                asr_text=payload["asr_text"],
                translation_text=payload["translation_text"],
                new_words=payload.get("new_words", []),
                is_eos=bool(payload.get("is_eos", False)),
                raw_translation_text=payload.get("raw_translation_text"),
                emission_policy_action=payload.get("emission_policy_action"),
                translation_prompt_num_cached_tokens=payload.get(
                    "translation_prompt_num_cached_tokens"
                ),
                translation_prompt_num_tokens=payload.get("translation_prompt_num_tokens"),
                partial_accepted_target=payload.get("partial_accepted_target"),
                partial_accepted_token_count=payload.get("partial_accepted_token_count"),
                partial_draft_target=payload.get("partial_draft_target"),
                alignatt_metadata=payload.get("alignatt_metadata"),
                translation_timings_ms=payload.get("translation_timings_ms"),
            )
        )
    return updates


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("Replay output dir must differ from the input dir.")

    manifest = load_json(input_dir / MANIFEST_FILENAME)
    hypothesis = load_json(input_dir / HYPOTHESIS_FILENAME)
    raw_updates = load_stream_updates(input_dir / STREAM_UPDATES_FILENAME)
    source_lang_code = manifest.get("source_language_code", DEFAULT_SOURCE_LANG_CODE)
    target_lang_code = manifest.get("target_language_code", DEFAULT_TARGET_LANG_CODE)
    candidate_asr_path = input_dir / final_asr_filename(source_lang_code)
    if not candidate_asr_path.exists():
        candidate_asr_path = input_dir / FINAL_ASR_FILENAME
    final_asr_text = candidate_asr_path.read_text(encoding="utf-8").strip()
    final_translation_text = hypothesis["prediction"].strip()

    emitted_updates, word_delays_ms, word_elapsed_ms = replay_stream_updates(
        raw_updates,
        final_translation_text=final_translation_text,
        emit_policy=args.emit_policy,
        max_tail_rewrite_words=args.max_tail_rewrite_words,
        target_lang_code=target_lang_code,
    )
    replayed_final_translation_text = (
        emitted_updates[-1].translation_text if emitted_updates else final_translation_text
    )

    source_runtime_config = dict(manifest.get("runtime_config", {}))
    replay_depth = int(source_runtime_config.get("translation_replay_depth", 0) or 0) + 1
    runtime_config = dict(source_runtime_config)
    runtime_config["translation_emit_policy"] = args.emit_policy
    runtime_config["translation_max_tail_rewrite_words"] = args.max_tail_rewrite_words
    runtime_config["translation_artifact_origin"] = "replayed_stream"
    runtime_config["translation_replay_input_dir"] = str(input_dir)
    runtime_config["translation_replay_depth"] = replay_depth
    runtime_config["translation_replay_input_emit_policy"] = source_runtime_config.get(
        "translation_emit_policy",
        RAW_PASSTHROUGH,
    )
    runtime_config["translation_replay_input_max_tail_rewrite_words"] = source_runtime_config.get(
        "translation_max_tail_rewrite_words",
        args.max_tail_rewrite_words,
    )

    artifacts = InferenceArtifacts(
        wav_path=manifest["wav_path"],
        chunk_ms=manifest["chunk_ms"],
        translation_variant=manifest.get("translation_variant"),
        source_language=manifest["source_language"],
        target_language=manifest["target_language"],
        source_language_code=source_lang_code,
        target_language_code=target_lang_code,
        latency_unit=manifest["latency_unit"],
        audio_duration_ms=manifest["audio_duration_ms"],
        final_asr_text=final_asr_text,
        final_translation_text=replayed_final_translation_text,
        translation_word_delays_ms=word_delays_ms,
        translation_word_elapsed_ms=word_elapsed_ms,
        updates=emitted_updates,
        runtime_config=runtime_config,
    )
    written_files = write_inference_artifacts(artifacts, output_dir)
    print(f"Re-emitted inference artifacts to {output_dir}")
    for label, path in written_files.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
