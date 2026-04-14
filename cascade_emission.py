from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from cascade_artifacts import StreamUpdate


FREEZE_MAJOR_TAIL_REWRITES = "freeze_major_tail_rewrites"
RAW_PASSTHROUGH = "raw_passthrough"
FINAL_PASSTHROUGH = "final_passthrough"
FROZEN_MAJOR_TAIL_REWRITE = "frozen_major_tail_rewrite"
FINAL_FORCE_MATCH = "final_force_match"


def longest_common_prefix_words(previous: list[str], current: list[str]) -> int:
    prefix_len = 0
    for left, right in zip(previous, current):
        if left != right:
            break
        prefix_len += 1
    return prefix_len


def register_translation_timestamps(
    previous_translation: str,
    current_translation: str,
    timestamp_ms: float,
    metric_timestamps_ms: list[float],
) -> list[str]:
    previous_words = previous_translation.split()
    current_words = current_translation.split()
    prefix_len = longest_common_prefix_words(previous_words, current_words)

    del metric_timestamps_ms[prefix_len:]
    for _ in current_words[prefix_len:]:
        metric_timestamps_ms.append(timestamp_ms)

    return current_words[prefix_len:]


def register_translation_words(
    previous_translation: str,
    current_translation: str,
    audio_processed_ms: float,
    word_delays_ms: list[float],
) -> list[str]:
    return register_translation_timestamps(
        previous_translation,
        current_translation,
        audio_processed_ms,
        word_delays_ms,
    )


def stabilize_emitted_translation(
    previous_translation: str,
    raw_translation: str,
    *,
    max_tail_rewrite_words: int,
    is_final: bool,
) -> tuple[str, str]:
    previous_translation = previous_translation.strip()
    raw_translation = raw_translation.strip()

    if is_final:
        return raw_translation, FINAL_PASSTHROUGH

    if max_tail_rewrite_words < 0 or not previous_translation or not raw_translation:
        return raw_translation, RAW_PASSTHROUGH

    previous_words = previous_translation.split()
    raw_words = raw_translation.split()
    prefix_len = longest_common_prefix_words(previous_words, raw_words)
    allowed_rewrite_start = max(0, len(previous_words) - max_tail_rewrite_words)
    if prefix_len >= allowed_rewrite_start:
        return raw_translation, RAW_PASSTHROUGH

    return previous_translation, FROZEN_MAJOR_TAIL_REWRITE


def apply_emission_policy(
    emit_policy: str,
    previous_translation: str,
    raw_translation: str,
    *,
    max_tail_rewrite_words: int,
    is_final: bool,
) -> tuple[str, str]:
    if emit_policy == FREEZE_MAJOR_TAIL_REWRITES:
        return stabilize_emitted_translation(
            previous_translation,
            raw_translation,
            max_tail_rewrite_words=max_tail_rewrite_words,
            is_final=is_final,
        )

    action = FINAL_PASSTHROUGH if is_final else RAW_PASSTHROUGH
    return raw_translation.strip(), action


def replay_stream_updates(
    raw_updates: Sequence[StreamUpdate],
    *,
    final_translation_text: str,
    emit_policy: str,
    max_tail_rewrite_words: int,
) -> tuple[list[StreamUpdate], list[float], list[float]]:
    emitted_updates: list[StreamUpdate] = []
    word_delays_ms: list[float] = []
    word_elapsed_ms: list[float] = []
    previous_emitted_translation = ""
    previous_raw_translation = ""
    last_update_index = len(raw_updates) - 1

    for index, update in enumerate(raw_updates):
        raw_translation = (update.raw_translation_text or update.translation_text).strip()
        register_translation_timestamps(
            previous_raw_translation,
            raw_translation,
            update.wallclock_elapsed_ms,
            word_elapsed_ms,
        )
        emitted_translation, action = apply_emission_policy(
            emit_policy,
            previous_emitted_translation,
            raw_translation,
            max_tail_rewrite_words=max_tail_rewrite_words,
            is_final=index == last_update_index,
        )
        new_words = register_translation_words(
            previous_emitted_translation,
            emitted_translation,
            update.audio_processed_ms,
            word_delays_ms,
        )
        emitted_updates.append(
            replace(
                update,
                translation_text=emitted_translation,
                new_words=new_words,
                raw_translation_text=raw_translation,
                emission_policy_action=action,
            )
        )
        previous_emitted_translation = emitted_translation
        previous_raw_translation = raw_translation

    final_translation_text = final_translation_text.strip()
    if emitted_updates and previous_emitted_translation != final_translation_text:
        last_update = emitted_updates[-1]
        new_words = register_translation_words(
            previous_emitted_translation,
            final_translation_text,
            last_update.audio_processed_ms,
            word_delays_ms,
        )
        emitted_updates[-1] = replace(
            last_update,
            translation_text=final_translation_text,
            new_words=new_words,
            raw_translation_text=(last_update.raw_translation_text or last_update.translation_text),
            emission_policy_action=FINAL_FORCE_MATCH,
        )

    return emitted_updates, word_delays_ms, word_elapsed_ms
