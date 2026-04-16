from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from cascade_artifacts import StreamUpdate
from cascade_text_surface import normalize_incremental_target_text, split_target_emission_units


FREEZE_MAJOR_TAIL_REWRITES = "freeze_major_tail_rewrites"
FREEZE_NONEXPANDING_MAJOR_REWRITES = "freeze_nonexpanding_major_rewrites"
RAW_PASSTHROUGH = "raw_passthrough"
FINAL_PASSTHROUGH = "final_passthrough"
FROZEN_MAJOR_TAIL_REWRITE = "frozen_major_tail_rewrite"
FROZEN_NONEXPANDING_MAJOR_TAIL_REWRITE = "frozen_nonexpanding_major_tail_rewrite"
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
    *,
    target_lang_code: str | None = None,
) -> list[str]:
    previous_words = split_target_emission_units(
        previous_translation, target_lang_code=target_lang_code
    )
    current_words = split_target_emission_units(
        current_translation, target_lang_code=target_lang_code
    )
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
    *,
    target_lang_code: str | None = None,
) -> list[str]:
    return register_translation_timestamps(
        previous_translation,
        current_translation,
        audio_processed_ms,
        word_delays_ms,
        target_lang_code=target_lang_code,
    )


def major_rewrite_exceeds_tail_window(
    previous_words: list[str],
    raw_words: list[str],
    *,
    max_tail_rewrite_words: int,
) -> bool:
    prefix_len = longest_common_prefix_words(previous_words, raw_words)
    allowed_rewrite_start = max(0, len(previous_words) - max_tail_rewrite_words)
    return prefix_len < allowed_rewrite_start


def stabilize_emitted_translation(
    previous_translation: str,
    raw_translation: str,
    *,
    max_tail_rewrite_words: int,
    is_final: bool,
    target_lang_code: str | None = None,
) -> tuple[str, str]:
    previous_translation = previous_translation.strip()
    raw_translation = raw_translation.strip()

    if is_final:
        return raw_translation, FINAL_PASSTHROUGH

    if max_tail_rewrite_words < 0 or not previous_translation or not raw_translation:
        return raw_translation, RAW_PASSTHROUGH

    previous_words = split_target_emission_units(
        previous_translation, target_lang_code=target_lang_code
    )
    raw_words = split_target_emission_units(
        raw_translation, target_lang_code=target_lang_code
    )
    if not major_rewrite_exceeds_tail_window(
        previous_words,
        raw_words,
        max_tail_rewrite_words=max_tail_rewrite_words,
    ):
        return raw_translation, RAW_PASSTHROUGH

    return previous_translation, FROZEN_MAJOR_TAIL_REWRITE


def stabilize_nonexpanding_major_rewrites(
    previous_translation: str,
    raw_translation: str,
    *,
    max_tail_rewrite_words: int,
    is_final: bool,
    target_lang_code: str | None = None,
) -> tuple[str, str]:
    previous_translation = previous_translation.strip()
    raw_translation = raw_translation.strip()

    if is_final:
        return raw_translation, FINAL_PASSTHROUGH

    if max_tail_rewrite_words < 0 or not previous_translation or not raw_translation:
        return raw_translation, RAW_PASSTHROUGH

    previous_words = split_target_emission_units(
        previous_translation, target_lang_code=target_lang_code
    )
    raw_words = split_target_emission_units(
        raw_translation, target_lang_code=target_lang_code
    )
    if (
        major_rewrite_exceeds_tail_window(
            previous_words,
            raw_words,
            max_tail_rewrite_words=max_tail_rewrite_words,
        )
        and len(raw_words) <= len(previous_words)
    ):
        return previous_translation, FROZEN_NONEXPANDING_MAJOR_TAIL_REWRITE

    return raw_translation, RAW_PASSTHROUGH


def apply_emission_policy(
    emit_policy: str,
    previous_translation: str,
    raw_translation: str,
    *,
    max_tail_rewrite_words: int,
    is_final: bool,
    target_lang_code: str | None = None,
) -> tuple[str, str]:
    previous_translation = normalize_incremental_target_text(previous_translation)
    raw_translation = normalize_incremental_target_text(raw_translation)

    if emit_policy == FREEZE_MAJOR_TAIL_REWRITES:
        return stabilize_emitted_translation(
            previous_translation,
            raw_translation,
            max_tail_rewrite_words=max_tail_rewrite_words,
            is_final=is_final,
            target_lang_code=target_lang_code,
        )
    if emit_policy == FREEZE_NONEXPANDING_MAJOR_REWRITES:
        return stabilize_nonexpanding_major_rewrites(
            previous_translation,
            raw_translation,
            max_tail_rewrite_words=max_tail_rewrite_words,
            is_final=is_final,
            target_lang_code=target_lang_code,
        )

    action = FINAL_PASSTHROUGH if is_final else RAW_PASSTHROUGH
    return raw_translation.strip(), action


def replay_stream_updates(
    raw_updates: Sequence[StreamUpdate],
    *,
    final_translation_text: str,
    emit_policy: str,
    max_tail_rewrite_words: int,
    target_lang_code: str | None = None,
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
            target_lang_code=target_lang_code,
        )
        emitted_translation, action = apply_emission_policy(
            emit_policy,
            previous_emitted_translation,
            raw_translation,
            max_tail_rewrite_words=max_tail_rewrite_words,
            is_final=index == last_update_index,
            target_lang_code=target_lang_code,
        )
        new_words = register_translation_words(
            previous_emitted_translation,
            emitted_translation,
            update.audio_processed_ms,
            word_delays_ms,
            target_lang_code=target_lang_code,
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

    final_translation_text = normalize_incremental_target_text(final_translation_text)
    if emitted_updates and previous_emitted_translation != final_translation_text:
        last_update = emitted_updates[-1]
        new_words = register_translation_words(
            previous_emitted_translation,
            final_translation_text,
            last_update.audio_processed_ms,
            word_delays_ms,
            target_lang_code=target_lang_code,
        )
        emitted_updates[-1] = replace(
            last_update,
            translation_text=final_translation_text,
            new_words=new_words,
            raw_translation_text=(last_update.raw_translation_text or last_update.translation_text),
            emission_policy_action=FINAL_FORCE_MATCH,
        )

    return emitted_updates, word_delays_ms, word_elapsed_ms
