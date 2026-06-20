"""Shared audio-alignment primitives for the Gemma AlignAtt ASR backend.

Self-contained helpers the vLLM backend (``gemma_vllm_qk_fast``) relies on:
audio-span detection, audio-position → seconds conversion, per-token to
per-word aggregation, heads-file I/O, monotonicity score, and the
word-end-offset / monotone-envelope projections. Nothing here depends on
``transformers`` — the geometry (audio token span, ``audio_ms_per_token``)
is a Gemma processor fact, not an implementation choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import json
import string

from alignatt4llm.alignment.base import WordAlignment
from alignatt4llm.mt.base import AlignAttHead


# Gemma 4 E4B processor facts. The audio tower tokenizes at 40 ms/token and
# the processor caps the span at 750 tokens (= 30 s). We surface the cap so
# callers can raise explicitly on overflow rather than silently dropping a
# tail that has no valid timestamps.
GEMMA_AUDIO_TOKEN_ID_DEFAULT = 258881
GEMMA_AUDIO_MS_PER_TOKEN_DEFAULT = 40.0
GEMMA_AUDIO_MAX_SECONDS_DEFAULT = 30.0


PUNCTUATION_STRIP = string.punctuation + "”’)]}"
PUNCTUATION_LEADING = "\"'`“”‘’([{"
# Causal decoder tokens often peak before the acoustic end of their word.
# The next word's first token usually lands after that boundary, so we place
# the word end inside that interval instead of at the current word's last
# token peak. A slight bias toward the next-word anchor reduces systematic
# early word-end estimates without waiting for the full next word.
#
# This is a timestamp-quality correction only. It does not change when the
# runtime is allowed to emit a word; the real emission time is still the
# chunk boundary recorded separately in the commit log.
WORD_END_BOUNDARY_NEXT_WEIGHT = 0.6


class GemmaAudioTooLongError(ValueError):
    """Raised when an audio chunk exceeds the Gemma audio-encoder cap."""


@dataclass(frozen=True)
class AudioSpan:
    prompt_start: int
    prompt_end: int  # exclusive
    ms_per_token: float

    @property
    def length(self) -> int:
        return self.prompt_end - self.prompt_start


def detect_audio_span(
    input_ids: Sequence[int],
    *,
    audio_token_id: int,
    audio_ms_per_token: float,
) -> AudioSpan | None:
    """Find the contiguous audio-placeholder span in a rendered prompt.

    Gemma4Processor inserts ``boa_token, audio_token * N, eoa_token`` to
    represent one audio input. We locate that contiguous run and expose its
    position range for downstream attention extraction.
    """
    prompt_start: int | None = None
    for idx, token_id in enumerate(input_ids):
        if int(token_id) == int(audio_token_id):
            prompt_start = idx
            break
    if prompt_start is None:
        return None
    prompt_end = prompt_start + 1
    while prompt_end < len(input_ids) and int(input_ids[prompt_end]) == int(audio_token_id):
        prompt_end += 1
    return AudioSpan(
        prompt_start=prompt_start,
        prompt_end=prompt_end,
        ms_per_token=float(audio_ms_per_token),
    )


def audio_position_to_end_seconds(
    position: int | None,
    *,
    ms_per_token: float,
    audio_duration_s: float,
) -> float | None:
    """Audio-token index -> upper-bound end time for that token.

    Position ``i`` covers ``[i * ms_per_token, (i + 1) * ms_per_token)``.
    We report the upper bound as the end time so that cutting after a
    given token yields audio that fully contains the attended frame.
    """
    if position is None:
        return None
    end_s = (float(position) + 1.0) * float(ms_per_token) / 1000.0
    return min(end_s, float(audio_duration_s))


def split_text_into_word_spans(text: str) -> list[tuple[int, int, str]]:
    """Match the word-unit convention used by Qwen's forced aligner.

    Strips leading quotes/brackets and trailing punctuation, returning the
    residual word surface + its character span in ``text``. Empty words
    (pure-punctuation tokens) are dropped.
    """
    words: list[tuple[int, int, str]] = []
    idx = 0
    length = len(text)
    while idx < length:
        while idx < length and text[idx].isspace():
            idx += 1
        start = idx
        while idx < length and not text[idx].isspace():
            idx += 1
        end = idx
        while start < end and text[start] in PUNCTUATION_LEADING:
            start += 1
        while end > start and text[end - 1] in PUNCTUATION_STRIP:
            end -= 1
        if start < end:
            words.append((start, end, text[start:end]))
    return words


def _map_chars_to_words(
    text: str, word_spans: Sequence[tuple[int, int, str]]
) -> list[int | None]:
    mapping: list[int | None] = [None] * len(text)
    for word_idx, (start, end, _) in enumerate(word_spans):
        for char_idx in range(start, min(end, len(text))):
            mapping[char_idx] = word_idx
    return mapping


def aggregate_token_timings_to_words(
    text: str,
    *,
    generated_ids: Sequence[int],
    tokenizer,
    token_end_times_s: Sequence[float | None],
    audio_duration_s: float,
) -> list[WordAlignment]:
    """Group per-token end-times into word-level timestamps.

    Start-time is the min aligned token time inside the word. End-time uses
    the word's last aligned token, then refines that boundary toward the
    next word's first aligned token when both anchors are available and
    time-ordered. This reduces the systematic early bias of causal-token
    argmaxes for short words such as ``in`` / ``for`` / ``by``.

    The order of operations matters:
      1. head aggregation chooses one token-level audio index per token
      2. monotone projection removes local leftward regressions
      3. the scalar offset de-biases those token end times globally
      4. this word-level rule converts token anchors into word ends

    Keeping the global offset outside this function is deliberate. The
    offset is calibrated once per model/language pair, whereas this helper
    should stay purely geometric once token end-times are fixed.

    Tokens with no alignment (``None``) are ignored; unaligned words inherit
    the previous word's end-time as a monotone fallback.
    """
    if len(generated_ids) != len(token_end_times_s):
        raise ValueError("generated_ids and token_end_times_s length mismatch")

    token_surfaces: list[str] = []
    for token_id in generated_ids:
        piece = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        token_surfaces.append(piece)

    cumulative_prefix: list[int] = []
    offset = 0
    for piece in token_surfaces:
        cumulative_prefix.append(offset)
        offset += len(piece)
    full_decoded = "".join(token_surfaces)

    word_spans = split_text_into_word_spans(text)
    if full_decoded == text:
        char_to_word_idx = _map_chars_to_words(text, word_spans)
    else:
        decoded_word_spans = split_text_into_word_spans(full_decoded)
        char_to_word_idx = _map_chars_to_words(full_decoded, decoded_word_spans)
        word_spans = decoded_word_spans

    per_word_ends: dict[int, float] = {}
    per_word_starts: dict[int, float] = {}
    for piece, piece_start, end_time_s in zip(
        token_surfaces, cumulative_prefix, token_end_times_s
    ):
        if end_time_s is None:
            continue
        piece_end = piece_start + len(piece)
        if piece_end <= piece_start:
            continue
        for char_idx in range(piece_start, min(piece_end, len(char_to_word_idx))):
            word_idx = char_to_word_idx[char_idx]
            if word_idx is None:
                continue
            if word_idx not in per_word_ends or end_time_s > per_word_ends[word_idx]:
                per_word_ends[word_idx] = float(end_time_s)
            if word_idx not in per_word_starts or end_time_s < per_word_starts[word_idx]:
                per_word_starts[word_idx] = float(end_time_s)

    words: list[WordAlignment] = []
    last_end = 0.0
    for word_idx, (_, _, surface) in enumerate(word_spans):
        end_s = per_word_ends.get(word_idx)
        if end_s is None:
            end_s = last_end
        else:
            next_start_s = per_word_starts.get(word_idx + 1)
            if next_start_s is not None and next_start_s > end_s:
                weight = float(WORD_END_BOUNDARY_NEXT_WEIGHT)
                end_s = (
                    (1.0 - weight) * float(end_s)
                    + weight * float(next_start_s)
                )
        start_s = per_word_starts.get(word_idx, last_end)
        end_s = min(max(end_s, start_s), float(audio_duration_s))
        start_s = min(start_s, end_s)
        words.append(
            WordAlignment(
                text=surface,
                start_time=float(start_s),
                end_time=float(end_s),
            )
        )
        last_end = end_s
    return words


def load_audio_alignment_heads(
    path: str, *, top_k: int
) -> tuple[list[AlignAttHead], float]:
    """Load calibrated alignment heads from a JSON file.

    The file has the same shape as the MT head files
    (``token_alignment_heads`` array with ``layer``, ``head``, ``ts``) plus
    an optional ``word_end_offset_seconds`` scalar subtracted from every
    predicted word-end time at inference — correcting the systematic lag
    between a causal LLM's attention peak and the acoustic word boundary.

    Heads and offset are one calibrated bundle:
      - the head ranking controls which per-token argmaxes survive
      - the offset corrects the residual left/right bias of that exact set

    In other words, the offset is not a universal constant for "Gemma". It
    only makes sense for the head file it ships with and the teacher used to
    fit it. Keeping them in the same JSON prevents silent config drift.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    heads = [
        AlignAttHead(
            layer=int(entry["layer"]),
            head=int(entry["head"]),
            ts=float(entry.get("ts", 0.0)),
        )
        for entry in payload.get("token_alignment_heads", [])[:top_k]
    ]
    offset = float(payload.get("word_end_offset_seconds", 0.0))
    return heads, offset


def monotonicity_score(audio_positions: Sequence[int | None]) -> float:
    """Fraction of consecutive token pairs with non-decreasing audio index.

    Streaming AlignAtt requires alignment heads whose argmaxes move forward
    in audio time. Returns ``0`` when fewer than two valid positions exist.
    """
    filtered = [int(pos) for pos in audio_positions if pos is not None]
    if len(filtered) < 2:
        return 0.0
    pairs = zip(filtered[:-1], filtered[1:])
    non_decreasing = sum(1 for prev, nxt in pairs if nxt >= prev)
    return non_decreasing / float(len(filtered) - 1)


def _apply_word_end_offset(
    values: Sequence[float | None],
    *,
    offset_s: float,
    audio_duration_s: float,
) -> list[float | None]:
    """Subtract the calibrated lag and clamp to ``[0, audio_duration]``.

    The offset is a single scalar fit once per ``(language, model)`` on
    the same teacher that selects the heads — a generic constant, not a
    per-example hack.

    This correction is useful because the token-level signal is structured
    but biased: without it, timestamps are often consistently too early or
    too late even when their ranking is informative. Applying the offset can
    make word-end estimates much more plausible, but it does *not* change
    actual latency. Only the emission timestamp can do that.
    """
    if not offset_s:
        return list(values)
    output: list[float | None] = []
    for value in values:
        if value is None:
            output.append(None)
            continue
        shifted = float(value) - float(offset_s)
        shifted = max(0.0, min(shifted, float(audio_duration_s)))
        output.append(shifted)
    return output


def _enforce_monotone(values: Sequence[float | None]) -> list[float | None]:
    """Project the end-time sequence onto its monotone envelope.

    A per-token argmax can regress locally even in an otherwise monotone
    head. The aligner contract requires a monotone word-end sequence; the
    cleanest generic fix is a left-to-right running max. ``None`` values
    are only filled forward, never backward, so the fallback does not mask
    tokens that legitimately fail to align.

    This projection is deliberately lightweight. If timestamps remain far
    too early after this step, the problem is usually not "more smoothing"
    but either a bad head consensus or a genuinely unstable decode.
    """
    output: list[float | None] = []
    running_max: float = 0.0
    for value in values:
        if value is None:
            output.append(None)
            continue
        value = max(float(value), running_max)
        running_max = value
        output.append(value)
    return output
