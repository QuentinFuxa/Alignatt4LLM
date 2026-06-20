"""Cascade-faithful Qwen local-agreement capture for one source stream.

This module mirrors the source-side strategy we actually want to study:

1. keep only the active utterance tail audio, exactly like the cascade;
2. inside that tail, re-inject the locally-stable prefix into Qwen's prompt;
3. compute local agreement on the tail hypothesis;
4. commit to the public source only up to the last sentence-final punctuation,
   again like the cascade;
5. keep absolute timestamps for every visible word so offline holdback-x
   policies can be simulated later.

That combination is what prevents prompt-length blow-up on long files while
still letting us test "local agreement vs holdback-x" on top of the same
online mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from alignatt4llm.source_surface import (
    lexical_word_count,
    lexical_word_surfaces,
    project_char_lcp_to_word_prefix_text,
)


def longest_common_prefix(left: str, right: str) -> str:
    for idx in range(min(len(left), len(right))):
        if left[idx] != right[idx]:
            return left[:idx]
    return left[: min(len(left), len(right))]


def _join_segments(segments: list[str], tail_text: str) -> str:
    items = [segment.strip() for segment in segments if segment.strip()]
    if tail_text.strip():
        items.append(tail_text.strip())
    return " ".join(items).strip()


def _serialize_forced_align_words(
    items: Any,
    *,
    time_offset_s: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items or ():
        rows.append(
            {
                "text": str(item.text),
                "start_time": float(item.start_time) + float(time_offset_s),
                "end_time": float(item.end_time) + float(time_offset_s),
            }
        )
    return rows


def _find_commit_char_index(candidate_text: str) -> int:
    rightest_punct_idx = max(
        candidate_text.rfind(". "),
        candidate_text.rfind("! "),
        candidate_text.rfind("? "),
    )
    if rightest_punct_idx == -1 and candidate_text.endswith((".", "!", "?")):
        rightest_punct_idx = len(candidate_text) - 1
    return int(rightest_punct_idx)


def normalize_partial_asr_hypothesis(text: str) -> str:
    """Match the cascade's live-tail rendering without importing the runtime."""

    text = text.rstrip()
    while text.endswith((".", "!", "?")):
        text = text[:-1].rstrip()
    return text


@dataclass
class QwenLocalAgreementStream:
    """Own one cascade-style Qwen source stream.

    ``stable_tail_prefix_text`` is only the stable prefix inside the active
    utterance tail. Once we commit a sentence-final segment, both the active
    tail audio and that reinjected prefix reset, exactly to avoid prompt growth
    from old committed sentences.
    """

    asr: Any
    language: str
    context: str = ""

    def __post_init__(self) -> None:
        self.prompt_raw = self.asr._build_text_prompt(
            context=self.context,
            force_language=self.language,
        )
        self.tail_start_time_s = 0.0
        self.previous_tail_hypothesis_text = ""
        self.stable_tail_prefix_text = ""
        self.public_segments: list[str] = []
        self.public_committed_words: list[dict[str, Any]] = []
        self.stream_trace: list[dict[str, Any]] = []

    def _decode_tail_hypothesis(
        self,
        *,
        audio_tail: np.ndarray,
    ) -> tuple[str, str, str]:
        from qwen_asr import parse_asr_output

        prompt = self.prompt_raw + self.stable_tail_prefix_text
        outputs = self.asr.model.generate(
            [
                {
                    "prompt": prompt,
                    "multi_modal_data": {"audio": [audio_tail]},
                }
            ],
            sampling_params=self.asr.sampling_params,
            use_tqdm=False,
        )
        continuation_text = str(outputs[0].outputs[0].text)
        raw_decoded = self.stable_tail_prefix_text + continuation_text
        hypothesis_language, hypothesis_text = parse_asr_output(
            raw_decoded,
            user_language=self.language,
        )
        return continuation_text, hypothesis_language, hypothesis_text.strip()

    def _align_tail_hypothesis(
        self,
        *,
        audio_tail: np.ndarray,
        sample_rate: int,
        tail_hypothesis_text: str,
        hypothesis_language: str,
    ) -> tuple[list[dict[str, Any]], str | None]:
        if not tail_hypothesis_text:
            return [], None
        try:
            results = self.asr.forced_aligner.align(
                audio=(audio_tail, sample_rate),
                text=tail_hypothesis_text,
                language=hypothesis_language or self.language,
            )
        except Exception as exc:  # noqa: BLE001 - durable capture beats crash
            return [], f"{type(exc).__name__}: {exc}"
        if not results:
            return [], "forced_aligner_returned_no_results"
        return _serialize_forced_align_words(
            results[0],
            time_offset_s=float(self.tail_start_time_s),
        ), None

    def step(
        self,
        *,
        audio_tail: np.ndarray,
        sample_rate: int,
        chunk_idx: int,
        audio_processed_s: float,
        wallclock_s: float,
        is_final_chunk: bool,
    ) -> dict[str, Any]:
        prompt_prefix_before = self.stable_tail_prefix_text
        continuation_text, hypothesis_language, tail_hypothesis_text = (
            self._decode_tail_hypothesis(audio_tail=audio_tail)
        )
        aligned_tail_words, alignment_error = self._align_tail_hypothesis(
            audio_tail=audio_tail,
            sample_rate=sample_rate,
            tail_hypothesis_text=tail_hypothesis_text,
            hypothesis_language=hypothesis_language,
        )

        previous_tail_hypothesis_text = self.previous_tail_hypothesis_text
        raw_lcp_text = longest_common_prefix(
            previous_tail_hypothesis_text,
            tail_hypothesis_text,
        )
        tail_candidate_text = (
            tail_hypothesis_text
            if is_final_chunk
            else project_char_lcp_to_word_prefix_text(tail_hypothesis_text, raw_lcp_text)
        )

        stable_prefix_before = self.stable_tail_prefix_text
        stable_word_count_before = lexical_word_count(stable_prefix_before)
        append_only_conflict = not tail_candidate_text.startswith(stable_prefix_before)
        stable_prefix_after = (
            stable_prefix_before if append_only_conflict else tail_candidate_text
        )

        alignment_count_match = len(aligned_tail_words) == lexical_word_count(
            tail_hypothesis_text
        )
        commit_char_index = -1
        committed_segment_text = ""
        remainder_tail_text = tail_hypothesis_text
        committed_tail_word_count = 0
        new_committed_words: list[dict[str, Any]] = []

        if is_final_chunk:
            committed_segment_text = tail_hypothesis_text.strip()
            remainder_tail_text = ""
            committed_tail_word_count = lexical_word_count(committed_segment_text)
        else:
            commit_char_index = _find_commit_char_index(stable_prefix_after)
            if commit_char_index >= 0:
                committed_segment_text = tail_hypothesis_text[: commit_char_index + 1].strip()
                remainder_tail_text = tail_hypothesis_text[commit_char_index + 1 :].strip()
                committed_tail_word_count = lexical_word_count(committed_segment_text)

        committed_segment_end_time_s: float | None = None
        tail_start_time_before = float(self.tail_start_time_s)
        if committed_tail_word_count > 0 and alignment_count_match:
            if committed_tail_word_count <= len(aligned_tail_words):
                committed_segment_end_time_s = float(
                    aligned_tail_words[committed_tail_word_count - 1]["end_time"]
                )
                tail_surfaces = lexical_word_surfaces(tail_hypothesis_text)
                for word_idx in range(committed_tail_word_count):
                    aligned_word = aligned_tail_words[word_idx]
                    row = {
                        "word_index": len(self.public_committed_words) + len(new_committed_words),
                        "surface": tail_surfaces[word_idx],
                        "start_time_s": float(aligned_word["start_time"]),
                        "end_time_s": float(aligned_word["end_time"]),
                        "committed_at_audio_processed_s": float(audio_processed_s),
                        "committed_at_wallclock_s": float(wallclock_s),
                        "chunk_idx": int(chunk_idx),
                    }
                    new_committed_words.append(row)

        did_commit_segment = bool(committed_segment_text) and (
            is_final_chunk or committed_segment_end_time_s is not None
        )
        if did_commit_segment:
            self.public_segments.append(committed_segment_text)
            self.public_committed_words.extend(new_committed_words)
            self.tail_start_time_s = (
                float(committed_segment_end_time_s)
                if committed_segment_end_time_s is not None
                else float(audio_processed_s)
            )
            self.previous_tail_hypothesis_text = remainder_tail_text
            # The remainder starts a fresh active tail. It has only been seen
            # once in that new frame of reference, so its stable prefix is empty.
            self.stable_tail_prefix_text = ""
        else:
            self.previous_tail_hypothesis_text = tail_hypothesis_text
            self.stable_tail_prefix_text = stable_prefix_after

        visible_tail_text = normalize_partial_asr_hypothesis(
            remainder_tail_text if did_commit_segment else tail_hypothesis_text
        )
        global_hypothesis_text = _join_segments(self.public_segments, visible_tail_text)
        global_candidate_text = _join_segments(
            self.public_segments if did_commit_segment else self.public_segments,
            (
                ""
                if did_commit_segment
                else normalize_partial_asr_hypothesis(stable_prefix_after)
            ),
        )
        visible_tail_words = (
            aligned_tail_words[committed_tail_word_count:]
            if did_commit_segment
            else aligned_tail_words
        )
        global_words = [
            {
                "text": str(row["surface"]),
                "start_time": float(row["start_time_s"]),
                "end_time": float(row["end_time_s"]),
            }
            for row in self.public_committed_words
        ] + visible_tail_words
        global_alignment_count_match = len(global_words) == lexical_word_count(
            global_hypothesis_text
        )

        self.stream_trace.append(
            {
                "update_idx": len(self.stream_trace),
                "chunk_idx": int(chunk_idx),
                "audio_processed_s": float(audio_processed_s),
                "wallclock_s": float(wallclock_s),
                "public_asr_text": global_hypothesis_text,
            }
        )

        return {
            "chunk_idx": int(chunk_idx),
            "audio_processed_s": float(audio_processed_s),
            "wallclock_s": float(wallclock_s),
            "is_final_chunk": bool(is_final_chunk),
            "result_is_valid": True,
            "tail_start_time_s_before": tail_start_time_before,
            "tail_start_time_s_after": float(self.tail_start_time_s),
            "prompt_prefix_text": prompt_prefix_before,
            "prompt_prefix_word_count": lexical_word_count(prompt_prefix_before),
            "continuation_text": continuation_text,
            "hypothesis_language": hypothesis_language or self.language,
            "tail_hypothesis_text": tail_hypothesis_text,
            "tail_hypothesis_word_count": lexical_word_count(tail_hypothesis_text),
            "previous_tail_hypothesis_text": previous_tail_hypothesis_text,
            "tail_lcp_text": raw_lcp_text,
            "tail_lcp_word_count": lexical_word_count(
                project_char_lcp_to_word_prefix_text(tail_hypothesis_text, raw_lcp_text)
            ),
            "local_agreement_tail_candidate_text": tail_candidate_text,
            "local_agreement_tail_candidate_word_count": lexical_word_count(
                tail_candidate_text
            ),
            "stable_tail_prefix_text_before": stable_prefix_before,
            "stable_tail_prefix_word_count_before": int(stable_word_count_before),
            "stable_tail_prefix_text_after": self.stable_tail_prefix_text,
            "stable_tail_prefix_word_count_after": lexical_word_count(
                self.stable_tail_prefix_text
            ),
            "append_only_conflict": bool(append_only_conflict),
            "alignment_word_count": len(aligned_tail_words),
            "alignment_count_match": bool(alignment_count_match),
            "did_commit_segment": bool(did_commit_segment),
            "committed_segment_text": committed_segment_text,
            "committed_segment_word_count": int(committed_tail_word_count),
            "committed_segment_end_time_s": committed_segment_end_time_s,
            "remainder_tail_text": remainder_tail_text,
            "remainder_tail_word_count": lexical_word_count(remainder_tail_text),
            "new_committed_words": new_committed_words,
            "hypothesis_text": global_hypothesis_text,
            "hypothesis_word_count": lexical_word_count(global_hypothesis_text),
            "local_agreement_candidate_text": global_candidate_text,
            "local_agreement_candidate_word_count": lexical_word_count(
                global_candidate_text
            ),
            "public_asr_text": global_hypothesis_text,
            "words": global_words,
            "diagnostics": {
                "alignment_error": alignment_error,
                "global_alignment_count_match": bool(global_alignment_count_match),
            },
        }
