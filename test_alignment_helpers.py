"""Pure-Python invariants for the Gemma audio alignment helpers.

These tests do not load any model; they lock the data-flow rules that
make the rest of the pipeline correct. We want:

- ``detect_audio_span`` actually finds a contiguous audio-token span
- ``audio_position_to_end_seconds`` respects the known 40 ms calibration
- ``_enforce_monotone`` is idempotent and non-decreasing
- ``split_text_into_word_spans`` matches Qwen's punctuation-stripping
  word-unit convention so the downstream cascade treats the two
  backends identically
- ``aggregate_token_timings_to_words`` turns per-token end-times into
  per-word end-times without introducing non-monotone regressions
"""

from __future__ import annotations


def test_detect_audio_span_returns_contiguous_range():
    from gemma_alignment_probe import detect_audio_span

    ids = [1, 2, 3, 258881, 258881, 258881, 7]
    span = detect_audio_span(ids, audio_token_id=258881, audio_ms_per_token=40.0)
    assert span is not None
    assert span.prompt_start == 3
    assert span.prompt_end == 6
    assert span.length == 3
    assert span.ms_per_token == 40.0


def test_detect_audio_span_returns_none_when_absent():
    from gemma_alignment_probe import detect_audio_span

    assert detect_audio_span([1, 2, 3], audio_token_id=258881, audio_ms_per_token=40.0) is None


def test_audio_position_uses_40ms_calibration():
    from gemma_alignment_probe import audio_position_to_end_seconds

    assert audio_position_to_end_seconds(0, ms_per_token=40.0, audio_duration_s=30.0) == 0.04
    assert audio_position_to_end_seconds(24, ms_per_token=40.0, audio_duration_s=30.0) == 1.0
    # Clamps to audio duration.
    assert audio_position_to_end_seconds(1000, ms_per_token=40.0, audio_duration_s=3.5) == 3.5
    assert audio_position_to_end_seconds(None, ms_per_token=40.0, audio_duration_s=1.0) is None


def test_enforce_monotone_projects_to_running_max():
    from gemma_alignment_probe import _enforce_monotone

    assert _enforce_monotone([0.1, 0.2, 0.15, 0.3, 0.25]) == [0.1, 0.2, 0.2, 0.3, 0.3]
    assert _enforce_monotone([0.0, None, 0.3, None, 0.2]) == [0.0, None, 0.3, None, 0.3]


def test_split_text_into_word_spans_strips_trailing_punctuation():
    from gemma_alignment_probe import split_text_into_word_spans

    spans = split_text_into_word_spans("Hello, world!")
    assert [span[2] for span in spans] == ["Hello", "world"]

    spans2 = split_text_into_word_spans("(hello)")
    assert [span[2] for span in spans2] == ["hello"]


def test_aggregate_token_timings_preserves_monotonicity():
    _TOKEN_TABLE = {
        10: "Hel",
        11: "lo",
        12: ",",
        13: " wor",
        14: "ld",
        15: "!",
    }

    class StubTokenizer:
        def decode(self, ids, skip_special_tokens=False):
            return "".join(_TOKEN_TABLE[i] for i in ids)

    generated_ids = [10, 11, 12, 13, 14, 15]
    # Per-token end times (monotone).
    token_ends = [0.12, 0.20, 0.20, 0.50, 0.80, 0.80]

    from gemma_alignment_probe import aggregate_token_timings_to_words

    words = aggregate_token_timings_to_words(
        "Hello, world!",
        generated_ids=generated_ids,
        tokenizer=StubTokenizer(),
        token_end_times_s=token_ends,
        audio_duration_s=1.0,
    )
    # Two words: "Hello", "world"
    assert [w.text for w in words] == ["Hello", "world"]
    # Each word's end time is monotone non-decreasing.
    assert words[0].end_time <= words[1].end_time
    assert words[0].end_time <= 0.20 + 1e-9
    assert words[1].end_time <= 0.80 + 1e-9


def test_monotonicity_score_rewards_forward_progress():
    from gemma_alignment_probe import monotonicity_score

    assert monotonicity_score([0, 1, 2, 3]) == 1.0
    assert monotonicity_score([0, 0, 0, 0]) == 1.0
    # One backward jump across three transitions -> 2/3.
    assert abs(monotonicity_score([0, 1, 0, 2]) - (2.0 / 3.0)) < 1e-9
    assert monotonicity_score([None, None]) == 0.0


def test_audio_too_long_raises_with_explicit_error():
    """Long-audio guard must fail loudly (PLAN.md Phase 5)."""
    import numpy as np
    from gemma_alignment_probe import (
        GemmaAttentionAlignmentBackend,
        GemmaAudioTooLongError,
    )

    # Build a backend without loading the model; only the guard is exercised.
    backend = GemmaAttentionAlignmentBackend.__new__(GemmaAttentionAlignmentBackend)
    backend.max_audio_seconds = 30.0
    audio = np.zeros(31 * 16000, dtype=np.float32)  # 31 s

    raised = False
    try:
        backend._enforce_audio_cap(audio, sample_rate=16000)
    except GemmaAudioTooLongError as exc:
        raised = True
        assert "31" in str(exc) and "30" in str(exc)
    assert raised, "audio past cap must raise GemmaAudioTooLongError, not silently truncate"

    # In-cap audio returns the duration without raising.
    short = np.zeros(5 * 16000, dtype=np.float32)
    assert abs(backend._enforce_audio_cap(short, sample_rate=16000) - 5.0) < 1e-6


def test_qk_fast_prefix_slicing_preserves_audio_features():
    import torch
    from gemma_alignment_probe import GemmaAttentionAlignmentBackend

    inputs = {
        "input_ids": torch.tensor([[11, 12, 13, 14]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        "audio_features": torch.randn(1, 80, 16),
        "non_tensor": "keep",
    }
    sliced = GemmaAttentionAlignmentBackend._slice_inputs_to_prefix(inputs, 2)

    assert sliced["input_ids"].tolist() == [[11, 12]]
    assert sliced["attention_mask"].tolist() == [[1, 1]]
    assert tuple(sliced["audio_features"].shape) == (1, 80, 16)
    assert sliced["non_tensor"] == "keep"


def test_gemma_onepass_backend_uses_public_runtime_id():
    from gemma_alignment_probe import GemmaAttentionAlignmentBackend

    assert GemmaAttentionAlignmentBackend.name == "gemma_onepass_qk_fast"


def _run_all() -> None:
    failures = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception as exc:  # pragma: no cover - surface every failure
                failures.append((name, exc))
            else:
                print(f"ok  {name}")
    if failures:
        print("")
        for name, exc in failures:
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    _run_all()
