from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cascade.runtime import CascadeRuntimeConfig, CascadeSession, LoadedModelBundle
from simulstream.server.speech_processors import SAMPLE_RATE


def _session(*, min_commit_words: int) -> CascadeSession:
    config = CascadeRuntimeConfig(
        source_lang="English",
        target_lang="Simplified Chinese",
        asr_punctuation_min_commit_words=min_commit_words,
    )
    session = CascadeSession(LoadedModelBundle(config))
    session.state.source = np.zeros(SAMPLE_RATE * 10, dtype=np.float32)
    return session


def _word_alignment(end_time: float):
    return SimpleNamespace(end_time=end_time)


def test_short_punctuation_lcp_commit_is_legacy_default():
    session = _session(min_commit_words=0)
    result = SimpleNamespace(
        words=[
            _word_alignment(0.25),
            _word_alignment(0.50),
            _word_alignment(0.75),
            _word_alignment(1.00),
        ]
    )

    commit = session._try_commit_punctuation_lcp(
        asr_hypo="Hi everyone. I'm Jenny",
        result=result,
        lcp_text="Hi everyone.",
    )

    assert isinstance(commit, dict)
    assert commit["committed_text"] == "Hi everyone."
    assert commit["remainder_text"] == "I'm Jenny"


def test_short_punctuation_lcp_commit_can_be_delayed():
    session = _session(min_commit_words=3)
    result = SimpleNamespace(
        words=[
            _word_alignment(0.25),
            _word_alignment(0.50),
            _word_alignment(0.75),
            _word_alignment(1.00),
        ]
    )

    commit = session._try_commit_punctuation_lcp(
        asr_hypo="Hi everyone. I'm Jenny",
        result=result,
        lcp_text="Hi everyone.",
    )

    assert commit is None
    assert session.state.utt_sources == [""]


def test_asr_punctuation_min_commit_words_rejects_negative_values():
    with pytest.raises(ValueError, match="asr_punctuation_min_commit_words"):
        CascadeRuntimeConfig(asr_punctuation_min_commit_words=-1)


def test_committed_transcript_tail_takes_last_words_only():
    from cascade.runtime import committed_transcript_tail

    segments = ["", "hello world this is", "a committed transcript"]
    assert committed_transcript_tail(segments, max_words=4) == "is a committed transcript"
    assert committed_transcript_tail(segments, max_words=0) == ""
    assert committed_transcript_tail([], max_words=10) == ""
    assert committed_transcript_tail(["", "  "], max_words=10) == ""


def test_qwen_backend_passes_context_to_asr_transcribe():
    import numpy as np
    from types import SimpleNamespace
    from cascade.alignment.qwen_forced_backend import QwenAlignmentBackend

    captured = {}

    class _FakeASR:
        def transcribe(self, audio_tuple, *, language, context, return_time_stamps):
            captured["context"] = context
            captured["language"] = language
            return [SimpleNamespace(text="hi", time_stamps=())]

    backend = QwenAlignmentBackend.__new__(QwenAlignmentBackend)
    backend.asr = _FakeASR()
    backend.aligner = None

    result = backend.transcribe_and_align(
        np.zeros(1600, dtype=np.float32),
        sample_rate=16000,
        language="English",
        context="previously committed words",
    )

    assert captured["context"] == "previously committed words"
    assert captured["language"] == "English"
    # Empty timestamps short-circuit alignment; the context still reached ASR.
    assert result is None or result.text == "hi"
