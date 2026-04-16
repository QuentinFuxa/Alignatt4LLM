"""Unit tests for the EOS-flush behaviour of the ASR commit rules.

Both the default ``punctuation_lcp`` commit rule and the opt-in
``alignatt_frontier`` rule have a failure mode at end-of-stream: the
final ~500 ms of audio worth of words can't clear the "safely behind the
frontier" gate (alignatt_frontier) or lack a sentence-final punctuation
cue (punctuation_lcp). Before the EOS-flush fix the public hypothesis
lost the trailing 1-2 words on every clip, costing BLEU/chrF/COMET on
every submission.

These tests pin the fix: ``is_final_chunk=True`` on both commit helpers
flushes the full remaining ASR hypothesis. No model / GPU needed — the
tests construct an ``AlignmentResult`` directly and call the commit
helpers in-process.
"""
from __future__ import annotations

import numpy as np

from alignment_backend import AlignmentResult, WordAlignment
from cascade_runtime import CascadeRuntimeConfig, LoadedModelBundle


def _new_session(**overrides):
    # A bundle + session with no model loading. The commit rules touch
    # only CascadeState + CascadeRuntimeConfig, not the backends.
    config = CascadeRuntimeConfig(**overrides)
    bundle = LoadedModelBundle(config)
    return bundle.new_session()


def _result(text: str, word_timings: list[tuple[str, float, float]]) -> AlignmentResult:
    return AlignmentResult(
        text=text,
        words=tuple(
            WordAlignment(text=t, start_time=float(s), end_time=float(e))
            for t, s, e in word_timings
        ),
        audio_duration_s=float(word_timings[-1][2]) if word_timings else 0.0,
    )


# ---------------- alignatt_frontier ----------------

def test_alignatt_frontier_non_final_respects_margin():
    # audio ends at 10.0 s, margin 500 ms ⇒ safe_end_time_s = 9.5 s.
    # Words at 9.6 and 9.9 must NOT commit under a normal (non-final) chunk.
    session = _new_session(
        asr_commit_mode="alignatt_frontier",
        asr_alignatt_frontier_margin_ms=500.0,
    )
    # prime LCP via two identical hypotheses (LCP = full asr_hypo)
    hypo = "hi I am Si Yuan from Fudan University and I"
    session.state.asr_hypotheses = [hypo, hypo]
    audio = np.zeros(int(10.0 * 16000), dtype=np.float32)
    result = _result(
        hypo,
        [
            ("hi", 0.0, 0.5),
            ("I", 0.5, 0.8),
            ("am", 0.8, 1.2),
            ("Si", 1.2, 1.8),
            ("Yuan", 1.8, 2.3),
            ("from", 2.3, 2.7),
            ("Fudan", 2.7, 3.3),
            ("University", 3.3, 4.4),
            ("and", 9.6, 9.8),   # past the 9.5 s margin
            ("I", 9.8, 9.9),     # past the 9.5 s margin
        ],
    )

    session._try_commit_alignatt_frontier(
        asr_hypo=hypo, result=result, lcp_text=hypo, audio=audio,
        is_final_chunk=False,
    )
    # Exactly the first 8 words (up through "University") should commit; the
    # two trailing words beyond the margin stay in the asr_hypotheses tail.
    assert len(session.state.utt_sources) == 2, "one commit expected"
    committed_text = session.state.utt_sources[-1]
    assert committed_text.split() == hypo.split()[:8]
    assert session.state.asr_hypotheses[-1].strip() == "and I"


def test_alignatt_frontier_final_commits_past_margin():
    # Same setup, but is_final_chunk=True ⇒ all words must commit.
    session = _new_session(
        asr_commit_mode="alignatt_frontier",
        asr_alignatt_frontier_margin_ms=500.0,
    )
    hypo = "hi I am Si Yuan from Fudan University and I"
    session.state.asr_hypotheses = [hypo, hypo]
    audio = np.zeros(int(10.0 * 16000), dtype=np.float32)
    result = _result(
        hypo,
        [
            ("hi", 0.0, 0.5),
            ("I", 0.5, 0.8),
            ("am", 0.8, 1.2),
            ("Si", 1.2, 1.8),
            ("Yuan", 1.8, 2.3),
            ("from", 2.3, 2.7),
            ("Fudan", 2.7, 3.3),
            ("University", 3.3, 4.4),
            ("and", 9.6, 9.8),
            ("I", 9.8, 9.9),
        ],
    )

    session._try_commit_alignatt_frontier(
        asr_hypo=hypo, result=result, lcp_text=hypo, audio=audio,
        is_final_chunk=True,
    )
    assert len(session.state.utt_sources) == 2
    committed_text = session.state.utt_sources[-1]
    # Every word flushes; asr_hypotheses tail is now empty.
    assert committed_text.split() == hypo.split()
    assert session.state.asr_hypotheses[-1].strip() == ""


def test_alignatt_frontier_final_bypasses_lcp_gate():
    # Even with an empty LCP (nothing to trust across successive hypotheses),
    # is_final_chunk=True must still flush because no further audio is coming.
    session = _new_session(
        asr_commit_mode="alignatt_frontier",
        asr_alignatt_frontier_margin_ms=500.0,
    )
    hypo = "hi there"
    # lcp_text empty ⇒ non-final path would return early
    audio = np.zeros(int(5.0 * 16000), dtype=np.float32)
    result = _result(
        hypo,
        [("hi", 0.0, 0.5), ("there", 0.5, 1.0)],
    )
    session.state.asr_hypotheses = [hypo, hypo]

    session._try_commit_alignatt_frontier(
        asr_hypo=hypo, result=result, lcp_text="", audio=audio,
        is_final_chunk=True,
    )
    assert session.state.utt_sources[-1].split() == ["hi", "there"]


# ---------------- punctuation_lcp ----------------

def test_punctuation_lcp_non_final_requires_terminal_punct():
    session = _new_session(asr_commit_mode="punctuation_lcp")
    hypo = "hello world we are here"
    session.state.asr_hypotheses = [hypo, hypo]
    result = _result(
        hypo,
        [
            ("hello", 0.0, 0.5),
            ("world", 0.5, 1.0),
            ("we", 1.0, 1.3),
            ("are", 1.3, 1.6),
            ("here", 1.6, 2.1),
        ],
    )
    # No ". ! ?" in lcp_text and hypo does not end in one either → no commit.
    session._try_commit_punctuation_lcp(
        asr_hypo=hypo, result=result, lcp_text=hypo, is_final_chunk=False,
    )
    assert len(session.state.utt_sources) == 1


def test_punctuation_lcp_final_commits_without_terminal_punct():
    session = _new_session(asr_commit_mode="punctuation_lcp")
    hypo = "hello world we are here"
    session.state.asr_hypotheses = [hypo, hypo]
    result = _result(
        hypo,
        [
            ("hello", 0.0, 0.5),
            ("world", 0.5, 1.0),
            ("we", 1.0, 1.3),
            ("are", 1.3, 1.6),
            ("here", 1.6, 2.1),
        ],
    )
    session._try_commit_punctuation_lcp(
        asr_hypo=hypo, result=result, lcp_text=hypo, is_final_chunk=True,
    )
    assert len(session.state.utt_sources) == 2
    assert session.state.utt_sources[-1] == hypo
    assert session.state.asr_hypotheses[-1] == ""


def test_punctuation_lcp_final_still_handles_terminal_punct_cleanly():
    # Regression guard: the final-flush path must not break the normal
    # punctuation behaviour when a sentence terminal IS present.
    session = _new_session(asr_commit_mode="punctuation_lcp")
    hypo = "hello world."
    session.state.asr_hypotheses = [hypo, hypo]
    result = _result(
        hypo,
        [("hello", 0.0, 0.5), ("world", 0.5, 1.0)],
    )
    session._try_commit_punctuation_lcp(
        asr_hypo=hypo, result=result, lcp_text=hypo, is_final_chunk=True,
    )
    # Final flush commits everything, period included.
    assert session.state.utt_sources[-1] == hypo
    assert session.state.asr_hypotheses[-1] == ""


# ---------------- finalize_stream wiring ----------------

def test_transcribe_audio_accepts_is_final_chunk_kwarg():
    # Lightweight integration check: the public signature must accept the
    # kwarg, because finalize_stream relies on it. We don't run the ASR
    # backend here; we only verify the signature contract.
    import inspect

    from cascade_runtime import CascadeSession

    sig = inspect.signature(CascadeSession.transcribe_audio)
    assert "is_final_chunk" in sig.parameters
    assert sig.parameters["is_final_chunk"].default is False


def test_finalize_stream_passes_is_final_chunk_to_transcribe_audio(monkeypatch):
    session = _new_session()
    calls: list[dict] = []

    def fake_transcribe_audio(self, *, is_final_chunk: bool = False):
        calls.append({"is_final_chunk": is_final_chunk})
        return "dummy"

    # The runtime also calls render_translation, which wants the MT backend.
    # Stub it to avoid the load.
    def fake_render_translation(self):
        return "dummy_translation", None

    from cascade_runtime import CascadeSession

    monkeypatch.setattr(CascadeSession, "transcribe_audio", fake_transcribe_audio)
    monkeypatch.setattr(CascadeSession, "render_translation", fake_render_translation)

    session.finalize_stream()

    assert len(calls) == 1
    assert calls[0]["is_final_chunk"] is True
