"""Unit tests for the live attention trace formatter and adapters.

Pure-Python: no model load, no numpy/torch, runnable under ``.venv-dev``.
The ASR delta is duck-typed with ``SimpleNamespace`` (matching the project's
lightweight-fixture test style) so this never imports the streaming stack.
"""
from __future__ import annotations

from types import SimpleNamespace

from alignatt4llm.alignment.attention_trace import (
    AttentionTraceTokenEvent,
    asr_trace_events_from_delta,
    format_attention_trace_line,
    mt_trace_events_from_metadata,
)


def test_format_commit_line_has_no_mass_or_frontier_tail():
    line = format_attention_trace_line(
        AttentionTraceTokenEvent(
            chunk_idx=3,
            verdict="commit",
            token_text="the",
            source_locator="src@18 (0.76s)",
        )
    )
    assert line.startswith("[chunk   3] commit ")
    assert '"the"' in line
    assert "→ src@18 (0.76s)" in line
    assert "mass" not in line
    assert "frontier" not in line


def test_format_hold_line_appends_frontier_cut():
    line = format_attention_trace_line(
        AttentionTraceTokenEvent(
            chunk_idx=4,
            verdict="HOLD",
            token_text="attends",
            source_locator="src@31 (1.28s)",
            crossed_frontier=True,
        )
    )
    assert " HOLD  " in line
    assert line.endswith("> frontier → cut")


def test_format_mt_line_includes_mass_and_cjk_token():
    line = format_attention_trace_line(
        AttentionTraceTokenEvent(
            chunk_idx=7,
            verdict="commit",
            token_text="我们",
            source_locator="src@12",
            source_accessible_mass=0.91,
            source_inaccessible_mass=0.04,
        )
    )
    assert '"我们"' in line
    assert "mass acc 0.91 inacc 0.04" in line


def test_whitespace_in_token_never_breaks_the_line():
    line = format_attention_trace_line(
        AttentionTraceTokenEvent(
            chunk_idx=1,
            verdict="commit",
            token_text="a\nb\tc",
            source_locator="src@5",
        )
    )
    assert "\n" not in line and "\t" not in line
    assert "\\n" in line and "\\t" in line


def test_asr_adapter_builds_commit_and_hold_events():
    delta = SimpleNamespace(
        new_committed_tokens=[
            SimpleNamespace(text="the", end_frame_abs=18),
            SimpleNamespace(text="model", end_frame_abs=24),
        ],
        held_tokens=[("attends", 31)],
    )
    events = asr_trace_events_from_delta(chunk_idx=4, delta=delta, ms_per_token=40.0)

    assert [e.verdict for e in events] == ["commit", "commit", "HOLD"]
    # frame 18 → (18 + 1) * 40 / 1000 = 0.76s, mirroring last_committed_end_seconds.
    assert events[0].source_locator == "src@18 (0.76s)"
    assert events[1].source_locator == "src@24 (1.00s)"
    assert events[2].source_locator == "src@31 (1.28s)"
    assert events[2].crossed_frontier is True
    # ASR carries no attention mass.
    assert all(e.source_accessible_mass is None for e in events)


def test_asr_adapter_with_no_held_tokens():
    delta = SimpleNamespace(
        new_committed_tokens=[SimpleNamespace(text="ok", end_frame_abs=0)],
        held_tokens=[],
    )
    events = asr_trace_events_from_delta(chunk_idx=1, delta=delta, ms_per_token=40.0)
    assert [e.verdict for e in events] == ["commit"]
    assert events[0].source_locator == "src@0 (0.04s)"


def test_mt_adapter_reads_positions_provenance_and_acceptance():
    meta = {
        "aligned_source_local_positions": [12, 18, 27],
        "provenance_per_draft_token": [
            {"source_accessible": 0.91, "source_inaccessible": 0.04},
            {"source_accessible": 0.88, "source_inaccessible": 0.07},
            {"source_accessible": 0.31, "source_inaccessible": 0.58},
        ],
        "accepted_candidate_token_count": 2,
        "unsafe_reason": "frontier",
        "blocked_source_local_position": 27,
    }
    events = mt_trace_events_from_metadata(
        chunk_idx=8,
        draft_token_texts=["我们", "需要", "增长"],
        alignatt_metadata=meta,
    )
    assert [e.verdict for e in events] == ["commit", "commit", "HOLD"]
    assert events[0].source_locator == "src@12"
    assert events[0].source_accessible_mass == 0.91
    assert events[2].crossed_frontier is True
    line = format_attention_trace_line(events[2])
    assert "mass acc 0.31 inacc 0.58" in line
    assert line.endswith("> frontier → cut")


def test_mt_adapter_tolerates_missing_metadata():
    events = mt_trace_events_from_metadata(
        chunk_idx=0, draft_token_texts=["a", "b"], alignatt_metadata=None
    )
    # No positions/provenance: default acceptance = all committed, locator src@?.
    assert [e.verdict for e in events] == ["commit", "commit"]
    assert events[0].source_locator == "src@?"
    assert events[0].source_accessible_mass is None
    assert events[0].crossed_frontier is False
