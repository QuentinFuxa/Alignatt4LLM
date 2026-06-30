"""Live attention trace: show where each streamed token's attention lands.

This module is pure formatting and data-shaping. It deliberately has no
torch/transformers dependency so it can be unit-tested without loading a model.

Two streaming paths feed one renderer:

- ASR (``alignatt-gemma-asr``): the source axis is the audio timeline, so a
  token's locator is ``src@<frame> (<seconds>)``. The Gemma ASR path computes
  only audio-frame argmaxes, so ASR lines carry no attention-mass tail.
- MT (``alignatt-batch`` / ``alignatt-compare``): the source axis is the source
  prompt, so the locator is ``src@<source-token>`` and each line carries the
  accessible / inaccessible attention-mass split that drives the cut decision.

The only domain-conditional pieces of a line are the locator string and the
optional mass tail, so both paths share :func:`format_attention_trace_line`.
"""
from __future__ import annotations

import sys
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from alignatt4llm.alignment.gemma_alignatt_stream import StreamStepDelta

# Fixed display widths so columns line up across mixed Latin/CJK tokens.
_TOKEN_FIELD_CELLS = 12
_VERDICT_CELLS = 6


def _display_width(text: str) -> int:
    """East-Asian-aware display width (wide/fullwidth glyphs count as 2)."""
    width = 0
    for ch in text:
        width += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return width


def _escape_inline(text: str) -> str:
    """Collapse whitespace that would break the one-event-per-line invariant."""
    return text.replace("\t", "\\t").replace("\r", "\\r").replace("\n", "\\n")


def _clip_to_cells(text: str, cells: int) -> str:
    """Clip ``text`` to at most ``cells`` display columns (whole glyphs only)."""
    out: list[str] = []
    width = 0
    for ch in text:
        w = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if width + w > cells:
            break
        out.append(ch)
        width += w
    return "".join(out)


def _token_field(text: str) -> str:
    """Render the quoted token, clipped and right-padded to a fixed width."""
    tok = _clip_to_cells(_escape_inline(text), _TOKEN_FIELD_CELLS - 2)
    quoted = f'"{tok}"'
    pad = max(1, _TOKEN_FIELD_CELLS - _display_width(quoted))
    return quoted + " " * pad


@dataclass(frozen=True)
class AttentionTraceTokenEvent:
    """One token's attention decision, ready to format.

    ``source_locator`` is preformatted by the producing adapter ("src@18
    (0.76s)" for ASR, "src@12" for MT) so the renderer stays domain-agnostic.
    The mass fields are populated on the MT path only.
    """

    chunk_idx: int
    verdict: str  # "commit" | "HOLD"
    token_text: str
    source_locator: str
    crossed_frontier: bool = False
    source_accessible_mass: float | None = None
    source_inaccessible_mass: float | None = None


def format_attention_trace_line(event: AttentionTraceTokenEvent) -> str:
    """Render one trace event as a single aligned line (no trailing newline)."""
    verb = f"{event.verdict:<{_VERDICT_CELLS}}"
    line = (
        f"[chunk {event.chunk_idx:>3}] {verb} "
        f"{_token_field(event.token_text)}→ {event.source_locator}"
    )
    if event.source_accessible_mass is not None:
        inacc = (
            event.source_inaccessible_mass
            if event.source_inaccessible_mass is not None
            else 0.0
        )
        line += f"  mass acc {event.source_accessible_mass:.2f} inacc {inacc:.2f}"
    if event.crossed_frontier:
        line += " > frontier → cut"
    return line


def make_stderr_trace_printer(
    level: str = "all",
) -> Callable[[list[AttentionTraceTokenEvent]], None]:
    """Build a sink that prints trace events to stderr, one per line.

    ``level`` is ``"all"`` (commits + holds) or ``"commits"`` (commits only).
    Shared by every CLI so the trace looks identical for ASR and MT. stderr
    keeps the trace out of stdout/file artifacts.
    """

    def _printer(events: list[AttentionTraceTokenEvent]) -> None:
        for event in events:
            if level == "commits" and event.verdict != "commit":
                continue
            print(format_attention_trace_line(event), file=sys.stderr, flush=True)

    return _printer


def _asr_seconds(frame_abs: int, ms_per_token: float) -> float:
    """Absolute audio-frame index → end time in seconds.

    Mirrors ``GemmaAlignAttStream.last_committed_end_seconds`` so the trace
    matches the per-token commit log: a frame's end is ``(frame + 1) * ms``.
    """
    return float(frame_abs + 1) * float(ms_per_token) / 1000.0


def asr_trace_events_from_delta(
    *,
    chunk_idx: int,
    delta: "StreamStepDelta",
    ms_per_token: float,
    aborted_by_rewind: bool = False,
) -> list[AttentionTraceTokenEvent]:
    """Build trace events for one ASR streaming step from its delta.

    Committed tokens carry their absolute frame anchor directly; held tokens
    come from ``delta.held_tokens`` (text, absolute-frame) which the stream
    populates only when tracing is enabled.
    """
    events: list[AttentionTraceTokenEvent] = []
    for tok in delta.new_committed_tokens:
        frame = int(tok.end_frame_abs)
        events.append(
            AttentionTraceTokenEvent(
                chunk_idx=chunk_idx,
                verdict="commit",
                token_text=tok.text,
                source_locator=f"src@{frame} ({_asr_seconds(frame, ms_per_token):.2f}s)",
            )
        )
    for text, frame in getattr(delta, "held_tokens", None) or []:
        frame = int(frame)
        events.append(
            AttentionTraceTokenEvent(
                chunk_idx=chunk_idx,
                verdict="HOLD",
                token_text=str(text),
                source_locator=f"src@{frame} ({_asr_seconds(frame, ms_per_token):.2f}s)",
                crossed_frontier=True,
            )
        )
    return events


def _provenance_mass(row: Any, key: str) -> float | None:
    if isinstance(row, Mapping):
        value = row.get(key)
        return None if value is None else float(value)
    return None


def mt_trace_events_from_metadata(
    *,
    chunk_idx: int,
    draft_token_texts: Sequence[str],
    alignatt_metadata: Mapping[str, Any] | None,
) -> list[AttentionTraceTokenEvent]:
    """Build trace events for one MT draft from ``MTBackendResult.alignatt_metadata``.

    No MT struct change is needed: every field is already present
    (``aligned_source_local_positions``, ``provenance_per_draft_token``,
    ``accepted_candidate_token_count``, ``unsafe_reason`` /
    ``blocked_source_local_position``).
    """
    meta: Mapping[str, Any] = alignatt_metadata or {}
    positions = list(meta.get("aligned_source_local_positions") or [])
    provenance = list(meta.get("provenance_per_draft_token") or [])
    accepted_raw = meta.get("accepted_candidate_token_count")
    accepted = int(accepted_raw) if accepted_raw is not None else len(draft_token_texts)
    cut_present = (
        meta.get("unsafe_reason") is not None
        or meta.get("blocked_source_local_position") is not None
    )

    events: list[AttentionTraceTokenEvent] = []
    for i, text in enumerate(draft_token_texts):
        pos = positions[i] if i < len(positions) else None
        locator = f"src@{int(pos)}" if pos is not None else "src@?"
        row = provenance[i] if i < len(provenance) else None
        committed = i < accepted
        events.append(
            AttentionTraceTokenEvent(
                chunk_idx=chunk_idx,
                verdict="commit" if committed else "HOLD",
                token_text=str(text),
                source_locator=locator,
                crossed_frontier=(not committed) and cut_present,
                source_accessible_mass=_provenance_mass(row, "source_accessible"),
                source_inaccessible_mass=_provenance_mass(row, "source_inaccessible"),
            )
        )
    return events
