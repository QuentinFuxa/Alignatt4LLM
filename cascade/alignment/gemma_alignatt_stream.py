"""Token-level AlignAtt streaming state for the Gemma vLLM ASR backend.

Mirrors the policy of simul_whisper (``ufal/SimulStreaming``) adapted to
Gemma's chat-template prefill. The point of this module is to hold the
one invariant that makes AlignAtt streaming correct:

    The forced decoder prefix sent to the model corresponds exactly to
    committed tokens whose aligned audio-frame end lies strictly inside
    the currently visible audio window. Tokens whose alignment has
    fallen out of the window are kept in the external transcript but
    are never re-fed to the model.

All streaming state lives here (not scattered across the session) so
that invariant can be audited in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

if TYPE_CHECKING:
    from cascade.alignment.gemma_vllm_asr_backend import GemmaVLLMASRBackend


@dataclass(frozen=True)
class CommittedToken:
    """One committed decoder token with its absolute audio-frame anchor."""

    token_id: int
    text: str
    end_frame_abs: int


@dataclass(frozen=True)
class AlignAttStepRaw:
    """Backend-level per-step primitive output.

    All frame indices are relative to the audio window that was fed to
    the backend. ``content_frame_len`` is the number of real audio
    frames in that window (i.e. before any padding that the audio
    encoder may add internally).
    """

    generated_token_ids: tuple[int, ...]
    per_token_audio_frame_argmax: tuple[int, ...]
    content_frame_len: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamStepDelta:
    new_committed_tokens: list[CommittedToken] = field(default_factory=list)
    partial_tail_text: str = ""
    audio_window_start_frame_abs: int = 0
    audio_window_content_frames: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def committed_delta_text(self) -> str:
        return "".join(t.text for t in self.new_committed_tokens)

    @property
    def last_committed_end_frame_abs(self) -> int | None:
        if not self.new_committed_tokens:
            return None
        return int(self.new_committed_tokens[-1].end_frame_abs)


class GemmaAlignAttStream:
    """Owns the full AlignAtt streaming state for one ASR stream."""

    def __init__(
        self,
        *,
        backend: "GemmaVLLMASRBackend",
        language: str,
        frame_threshold: int = 4,
        rewind_threshold: int = 200,
        commit_policy: str = "frontier_flush",
        max_new_tokens: int | None = None,
    ) -> None:
        self.backend = backend
        self.language = str(language)
        self.frame_threshold = int(frame_threshold)
        self.rewind_threshold = int(rewind_threshold)
        self.commit_policy = str(commit_policy)
        self.max_new_tokens = int(max_new_tokens) if max_new_tokens else int(
            backend.max_new_tokens
        )

        sr = int(backend.sample_rate)
        ms = float(backend.audio_ms_per_token)
        self._samples_per_audio_frame = max(1, int(round(ms * sr / 1000.0)))
        self._max_window_samples = int(round(float(backend.max_audio_seconds) * sr))

        self.committed_tokens: list[CommittedToken] = []
        self._out_of_window_count: int = 0

    # ------------------------------------------------------------------
    # External accessors
    # ------------------------------------------------------------------

    @property
    def committed_text(self) -> str:
        return "".join(t.text for t in self.committed_tokens)

    @property
    def samples_per_audio_frame(self) -> int:
        return self._samples_per_audio_frame

    @property
    def max_window_samples(self) -> int:
        return self._max_window_samples

    def reset(self) -> None:
        self.committed_tokens = []
        self._out_of_window_count = 0

    # ------------------------------------------------------------------
    # Commit-time absolute end time for the last committed token
    # ------------------------------------------------------------------

    def last_committed_end_seconds(self) -> float:
        if not self.committed_tokens:
            return 0.0
        frame = int(self.committed_tokens[-1].end_frame_abs)
        ms_per_frame = float(self.backend.audio_ms_per_token)
        return float(frame + 1) * ms_per_frame / 1000.0

    # ------------------------------------------------------------------
    # Per-chunk step
    # ------------------------------------------------------------------

    def step(
        self,
        utterance_audio: np.ndarray,
        *,
        is_final_chunk: bool = False,
    ) -> StreamStepDelta:
        """Run one AlignAtt streaming step on the full utterance audio so far.

        ``utterance_audio`` is the complete live waveform since the start
        of this stream (the stream does not own audio — it is passed in
        every step). The stream owns only the committed-token list and
        the policy.
        """
        audio = np.asarray(utterance_audio, dtype=np.float32)
        if len(audio) == 0:
            return StreamStepDelta()

        # 1. Audio window: the trailing `max_window_samples`. Align the
        #    window start UP to an audio-frame boundary so per-token
        #    argmax frame indices are integers with a stable absolute-
        #    frame offset. Rounding UP (not down) is required to keep
        #    `len(audio_window) <= max_window_samples`; rounding down
        #    would enlarge the window past the encoder cap on partial-
        #    frame offsets.
        if len(audio) <= self._max_window_samples:
            win_start_sample = 0
        else:
            raw_start = len(audio) - self._max_window_samples
            win_start_sample = (
                (raw_start + self._samples_per_audio_frame - 1)
                // self._samples_per_audio_frame
            ) * self._samples_per_audio_frame
        audio_window = np.ascontiguousarray(audio[win_start_sample:])
        win_start_frame_abs = win_start_sample // self._samples_per_audio_frame
        expected_content_frames = int(
            np.ceil(len(audio_window) / float(self._samples_per_audio_frame))
        )

        # 2. Drop committed tokens whose alignment has fallen outside
        #    the window. They stay in ``self.committed_tokens`` for the
        #    external transcript; they are just no longer part of the
        #    model prompt. The comparison is ``>=`` because a token
        #    whose aligned frame sits exactly at the window-start frame
        #    is still supported by audio visible to the model.
        retained: list[CommittedToken] = [
            tok for tok in self.committed_tokens if tok.end_frame_abs >= win_start_frame_abs
        ]
        dropped_count = len(self.committed_tokens) - len(retained)
        self._out_of_window_count += max(0, dropped_count)
        forced_prefix_token_ids = tuple(int(tok.token_id) for tok in retained)

        # 3. Budget check. Over-budget is a real failure mode, not a
        #    place to silently trim audio away from the prefix (that's
        #    the exact desync bug the redesign removes). If this ever
        #    fires in practice we surface it through diagnostics and
        #    skip the step.
        fits = self.backend.can_fit_step(
            audio_window_samples=len(audio_window),
            forced_prefix_token_count=len(forced_prefix_token_ids),
            language=self.language,
        )
        if not fits:
            return StreamStepDelta(
                audio_window_start_frame_abs=int(win_start_frame_abs),
                audio_window_content_frames=int(expected_content_frames),
                diagnostics={
                    "commit_policy": str(self.commit_policy),
                    "budget_exceeded": True,
                    "forced_prefix_token_count": len(forced_prefix_token_ids),
                    "audio_window_samples": int(len(audio_window)),
                },
            )

        # 4. One generate + attention-capture step.
        step_raw = self.backend.alignatt_step(
            audio_window=audio_window,
            forced_prefix_token_ids=forced_prefix_token_ids,
            language=self.language,
            max_new_tokens=self.max_new_tokens,
        )
        if step_raw is None or not step_raw.generated_token_ids:
            return StreamStepDelta(
                audio_window_start_frame_abs=int(win_start_frame_abs),
                audio_window_content_frames=int(expected_content_frames),
                diagnostics={
                    "commit_policy": str(self.commit_policy),
                    "empty_generation": True,
                    "forced_prefix_token_count": len(forced_prefix_token_ids),
                    **(
                        {"backend": dict(step_raw.diagnostics)}
                        if step_raw is not None and step_raw.diagnostics
                        else {}
                    ),
                },
            )

        content_frame_len = int(step_raw.content_frame_len)
        if content_frame_len <= 0:
            return StreamStepDelta(
                audio_window_start_frame_abs=int(win_start_frame_abs),
                audio_window_content_frames=int(expected_content_frames),
                diagnostics={
                    "commit_policy": str(self.commit_policy),
                    "empty_content_frames": True,
                },
            )

        # 5. Token-level commit walk (simul_whisper §4).
        #
        # For each generated token i:
        #   - the raw attention argmax is clipped into the visible audio window;
        #   - under ``frontier_flush`` we project the local path onto the least
        #     monotone majorant anchored at the previous commit frontier, then
        #     commit the maximal prefix that lies strictly before the trailing
        #     ``frame_threshold`` band of the chunk;
        #   - under ``rewind_abort`` we preserve the historical behaviour:
        #     stop when the token enters the trailing band, or abort the chunk
        #     if it rewinds too far before the previous commit frontier.
        #
        # On the final chunk, commit every generated token; no more
        # audio is coming so holding back is pointless.
        gen_ids = list(int(t) for t in step_raw.generated_token_ids)
        argmaxes = list(int(a) for a in step_raw.per_token_audio_frame_argmax)
        n = min(len(gen_ids), len(argmaxes))

        if retained:
            prev_commit_frame_local = max(
                0, int(retained[-1].end_frame_abs) - int(win_start_frame_abs)
            )
        else:
            prev_commit_frame_local = 0

        frontier_stop_local = max(0, content_frame_len - self.frame_threshold)
        clipped_argmaxes: list[int] = []
        projected_argmaxes: list[int] = []
        raw_rewind_count = 0
        raw_max_rewind_frames = 0
        projected_repair_count = 0
        projected_max_repair_frames = 0
        running_floor = int(prev_commit_frame_local)
        for i in range(n):
            am = argmaxes[i]
            if am < 0:
                am = 0
            if am >= content_frame_len:
                am = content_frame_len - 1
            clipped_argmaxes.append(int(am))

            if am < running_floor:
                raw_rewind_count += 1
                raw_max_rewind_frames = max(raw_max_rewind_frames, running_floor - am)

            projected = max(int(am), int(running_floor))
            if projected != am:
                projected_repair_count += 1
                projected_max_repair_frames = max(
                    projected_max_repair_frames,
                    projected - am,
                )
            projected_argmaxes.append(int(projected))
            running_floor = int(projected)

        new_committed: list[CommittedToken] = []
        aborted_by_rewind = False
        accepted_count = 0
        if self.commit_policy == "rewind_abort":
            for i in range(n):
                am = clipped_argmaxes[i]
                if not is_final_chunk:
                    if am >= frontier_stop_local:
                        break
                    if am < prev_commit_frame_local - self.rewind_threshold:
                        aborted_by_rewind = True
                        break

                new_committed.append(
                    CommittedToken(
                        token_id=int(gen_ids[i]),
                        text=self.backend.decode_single_token(int(gen_ids[i])),
                        end_frame_abs=int(win_start_frame_abs) + int(am),
                    )
                )
                accepted_count = i + 1
        else:
            for i in range(n):
                am = projected_argmaxes[i]
                if not is_final_chunk and am >= frontier_stop_local:
                    break
                new_committed.append(
                    CommittedToken(
                        token_id=int(gen_ids[i]),
                        text=self.backend.decode_single_token(int(gen_ids[i])),
                        end_frame_abs=int(win_start_frame_abs) + int(am),
                    )
                )
                accepted_count = i + 1

        # Tail = tokens generated but not committed on this step.
        # Decoded purely so callers can render a display-only live
        # partial; the tail is NOT added to self.committed_tokens.
        if accepted_count < n and not aborted_by_rewind:
            tail_ids = gen_ids[accepted_count:n]
            partial_tail_text = self.backend.decode_token_ids(tail_ids)
        else:
            partial_tail_text = ""

        self.committed_tokens.extend(new_committed)

        diagnostics = {
            "commit_policy": str(self.commit_policy),
            "generated_count": int(n),
            "accepted_count": int(len(new_committed)),
            "aborted_by_rewind": bool(aborted_by_rewind),
            "frontier_stop_local": int(frontier_stop_local),
            "forced_prefix_token_count": int(len(forced_prefix_token_ids)),
            "out_of_window_cumulative": int(self._out_of_window_count),
            "content_frame_len": int(content_frame_len),
            "raw_rewind_count": int(raw_rewind_count),
            "raw_max_rewind_frames": int(raw_max_rewind_frames),
            "projected_repair_count": int(projected_repair_count),
            "projected_max_repair_frames": int(projected_max_repair_frames),
            "audio_window_samples": int(len(audio_window)),
            "audio_window_s": float(len(audio_window)) / float(self.backend.sample_rate),
        }
        if step_raw.diagnostics:
            diagnostics["backend"] = dict(step_raw.diagnostics)

        return StreamStepDelta(
            new_committed_tokens=new_committed,
            partial_tail_text=partial_tail_text,
            audio_window_start_frame_abs=int(win_start_frame_abs),
            audio_window_content_frames=int(content_frame_len),
            diagnostics=diagnostics,
        )
