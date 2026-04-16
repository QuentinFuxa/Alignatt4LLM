"""Gemma-only attention-based source alignment backend.

The paper-level question driving this code is whether a multimodal LLM's
self-attention to its own audio-placeholder tokens is strong enough to
replace an external forced aligner. Gemma 4 encodes audio as a contiguous
span of ``audio_token_id`` placeholders in the LLM input sequence at a
fixed ``audio_ms_per_token`` rate (40 ms on E4B). A generated transcript
token's attention distribution over that span is therefore a direct
text-to-audio alignment signal, callable at decode time.

The mechanism implemented here is exactly the one the MT side already
uses for text-to-text AlignAtt (see ``cascade_mt_backend``): a fixed
small set of alignment heads, per-head z-score normalization, median
filtering on the source axis, mean across heads, and argmax — only the
source axis here is the audio-placeholder span instead of a text source.
Audio-position argmaxes are converted to milliseconds via the processor's
``audio_ms_per_token`` (the authoritative calibration). Token-level
timestamps are then grouped into Qwen-style word-level timestamps using
the tokenizer's ``offset_mapping`` so the downstream cascade contract is
unchanged.

No lexical heuristics, no content-aware adjustments, no punctuation
tricks: every knob is a generic attention or tokenization artifact.

The teacher-forced alignment path supports two explicit probe backends:

- ``"eager"`` materializes self-attention weights directly
- ``"qk_fast"`` reconstructs transcript-token rows into the audio span
  from captured layer inputs plus KV snapshots under ``sdpa``-style
  attention, mirroring the MT AlignAtt fast path
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Sequence
import json
import math
import string

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from alignment_backend import (
    AlignAttObserverToken,
    AlignAttProvenanceBreakdown,
    AlignmentBackend,
    AlignmentResult,
    WordAlignment,
)
from cascade_mt_backend import (
    AlignAttHead,
    LayerInputCapture,
    SelectedAttentionRecorder,
    SelectedLayerInputRecorder,
    compute_alignatt_source_argmaxes,
    compute_key_states_from_layer_input_capture,
    compute_query_states_from_layer_input_capture,
    extract_source_attention_rows_per_token_from_fast_path,
    extract_source_attention_rows_per_token,
    map_attention_head_to_key_value_head,
)


GEMMA_AUDIO_TOKEN_ID_DEFAULT = 258881
GEMMA_AUDIO_MS_PER_TOKEN_DEFAULT = 40.0
# Gemma4 processor caps the audio token span at ``audio_seq_length`` tokens
# (750 on E4B = 30.0 s at 40 ms/token). Audio past that boundary is silently
# dropped by the processor, which would invalidate any timestamp produced for
# the dropped tail. We enforce the same cap explicitly so the failure is
# visible at the call site instead of corrupting downstream metrics.
GEMMA_AUDIO_MAX_SECONDS_DEFAULT = 30.0


class GemmaAudioTooLongError(ValueError):
    """Raised when an audio chunk exceeds the Gemma audio-encoder cap."""


class GemmaAudioQKFastError(RuntimeError):
    """Raised when the audio ``qk_fast`` probe cannot reconstruct rows."""


PUNCTUATION_STRIP = string.punctuation + "”’)]}"
PUNCTUATION_LEADING = "\"'`“”‘’([{"


@dataclass(frozen=True)
class AudioSpan:
    prompt_start: int
    prompt_end: int  # exclusive
    ms_per_token: float

    @property
    def length(self) -> int:
        return self.prompt_end - self.prompt_start


@dataclass(frozen=True)
class TokenTiming:
    token_id: int
    token_str: str
    aligned_audio_position: int | None
    end_time: float | None


@dataclass(frozen=True)
class AudioAttentionProbeResult:
    source_attention_rows_per_token: tuple[torch.Tensor, ...]
    probe_backend: str
    diagnostics: dict[str, Any]


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
    (pure-punctuation tokens) are dropped. This mirrors the source-unit
    logic already used by ``cascade_source_frontier.iter_source_word_spans``.
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


def aggregate_token_timings_to_words(
    text: str,
    *,
    generated_ids: Sequence[int],
    tokenizer,
    token_end_times_s: Sequence[float | None],
    audio_duration_s: float,
) -> list[WordAlignment]:
    """Group per-token end-times into word-level timestamps.

    Uses tokenizer's ``decode`` to recover each token's surface characters
    and runs them against the final transcript's word spans. Each word's
    end-time is the max end-time of any token whose characters overlap
    the word span; start-time is the min. Tokens with no alignment
    (``None``) are ignored for that word. If a word has no aligned token,
    the previous word's end-time is used as a monotone fallback.
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
        # Fall back to word-by-word alignment on the decoded string,
        # then re-anchor to the normalized text via longest-common-prefix.
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


def _map_chars_to_words(
    text: str, word_spans: Sequence[tuple[int, int, str]]
) -> list[int | None]:
    mapping: list[int | None] = [None] * len(text)
    for word_idx, (start, end, _) in enumerate(word_spans):
        for char_idx in range(start, min(end, len(text))):
            mapping[char_idx] = word_idx
    return mapping


def load_audio_alignment_heads(
    path: str, *, top_k: int
) -> tuple[list[AlignAttHead], float]:
    """Load calibrated alignment heads from a JSON file.

    The file has the same shape as the MT head files
    (``token_alignment_heads`` array with ``layer``, ``head``, ``ts``) plus
    an optional ``word_end_offset_seconds`` scalar that is subtracted from
    every predicted word-end time at inference. That offset corrects the
    systematic lag between a causal LLM's attention peak and the acoustic
    word boundary — a single generic constant, not a per-example hack.
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


def save_audio_alignment_heads(
    path: str,
    *,
    scored_heads: Sequence[dict],
    model_name: str,
    language: str,
    scoring_notes: dict[str, object] | None = None,
    word_end_offset_seconds: float = 0.0,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_name,
        "language": language,
        "score_name": "audio_alignment_monotonicity_mae",
        "word_end_offset_seconds": float(word_end_offset_seconds),
        "scoring_notes": scoring_notes or {},
        "token_alignment_heads": list(scored_heads),
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def monotonicity_score(audio_positions: Sequence[int | None]) -> float:
    """Fraction of consecutive token pairs with non-decreasing audio index.

    This is the source-axis analogue of AlignAtt's quality signal: an
    alignment head useful for streaming must move forward in time. We
    ignore ``None`` positions. Returns ``0`` when fewer than two valid
    positions exist.
    """
    filtered = [int(pos) for pos in audio_positions if pos is not None]
    if len(filtered) < 2:
        return 0.0
    pairs = zip(filtered[:-1], filtered[1:])
    non_decreasing = sum(1 for prev, nxt in pairs if nxt >= prev)
    return non_decreasing / float(len(filtered) - 1)


@dataclass
class _GeneratedCapture:
    generated_ids: list[int]
    per_token_layer_attentions: list[dict[int, torch.Tensor]]


class GemmaAttentionAlignmentBackend(AlignmentBackend):
    name = "gemma_onepass_qk_fast"

    def __init__(
        self,
        *,
        model_name: str,
        runtime_config: SimpleNamespace,
        audio_heads_path: str | None = None,
        audio_heads_top_k: int = 8,
        filter_width: int = 7,
        max_new_tokens: int = 256,
        audio_token_id: int = GEMMA_AUDIO_TOKEN_ID_DEFAULT,
        audio_ms_per_token: float = GEMMA_AUDIO_MS_PER_TOKEN_DEFAULT,
        max_audio_seconds: float = GEMMA_AUDIO_MAX_SECONDS_DEFAULT,
        audio_align_probe_mode: str | None = None,
    ):
        self.model_name = model_name
        self.runtime_config = runtime_config
        self.audio_heads_path = audio_heads_path
        self.audio_heads_top_k = int(audio_heads_top_k)
        self.filter_width = int(filter_width)
        self.max_new_tokens = int(max_new_tokens)
        self.audio_token_id = int(audio_token_id)
        self.audio_ms_per_token = float(audio_ms_per_token)
        self.max_audio_seconds = float(max_audio_seconds)
        self.device = str(getattr(runtime_config, "gemma_transformers_device", "cuda:0"))
        self.dtype = getattr(torch, str(getattr(runtime_config, "gemma_transformers_dtype", "bfloat16")))
        self.fast_attention_implementation = str(
            getattr(runtime_config, "gemma_transformers_fast_attention", "sdpa")
        )
        self.audio_align_probe_mode = str(
            audio_align_probe_mode
            or getattr(runtime_config, "gemma_audio_align_probe_mode", "qk_fast")
        )

        self.model = None
        self.processor = None
        self.tokenizer = None
        self.alignatt_heads: list[AlignAttHead] = []
        self.alignatt_recorder: SelectedAttentionRecorder | None = None
        self.alignatt_layer_input_recorder: SelectedLayerInputRecorder | None = None
        self.qk_fast_probe_supported: bool | None = None
        # Subtracted from every predicted word-end time. Calibrated once per
        # (language, model) alongside the heads file. Defaults to 0 so
        # uncalibrated runs still work.
        self.word_end_offset_s: float = 0.0

    def load(self) -> None:
        from transformers import (
            AutoModelForMultimodalLM,
            AutoProcessor,
            modeling_utils,
        )

        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                local_files_only=True,
            )
            self.tokenizer = self.processor.tokenizer
            ms_per_token = getattr(self.processor, "audio_ms_per_token", None)
            if ms_per_token is not None:
                self.audio_ms_per_token = float(ms_per_token)
            audio_seq_length = getattr(self.processor, "audio_seq_length", None)
            if audio_seq_length is not None and ms_per_token is not None:
                self.max_audio_seconds = float(audio_seq_length) * float(ms_per_token) / 1000.0

        if self.model is None:
            original_warmup = None
            if hasattr(modeling_utils, "caching_allocator_warmup"):
                original_warmup = modeling_utils.caching_allocator_warmup
                modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
            try:
                self.model = AutoModelForMultimodalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.dtype,
                    device_map=self.device,
                    trust_remote_code=True,
                    local_files_only=True,
                    attn_implementation=self.fast_attention_implementation,
                    low_cpu_mem_usage=True,
                )
            finally:
                if original_warmup is not None:
                    modeling_utils.caching_allocator_warmup = original_warmup
            self.model.eval()
            audio_token_id = getattr(self.model.config, "audio_token_id", None)
            if audio_token_id is not None:
                self.audio_token_id = int(audio_token_id)

        if self.alignatt_heads == [] and self.audio_heads_path:
            if Path(self.audio_heads_path).exists():
                self.alignatt_heads, self.word_end_offset_s = load_audio_alignment_heads(
                    self.audio_heads_path,
                    top_k=self.audio_heads_top_k,
                )
        self._ensure_probe_recorders()

    def _ensure_probe_recorders(self) -> None:
        if self.model is None or not self.alignatt_heads:
            return
        if self.alignatt_recorder is None:
            self.alignatt_recorder = SelectedAttentionRecorder(
                model=self.model,
                alignatt_heads=self.alignatt_heads,
            )
        if self.alignatt_layer_input_recorder is None and self.alignatt_heads:
            self.alignatt_layer_input_recorder = SelectedLayerInputRecorder(
                model=self.model,
                alignatt_heads=self.alignatt_heads,
            )

    def _enforce_audio_cap(self, audio: np.ndarray, *, sample_rate: int) -> float:
        """Raise if ``audio`` exceeds the Gemma encoder's audio cap.

        The processor silently truncates anything past ``audio_seq_length``
        tokens, so any timestamp produced for the dropped tail would be
        invalid. Failing loudly here forces callers to chunk explicitly
        rather than silently degrade their metrics.
        """
        duration_s = float(len(audio)) / float(sample_rate)
        if duration_s > self.max_audio_seconds + 1e-3:
            raise GemmaAudioTooLongError(
                f"Audio is {duration_s:.3f}s but Gemma encoder cap is "
                f"{self.max_audio_seconds:.3f}s. Chunk the input or raise "
                "the cap explicitly via max_audio_seconds."
            )
        return duration_s

    # -------------------- core prompt / inference machinery ---------------

    def _render_asr_messages(
        self, audio: np.ndarray, *, language: str
    ) -> list[dict]:
        # Audio block comes *before* the text block — this matches the
        # Gemma cookbook exactly. Swapping the order changes where the
        # audio placeholders land in the rendered chat template and (per
        # quick ablation on this clip) is the difference between clean
        # transcription and hallucinated content.
        del language  # cookbook uses "original language" directly
        return [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": np.asarray(audio, dtype=np.float32)},
                    {
                        "type": "text",
                        "text": (
                            "Transcribe the following speech segment in its original language. "
                            "Follow these specific instructions for formatting the answer:\n"
                            "* Only output the transcription, with no newlines.\n"
                            "* When transcribing numbers, write the digits, i.e. write 1.7 "
                            "and not one point seven, and write 3 instead of three."
                        ),
                    },
                ],
            }
        ]

    @contextmanager
    def _temporary_attention_implementation(self, attn_implementation: str):
        if self.model is None:
            raise RuntimeError("Gemma alignment backend is not loaded. Call load() first.")

        configs: list[object] = []
        candidates = (
            self.model,
            getattr(self.model, "model", None),
            getattr(getattr(self.model, "model", None), "language_model", None),
        )
        seen_ids: set[int] = set()
        for candidate in candidates:
            config = getattr(candidate, "config", None)
            if config is None or id(config) in seen_ids:
                continue
            seen_ids.add(id(config))
            configs.append(config)
        original_impls = [getattr(c, "_attn_implementation", None) for c in configs]
        for config in configs:
            if hasattr(config, "_attn_implementation"):
                config._attn_implementation = attn_implementation
        try:
            yield
        finally:
            for config, original in zip(configs, original_impls):
                if original is not None and hasattr(config, "_attn_implementation"):
                    config._attn_implementation = original

    @contextmanager
    def _default_attention_implementation(self):
        """Force default (non-eager) attention for free-run ASR quality.

        The ablation in run_gemma_asr_fairness.py showed that eager attention
        destroys free-run ASR (WER 0.81+) while default attention produces
        correct transcripts (WER 0.03-0.26). This context manager ensures
        default attention regardless of how the model was loaded.
        """
        with self._temporary_attention_implementation("sdpa"):
            yield

    @contextmanager
    def _eager_attention_implementation(self):
        """Force eager attention so attention weights are materialized.

        ``SelectedAttentionRecorder`` reads the hook output's second tuple
        element, which only exists when the model runs with eager attention.
        """
        with self._temporary_attention_implementation("eager"):
            yield

    def _prepare_inputs(self, audio: np.ndarray, *, language: str) -> tuple[dict, list[int]]:
        messages = self._render_asr_messages(audio, language=language)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        # Cookbook-style: a single ``.to(device)`` only. The BatchFeature
        # keeps floats in their native dtype; letting the model auto-cast
        # internally matches the reference setup and avoids a bf16
        # quantization pass over the mel features before the audio tower.
        inputs = inputs.to(self.model.device)
        input_ids = inputs["input_ids"][0].tolist()
        return dict(inputs), input_ids

    def _build_all_layer_recorder(self) -> SelectedAttentionRecorder:
        """Hook every text-model layer so we can score the full (layer, head) grid.

        ``SelectedAttentionRecorder`` captures the whole attention tensor
        per hooked layer (all heads), filtering by head happens later. So a
        recorder constructed with one synthetic ``AlignAttHead`` per layer
        is exactly the "all layers" capture calibration needs. This works
        uniformly because Gemma4 in transformers 5.x no longer populates
        ``outputs.attentions`` — hooks are the supported extraction path.
        """
        if self.model is None:
            raise RuntimeError("Gemma alignment backend is not loaded.")
        text_config = getattr(self.model.config, "text_config", None)
        num_layers = int(getattr(text_config, "num_hidden_layers", 0))
        if num_layers <= 0:
            raise RuntimeError(
                "Could not resolve num_hidden_layers on Gemma4 text config."
            )
        synthetic_heads = [
            AlignAttHead(layer=layer_idx, head=0, ts=0.0)
            for layer_idx in range(num_layers)
        ]
        return SelectedAttentionRecorder(
            model=self.model,
            alignatt_heads=synthetic_heads,
        )

    def _generate_with_attention(
        self,
        inputs: dict,
        audio_span: AudioSpan,
        *,
        record_all_heads: bool = False,
        recorder_override: SelectedAttentionRecorder | None = None,
    ) -> _GeneratedCapture:
        """Autoregressively decode, capturing per-step attention via hooks.

        ``record_all_heads=True`` hooks every text-model layer so head-
        selection calibration can pull any head; otherwise the configured
        runtime recorder (over the selected heads only) is used. Forcing
        eager attention is required so ``attn_weights`` is materialized for
        the hook.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Gemma alignment backend is not loaded.")

        if record_all_heads:
            recorder = self._build_all_layer_recorder()
        else:
            recorder = recorder_override or self.alignatt_recorder

        eos_token_ids = set(self._resolve_stop_token_ids())
        generated_ids: list[int] = []
        per_token_captures: list[dict[int, torch.Tensor]] = []

        model_kwargs = {k: v for k, v in inputs.items()}
        past_key_values = None

        with self._eager_attention_implementation(), torch.no_grad():
            # The first pass processes the audio features and populates the
            # KV cache. Its own attentions belong to the prompt / audio
            # tokens, not to any generated token, so we do not capture them.
            outputs = self.model(
                **model_kwargs,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values

            for step in range(self.max_new_tokens):
                logits = outputs.logits[0, -1, :].float()
                next_token_id = int(logits.argmax().item())
                if next_token_id in eos_token_ids:
                    break
                generated_ids.append(next_token_id)

                step_input_ids = torch.tensor(
                    [[next_token_id]], device=self.model.device
                )
                step_kwargs: dict = {
                    "input_ids": step_input_ids,
                    "past_key_values": past_key_values,
                    "use_cache": True,
                }
                if recorder is not None:
                    with recorder.capture() as captured:
                        outputs = self.model(**step_kwargs)
                    per_token_captures.append(
                        {
                            k: (v.detach().to("cpu") if record_all_heads else v.clone())
                            for k, v in captured.items()
                        }
                    )
                else:
                    outputs = self.model(**step_kwargs)
                past_key_values = outputs.past_key_values

        return _GeneratedCapture(
            generated_ids=generated_ids,
            per_token_layer_attentions=per_token_captures,
        )

    def _resolve_stop_token_ids(self) -> tuple[int, ...]:
        tokenizer = self.tokenizer
        stops: set[int] = set()
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None:
            stops.add(int(eos))
        # Pull multi-eos from config if present (Gemma4 ships with [1, 106]).
        config_eos = getattr(self.model.config, "eos_token_id", None)
        if isinstance(config_eos, (list, tuple)):
            stops.update(int(t) for t in config_eos)
        elif isinstance(config_eos, int):
            stops.add(int(config_eos))
        # Stop when the model emits end-of-turn / other special tokens.
        end_of_turn = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if isinstance(end_of_turn, int) and end_of_turn >= 0:
            stops.add(end_of_turn)
        return tuple(sorted(stops))

    def _decode_transcript_and_audio_rows_qk_fast(
        self,
        *,
        inputs: dict,
        input_ids: Sequence[int],
        audio_span: AudioSpan,
    ) -> tuple[list[int], list[torch.Tensor], dict[str, Any]]:
        """Greedy ASR decode with replay-free qk_fast audio alignment.

        This is the runtime one-pass path: prompt once under fast attention,
        then decode token-by-token while capturing the current token's layer
        inputs and reconstructing its attention row into the audio span from
        the live KV cache. No eager pass and no teacher-forced replay.
        """
        if self.model is None or self.alignatt_layer_input_recorder is None:
            raise RuntimeError(
                "Gemma qk_fast decode needs a loaded model and selected alignment heads."
            )

        generation_stop_token_ids = set(self._resolve_stop_token_ids())
        audio_positions = list(range(audio_span.prompt_start, audio_span.prompt_end))
        prompt_forward_start = perf_counter()

        with self._temporary_attention_implementation(self.fast_attention_implementation), torch.no_grad():
            outputs = self.model(
                **inputs,
                use_cache=True,
                return_dict=True,
            )
        prompt_forward_ms = (perf_counter() - prompt_forward_start) * 1000.0
        prompt_snapshot_start = perf_counter()
        prompt_kv_snapshot = self._snapshot_kv(outputs.past_key_values, len(input_ids))
        prompt_kv_snapshot_ms = (perf_counter() - prompt_snapshot_start) * 1000.0
        past_key_values = outputs.past_key_values

        generated_ids: list[int] = []
        per_token_audio_rows: list[torch.Tensor] = []
        self.qk_fast_probe_supported = None
        decode_loop_start = perf_counter()
        decode_step_ms_total = 0.0
        qk_reconstruction_ms_total = 0.0
        qk_reconstruction_success_count = 0
        qk_reconstruction_empty_count = 0

        for _ in range(self.max_new_tokens):
            logits = outputs.logits[0, -1, :].float()
            next_token_id = int(logits.argmax().item())
            if next_token_id in generation_stop_token_ids:
                break

            generated_ids.append(next_token_id)
            decode_step_start = perf_counter()
            outputs, captured_layer_inputs = self._run_suffix_forward(
                token_ids=[next_token_id],
                past_key_values=past_key_values,
                attention_implementation=self.fast_attention_implementation,
                capture_recorder=self.alignatt_layer_input_recorder,
            )
            decode_step_ms_total += (perf_counter() - decode_step_start) * 1000.0
            past_key_values = outputs.past_key_values

            qk_start = perf_counter()
            if captured_layer_inputs:
                self.qk_fast_probe_supported = True
                qk_rows, _ = extract_source_attention_rows_per_token_from_fast_path(
                    layer_inputs_by_layer=captured_layer_inputs,
                    prompt_kv_snapshot=prompt_kv_snapshot,
                    runtime_past_key_values=past_key_values,
                    alignatt_heads=self.alignatt_heads,
                    source_positions=audio_positions,
                )
            else:
                self.qk_fast_probe_supported = False
                qk_rows = []
            qk_reconstruction_ms_total += (perf_counter() - qk_start) * 1000.0

            if qk_rows:
                qk_reconstruction_success_count += 1
                per_token_audio_rows.append(qk_rows[-1])
            else:
                qk_reconstruction_empty_count += 1
                per_token_audio_rows.append(
                    torch.zeros(
                        len(self.alignatt_heads),
                        len(audio_positions),
                        dtype=torch.float32,
                    )
                )

        decode_loop_ms = (perf_counter() - decode_loop_start) * 1000.0
        generated_token_count = len(generated_ids)
        timing_components_ms = {
            "prompt_forward": prompt_forward_ms,
            "prompt_kv_snapshot": prompt_kv_snapshot_ms,
            "decode_loop_total": decode_loop_ms,
            "decode_step_total": decode_step_ms_total,
            "decode_step_avg": (
                decode_step_ms_total / float(generated_token_count)
                if generated_token_count > 0
                else 0.0
            ),
            "qk_reconstruction_total": qk_reconstruction_ms_total,
            "qk_reconstruction_avg": (
                qk_reconstruction_ms_total / float(generated_token_count)
                if generated_token_count > 0
                else 0.0
            ),
        }
        diagnostics = {
            "selected_head_count": len(self.alignatt_heads),
            "audio_span_length": len(audio_positions),
            "generated_token_count": generated_token_count,
            "qk_fast_layer_input_capture_supported": bool(self.qk_fast_probe_supported),
            "qk_reconstruction_successful_token_count": qk_reconstruction_success_count,
            "qk_reconstruction_empty_token_count": qk_reconstruction_empty_count,
            "timings_ms": {
                key: round(float(value), 3)
                for key, value in timing_components_ms.items()
            },
        }
        return generated_ids, per_token_audio_rows, diagnostics

    def _build_compact_observer_tokens(
        self,
        *,
        token_ids: Sequence[int],
        aligned_source_positions: Sequence[int | None],
        provenance: Sequence[AlignAttProvenanceBreakdown | None] | None = None,
        blocked_source_local_position: int | None = None,
        blocked_source_unit_index: int | None = None,
    ) -> tuple[AlignAttObserverToken, ...]:
        if self.tokenizer is None:
            raise RuntimeError("Gemma alignment backend is not loaded.")
        if len(token_ids) != len(aligned_source_positions):
            raise ValueError("token_ids and aligned_source_positions length mismatch")
        if provenance is not None and len(provenance) != len(token_ids):
            raise ValueError("provenance and token_ids length mismatch")

        observer_tokens: list[AlignAttObserverToken] = []
        for token_index, (token_id, aligned_position) in enumerate(
            zip(token_ids, aligned_source_positions)
        ):
            provenance_entry = None if provenance is None else provenance[token_index]
            observer_tokens.append(
                AlignAttObserverToken(
                    token_id=int(token_id),
                    token_str=self.tokenizer.decode(
                        [int(token_id)], skip_special_tokens=False
                    ),
                    aligned_source_position=(
                        None if aligned_position is None else int(aligned_position)
                    ),
                    source_accessible_mass=(
                        None
                        if provenance_entry is None
                        else float(provenance_entry.source_accessible)
                    ),
                    blocked_source_local_position=blocked_source_local_position,
                    blocked_source_unit_index=blocked_source_unit_index,
                    provenance=provenance_entry,
                )
            )
        return tuple(observer_tokens)

    # -------------------- default-attention ASR ----------------------------

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        language: str,
    ) -> str | None:
        """Free-run ASR with default (non-eager) attention.

        Uses model.generate() directly rather than step-by-step decoding
        with attention hooks. This produces correct transcripts (WER 0.03-0.26)
        whereas eager attention hallucinates (WER 0.81+).
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Gemma alignment backend is not loaded. Call load() first.")

        audio = np.asarray(audio, dtype=np.float32)
        self._enforce_audio_cap(audio, sample_rate=sample_rate)

        messages = self._render_asr_messages(audio, language=language)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        with self._default_attention_implementation(), torch.no_grad():
            outputs = self.model.generate(
                **dict(inputs),
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        text = self.processor.decode(
            outputs[0][input_len:], skip_special_tokens=True
        ).strip()
        return text if text else None

    # -------------------- forced alignment --------------------------------

    def align_transcript(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        language: str,
        transcript: str,
    ) -> AlignmentResult | None:
        """Teacher-forced alignment: ``audio + known transcript`` -> word timestamps.

        This is the direct replacement for Qwen3-ForcedAligner-0.6B: given
        the audio and the text, we prefill the transcript in the assistant
        turn and extract per-transcript-token alignment rows into the audio
        span. Depending on ``gemma_audio_align_probe_mode`` that extraction
        either reads materialized attentions directly (``eager``) or
        reconstructs the rows from layer inputs plus KV snapshots
        (``qk_fast``). This decouples the novel part of this research
        (attention-based alignment) from the quality of Gemma's own ASR
        head, and is the right experiment to run first when the model's
        free-running transcription is noisy.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Gemma alignment backend is not loaded.")
        total_start = perf_counter()
        transcript = transcript.strip()
        if not transcript:
            return None

        audio = np.asarray(audio, dtype=np.float32)
        audio_duration_s = self._enforce_audio_cap(audio, sample_rate=sample_rate)

        inputs, input_ids, transcript_span = self._prepare_forced_alignment_inputs(
            audio, language=language, transcript=transcript
        )
        audio_span = detect_audio_span(
            input_ids,
            audio_token_id=self.audio_token_id,
            audio_ms_per_token=self.audio_ms_per_token,
        )
        if audio_span is None or audio_span.length <= 0:
            return None
        if transcript_span is None or transcript_span[1] <= transcript_span[0]:
            return None

        transcript_token_ids = input_ids[transcript_span[0] : transcript_span[1]]

        if not self.alignatt_heads:
            return AlignmentResult(
                text=transcript,
                words=(),
                audio_duration_s=audio_duration_s,
                observer_tokens=(),
                diagnostics={
                    "backend": self.name,
                    "mode": "forced_alignment",
                    "audio_span_length": audio_span.length,
                    "reason": "no_audio_alignment_heads_calibrated",
                },
            )

        probe_result = self._extract_transcript_audio_rows(
            inputs=inputs,
            input_ids=input_ids,
            transcript_span=transcript_span,
            audio_span=audio_span,
        )
        postprocess_start = perf_counter()
        token_audio_positions = self._aggregate_audio_positions_from_rows(
            probe_result.source_attention_rows_per_token
        )

        token_end_times_s = [
            audio_position_to_end_seconds(
                pos,
                ms_per_token=audio_span.ms_per_token,
                audio_duration_s=audio_duration_s,
            )
            for pos in token_audio_positions
        ]
        token_end_times_s = _enforce_monotone(token_end_times_s)
        token_end_times_s = _apply_word_end_offset(
            token_end_times_s,
            offset_s=self.word_end_offset_s,
            audio_duration_s=audio_duration_s,
        )

        words = aggregate_token_timings_to_words(
            transcript,
            generated_ids=transcript_token_ids,
            tokenizer=self.tokenizer,
            token_end_times_s=token_end_times_s,
            audio_duration_s=audio_duration_s,
        )
        postprocess_ms = (perf_counter() - postprocess_start) * 1000.0
        total_alignment_ms = (perf_counter() - total_start) * 1000.0
        observer_tokens = self._build_compact_observer_tokens(
            token_ids=transcript_token_ids,
            aligned_source_positions=token_audio_positions,
        )
        timings_ms = dict(probe_result.diagnostics.get("timings_ms", {}))
        timings_ms["timing_aggregation"] = round(float(postprocess_ms), 3)
        timings_ms["total_alignment"] = round(float(total_alignment_ms), 3)
        return AlignmentResult(
            text=transcript,
            words=tuple(words),
            audio_duration_s=audio_duration_s,
            observer_tokens=observer_tokens,
            diagnostics={
                "backend": self.name,
                "mode": "forced_alignment",
                "probe_backend": probe_result.probe_backend,
                "audio_span_length": audio_span.length,
                "audio_ms_per_token": audio_span.ms_per_token,
                "monotonicity": monotonicity_score(token_audio_positions),
                "aligned_audio_positions": token_audio_positions,
                "transcript_token_count": len(transcript_token_ids),
                "observer_token_count": len(observer_tokens),
                "word_end_offset_s": self.word_end_offset_s,
                **{
                    k: v for k, v in probe_result.diagnostics.items() if k != "timings_ms"
                },
                "timings_ms": timings_ms,
            },
        )

    def _prepare_forced_alignment_inputs(
        self,
        audio: np.ndarray,
        *,
        language: str,
        transcript: str,
    ) -> tuple[dict, list[int], tuple[int, int] | None]:
        """Build the ``[prompt, audio, assistant=transcript]`` input.

        Uses ``continue_final_message=True`` so the assistant turn is a
        prefill of the known transcript rather than an empty generation
        prompt; attention from those assistant tokens into the audio span
        is exactly what we want to extract.

        **Ordering note.** The forced-alignment path intentionally uses
        **text-before-audio** ordering, *not* the cookbook's audio-first
        ordering used by free-run ASR. Rationale — measured on smoke18
        with a Qwen teacher:

        - text-first calibration: MAE 177 ms, monotonicity 0.98
        - audio-first calibration: MAE 502 ms, monotonicity 0.76

        Swapping the order moves the audio placeholders inside the chat
        template and makes the assistant-token attention into the audio
        span noticeably less peaked. Since forced alignment doesn't use
        the cookbook's ASR output at all (the transcript is prefilled),
        the cookbook's layout constraint does not apply here. We keep
        the cookbook ordering in :meth:`_render_asr_messages` for
        free-run ASR and use the alignment-friendly ordering here.
        """
        del language  # prompt wording is fixed per (language, model) pair
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Transcribe the following speech segment in its original language. "
                            "Follow these specific instructions for formatting the answer:\n"
                            "* Only output the transcription, with no newlines.\n"
                            "* When transcribing numbers, write the digits, i.e. write 1.7 "
                            "and not one point seven, and write 3 instead of three."
                        ),
                    },
                    {"type": "audio", "audio": np.asarray(audio, dtype=np.float32)},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": transcript},
                ],
            },
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            continue_final_message=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        input_ids = inputs["input_ids"][0].tolist()
        transcript_span = self._locate_transcript_span(input_ids, transcript)
        return dict(inputs), input_ids, transcript_span

    def _locate_transcript_span(
        self, input_ids: list[int], transcript: str
    ) -> tuple[int, int] | None:
        """Find the contiguous input_ids run that decodes back to the transcript.

        We re-tokenize the transcript with ``add_special_tokens=False`` and
        search for that subsequence in the rendered input_ids, starting
        from the end (to land inside the assistant turn). This avoids
        having to re-parse the chat template and stays robust to any
        trailing special tokens the template may append.
        """
        transcript_token_ids = self.tokenizer(
            transcript, add_special_tokens=False
        )["input_ids"]
        if not transcript_token_ids:
            return None
        target = list(transcript_token_ids)
        n = len(target)
        for start in range(len(input_ids) - n, -1, -1):
            if input_ids[start : start + n] == target:
                return start, start + n
        return None

    def _run_forward_capture_transcript_attention(
        self,
        *,
        inputs: dict,
        transcript_span: tuple[int, int],
    ) -> list[dict[int, torch.Tensor]]:
        """Run one forward pass, capture attention at transcript positions only."""
        recorder = self._build_all_layer_recorder()
        with self._eager_attention_implementation(), torch.no_grad():
            with recorder.capture() as captured:
                _ = self.model(
                    **inputs,
                    use_cache=False,
                    return_dict=True,
                )
            per_layer = {k: v.clone() for k, v in captured.items()}

        # Each layer tensor has shape (batch, heads, seq_len, seq_len). We
        # slice out only the transcript rows (one per transcript token) and
        # CPU-offload to keep GPU memory flat.
        start, end = transcript_span
        per_token_captures: list[dict[int, torch.Tensor]] = []
        num_transcript_tokens = end - start
        for token_offset in range(num_transcript_tokens):
            row = {}
            query_index = start + token_offset
            for layer_idx, tensor in per_layer.items():
                # Keep only the row for this query; trim the key-length to
                # avoid shipping the whole sequence if it's large.
                row[layer_idx] = tensor[:, :, query_index : query_index + 1, :].detach().cpu()
            per_token_captures.append(row)
        return per_token_captures

    @staticmethod
    def _snapshot_kv(past_kv, length: int):
        if past_kv is None:
            return None
        if hasattr(past_kv, "layers"):
            snapshot = []
            for layer_idx, layer in enumerate(past_kv.layers):
                keys = getattr(layer, "keys", None)
                values = getattr(layer, "values", None)
                if keys is None or values is None or getattr(keys, "numel", lambda: 0)() == 0:
                    continue
                seq_length = int(layer.get_seq_length()) if hasattr(layer, "get_seq_length") else int(length)
                snapshot.append(
                    (
                        layer_idx,
                        keys[:, :, :length, :].detach().clone(),
                        values[:, :, :length, :].detach().clone(),
                        seq_length,
                    )
                )
            return snapshot
        if hasattr(past_kv, "key_cache"):
            return [
                (
                    layer_idx,
                    key[:, :, :length, :].detach().clone(),
                    value[:, :, :length, :].detach().clone(),
                    int(length),
                )
                for layer_idx, (key, value) in enumerate(
                    zip(past_kv.key_cache, past_kv.value_cache)
                )
            ]
        if isinstance(past_kv, (list, tuple)):
            return [
                (
                    layer_idx,
                    key[:, :, :length, :].detach().clone(),
                    value[:, :, :length, :].detach().clone(),
                    int(length),
                )
                for layer_idx, (key, value) in enumerate(past_kv)
            ]
        return None

    def _restore_kv(self, snapshot, length: int):
        if self.model is None:
            raise RuntimeError("Gemma alignment backend is not loaded. Call load() first.")
        past_kv = DynamicCache(config=self.model.config)
        for layer_idx, key, value, _seq_length in snapshot:
            past_kv.update(
                key[:, :, :length, :],
                value[:, :, :length, :],
                layer_idx=layer_idx,
            )
            layer = past_kv.layers[layer_idx]
            if hasattr(layer, "cumulative_length"):
                layer.cumulative_length = int(length)
        return past_kv

    @staticmethod
    def _slice_inputs_to_prefix(inputs: dict, prefix_length: int) -> dict:
        input_ids = inputs.get("input_ids")
        if input_ids is None or not torch.is_tensor(input_ids):
            raise RuntimeError("Forced-alignment replay requires tensor input_ids.")
        full_length = int(input_ids.shape[-1])
        sliced: dict[str, Any] = {}
        for key, value in inputs.items():
            if not torch.is_tensor(value):
                sliced[key] = value
                continue
            if value.ndim >= 1 and int(value.shape[-1]) == full_length:
                sliced[key] = value[..., :prefix_length].contiguous()
            else:
                sliced[key] = value
        return sliced

    def _run_suffix_forward(
        self,
        *,
        token_ids: Sequence[int],
        past_key_values,
        attention_implementation: str,
        capture_recorder=None,
    ):
        if self.model is None:
            raise RuntimeError("Gemma alignment backend is not loaded. Call load() first.")
        step_kwargs: dict[str, Any] = {
            "input_ids": torch.tensor([list(token_ids)], device=self.model.device),
            "past_key_values": past_key_values,
            "use_cache": True,
            "return_dict": True,
        }
        with self._temporary_attention_implementation(attention_implementation):
            if capture_recorder is not None:
                with capture_recorder.capture() as captured:
                    with torch.no_grad():
                        outputs = self.model(**step_kwargs)
                return outputs, captured
            with torch.no_grad():
                outputs = self.model(**step_kwargs)
            return outputs, None

    def _extract_audio_rows_from_eager_captures(
        self,
        *,
        per_token_captures: Sequence[dict[int, torch.Tensor]],
        audio_span: AudioSpan,
    ) -> list[torch.Tensor]:
        audio_positions = list(range(audio_span.prompt_start, audio_span.prompt_end))
        per_token_rows: list[torch.Tensor] = []
        for captured in per_token_captures:
            rows_per_token = extract_source_attention_rows_per_token(
                layer_attentions_by_layer=captured,
                alignatt_heads=self.alignatt_heads,
                source_positions=audio_positions,
            )
            if not rows_per_token:
                per_token_rows.append(
                    torch.zeros(len(self.alignatt_heads), len(audio_positions))
                )
                continue
            per_token_rows.append(rows_per_token[-1])
        return per_token_rows

    def _aggregate_audio_positions_from_rows(
        self,
        source_attention_rows_per_token: Sequence[torch.Tensor],
    ) -> list[int | None]:
        argmaxes = compute_alignatt_source_argmaxes(
            source_attention_rows_per_token,
            filter_width=self.filter_width,
        )
        return [int(pos) if pos is not None else None for pos in argmaxes]

    def _extract_transcript_audio_rows_eager(
        self,
        *,
        inputs: dict,
        transcript_span: tuple[int, int],
        audio_span: AudioSpan,
    ) -> AudioAttentionProbeResult:
        capture_start = perf_counter()
        per_token_captures = self._run_forward_capture_transcript_attention(
            inputs=inputs,
            transcript_span=transcript_span,
        )
        capture_ms = (perf_counter() - capture_start) * 1000.0
        row_extract_start = perf_counter()
        rows = tuple(
            self._extract_audio_rows_from_eager_captures(
                per_token_captures=per_token_captures,
                audio_span=audio_span,
            )
        )
        row_extract_ms = (perf_counter() - row_extract_start) * 1000.0
        return AudioAttentionProbeResult(
            source_attention_rows_per_token=rows,
            probe_backend="eager",
            diagnostics={
                "alignment_attention": "eager",
                "fast_attention_implementation": None,
                "qk_fast_reconstruction_succeeded": False,
                "selected_head_count": len(self.alignatt_heads),
                "audio_span_length": audio_span.length,
                "timings_ms": {
                    "attention_capture": round(float(capture_ms), 3),
                    "row_extraction": round(float(row_extract_ms), 3),
                },
            },
        )

    def _extract_transcript_audio_rows_qk_fast(
        self,
        *,
        inputs: dict,
        input_ids: Sequence[int],
        transcript_span: tuple[int, int],
        audio_span: AudioSpan,
    ) -> AudioAttentionProbeResult:
        self.qk_fast_probe_supported = False
        self._ensure_probe_recorders()
        if self.alignatt_layer_input_recorder is None:
            raise GemmaAudioQKFastError(
                "Audio qk_fast probe requires calibrated alignment heads and an initialized "
                "layer-input recorder."
            )

        transcript_start, transcript_end = transcript_span
        transcript_token_ids = list(input_ids[transcript_start:transcript_end])
        if not transcript_token_ids:
            raise GemmaAudioQKFastError("Transcript span is empty; nothing to replay.")
        if transcript_start <= 0:
            raise GemmaAudioQKFastError(
                "Transcript replay prefix is empty; expected a multimodal prompt prefix "
                "before the assistant transcript span."
            )
        if audio_span.prompt_end > transcript_start:
            raise GemmaAudioQKFastError(
                "Audio span extends into the transcript replay suffix; qk_fast assumes the "
                "audio placeholders live entirely in the prompt prefix."
            )

        prefix_inputs = self._slice_inputs_to_prefix(inputs, transcript_start)
        prefix_forward_start = perf_counter()
        with self._temporary_attention_implementation(self.fast_attention_implementation), torch.no_grad():
            prefix_outputs = self.model(
                **prefix_inputs,
                use_cache=True,
                return_dict=True,
            )
        prefix_forward_ms = (perf_counter() - prefix_forward_start) * 1000.0
        prompt_snapshot_start = perf_counter()
        prompt_kv_snapshot = self._snapshot_kv(prefix_outputs.past_key_values, transcript_start)
        prompt_kv_snapshot_ms = (perf_counter() - prompt_snapshot_start) * 1000.0
        if prompt_kv_snapshot is None:
            self.qk_fast_probe_supported = False
            raise GemmaAudioQKFastError(
                "Audio qk_fast probe could not snapshot prompt KV states from the prefix run."
            )

        prompt_past_key_values = self._restore_kv(prompt_kv_snapshot, transcript_start)
        suffix_forward_start = perf_counter()
        outputs, captured_layer_inputs = self._run_suffix_forward(
            token_ids=transcript_token_ids,
            past_key_values=prompt_past_key_values,
            attention_implementation=self.fast_attention_implementation,
            capture_recorder=self.alignatt_layer_input_recorder,
        )
        suffix_forward_ms = (perf_counter() - suffix_forward_start) * 1000.0
        self.qk_fast_probe_supported = bool(captured_layer_inputs)
        if not captured_layer_inputs:
            raise GemmaAudioQKFastError(
                "Audio qk_fast probe did not capture any layer inputs under "
                f"{self.fast_attention_implementation!r}."
            )

        audio_positions = list(range(audio_span.prompt_start, audio_span.prompt_end))
        qk_start = perf_counter()
        rows, _provenance = extract_source_attention_rows_per_token_from_fast_path(
            layer_inputs_by_layer=captured_layer_inputs,
            prompt_kv_snapshot=prompt_kv_snapshot,
            runtime_past_key_values=outputs.past_key_values,
            alignatt_heads=self.alignatt_heads,
            source_positions=audio_positions,
        )
        qk_reconstruction_ms = (perf_counter() - qk_start) * 1000.0
        if not rows:
            raise GemmaAudioQKFastError(
                "Audio qk_fast probe captured layer inputs but could not reconstruct any "
                "audio attention rows."
            )
        return AudioAttentionProbeResult(
            source_attention_rows_per_token=tuple(rows),
            probe_backend="qk_fast",
            diagnostics={
                "alignment_attention": "qk_fast",
                "fast_attention_implementation": self.fast_attention_implementation,
                "qk_fast_reconstruction_succeeded": True,
                "qk_fast_prompt_token_count": int(transcript_start),
                "qk_fast_transcript_token_count": int(len(transcript_token_ids)),
                "selected_head_count": len(self.alignatt_heads),
                "audio_span_length": len(audio_positions),
                "timings_ms": {
                    "prefix_forward": round(float(prefix_forward_ms), 3),
                    "prompt_kv_snapshot": round(float(prompt_kv_snapshot_ms), 3),
                    "suffix_forward": round(float(suffix_forward_ms), 3),
                    "qk_reconstruction_total": round(float(qk_reconstruction_ms), 3),
                },
            },
        )

    def _extract_transcript_audio_rows(
        self,
        *,
        inputs: dict,
        input_ids: Sequence[int],
        transcript_span: tuple[int, int],
        audio_span: AudioSpan,
    ) -> AudioAttentionProbeResult:
        probe_mode = str(self.audio_align_probe_mode)
        if probe_mode == "eager":
            return self._extract_transcript_audio_rows_eager(
                inputs=inputs,
                transcript_span=transcript_span,
                audio_span=audio_span,
            )
        if probe_mode == "qk_fast":
            return self._extract_transcript_audio_rows_qk_fast(
                inputs=inputs,
                input_ids=input_ids,
                transcript_span=transcript_span,
                audio_span=audio_span,
            )
        raise ValueError(f"Unknown gemma_audio_align_probe_mode: {probe_mode!r}")

    # -------------------- alignment backend API --------------------------

    def transcribe_and_align(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        language: str,
        streaming_prefix_text: str = "",
        streaming_prefix_words: tuple[WordAlignment, ...] = (),
    ) -> AlignmentResult | None:
        if self.model is None:
            raise RuntimeError("Gemma alignment backend is not loaded. Call load() first.")
        if streaming_prefix_text or streaming_prefix_words:
            raise NotImplementedError(
                "Transformers-based gemma_onepass_qk_fast does not yet support "
                "prompt-prefix streaming. Use gemma_vllm_qk_fast with "
                "asr_streaming_prefix_enabled=True for that path."
            )
        total_start = perf_counter()

        audio = np.asarray(audio, dtype=np.float32)
        audio_duration_s = self._enforce_audio_cap(audio, sample_rate=sample_rate)

        inputs, input_ids = self._prepare_inputs(audio, language=language)
        audio_span = detect_audio_span(
            input_ids,
            audio_token_id=self.audio_token_id,
            audio_ms_per_token=self.audio_ms_per_token,
        )
        if audio_span is None or audio_span.length <= 0:
            return None

        self._ensure_probe_recorders()
        if not self.alignatt_heads or self.alignatt_layer_input_recorder is None:
            text = self.transcribe(audio, sample_rate=sample_rate, language=language) or ""
            return AlignmentResult(
                text=text,
                words=(),
                audio_duration_s=audio_duration_s,
                observer_tokens=(),
                diagnostics={
                    "backend": self.name,
                    "audio_span_length": audio_span.length,
                    "reason": "no_audio_alignment_heads_calibrated",
                },
            )

        generated_ids, per_token_audio_rows, decode_diagnostics = self._decode_transcript_and_audio_rows_qk_fast(
            inputs=inputs,
            input_ids=input_ids,
            audio_span=audio_span,
        )
        postprocess_start = perf_counter()
        text = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()
        token_audio_positions = compute_alignatt_source_argmaxes(
            per_token_audio_rows,
            filter_width=self.filter_width,
        )
        token_end_times_s = [
            audio_position_to_end_seconds(
                pos,
                ms_per_token=audio_span.ms_per_token,
                audio_duration_s=audio_duration_s,
            )
            for pos in token_audio_positions
        ]
        token_end_times_s = _enforce_monotone(token_end_times_s)
        token_end_times_s = _apply_word_end_offset(
            token_end_times_s,
            offset_s=self.word_end_offset_s,
            audio_duration_s=audio_duration_s,
        )

        words = aggregate_token_timings_to_words(
            text,
            generated_ids=generated_ids,
            tokenizer=self.tokenizer,
            token_end_times_s=token_end_times_s,
            audio_duration_s=audio_duration_s,
        )
        postprocess_ms = (perf_counter() - postprocess_start) * 1000.0
        total_backend_ms = (perf_counter() - total_start) * 1000.0
        timings_ms = dict(decode_diagnostics.get("timings_ms", {}))
        timings_ms["timing_aggregation"] = round(float(postprocess_ms), 3)
        timings_ms["total_backend"] = round(float(total_backend_ms), 3)
        observer_tokens = self._build_compact_observer_tokens(
            token_ids=generated_ids,
            aligned_source_positions=token_audio_positions,
        )

        return AlignmentResult(
            text=text,
            words=tuple(words),
            audio_duration_s=audio_duration_s,
            observer_tokens=observer_tokens,
            diagnostics={
                "backend": self.name,
                "audio_span_length": audio_span.length,
                "audio_ms_per_token": audio_span.ms_per_token,
                "monotonicity": monotonicity_score(token_audio_positions),
                "aligned_audio_positions": token_audio_positions,
                "word_end_offset_s": self.word_end_offset_s,
                "probe_backend": "qk_fast_onepass",
                "qk_fast_reconstruction_succeeded": bool(self.qk_fast_probe_supported),
                "observer_token_count": len(observer_tokens),
                **{
                    k: v for k, v in decode_diagnostics.items() if k != "timings_ms"
                },
                "timings_ms": timings_ms,
            },
        )

    def _aggregate_audio_positions(
        self,
        *,
        per_token_captures: Sequence[dict[int, torch.Tensor]],
        audio_span: AudioSpan,
    ) -> list[int | None]:
        if not per_token_captures:
            return []
        audio_positions = list(range(audio_span.prompt_start, audio_span.prompt_end))
        per_token_rows: list[torch.Tensor] = []
        for captured in per_token_captures:
            rows_per_token = extract_source_attention_rows_per_token(
                layer_attentions_by_layer=captured,
                alignatt_heads=self.alignatt_heads,
                source_positions=audio_positions,
            )
            if not rows_per_token:
                per_token_rows.append(
                    torch.zeros(len(self.alignatt_heads), len(audio_positions))
                )
                continue
            per_token_rows.append(rows_per_token[-1])

        argmaxes = compute_alignatt_source_argmaxes(
            per_token_rows, filter_width=self.filter_width
        )
        return [int(pos) if pos is not None else None for pos in argmaxes]

    # -------------------- qk_fast audio alignment -------------------------

    def _run_forward_extract_audio_positions_qk_fast(
        self,
        *,
        inputs: dict,
        audio_span: AudioSpan,
        transcript_span: tuple[int, int],
    ) -> list[int | None]:
        """Forward pass with eager attention, extract audio alignment via Q*K.

        Runs the model with eager attention (same implementation the heads
        were calibrated against), but captures only the selected layers'
        hidden states via ``SelectedLayerInputRecorder`` instead of building
        an all-layer attention recorder.  This avoids retaining ~42 full
        attention matrices and computes Q*K^T only for the selected
        alignment heads at transcript-to-audio positions.
        """
        if self.alignatt_layer_input_recorder is None:
            return []

        with self._eager_attention_implementation(), torch.no_grad():
            with self.alignatt_layer_input_recorder.capture() as captured_layer_inputs:
                _ = self.model(
                    **inputs,
                    use_cache=False,
                    return_dict=True,
                )

        if not captured_layer_inputs:
            return []

        per_token_audio_rows = self._extract_audio_attention_rows_qk_fast(
            captured_layer_inputs=captured_layer_inputs,
            audio_span=audio_span,
            transcript_span=transcript_span,
        )
        if not per_token_audio_rows:
            return []

        argmaxes = compute_alignatt_source_argmaxes(
            per_token_audio_rows, filter_width=self.filter_width
        )
        return [int(pos) if pos is not None else None for pos in argmaxes]

    def _extract_audio_attention_rows_qk_fast(
        self,
        *,
        captured_layer_inputs: dict[int, LayerInputCapture],
        audio_span: AudioSpan,
        transcript_span: tuple[int, int],
    ) -> list[torch.Tensor]:
        """Compute per-transcript-token attention over audio positions via Q*K.

        For each selected alignment head: project hidden states to Q (at
        transcript positions) and K (at all positions), compute the scaled
        dot-product logits, apply a causal + sliding-window mask, softmax
        over the full key dimension, then extract the audio-span columns.

        Returns a list of tensors (one per transcript token), each shaped
        ``(num_selected_heads, audio_span_length)`` — the same contract as
        :meth:`_aggregate_audio_positions` produces from the eager path.
        """
        transcript_start, transcript_end = transcript_span
        num_transcript_tokens = transcript_end - transcript_start
        if num_transcript_tokens <= 0:
            return []

        audio_start = audio_span.prompt_start
        audio_end = audio_span.prompt_end
        if audio_end <= audio_start:
            return []

        first_capture = next(iter(captured_layer_inputs.values()))
        device = first_capture.hidden_states.device

        audio_index_tensor = torch.arange(
            audio_start, audio_end, device=device, dtype=torch.long,
        )

        query_states_by_layer: dict[int, torch.Tensor] = {}
        key_states_by_layer: dict[int, torch.Tensor] = {}
        head_row_matrices: list[torch.Tensor] = []

        for head in self.alignatt_heads:
            layer_idx = int(head.layer)
            capture = captured_layer_inputs.get(layer_idx)
            if capture is None:
                continue

            qs = query_states_by_layer.get(layer_idx)
            if qs is None:
                qs = compute_query_states_from_layer_input_capture(capture)
                if qs is None:
                    continue
                query_states_by_layer[layer_idx] = qs

            ks = key_states_by_layer.get(layer_idx)
            if ks is None:
                ks = compute_key_states_from_layer_input_capture(capture)
                if ks is None:
                    continue
                key_states_by_layer[layer_idx] = ks

            num_attention_heads = int(qs.shape[1])
            num_key_value_heads = int(ks.shape[1])
            head_index = int(head.head)
            if head_index < 0 or head_index >= num_attention_heads:
                continue

            kv_head_index = map_attention_head_to_key_value_head(
                head_index,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
            )

            q_head = qs[0, head_index, transcript_start:transcript_end, :].float()
            k_all = ks[0, kv_head_index, :, :].float()
            full_seq_len = k_all.shape[0]

            full_logits = torch.matmul(q_head, k_all.transpose(0, 1))

            scaling = float(getattr(capture.module, "scaling", 1.0))
            if scaling != 1.0:
                full_logits = full_logits * scaling

            # Causal mask: transcript token i (absolute pos transcript_start+i)
            # can only attend to key positions 0 .. transcript_start+i.
            query_abs = transcript_start + torch.arange(
                num_transcript_tokens, device=device,
            )
            key_pos = torch.arange(full_seq_len, device=device)
            causal_mask = key_pos.unsqueeze(0) > query_abs.unsqueeze(1)

            sliding_window = getattr(capture.module, "sliding_window", None)
            if sliding_window is not None and int(sliding_window) > 0:
                min_key = (query_abs - int(sliding_window) + 1).clamp_min(0)
                causal_mask = causal_mask | (key_pos.unsqueeze(0) < min_key.unsqueeze(1))

            full_logits = full_logits.masked_fill(causal_mask, float("-inf"))
            full_weights = torch.softmax(full_logits, dim=-1)

            audio_weights = full_weights[:, audio_index_tensor]
            head_row_matrices.append(audio_weights)

        if not head_row_matrices:
            return []

        # (num_heads, num_transcript, num_audio) -> per-token list of (num_heads, num_audio)
        stacked = torch.stack(head_row_matrices, dim=0)
        return [stacked[:, i, :] for i in range(num_transcript_tokens)]

    # -------------------- forced-alignment calibration --------------------

    def calibrate_alignment_heads_forced(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        language: str,
        teacher: AlignmentResult,
        top_k: int | None = None,
    ) -> list[dict]:
        """Score (layer, head) using the teacher's transcript as teacher-forced input.

        This is the defensible calibration: we compare predicted word-end
        times against the teacher using the *same* word sequence (no
        alignment of two different transcripts), so MAE is a real
        alignment-quality signal rather than a transcription-quality
        artifact.
        """
        if self.model is None:
            raise RuntimeError("Gemma alignment backend is not loaded.")
        transcript = teacher.text.strip()
        if not transcript:
            return []

        audio = np.asarray(audio, dtype=np.float32)
        audio_duration_s = self._enforce_audio_cap(audio, sample_rate=sample_rate)

        inputs, input_ids, transcript_span = self._prepare_forced_alignment_inputs(
            audio, language=language, transcript=transcript
        )
        audio_span = detect_audio_span(
            input_ids,
            audio_token_id=self.audio_token_id,
            audio_ms_per_token=self.audio_ms_per_token,
        )
        if audio_span is None or audio_span.length <= 0 or transcript_span is None:
            return []

        per_token_captures = self._run_forward_capture_transcript_attention(
            inputs=inputs, transcript_span=transcript_span
        )
        transcript_token_ids = input_ids[transcript_span[0] : transcript_span[1]]
        if not transcript_token_ids:
            return []

        num_layers = len(per_token_captures[0])
        layer_names = sorted(per_token_captures[0].keys())
        num_heads = int(per_token_captures[0][layer_names[0]].shape[1])
        audio_positions = list(range(audio_span.prompt_start, audio_span.prompt_end))

        scored: list[dict] = []
        for layer_idx in layer_names:
            for head_idx in range(num_heads):
                head = AlignAttHead(layer=layer_idx, head=head_idx, ts=0.0)
                per_token_rows: list[torch.Tensor] = []
                for captured in per_token_captures:
                    rows = extract_source_attention_rows_per_token(
                        layer_attentions_by_layer=captured,
                        alignatt_heads=[head],
                        source_positions=audio_positions,
                    )
                    if not rows:
                        per_token_rows.append(torch.zeros(1, len(audio_positions)))
                        continue
                    per_token_rows.append(rows[-1])

                argmaxes = compute_alignatt_source_argmaxes(
                    per_token_rows, filter_width=self.filter_width
                )
                token_ends = [
                    audio_position_to_end_seconds(
                        pos,
                        ms_per_token=audio_span.ms_per_token,
                        audio_duration_s=audio_duration_s,
                    )
                    for pos in argmaxes
                ]
                token_ends = _enforce_monotone(token_ends)
                predicted_words = aggregate_token_timings_to_words(
                    transcript,
                    generated_ids=transcript_token_ids,
                    tokenizer=self.tokenizer,
                    token_end_times_s=token_ends,
                    audio_duration_s=audio_duration_s,
                )
                mae = _word_end_mae(predicted_words, teacher.words)
                mono = monotonicity_score(argmaxes)
                coverage = sum(1 for p in argmaxes if p is not None) / max(1, len(argmaxes))
                scored.append(
                    {
                        "layer": int(layer_idx),
                        "head": int(head_idx),
                        "mae_seconds": float(mae) if mae is not None else None,
                        "monotonicity": float(mono),
                        "coverage": float(coverage),
                        "ts": _combine_head_score(mae, mono, coverage),
                    }
                )
        scored.sort(key=lambda entry: entry["ts"], reverse=True)
        if top_k is not None:
            scored = scored[: int(top_k)]
        return scored

    # -------------------- calibration / head selection -------------------

    def calibrate_alignment_heads(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        language: str,
        teacher: AlignmentResult,
        top_k: int | None = None,
    ) -> list[dict]:
        """Score every (layer, head) on one audio using a teacher alignment.

        For each attention head we (a) compute a monotonicity score and
        (b) compute MAE against the teacher's per-word end times. The
        returned scored list is sorted by a combined rank so a small
        ``top_k`` slice can be written back as a runtime head set.

        The teacher is typically the Qwen forced aligner on the same audio;
        this is used only for head ranking — not as ground truth — in line
        with the PLAN.md hierarchy of supervision.
        """
        if self.model is None:
            raise RuntimeError("Gemma alignment backend is not loaded.")

        inputs, input_ids = self._prepare_inputs(audio, language=language)
        audio_span = detect_audio_span(
            input_ids,
            audio_token_id=self.audio_token_id,
            audio_ms_per_token=self.audio_ms_per_token,
        )
        if audio_span is None or audio_span.length <= 0:
            return []
        audio_duration_s = float(len(audio)) / float(sample_rate)

        capture = self._generate_with_attention(
            inputs,
            audio_span,
            record_all_heads=True,
        )
        generated_ids = capture.generated_ids
        if not generated_ids:
            return []
        num_layers = len(capture.per_token_layer_attentions[0])
        layer_names = sorted(capture.per_token_layer_attentions[0].keys())
        num_heads = int(capture.per_token_layer_attentions[0][layer_names[0]].shape[1])

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        audio_positions = list(range(audio_span.prompt_start, audio_span.prompt_end))

        scored: list[dict] = []
        for layer_idx in layer_names:
            for head_idx in range(num_heads):
                head = AlignAttHead(layer=layer_idx, head=head_idx, ts=0.0)
                per_token_rows: list[torch.Tensor] = []
                for captured in capture.per_token_layer_attentions:
                    rows = extract_source_attention_rows_per_token(
                        layer_attentions_by_layer=captured,
                        alignatt_heads=[head],
                        source_positions=audio_positions,
                    )
                    if not rows:
                        per_token_rows.append(
                            torch.zeros(1, len(audio_positions))
                        )
                        continue
                    per_token_rows.append(rows[-1])

                argmaxes = compute_alignatt_source_argmaxes(
                    per_token_rows, filter_width=self.filter_width
                )
                token_ends = [
                    audio_position_to_end_seconds(
                        pos,
                        ms_per_token=audio_span.ms_per_token,
                        audio_duration_s=audio_duration_s,
                    )
                    for pos in argmaxes
                ]
                token_ends = _enforce_monotone(token_ends)
                predicted_words = aggregate_token_timings_to_words(
                    text,
                    generated_ids=generated_ids,
                    tokenizer=self.tokenizer,
                    token_end_times_s=token_ends,
                    audio_duration_s=audio_duration_s,
                )

                mae = _word_end_mae(predicted_words, teacher.words)
                mono = monotonicity_score(argmaxes)
                coverage = sum(1 for p in argmaxes if p is not None) / max(1, len(argmaxes))
                scored.append(
                    {
                        "layer": int(layer_idx),
                        "head": int(head_idx),
                        "mae_seconds": float(mae) if mae is not None else None,
                        "monotonicity": float(mono),
                        "coverage": float(coverage),
                        "ts": _combine_head_score(mae, mono, coverage),
                    }
                )

        scored.sort(key=lambda entry: entry["ts"], reverse=True)
        if top_k is not None:
            scored = scored[: int(top_k)]
        return scored


def _combine_head_score(
    mae_seconds: float | None, monotonicity: float, coverage: float
) -> float:
    """Rank heads by monotonicity / (1 + MAE). Coverage breaks ties.

    Monotonicity is the dominant signal because streaming needs forward
    progress; MAE is a soft quality term; coverage ensures heads that
    attend somewhere on the audio axis at all are preferred. The function
    is generic and contains no calibration constants tuned to any
    particular example.
    """
    if mae_seconds is None or math.isnan(mae_seconds) or math.isinf(mae_seconds):
        mae_term = 0.0
    else:
        mae_term = 1.0 / (1.0 + float(mae_seconds))
    return float(monotonicity) * 0.7 + mae_term * 0.25 + float(coverage) * 0.05


def _word_end_mae(
    predicted: Sequence[WordAlignment],
    teacher: Sequence[WordAlignment],
) -> float | None:
    if not predicted or not teacher:
        return None
    n = min(len(predicted), len(teacher))
    if n <= 0:
        return None
    errors = [abs(predicted[i].end_time - teacher[i].end_time) for i in range(n)]
    return float(sum(errors) / len(errors))


def _apply_word_end_offset(
    values: Sequence[float | None],
    *,
    offset_s: float,
    audio_duration_s: float,
) -> list[float | None]:
    """Subtract the calibrated lag and clamp to ``[0, audio_duration]``.

    The offset corrects a systematic lag between a causal LLM's attention
    peak and the acoustic word boundary; it is a single scalar fit once
    per ``(language, model)`` on the same teacher that selects the heads.
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

    A per-token argmax can locally regress even in an otherwise monotone
    head. The aligner contract requires a monotone word-end sequence, and
    the cleanest generic fix is a left-to-right running max; we only
    fill ``None`` forward, never backward, so the fallback does not mask
    tokens that legitimately fail to align.
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
