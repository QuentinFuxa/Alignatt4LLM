from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
import unicodedata

import torch
import torch.nn.functional as F
from cascade.translation_variants import RenderedTranslationPrompt, TranslationVariant
from cascade.source_frontier import SourceAccessibilityFrontier


@dataclass(frozen=True)
class AlignAttHead:
    layer: int
    head: int
    ts: float


@dataclass
class PromptCacheState:
    full_prompt_ids: list[int] = field(default_factory=list)
    prompt_kv_snapshot: list[tuple[int, torch.Tensor, torch.Tensor, int]] | None = None


@dataclass
class DraftDecodingResult:
    draft_generated_ids: list[int]
    prompt_num_tokens: int
    num_cached_tokens: int | None
    stop_reason: str | int | None
    prompt_kv_snapshot: list[tuple[int, torch.Tensor, torch.Tensor, int]] | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenProvenanceBreakdown:
    """Per-token attention mass distribution across prompt regions."""
    source_accessible: float
    source_inaccessible: float
    non_source_prompt: float
    suffix: float


@dataclass
class AlignAttProbeResult:
    accepted_candidate_ids: list[int]
    aligned_source_local_positions: list[int | None]
    unsafe_reason: str | None = None
    unsafe_target_token_index: int | None = None
    blocked_source_local_position: int | None = None
    blocked_source_unit_index: int | None = None
    stop_reason: str | int | None = None
    probe_backend: str | None = None
    provenance: list[TokenProvenanceBreakdown] | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class LayerInputCapture:
    module: Any
    hidden_states: torch.Tensor
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None


class IncrementalAlignAttTracker:
    def __init__(self, *, filter_width: int, normalization: str = "zscore"):
        self.filter_width = int(filter_width)
        self.normalization = str(normalization)
        if self.normalization not in {"zscore", "raw"}:
            raise ValueError(
                "AlignAtt online normalization must be 'zscore' or 'raw', "
                f"got {self.normalization!r}."
            )
        self.token_count = 0
        self.running_mean: torch.Tensor | None = None
        self.running_m2: torch.Tensor | None = None
        self.aligned_source_local_positions: list[int | None] = []

    def _observe(self, rows: torch.Tensor) -> None:
        self.token_count += 1
        if self.running_mean is None or self.running_m2 is None:
            self.running_mean = rows.clone()
            self.running_m2 = torch.zeros_like(rows)
            return
        delta = rows - self.running_mean
        self.running_mean = self.running_mean + delta / float(self.token_count)
        delta2 = rows - self.running_mean
        self.running_m2 = self.running_m2 + delta * delta2

    def update_with_per_head_positions(
        self,
        source_attention_rows: torch.Tensor,
    ) -> tuple[int | None, list[int]]:
        if source_attention_rows.ndim != 2:
            raise ValueError(
                "source_attention_rows must have shape [num_heads, source_token_count] "
                f"but got {tuple(source_attention_rows.shape)}"
            )

        rows = source_attention_rows.detach().float()
        if rows.shape[-1] <= 0:
            self.aligned_source_local_positions.append(None)
            return None, []

        if self.normalization == "raw" or self.token_count <= 1:
            normalized_rows = rows
        else:
            variance = self.running_m2 / max(1, self.token_count)
            normalized_rows = (rows - self.running_mean) / variance.sqrt().clamp_min(1e-6)
        smoothed_rows = median_filter_last_dim(normalized_rows, self.filter_width)
        per_head_positions = torch.argmax(smoothed_rows, dim=-1).tolist()
        aligned_position = _median_position(per_head_positions)
        self._observe(rows)
        self.aligned_source_local_positions.append(aligned_position)
        return aligned_position, [int(position) for position in per_head_positions]

    def update(self, source_attention_rows: torch.Tensor) -> int | None:
        aligned_position, _ = self.update_with_per_head_positions(source_attention_rows)
        return aligned_position


class SelectedAttentionRecorder:
    def __init__(self, *, model, alignatt_heads: Sequence[AlignAttHead]):
        self._capture_active = False
        self._hooks = []

        model_layers = self._resolve_text_layers(model)
        if model_layers is None:
            raise RuntimeError("Gemma text layers are not available for AlignAtt recording.")

        layer_to_heads: dict[int, list[int]] = {}
        for alignatt_head in alignatt_heads:
            layer_to_heads.setdefault(int(alignatt_head.layer), []).append(int(alignatt_head.head))
        self.layer_to_heads = {
            layer_idx: tuple(sorted(set(head_ids)))
            for layer_idx, head_ids in layer_to_heads.items()
        }

        for layer_idx in sorted(self.layer_to_heads):
            self._hooks.append(
                model_layers[layer_idx].self_attn.register_forward_hook(
                    self._make_hook(layer_idx),
                )
            )

    @staticmethod
    def _resolve_text_layers(model):
        candidates = (
            ("model", "layers"),
            ("model", "language_model", "layers"),
            ("language_model", "layers"),
            ("base_model", "layers"),
            ("base_model", "language_model", "layers"),
            ("text_model", "layers"),
            ("model", "text_model", "layers"),
        )
        for path in candidates:
            current = model
            for attr in path:
                current = getattr(current, attr, None)
                if current is None:
                    break
            if current is not None:
                return current
        return None

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            if not self._capture_active:
                return
            if not isinstance(output, tuple) or len(output) < 2:
                return
            attn_weights = output[1]
            if attn_weights is None:
                return
            self._captured_layer_attentions[layer_idx] = attn_weights.detach().float()

        return hook

    @contextmanager
    def capture(self) -> dict[int, torch.Tensor]:
        if self._capture_active:
            raise RuntimeError("Nested AlignAtt attention capture is not supported.")

        self._capture_active = True
        self._captured_layer_attentions: dict[int, torch.Tensor] = {}
        try:
            yield self._captured_layer_attentions
        finally:
            self._capture_active = False
            self._captured_layer_attentions = {}


class SelectedLayerInputRecorder:
    def __init__(self, *, model, alignatt_heads: Sequence[AlignAttHead]):
        self._capture_active = False
        self._hooks = []

        model_layers = SelectedAttentionRecorder._resolve_text_layers(model)
        if model_layers is None:
            raise RuntimeError("Gemma text layers are not available for AlignAtt layer-input recording.")

        layer_indices = sorted({int(alignatt_head.layer) for alignatt_head in alignatt_heads})
        for layer_idx in layer_indices:
            self._hooks.append(
                model_layers[layer_idx].self_attn.register_forward_hook(
                    self._make_hook(layer_idx),
                    with_kwargs=True,
                )
            )

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, kwargs, output):
            del output
            if not self._capture_active:
                return
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and len(inputs) > 0:
                hidden_states = inputs[0]
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None and len(inputs) > 1:
                position_embeddings = inputs[1]
            if hidden_states is None:
                return
            normalized_position_embeddings = None
            if (
                isinstance(position_embeddings, tuple)
                and len(position_embeddings) == 2
                and position_embeddings[0] is not None
                and position_embeddings[1] is not None
            ):
                normalized_position_embeddings = (
                    position_embeddings[0].detach(),
                    position_embeddings[1].detach(),
                )
            self._captured_layer_inputs[layer_idx] = LayerInputCapture(
                module=module,
                hidden_states=hidden_states.detach(),
                position_embeddings=normalized_position_embeddings,
            )

        return hook

    @contextmanager
    def capture(self) -> dict[int, LayerInputCapture]:
        if self._capture_active:
            raise RuntimeError("Nested AlignAtt layer-input capture is not supported.")

        self._capture_active = True
        self._captured_layer_inputs: dict[int, LayerInputCapture] = {}
        try:
            yield self._captured_layer_inputs
        finally:
            self._capture_active = False
            self._captured_layer_inputs = {}


@dataclass
class AlignAttAcceptance:
    accepted_generated_ids: list[int]
    alignatt_metadata: dict[str, Any] | None


@dataclass
class AlignAttUnitPolicyDecision:
    accepted_candidate_ids: list[int]
    unsafe_reason: str | None
    unsafe_target_token_index: int | None
    unsafe_token_id: int | None
    blocked_source_local_position: int | None
    blocked_source_unit_index: int | None
    stop_reason: str | int | None
    metadata: dict[str, Any]


@dataclass
class MTBackendResult:
    draft_text: str
    acceptance_text: str
    draft_generated_token_ids: tuple[int, ...] = ()
    accepted_generated_token_ids: tuple[int, ...] = ()
    draft_token_ids: tuple[int, ...] = ()
    accepted_token_ids: tuple[int, ...] = ()
    num_cached_tokens: int | None = None
    prompt_num_tokens: int | None = None
    stop_reason: str | int | None = None
    alignatt_metadata: dict[str, Any] | None = None
    timings_ms: dict[str, float] | None = None


@dataclass(frozen=True)
class PromptSourceUnitSpan:
    unit_index: int
    text: str
    prompt_token_positions: tuple[int, ...]
    is_accessible: bool
    start_ms: float | None
    end_ms: float | None


@dataclass(frozen=True)
class PromptSourceMap:
    source_text: str
    source_token_positions: tuple[int, ...]
    source_unit_spans: tuple[PromptSourceUnitSpan, ...]
    accessible_source_token_count: int
    accessible_unit_count: int
    total_unit_count: int
    current_source_ms: float
    inaccessible_ms: float
    is_final: bool


@dataclass(frozen=True)
class RenderedPromptWithSourceMap:
    prompt_token_ids: tuple[int, ...]
    prompt_text: str
    source_map: PromptSourceMap | None


def build_mt_backend(
    *,
    model_name: str,
    runtime_config: SimpleNamespace,
) -> "BaseMTBackend":
    backend_name = getattr(runtime_config, "mt_backend_name", "gemma_vllm_alignatt")
    if backend_name == "gemma_vllm_alignatt":
        from cascade.mt.gemma_vllm_backend import GemmaVLLMMTBackend

        return GemmaVLLMMTBackend(
            model_name=model_name,
            runtime_config=runtime_config,
        )
    if backend_name == "milmmt_vllm_alignatt":
        from cascade.mt.gemma_vllm_backend import MiLMMTVLLMMTBackend

        return MiLMMTVLLMMTBackend(
            model_name=model_name,
            runtime_config=runtime_config,
        )
    raise ValueError(f"Unknown mt_backend_name: {backend_name!r}")


class BaseMTBackend(ABC):
    def __init__(self, *, model_name: str, runtime_config: SimpleNamespace):
        self.model_name = model_name
        self.runtime_config = runtime_config
        self.tokenizer = None

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def translate(
        self,
        *,
        rendered_prompt: RenderedTranslationPrompt,
        variant: TranslationVariant,
        is_partial: bool,
        prompt_cache_state: PromptCacheState | None = None,
    ) -> MTBackendResult:
        raise NotImplementedError

    def reset_caches(self) -> None:
        """Drop any per-run prompt cache state so reruns are independent."""
        return None

    def resolve_generation_stop_token_ids(self) -> tuple[int, ...]:
        if self.tokenizer is None:
            raise RuntimeError("Gemma tokenizer is not loaded. Run load() first.")

        stop_ids = {
            int(token_id)
            for token_id in getattr(self.tokenizer, "all_special_ids", [])
            if token_id is not None and int(token_id) >= 0
        }
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is not None and int(eos_token_id) >= 0:
            stop_ids.add(int(eos_token_id))
        return tuple(sorted(stop_ids))

    def render_prompt_token_ids(self, rendered_prompt: RenderedTranslationPrompt) -> list[int]:
        if self.tokenizer is None:
            raise RuntimeError("Gemma tokenizer is not loaded. Run load() first.")
        template_kwargs: dict[str, Any] = {
            "tokenize": True,
            "return_dict": True,
        }
        if rendered_prompt.continue_final_message:
            template_kwargs["continue_final_message"] = True
        else:
            template_kwargs["add_generation_prompt"] = True
        prompt_token_ids = self.tokenizer.apply_chat_template(
            rendered_prompt.messages,
            **template_kwargs,
        )
        if hasattr(prompt_token_ids, "keys") and "input_ids" in prompt_token_ids:
            prompt_token_ids = prompt_token_ids["input_ids"]
        elif hasattr(prompt_token_ids, "ids"):
            prompt_token_ids = prompt_token_ids.ids
        return list(prompt_token_ids)

    def render_prompt_text(self, rendered_prompt: RenderedTranslationPrompt) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Gemma tokenizer is not loaded. Run load() first.")
        template_kwargs: dict[str, Any] = {
            "tokenize": False,
        }
        if rendered_prompt.continue_final_message:
            template_kwargs["continue_final_message"] = True
        else:
            template_kwargs["add_generation_prompt"] = True
        return str(self.tokenizer.apply_chat_template(rendered_prompt.messages, **template_kwargs))

    def render_prompt_package(
        self,
        rendered_prompt: RenderedTranslationPrompt,
    ) -> RenderedPromptWithSourceMap:
        prompt_token_ids = tuple(self.render_prompt_token_ids(rendered_prompt))
        prompt_text = self.render_prompt_text(rendered_prompt)
        source_map = build_prompt_source_map(
            tokenizer=self.tokenizer,
            rendered_prompt=rendered_prompt,
            prompt_text=prompt_text,
        )
        return RenderedPromptWithSourceMap(
            prompt_token_ids=prompt_token_ids,
            prompt_text=prompt_text,
            source_map=source_map,
        )

    def compute_max_tokens(
        self,
        *,
        prompt_tokens: int,
        source_text: str,
        is_partial: bool,
        assistant_prefill: str,
        is_final_source: bool = False,
    ) -> int:
        if self.tokenizer is None:
            raise RuntimeError("Gemma tokenizer is not loaded. Run load() first.")
        source_tokens = len(self.tokenizer(source_text, add_special_tokens=False)["input_ids"])
        if is_partial and not is_final_source:
            desired_max_tokens = max(
                self.runtime_config.partial_translation_min_new_tokens,
                int(source_tokens * self.runtime_config.partial_translation_token_budget_ratio)
                + self.runtime_config.partial_translation_token_budget_buffer,
            )
            max_token_cap = self.runtime_config.partial_max_new_tokens
        else:
            desired_max_tokens = max(
                self.runtime_config.translation_min_new_tokens,
                int(source_tokens * self.runtime_config.translation_token_budget_ratio)
                + self.runtime_config.translation_token_budget_buffer,
            )
            max_token_cap = self.runtime_config.max_new_tokens

        available_max_tokens = (
            getattr(
                self.runtime_config,
                "mt_max_model_len",
                self.runtime_config.gemma_max_model_len,
            )
            - prompt_tokens
            - self.runtime_config.translation_generation_margin
        )
        if available_max_tokens < 1:
            raise RuntimeError(
                "MT prompt exhausted the context window: "
                f"prompt_tokens={prompt_tokens} "
                "max_model_len="
                f"{getattr(self.runtime_config, 'mt_max_model_len', self.runtime_config.gemma_max_model_len)}"
            )
        return min(max_token_cap, desired_max_tokens, available_max_tokens)

    @staticmethod
    def apply_repetition_penalty(
        logits: torch.Tensor,
        *,
        prior_token_ids: Sequence[int],
        repetition_penalty: float,
    ) -> torch.Tensor:
        if repetition_penalty <= 1.0:
            return logits
        for token_id in set(int(token_id) for token_id in prior_token_ids):
            if token_id < 0 or token_id >= logits.shape[-1]:
                continue
            if logits[token_id] > 0:
                logits[token_id] /= repetition_penalty
            else:
                logits[token_id] *= repetition_penalty
        return logits

    def decode_candidate_text(
        self,
        *,
        generated_ids: Sequence[int],
        assistant_prefill: str,
        variant: TranslationVariant,
        is_partial: bool,
    ) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Gemma tokenizer is not loaded. Run load() first.")
        generated_text = self.tokenizer.decode(
            list(generated_ids),
            skip_special_tokens=False,
        )
        return variant.normalize_output(
            generated_text=generated_text,
            assistant_prefill=assistant_prefill,
            is_partial=is_partial,
        )

    def encode_semantic_target_token_ids(self, text: str) -> tuple[int, ...]:
        if self.tokenizer is None:
            raise RuntimeError("Gemma tokenizer is not loaded. Run load() first.")
        normalized_text = text.strip()
        if not normalized_text:
            return ()
        token_ids = self.tokenizer(normalized_text, add_special_tokens=False)["input_ids"]
        return tuple(int(token_id) for token_id in token_ids)


class AlignAttDecoderPolicy:
    TERMINAL_PUNCTUATION_CHARS = frozenset(".!?。！？")
    TERMINAL_PUNCTUATION_CLOSERS = frozenset('"\'”’」』）)]】》〉')

    def __init__(self, *, tokenizer, runtime_config: SimpleNamespace):
        self.tokenizer = tokenizer
        self.runtime_config = runtime_config

    def alignatt_filter_width(self) -> int:
        return int(getattr(self.runtime_config, "translation_alignatt_filter_width", 7))

    def alignatt_acceptance_variant(self) -> str:
        return str(
            getattr(
                self.runtime_config,
                "translation_alignatt_acceptance_variant",
                "token",
            )
        )

    def source_regression_action(self) -> str:
        return str(
            getattr(
                self.runtime_config,
                "translation_alignatt_source_regression_action",
                "stop",
            )
        )

    def source_frontier_action(self) -> str:
        return str(
            getattr(
                self.runtime_config,
                "translation_alignatt_source_frontier_action",
                "stop",
            )
        )

    @staticmethod
    def _finite_float(value: float | int | None) -> float | None:
        if value is None:
            return None
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None

    @staticmethod
    def _provenance_has_nonfinite(row: TokenProvenanceBreakdown) -> bool:
        return any(
            not math.isfinite(float(value))
            for value in (
                row.source_accessible,
                row.source_inaccessible,
                row.non_source_prompt,
                row.suffix,
            )
        )

    def _target_uses_char_stability_units(self) -> bool:
        target_lang = str(getattr(self.runtime_config, "target_lang", "")).strip().lower()
        target_code = str(
            getattr(self.runtime_config, "target_lang_code", "")
        ).strip().lower()
        return target_code in {"zh", "zh-cn", "ja"} or target_lang in {
            "zh",
            "zh-cn",
            "ja",
            "chinese",
            "simplified chinese",
            "traditional chinese",
            "chinese (simplified)",
            "chinese (traditional)",
            "japanese",
        }

    @staticmethod
    def _decoded_char_stability_unit_count(text: str) -> int:
        return len(AlignAttDecoderPolicy._decoded_char_stability_unit_end_offsets(text))

    @classmethod
    def _decoded_char_stability_unit_end_offsets(cls, text: str) -> list[int]:
        """Return decoded-text end offsets for stable units in CJK-style output.

        Chinese/Japanese characters and punctuation can commit one by one, but
        Latin words, acronyms, and numbers should not commit character by
        character. They become stable only once a following boundary closes the
        run. This matters for EN->ZH drafts containing names, years, model
        names, URLs, or acronyms.
        """
        end_offsets: list[int] = []
        pending_spacing_run_start: int | None = None
        for index, char in enumerate(text):
            if char == "\ufffd":
                continue
            if char.isspace():
                if pending_spacing_run_start is not None:
                    end_offsets.append(index)
                    pending_spacing_run_start = None
                continue
            if cls._is_non_spacing_script_char(char):
                if pending_spacing_run_start is not None:
                    end_offsets.append(index)
                    pending_spacing_run_start = None
                end_offsets.append(index + 1)
                continue
            if cls._is_spacing_run_char(char):
                if pending_spacing_run_start is None:
                    pending_spacing_run_start = index
                continue
            if pending_spacing_run_start is not None:
                end_offsets.append(index)
                pending_spacing_run_start = None
            end_offsets.append(index + 1)
        return end_offsets

    def _decode_generated_ids_for_boundaries(self, ids: Sequence[int]) -> str:
        return str(
            self.tokenizer.decode(
                list(ids),
                skip_special_tokens=False,
            )
        )

    def _decoded_char_stability_spans(
        self,
        generated_ids: Sequence[int],
    ) -> list[tuple[int, int]]:
        ids = list(generated_ids)
        if not ids:
            return []
        decoded_text = self._decode_generated_ids_for_boundaries(ids)
        unit_end_offsets = self._decoded_char_stability_unit_end_offsets(decoded_text)
        if not unit_end_offsets:
            return []
        spans: list[tuple[int, int]] = []
        current_start = 0
        search_start = 1
        for unit_end_offset in unit_end_offsets:
            target_prefix = decoded_text[:unit_end_offset]
            for end in range(search_start, len(ids) + 1):
                prefix_text = self._decode_generated_ids_for_boundaries(ids[:end])
                if "\ufffd" in prefix_text:
                    continue
                if len(prefix_text) < unit_end_offset:
                    continue
                if prefix_text[:unit_end_offset] != target_prefix:
                    continue
                if end > current_start:
                    spans.append((current_start, end))
                    current_start = end
                    search_start = end
                break
        return spans

    @staticmethod
    def _token_visible_chars(token: str) -> str:
        if not token:
            return ""
        if token.startswith(("▁", "Ġ")):
            return token[1:]
        return token

    @staticmethod
    def _is_non_spacing_script_char(ch: str) -> bool:
        if not ch:
            return False
        cp = ord(ch)
        # CJK Unified Ideographs and common extensions, plus Japanese kana.
        # These scripts do not delimit words with whitespace, so each such
        # character acts as its own target stability unit.
        return (
            0x3040 <= cp <= 0x30FF  # Hiragana / Katakana
            or 0x3400 <= cp <= 0x4DBF  # CJK Extension A
            or 0x4E00 <= cp <= 0x9FFF  # CJK Unified Ideographs
            or 0xF900 <= cp <= 0xFAFF  # CJK Compatibility Ideographs
            or 0x20000 <= cp <= 0x2A6DF  # CJK Extension B
            or 0x2A700 <= cp <= 0x2B73F  # CJK Extension C
            or 0x2B740 <= cp <= 0x2B81F  # CJK Extension D
            or 0x2B820 <= cp <= 0x2CEAF  # CJK Extension E
        )

    @staticmethod
    def _is_spacing_run_char(ch: str) -> bool:
        if not ch:
            return False
        if ch in {"_", "-", "'", "’"}:
            return True
        category = unicodedata.category(ch)
        return category[0] in {"L", "M", "N"}

    @classmethod
    def token_starts_stability_unit(
        cls,
        token: str,
        *,
        is_first_token: bool = False,
    ) -> bool:
        """Return True when a token opens a new target stability unit.

        A stability unit is the minimal prefix of generated text that cannot be
        retroactively altered by future decoding steps. For whitespace-segmented
        languages (en->de, en->it, ...) that unit is a full word, signalled by a
        leading SentencePiece ``▁`` / byte-pair ``Ġ`` / raw whitespace prefix.
        For non-spacing scripts (en->zh, en->ja) each CJK/kana character is its
        own unit, so any token whose first visible character lives in those
        ranges also starts a new unit.
        """
        if not token:
            return False
        if is_first_token:
            return True
        if token.startswith(("▁", "Ġ")):
            return True
        if token[0].isspace():
            return True
        if token.startswith("<0x0A>"):
            return True
        visible = cls._token_visible_chars(token)
        if visible and cls._is_non_spacing_script_char(visible[0]):
            return True
        return False

    # Back-compat alias for callers that still refer to the legacy name.
    @classmethod
    def token_starts_new_word(cls, token: str) -> bool:
        return cls.token_starts_stability_unit(token)

    def trim_to_last_stability_unit(self, generated_ids: Sequence[int]) -> list[int]:
        """Drop the trailing, possibly-incomplete target stability unit."""
        if not generated_ids:
            return []
        if self._target_uses_char_stability_units():
            spans = self._decoded_char_stability_spans(generated_ids)
            if not spans:
                return []
            return list(generated_ids[: spans[-1][1]])
        token_strings = self.tokenizer.convert_ids_to_tokens(list(generated_ids))
        unit_start_indices = [
            idx
            for idx, token in enumerate(token_strings)
            if self.token_starts_stability_unit(str(token), is_first_token=(idx == 0))
        ]
        if len(unit_start_indices) <= 1:
            return []
        return list(generated_ids[: unit_start_indices[-1]])

    # Back-compat alias. New code should call ``trim_to_last_stability_unit``.
    def trim_to_last_complete_word(self, generated_ids: Sequence[int]) -> list[int]:
        return self.trim_to_last_stability_unit(generated_ids)

    def cut_last_target_stability_units(
        self,
        generated_ids: Sequence[int],
        *,
        cutoff_units: int,
    ) -> list[int]:
        """Return ``generated_ids`` after dropping the last N target units.

        The unit definition matches ``trim_to_last_stability_unit``: whitespace
        languages use word starts, while CJK-style no-space scripts can advance
        character by character.
        """
        ids = list(generated_ids)
        if not ids:
            return []
        cutoff_units = int(cutoff_units)
        if cutoff_units <= 0:
            return ids
        if self._target_uses_char_stability_units():
            spans = self._decoded_char_stability_spans(ids)
            keep_unit_count = max(0, len(spans) - cutoff_units)
            if keep_unit_count <= 0:
                return []
            keep_end = spans[keep_unit_count - 1][1]
            return ids[:keep_end]
        token_strings = self.tokenizer.convert_ids_to_tokens(ids)
        unit_start_indices = [
            idx
            for idx, token in enumerate(token_strings)
            if self.token_starts_stability_unit(str(token), is_first_token=(idx == 0))
        ]
        if not unit_start_indices:
            return []
        keep_unit_count = max(0, len(unit_start_indices) - cutoff_units)
        if keep_unit_count <= 0:
            return []
        keep_end = (
            unit_start_indices[keep_unit_count]
            if keep_unit_count < len(unit_start_indices)
            else len(ids)
        )
        return ids[:keep_end]

    def keep_first_target_stability_units(
        self,
        generated_ids: Sequence[int],
        *,
        max_units: int,
    ) -> list[int]:
        """Return at most the first ``max_units`` complete target units."""
        ids = list(generated_ids)
        if not ids:
            return []
        max_units = int(max_units)
        if max_units <= 0:
            return []
        unit_end_indices = self.target_stability_unit_end_token_indices(ids)
        if len(unit_end_indices) <= max_units:
            return ids
        keep_end = unit_end_indices[max_units - 1]
        return ids[:keep_end]

    def count_target_stability_units(self, generated_ids: Sequence[int]) -> int:
        """Count target stability units represented by a generated-id prefix."""
        ids = list(generated_ids)
        if not ids:
            return 0
        if self._target_uses_char_stability_units():
            return len(self._decoded_char_stability_spans(ids))
        token_strings = self.tokenizer.convert_ids_to_tokens(ids)
        return sum(
            1
            for idx, token in enumerate(token_strings)
            if self.token_starts_stability_unit(str(token), is_first_token=(idx == 0))
        )

    def target_stability_unit_end_token_indices(
        self,
        generated_ids: Sequence[int],
    ) -> list[int]:
        """Return token-prefix ends that align with target stability units."""
        ids = list(generated_ids)
        if not ids:
            return []
        if self._target_uses_char_stability_units():
            return [int(end) for _, end in self._decoded_char_stability_spans(ids)]
        token_strings = self.tokenizer.convert_ids_to_tokens(ids)
        unit_start_indices = [
            idx
            for idx, token in enumerate(token_strings)
            if self.token_starts_stability_unit(str(token), is_first_token=(idx == 0))
        ]
        if not unit_start_indices:
            return []
        unit_end_indices = [
            int(unit_start_indices[index + 1])
            for index in range(len(unit_start_indices) - 1)
        ]
        unit_end_indices.append(len(ids))
        return sorted({end for end in unit_end_indices if 0 < end <= len(ids)})

    def token_closes_target_stability_unit(
        self,
        *,
        accepted_candidate_ids: Sequence[int],
        next_token_id: int,
    ) -> bool:
        """Return True when ``next_token_id`` closes a stable unit boundary.

        The token itself is not accepted. This only tells ``finalize_partial``
        whether the accepted prefix immediately before an unsafe token is now
        complete enough to commit. For EN->ZH this catches punctuation or a CJK
        character that closes a preceding Latin/number run.
        """
        if not self._target_uses_char_stability_units():
            return False
        accepted_ids = list(accepted_candidate_ids)
        accepted_count = self.count_target_stability_units(accepted_ids)
        combined_count = self.count_target_stability_units(
            [*accepted_ids, int(next_token_id)]
        )
        return combined_count > accepted_count

    def should_stop_in_loop(
        self,
        *,
        current_source_local_position: int | None,
        accessible_source_token_count: int,
        source_inaccessible_mass: float | None = None,
    ) -> tuple[str | None, int | None]:
        if current_source_local_position is None:
            return None, None

        border_margin = int(
            getattr(self.runtime_config, "translation_alignatt_border_margin", 0)
        )
        frontier = max(0, int(accessible_source_token_count)) + border_margin
        if current_source_local_position >= frontier:
            frontier_min_inaccessible_mass = float(
                getattr(
                    self.runtime_config,
                    "translation_alignatt_frontier_min_inaccessible_mass",
                    0.0,
                )
            )
            if (
                frontier_min_inaccessible_mass > 0.0
                and source_inaccessible_mass is not None
                and float(source_inaccessible_mass) < frontier_min_inaccessible_mass
            ):
                return None, current_source_local_position
            return "source_frontier", current_source_local_position
        return None, current_source_local_position

    def should_stop_for_provenance_mass(
        self,
        *,
        source_accessible_mass: float,
        source_inaccessible_mass: float,
        non_source_prompt_mass: float | None = None,
    ) -> str | None:
        """Optional confidence gates from the source provenance partition.

        The argmax frontier answers "where is the strongest source point?".
        These gates answer the complementary question: "how much of this token's
        attention is actually grounded in accessible source versus future
        source?". Defaults preserve the historical argmax-only policy.
        """
        accessible = self._finite_float(source_accessible_mass)
        inaccessible = self._finite_float(source_inaccessible_mass)
        if accessible is None or inaccessible is None:
            return "provenance_nonfinite"
        max_inaccessible = float(
            getattr(
                self.runtime_config,
                "translation_alignatt_max_inaccessible_source_mass",
                1.0,
            )
        )
        if max_inaccessible < 1.0 and inaccessible > max_inaccessible:
            return "provenance_inaccessible_high"

        non_source = self._finite_float(non_source_prompt_mass)
        max_non_source_prompt = float(
            getattr(
                self.runtime_config,
                "translation_alignatt_max_non_source_prompt_mass",
                1.0,
            )
        )
        if (
            max_non_source_prompt < 1.0
            and non_source is not None
            and non_source > max_non_source_prompt
        ):
            return "provenance_non_source_high"

        min_margin = float(
            getattr(
                self.runtime_config,
                "translation_alignatt_min_accessible_inaccessible_margin",
                -1.0,
            )
        )
        if min_margin > -1.0 and accessible - inaccessible < min_margin:
            return "provenance_margin_weak"
        return None

    @staticmethod
    def _mean_accessible_source_mass(
        provenance_mass: Sequence[TokenProvenanceBreakdown],
        *,
        token_count: int,
    ) -> float | None:
        return AlignAttDecoderPolicy._mean_accessible_source_mass_range(
            provenance_mass,
            start=0,
            end=int(token_count),
        )

    @staticmethod
    def _mean_accessible_source_mass_range(
        provenance_mass: Sequence[TokenProvenanceBreakdown],
        *,
        start: int,
        end: int,
    ) -> float | None:
        start_index = max(0, int(start))
        end_index = int(end)
        count = end_index - start_index
        if count <= 0 or len(provenance_mass) < end_index:
            return None
        values = [
            AlignAttDecoderPolicy._finite_float(row.source_accessible)
            for row in provenance_mass[start_index:end_index]
        ]
        if any(value is None for value in values):
            return None
        return sum(float(value) for value in values) / float(count)

    def _recent_accessible_source_mass(
        self,
        generated_ids: Sequence[int],
        *,
        provenance_mass: Sequence[TokenProvenanceBreakdown],
        token_count: int,
        recent_units: int,
    ) -> float | None:
        count = int(token_count)
        if count <= 0:
            return None
        units = int(recent_units)
        if units <= 0:
            return self._mean_accessible_source_mass(
                provenance_mass,
                token_count=count,
            )
        prefix_ids = list(generated_ids[:count])
        unit_ends = [
            end
            for end in self.target_stability_unit_end_token_indices(prefix_ids)
            if end <= count
        ]
        if not unit_ends:
            return None
        start = 0 if len(unit_ends) <= units else unit_ends[-units - 1]
        return self._mean_accessible_source_mass_range(
            provenance_mass,
            start=start,
            end=count,
        )

    def _recent_unit_accessible_source_mass_floor(
        self,
        generated_ids: Sequence[int],
        *,
        provenance_mass: Sequence[TokenProvenanceBreakdown],
        token_count: int,
        recent_units: int,
    ) -> float | None:
        count = int(token_count)
        if count <= 0:
            return None
        units = int(recent_units)
        if units <= 0:
            return self._mean_accessible_source_mass(
                provenance_mass,
                token_count=count,
            )
        prefix_ids = list(generated_ids[:count])
        unit_ends = [
            int(end)
            for end in self.target_stability_unit_end_token_indices(prefix_ids)
            if 0 < int(end) <= count
        ]
        if not unit_ends:
            return None
        unit_starts = [0, *unit_ends[:-1]]
        unit_means: list[float] = []
        for start, end in list(zip(unit_starts, unit_ends))[-units:]:
            mean_accessible = self._mean_accessible_source_mass_range(
                provenance_mass,
                start=start,
                end=end,
            )
            if mean_accessible is None:
                return None
            unit_means.append(float(mean_accessible))
        return min(unit_means) if unit_means else None

    @staticmethod
    def _accepted_prefix_provenance_passes(
        *,
        mean_accessible: float | None,
        recent_mean_accessible: float | None,
        recent_min_unit_accessible: float | None,
        threshold: float,
    ) -> bool:
        return (
            mean_accessible is not None
            and mean_accessible >= threshold
            and recent_mean_accessible is not None
            and recent_mean_accessible >= threshold
            and recent_min_unit_accessible is not None
            and recent_min_unit_accessible >= threshold
        )

    def trim_for_accepted_prefix_provenance(
        self,
        generated_ids: Sequence[int],
        *,
        provenance_mass: Sequence[TokenProvenanceBreakdown] | None,
    ) -> tuple[list[int], float | None, float | None, float | None, bool]:
        threshold = float(
            getattr(
                self.runtime_config,
                "translation_alignatt_min_accepted_accessible_source_mass",
                0.0,
            )
        )
        recent_units = int(
            getattr(
                self.runtime_config,
                "translation_alignatt_accepted_accessible_source_mass_recent_units",
                2,
            )
        )
        ids = list(generated_ids)
        if threshold <= 0.0 or not ids or not provenance_mass:
            return ids, self._mean_accessible_source_mass(
                provenance_mass or [],
                token_count=len(ids),
            ), self._recent_accessible_source_mass(
                ids,
                provenance_mass=provenance_mass or [],
                token_count=len(ids),
                recent_units=recent_units,
            ), self._recent_unit_accessible_source_mass_floor(
                ids,
                provenance_mass=provenance_mass or [],
                token_count=len(ids),
                recent_units=recent_units,
            ), False
        if len(provenance_mass) < len(ids):
            return ids, None, None, None, False

        original_mean = self._mean_accessible_source_mass(
            provenance_mass,
            token_count=len(ids),
        )
        original_recent_mean = self._recent_accessible_source_mass(
            ids,
            provenance_mass=provenance_mass,
            token_count=len(ids),
            recent_units=recent_units,
        )
        original_recent_min_unit = self._recent_unit_accessible_source_mass_floor(
            ids,
            provenance_mass=provenance_mass,
            token_count=len(ids),
            recent_units=recent_units,
        )
        if self._accepted_prefix_provenance_passes(
            mean_accessible=original_mean,
            recent_mean_accessible=original_recent_mean,
            recent_min_unit_accessible=original_recent_min_unit,
            threshold=threshold,
        ):
            return ids, original_mean, original_recent_mean, original_recent_min_unit, False

        for end in reversed(self.target_stability_unit_end_token_indices(ids)):
            if end >= len(ids):
                continue
            mean_accessible = self._mean_accessible_source_mass(
                provenance_mass,
                token_count=end,
            )
            recent_mean_accessible = self._recent_accessible_source_mass(
                ids,
                provenance_mass=provenance_mass,
                token_count=end,
                recent_units=recent_units,
            )
            recent_min_unit_accessible = self._recent_unit_accessible_source_mass_floor(
                ids,
                provenance_mass=provenance_mass,
                token_count=end,
                recent_units=recent_units,
            )
            if self._accepted_prefix_provenance_passes(
                mean_accessible=mean_accessible,
                recent_mean_accessible=recent_mean_accessible,
                recent_min_unit_accessible=recent_min_unit_accessible,
                threshold=threshold,
            ):
                return (
                    ids[:end],
                    mean_accessible,
                    recent_mean_accessible,
                    recent_min_unit_accessible,
                    True,
                )
        return [], original_mean, original_recent_mean, original_recent_min_unit, bool(ids)

    @classmethod
    def _ends_with_terminal_punctuation(cls, text: str) -> bool:
        stripped = text.rstrip()
        while stripped and stripped[-1] in cls.TERMINAL_PUNCTUATION_CLOSERS:
            stripped = stripped[:-1].rstrip()
        return bool(stripped) and stripped[-1] in cls.TERMINAL_PUNCTUATION_CHARS

    @staticmethod
    def _max_source_mass_range(
        provenance_mass: Sequence[TokenProvenanceBreakdown],
        *,
        start: int,
        end: int,
    ) -> float | None:
        start_index = max(0, int(start))
        end_index = int(end)
        if end_index <= start_index or len(provenance_mass) < end_index:
            return None
        values: list[float] = []
        for row in provenance_mass[start_index:end_index]:
            accessible = AlignAttDecoderPolicy._finite_float(row.source_accessible)
            inaccessible = AlignAttDecoderPolicy._finite_float(row.source_inaccessible)
            if accessible is None or inaccessible is None:
                return None
            values.append(accessible + inaccessible)
        return max(values) if values else None

    def terminal_punctuation_min_source_mass(self) -> float:
        configured = float(
            getattr(
                self.runtime_config,
                "translation_alignatt_terminal_punctuation_min_source_mass",
                0.06,
            )
        )
        regression_relaxation = float(
            getattr(
                self.runtime_config,
                "translation_alignatt_source_regression_min_source_mass",
                0.0,
            )
        )
        return max(configured, regression_relaxation)

    def trim_low_source_terminal_punctuation(
        self,
        generated_ids: Sequence[int],
        *,
        provenance_mass: Sequence[TokenProvenanceBreakdown] | None,
    ) -> tuple[list[int], bool, float | None, int]:
        ids = list(generated_ids)
        if not ids or not provenance_mass:
            return ids, False, None, 0
        if not bool(
            getattr(
                self.runtime_config,
                "translation_alignatt_defer_low_source_terminal_punctuation",
                False,
            )
        ):
            return ids, False, None, 0
        threshold = self.terminal_punctuation_min_source_mass()
        unit_ends = self.target_stability_unit_end_token_indices(ids)
        if not unit_ends:
            return ids, False, None, 0
        trimmed_unit_count = 0
        last_source_mass: float | None = None
        while unit_ends and self._ends_with_terminal_punctuation(
            self._decode_generated_ids_for_boundaries(ids[: unit_ends[-1]])
        ):
            end = unit_ends[-1]
            start = 0 if len(unit_ends) == 1 else unit_ends[-2]
            source_mass = self._max_source_mass_range(
                provenance_mass,
                start=start,
                end=end,
            )
            last_source_mass = source_mass
            if source_mass is None or source_mass >= threshold:
                break
            ids = ids[:start]
            unit_ends = unit_ends[:-1]
            trimmed_unit_count += 1
        return ids, trimmed_unit_count > 0, last_source_mass, trimmed_unit_count

    def trim_for_source_lookback_holdback(
        self,
        generated_ids: Sequence[int],
        *,
        provenance_mass: Sequence[TokenProvenanceBreakdown] | None,
        aligned_source_local_positions: Sequence[int | None],
        blocked_source_local_position: int | None,
    ) -> tuple[list[int], bool, int | None, int, int | None]:
        ids = list(generated_ids)
        if not ids or not provenance_mass:
            return ids, False, None, 0, None
        if not bool(
            getattr(
                self.runtime_config,
                "translation_alignatt_source_lookback_holdback",
                False,
            )
        ):
            return ids, False, None, 0, None
        if blocked_source_local_position is None:
            return ids, False, None, 0, None
        lookback_units = int(
            getattr(self.runtime_config, "translation_alignatt_source_lookback_units", 2)
        )
        min_source_mass = float(
            getattr(
                self.runtime_config,
                "translation_alignatt_source_lookback_min_source_mass",
                0.05,
            )
        )
        min_source_position = int(
            getattr(
                self.runtime_config,
                "translation_alignatt_source_lookback_min_source_position",
                3,
            )
        )
        cutoff_position = max(0, int(blocked_source_local_position) - lookback_units)
        earliest_trim_start: int | None = None
        earliest_trim_source_position: int | None = None
        unit_ends = self.target_stability_unit_end_token_indices(ids)
        start = 0
        for end in unit_ends:
            if end > len(ids):
                break
            unit_source_positions: list[int] = []
            for token_index in range(start, end):
                if token_index >= len(provenance_mass) or token_index >= len(
                    aligned_source_local_positions
                ):
                    continue
                row = provenance_mass[token_index]
                accessible = self._finite_float(row.source_accessible)
                inaccessible = self._finite_float(row.source_inaccessible)
                if accessible is None or inaccessible is None:
                    continue
                if accessible + inaccessible < min_source_mass:
                    continue
                source_position = aligned_source_local_positions[token_index]
                if source_position is None:
                    continue
                unit_source_positions.append(int(source_position))
            if unit_source_positions:
                unit_max_source_position = max(unit_source_positions)
                if (
                    unit_max_source_position >= min_source_position
                    and unit_max_source_position >= cutoff_position
                ):
                    earliest_trim_start = start
                    earliest_trim_source_position = unit_max_source_position
                    break
            start = end
        if earliest_trim_start is None:
            return ids, False, cutoff_position, 0, None
        trimmed_ids = ids[:earliest_trim_start]
        trimmed_unit_count = max(0, len(unit_ends) - self.count_target_stability_units(trimmed_ids))
        return (
            trimmed_ids,
            trimmed_ids != ids,
            cutoff_position,
            trimmed_unit_count,
            earliest_trim_source_position,
        )

    def should_stop_for_source_regression(
        self,
        *,
        current_source_local_position: int | None,
        max_accepted_source_local_position: int | None,
        accessible_source_token_count: int | None = None,
        source_accessible_mass: float | None = None,
        source_inaccessible_mass: float | None = None,
    ) -> str | None:
        max_regression = int(
            getattr(self.runtime_config, "translation_alignatt_max_source_regression", -1)
        )
        if (
            max_regression < 0
            or current_source_local_position is None
            or max_accepted_source_local_position is None
        ):
            return None
        activation_mode = str(
            getattr(
                self.runtime_config,
                "translation_alignatt_source_regression_activation_mode",
                "always",
            )
        )
        if activation_mode == "frontier_reached":
            if accessible_source_token_count is None:
                return None
            last_accessible_position = max(0, int(accessible_source_token_count) - 1)
            activation_slack = max(0, max_regression) + max(
                0,
                int(
                    getattr(
                        self.runtime_config,
                        "translation_alignatt_source_regression_activation_slack_tokens",
                        0,
                    )
                ),
            )
            if int(max_accepted_source_local_position) < (
                last_accessible_position - activation_slack
            ):
                return None
        if int(current_source_local_position) < (
            int(max_accepted_source_local_position) - max_regression
        ):
            min_inaccessible_mass = float(
                getattr(
                    self.runtime_config,
                    "translation_alignatt_source_regression_min_inaccessible_mass",
                    0.0,
                )
            )
            if min_inaccessible_mass > 0.0:
                inaccessible = self._finite_float(source_inaccessible_mass)
                if inaccessible is None or inaccessible < min_inaccessible_mass:
                    return None
            min_source_mass = float(
                getattr(
                    self.runtime_config,
                    "translation_alignatt_source_regression_min_source_mass",
                    0.0,
                )
            )
            if min_source_mass > 0.0:
                accessible = self._finite_float(source_accessible_mass)
                inaccessible = self._finite_float(source_inaccessible_mass)
                if accessible is not None and inaccessible is not None:
                    if accessible + inaccessible < min_source_mass:
                        return None
            return "source_regression"
        return None

    def _source_regression_unit_evidence(
        self,
        *,
        aligned_source_local_positions: Sequence[int | None],
        provenance_mass: Sequence[TokenProvenanceBreakdown] | None,
        start: int,
        end: int,
    ) -> tuple[int | None, list[int], float | None, float | None]:
        positions: list[int] = []
        accessible_values: list[float] = []
        inaccessible_values: list[float] = []
        for token_index in range(max(0, int(start)), max(0, int(end))):
            if token_index < len(aligned_source_local_positions):
                position = aligned_source_local_positions[token_index]
                if position is not None:
                    positions.append(int(position))
            if provenance_mass is None or token_index >= len(provenance_mass):
                continue
            row = provenance_mass[token_index]
            accessible = self._finite_float(row.source_accessible)
            inaccessible = self._finite_float(row.source_inaccessible)
            if accessible is not None:
                accessible_values.append(accessible)
            if inaccessible is not None:
                inaccessible_values.append(inaccessible)

        unit_position = max(positions) if positions else None
        unit_accessible = max(accessible_values) if accessible_values else None
        unit_inaccessible = max(inaccessible_values) if inaccessible_values else None
        return unit_position, positions, unit_accessible, unit_inaccessible

    def trim_for_source_regression_target_units(
        self,
        generated_ids: Sequence[int],
        *,
        aligned_source_local_positions: Sequence[int | None],
        provenance_mass: Sequence[TokenProvenanceBreakdown] | None,
        source_map: PromptSourceMap | None,
    ) -> tuple[list[int], bool, int, int, str | None, int | None, int | None, int]:
        ids = list(generated_ids)
        max_regression = int(
            getattr(self.runtime_config, "translation_alignatt_max_source_regression", -1)
        )
        if max_regression < 0 or not ids:
            return ids, False, 0, 0, None, None, None, 0

        unit_ends = [
            int(end)
            for end in self.target_stability_unit_end_token_indices(ids)
            if 0 < int(end) <= len(ids)
        ]
        if not unit_ends:
            return ids, False, 0, 0, None, None, None, 0

        accepted_end = 0
        max_accepted_source_local_position: int | None = None
        accepted_source_local_positions: list[int] = []
        regression_streak = 0
        regression_bypassed_count = 0
        pending_trim_end: int | None = None
        pending_trim_reason: str | None = None
        pending_trim_reference_position: int | None = None
        pending_trim_unit_position: int | None = None
        trim_reason: str | None = None
        trim_reference_position: int | None = None
        trim_unit_position: int | None = None
        action = self.source_regression_action()

        for unit_end in unit_ends:
            unit_start = accepted_end
            (
                unit_position,
                unit_positions,
                unit_accessible_mass,
                unit_inaccessible_mass,
            ) = self._source_regression_unit_evidence(
                aligned_source_local_positions=aligned_source_local_positions,
                provenance_mass=provenance_mass,
                start=unit_start,
                end=unit_end,
            )
            reference_position = self.source_regression_reference_position(
                accepted_source_local_positions=accepted_source_local_positions,
                max_accepted_source_local_position=max_accepted_source_local_position,
            )
            reason = self.should_stop_for_source_regression(
                current_source_local_position=unit_position,
                max_accepted_source_local_position=reference_position,
                accessible_source_token_count=(
                    None
                    if source_map is None
                    else source_map.accessible_source_token_count
                ),
                source_accessible_mass=unit_accessible_mass,
                source_inaccessible_mass=unit_inaccessible_mass,
            )
            if reason is not None:
                if pending_trim_end is None:
                    pending_trim_end = accepted_end
                    pending_trim_reason = reason
                    pending_trim_reference_position = reference_position
                    pending_trim_unit_position = unit_position
                regression_streak += 1
                if (
                    action == "trim_target_unit"
                    and regression_streak >= self.source_regression_patience_tokens()
                ):
                    trim_reason = reason
                    trim_reference_position = reference_position
                    trim_unit_position = unit_position
                    accepted_end = pending_trim_end
                    break
                regression_bypassed_count += 1
                accepted_end = unit_end
                continue

            regression_streak = 0
            pending_trim_end = None
            pending_trim_reason = None
            pending_trim_reference_position = None
            pending_trim_unit_position = None
            accepted_end = unit_end
            for position in unit_positions:
                accepted_source_local_positions.append(int(position))
                max_accepted_source_local_position = max(
                    int(position),
                    -1
                    if max_accepted_source_local_position is None
                    else int(max_accepted_source_local_position),
                )

        if (
            action == "trim_unrecovered"
            and pending_trim_end is not None
            and regression_streak >= self.source_regression_patience_tokens()
        ):
            trim_reason = pending_trim_reason
            trim_reference_position = pending_trim_reference_position
            trim_unit_position = pending_trim_unit_position
            accepted_end = pending_trim_end

        trimmed_ids = ids[:accepted_end]
        trimmed = len(trimmed_ids) < len(ids)
        if not trimmed:
            return ids, False, 0, 0, None, None, None, regression_bypassed_count
        trimmed_units = max(
            0,
            len(unit_ends) - self.count_target_stability_units(trimmed_ids),
        )
        return (
            trimmed_ids,
            True,
            len(ids) - len(trimmed_ids),
            trimmed_units,
            trim_reason,
            trim_reference_position,
            trim_unit_position,
            regression_bypassed_count,
        )

    def trim_for_source_frontier_target_units(
        self,
        generated_ids: Sequence[int],
        *,
        aligned_source_local_positions: Sequence[int | None],
        provenance_mass: Sequence[TokenProvenanceBreakdown] | None,
        source_map: PromptSourceMap | None,
    ) -> tuple[list[int], bool, int, int, str | None, int | None, int]:
        ids = list(generated_ids)
        if self.source_frontier_action() != "trim_unrecovered" or not ids:
            return ids, False, 0, 0, None, None, 0
        if source_map is None:
            return ids, False, 0, 0, None, None, 0

        unit_ends = [
            int(end)
            for end in self.target_stability_unit_end_token_indices(ids)
            if 0 < int(end) <= len(ids)
        ]
        if not unit_ends:
            return ids, False, 0, 0, None, None, 0

        accepted_end = 0
        pending_trim_end: int | None = None
        pending_trim_reason: str | None = None
        pending_trim_unit_position: int | None = None
        frontier_bypassed_count = 0
        trim_reason: str | None = None
        trim_unit_position: int | None = None

        for unit_end in unit_ends:
            unit_start = accepted_end
            for token_index in range(unit_start, unit_end):
                current_position = (
                    aligned_source_local_positions[token_index]
                    if token_index < len(aligned_source_local_positions)
                    else None
                )
                source_inaccessible_mass = None
                if provenance_mass is not None and token_index < len(provenance_mass):
                    source_inaccessible_mass = self._finite_float(
                        provenance_mass[token_index].source_inaccessible
                    )
                reason, blocked_position = self.should_stop_in_loop(
                    current_source_local_position=current_position,
                    accessible_source_token_count=(
                        source_map.accessible_source_token_count
                    ),
                    source_inaccessible_mass=source_inaccessible_mass,
                )
                if reason is not None:
                    if pending_trim_end is None:
                        pending_trim_end = accepted_end
                        pending_trim_reason = reason
                        pending_trim_unit_position = blocked_position
                    frontier_bypassed_count += 1
                    continue

                pending_trim_end = None
                pending_trim_reason = None
                pending_trim_unit_position = None
            accepted_end = unit_end

        if pending_trim_end is not None:
            trim_reason = pending_trim_reason
            trim_unit_position = pending_trim_unit_position
            accepted_end = pending_trim_end

        trimmed_ids = ids[:accepted_end]
        trimmed = len(trimmed_ids) < len(ids)
        if not trimmed:
            return ids, False, 0, 0, None, None, frontier_bypassed_count
        trimmed_units = max(
            0,
            len(unit_ends) - self.count_target_stability_units(trimmed_ids),
        )
        return (
            trimmed_ids,
            True,
            len(ids) - len(trimmed_ids),
            trimmed_units,
            trim_reason,
            trim_unit_position,
            frontier_bypassed_count,
        )

    def source_regression_patience_tokens(self) -> int:
        return max(
            1,
            int(
                getattr(
                    self.runtime_config,
                    "translation_alignatt_source_regression_patience_tokens",
                    1,
                )
            ),
        )

    def should_stop_after_source_regression_patience(
        self,
        *,
        current_streak: int,
        source_regression_stop_reason: str | None,
    ) -> tuple[bool, int]:
        if source_regression_stop_reason is None:
            return False, 0
        next_streak = max(0, int(current_streak)) + 1
        return next_streak >= self.source_regression_patience_tokens(), next_streak

    def source_regression_reference_position(
        self,
        *,
        accepted_source_local_positions: Sequence[int],
        max_accepted_source_local_position: int | None,
    ) -> int | None:
        recent_tokens = int(
            getattr(
                self.runtime_config,
                "translation_alignatt_source_regression_recent_tokens",
                0,
            )
        )
        if recent_tokens > 0 and accepted_source_local_positions:
            recent_positions = [
                int(position)
                for position in accepted_source_local_positions[-recent_tokens:]
            ]
            reference_mode = str(
                getattr(
                    self.runtime_config,
                    "translation_alignatt_source_regression_reference_mode",
                    "max",
                )
            )
            if reference_mode == "median_recent":
                return _median_position(recent_positions)
            return max(recent_positions)
        return max_accepted_source_local_position

    def should_stop_for_token_argmax_frontier(
        self,
        *,
        current_source_local_position: int | None,
        accessible_source_token_count: int,
        source_accessible_mass: float | None,
        source_inaccessible_mass: float | None,
    ) -> tuple[str | None, int | None, float | None]:
        if not bool(
            getattr(
                self.runtime_config,
                "translation_alignatt_token_argmax_frontier_gate",
                False,
            )
        ):
            return None, current_source_local_position, None
        if current_source_local_position is None:
            return None, None, None
        accessible = self._finite_float(source_accessible_mass)
        inaccessible = self._finite_float(source_inaccessible_mass)
        if accessible is None or inaccessible is None:
            return None, current_source_local_position, None
        source_mass = accessible + inaccessible
        min_source_mass = float(
            getattr(
                self.runtime_config,
                "translation_alignatt_token_argmax_min_source_mass",
                0.05,
            )
        )
        if source_mass < min_source_mass:
            return None, current_source_local_position, source_mass
        frontier_margin = int(
            getattr(
                self.runtime_config,
                "translation_alignatt_token_argmax_frontier_margin",
                0,
            )
        )
        frontier = max(0, int(accessible_source_token_count)) + max(
            0,
            frontier_margin,
        )
        if int(current_source_local_position) >= frontier:
            return "token_argmax_source_frontier", current_source_local_position, source_mass
        return None, current_source_local_position, source_mass

    def token_argmax_frontier_patience_tokens(self) -> int:
        return max(
            1,
            int(
                getattr(
                    self.runtime_config,
                    "translation_alignatt_token_argmax_frontier_patience_tokens",
                    1,
                )
            ),
        )

    def should_stop_after_token_argmax_frontier_patience(
        self,
        *,
        current_streak: int,
        token_argmax_stop_reason: str | None,
    ) -> tuple[bool, int]:
        if token_argmax_stop_reason is None:
            return False, 0
        next_streak = max(0, int(current_streak)) + 1
        return next_streak >= self.token_argmax_frontier_patience_tokens(), next_streak

    def _unit_argmax_token_safe(
        self,
        *,
        current_source_local_position: int | None,
        source_map: PromptSourceMap,
        source_inaccessible_mass: float | None = None,
    ) -> tuple[bool, str | None, int | None]:
        reason, blocked_position = self.should_stop_in_loop(
            current_source_local_position=current_source_local_position,
            accessible_source_token_count=source_map.accessible_source_token_count,
            source_inaccessible_mass=source_inaccessible_mass,
        )
        return reason is None, reason, blocked_position

    def _unit_consensus_token_safe(
        self,
        *,
        per_head_source_local_positions: Sequence[int] | None = None,
        source_attention_rows: torch.Tensor | None = None,
        source_map: PromptSourceMap,
    ) -> tuple[bool, str | None, int | None, float | None]:
        if per_head_source_local_positions is not None:
            per_head_positions = torch.tensor(
                [int(position) for position in per_head_source_local_positions],
                dtype=torch.long,
            )
            if per_head_positions.numel() <= 0:
                return False, "attention_missing", None, None
        else:
            if source_attention_rows is None:
                return False, "attention_missing", None, None
            if source_attention_rows.ndim != 2 or source_attention_rows.shape[-1] <= 0:
                return False, "attention_missing", None, None
            rows = source_attention_rows.detach().float()
            if not bool(torch.isfinite(rows).all().item()):
                return False, "attention_nonfinite", None, None
            smoothed_rows = median_filter_last_dim(rows, self.alignatt_filter_width())
            per_head_positions = torch.argmax(smoothed_rows, dim=-1).to(torch.long)
        border_margin = int(
            getattr(self.runtime_config, "translation_alignatt_border_margin", 0)
        )
        frontier = max(0, int(source_map.accessible_source_token_count)) + border_margin
        within_frontier = per_head_positions < int(frontier)
        min_ratio = float(
            getattr(
                self.runtime_config,
                "translation_alignatt_unit_consensus_min_head_ratio",
                0.60,
            )
        )
        ratio = float(within_frontier.float().mean().item())
        if ratio < min_ratio:
            blocked_position = int(per_head_positions.max().item())
            return False, "head_consensus_frontier", blocked_position, ratio
        consensus_positions = [
            int(position)
            for position, is_within in zip(
                per_head_positions.tolist(),
                within_frontier.tolist(),
            )
            if bool(is_within)
        ]
        if not consensus_positions:
            return False, "head_consensus_frontier", None, ratio
        return True, None, int(max(consensus_positions)), ratio

    def _unit_conf_token_safe(
        self,
        *,
        current_source_local_position: int | None,
        per_head_source_local_positions: Sequence[int] | None,
        source_map: PromptSourceMap,
        source_inaccessible_mass: float | None = None,
    ) -> tuple[bool, str | None, int | None, float | None]:
        """Argmax frontier plus alignment-confidence gate for ``unit_conf``.

        The frontier decision is exactly ``_unit_argmax_token_safe`` (soft
        frontier and border margin included). On top of it, when
        ``translation_alignatt_min_alignment_confidence`` is positive, the
        token's head-agreement ratio must reach the floor; a dispersed head
        vote marks the alignment as unsettled and defers the unit. This is a
        different statistic from ``_unit_consensus_token_safe`` (which counts
        heads voting inside the frontier, not heads agreeing with each other).
        """
        token_safe, reason, blocked_position = self._unit_argmax_token_safe(
            current_source_local_position=current_source_local_position,
            source_map=source_map,
            source_inaccessible_mass=source_inaccessible_mass,
        )
        if not token_safe:
            return False, reason, blocked_position, None
        confidence = head_agreement_ratio(
            per_head_source_local_positions,
            current_source_local_position,
        )
        min_confidence = float(
            getattr(
                self.runtime_config,
                "translation_alignatt_min_alignment_confidence",
                0.0,
            )
        )
        if min_confidence > 0.0:
            if confidence is None:
                return False, "attention_missing", current_source_local_position, None
            if confidence < min_confidence:
                return (
                    False,
                    "unit_confidence_weak",
                    current_source_local_position,
                    confidence,
                )
        return True, None, current_source_local_position, confidence

    def accept_complete_target_units(
        self,
        *,
        generated_ids: Sequence[int],
        aligned_source_local_positions: Sequence[int | None],
        source_attention_rows: Sequence[torch.Tensor] | None,
        provenance_mass: Sequence[TokenProvenanceBreakdown] | None,
        source_map: PromptSourceMap,
        finish_reason: str | int | None,
        per_head_aligned_source_local_positions: Sequence[Sequence[int]] | None = None,
    ) -> AlignAttUnitPolicyDecision:
        """Accept the longest target-unit prefix justified by AlignAtt evidence."""
        variant = self.alignatt_acceptance_variant()
        ids = list(generated_ids)
        unit_end_indices = [
            int(end)
            for end in self.target_stability_unit_end_token_indices(ids)
            if 0 < int(end) <= len(ids)
        ]
        metadata: dict[str, Any] = {
            "alignatt_acceptance_variant": variant,
            "alignatt_unit_policy_complete_unit_end_token_indices": unit_end_indices,
            "alignatt_unit_policy_complete_unit_count": len(unit_end_indices),
            "alignatt_unit_policy_accepted_unit_count": 0,
            "alignatt_unit_policy_stop_reason": None,
            "alignatt_unit_policy_stop_token_index": None,
            "alignatt_unit_policy_border_margin": int(
                getattr(self.runtime_config, "translation_alignatt_border_margin", 0)
            ),
            "alignatt_unit_consensus_min_head_ratio": float(
                getattr(
                    self.runtime_config,
                    "translation_alignatt_unit_consensus_min_head_ratio",
                    0.60,
                )
            ),
            "alignatt_unit_consensus_normalization": str(
                getattr(
                    self.runtime_config,
                    "translation_alignatt_online_normalization",
                    "zscore",
                )
            ),
            "alignatt_source_bearing_min_source_mass": float(
                getattr(
                    self.runtime_config,
                    "translation_alignatt_source_bearing_min_source_mass",
                    0.05,
                )
            ),
            "alignatt_source_bearing_hard_inaccessible_cap": float(
                getattr(
                    self.runtime_config,
                    "translation_alignatt_source_bearing_hard_inaccessible_cap",
                    1.0,
                )
            ),
        }
        metadata["alignatt_min_alignment_confidence"] = float(
            getattr(
                self.runtime_config,
                "translation_alignatt_min_alignment_confidence",
                0.0,
            )
        )
        if variant not in {
            "unit_mass",
            "unit_mass_source_bearing",
            "unit_argmax",
            "unit_consensus",
            "unit_conf",
        }:
            raise ValueError(f"Not a unit AlignAtt acceptance variant: {variant!r}")
        if bool(source_map.is_final):
            metadata["alignatt_unit_policy_accepted_unit_count"] = len(unit_end_indices)
            return AlignAttUnitPolicyDecision(
                accepted_candidate_ids=ids,
                unsafe_reason=None,
                unsafe_target_token_index=None,
                unsafe_token_id=None,
                blocked_source_local_position=None,
                blocked_source_unit_index=None,
                stop_reason=finish_reason,
                metadata=metadata,
            )
        if not ids or not unit_end_indices:
            reason = "target_unit_incomplete"
            metadata["alignatt_unit_policy_stop_reason"] = reason
            return AlignAttUnitPolicyDecision(
                accepted_candidate_ids=[],
                unsafe_reason=reason,
                unsafe_target_token_index=0 if ids else None,
                unsafe_token_id=int(ids[0]) if ids else None,
                blocked_source_local_position=None,
                blocked_source_unit_index=None,
                stop_reason=f"alignatt:{reason}",
                metadata=metadata,
            )

        accepted_end = 0
        unsafe_reason: str | None = None
        unsafe_target_token_index: int | None = None
        unsafe_token_id: int | None = None
        blocked_source_local_position: int | None = None
        blocked_source_unit_index: int | None = None
        last_safe_source_position: int | None = None
        last_consensus_ratio: float | None = None
        last_unit_confidence: float | None = None
        unit_conf_unit_confidences: list[float | None] = []

        for unit_index, unit_end in enumerate(unit_end_indices):
            unit_start = accepted_end
            unit_safe = True
            unit_reason: str | None = None
            unit_blocked_position: int | None = None
            unit_failed_token_index: int | None = None
            unit_source_bearing_accessible: list[float] = []
            unit_source_bearing_inaccessible: list[float] = []
            unit_min_confidence: float | None = None
            for token_index in range(unit_start, unit_end):
                source_inaccessible_mass: float | None = None
                if provenance_mass is not None and token_index < len(provenance_mass):
                    row = provenance_mass[token_index]
                    if self._provenance_has_nonfinite(row):
                        unit_safe = False
                        unit_reason = "provenance_nonfinite"
                        unit_failed_token_index = token_index
                        break
                    source_inaccessible_mass = float(row.source_inaccessible)
                    if variant == "unit_mass_source_bearing":
                        source_accessible_mass = float(row.source_accessible)
                        source_mass = source_accessible_mass + source_inaccessible_mass
                        min_source_bearing_mass = float(
                            getattr(
                                self.runtime_config,
                                "translation_alignatt_source_bearing_min_source_mass",
                                0.05,
                            )
                        )
                        if source_mass < min_source_bearing_mass:
                            continue
                        hard_cap = float(
                            getattr(
                                self.runtime_config,
                                "translation_alignatt_source_bearing_hard_inaccessible_cap",
                                1.0,
                            )
                        )
                        if source_inaccessible_mass > hard_cap:
                            unit_safe = False
                            unit_reason = "source_bearing_inaccessible_hard_cap"
                            unit_failed_token_index = token_index
                            break
                        mass_reason = self.should_stop_for_provenance_mass(
                            source_accessible_mass=source_accessible_mass,
                            source_inaccessible_mass=source_inaccessible_mass,
                            non_source_prompt_mass=float(row.non_source_prompt),
                        )
                        if mass_reason is not None:
                            unit_safe = False
                            unit_reason = mass_reason
                            unit_failed_token_index = token_index
                            break
                        current_source_position = (
                            aligned_source_local_positions[token_index]
                            if token_index < len(aligned_source_local_positions)
                            else None
                        )
                        if current_source_position is None:
                            unit_safe = False
                            unit_reason = "attention_missing"
                            unit_failed_token_index = token_index
                            break
                        token_safe, reason, blocked_position = self._unit_argmax_token_safe(
                            current_source_local_position=current_source_position,
                            source_map=source_map,
                            source_inaccessible_mass=source_inaccessible_mass,
                        )
                        if not token_safe:
                            unit_safe = False
                            unit_reason = reason
                            unit_blocked_position = blocked_position
                            unit_failed_token_index = token_index
                            break
                        last_safe_source_position = current_source_position
                        unit_source_bearing_accessible.append(source_accessible_mass)
                        unit_source_bearing_inaccessible.append(source_inaccessible_mass)
                        continue
                    if variant == "unit_mass":
                        source_accessible_mass = float(row.source_accessible)
                        min_source_mass = float(
                            getattr(
                                self.runtime_config,
                                "translation_alignatt_min_source_mass",
                                0.0,
                            )
                        )
                        if (
                            min_source_mass > 0.0
                            and source_accessible_mass < min_source_mass
                        ):
                            unit_safe = False
                            unit_reason = "provenance_weak"
                            unit_failed_token_index = token_index
                            break
                        mass_reason = self.should_stop_for_provenance_mass(
                            source_accessible_mass=source_accessible_mass,
                            source_inaccessible_mass=source_inaccessible_mass,
                            non_source_prompt_mass=float(row.non_source_prompt),
                        )
                        if mass_reason is not None:
                            unit_safe = False
                            unit_reason = mass_reason
                            unit_failed_token_index = token_index
                            break
                elif variant in {"unit_mass", "unit_mass_source_bearing"}:
                    unit_safe = False
                    unit_reason = "provenance_missing"
                    unit_failed_token_index = token_index
                    break

                if variant in {"unit_mass", "unit_mass_source_bearing"}:
                    continue
                if variant == "unit_argmax":
                    current_source_position = (
                        aligned_source_local_positions[token_index]
                        if token_index < len(aligned_source_local_positions)
                        else None
                    )
                    token_safe, reason, blocked_position = self._unit_argmax_token_safe(
                        current_source_local_position=current_source_position,
                        source_map=source_map,
                        source_inaccessible_mass=source_inaccessible_mass,
                    )
                    if not token_safe:
                        unit_safe = False
                        unit_reason = reason
                        unit_blocked_position = blocked_position
                        unit_failed_token_index = token_index
                        break
                    last_safe_source_position = current_source_position
                    continue
                if variant == "unit_consensus":
                    per_head_positions = None
                    if (
                        per_head_aligned_source_local_positions is not None
                        and token_index < len(per_head_aligned_source_local_positions)
                    ):
                        per_head_positions = per_head_aligned_source_local_positions[
                            token_index
                        ]
                    token_source_attention_rows = None
                    if (
                        source_attention_rows is not None
                        and token_index < len(source_attention_rows)
                    ):
                        token_source_attention_rows = source_attention_rows[token_index]
                    if per_head_positions is None and token_source_attention_rows is None:
                        unit_safe = False
                        unit_reason = "attention_missing"
                        unit_failed_token_index = token_index
                        break
                    token_safe, reason, blocked_position, consensus_ratio = (
                        self._unit_consensus_token_safe(
                            per_head_source_local_positions=per_head_positions,
                            source_attention_rows=token_source_attention_rows,
                            source_map=source_map,
                        )
                    )
                    last_consensus_ratio = consensus_ratio
                    if not token_safe:
                        unit_safe = False
                        unit_reason = reason
                        unit_blocked_position = blocked_position
                        unit_failed_token_index = token_index
                        break
                    last_safe_source_position = blocked_position
                    continue
                if variant == "unit_conf":
                    current_source_position = (
                        aligned_source_local_positions[token_index]
                        if token_index < len(aligned_source_local_positions)
                        else None
                    )
                    per_head_positions = None
                    if (
                        per_head_aligned_source_local_positions is not None
                        and token_index < len(per_head_aligned_source_local_positions)
                    ):
                        per_head_positions = per_head_aligned_source_local_positions[
                            token_index
                        ]
                    token_safe, reason, blocked_position, confidence = (
                        self._unit_conf_token_safe(
                            current_source_local_position=current_source_position,
                            per_head_source_local_positions=per_head_positions,
                            source_map=source_map,
                            source_inaccessible_mass=source_inaccessible_mass,
                        )
                    )
                    if confidence is not None:
                        unit_min_confidence = (
                            confidence
                            if unit_min_confidence is None
                            else min(unit_min_confidence, confidence)
                        )
                    if not token_safe:
                        unit_safe = False
                        unit_reason = reason
                        unit_blocked_position = blocked_position
                        unit_failed_token_index = token_index
                        break
                    last_safe_source_position = current_source_position

            if unit_safe and variant == "unit_conf":
                unit_conf_unit_confidences.append(unit_min_confidence)
                last_unit_confidence = unit_min_confidence

            if unit_safe and variant == "unit_mass_source_bearing":
                source_bearing_count = len(unit_source_bearing_accessible)
                metadata["alignatt_source_bearing_last_unit_token_count"] = (
                    source_bearing_count
                )
                if source_bearing_count > 0:
                    mean_accessible = (
                        sum(unit_source_bearing_accessible) / float(source_bearing_count)
                    )
                    mean_inaccessible = (
                        sum(unit_source_bearing_inaccessible)
                        / float(source_bearing_count)
                    )
                    metadata["alignatt_source_bearing_last_unit_mean_accessible"] = (
                        mean_accessible
                    )
                    metadata["alignatt_source_bearing_last_unit_mean_inaccessible"] = (
                        mean_inaccessible
                    )
                    max_inaccessible = float(
                        getattr(
                            self.runtime_config,
                            "translation_alignatt_max_inaccessible_source_mass",
                            1.0,
                        )
                    )
                    if max_inaccessible < 1.0 and mean_inaccessible > max_inaccessible:
                        unit_safe = False
                        unit_reason = "source_bearing_inaccessible_high"
                        unit_failed_token_index = unit_start
                    else:
                        min_margin = float(
                            getattr(
                                self.runtime_config,
                                "translation_alignatt_min_accessible_inaccessible_margin",
                                -1.0,
                            )
                        )
                        if (
                            min_margin > -1.0
                            and mean_accessible - mean_inaccessible < min_margin
                        ):
                            unit_safe = False
                            unit_reason = "source_bearing_margin_weak"
                            unit_failed_token_index = unit_start
                else:
                    metadata["alignatt_source_bearing_last_unit_mean_accessible"] = None
                    metadata["alignatt_source_bearing_last_unit_mean_inaccessible"] = None
                    unit_safe = False
                    unit_reason = "source_bearing_missing"
                    unit_failed_token_index = unit_start

            if not unit_safe:
                unsafe_reason = unit_reason or "unit_policy_blocked"
                unsafe_target_token_index = (
                    unit_failed_token_index
                    if unit_failed_token_index is not None
                    else unit_start
                )
                unsafe_token_id = (
                    int(ids[unsafe_target_token_index])
                    if 0 <= unsafe_target_token_index < len(ids)
                    else None
                )
                blocked_source_local_position = unit_blocked_position
                if (
                    blocked_source_local_position is None
                    and unsafe_target_token_index is not None
                    and unsafe_target_token_index < len(aligned_source_local_positions)
                ):
                    blocked_source_local_position = aligned_source_local_positions[
                        unsafe_target_token_index
                    ]
                blocked_source_unit_index = source_local_position_to_unit_index(
                    source_map,
                    blocked_source_local_position,
                )
                metadata["alignatt_unit_policy_stop_reason"] = unsafe_reason
                metadata["alignatt_unit_policy_stop_token_index"] = (
                    unsafe_target_token_index
                )
                break

            accepted_end = unit_end
            metadata["alignatt_unit_policy_accepted_unit_count"] = unit_index + 1

        if unsafe_reason is None and accepted_end < len(ids):
            unsafe_reason = "target_unit_incomplete"
            unsafe_target_token_index = accepted_end
            unsafe_token_id = int(ids[accepted_end]) if accepted_end < len(ids) else None
            metadata["alignatt_unit_policy_stop_reason"] = unsafe_reason
            metadata["alignatt_unit_policy_stop_token_index"] = unsafe_target_token_index

        metadata["alignatt_unit_policy_last_safe_source_position"] = (
            last_safe_source_position
        )
        metadata["alignatt_unit_policy_last_consensus_ratio"] = last_consensus_ratio
        metadata["alignatt_unit_policy_last_unit_confidence"] = last_unit_confidence
        metadata["alignatt_unit_conf_unit_confidences"] = unit_conf_unit_confidences
        return AlignAttUnitPolicyDecision(
            accepted_candidate_ids=ids[:accepted_end],
            unsafe_reason=unsafe_reason,
            unsafe_target_token_index=unsafe_target_token_index,
            unsafe_token_id=unsafe_token_id,
            blocked_source_local_position=blocked_source_local_position,
            blocked_source_unit_index=blocked_source_unit_index,
            stop_reason=(
                f"alignatt:{unsafe_reason}"
                if unsafe_reason is not None
                else finish_reason
            ),
            metadata=metadata,
        )

    @staticmethod
    def generation_completed_normally(stop_reason: str | int | None) -> bool:
        if not isinstance(stop_reason, str):
            return False
        return stop_reason.strip().lower() in {
            "stop",
            "eos",
            "eos_token",
            "end_of_turn",
        }

    def should_bypass_alignatt_for_final_source(
        self,
        *,
        source_map: PromptSourceMap | None,
        stop_reason: str | int | None,
    ) -> bool:
        del stop_reason
        return source_map is not None and bool(source_map.is_final)

    def finalize_partial(
        self,
        *,
        accepted_candidate_ids: Sequence[int],
        aligned_source_local_positions: Sequence[int | None],
        source_map: PromptSourceMap | None,
        unsafe_reason: str | None,
        unsafe_target_token_index: int | None,
        unsafe_token_id: int | None,
        blocked_source_local_position: int | None,
        blocked_source_unit_index: int | None,
        stop_reason: str | int | None,
        probe_backend: str | None,
        provenance_mass: Sequence[TokenProvenanceBreakdown] | None = None,
        extra_alignatt_metadata: Mapping[str, Any] | None = None,
    ) -> AlignAttAcceptance:
        final_source_completed = self.should_bypass_alignatt_for_final_source(
            source_map=source_map,
            stop_reason=stop_reason,
        )
        if final_source_completed:
            unsafe_reason = None
            unsafe_target_token_index = None
            unsafe_token_id = None
            blocked_source_local_position = None
            blocked_source_unit_index = None
        unsafe_token_starts_new_unit = False
        unsafe_token_closes_stability_unit = False
        if unsafe_token_id is not None:
            unsafe_token = str(
                self.tokenizer.convert_ids_to_tokens([int(unsafe_token_id)])[0]
            )
            unsafe_token_starts_new_unit = self.token_starts_stability_unit(
                unsafe_token
            )
            unsafe_token_closes_stability_unit = self.token_closes_target_stability_unit(
                accepted_candidate_ids=accepted_candidate_ids,
                next_token_id=int(unsafe_token_id),
            )
        if final_source_completed:
            trimmed_generated_ids = list(accepted_candidate_ids)
        elif unsafe_token_starts_new_unit or unsafe_token_closes_stability_unit:
            trimmed_generated_ids = list(accepted_candidate_ids)
        else:
            trimmed_generated_ids = self.trim_to_last_stability_unit(
                accepted_candidate_ids
            )
        boundary_trimmed_ids = list(trimmed_generated_ids)
        source_frontier_action = self.source_frontier_action()
        source_frontier_trimmed = False
        source_frontier_trimmed_token_count = 0
        source_frontier_trimmed_unit_count = 0
        source_frontier_trim_reason: str | None = None
        source_frontier_trim_unit_position: int | None = None
        source_frontier_trim_bypassed_count = 0
        if not final_source_completed and source_frontier_action == "trim_unrecovered":
            (
                trimmed_generated_ids,
                source_frontier_trimmed,
                source_frontier_trimmed_token_count,
                source_frontier_trimmed_unit_count,
                source_frontier_trim_reason,
                source_frontier_trim_unit_position,
                source_frontier_trim_bypassed_count,
            ) = self.trim_for_source_frontier_target_units(
                trimmed_generated_ids,
                aligned_source_local_positions=aligned_source_local_positions,
                provenance_mass=provenance_mass,
                source_map=source_map,
            )
        source_frontier_trimmed_ids = list(trimmed_generated_ids)
        source_regression_action = self.source_regression_action()
        source_regression_trimmed = False
        source_regression_trimmed_token_count = 0
        source_regression_trimmed_unit_count = 0
        source_regression_trim_reason: str | None = None
        source_regression_trim_reference_position: int | None = None
        source_regression_trim_unit_position: int | None = None
        source_regression_trim_bypassed_count = 0
        if (
            not final_source_completed
            and source_regression_action in {"trim_target_unit", "trim_unrecovered"}
        ):
            (
                trimmed_generated_ids,
                source_regression_trimmed,
                source_regression_trimmed_token_count,
                source_regression_trimmed_unit_count,
                source_regression_trim_reason,
                source_regression_trim_reference_position,
                source_regression_trim_unit_position,
                source_regression_trim_bypassed_count,
            ) = self.trim_for_source_regression_target_units(
                trimmed_generated_ids,
                aligned_source_local_positions=aligned_source_local_positions,
                provenance_mass=provenance_mass,
                source_map=source_map,
            )
        source_regression_trimmed_ids = list(trimmed_generated_ids)
        (
            trimmed_generated_ids,
            source_lookback_trimmed,
            source_lookback_cutoff_position,
            source_lookback_trimmed_unit_count,
            source_lookback_trim_source_position,
        ) = (
            self.trim_for_source_lookback_holdback(
                trimmed_generated_ids,
                provenance_mass=provenance_mass,
                aligned_source_local_positions=aligned_source_local_positions,
                blocked_source_local_position=blocked_source_local_position,
            )
            if not final_source_completed
            else (list(trimmed_generated_ids), False, None, 0, None)
        )
        source_lookback_trimmed_ids = list(trimmed_generated_ids)
        hold_back_units = int(
            getattr(self.runtime_config, "translation_alignatt_hold_back_target_units", 0)
        )
        if hold_back_units > 0 and not final_source_completed:
            trimmed_generated_ids = self.cut_last_target_stability_units(
                trimmed_generated_ids,
                cutoff_units=hold_back_units,
            )
        hold_back_trimmed = list(trimmed_generated_ids) != source_lookback_trimmed_ids
        (
            trimmed_generated_ids,
            terminal_punctuation_trimmed,
            terminal_punctuation_source_mass,
            terminal_punctuation_trimmed_unit_count,
        ) = (
            self.trim_low_source_terminal_punctuation(
                trimmed_generated_ids,
                provenance_mass=provenance_mass,
            )
            if not final_source_completed
            else (list(trimmed_generated_ids), False, None, 0)
        )
        min_emit_units = int(
            getattr(self.runtime_config, "translation_alignatt_min_emit_target_units", 0)
        )
        emitted_unit_count = self.count_target_stability_units(trimmed_generated_ids)
        min_emit_blocked = (
            min_emit_units > 0
            and not final_source_completed
            and 0 < emitted_unit_count < min_emit_units
        )
        if min_emit_blocked:
            trimmed_generated_ids = []
        min_accessible_source_units = int(
            getattr(
                self.runtime_config,
                "translation_alignatt_min_accessible_source_units",
                0,
            )
        )
        source_context_mode = str(
            getattr(
                self.runtime_config,
                "translation_alignatt_min_accessible_source_units_mode",
                "block",
            )
        )
        accessible_source_unit_count = (
            0 if source_map is None else int(source_map.accessible_unit_count)
        )
        source_context_under_min = (
            min_accessible_source_units > 0
            and not final_source_completed
            and accessible_source_unit_count < min_accessible_source_units
        )
        source_context_blocked = source_context_under_min and source_context_mode == "block"
        source_context_cap_applied = False
        source_context_cap_target_units: int | None = None
        if source_context_blocked:
            trimmed_generated_ids = []
        elif source_context_under_min and source_context_mode == "target_unit_cap":
            source_context_cap_target_units = accessible_source_unit_count
            capped_generated_ids = self.keep_first_target_stability_units(
                trimmed_generated_ids,
                max_units=source_context_cap_target_units,
            )
            source_context_cap_applied = list(capped_generated_ids) != list(
                trimmed_generated_ids
            )
            trimmed_generated_ids = capped_generated_ids
        accepted_prefix_mean_accessible_mass: float | None = None
        accepted_prefix_recent_mean_accessible_mass: float | None = None
        accepted_prefix_recent_min_unit_accessible_mass: float | None = None
        accepted_prefix_provenance_trimmed = False
        if (
            not final_source_completed
            and not min_emit_blocked
            and not source_context_blocked
        ):
            (
                trimmed_generated_ids,
                accepted_prefix_mean_accessible_mass,
                accepted_prefix_recent_mean_accessible_mass,
                accepted_prefix_recent_min_unit_accessible_mass,
                accepted_prefix_provenance_trimmed,
            ) = self.trim_for_accepted_prefix_provenance(
                trimmed_generated_ids,
                provenance_mass=provenance_mass,
            )
        word_boundary_trimmed = boundary_trimmed_ids != list(accepted_candidate_ids)
        alignatt_metadata = {
            "source_token_count": 0 if source_map is None else len(source_map.source_token_positions),
            "source_unit_count": 0 if source_map is None else source_map.total_unit_count,
            "accessible_source_unit_count": accessible_source_unit_count,
            "accessible_source_local_end_exclusive": 0
            if source_map is None
            else source_map.accessible_source_token_count,
            "aligned_source_local_positions": list(aligned_source_local_positions),
            "aligned_source_unit_indices": [
                source_local_position_to_unit_index(source_map, position)
                for position in aligned_source_local_positions
            ],
            "alignatt_acceptance_variant": self.alignatt_acceptance_variant(),
            "alignatt_online_normalization": str(
                getattr(
                    self.runtime_config,
                    "translation_alignatt_online_normalization",
                    "zscore",
                )
            ),
            "alignatt_frontier_min_inaccessible_mass": float(
                getattr(
                    self.runtime_config,
                    "translation_alignatt_frontier_min_inaccessible_mass",
                    0.0,
                )
            ),
            "alignatt_source_frontier_action": source_frontier_action,
            "alignatt_source_frontier_trimmed": source_frontier_trimmed,
            "alignatt_source_frontier_trimmed_token_count": (
                source_frontier_trimmed_token_count
            ),
            "alignatt_source_frontier_trimmed_unit_count": (
                source_frontier_trimmed_unit_count
            ),
            "alignatt_source_frontier_trim_reason": source_frontier_trim_reason,
            "alignatt_source_frontier_trim_unit_position": (
                source_frontier_trim_unit_position
            ),
            "alignatt_source_frontier_trim_bypassed_count": (
                source_frontier_trim_bypassed_count
            ),
            "alignatt_source_frontier_trimmed_before_regression": (
                source_frontier_trimmed_ids != boundary_trimmed_ids
            ),
            "alignatt_max_inaccessible_source_mass": float(
                getattr(
                    self.runtime_config,
                    "translation_alignatt_max_inaccessible_source_mass",
                    1.0,
                )
            ),
            "alignatt_max_non_source_prompt_mass": float(
                getattr(
                    self.runtime_config,
                    "translation_alignatt_max_non_source_prompt_mass",
                    1.0,
                )
            ),
            "alignatt_min_accessible_inaccessible_margin": float(
                getattr(
                    self.runtime_config,
                    "translation_alignatt_min_accessible_inaccessible_margin",
                    -1.0,
                )
            ),
            "alignatt_min_accepted_accessible_source_mass": float(
                getattr(
                    self.runtime_config,
                    "translation_alignatt_min_accepted_accessible_source_mass",
                    0.0,
                )
            ),
            "alignatt_accepted_accessible_source_mass_recent_units": int(
                getattr(
                    self.runtime_config,
                    "translation_alignatt_accepted_accessible_source_mass_recent_units",
                    2,
                )
            ),
            "alignatt_accepted_prefix_mean_source_accessible_mass": (
                accepted_prefix_mean_accessible_mass
            ),
            "alignatt_accepted_prefix_recent_mean_source_accessible_mass": (
                accepted_prefix_recent_mean_accessible_mass
            ),
            "alignatt_accepted_prefix_recent_min_unit_source_accessible_mass": (
                accepted_prefix_recent_min_unit_accessible_mass
            ),
            "alignatt_accepted_prefix_provenance_trimmed": (
                accepted_prefix_provenance_trimmed
            ),
            "alignatt_min_accessible_source_units": min_accessible_source_units,
            "alignatt_min_accessible_source_units_mode": source_context_mode,
            "alignatt_source_context_accessible_units": accessible_source_unit_count,
            "alignatt_source_context_under_min": source_context_under_min,
            "alignatt_source_context_blocked": source_context_blocked,
            "alignatt_source_context_cap_applied": source_context_cap_applied,
            "alignatt_source_context_cap_target_units": source_context_cap_target_units,
            "unsafe_target_token_index": unsafe_target_token_index,
            "unsafe_reason": unsafe_reason,
            "unsafe_token_id": unsafe_token_id,
            "unsafe_token_starts_new_unit": unsafe_token_starts_new_unit,
            "unsafe_token_closes_stability_unit": unsafe_token_closes_stability_unit,
            "blocked_source_local_position": blocked_source_local_position,
            "blocked_source_unit_index": blocked_source_unit_index,
            "accepted_candidate_token_count": len(accepted_candidate_ids),
            "accepted_token_count": len(trimmed_generated_ids),
            "target_stability_unit_end_token_indices": (
                self.target_stability_unit_end_token_indices(accepted_candidate_ids)
            ),
            "accepted_target_stability_unit_count": (
                self.count_target_stability_units(trimmed_generated_ids)
            ),
            "word_boundary_trimmed": word_boundary_trimmed,
            "alignatt_source_regression_action": source_regression_action,
            "alignatt_source_regression_trimmed": source_regression_trimmed,
            "alignatt_source_regression_trimmed_token_count": (
                source_regression_trimmed_token_count
            ),
            "alignatt_source_regression_trimmed_unit_count": (
                source_regression_trimmed_unit_count
            ),
            "alignatt_source_regression_trim_reason": (
                source_regression_trim_reason
            ),
            "alignatt_source_regression_trim_reference_position": (
                source_regression_trim_reference_position
            ),
            "alignatt_source_regression_trim_unit_position": (
                source_regression_trim_unit_position
            ),
            "alignatt_source_regression_trim_bypassed_count": (
                source_regression_trim_bypassed_count
            ),
            "alignatt_source_regression_trimmed_before_lookback": (
                source_regression_trimmed_ids
                != boundary_trimmed_ids
            ),
            "alignatt_source_lookback_holdback": bool(
                getattr(
                    self.runtime_config,
                    "translation_alignatt_source_lookback_holdback",
                    False,
                )
            ),
            "alignatt_source_lookback_units": int(
                getattr(
                    self.runtime_config,
                    "translation_alignatt_source_lookback_units",
                    2,
                )
            ),
            "alignatt_source_lookback_min_source_mass": float(
                getattr(
                    self.runtime_config,
                    "translation_alignatt_source_lookback_min_source_mass",
                    0.05,
                )
            ),
            "alignatt_source_lookback_min_source_position": int(
                getattr(
                    self.runtime_config,
                    "translation_alignatt_source_lookback_min_source_position",
                    3,
                )
            ),
            "alignatt_source_lookback_trimmed": source_lookback_trimmed,
            "alignatt_source_lookback_cutoff_position": (
                source_lookback_cutoff_position
            ),
            "alignatt_source_lookback_trimmed_unit_count": (
                source_lookback_trimmed_unit_count
            ),
            "alignatt_source_lookback_trim_source_position": (
                source_lookback_trim_source_position
            ),
            "alignatt_hold_back_target_units": hold_back_units,
            "alignatt_hold_back_trimmed": hold_back_trimmed,
            "alignatt_token_argmax_frontier_gate": bool(
                getattr(
                    self.runtime_config,
                    "translation_alignatt_token_argmax_frontier_gate",
                    False,
                )
            ),
            "alignatt_token_argmax_min_source_mass": float(
                getattr(
                    self.runtime_config,
                    "translation_alignatt_token_argmax_min_source_mass",
                    0.05,
                )
            ),
            "alignatt_token_argmax_frontier_margin": int(
                getattr(
                    self.runtime_config,
                    "translation_alignatt_token_argmax_frontier_margin",
                    0,
                )
            ),
            "alignatt_token_argmax_frontier_patience_tokens": (
                self.token_argmax_frontier_patience_tokens()
            ),
            "alignatt_defer_low_source_terminal_punctuation": bool(
                getattr(
                    self.runtime_config,
                    "translation_alignatt_defer_low_source_terminal_punctuation",
                    False,
                )
            ),
            "alignatt_terminal_punctuation_min_source_mass": float(
                self.terminal_punctuation_min_source_mass()
            ),
            "alignatt_terminal_punctuation_trimmed": terminal_punctuation_trimmed,
            "alignatt_terminal_punctuation_source_mass": (
                terminal_punctuation_source_mass
            ),
            "alignatt_terminal_punctuation_trimmed_unit_count": (
                terminal_punctuation_trimmed_unit_count
            ),
            "alignatt_min_emit_target_units": min_emit_units,
            "alignatt_emitted_target_unit_count": emitted_unit_count,
            "alignatt_min_emit_blocked": min_emit_blocked,
            "alignatt_max_source_regression": int(
                getattr(
                    self.runtime_config,
                    "translation_alignatt_max_source_regression",
                    -1,
                )
            ),
            "alignatt_source_regression_min_source_mass": float(
                getattr(
                    self.runtime_config,
                    "translation_alignatt_source_regression_min_source_mass",
                    0.0,
                )
            ),
            "alignatt_source_regression_recent_tokens": int(
                getattr(
                    self.runtime_config,
                    "translation_alignatt_source_regression_recent_tokens",
                    0,
                )
            ),
            "alignatt_source_regression_activation_mode": str(
                getattr(
                    self.runtime_config,
                    "translation_alignatt_source_regression_activation_mode",
                    "always",
                )
            ),
            "alignatt_source_regression_activation_slack_tokens": int(
                getattr(
                    self.runtime_config,
                    "translation_alignatt_source_regression_activation_slack_tokens",
                    0,
                )
            ),
            "alignatt_source_regression_patience_tokens": (
                self.source_regression_patience_tokens()
            ),
            "final_source_completed_full_accept": final_source_completed,
            "stop_reason": stop_reason,
            "acceptance_policy": getattr(
                self.runtime_config,
                "translation_acceptance_policy",
                "alignatt",
            ),
            "current_source_ms": None if source_map is None else source_map.current_source_ms,
            "inaccessible_ms": None if source_map is None else source_map.inaccessible_ms,
            "probe_mode": "prefix_online_batched",
            "probe_backend": probe_backend,
        }
        if extra_alignatt_metadata:
            alignatt_metadata.update(dict(extra_alignatt_metadata))
        return AlignAttAcceptance(
            accepted_generated_ids=trimmed_generated_ids,
            alignatt_metadata=alignatt_metadata,
        )


def load_alignatt_heads(path: str, *, top_k: int) -> list[AlignAttHead]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        AlignAttHead(
            layer=int(head["layer"]),
            head=int(head["head"]),
            ts=float(head["ts"]),
        )
        for head in payload.get("token_alignment_heads", [])[:top_k]
    ]


def load_alignatt_heads_by_direction(
    paths_by_direction: Mapping[str, str],
    *,
    top_k: int,
) -> dict[str, list[AlignAttHead]]:
    """Load per-direction head lists, e.g. ``{'en-de': [...], 'en-zh': [...]}``."""
    return {
        direction: load_alignatt_heads(path, top_k=top_k)
        for direction, path in paths_by_direction.items()
    }


def shared_kernel_alignatt_heads(
    head_sets_by_direction: Mapping[str, Sequence[AlignAttHead]],
) -> list[AlignAttHead]:
    """Return heads that appear in every direction, ranked by mean ``ts``.

    The comparison is by ``(layer, head)`` identity; translation scores are
    averaged across directions so downstream code can still rank heads and cap
    to a budget.
    """
    if not head_sets_by_direction:
        return []
    direction_id_sets = []
    score_sums: dict[tuple[int, int], float] = {}
    score_counts: dict[tuple[int, int], int] = {}
    for heads in head_sets_by_direction.values():
        ids = set()
        for h in heads:
            key = (int(h.layer), int(h.head))
            ids.add(key)
            score_sums[key] = score_sums.get(key, 0.0) + float(h.ts)
            score_counts[key] = score_counts.get(key, 0) + 1
        direction_id_sets.append(ids)

    shared = set.intersection(*direction_id_sets)
    result = [
        AlignAttHead(layer=layer, head=head, ts=score_sums[(layer, head)] / score_counts[(layer, head)])
        for (layer, head) in shared
    ]
    result.sort(key=lambda h: h.ts, reverse=True)
    return result


def write_alignatt_heads_file(
    heads: Sequence[AlignAttHead],
    path: str | Path,
    *,
    direction: str | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Serialise a head list in the same JSON shape as ``load_alignatt_heads``.

    This lets Phase 4 head-set experiments write the constructed regime
    (shared kernel, multilingual union, ...) to a temp file and point
    ``translation_alignatt_heads_path`` at it, without teaching the runtime a
    separate code path for each regime.
    """
    payload: dict[str, Any] = {
        "token_alignment_heads": [
            {"layer": int(h.layer), "head": int(h.head), "ts": float(h.ts)}
            for h in heads
        ],
    }
    if direction is not None:
        payload["direction"] = direction
    if extra_metadata:
        for key, value in extra_metadata.items():
            payload.setdefault(key, value)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def multilingual_union_alignatt_heads(
    head_sets_by_direction: Mapping[str, Sequence[AlignAttHead]],
    *,
    max_heads: int | None = None,
) -> list[AlignAttHead]:
    """Return the ``(layer, head)`` union across directions, ranked by mean ``ts``.

    When ``max_heads`` is provided, the list is truncated to the strongest
    heads, which is useful when comparing a concentrated multilingual head set
    against the per-direction top-k baseline in a head-set sweep.
    """
    score_sums: dict[tuple[int, int], float] = {}
    score_counts: dict[tuple[int, int], int] = {}
    for heads in head_sets_by_direction.values():
        for h in heads:
            key = (int(h.layer), int(h.head))
            score_sums[key] = score_sums.get(key, 0.0) + float(h.ts)
            score_counts[key] = score_counts.get(key, 0) + 1

    result = [
        AlignAttHead(layer=layer, head=head, ts=score_sums[(layer, head)] / score_counts[(layer, head)])
        for (layer, head) in score_sums
    ]
    result.sort(key=lambda h: h.ts, reverse=True)
    if max_heads is not None:
        result = result[: int(max_heads)]
    return result


def project_char_span_to_token_indices(
    offsets: Sequence[tuple[int, int]],
    start_char: int,
    end_char: int,
) -> list[int]:
    indices = []
    for idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_end <= start_char:
            continue
        if tok_start >= end_char:
            break
        if tok_start < end_char and tok_end > start_char:
            indices.append(idx)
    return indices


def build_prompt_source_map(
    *,
    tokenizer,
    rendered_prompt: RenderedTranslationPrompt,
    prompt_text: str,
) -> PromptSourceMap | None:
    source_frontier = rendered_prompt.source_frontier
    if source_frontier is None or not rendered_prompt.source_text:
        return None

    current_user_message = rendered_prompt.messages[rendered_prompt.current_user_message_index]["content"]
    user_char_start = prompt_text.rfind(current_user_message)
    if user_char_start < 0:
        return None

    source_rel_start, source_rel_end = rendered_prompt.source_text_char_span_in_user_message
    source_char_start = user_char_start + source_rel_start
    source_char_end = user_char_start + source_rel_end
    return build_prompt_source_map_from_char_span(
        tokenizer=tokenizer,
        source_frontier=source_frontier,
        prompt_text=prompt_text,
        source_char_start=source_char_start,
        source_char_end=source_char_end,
    )


def build_prompt_source_map_from_char_span(
    *,
    tokenizer,
    source_frontier,
    prompt_text: str,
    source_char_start: int,
    source_char_end: int,
) -> PromptSourceMap | None:
    if source_frontier is None or not source_frontier.source_text:
        return None

    prompt_offsets = tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )["offset_mapping"]
    normalized_offsets = [tuple(map(int, off)) for off in prompt_offsets]

    source_token_positions = project_char_span_to_token_indices(
        normalized_offsets,
        source_char_start,
        source_char_end,
    )
    if not source_token_positions:
        return None

    unit_spans: list[PromptSourceUnitSpan] = []
    accessible_source_token_count = 0
    for unit_index, unit in enumerate(source_frontier.units):
        unit_prompt_positions = tuple(
            project_char_span_to_token_indices(
                normalized_offsets,
                source_char_start + unit.char_start,
                source_char_start + unit.char_end,
            )
        )
        if unit.is_accessible:
            accessible_source_token_count += len(unit_prompt_positions)
        unit_spans.append(
            PromptSourceUnitSpan(
                unit_index=unit_index,
                text=unit.text,
                prompt_token_positions=unit_prompt_positions,
                is_accessible=unit.is_accessible,
                start_ms=unit.start_ms,
                end_ms=unit.end_ms,
            )
        )

    return PromptSourceMap(
        source_text=source_frontier.source_text,
        source_token_positions=tuple(source_token_positions),
        source_unit_spans=tuple(unit_spans),
        accessible_source_token_count=accessible_source_token_count,
        accessible_unit_count=source_frontier.accessible_unit_count,
        total_unit_count=len(source_frontier.units),
        current_source_ms=source_frontier.current_source_ms,
        inaccessible_ms=source_frontier.inaccessible_ms,
        is_final=source_frontier.is_final,
    )


def extract_source_attention_rows(
    *,
    layer_attentions_by_layer: Mapping[int, torch.Tensor] | None,
    alignatt_heads: Sequence[AlignAttHead],
    source_positions: Sequence[int],
) -> torch.Tensor | None:
    rows_per_token = extract_source_attention_rows_per_token(
        layer_attentions_by_layer=layer_attentions_by_layer,
        alignatt_heads=alignatt_heads,
        source_positions=source_positions,
    )
    if not rows_per_token:
        return None
    return rows_per_token[-1]


def rotate_half(values: torch.Tensor) -> torch.Tensor:
    first_half = values[..., : values.shape[-1] // 2]
    second_half = values[..., values.shape[-1] // 2 :]
    return torch.cat((-second_half, first_half), dim=-1)


def apply_rotary_pos_emb_to_query(
    query_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (query_states * cos) + (rotate_half(query_states) * sin)


def map_attention_head_to_key_value_head(
    head_index: int,
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
) -> int:
    if num_key_value_heads <= 0:
        raise ValueError("num_key_value_heads must be positive")
    if num_attention_heads <= 0:
        raise ValueError("num_attention_heads must be positive")
    if num_attention_heads == num_key_value_heads:
        return int(head_index)
    heads_per_group = max(1, num_attention_heads // num_key_value_heads)
    return min(num_key_value_heads - 1, int(head_index) // heads_per_group)


def compute_query_states_from_layer_input_capture(
    capture: LayerInputCapture,
) -> torch.Tensor | None:
    hidden_states = capture.hidden_states
    if hidden_states.ndim != 3:
        return None

    module = capture.module
    head_dim = getattr(module, "head_dim", None)
    if head_dim is None:
        return None

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, int(head_dim))
    query_states = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    q_norm = getattr(module, "q_norm", None)
    if q_norm is not None:
        query_states = q_norm(query_states)

    if capture.position_embeddings is not None:
        cos, sin = capture.position_embeddings
        query_states = apply_rotary_pos_emb_to_query(query_states, cos, sin)
    return query_states.detach()


def compute_key_states_from_layer_input_capture(
    capture: LayerInputCapture,
) -> torch.Tensor | None:
    hidden_states = capture.hidden_states
    if hidden_states.ndim != 3:
        return None

    module = capture.module
    head_dim = getattr(module, "head_dim", None)
    if head_dim is None:
        return None

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, int(head_dim))
    key_states = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    k_norm = getattr(module, "k_norm", None)
    if k_norm is not None:
        key_states = k_norm(key_states)

    if capture.position_embeddings is not None:
        cos, sin = capture.position_embeddings
        key_states = apply_rotary_pos_emb_to_query(key_states, cos, sin)
    return key_states.detach()


def snapshot_to_layer_key_cache(
    prompt_kv_snapshot: Sequence[tuple[int, torch.Tensor, torch.Tensor, int]] | None,
) -> dict[int, torch.Tensor]:
    if prompt_kv_snapshot is None:
        return {}
    return {
        int(layer_idx): key.detach()
        for layer_idx, key, _value, _seq_length in prompt_kv_snapshot
    }


def runtime_cache_to_layer_key_cache(past_key_values) -> dict[int, torch.Tensor]:
    if past_key_values is None:
        return {}
    if hasattr(past_key_values, "layers"):
        key_cache_by_layer: dict[int, torch.Tensor] = {}
        for layer_idx, layer in enumerate(past_key_values.layers):
            keys = getattr(layer, "keys", None)
            if keys is None or getattr(keys, "numel", lambda: 0)() == 0:
                continue
            key_cache_by_layer[int(layer_idx)] = keys.detach()
        return key_cache_by_layer
    if hasattr(past_key_values, "key_cache"):
        return {
            int(layer_idx): key.detach()
            for layer_idx, key in enumerate(past_key_values.key_cache)
        }
    if isinstance(past_key_values, (list, tuple)):
        return {
            int(layer_idx): key.detach()
            for layer_idx, (key, _value) in enumerate(past_key_values)
        }
    return {}


def runtime_cache_to_shared_layer_key_cache(past_key_values) -> dict[int, torch.Tensor]:
    shared_layers = getattr(past_key_values, "shared_layers", None)
    if not isinstance(shared_layers, Mapping):
        return {}

    key_cache_by_layer: dict[int, torch.Tensor] = {}
    for layer_idx, layer_kv in shared_layers.items():
        if not isinstance(layer_kv, (list, tuple)) or not layer_kv:
            continue
        keys = layer_kv[0]
        if keys is None or getattr(keys, "numel", lambda: 0)() == 0:
            continue
        key_cache_by_layer[int(layer_idx)] = keys.detach()
    return key_cache_by_layer


def resolve_prompt_and_suffix_key_states_for_layer(
    *,
    layer_idx: int,
    capture: LayerInputCapture,
    prompt_key_cache_by_layer: Mapping[int, torch.Tensor],
    runtime_key_cache_by_layer: Mapping[int, torch.Tensor],
    runtime_shared_key_cache_by_layer: Mapping[int, torch.Tensor],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    module = capture.module
    is_kv_shared_layer = bool(getattr(module, "is_kv_shared_layer", False))
    prompt_key_layer_idx = (
        int(getattr(module, "kv_shared_layer_index"))
        if is_kv_shared_layer
        else int(layer_idx)
    )
    prompt_key_cache = prompt_key_cache_by_layer.get(prompt_key_layer_idx)
    if prompt_key_cache is None:
        return None, None

    # Sliding-window layers can evict old prompt keys as new suffix tokens arrive, so
    # their visible suffix keys are more reliable when reconstructed from the current
    # layer inputs. Full-attention layers can reuse the runtime KV cache directly.
    if getattr(module, "sliding_window", None) is not None:
        return prompt_key_cache, None

    full_key_cache = (
        runtime_shared_key_cache_by_layer.get(prompt_key_layer_idx)
        if is_kv_shared_layer
        else runtime_key_cache_by_layer.get(prompt_key_layer_idx)
    )
    if full_key_cache is None:
        return prompt_key_cache, None

    prompt_cache_length = int(prompt_key_cache.shape[2])
    if int(full_key_cache.shape[2]) < prompt_cache_length:
        return prompt_key_cache, None
    return prompt_key_cache, full_key_cache[:, :, prompt_cache_length:, :].detach()


def apply_causal_and_window_mask_to_suffix_logits(
    suffix_logits: torch.Tensor,
    *,
    prompt_length: int,
    sliding_window: int | None,
) -> torch.Tensor:
    seq_len = suffix_logits.shape[0]
    if seq_len <= 0:
        return suffix_logits

    masked = suffix_logits.clone()
    future_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=masked.device, dtype=torch.bool),
        diagonal=1,
    )
    masked = masked.masked_fill(future_mask, float("-inf"))

    if sliding_window is None or sliding_window <= 0:
        return masked

    query_positions = prompt_length + torch.arange(seq_len, device=masked.device)
    key_positions = prompt_length + torch.arange(seq_len, device=masked.device)
    min_positions = (query_positions - int(sliding_window) + 1).clamp_min(0)
    window_mask = key_positions.unsqueeze(0) < min_positions.unsqueeze(1)
    masked = masked.masked_fill(window_mask, float("-inf"))
    return masked


def apply_window_mask_to_prompt_logits(
    prompt_logits: torch.Tensor,
    *,
    prompt_length: int,
    sliding_window: int | None,
) -> torch.Tensor:
    if sliding_window is None or sliding_window <= 0 or prompt_length <= 0:
        return prompt_logits

    masked = prompt_logits.clone()
    query_positions = prompt_length + torch.arange(masked.shape[0], device=masked.device)
    key_positions = torch.arange(prompt_length, device=masked.device)
    min_positions = (query_positions - int(sliding_window) + 1).clamp_min(0)
    window_mask = key_positions.unsqueeze(0) < min_positions.unsqueeze(1)
    masked = masked.masked_fill(window_mask, float("-inf"))
    return masked


def extract_source_attention_rows_per_token_from_fast_path(
    *,
    layer_inputs_by_layer: Mapping[int, LayerInputCapture] | None,
    prompt_kv_snapshot: Sequence[tuple[int, torch.Tensor, torch.Tensor, int]] | None,
    runtime_past_key_values=None,
    alignatt_heads: Sequence[AlignAttHead],
    source_positions: Sequence[int],
    accessible_source_token_count: int | None = None,
) -> tuple[list[torch.Tensor], list[TokenProvenanceBreakdown]]:
    if not layer_inputs_by_layer or not prompt_kv_snapshot or not alignatt_heads or not source_positions:
        return [], []

    prompt_key_cache_by_layer = snapshot_to_layer_key_cache(prompt_kv_snapshot)
    runtime_key_cache_by_layer = runtime_cache_to_layer_key_cache(runtime_past_key_values)
    runtime_shared_key_cache_by_layer = runtime_cache_to_shared_layer_key_cache(
        runtime_past_key_values
    )
    source_index_tensor = None
    query_states_by_layer: dict[int, torch.Tensor] = {}
    resolved_key_states_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    head_row_matrices: list[torch.Tensor] = []

    compute_provenance = accessible_source_token_count is not None
    accessible_source_idxs: torch.Tensor | None = None
    inaccessible_source_idxs: torch.Tensor | None = None
    provenance_mass_sums: torch.Tensor | None = None
    provenance_head_count = 0

    for alignatt_head in alignatt_heads:
        layer_idx = int(alignatt_head.layer)
        capture = layer_inputs_by_layer.get(layer_idx)
        if capture is None:
            continue

        query_states = query_states_by_layer.get(layer_idx)
        if query_states is None:
            query_states = compute_query_states_from_layer_input_capture(capture)
            if query_states is None:
                continue
            query_states_by_layer[layer_idx] = query_states

        resolved_key_states = resolved_key_states_by_layer.get(layer_idx)
        if resolved_key_states is None:
            prompt_key_cache, suffix_key_states = resolve_prompt_and_suffix_key_states_for_layer(
                layer_idx=layer_idx,
                capture=capture,
                prompt_key_cache_by_layer=prompt_key_cache_by_layer,
                runtime_key_cache_by_layer=runtime_key_cache_by_layer,
                runtime_shared_key_cache_by_layer=runtime_shared_key_cache_by_layer,
            )
            if prompt_key_cache is None:
                continue
            if suffix_key_states is None:
                suffix_key_states = compute_key_states_from_layer_input_capture(capture)
            if suffix_key_states is None:
                continue
            resolved_key_states = (
                prompt_key_cache,
                suffix_key_states,
            )
            resolved_key_states_by_layer[layer_idx] = resolved_key_states

        prompt_key_cache, suffix_key_states = resolved_key_states

        num_attention_heads = int(query_states.shape[1])
        num_key_value_heads = int(prompt_key_cache.shape[1])
        head_index = int(alignatt_head.head)
        if head_index < 0 or head_index >= num_attention_heads:
            continue

        if source_index_tensor is None:
            source_index_tensor = torch.tensor(
                list(source_positions),
                device=prompt_key_cache.device,
                dtype=torch.long,
            )

        prompt_valid = (source_index_tensor >= 0) & (source_index_tensor < prompt_key_cache.shape[2])
        kv_head_index = map_attention_head_to_key_value_head(
            head_index,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
        )

        query_head = query_states[0, head_index, :, :].float()
        prompt_key_head = prompt_key_cache[0, kv_head_index, :, :].float()
        suffix_key_head = suffix_key_states[0, kv_head_index, :, :].float()

        prompt_logits = torch.matmul(query_head, prompt_key_head.transpose(0, 1))
        suffix_logits = torch.matmul(query_head, suffix_key_head.transpose(0, 1))
        scaling = float(getattr(capture.module, "scaling", 1.0))
        if scaling != 1.0:
            prompt_logits = prompt_logits * scaling
            suffix_logits = suffix_logits * scaling

        sliding_window = getattr(capture.module, "sliding_window", None)
        prompt_logits = apply_window_mask_to_prompt_logits(
            prompt_logits,
            prompt_length=prompt_key_head.shape[0],
            sliding_window=sliding_window,
        )
        suffix_logits = apply_causal_and_window_mask_to_suffix_logits(
            suffix_logits,
            prompt_length=prompt_key_head.shape[0],
            sliding_window=sliding_window,
        )
        full_logits = torch.cat([prompt_logits, suffix_logits], dim=-1)
        full_weights = torch.softmax(full_logits, dim=-1)

        row_matrix = torch.zeros(
            int(query_states.shape[2]),
            len(source_positions),
            device=query_states.device,
            dtype=torch.float32,
        )
        if torch.any(prompt_valid):
            row_matrix[:, prompt_valid] = full_weights[:, source_index_tensor[prompt_valid]]
        head_row_matrices.append(row_matrix)

        if compute_provenance:
            prompt_length = int(prompt_key_head.shape[0])
            nq = int(full_weights.shape[0])
            if provenance_mass_sums is None:
                provenance_mass_sums = torch.zeros(nq, 4, device=full_weights.device)
                accessible_source_idxs = torch.tensor(
                    list(source_positions[:accessible_source_token_count]),
                    device=full_weights.device, dtype=torch.long,
                )
                inaccessible_source_idxs = torch.tensor(
                    list(source_positions[accessible_source_token_count:]),
                    device=full_weights.device, dtype=torch.long,
                )

            suffix_mass = full_weights[:, prompt_length:].sum(dim=-1)

            acc_valid = (accessible_source_idxs >= 0) & (accessible_source_idxs < prompt_length)
            accessible_mass = torch.zeros(nq, device=full_weights.device)
            if acc_valid.any():
                accessible_mass = full_weights[:, accessible_source_idxs[acc_valid]].sum(dim=-1)

            inaccessible_mass = torch.zeros(nq, device=full_weights.device)
            if inaccessible_source_idxs.numel() > 0:
                inacc_valid = (inaccessible_source_idxs >= 0) & (inaccessible_source_idxs < prompt_length)
                if inacc_valid.any():
                    inaccessible_mass = full_weights[:, inaccessible_source_idxs[inacc_valid]].sum(dim=-1)

            non_source_mass = (1.0 - accessible_mass - inaccessible_mass - suffix_mass).clamp_min(0.0)

            provenance_mass_sums[:, 0] += accessible_mass
            provenance_mass_sums[:, 1] += inaccessible_mass
            provenance_mass_sums[:, 2] += non_source_mass
            provenance_mass_sums[:, 3] += suffix_mass
            provenance_head_count += 1

    if not head_row_matrices:
        return [], []

    stacked = torch.stack(head_row_matrices, dim=0)
    source_rows = [stacked[:, query_index, :] for query_index in range(stacked.shape[1])]

    provenance: list[TokenProvenanceBreakdown] = []
    if provenance_mass_sums is not None and provenance_head_count > 0:
        avg = provenance_mass_sums / float(provenance_head_count)
        provenance = [
            TokenProvenanceBreakdown(
                source_accessible=float(avg[q, 0]),
                source_inaccessible=float(avg[q, 1]),
                non_source_prompt=float(avg[q, 2]),
                suffix=float(avg[q, 3]),
            )
            for q in range(avg.shape[0])
        ]

    return source_rows, provenance


def extract_source_qk_rows_per_token(
    *,
    layer_inputs_by_layer: Mapping[int, LayerInputCapture] | None,
    prompt_kv_snapshot: Sequence[tuple[int, torch.Tensor, torch.Tensor, int]] | None,
    alignatt_heads: Sequence[AlignAttHead],
    source_positions: Sequence[int],
) -> list[torch.Tensor]:
    if not layer_inputs_by_layer or not prompt_kv_snapshot or not alignatt_heads or not source_positions:
        return []

    key_cache_by_layer = snapshot_to_layer_key_cache(prompt_kv_snapshot)
    source_index_tensor = None
    query_states_by_layer: dict[int, torch.Tensor] = {}
    head_row_matrices: list[torch.Tensor] = []

    for alignatt_head in alignatt_heads:
        layer_idx = int(alignatt_head.layer)
        capture = layer_inputs_by_layer.get(layer_idx)
        key_cache = key_cache_by_layer.get(layer_idx)
        if capture is None or key_cache is None:
            continue

        query_states = query_states_by_layer.get(layer_idx)
        if query_states is None:
            query_states = compute_query_states_from_layer_input_capture(capture)
            if query_states is None:
                continue
            query_states_by_layer[layer_idx] = query_states

        num_attention_heads = int(query_states.shape[1])
        num_key_value_heads = int(key_cache.shape[1])
        head_index = int(alignatt_head.head)
        if head_index < 0 or head_index >= num_attention_heads:
            continue

        if source_index_tensor is None:
            source_index_tensor = torch.tensor(
                list(source_positions),
                device=key_cache.device,
                dtype=torch.long,
            )

        valid = (source_index_tensor >= 0) & (source_index_tensor < key_cache.shape[2])
        row_matrix = torch.zeros(
            int(query_states.shape[2]),
            len(source_positions),
            device=query_states.device,
            dtype=torch.float32,
        )
        if torch.any(valid):
            kv_head_index = map_attention_head_to_key_value_head(
                head_index,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
            )
            query_head = query_states[0, head_index, :, :].float()
            key_head = key_cache[0, kv_head_index, source_index_tensor[valid], :].float()
            row_matrix[:, valid] = torch.matmul(query_head, key_head.transpose(0, 1))
        scaling = float(getattr(capture.module, "scaling", 1.0))
        if scaling != 1.0:
            row_matrix = row_matrix * scaling
        head_row_matrices.append(row_matrix)

    if not head_row_matrices:
        return []

    stacked = torch.stack(head_row_matrices, dim=0)
    return [stacked[:, query_index, :] for query_index in range(stacked.shape[1])]


def extract_source_attention_rows_per_token(
    *,
    layer_attentions_by_layer: Mapping[int, torch.Tensor] | None,
    alignatt_heads: Sequence[AlignAttHead],
    source_positions: Sequence[int],
) -> list[torch.Tensor]:
    if not layer_attentions_by_layer or not alignatt_heads or not source_positions:
        return []

    source_index_tensor = None
    max_context_length = 0
    query_length = 0
    for alignatt_head in alignatt_heads:
        layer_attn = layer_attentions_by_layer.get(int(alignatt_head.layer))
        if layer_attn is None:
            continue
        head_matrix = layer_attn[0, alignatt_head.head, :, :]
        max_context_length = max(max_context_length, int(head_matrix.shape[-1]))
        query_length = max(query_length, int(head_matrix.shape[0]))
    if max_context_length <= 0 or query_length <= 0:
        return []

    head_row_matrices: list[torch.Tensor] = []
    for alignatt_head in alignatt_heads:
        layer_attn = layer_attentions_by_layer.get(int(alignatt_head.layer))
        if layer_attn is None:
            continue
        head_matrix = layer_attn[0, alignatt_head.head, :, :]
        context_length = int(head_matrix.shape[-1])
        global_offset = max_context_length - context_length
        if source_index_tensor is None:
            source_index_tensor = torch.tensor(
                list(source_positions),
                device=head_matrix.device,
                dtype=torch.long,
            )
        local_positions = source_index_tensor - int(global_offset)
        valid = (local_positions >= 0) & (local_positions < context_length)
        row_matrix = torch.zeros(
            int(head_matrix.shape[0]),
            len(source_positions),
            device=head_matrix.device,
            dtype=head_matrix.dtype,
        )
        if torch.any(valid):
            row_matrix[:, valid] = head_matrix.index_select(-1, local_positions[valid])
        head_row_matrices.append(row_matrix)
    if not head_row_matrices:
        return []

    stacked = torch.stack(head_row_matrices, dim=0)
    return [stacked[:, query_index, :] for query_index in range(stacked.shape[1])]


def source_local_position_to_unit_index(
    source_map: PromptSourceMap | None,
    source_local_position: int | None,
) -> int | None:
    if source_map is None or source_local_position is None:
        return None
    if source_local_position < 0 or source_local_position >= len(source_map.source_token_positions):
        return None

    prompt_token_position = source_map.source_token_positions[source_local_position]
    for unit_span in source_map.source_unit_spans:
        if prompt_token_position in unit_span.prompt_token_positions:
            return int(unit_span.unit_index)
    return None


def median_filter_last_dim(values: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 1 or values.shape[-1] <= 1:
        return values
    radius = width // 2
    padded = F.pad(values, (radius, radius), mode="replicate")
    windows = padded.unfold(-1, width, 1)
    return windows.median(dim=-1).values


def _prepare_alignatt_attention_tensor(
    source_attention_rows_per_token: Sequence[torch.Tensor],
    *,
    filter_width: int,
) -> torch.Tensor | None:
    if not source_attention_rows_per_token:
        return None

    attention_tensor = torch.stack(list(source_attention_rows_per_token), dim=1)
    if attention_tensor.shape[-1] <= 0:
        return None

    std, mean = torch.std_mean(attention_tensor, dim=1, keepdim=True, unbiased=False)
    attention_tensor = (attention_tensor - mean) / std.clamp_min(1e-6)
    return median_filter_last_dim(attention_tensor, filter_width)


def _median_position(positions: Sequence[int]) -> int:
    ordered = sorted(int(position) for position in positions)
    return int(ordered[(len(ordered) - 1) // 2])


def head_agreement_ratio(
    per_head_positions: Sequence[int] | None,
    consensus_position: int | None,
    *,
    tolerance: int = 1,
) -> float | None:
    """Fraction of attention heads whose source argmax agrees with the consensus.

    Agreement means the head's argmax lies within ``tolerance`` source tokens of
    the consensus position. This is the alignment-confidence statistic shared by
    the recorded diagnostics and the ``unit_conf`` acceptance gate, so the
    feature studied offline is exactly the feature gated online.
    """
    if per_head_positions is None or consensus_position is None:
        return None
    positions = [int(position) for position in per_head_positions]
    if not positions:
        return None
    consensus = int(consensus_position)
    agreeing = sum(
        1 for position in positions if abs(position - consensus) <= int(tolerance)
    )
    return agreeing / float(len(positions))


def compute_alignatt_source_argmaxes(
    source_attention_rows_per_token: Sequence[torch.Tensor],
    *,
    filter_width: int,
) -> list[int | None]:
    # Robustify at the head-consensus level before the per-token frontier test.
    # Equal-weight averaging can be dragged by a minority of noisy heads;
    # `median_argmax` keeps the monotone argmax pipeline while avoiding
    # benchmark-specific head blacklists.
    #
    # We intentionally do not expose alternative aggregations in the
    # maintained runtime anymore:
    #   - mean / weighted-mean were too sensitive to one bad head
    #   - weighted-median matched the unweighted median on the maintained
    #     head set, so the extra knob bought no measurable benefit
    # The robust consensus is the point here; not squeezing a tiny gain by
    # retuning the aggregator per benchmark.
    attention_tensor = _prepare_alignatt_attention_tensor(
        source_attention_rows_per_token,
        filter_width=filter_width,
    )
    if attention_tensor is None:
        if not source_attention_rows_per_token:
            return []
        return [None] * len(source_attention_rows_per_token)

    per_head_positions = torch.argmax(attention_tensor, dim=-1).transpose(0, 1).tolist()
    return [_median_position(query_positions) for query_positions in per_head_positions]


def compute_prefix_online_alignatt_source_argmaxes(
    source_attention_rows_per_token: Sequence[torch.Tensor],
    *,
    filter_width: int,
    normalization: str = "zscore",
) -> list[int | None]:
    if not source_attention_rows_per_token:
        return []

    tracker = IncrementalAlignAttTracker(
        filter_width=filter_width,
        normalization=normalization,
    )
    return [tracker.update(source_attention_rows) for source_attention_rows in source_attention_rows_per_token]


def compute_prefix_online_alignatt_per_head_source_argmaxes(
    source_attention_rows_per_token: Sequence[torch.Tensor],
    *,
    filter_width: int,
    normalization: str = "zscore",
) -> list[list[int]]:
    if not source_attention_rows_per_token:
        return []

    tracker = IncrementalAlignAttTracker(
        filter_width=filter_width,
        normalization=normalization,
    )
    per_token_positions: list[list[int]] = []
    for source_attention_rows in source_attention_rows_per_token:
        _, per_head_positions = tracker.update_with_per_head_positions(
            source_attention_rows
        )
        per_token_positions.append(per_head_positions)
    return per_token_positions
