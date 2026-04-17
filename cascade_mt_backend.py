from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from cascade_translation_variants import RenderedTranslationPrompt, TranslationVariant
from cascade_source_frontier import SourceAccessibilityFrontier


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
    rewind_from_local_position: int | None = None
    rewind_to_local_position: int | None = None
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
    def __init__(self, *, filter_width: int):
        self.filter_width = int(filter_width)
        self.token_count = 0
        self.running_mean: torch.Tensor | None = None
        self.running_m2: torch.Tensor | None = None
        self.aligned_source_local_positions: list[int | None] = []

    def update(self, source_attention_rows: torch.Tensor) -> int | None:
        if source_attention_rows.ndim != 2:
            raise ValueError(
                "source_attention_rows must have shape [num_heads, source_token_count] "
                f"but got {tuple(source_attention_rows.shape)}"
            )

        rows = source_attention_rows.detach().float()
        if rows.shape[-1] <= 0:
            self.aligned_source_local_positions.append(None)
            return None

        self.token_count += 1
        if self.running_mean is None or self.running_m2 is None:
            self.running_mean = rows.clone()
            self.running_m2 = torch.zeros_like(rows)
        else:
            delta = rows - self.running_mean
            self.running_mean = self.running_mean + delta / float(self.token_count)
            delta2 = rows - self.running_mean
            self.running_m2 = self.running_m2 + delta * delta2

        variance = self.running_m2 / max(1, self.token_count)
        normalized_rows = (rows - self.running_mean) / variance.sqrt().clamp_min(1e-6)
        smoothed_rows = median_filter_last_dim(normalized_rows, self.filter_width)
        averaged_row = smoothed_rows.mean(dim=0)
        aligned_position = int(torch.argmax(averaged_row, dim=-1).item())
        self.aligned_source_local_positions.append(aligned_position)
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
    current_audio_ms: float
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
        from gemma_vllm_mt_backend import VLLMAlignAttGemmaMTBackend

        return VLLMAlignAttGemmaMTBackend(
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
    ) -> int:
        if self.tokenizer is None:
            raise RuntimeError("Gemma tokenizer is not loaded. Run load() first.")
        source_tokens = len(self.tokenizer(source_text, add_special_tokens=False)["input_ids"])
        if is_partial:
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
            self.runtime_config.gemma_max_model_len
            - prompt_tokens
            - self.runtime_config.translation_generation_margin
        )
        if available_max_tokens < 1:
            raise RuntimeError(
                f"Gemma prompt exhausted the context window: prompt_tokens={prompt_tokens} "
                f"gemma_max_model_len={self.runtime_config.gemma_max_model_len}"
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
    def __init__(self, *, tokenizer, runtime_config: SimpleNamespace):
        self.tokenizer = tokenizer
        self.runtime_config = runtime_config

    def alignatt_filter_width(self) -> int:
        return int(getattr(self.runtime_config, "translation_alignatt_filter_width", 7))

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

    def should_stop_in_loop(
        self,
        *,
        current_source_local_position: int | None,
        last_aligned_source_local_position: int | None,
        accessible_source_token_count: int,
        source_inaccessible_mass: float | None = None,
    ) -> tuple[str | None, int | None, int | None, int | None]:
        if current_source_local_position is None:
            return None, None, None, None

        rewind_threshold = int(
            getattr(self.runtime_config, "translation_alignatt_rewind_threshold", 3)
        )
        if (
            last_aligned_source_local_position is not None
            and last_aligned_source_local_position - current_source_local_position > rewind_threshold
        ):
            return (
                "rewind",
                current_source_local_position,
                last_aligned_source_local_position,
                current_source_local_position,
            )

        border_margin = int(
            getattr(self.runtime_config, "translation_alignatt_border_margin", 0)
        )
        frontier = max(0, int(accessible_source_token_count)) + border_margin
        if current_source_local_position >= frontier:
            return "source_frontier", current_source_local_position, None, None
        return None, current_source_local_position, None, None

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
        rewind_from_local_position: int | None,
        rewind_to_local_position: int | None,
        stop_reason: str | int | None,
        probe_backend: str | None,
    ) -> AlignAttAcceptance:
        unsafe_token_starts_new_unit = False
        if unsafe_token_id is not None:
            unsafe_token = str(
                self.tokenizer.convert_ids_to_tokens([int(unsafe_token_id)])[0]
            )
            unsafe_token_starts_new_unit = self.token_starts_stability_unit(
                unsafe_token
            )
        if unsafe_token_starts_new_unit:
            trimmed_generated_ids = list(accepted_candidate_ids)
        else:
            trimmed_generated_ids = self.trim_to_last_stability_unit(
                accepted_candidate_ids
            )
        word_boundary_trimmed = list(trimmed_generated_ids) != list(accepted_candidate_ids)
        alignatt_metadata = {
            "source_token_count": 0 if source_map is None else len(source_map.source_token_positions),
            "source_unit_count": 0 if source_map is None else source_map.total_unit_count,
            "accessible_source_unit_count": 0
            if source_map is None
            else source_map.accessible_unit_count,
            "accessible_source_local_end_exclusive": 0
            if source_map is None
            else source_map.accessible_source_token_count,
            "aligned_source_local_positions": list(aligned_source_local_positions),
            "unsafe_target_token_index": unsafe_target_token_index,
            "unsafe_reason": unsafe_reason,
            "unsafe_token_id": unsafe_token_id,
            "unsafe_token_starts_new_unit": unsafe_token_starts_new_unit,
            "blocked_source_local_position": blocked_source_local_position,
            "blocked_source_unit_index": blocked_source_unit_index,
            "rewind_from_local_position": rewind_from_local_position,
            "rewind_to_local_position": rewind_to_local_position,
            "accepted_candidate_token_count": len(accepted_candidate_ids),
            "accepted_token_count": len(trimmed_generated_ids),
            "word_boundary_trimmed": word_boundary_trimmed,
            "stop_reason": stop_reason,
            "current_audio_ms": None if source_map is None else source_map.current_audio_ms,
            "inaccessible_ms": None if source_map is None else source_map.inaccessible_ms,
            "probe_mode": "prefix_online_batched",
            "probe_backend": probe_backend,
        }
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
        current_audio_ms=source_frontier.current_audio_ms,
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


def compute_alignatt_source_argmaxes(
    source_attention_rows_per_token: Sequence[torch.Tensor],
    *,
    filter_width: int,
) -> list[int | None]:
    if not source_attention_rows_per_token:
        return []

    attention_tensor = torch.stack(list(source_attention_rows_per_token), dim=1)
    if attention_tensor.shape[-1] <= 0:
        return [None] * attention_tensor.shape[1]

    std, mean = torch.std_mean(attention_tensor, dim=1, keepdim=True, unbiased=False)
    attention_tensor = (attention_tensor - mean) / std.clamp_min(1e-6)
    attention_tensor = median_filter_last_dim(attention_tensor, filter_width)
    attention_tensor = attention_tensor.mean(dim=0)
    return [int(position) for position in torch.argmax(attention_tensor, dim=-1).tolist()]


def compute_prefix_online_alignatt_source_argmaxes(
    source_attention_rows_per_token: Sequence[torch.Tensor],
    *,
    filter_width: int,
) -> list[int | None]:
    if not source_attention_rows_per_token:
        return []

    tracker = IncrementalAlignAttTracker(filter_width=filter_width)
    return [tracker.update(source_attention_rows) for source_attention_rows in source_attention_rows_per_token]
