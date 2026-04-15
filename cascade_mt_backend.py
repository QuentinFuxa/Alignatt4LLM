from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, modeling_utils
from transformers.cache_utils import DynamicCache

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
    accepted_candidate_ids: list[int]
    prompt_num_tokens: int
    num_cached_tokens: int | None
    stop_reason: str | int | None
    source_attention_rows_per_token: list[torch.Tensor]
    unsafe_reason: str | None = None
    unsafe_target_token_index: int | None = None
    rewind_from_local_position: int | None = None
    rewind_to_local_position: int | None = None


@dataclass
class AlignAttAcceptance:
    accepted_generated_ids: list[int]
    alignatt_metadata: dict[str, Any] | None


@dataclass
class MTBackendResult:
    draft_text: str
    acceptance_text: str
    draft_token_ids: tuple[int, ...] = ()
    accepted_token_ids: tuple[int, ...] = ()
    num_cached_tokens: int | None = None
    prompt_num_tokens: int | None = None
    stop_reason: str | int | None = None
    alignatt_metadata: dict[str, Any] | None = None


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
    backend_kind = str(getattr(runtime_config, "translation_mt_backend", "vllm"))
    if backend_kind == "vllm":
        return VLLMGemmaMTBackend(model_name=model_name, runtime_config=runtime_config)
    if backend_kind == "transformers_alignatt":
        return TransformersAlignAttGemmaMTBackend(
            model_name=model_name,
            runtime_config=runtime_config,
        )
    raise ValueError(f"Unknown translation_mt_backend: {backend_kind}")


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
    ) -> MTBackendResult:
        raise NotImplementedError

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
            max_token_cap = (
                self.runtime_config.partial_followup_max_new_tokens
                if assistant_prefill.strip()
                else self.runtime_config.partial_max_new_tokens
            )
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


class AlignAttDecoderPolicy:
    def __init__(self, *, tokenizer, runtime_config: SimpleNamespace):
        self.tokenizer = tokenizer
        self.runtime_config = runtime_config

    @staticmethod
    def token_starts_new_word(token: str) -> bool:
        if not token:
            return False
        if token.startswith(("▁", "Ġ")):
            return True
        if token[0].isspace():
            return True
        if token.startswith("<0x0A>"):
            return True
        return False

    def trim_to_last_complete_word(self, generated_ids: Sequence[int]) -> list[int]:
        if not generated_ids:
            return []
        token_strings = self.tokenizer.convert_ids_to_tokens(list(generated_ids))
        word_start_indices = [
            idx for idx, token in enumerate(token_strings) if self.token_starts_new_word(str(token))
        ]
        if len(word_start_indices) <= 1:
            return []
        return list(generated_ids[: word_start_indices[-1]])

    def should_stop_in_loop(
        self,
        *,
        source_attention_rows_per_token: Sequence[torch.Tensor],
        last_aligned_source_local_position: int | None,
        accessible_source_token_count: int,
    ) -> tuple[str | None, int | None, int | None, int | None]:
        if not source_attention_rows_per_token:
            return None, None, None, None
        aligned_source_local_positions = compute_alignatt_source_argmaxes(
            source_attention_rows_per_token,
            filter_width=7,
        )
        current_source_local_position = aligned_source_local_positions[-1]
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

        if current_source_local_position >= max(0, int(accessible_source_token_count)):
            return "source_frontier", current_source_local_position, None, None
        return None, current_source_local_position, None, None

    def finalize_partial(
        self,
        *,
        accepted_candidate_ids: Sequence[int],
        source_attention_rows_per_token: Sequence[torch.Tensor],
        source_map: PromptSourceMap | None,
        unsafe_reason: str | None,
        unsafe_target_token_index: int | None,
        rewind_from_local_position: int | None,
        rewind_to_local_position: int | None,
        stop_reason: str | int | None,
    ) -> AlignAttAcceptance:
        aligned_source_local_positions: list[int | None] = []
        if source_attention_rows_per_token:
            aligned_source_local_positions = compute_alignatt_source_argmaxes(
                source_attention_rows_per_token,
                filter_width=7,
            )

        trimmed_generated_ids = self.trim_to_last_complete_word(accepted_candidate_ids)
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
            "aligned_source_local_positions": aligned_source_local_positions,
            "unsafe_target_token_index": unsafe_target_token_index,
            "unsafe_reason": unsafe_reason,
            "rewind_from_local_position": rewind_from_local_position,
            "rewind_to_local_position": rewind_to_local_position,
            "accepted_candidate_token_count": len(accepted_candidate_ids),
            "accepted_token_count": len(trimmed_generated_ids),
            "word_boundary_trimmed": word_boundary_trimmed,
            "stop_reason": stop_reason,
            "current_audio_ms": None if source_map is None else source_map.current_audio_ms,
            "inaccessible_ms": None if source_map is None else source_map.inaccessible_ms,
        }
        return AlignAttAcceptance(
            accepted_generated_ids=trimmed_generated_ids,
            alignatt_metadata=alignatt_metadata,
        )


class VLLMGemmaMTBackend(BaseMTBackend):
    def __init__(self, *, model_name: str, runtime_config: SimpleNamespace):
        super().__init__(model_name=model_name, runtime_config=runtime_config)
        self.llm = None

    def load(self) -> None:
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                local_files_only=True,
            )
        if self.llm is None:
            from vllm import LLM

            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=max(1, torch.cuda.device_count()),
                max_model_len=self.runtime_config.gemma_max_model_len,
                gpu_memory_utilization=self.runtime_config.gemma_gpu_memory_utilization,
                enforce_eager=self.runtime_config.gemma_enforce_eager,
                enable_prefix_caching=self.runtime_config.gemma_enable_prefix_caching,
                trust_remote_code=True,
            )

    def translate(
        self,
        *,
        rendered_prompt: RenderedTranslationPrompt,
        variant: TranslationVariant,
        is_partial: bool,
    ) -> MTBackendResult:
        if self.tokenizer is None or self.llm is None:
            raise RuntimeError("MT backend is not loaded. Run load() first.")
        from vllm import SamplingParams

        prompt_package = self.render_prompt_package(rendered_prompt)
        prompt_num_tokens = len(prompt_package.prompt_token_ids)
        sampling_kwargs: dict[str, Any] = {
            "temperature": self.runtime_config.temperature,
            "max_tokens": self.compute_max_tokens(
                prompt_tokens=prompt_num_tokens,
                source_text=rendered_prompt.source_text,
                is_partial=is_partial,
                assistant_prefill=rendered_prompt.assistant_prefill,
            ),
            "repetition_penalty": self.runtime_config.repetition_penalty,
            "skip_reading_prefix_cache": False,
        }

        generation_stop_token_ids = self.resolve_generation_stop_token_ids()
        if generation_stop_token_ids:
            sampling_kwargs["stop_token_ids"] = list(generation_stop_token_ids)

        outputs = self.llm.generate(
            {"prompt_token_ids": list(prompt_package.prompt_token_ids)},
            SamplingParams(**sampling_kwargs),
        )
        request_output = outputs[0]
        completion = request_output.outputs[0]
        generated_ids = tuple(int(token_id) for token_id in completion.token_ids)
        draft_text = self.decode_candidate_text(
            generated_ids=generated_ids,
            assistant_prefill=rendered_prompt.assistant_prefill,
            variant=variant,
            is_partial=is_partial,
        )
        return MTBackendResult(
            draft_text=draft_text,
            acceptance_text=draft_text,
            draft_token_ids=generated_ids,
            accepted_token_ids=generated_ids,
            num_cached_tokens=request_output.num_cached_tokens,
            prompt_num_tokens=prompt_num_tokens,
            stop_reason=completion.stop_reason,
        )


class TransformersAlignAttGemmaMTBackend(BaseMTBackend):
    def __init__(self, *, model_name: str, runtime_config: SimpleNamespace):
        super().__init__(model_name=model_name, runtime_config=runtime_config)
        self.model = None
        self.device = str(getattr(runtime_config, "gemma_transformers_device", "cuda:0"))
        self.dtype = getattr(torch, str(getattr(runtime_config, "gemma_transformers_dtype", "bfloat16")))
        self.alignatt_heads: list[AlignAttHead] = []
        self.prompt_cache = PromptCacheState()
        self.policy = None

    def load(self) -> None:
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                local_files_only=True,
            )
        if self.model is None:
            original_caching_allocator_warmup = None
            if hasattr(modeling_utils, "caching_allocator_warmup"):
                original_caching_allocator_warmup = modeling_utils.caching_allocator_warmup
                modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.dtype,
                    device_map=self.device,
                    trust_remote_code=True,
                    local_files_only=True,
                    attn_implementation="eager",
                    low_cpu_mem_usage=True,
                )
            finally:
                if original_caching_allocator_warmup is not None:
                    modeling_utils.caching_allocator_warmup = original_caching_allocator_warmup
            self.model.eval()
        if not self.alignatt_heads:
            self.alignatt_heads = load_alignatt_heads(
                getattr(self.runtime_config, "translation_alignatt_heads_path"),
                top_k=int(getattr(self.runtime_config, "translation_alignatt_top_k_heads", 8)),
            )
        if self.policy is None:
            self.policy = AlignAttDecoderPolicy(
                tokenizer=self.tokenizer,
                runtime_config=self.runtime_config,
            )

    @staticmethod
    def _common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
        size = min(len(a), len(b))
        idx = 0
        while idx < size and a[idx] == b[idx]:
            idx += 1
        return idx

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
            raise RuntimeError("MT backend is not loaded. Run load() first.")
        past_kv = DynamicCache(config=self.model.config)
        for layer_idx, key, value, seq_length in snapshot:
            past_kv.update(
                key[:, :, :length, :],
                value[:, :, :length, :],
                layer_idx=layer_idx,
            )
            layer = past_kv.layers[layer_idx]
            if hasattr(layer, "cumulative_length"):
                layer.cumulative_length = int(seq_length)
        return past_kv

    def _forward_prompt_with_cache(
        self,
        *,
        prompt_ids: Sequence[int],
        output_attentions: bool,
    ):
        if self.model is None:
            raise RuntimeError("MT backend is not loaded. Run load() first.")
        if not bool(getattr(self.runtime_config, "gemma_transformers_prompt_kv_reuse", False)):
            device = next(self.model.parameters()).device
            with torch.no_grad():
                outputs = self.model(
                    input_ids=torch.tensor([list(prompt_ids)], device=device),
                    use_cache=True,
                    output_attentions=output_attentions,
                )
            return outputs, outputs.past_key_values, 0
        prev_ids = list(self.prompt_cache.full_prompt_ids)
        shared_len = 0
        if self.prompt_cache.prompt_kv_snapshot is not None and prev_ids:
            shared_len = self._common_prefix_len(prompt_ids, prev_ids)
        if shared_len == len(prompt_ids) and shared_len > 0:
            shared_len -= 1

        device = next(self.model.parameters()).device
        if shared_len > 0 and self.prompt_cache.prompt_kv_snapshot is not None:
            past_kv = self._restore_kv(self.prompt_cache.prompt_kv_snapshot, shared_len)
            delta_ids = list(prompt_ids[shared_len:])
            with torch.no_grad():
                outputs = self.model(
                    input_ids=torch.tensor([delta_ids], device=device),
                    past_key_values=past_kv,
                    use_cache=True,
                    output_attentions=output_attentions,
                )
            past_kv = outputs.past_key_values
        else:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=torch.tensor([list(prompt_ids)], device=device),
                    use_cache=True,
                    output_attentions=output_attentions,
                )
            past_kv = outputs.past_key_values
            shared_len = 0

        self.prompt_cache.prompt_kv_snapshot = self._snapshot_kv(past_kv, len(prompt_ids))
        self.prompt_cache.full_prompt_ids = list(prompt_ids)
        return outputs, past_kv, shared_len

    def decode_draft(
        self,
        *,
        prompt_token_ids: Sequence[int],
        source_map: PromptSourceMap | None,
        max_new_tokens: int,
        is_partial: bool,
    ) -> DraftDecodingResult:
        if self.model is None or self.tokenizer is None or self.policy is None:
            raise RuntimeError("MT backend is not loaded. Run load() first.")

        outputs, past_key_values, num_cached_tokens = self._forward_prompt_with_cache(
            prompt_ids=prompt_token_ids,
            output_attentions=bool(
                is_partial and self.alignatt_heads and source_map and source_map.source_token_positions
            ),
        )
        device = next(self.model.parameters()).device
        generation_stop_token_ids = set(self.resolve_generation_stop_token_ids())

        draft_generated_ids: list[int] = []
        accepted_candidate_ids: list[int] = []
        source_attention_rows_per_token: list[torch.Tensor] = []
        unsafe_reason: str | None = None
        unsafe_target_token_index: int | None = None
        rewind_from_local_position: int | None = None
        rewind_to_local_position: int | None = None
        stop_reason: str | int | None = None
        prior_token_ids = list(prompt_token_ids)
        last_aligned_source_local_position: int | None = None

        for _ in range(max_new_tokens):
            logits = outputs.logits[0, -1, :].float()
            logits = self.apply_repetition_penalty(
                logits,
                prior_token_ids=prior_token_ids[-64:],
                repetition_penalty=float(self.runtime_config.repetition_penalty),
            )
            next_token_id = int(logits.argmax().item())

            if next_token_id in generation_stop_token_ids:
                if next_token_id == getattr(self.tokenizer, "eos_token_id", None):
                    stop_reason = "eos"
                else:
                    stop_reason = self.tokenizer.convert_ids_to_tokens(next_token_id)
                break

            draft_generated_ids.append(next_token_id)
            prior_token_ids.append(next_token_id)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=torch.tensor([[next_token_id]], device=device),
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=bool(
                        is_partial and self.alignatt_heads and source_map and source_map.source_token_positions
                    ),
                )
            past_key_values = outputs.past_key_values

            if is_partial and self.alignatt_heads and source_map and source_map.source_token_positions:
                source_attention_rows = extract_source_attention_rows(
                    attentions=outputs.attentions,
                    alignatt_heads=self.alignatt_heads,
                    source_positions=source_map.source_token_positions,
                )
                if source_attention_rows is not None:
                    source_attention_rows_per_token.append(source_attention_rows)
                    (
                        unsafe_reason,
                        current_aligned_source_local_position,
                        rewind_from_local_position,
                        rewind_to_local_position,
                    ) = self.policy.should_stop_in_loop(
                        source_attention_rows_per_token=source_attention_rows_per_token,
                        last_aligned_source_local_position=last_aligned_source_local_position,
                        accessible_source_token_count=source_map.accessible_source_token_count,
                    )
                    if unsafe_reason == "rewind":
                        unsafe_target_token_index = 0
                        accepted_candidate_ids = []
                        stop_reason = "alignatt:rewind"
                        break
                    if unsafe_reason == "source_frontier":
                        unsafe_target_token_index = len(draft_generated_ids) - 1
                        stop_reason = "alignatt:source_frontier"
                        break
                    last_aligned_source_local_position = current_aligned_source_local_position

            accepted_candidate_ids.append(next_token_id)

        return DraftDecodingResult(
            draft_generated_ids=draft_generated_ids,
            accepted_candidate_ids=accepted_candidate_ids,
            prompt_num_tokens=len(prompt_token_ids),
            num_cached_tokens=num_cached_tokens,
            stop_reason=stop_reason,
            source_attention_rows_per_token=source_attention_rows_per_token,
            unsafe_reason=unsafe_reason,
            unsafe_target_token_index=unsafe_target_token_index,
            rewind_from_local_position=rewind_from_local_position,
            rewind_to_local_position=rewind_to_local_position,
        )

    def translate(
        self,
        *,
        rendered_prompt: RenderedTranslationPrompt,
        variant: TranslationVariant,
        is_partial: bool,
    ) -> MTBackendResult:
        if self.model is None or self.tokenizer is None or self.policy is None:
            raise RuntimeError("MT backend is not loaded. Run load() first.")

        prompt_package = self.render_prompt_package(rendered_prompt)
        max_new_tokens = self.compute_max_tokens(
            prompt_tokens=len(prompt_package.prompt_token_ids),
            source_text=rendered_prompt.source_text,
            is_partial=is_partial,
            assistant_prefill=rendered_prompt.assistant_prefill,
        )
        draft_result = self.decode_draft(
            prompt_token_ids=prompt_package.prompt_token_ids,
            source_map=prompt_package.source_map,
            max_new_tokens=max_new_tokens,
            is_partial=is_partial,
        )

        draft_text = self.decode_candidate_text(
            generated_ids=draft_result.draft_generated_ids,
            assistant_prefill=rendered_prompt.assistant_prefill,
            variant=variant,
            is_partial=is_partial,
        )

        if is_partial:
            acceptance = self.policy.finalize_partial(
                accepted_candidate_ids=draft_result.accepted_candidate_ids,
                source_attention_rows_per_token=draft_result.source_attention_rows_per_token,
                source_map=prompt_package.source_map,
                unsafe_reason=draft_result.unsafe_reason,
                unsafe_target_token_index=draft_result.unsafe_target_token_index,
                rewind_from_local_position=draft_result.rewind_from_local_position,
                rewind_to_local_position=draft_result.rewind_to_local_position,
                stop_reason=draft_result.stop_reason,
            )
            accepted_token_ids = tuple(int(token_id) for token_id in acceptance.accepted_generated_ids)
            acceptance_text = self.decode_candidate_text(
                generated_ids=accepted_token_ids,
                assistant_prefill=rendered_prompt.assistant_prefill,
                variant=variant,
                is_partial=True,
            )
            alignatt_metadata = acceptance.alignatt_metadata
        else:
            accepted_token_ids = tuple(int(token_id) for token_id in draft_result.draft_generated_ids)
            acceptance_text = draft_text
            alignatt_metadata = None

        return MTBackendResult(
            draft_text=draft_text,
            acceptance_text=acceptance_text,
            draft_token_ids=tuple(int(token_id) for token_id in draft_result.draft_generated_ids),
            accepted_token_ids=accepted_token_ids,
            num_cached_tokens=draft_result.num_cached_tokens,
            prompt_num_tokens=draft_result.prompt_num_tokens,
            stop_reason=draft_result.stop_reason,
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
    attentions,
    alignatt_heads: Sequence[AlignAttHead],
    source_positions: Sequence[int],
) -> torch.Tensor | None:
    if not attentions or not alignatt_heads or not source_positions:
        return None
    max_context_length = 0
    for alignatt_head in alignatt_heads:
        if alignatt_head.layer >= len(attentions):
            continue
        layer_attn = attentions[alignatt_head.layer]
        if layer_attn is None:
            continue
        head_vector = layer_attn[0, alignatt_head.head, -1, :].detach().float()
        max_context_length = max(max_context_length, int(head_vector.shape[-1]))
    if max_context_length <= 0:
        return None

    head_rows: list[torch.Tensor] = []
    for alignatt_head in alignatt_heads:
        if alignatt_head.layer >= len(attentions):
            continue
        layer_attn = attentions[alignatt_head.layer]
        if layer_attn is None:
            continue
        head_vector = layer_attn[0, alignatt_head.head, -1, :].detach().float()
        context_length = int(head_vector.shape[-1])
        global_offset = max_context_length - context_length
        row = []
        for source_position in source_positions:
            local_position = int(source_position) - global_offset
            if 0 <= local_position < context_length:
                row.append(head_vector[local_position])
            else:
                row.append(torch.tensor(0.0, device=head_vector.device))
        head_rows.append(torch.stack(row))
    if not head_rows:
        return None
    return torch.stack(head_rows, dim=0)


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
