"""Gemma-family MT backends via vLLM AlignAtt.

Mechanics rewrite of the Transformers MT backend under vLLM. Phase 1 covered
draft generation and prompt parity; Phase 2 adds the MT AlignAtt observer and
the 4-way provenance partition that the runtime policy needs.

Design notes
------------

The observer is built on the same Gemma-family attention substrate as the
ASR-side vLLM seam:

- a custom ``worker_cls`` installs the observer **before** engine build so
  compile/cudagraph capture see the patched attention ``forward``;
- per-layer tensor buffers capture K at every prompt position *and* Q+K at
  every decode position (the MT provenance partition needs the full
  ``softmax([prompt_K | suffix_K])`` weights, not just ``softmax(prompt_K)``);
- a compact payload is fetched back once decoding finishes and re-used to
  reconstruct the attention rows and 4-way mass partition that the existing
  runtime policy already consumes.

``is_partial=False`` still decodes a full translation with no observer work.
``is_partial=True`` runs the full AlignAtt probe and returns real
``alignatt:source_frontier`` / ``alignatt:provenance_weak`` stop reasons
plus a ``provenance_per_draft_token`` list — the same semantic surface
the Transformers backend produces.

``GemmaVLLMMTBackend`` is the submitted IWSLT MT route. ``MiLMMTVLLMMTBackend``
is an explicit experimental route that reuses the same Q/K observer substrate
with MiLMMT's raw translation prompt.
"""
from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

from cascade.mt.base import (
    AlignAttDecoderPolicy,
    BaseMTBackend,
    MTBackendResult,
    PromptSourceMap,
    PromptSourceUnitSpan,
    PromptCacheState,
    RenderedPromptWithSourceMap,
    TokenProvenanceBreakdown,
    compute_prefix_online_alignatt_source_argmaxes,
    load_alignatt_heads,
    project_char_span_to_token_indices,
    source_local_position_to_unit_index,
)
from cascade.translation_variants import RenderedTranslationPrompt, TranslationVariant

from cascade.mt.gemma_vllm_observer import (
    _MT_OBSERVER_BOOTSTRAP_ENV,
    _encode_mt_observer_bootstrap,
    reconstruct_mt_attention_rows,
)


MILMMT_NUM_TEXT_LAYERS = 34
MILMMT_NUM_ATTENTION_HEADS = 8
GEMMA4_E4B_SPECULATIVE_ASSISTANT_MODEL_ID = "google/gemma-4-E4B-it-assistant"


def _hf_hub_roots() -> list[Path]:
    roots: list[Path] = []
    for candidate in (
        os.environ.get("HF_HUB_CACHE"),
        os.path.join(os.environ.get("HF_HOME", ""), "hub") if os.environ.get("HF_HOME") else None,
        "/home/.cache/huggingface/hub",
        os.path.join(os.path.expanduser("~/.cache/huggingface/hub")),
    ):
        if not candidate:
            continue
        path = Path(candidate)
        if path not in roots:
            roots.append(path)
    return roots


def _resolve_default_gemma_speculative_assistant_model() -> str:
    """Prefer a cached Gemma assistant snapshot, fall back to the HF model id."""
    override = os.environ.get("CASCADE_GEMMA_ASSISTANT_SNAPSHOT")
    if override:
        return override

    for root in _hf_hub_roots():
        snapshots_root = (
            root
            / "models--google--gemma-4-E4B-it-assistant"
            / "snapshots"
        )
        if not snapshots_root.is_dir():
            continue
        snapshots = sorted(
            (path for path in snapshots_root.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if snapshots:
            return str(snapshots[0])
    return GEMMA4_E4B_SPECULATIVE_ASSISTANT_MODEL_ID


def _milmmt_language_name(language: str) -> str:
    if language in {"Simplified Chinese", "Chinese"}:
        return "Chinese (Simplified)"
    return language


class MiLMMTVLLMMTBackend(BaseMTBackend):
    backend_name = "milmmt_vllm_alignatt"
    model_family = "MiLMMT"
    context_config_attr = "mt_max_model_len"

    def __init__(self, *, model_name: str, runtime_config: SimpleNamespace):
        super().__init__(model_name=model_name, runtime_config=runtime_config)
        self.llm = None
        self.policy: AlignAttDecoderPolicy | None = None
        self.alignatt_heads: list[Any] = []
        self.enforce_eager = bool(
            getattr(runtime_config, "mt_vllm_enforce_eager", False)
        )
        self.enable_prefix_caching = bool(
            getattr(runtime_config, "mt_vllm_enable_prefix_caching", False)
        )
        self.cudagraph_mode = getattr(runtime_config, "mt_vllm_cudagraph_mode", "full")
        self.gpu_memory_utilization = float(
            getattr(runtime_config, "mt_vllm_gpu_memory_utilization", 0.5)
        )
        self.enable_speculative_decoding = bool(
            getattr(runtime_config, "mt_vllm_enable_speculative_decoding", False)
        )
        self.num_speculative_tokens = int(
            getattr(runtime_config, "mt_vllm_num_speculative_tokens", 4)
        )
        self.speculative_assistant_model = getattr(
            runtime_config, "mt_vllm_speculative_assistant_model", None
        )
        # Observer sizing: max_prompt_tokens is bounded by the active MT
        # context cap,
        # max_decode_tokens by the per-run generation cap (we pick the larger of
        # the full and partial caps so one configuration covers both calls).
        self.max_prompt_tokens = int(
            getattr(runtime_config, self.context_config_attr, 1024)
        )
        self.max_decode_tokens = max(
            int(getattr(runtime_config, "max_new_tokens", 160)),
            int(getattr(runtime_config, "partial_max_new_tokens", 48)),
        )
        self._last_generated_token_ids: list[int] | None = None

    # -- load ---------------------------------------------------------------
    def _build_compilation_config(self) -> dict[str, Any] | None:
        if self.enforce_eager or self.cudagraph_mode is None:
            return None
        return {"cudagraph_mode": str(self.cudagraph_mode)}

    def _resolve_speculative_assistant_model(self) -> str | None:
        configured = getattr(
            self.runtime_config,
            "mt_vllm_speculative_assistant_model",
            self.speculative_assistant_model,
        )
        if configured:
            return str(configured)
        if self.model_family == "Gemma":
            return _resolve_default_gemma_speculative_assistant_model()
        return None

    def build_speculative_config(self) -> dict[str, Any] | None:
        enabled = bool(
            getattr(
                self.runtime_config,
                "mt_vllm_enable_speculative_decoding",
                self.enable_speculative_decoding,
            )
        )
        if not enabled:
            return None
        num_tokens = int(
            getattr(
                self.runtime_config,
                "mt_vllm_num_speculative_tokens",
                self.num_speculative_tokens,
            )
        )
        if num_tokens < 1:
            raise ValueError(
                "mt_vllm_num_speculative_tokens must be >= 1 when speculative "
                f"decoding is enabled, got {num_tokens!r}."
            )
        assistant_model = self._resolve_speculative_assistant_model()
        if not assistant_model:
            raise ValueError(
                "MT speculative decoding requires "
                "mt_vllm_speculative_assistant_model for this backend."
            )
        return {
            "model": assistant_model,
            "num_speculative_tokens": num_tokens,
        }

    def build_llm_init_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "max_model_len": int(self.max_prompt_tokens),
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enforce_eager": self.enforce_eager,
            "enable_prefix_caching": self.enable_prefix_caching,
            "compilation_config": self._build_compilation_config(),
            "worker_cls": "cascade.mt.gemma_vllm_worker.GemmaVLLMMTWorker",
        }
        speculative_config = self.build_speculative_config()
        if speculative_config is not None:
            kwargs["speculative_config"] = speculative_config
        return kwargs

    def build_sampling_params_kwargs(
        self,
        *,
        max_new_tokens: int,
        stop_token_ids: list[int],
    ) -> dict[str, Any]:
        return {
            "temperature": 0.0,
            "top_k": 1,
            "max_tokens": int(max_new_tokens),
            "repetition_penalty": float(
                getattr(self.runtime_config, "repetition_penalty", 1.0)
            ),
            "stop_token_ids": stop_token_ids or None,
            "skip_special_tokens": False,
        }

    def load(self) -> None:
        from transformers import AutoTokenizer

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                local_files_only=True,
            )

        if not self.alignatt_heads:
            heads_path = getattr(self.runtime_config, "translation_alignatt_heads_path", None)
            top_k = int(getattr(self.runtime_config, "translation_alignatt_top_k_heads", 8))
            if heads_path:
                self.alignatt_heads = load_alignatt_heads(heads_path, top_k=top_k)
                self._validate_alignatt_heads()
        if not self.alignatt_heads:
            raise RuntimeError(
                f"{type(self).__name__} requires translation_alignatt_heads_path "
                "to load MT AlignAtt heads."
            )

        if self.llm is None:
            from vllm import LLM

            bootstrap_prev = os.environ.get(_MT_OBSERVER_BOOTSTRAP_ENV)
            os.environ[_MT_OBSERVER_BOOTSTRAP_ENV] = _encode_mt_observer_bootstrap(
                selected_heads=[
                    {"layer": int(h.layer), "head": int(h.head)}
                    for h in self.alignatt_heads
                ],
                max_prompt_tokens=int(self.max_prompt_tokens),
                max_decode_tokens=int(self.max_decode_tokens),
            )
            # vLLM gates callable-form collective_rpc behind this switch; the
            # MT worker bootstrap needs it for the same reason the ASR one does.
            os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
            try:
                llm_kwargs = self.build_llm_init_kwargs()
                self.llm = LLM(**llm_kwargs)
            finally:
                if bootstrap_prev is None:
                    os.environ.pop(_MT_OBSERVER_BOOTSTRAP_ENV, None)
                else:
                    os.environ[_MT_OBSERVER_BOOTSTRAP_ENV] = bootstrap_prev

        if self.policy is None:
            self.policy = AlignAttDecoderPolicy(
                tokenizer=self.tokenizer,
                runtime_config=self.runtime_config,
            )
        speculative_config = self.build_speculative_config()
        print(
            f"[{self.backend_name}] MT backend loaded; "
            f"heads={len(self.alignatt_heads)} "
            f"max_prompt_tokens={self.max_prompt_tokens} "
            f"max_decode_tokens={self.max_decode_tokens} "
            f"enforce_eager={self.enforce_eager} "
            f"cudagraph_mode={self.cudagraph_mode} "
            f"enable_prefix_caching={self.enable_prefix_caching} "
            f"speculative_decoding={speculative_config is not None} "
            f"num_speculative_tokens="
            f"{None if speculative_config is None else speculative_config['num_speculative_tokens']}",
            flush=True,
        )

    def reset_caches(self) -> None:
        self._last_generated_token_ids = None

    def refresh_alignatt_artifacts(self) -> None:
        heads_path = getattr(self.runtime_config, "translation_alignatt_heads_path", None)
        top_k = int(getattr(self.runtime_config, "translation_alignatt_top_k_heads", 8))
        if heads_path:
            self.alignatt_heads = load_alignatt_heads(heads_path, top_k=top_k)
            self._validate_alignatt_heads()

    def _validate_alignatt_heads(self) -> None:
        invalid: list[dict[str, int]] = []
        for head in self.alignatt_heads:
            layer = int(head.layer)
            head_idx = int(head.head)
            if not (0 <= layer < MILMMT_NUM_TEXT_LAYERS) or not (
                0 <= head_idx < MILMMT_NUM_ATTENTION_HEADS
            ):
                invalid.append({"layer": layer, "head": head_idx})
        if invalid:
            raise ValueError(
                f"{self.model_family} MT AlignAtt heads must satisfy "
                f"0 <= layer < {MILMMT_NUM_TEXT_LAYERS} and "
                f"0 <= head < {MILMMT_NUM_ATTENTION_HEADS}; invalid={invalid}"
            )

    def _render_milmmt_prompt_text(
        self,
        rendered_prompt: RenderedTranslationPrompt,
    ) -> tuple[str, tuple[int, int]]:
        source_lang = _milmmt_language_name(
            str(getattr(self.runtime_config, "source_lang", "English"))
        )
        target_lang = _milmmt_language_name(
            str(getattr(self.runtime_config, "target_lang", "German"))
        )
        source_text = rendered_prompt.source_text
        prefix = (
            f"Translate this from {source_lang} to {target_lang}:\n"
            f"{source_lang}: "
        )
        suffix = f"\n{target_lang}:{rendered_prompt.assistant_prefill}"
        source_start = len(prefix)
        prompt_text = f"{prefix}{source_text}{suffix}"
        return prompt_text, (source_start, source_start + len(source_text))

    def render_prompt_token_ids(self, rendered_prompt: RenderedTranslationPrompt) -> list[int]:
        if self.tokenizer is None:
            raise RuntimeError(f"{self.model_family} tokenizer is not loaded. Run load() first.")
        prompt_text, _ = self._render_milmmt_prompt_text(rendered_prompt)
        return list(
            self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        )

    def render_prompt_text(self, rendered_prompt: RenderedTranslationPrompt) -> str:
        prompt_text, _ = self._render_milmmt_prompt_text(rendered_prompt)
        return prompt_text

    def render_prompt_package(
        self,
        rendered_prompt: RenderedTranslationPrompt,
    ) -> RenderedPromptWithSourceMap:
        if self.tokenizer is None:
            raise RuntimeError(f"{self.model_family} tokenizer is not loaded. Run load() first.")
        prompt_text, source_char_span = self._render_milmmt_prompt_text(rendered_prompt)
        prompt_token_ids = tuple(
            self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        )
        source_map = self._build_milmmt_source_map(
            rendered_prompt=rendered_prompt,
            prompt_text=prompt_text,
            source_char_span=source_char_span,
        )
        return RenderedPromptWithSourceMap(
            prompt_token_ids=prompt_token_ids,
            prompt_text=prompt_text,
            source_map=source_map,
        )

    def _build_milmmt_source_map(
        self,
        *,
        rendered_prompt: RenderedTranslationPrompt,
        prompt_text: str,
        source_char_span: tuple[int, int],
    ) -> PromptSourceMap | None:
        if self.tokenizer is None:
            raise RuntimeError(f"{self.model_family} tokenizer is not loaded. Run load() first.")
        source_frontier = rendered_prompt.source_frontier
        if source_frontier is None or not rendered_prompt.source_text:
            return None

        source_char_start, source_char_end = source_char_span
        prompt_offsets = self.tokenizer(
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

    # -- helpers for collective_rpc ----------------------------------------
    def _prepare_mt_observer(self, *, prompt_length: int) -> dict[str, Any]:
        assert self.llm is not None
        results = self.llm.collective_rpc(
            "prepare_mt_observer",
            args=(int(prompt_length),),
        )
        if len(results) != 1:
            raise RuntimeError(
                f"Expected a single vLLM worker, got {len(results)} observer preparations."
            )
        return results[0]

    def _fetch_mt_observer_payload(self) -> dict[str, Any] | None:
        assert self.llm is not None
        payloads = self.llm.collective_rpc("fetch_mt_observer_payload")
        if len(payloads) != 1:
            raise RuntimeError(
                f"Expected a single vLLM worker, got {len(payloads)} observer payloads."
            )
        return payloads[0]

    # -- translate ----------------------------------------------------------
    def translate(
        self,
        *,
        rendered_prompt: RenderedTranslationPrompt,
        variant: TranslationVariant,
        is_partial: bool,
        prompt_cache_state: PromptCacheState | None = None,
    ) -> MTBackendResult:
        if self.llm is None or self.tokenizer is None or self.policy is None:
            raise RuntimeError("vLLM MT backend is not loaded. Run load() first.")

        from vllm import SamplingParams

        total_start = perf_counter()

        prompt_render_start = perf_counter()
        prompt_package = self.render_prompt_package(rendered_prompt)
        prompt_render_ms = (perf_counter() - prompt_render_start) * 1000.0

        prompt_token_ids = list(prompt_package.prompt_token_ids)
        prompt_length = len(prompt_token_ids)
        if prompt_length > self.max_prompt_tokens:
            raise RuntimeError(
                f"Rendered MT prompt is {prompt_length} tokens, exceeds the "
                f"observer's max_prompt_tokens={self.max_prompt_tokens}. Lower "
                "the prompt or raise the config cap."
            )

        max_new_tokens = self.compute_max_tokens(
            prompt_tokens=prompt_length,
            source_text=rendered_prompt.source_text,
            is_partial=is_partial,
            assistant_prefill=rendered_prompt.assistant_prefill,
        )
        if max_new_tokens > self.max_decode_tokens:
            raise RuntimeError(
                f"compute_max_tokens returned {max_new_tokens} but observer was "
                f"configured for max_decode_tokens={self.max_decode_tokens}. "
                "Increase runtime_config.max_new_tokens before load()."
            )

        stop_token_ids = list(self.resolve_generation_stop_token_ids())
        sampling_params = SamplingParams(
            **self.build_sampling_params_kwargs(
                max_new_tokens=max_new_tokens,
                stop_token_ids=stop_token_ids,
            )
        )

        prepare_start = perf_counter()
        prepare_diag = self._prepare_mt_observer(prompt_length=prompt_length)
        prepare_ms = (perf_counter() - prepare_start) * 1000.0

        generate_start = perf_counter()
        outputs = self.llm.generate(
            [{"prompt_token_ids": prompt_token_ids}],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        generate_ms = (perf_counter() - generate_start) * 1000.0

        if not outputs or not outputs[0].outputs:
            raise RuntimeError("vLLM MT produced no completion output.")

        completion = outputs[0].outputs[0]
        special_ids = {
            int(tid)
            for tid in (getattr(self.tokenizer, "all_special_ids", None) or [])
        }
        raw_ids = [int(tid) for tid in completion.token_ids]
        # Keep the "raw" generated id sequence (including any trailing stop
        # token) for observer alignment: the observer captured Q for every
        # token the model emitted, so n_generated in the observer == len(raw_ids).
        observer_raw_count = len(raw_ids)
        # For the returned *draft* sequence, trim trailing specials so the
        # text surface matches the Transformers MT backend.
        trimmed_ids = list(raw_ids)
        while trimmed_ids and trimmed_ids[-1] in special_ids:
            trimmed_ids.pop()
        draft_generated_ids = trimmed_ids
        prompt_num_tokens = len(outputs[0].prompt_token_ids or prompt_token_ids)
        finish_reason = completion.finish_reason

        draft_text = self.decode_candidate_text(
            generated_ids=draft_generated_ids,
            assistant_prefill=rendered_prompt.assistant_prefill,
            variant=variant,
            is_partial=is_partial,
        )
        draft_token_ids = self.encode_semantic_target_token_ids(draft_text)

        if not is_partial:
            accepted_generated_token_ids = tuple(int(tid) for tid in draft_generated_ids)
            acceptance_text = draft_text
            accepted_token_ids = draft_token_ids
            alignatt_metadata = None
            stop_reason: str | int | None = finish_reason
            timings_ms = {
                "prompt_render": prompt_render_ms,
                "prepare_observer": prepare_ms,
                "generate": generate_ms,
                "total": (perf_counter() - total_start) * 1000.0,
            }
            return MTBackendResult(
                draft_text=draft_text,
                acceptance_text=acceptance_text,
                draft_generated_token_ids=tuple(int(tid) for tid in draft_generated_ids),
                accepted_generated_token_ids=accepted_generated_token_ids,
                draft_token_ids=draft_token_ids,
                accepted_token_ids=accepted_token_ids,
                num_cached_tokens=None,
                prompt_num_tokens=prompt_num_tokens,
                stop_reason=stop_reason,
                alignatt_metadata=alignatt_metadata,
                timings_ms=timings_ms,
            )

        acceptance_policy = str(
            getattr(self.runtime_config, "translation_acceptance_policy", "alignatt")
        )
        if acceptance_policy == "cut_last_target_units":
            cutoff_units = int(
                getattr(self.runtime_config, "translation_static_cutoff_units", 0)
            )
            accepted_generated_token_ids = tuple(
                int(tid)
                for tid in self.policy.cut_last_target_stability_units(
                    draft_generated_ids,
                    cutoff_units=cutoff_units,
                )
            )
            acceptance_text = self.decode_candidate_text(
                generated_ids=accepted_generated_token_ids,
                assistant_prefill=rendered_prompt.assistant_prefill,
                variant=variant,
                is_partial=True,
            )
            accepted_token_ids = self.encode_semantic_target_token_ids(acceptance_text)
            alignatt_metadata = {
                "acceptance_policy": acceptance_policy,
                "static_cutoff_units": cutoff_units,
                "accepted_candidate_token_count": len(draft_generated_ids),
                "accepted_token_count": len(accepted_generated_token_ids),
                "stop_reason": finish_reason,
            }
            timings_ms = {
                "prompt_render": prompt_render_ms,
                "prepare_observer": prepare_ms,
                "generate": generate_ms,
                "total": (perf_counter() - total_start) * 1000.0,
            }
            return MTBackendResult(
                draft_text=draft_text,
                acceptance_text=acceptance_text,
                draft_generated_token_ids=tuple(int(tid) for tid in draft_generated_ids),
                accepted_generated_token_ids=accepted_generated_token_ids,
                draft_token_ids=draft_token_ids,
                accepted_token_ids=accepted_token_ids,
                num_cached_tokens=None,
                prompt_num_tokens=prompt_num_tokens,
                stop_reason=finish_reason,
                alignatt_metadata=alignatt_metadata,
                timings_ms=timings_ms,
            )
        if acceptance_policy != "alignatt":
            raise ValueError(f"Unknown translation_acceptance_policy: {acceptance_policy!r}")

        # Partial translation: fetch observer and run the AlignAtt policy.
        fetch_start = perf_counter()
        capture_payload = self._fetch_mt_observer_payload()
        fetch_ms = (perf_counter() - fetch_start) * 1000.0

        source_map = prompt_package.source_map
        if source_map is None or not source_map.source_token_positions:
            # No source map means we can't align anything. Treat this like the
            # Transformers backend's degenerate case: accept the whole draft.
            accepted_generated_token_ids = tuple(int(tid) for tid in draft_generated_ids)
            acceptance_text = draft_text
            accepted_token_ids = draft_token_ids
            alignatt_metadata = {
                "alignatt_degenerate_no_source_map": True,
                "stop_reason": finish_reason,
                "accepted_candidate_token_count": len(draft_generated_ids),
                "accepted_token_count": len(accepted_generated_token_ids),
            }
            stop_reason = finish_reason
            timings_ms = {
                "prompt_render": prompt_render_ms,
                "prepare_observer": prepare_ms,
                "generate": generate_ms,
                "fetch_observer": fetch_ms,
                "total": (perf_counter() - total_start) * 1000.0,
            }
            return MTBackendResult(
                draft_text=draft_text,
                acceptance_text=acceptance_text,
                draft_generated_token_ids=tuple(int(tid) for tid in draft_generated_ids),
                accepted_generated_token_ids=accepted_generated_token_ids,
                draft_token_ids=draft_token_ids,
                accepted_token_ids=accepted_token_ids,
                num_cached_tokens=None,
                prompt_num_tokens=prompt_num_tokens,
                stop_reason=stop_reason,
                alignatt_metadata=alignatt_metadata,
                timings_ms=timings_ms,
            )

        reconstruct_start = perf_counter()
        reconstruction = reconstruct_mt_attention_rows(
            capture_payload,
            alignatt_heads=self.alignatt_heads,
            source_positions=source_map.source_token_positions,
            accessible_source_token_count=source_map.accessible_source_token_count,
        )
        reconstruct_ms = (perf_counter() - reconstruct_start) * 1000.0

        source_rows = reconstruction.source_attention_rows_per_token
        provenance_mass = reconstruction.provenance_mass_per_token

        # The observer captured Q for every token the model emitted, including
        # any trailing stop token that the Transformers backend would never
        # have generated (its ``.generate(...)`` path stops *before* emitting
        # EOS). To keep policy decisions and the reported provenance surface
        # semantically equivalent between the two backends, we operate only
        # over the trimmed draft length and drop observer state for the
        # trailing stop tokens.
        observer_token_count = len(source_rows)
        draft_token_count = len(draft_generated_ids)
        operating_count = min(observer_token_count, draft_token_count)
        operating_ids = draft_generated_ids[:operating_count]
        source_rows = source_rows[:operating_count]
        provenance_mass = provenance_mass[:operating_count]

        if operating_count == 0 or not source_rows:
            # Observer produced no signal — stay conservative and accept nothing
            # new. The runtime will reuse the previously-accepted prefix.
            accepted_candidate_ids: list[int] = []
            unsafe_reason: str | None = "observer_empty"
            unsafe_target_token_index = 0
            unsafe_token_id = None
            blocked_source_local_position = None
            blocked_source_unit_index = None
            aligned_source_local_positions: list[int | None] = []
            stop_reason = "alignatt:observer_empty"
        else:
            aligned_source_local_positions = compute_prefix_online_alignatt_source_argmaxes(
                source_rows,
                filter_width=self.policy.alignatt_filter_width(),
            )
            argmax_mass_threshold = float(
                getattr(
                    self.runtime_config,
                    "translation_alignatt_argmax_mass_threshold",
                    0.0,
                )
            )
            argmax_raw_mass_per_token: list[float | None] = []
            for token_index, row_tensor in enumerate(source_rows):
                if token_index >= len(aligned_source_local_positions):
                    argmax_raw_mass_per_token.append(None)
                    continue
                pos = aligned_source_local_positions[token_index]
                if pos is None or row_tensor.shape[-1] <= 0:
                    argmax_raw_mass_per_token.append(None)
                    continue
                argmax_raw_mass_per_token.append(
                    float(row_tensor[..., int(pos)].mean().item())
                )
            accepted_candidate_ids = []
            unsafe_reason = None
            unsafe_target_token_index = None
            unsafe_token_id = None
            blocked_source_local_position = None
            blocked_source_unit_index = None
            stop_reason = finish_reason
            min_source_mass = float(
                getattr(self.runtime_config, "translation_alignatt_min_source_mass", 0.0)
            )
            for token_index, (token_id, current_source_local_position) in enumerate(
                zip(operating_ids, aligned_source_local_positions)
            ):
                source_accessible_mass: float | None = None
                source_inaccessible_mass: float | None = None
                if token_index < len(provenance_mass):
                    source_accessible_mass = float(provenance_mass[token_index][0])
                    source_inaccessible_mass = float(provenance_mass[token_index][1])
                unsafe_reason, _ = self.policy.should_stop_in_loop(
                    current_source_local_position=current_source_local_position,
                    accessible_source_token_count=source_map.accessible_source_token_count,
                    source_inaccessible_mass=source_inaccessible_mass,
                )
                if unsafe_reason == "source_frontier":
                    unsafe_target_token_index = token_index
                    unsafe_token_id = int(token_id)
                    blocked_source_local_position = current_source_local_position
                    blocked_source_unit_index = source_local_position_to_unit_index(
                        source_map, current_source_local_position
                    )
                    stop_reason = "alignatt:source_frontier"
                    break
                if (
                    argmax_mass_threshold > 0.0
                    and token_index < len(argmax_raw_mass_per_token)
                    and argmax_raw_mass_per_token[token_index] is not None
                    and argmax_raw_mass_per_token[token_index] < argmax_mass_threshold
                ):
                    unsafe_reason = "argmax_mass_weak"
                    unsafe_target_token_index = token_index
                    unsafe_token_id = int(token_id)
                    stop_reason = "alignatt:argmax_mass_weak"
                    break
                if (
                    min_source_mass > 0.0
                    and source_accessible_mass is not None
                    and source_accessible_mass < min_source_mass
                ):
                    unsafe_reason = "provenance_weak"
                    unsafe_target_token_index = token_index
                    unsafe_token_id = int(token_id)
                    stop_reason = "alignatt:provenance_weak"
                    break
                if (
                    source_accessible_mass is not None
                    and source_inaccessible_mass is not None
                ):
                    provenance_stop_reason = self.policy.should_stop_for_provenance_mass(
                        source_accessible_mass=source_accessible_mass,
                        source_inaccessible_mass=source_inaccessible_mass,
                    )
                    if provenance_stop_reason is not None:
                        unsafe_reason = provenance_stop_reason
                        unsafe_target_token_index = token_index
                        unsafe_token_id = int(token_id)
                        stop_reason = f"alignatt:{provenance_stop_reason}"
                        break
                accepted_candidate_ids.append(int(token_id))

        acceptance = self.policy.finalize_partial(
            accepted_candidate_ids=accepted_candidate_ids,
            aligned_source_local_positions=aligned_source_local_positions,
            source_map=source_map,
            unsafe_reason=unsafe_reason,
            unsafe_target_token_index=unsafe_target_token_index,
            unsafe_token_id=unsafe_token_id,
            blocked_source_local_position=blocked_source_local_position,
            blocked_source_unit_index=blocked_source_unit_index,
            stop_reason=stop_reason,
            probe_backend="vllm_mt_observer",
        )
        accepted_generated_token_ids = tuple(
            int(tid) for tid in acceptance.accepted_generated_ids
        )
        acceptance_text = self.decode_candidate_text(
            generated_ids=accepted_generated_token_ids,
            assistant_prefill=rendered_prompt.assistant_prefill,
            variant=variant,
            is_partial=True,
        )
        accepted_token_ids = self.encode_semantic_target_token_ids(acceptance_text)
        alignatt_metadata = dict(acceptance.alignatt_metadata or {})
        if provenance_mass:
            alignatt_metadata["provenance_per_draft_token"] = [
                {
                    "source_accessible": row[0],
                    "source_inaccessible": row[1],
                    "non_source_prompt": row[2],
                    "suffix": row[3],
                }
                for row in provenance_mass
            ]
        alignatt_metadata["observer_diagnostics"] = reconstruction.diagnostics
        alignatt_metadata["observer_raw_token_count"] = observer_raw_count
        alignatt_metadata["observer_operating_token_count"] = operating_count
        alignatt_metadata["prepare_diagnostics"] = prepare_diag
        if capture_payload is not None:
            alignatt_metadata["observer_debug"] = capture_payload.get("debug", {})

        total_ms = (perf_counter() - total_start) * 1000.0
        timings_ms = {
            "prompt_render": prompt_render_ms,
            "prepare_observer": prepare_ms,
            "generate": generate_ms,
            "fetch_observer": fetch_ms,
            "reconstruct": reconstruct_ms,
            "total": total_ms,
        }

        return MTBackendResult(
            draft_text=draft_text,
            acceptance_text=acceptance_text,
            draft_generated_token_ids=tuple(int(tid) for tid in draft_generated_ids),
            accepted_generated_token_ids=accepted_generated_token_ids,
            draft_token_ids=draft_token_ids,
            accepted_token_ids=accepted_token_ids,
            num_cached_tokens=None,
            prompt_num_tokens=prompt_num_tokens,
            stop_reason=acceptance.alignatt_metadata.get("stop_reason", stop_reason)
            if acceptance.alignatt_metadata
            else stop_reason,
            alignatt_metadata=alignatt_metadata,
            timings_ms=timings_ms,
        )


class GemmaVLLMMTBackend(MiLMMTVLLMMTBackend):
    """Submitted Gemma-4 E4B-it MT AlignAtt backend.

    Gemma and MiLMMT share the vLLM Q/K observer mechanics, but Gemma keeps the
    chat-template prompt contract used by the submitted system and paper.
    """

    backend_name = "gemma_vllm_alignatt"
    model_family = "Gemma"
    context_config_attr = "gemma_max_model_len"

    def _validate_alignatt_heads(self) -> None:
        return None

    def render_prompt_token_ids(
        self,
        rendered_prompt: RenderedTranslationPrompt,
    ) -> list[int]:
        return BaseMTBackend.render_prompt_token_ids(self, rendered_prompt)

    def render_prompt_text(self, rendered_prompt: RenderedTranslationPrompt) -> str:
        return BaseMTBackend.render_prompt_text(self, rendered_prompt)

    def render_prompt_package(
        self,
        rendered_prompt: RenderedTranslationPrompt,
    ) -> RenderedPromptWithSourceMap:
        return BaseMTBackend.render_prompt_package(self, rendered_prompt)
