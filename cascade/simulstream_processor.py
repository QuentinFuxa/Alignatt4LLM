"""SimulStream SpeechProcessor wrapping the instantiable AlignAtt cascade."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List

from simulstream.server.speech_processors import SpeechProcessor
from simulstream.server.speech_processors.incremental_output import IncrementalOutput

from cascade.incremental_output import (
    append_only_incremental_output,
    empty_incremental_output,
)
from cascade.runtime import (
    LANGUAGE_CODE_TO_NAME,
    LANGUAGE_NAME_TO_CODE,
    CascadeRuntimeConfig,
    LoadedModelBundle,
    alignatt_heads_path_for,
)
from cascade.text_surface import join_public_emission_units, split_public_emission_units


def _resolve_language(language: str) -> tuple[str, str]:
    """Accept either a language name ("Simplified Chinese") or code ("zh")
    and return (canonical_name, canonical_code). The Docker path feeds the
    human-readable name from simulstream's --tgt-lang / --src-lang flags,
    whereas the Python batch path feeds the code directly."""
    if language in LANGUAGE_NAME_TO_CODE:
        return language, LANGUAGE_NAME_TO_CODE[language]
    if language in LANGUAGE_CODE_TO_NAME:
        return LANGUAGE_CODE_TO_NAME[language], language
    return language, language


class CascadeAlignAttProcessor(SpeechProcessor):
    """SimulStream processor backed by a per-instance ``CascadeSession``."""

    _bundle: LoadedModelBundle | None = None
    _bundle_signature: tuple | None = None

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self._runtime_config = self._build_runtime_config(config)
        self._default_paper_context_path = self._runtime_config.paper_context_path
        self._chunk_ms = int(getattr(config, "chunk_ms", 450))
        self._target_lang_code = getattr(config, "target_lang_code", "de")
        self._source_lang_code = getattr(config, "source_lang_code", "en")
        bundle = type(self)._ensure_bundle(self._runtime_config)
        self._session = bundle.new_session()
        self._emitted_units: list[str] = []
        self._emission_events: list[dict[str, Any]] = []

    @staticmethod
    def _build_runtime_config(config: SimpleNamespace) -> CascadeRuntimeConfig:
        source_lang = LANGUAGE_CODE_TO_NAME.get(
            getattr(config, "source_lang_code", "en"), "English"
        )
        target_lang = LANGUAGE_CODE_TO_NAME.get(
            getattr(config, "target_lang_code", "de"), "German"
        )
        runtime_config = CascadeRuntimeConfig(
            source_lang=source_lang,
            target_lang=target_lang,
            alignment_backend_name=str(
                getattr(config, "alignment_backend_name", "qwen_forced")
            ),
            mt_backend_name=str(
                getattr(config, "mt_backend_name", "gemma_vllm_alignatt")
            ),
        )
        override_keys = [
            "min_start_seconds",
            "max_history_utterances",
            "partial_max_new_tokens",
            "translation_alignatt_inaccessible_ms",
            "translation_alignatt_source_lcp_stability",
            "translation_alignatt_source_lcp_append_slack_units",
            "translation_alignatt_acceptance_variant",
            "translation_alignatt_online_normalization",
            "translation_alignatt_border_margin",
            "translation_alignatt_min_source_mass",
            "translation_alignatt_frontier_min_inaccessible_mass",
            "translation_alignatt_source_frontier_action",
            "translation_alignatt_max_inaccessible_source_mass",
            "translation_alignatt_max_non_source_prompt_mass",
            "translation_alignatt_min_accessible_inaccessible_margin",
            "translation_alignatt_min_accepted_accessible_source_mass",
            "translation_alignatt_accepted_accessible_source_mass_recent_units",
            "translation_alignatt_argmax_mass_threshold",
            "translation_alignatt_min_accessible_source_units",
            "translation_alignatt_min_accessible_source_units_mode",
            "translation_alignatt_hold_back_target_units",
            "translation_alignatt_min_emit_target_units",
            "translation_alignatt_max_source_regression",
            "translation_alignatt_source_regression_min_source_mass",
            "translation_alignatt_source_regression_min_inaccessible_mass",
            "translation_alignatt_source_regression_recent_tokens",
            "translation_alignatt_source_regression_reference_mode",
            "translation_alignatt_source_regression_activation_mode",
            "translation_alignatt_source_regression_activation_slack_tokens",
            "translation_alignatt_source_regression_patience_tokens",
            "translation_alignatt_source_regression_action",
            "translation_alignatt_unit_consensus_min_head_ratio",
            "translation_alignatt_min_alignment_confidence",
            "translation_alignatt_source_bearing_min_source_mass",
            "translation_alignatt_source_bearing_hard_inaccessible_cap",
            "translation_alignatt_token_argmax_frontier_gate",
            "translation_alignatt_token_argmax_min_source_mass",
            "translation_alignatt_token_argmax_frontier_margin",
            "translation_alignatt_token_argmax_frontier_patience_tokens",
            "translation_alignatt_source_lookback_holdback",
            "translation_alignatt_source_lookback_units",
            "translation_alignatt_source_lookback_min_source_mass",
            "translation_alignatt_source_lookback_min_source_position",
            "translation_alignatt_defer_low_source_terminal_punctuation",
            "translation_alignatt_terminal_punctuation_min_source_mass",
            "translation_alignatt_heads_path",
            "translation_alignatt_top_k_heads",
            "translation_alignatt_filter_width",
            "translation_alignatt_probe_mode",
            "translation_acceptance_policy",
            "translation_static_cutoff_units",
            "temperature",
            "repetition_penalty",
            "gemma_audio_alignment_heads_path",
            "gemma_audio_alignment_top_k_heads",
            "gemma_audio_alignment_filter_width",
            "gemma_audio_alignment_max_new_tokens",
            "gemma_audio_eos_only_rescue_enabled",
            "gemma_audio_eos_only_rescue_max_new_tokens",
            "asr_alignatt_commit_policy",
            "asr_alignatt_frame_threshold",
            "asr_alignatt_rewind_threshold",
            "asr_punctuation_min_commit_words",
            "asr_context_committed_words",
            "mt_vllm_enforce_eager",
            "mt_vllm_cudagraph_mode",
            "mt_vllm_enable_prefix_caching",
            "mt_vllm_gpu_memory_utilization",
            "mt_vllm_enable_speculative_decoding",
            "mt_vllm_speculative_assistant_model",
            "mt_vllm_num_speculative_tokens",
            "gemma_max_model_len",
            "milmmt_prompt_mode",
            "milmmt_prompt_add_bos",
            "milmmt_temperature",
            "milmmt_top_p",
            "milmmt_top_k",
            "milmmt_repetition_penalty",
            "mt_max_model_len",
            "asr_gpu_memory_utilization",
            "gemma_vllm_gpu_memory_utilization",
            "paper_context_path",
            "paper_context_mode",
            "paper_context_top_k",
            "paper_context_max_chars",
            "paper_context_history_window_words",
        ]
        overrides = {
            key: getattr(config, key)
            for key in override_keys
            if getattr(config, key, None) is not None
        }
        runtime_config.apply_overrides(**overrides)
        return runtime_config

    @staticmethod
    def _bundle_key(runtime_config: CascadeRuntimeConfig) -> tuple:
        # paper_context_path is intentionally *not* in the bundle key: the
        # PaperContextSelector is trivially cheap to rebuild from a JSON
        # artifact and its load cost is negligible against a ~5 min ASR +
        # MT model reload. Swapping artifacts (or toggling context modes)
        # must stay hot — LoadedModelBundle.ensure_paper_context_selector
        # refreshes the selector lazily when the path changes.
        return (
            runtime_config.source_lang,
            runtime_config.target_lang,
            runtime_config.alignment_backend_name,
            runtime_config.mt_backend_name,
            runtime_config.translation_alignatt_heads_path,
            runtime_config.translation_alignatt_top_k_heads,
            runtime_config.translation_alignatt_filter_width,
            runtime_config.translation_alignatt_probe_mode,
            runtime_config.mt_max_model_len,
            runtime_config.mt_vllm_enforce_eager,
            runtime_config.mt_vllm_enable_prefix_caching,
            runtime_config.mt_vllm_cudagraph_mode,
            runtime_config.mt_vllm_gpu_memory_utilization,
            runtime_config.mt_vllm_enable_speculative_decoding,
            runtime_config.mt_vllm_speculative_assistant_model,
            runtime_config.mt_vllm_num_speculative_tokens,
            runtime_config.gemma_audio_alignment_heads_path,
            runtime_config.gemma_audio_alignment_top_k_heads,
            runtime_config.gemma_audio_alignment_filter_width,
            runtime_config.gemma_audio_alignment_max_new_tokens,
        )

    @classmethod
    def _ensure_bundle(cls, runtime_config: CascadeRuntimeConfig) -> LoadedModelBundle:
        bundle_key = cls._bundle_key(runtime_config)
        if cls._bundle is None or cls._bundle_signature != bundle_key:
            cls._bundle = LoadedModelBundle(runtime_config)
            cls._bundle.load()
            cls._bundle_signature = bundle_key
        else:
            cls._bundle.config = runtime_config
        return cls._bundle

    @classmethod
    def load_model(cls, config: SimpleNamespace) -> None:
        runtime_config = cls._build_runtime_config(config)
        cls._ensure_bundle(runtime_config)

    @property
    def speech_chunk_size(self) -> float:
        return self._chunk_ms / 1000.0

    @property
    def session(self):
        return self._session

    def process_chunk(self, waveform) -> IncrementalOutput:
        session_result = self._session.process_audio_chunk(waveform)
        if session_result is None:
            return empty_incremental_output()
        translation, _ = self._session.apply_translation_emit_policy(
            self._current_emitted_text(),
            session_result.raw_translation_text,
            is_final=False,
        )
        return self._compute_incremental_output(translation)

    def end_of_stream(self) -> IncrementalOutput:
        final_result = self._session.finalize_stream()
        translation, _ = self._session.apply_translation_emit_policy(
            self._current_emitted_text(),
            final_result.raw_translation_text,
            is_final=True,
        )
        return self._compute_incremental_output(translation)

    def set_source_language(self, language: str) -> None:
        lang_name, lang_code = _resolve_language(language)
        self._runtime_config.source_lang = lang_name
        self._source_lang_code = lang_code
        self._runtime_config.translation_alignatt_heads_path = alignatt_heads_path_for(
            lang_name,
            self._runtime_config.target_lang,
            mt_backend_name=self._runtime_config.mt_backend_name,
        )
        self._session.bundle.ensure_mt_backend()

    def set_target_language(self, language: str) -> None:
        lang_name, lang_code = _resolve_language(language)
        self._runtime_config.target_lang = lang_name
        self._target_lang_code = lang_code
        self._runtime_config.translation_alignatt_heads_path = alignatt_heads_path_for(
            self._runtime_config.source_lang,
            lang_name,
            mt_backend_name=self._runtime_config.mt_backend_name,
        )
        self._session.bundle.ensure_mt_backend()

    def set_paper_context_path(self, path: str | None) -> None:
        self._runtime_config.paper_context_path = path

    def tokens_to_string(self, tokens: List[str]) -> str:
        return join_public_emission_units(tokens, target_lang_code=self._target_lang_code)

    def clear(self) -> None:
        self._runtime_config.paper_context_path = self._default_paper_context_path
        self._session.clear()
        self._emitted_units = []
        self._emission_events = []

    def _current_emitted_text(self) -> str:
        return self.tokens_to_string(self._emitted_units)

    def emission_events(self) -> list[dict[str, Any]]:
        return [dict(event) for event in self._emission_events]

    def _record_emission_event(
        self,
        *,
        reason: str,
        accepted: bool,
        previous_units: list[str],
        candidate_units: list[str],
        candidate_translation: str,
        added_units: list[str] | None = None,
    ) -> None:
        common_prefix_unit_count = 0
        for left, right in zip(previous_units, candidate_units):
            if left != right:
                break
            common_prefix_unit_count += 1
        self._emission_events.append(
            {
                "event_idx": len(self._emission_events),
                "reason": reason,
                "accepted": bool(accepted),
                "previous_emitted_text": self.tokens_to_string(previous_units),
                "candidate_translation": candidate_translation,
                "previous_unit_count": len(previous_units),
                "candidate_unit_count": len(candidate_units),
                "common_prefix_unit_count": common_prefix_unit_count,
                "added_unit_count": 0 if added_units is None else len(added_units),
            }
        )

    def _compute_incremental_output(self, new_translation: str) -> IncrementalOutput:
        new_units = split_public_emission_units(
            new_translation, target_lang_code=self._target_lang_code
        )
        previous_units = list(self._emitted_units)
        if not new_units:
            self._record_emission_event(
                reason="empty_candidate",
                accepted=False,
                previous_units=previous_units,
                candidate_units=new_units,
                candidate_translation=new_translation,
            )
            return empty_incremental_output()
        if len(new_units) < len(previous_units):
            self._record_emission_event(
                reason="candidate_shorter_than_emitted",
                accepted=False,
                previous_units=previous_units,
                candidate_units=new_units,
                candidate_translation=new_translation,
            )
            return empty_incremental_output()
        if new_units[: len(previous_units)] != previous_units:
            self._record_emission_event(
                reason="candidate_not_append_prefix",
                accepted=False,
                previous_units=previous_units,
                candidate_units=new_units,
                candidate_translation=new_translation,
            )
            return empty_incremental_output()

        added_units = new_units[len(previous_units) :]
        if not added_units:
            self._record_emission_event(
                reason="no_new_units",
                accepted=False,
                previous_units=previous_units,
                candidate_units=new_units,
                candidate_translation=new_translation,
                added_units=added_units,
            )
            return empty_incremental_output()

        self._emitted_units = list(new_units)
        self._record_emission_event(
            reason="accepted",
            accepted=True,
            previous_units=previous_units,
            candidate_units=new_units,
            candidate_translation=new_translation,
            added_units=added_units,
        )
        added_string = self.tokens_to_string(added_units)
        return append_only_incremental_output(
            new_tokens=[added_string],
            new_string=added_string,
        )
