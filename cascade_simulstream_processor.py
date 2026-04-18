"""SimulStream SpeechProcessor wrapping the instantiable AlignAtt cascade."""

from __future__ import annotations

from types import SimpleNamespace
from typing import List

from simulstream.server.speech_processors import SpeechProcessor
from simulstream.server.speech_processors.incremental_output import IncrementalOutput

from cascade_incremental_output import (
    append_only_incremental_output,
    empty_incremental_output,
)
from cascade_runtime import (
    LANGUAGE_CODE_TO_NAME,
    LANGUAGE_NAME_TO_CODE,
    CascadeRuntimeConfig,
    LoadedModelBundle,
    alignatt_heads_path_for,
)
from cascade_text_surface import join_public_emission_units, split_public_emission_units


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
            "translation_alignatt_rewind_threshold",
            "translation_alignatt_border_margin",
            "translation_alignatt_min_source_mass",
            "translation_alignatt_argmax_mass_threshold",
            "translation_alignatt_top_k_heads",
            "translation_alignatt_filter_width",
            "translation_alignatt_probe_mode",
            "temperature",
            "repetition_penalty",
            "gemma_audio_align_probe_mode",
            "gemma_audio_alignment_heads_path",
            "gemma_audio_alignment_top_k_heads",
            "gemma_audio_alignment_filter_width",
            "gemma_audio_alignment_max_new_tokens",
            "asr_streaming_prefix_enabled",
            "asr_streaming_rollback_words",
            "asr_streaming_unfixed_chunks",
            "gemma_vllm_force_generate_api",
            "mt_vllm_enforce_eager",
            "mt_vllm_cudagraph_mode",
            "mt_vllm_enable_prefix_caching",
            "mt_vllm_gpu_memory_utilization",
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
            runtime_config.gemma_audio_alignment_heads_path,
            runtime_config.gemma_audio_align_probe_mode,
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
        )
        self._session.bundle.ensure_mt_backend()

    def set_target_language(self, language: str) -> None:
        lang_name, lang_code = _resolve_language(language)
        self._runtime_config.target_lang = lang_name
        self._target_lang_code = lang_code
        self._runtime_config.translation_alignatt_heads_path = alignatt_heads_path_for(
            self._runtime_config.source_lang,
            lang_name,
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

    def _current_emitted_text(self) -> str:
        return self.tokens_to_string(self._emitted_units)

    def _compute_incremental_output(self, new_translation: str) -> IncrementalOutput:
        new_units = split_public_emission_units(
            new_translation, target_lang_code=self._target_lang_code
        )
        previous_units = self._emitted_units
        if not new_units:
            return empty_incremental_output()
        if len(new_units) < len(previous_units):
            return empty_incremental_output()
        if new_units[: len(previous_units)] != previous_units:
            return empty_incremental_output()

        added_units = new_units[len(previous_units) :]
        if not added_units:
            return empty_incremental_output()

        self._emitted_units = list(new_units)
        added_string = self.tokens_to_string(added_units)
        return append_only_incremental_output(
            new_tokens=[added_string],
            new_string=added_string,
        )
