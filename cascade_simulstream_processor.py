"""SimulStream SpeechProcessor wrapping the AlignAtt cascade.

This is the canonical delivery path. All speed measurements and final
evaluations should go through this processor, not the research harness.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import List

import numpy as np

from simulstream.server.speech_processors import SpeechProcessor, SAMPLE_RATE
from simulstream.server.speech_processors.incremental_output import IncrementalOutput

from cascade_text_surface import split_target_emission_units

LANGUAGE_CODE_TO_NAME = {
    "en": "English",
    "de": "German",
    "it": "Italian",
    "zh": "Chinese",
}


class CascadeAlignAttProcessor(SpeechProcessor):
    """SimulStream processor backed by the Qwen3-ASR + Gemma AlignAtt cascade."""

    _core = None
    _models_loaded = False

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self._emitted_units: list[str] = []
        self._last_asr: str = ""
        self._last_committed_segments: int = 0
        self._target_lang_code: str = getattr(config, "target_lang_code", "de")
        self._source_lang_code: str = getattr(config, "source_lang_code", "en")
        self._chunk_ms: int = int(getattr(config, "chunk_ms", 450))
        self._apply_runtime_overrides(config)

    @property
    def speech_chunk_size(self) -> float:
        return self._chunk_ms / 1000.0

    def _apply_runtime_overrides(self, config: SimpleNamespace) -> None:
        core = self._get_core()
        override_keys = [
            "min_start_seconds",
            "max_history_utterances",
            "partial_max_new_tokens",
            "partial_followup_max_new_tokens",
            "translation_alignatt_inaccessible_ms",
            "translation_alignatt_rewind_threshold",
            "translation_alignatt_min_source_mass",
            "translation_alignatt_top_k_heads",
            "translation_alignatt_filter_width",
            "translation_alignatt_probe_mode",
            "translation_scheduler_stall_seconds",
            "temperature",
            "repetition_penalty",
        ]
        for key in override_keys:
            value = getattr(config, key, None)
            if value is not None:
                setattr(core.config, key, value)

    @classmethod
    def _get_core(cls):
        if cls._core is None:
            import qwen3asr_gemma_cascade_core as core
            cls._core = core
        return cls._core

    @classmethod
    def load_model(cls, config: SimpleNamespace) -> None:
        core = cls._get_core()
        source_lang = LANGUAGE_CODE_TO_NAME.get(
            getattr(config, "source_lang_code", "en"), "English"
        )
        target_lang = LANGUAGE_CODE_TO_NAME.get(
            getattr(config, "target_lang_code", "de"), "German"
        )
        core.config.source_lang = source_lang
        core.config.target_lang = target_lang
        core.config.translation_alignatt_heads_path = core.alignatt_heads_path_for(
            source_lang, target_lang
        )
        chunk_ms = int(getattr(config, "chunk_ms", 450))
        if hasattr(config, "speech_chunk_size"):
            chunk_ms = int(float(config.speech_chunk_size) * 1000)
        core.load_models()
        cls._models_loaded = True

    def process_chunk(self, waveform: np.float32) -> IncrementalOutput:
        core = self._get_core()
        state = core.state
        state.source = np.concatenate([state.source, np.asarray(waveform, dtype=np.float32)])

        if len(state.source) / SAMPLE_RATE < core.config.min_start_seconds:
            return IncrementalOutput([], "", [], "")

        current_asr = core.transcribe_audio()
        committed_segments = len(state.utt_sources)
        if not current_asr:
            return IncrementalOutput([], "", [], "")
        if (
            current_asr == self._last_asr
            and committed_segments == self._last_committed_segments
        ):
            return IncrementalOutput([], "", [], "")

        self._last_asr = current_asr
        self._last_committed_segments = committed_segments
        raw_translation, _ = core.translation_units.render_translation()
        translation, _ = core.apply_translation_emit_policy(
            self._current_emitted_text(),
            raw_translation,
            is_final=False,
        )
        return self._compute_incremental_output(translation)

    def end_of_stream(self) -> IncrementalOutput:
        core = self._get_core()
        core.transcribe_audio()
        raw_translation, _ = core.translation_units.render_translation()
        translation, _ = core.apply_translation_emit_policy(
            self._current_emitted_text(),
            raw_translation,
            is_final=True,
        )
        return self._compute_incremental_output(translation)

    def set_source_language(self, language: str) -> None:
        core = self._get_core()
        lang_name = LANGUAGE_CODE_TO_NAME.get(language, language)
        core.config.source_lang = lang_name
        self._source_lang_code = language
        core.config.translation_alignatt_heads_path = core.alignatt_heads_path_for(
            lang_name, core.config.target_lang
        )

    def set_target_language(self, language: str) -> None:
        core = self._get_core()
        lang_name = LANGUAGE_CODE_TO_NAME.get(language, language)
        core.config.target_lang = lang_name
        self._target_lang_code = language
        core.config.translation_alignatt_heads_path = core.alignatt_heads_path_for(
            core.config.source_lang, lang_name
        )

    def tokens_to_string(self, tokens: List[str]) -> str:
        if not tokens:
            return ""
        from cascade_text_surface import is_char_level_target_lang
        if is_char_level_target_lang(self._target_lang_code):
            return "".join(tokens)
        return " ".join(tokens)

    def clear(self) -> None:
        core = self._get_core()
        core.clear_state()
        self._emitted_units = []
        self._last_asr = ""
        self._last_committed_segments = 0

    def _current_emitted_text(self) -> str:
        return self.tokens_to_string(self._emitted_units)

    def _compute_incremental_output(self, new_translation: str) -> IncrementalOutput:
        new_units = split_target_emission_units(
            new_translation, target_lang_code=self._target_lang_code
        )
        previous_units = self._emitted_units
        common_prefix_len = 0
        for prev, cur in zip(previous_units, new_units):
            if prev != cur:
                break
            common_prefix_len += 1

        deleted_units = previous_units[common_prefix_len:]
        added_units = new_units[common_prefix_len:]
        self._emitted_units = list(new_units)
        return IncrementalOutput(
            new_tokens=added_units,
            new_string=self.tokens_to_string(added_units),
            deleted_tokens=deleted_units,
            deleted_string=self.tokens_to_string(deleted_units),
        )
