"""Experimental MiLMMT-46 MT backend via vLLM AlignAtt.

MiLMMT-46-4B is a Gemma3-based translation model. It can therefore reuse the
Gemma-family MT observer, but it should not reuse the chat prompt written for
Gemma instruction models. The model card recommends a direct translation
prompt, so this backend renders that prompt explicitly and keeps the current
source span visible to the AlignAtt source-map builder.
"""
from __future__ import annotations

from typing import Any

from cascade.mt.base import (
    RenderedPromptWithSourceMap,
    build_prompt_source_map_from_char_span,
)
from cascade.mt.gemma_vllm_backend import GemmaVLLMMTBackend
from cascade.translation_variants import RenderedTranslationPrompt


MILMMT_PROMPT_MODES = ("direct", "direct_preserve")

MILMMT_LANGUAGE_NAMES = {
    "ar": "Arabic",
    "az": "Azerbaijani",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "ca": "Catalan",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kk": "Kazakh",
    "km": "Khmer",
    "ko": "Korean",
    "lo": "Lao",
    "ms": "Malay",
    "my": "Burmese",
    "nb": "Norwegian",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sv": "Swedish",
    "ta": "Tamil",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "zh": "Chinese (Simplified)",
    "zh-Hant": "Chinese (Traditional)",
    "Arabic": "Arabic",
    "Azerbaijani": "Azerbaijani",
    "Bulgarian": "Bulgarian",
    "Bengali": "Bengali",
    "Catalan": "Catalan",
    "Czech": "Czech",
    "Danish": "Danish",
    "German": "German",
    "Greek": "Greek",
    "English": "English",
    "Spanish": "Spanish",
    "Persian": "Persian",
    "Finnish": "Finnish",
    "French": "French",
    "Hebrew": "Hebrew",
    "Hindi": "Hindi",
    "Croatian": "Croatian",
    "Hungarian": "Hungarian",
    "Indonesian": "Indonesian",
    "Italian": "Italian",
    "Japanese": "Japanese",
    "Kazakh": "Kazakh",
    "Khmer": "Khmer",
    "Korean": "Korean",
    "Lao": "Lao",
    "Malay": "Malay",
    "Burmese": "Burmese",
    "Norwegian": "Norwegian",
    "Dutch": "Dutch",
    "Polish": "Polish",
    "Portuguese": "Portuguese",
    "Romanian": "Romanian",
    "Russian": "Russian",
    "Slovak": "Slovak",
    "Slovenian": "Slovenian",
    "Swedish": "Swedish",
    "Tamil": "Tamil",
    "Thai": "Thai",
    "Tagalog": "Tagalog",
    "Turkish": "Turkish",
    "Urdu": "Urdu",
    "Uzbek": "Uzbek",
    "Vietnamese": "Vietnamese",
    "Cantonese": "Cantonese",
    "Chinese": "Chinese (Simplified)",
    "Simplified Chinese": "Chinese (Simplified)",
    "Traditional Chinese": "Chinese (Traditional)",
    "Chinese (Simplified)": "Chinese (Simplified)",
    "Chinese (Traditional)": "Chinese (Traditional)",
}


def milmmmt_language_name(lang: str) -> str:
    return MILMMT_LANGUAGE_NAMES.get(str(lang), str(lang))


def render_milmmmt_prompt_text(
    *,
    source_lang: str,
    target_lang: str,
    source_text: str,
    assistant_prefill: str = "",
    preserve_names_numbers_tags: bool = False,
) -> tuple[str, tuple[int, int]]:
    src_name = milmmmt_language_name(source_lang)
    tgt_name = milmmmt_language_name(target_lang)
    instruction_lines = [f"Translate this from {src_name} to {tgt_name}:"]
    if preserve_names_numbers_tags:
        instruction_lines.append(
            "Preserve names, numbers, acronyms, symbols, and tags from the source "
            "when they do not have a standard target-language rendering."
        )
    prefix = "\n".join(instruction_lines) + f"\n{src_name}: "
    source_start = len(prefix)
    suffix = f"\n{tgt_name}:"
    prompt_text = f"{prefix}{source_text}{suffix}{assistant_prefill}"
    return prompt_text, (source_start, source_start + len(source_text))


class MiLMMTVLLMMTBackend(GemmaVLLMMTBackend):
    backend_label = "milmmmt_vllm_alignatt"
    worker_cls = "cascade.mt.gemma_vllm_worker.GemmaVLLMMTWorker"

    def _source_and_target_langs(self) -> tuple[str, str]:
        return (
            str(getattr(self.runtime_config, "source_lang", "English")),
            str(getattr(self.runtime_config, "target_lang", "Chinese")),
        )

    def _prompt_mode(self) -> str:
        return str(getattr(self.runtime_config, "milmmmt_prompt_mode", "direct"))

    def resolve_generation_stop_token_ids(self) -> tuple[int, ...]:
        stop_ids = set(super().resolve_generation_stop_token_ids())
        if self.tokenizer is not None and hasattr(self.tokenizer, "convert_tokens_to_ids"):
            end_of_turn = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
            if end_of_turn is not None and int(end_of_turn) >= 0:
                stop_ids.add(int(end_of_turn))
        return tuple(sorted(stop_ids))

    def build_sampling_params_kwargs(
        self,
        *,
        max_new_tokens: int,
        stop_token_ids: list[int],
    ) -> dict[str, Any]:
        kwargs = super().build_sampling_params_kwargs(
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
        )
        kwargs.update(
            {
                "temperature": float(
                    getattr(self.runtime_config, "milmmmt_temperature", 0.0)
                ),
                "top_p": float(getattr(self.runtime_config, "milmmmt_top_p", 1.0)),
                "top_k": int(getattr(self.runtime_config, "milmmmt_top_k", 1)),
                "repetition_penalty": float(
                    getattr(self.runtime_config, "milmmmt_repetition_penalty", 1.0)
                ),
            }
        )
        return kwargs

    def render_prompt_package(
        self,
        rendered_prompt: RenderedTranslationPrompt,
    ) -> RenderedPromptWithSourceMap:
        if self.tokenizer is None:
            raise RuntimeError("MiLMMT tokenizer is not loaded. Run load() first.")

        mode = self._prompt_mode()
        if mode not in MILMMT_PROMPT_MODES:
            raise ValueError(f"Unknown milmmmt_prompt_mode: {mode!r}")

        source_lang, target_lang = self._source_and_target_langs()
        prompt_text, source_span = render_milmmmt_prompt_text(
            source_lang=source_lang,
            target_lang=target_lang,
            source_text=rendered_prompt.source_text,
            assistant_prefill=rendered_prompt.assistant_prefill,
            preserve_names_numbers_tags=(mode == "direct_preserve"),
        )
        encoded = self.tokenizer(prompt_text, add_special_tokens=False)
        prompt_token_ids = tuple(int(tid) for tid in encoded["input_ids"])
        source_map = build_prompt_source_map_from_char_span(
            tokenizer=self.tokenizer,
            source_frontier=rendered_prompt.source_frontier,
            prompt_text=prompt_text,
            source_char_start=source_span[0],
            source_char_end=source_span[1],
        )
        return RenderedPromptWithSourceMap(
            prompt_token_ids=prompt_token_ids,
            prompt_text=prompt_text,
            source_map=source_map,
        )
