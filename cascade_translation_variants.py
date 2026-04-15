from __future__ import annotations

from dataclasses import dataclass

from cascade_source_frontier import SourceAccessibilityFrontier
from cascade_text_surface import normalize_incremental_target_text


@dataclass(frozen=True)
class PrefixContinuationExample:
    source: str
    output: str
    is_partial: bool


@dataclass(frozen=True)
class RenderedTranslationPrompt:
    messages: list[dict[str, str]]
    source_text: str
    source_frontier: SourceAccessibilityFrontier | None
    current_user_message_index: int
    source_text_char_span_in_user_message: tuple[int, int]
    continue_final_message: bool = False
    assistant_prefill: str = ""


@dataclass(frozen=True)
class TranslationVariant:
    variant_id: str
    description: str
    max_history_utterances: int
    prompt_rules: tuple[str, ...] = ()
    partial_segment_rule: str = ""
    stable_segment_rule: str = ""
    system_prompt_template: str | None = None
    examples: tuple[PrefixContinuationExample, ...] = ()
    preserve_frozen_prefix: bool = False
    include_structured_scaffolding: bool = False

    @property
    def uses_structured_messages(self) -> bool:
        return self.system_prompt_template is not None

    @property
    def uses_structured_scaffolding(self) -> bool:
        return self.uses_structured_messages and self.include_structured_scaffolding

    def render_prompt(
        self,
        *,
        source_lang: str,
        target_lang: str,
        text: str,
        source_history: list[str],
        translation_history: list[str],
        is_partial: bool,
    ) -> str:
        rules = list(self.prompt_rules)
        rules.append(self.partial_segment_rule if is_partial else self.stable_segment_rule)
        prompt_sections = [
            f"You are a professional translator from {source_lang} to {target_lang}.",
            "Rules:\n- " + "\n- ".join(rules),
        ]

        if source_history:
            prompt_sections.append(
                "Previous source context for consistency only:\n" + "\n".join(source_history)
            )
        if translation_history:
            prompt_sections.append(
                "Previous translated context for consistency only:\n"
                + "\n".join(translation_history)
            )

        segment_label = (
            "Current source segment (possibly incomplete):"
            if is_partial
            else "Current source segment:"
        )
        prompt_sections.append(f"{segment_label}\n{text}")
        return "\n\n".join(prompt_sections)

    def render_messages(
        self,
        *,
        source_lang: str,
        target_lang: str,
        text: str,
        source_frontier: SourceAccessibilityFrontier | None,
        source_history: list[str],
        translation_history: list[str],
        is_partial: bool,
        assistant_prefill: str = "",
    ) -> RenderedTranslationPrompt:
        if not self.uses_structured_messages:
            user_content = self.render_prompt(
                source_lang=source_lang,
                target_lang=target_lang,
                text=text,
                source_history=source_history,
                translation_history=translation_history,
                is_partial=is_partial,
            )
            source_start = user_content.rfind(text)
            return RenderedTranslationPrompt(
                messages=[
                    {
                        "role": "user",
                        "content": user_content,
                    }
                ],
                source_text=text,
                source_frontier=source_frontier,
                current_user_message_index=0,
                source_text_char_span_in_user_message=(
                    source_start,
                    source_start + len(text),
                ),
            )

        messages: list[dict[str, str]] = []
        if self.system_prompt_template:
            messages.append(
                {
                    "role": "system",
                    "content": self.system_prompt_template.format(
                        source_lang=source_lang,
                        target_lang=target_lang,
                    ).strip(),
                }
            )
        if self.uses_structured_scaffolding:
            for example in self.examples:
                messages.append(
                    {
                        "role": "user",
                        "content": self._render_structured_user_message(
                            source_text=example.source,
                            context_block="(none)",
                            is_partial=example.is_partial,
                            assistant_prefix_seeded=False,
                        )[0],
                    }
                )
                messages.append({"role": "assistant", "content": example.output})

        current_user_message_index = len(messages)
        current_user_content, source_char_span = self._render_structured_user_message(
            source_text=text,
            context_block=self._render_context_block(
                source_history=source_history,
                translation_history=translation_history,
            ),
            is_partial=is_partial,
            assistant_prefix_seeded=bool(assistant_prefill.strip()),
        )
        messages.append(
            {
                "role": "user",
                "content": current_user_content,
            }
        )
        continue_final_message = bool(self.preserve_frozen_prefix and assistant_prefill.strip())
        if continue_final_message:
            messages.append({"role": "assistant", "content": assistant_prefill})
        return RenderedTranslationPrompt(
            messages=messages,
            source_text=text,
            source_frontier=source_frontier,
            current_user_message_index=current_user_message_index,
            source_text_char_span_in_user_message=source_char_span,
            continue_final_message=continue_final_message,
            assistant_prefill=assistant_prefill,
        )

    def normalize_output(
        self,
        *,
        generated_text: str,
        assistant_prefill: str,
        is_partial: bool,
    ) -> str:
        del is_partial
        candidate = generated_text.rstrip(" \n")
        if assistant_prefill:
            return normalize_incremental_target_text(assistant_prefill + candidate)
        return normalize_incremental_target_text(candidate)

    def _render_structured_user_message(
        self,
        *,
        source_text: str,
        context_block: str,
        is_partial: bool,
        assistant_prefix_seeded: bool,
    ) -> tuple[str, tuple[int, int]]:
        source_header = "[Current English ASR prefix]\n"
        content_sections = []
        if context_block != "(none)":
            content_sections.append(
                "[Confirmed earlier sentence pairs]\n"
                f"{context_block}"
            )
        content_sections.append(f"{source_header}{source_text}")
        content = "\n\n".join(content_sections)
        source_start = content.rfind(source_header) + len(source_header)
        return content, (source_start, source_start + len(source_text))

    @staticmethod
    def _render_context_block(
        *,
        source_history: list[str],
        translation_history: list[str],
    ) -> str:
        pairs = [
            (source.strip(), translation.strip())
            for source, translation in zip(source_history, translation_history)
            if source.strip() and translation.strip()
        ]
        if not pairs:
            return "(none)"
        return "\n\n".join(
            f"English: {source}\nGerman: {translation}" for source, translation in pairs
        )


PUNCTUATION_STABLE_SEGMENT_RULE = "The current source segment is punctuation-stable."

STRUCTURED_PREFIX_SYSTEM_PROMPT = """
Translate live incremental {source_lang} ASR into fluent, idiomatic {target_lang} suitable for speech translation.
Return only {target_lang} text.
If the assistant message already contains an accepted {target_lang} prefix, continue directly after it and preserve its wording.
Prefer natural {target_lang} syntax over word-for-word calques.
If the ASR prefix contains minor punctuation noise or local disfluencies, translate the most plausible intended meaning without adding new facts.
Translate from the beginning of the current sentence, preserve names and technical terms when they are already clear, and let the runtime decide which drafted tokens are committed.
""".strip()

def make_structured_prefix_variant(*, variant_id: str, description: str) -> TranslationVariant:
    return TranslationVariant(
        variant_id=variant_id,
        description=description,
        max_history_utterances=0,
        system_prompt_template=STRUCTURED_PREFIX_SYSTEM_PROMPT,
        preserve_frozen_prefix=True,
        include_structured_scaffolding=False,
    )


ALIGNATT_PREFIX_TRANSLATION_VARIANT = make_structured_prefix_variant(
    variant_id="alignatt_prefix",
    description=(
        "Use an accepted German prefix as assistant prefill and let runtime AlignAtt decide how much new target text is safe to emit."
    ),
)

TRANSLATION_VARIANTS = {
    ALIGNATT_PREFIX_TRANSLATION_VARIANT.variant_id: ALIGNATT_PREFIX_TRANSLATION_VARIANT,
}

DEFAULT_TRANSLATION_VARIANT_ID = ALIGNATT_PREFIX_TRANSLATION_VARIANT.variant_id
FOUNDATIONAL_TRANSLATION_VARIANT_ID = ALIGNATT_PREFIX_TRANSLATION_VARIANT.variant_id
