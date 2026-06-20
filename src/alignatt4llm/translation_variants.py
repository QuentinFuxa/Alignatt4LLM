from __future__ import annotations

from dataclasses import dataclass

from alignatt4llm.source_frontier import SourceAccessibilityFrontier
from alignatt4llm.text_surface import (
    normalize_incremental_target_text,
    strip_repeated_accepted_prefix,
)


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
    paper_context_block: str = ""


@dataclass(frozen=True)
class TranslationVariant:
    variant_id: str
    description: str
    max_history_utterances: int
    prompt_rules: tuple[str, ...] = ()
    partial_segment_rule: str = ""
    stable_segment_rule: str = ""
    system_prompt_template: str | None = None
    # Optional suffix appended to the rendered system prompt ONLY when a
    # non-empty paper_context_block is supplied at render time. Keeping this
    # conditional means the default (no-context) path remains byte-identical
    # to pre-context runs — critical for submission stability and for honest
    # A/B comparisons against the no-context baseline.
    paper_context_instruction_template: str | None = None
    examples: tuple[PrefixContinuationExample, ...] = ()
    preserve_frozen_prefix: bool = False
    include_structured_scaffolding: bool = False
    history_as_chat_turns: bool = False

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
        paper_context_block: str = "",
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
                paper_context_block=paper_context_block,
            )

        messages: list[dict[str, str]] = []
        if self.system_prompt_template:
            system_content = self.system_prompt_template.format(
                source_lang=source_lang,
                target_lang=target_lang,
            ).strip()
            paper_instruction = (
                self.paper_context_instruction_template
                if (paper_context_block.strip() and self.paper_context_instruction_template)
                else None
            )
            if paper_instruction:
                rendered_instruction = paper_instruction.format(
                    source_lang=source_lang,
                    target_lang=target_lang,
                ).strip()
                if rendered_instruction:
                    system_content = f"{system_content}\n{rendered_instruction}"
            messages.append(
                {
                    "role": "system",
                    "content": system_content,
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
                            source_lang=source_lang,
                            is_partial=example.is_partial,
                            assistant_prefix_seeded=False,
                        )[0],
                    }
                )
                messages.append({"role": "assistant", "content": example.output})

        history_pairs = self._history_pairs(
            source_history=source_history,
            translation_history=translation_history,
        )
        if self.history_as_chat_turns:
            for source_item, translation_item in history_pairs:
                messages.append({"role": "user", "content": source_item})
                messages.append({"role": "assistant", "content": translation_item})

        current_user_message_index = len(messages)
        current_user_content, source_char_span = self._render_structured_user_message(
            source_text=text,
            context_block=(
                "(none)"
                if self.history_as_chat_turns
                else self._render_context_block(
                    source_lang=source_lang,
                    target_lang=target_lang,
                    source_history=source_history,
                    translation_history=translation_history,
                )
            ),
            source_lang=source_lang,
            is_partial=is_partial,
            assistant_prefix_seeded=bool(assistant_prefill.strip()),
            paper_context_block=paper_context_block,
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
            paper_context_block=paper_context_block,
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
            continuation = strip_repeated_accepted_prefix(
                accepted_prefix=assistant_prefill,
                generated_continuation=candidate,
            )
            return normalize_incremental_target_text(assistant_prefill + continuation)
        return normalize_incremental_target_text(candidate)

    def _render_structured_user_message(
        self,
        *,
        source_text: str,
        context_block: str,
        source_lang: str,
        is_partial: bool,
        assistant_prefix_seeded: bool,
        paper_context_block: str = "",
    ) -> tuple[str, tuple[int, int]]:
        source_header = f"[Current {source_lang} ASR prefix]\n"
        content_sections: list[str] = []
        paper_block = paper_context_block.strip()
        if paper_block:
            if source_header in paper_block:
                raise ValueError(
                    "paper_context_block must not contain the current-source header; "
                    "found collision with "
                    f"{source_header!r}"
                )
            content_sections.append(paper_block)
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
        source_lang: str,
        target_lang: str,
        source_history: list[str],
        translation_history: list[str],
    ) -> str:
        pairs = TranslationVariant._history_pairs(
            source_history=source_history,
            translation_history=translation_history,
        )
        if not pairs:
            return "(none)"
        return "\n\n".join(
            f"{source_lang}: {source}\n{target_lang}: {translation}"
            for source, translation in pairs
        )

    @staticmethod
    def _history_pairs(
        *,
        source_history: list[str],
        translation_history: list[str],
    ) -> list[tuple[str, str]]:
        return [
            (source.strip(), translation.strip())
            for source, translation in zip(source_history, translation_history)
            if source.strip() and translation.strip()
        ]


PUNCTUATION_STABLE_SEGMENT_RULE = "The current source segment is punctuation-stable."

STRUCTURED_PREFIX_SYSTEM_PROMPT = """
Translate live incremental {source_lang} ASR into fluent, idiomatic {target_lang} suitable for speech translation.
Return only {target_lang} text.
If the assistant message already contains an accepted {target_lang} prefix, continue directly after it and preserve its wording.
Prefer natural {target_lang} syntax over word-for-word calques.
If the ASR prefix contains minor punctuation noise or local disfluencies, translate the most plausible intended meaning without adding new facts.
Translate from the beginning of the current sentence.
Keep {source_lang} personal names and technical acronyms verbatim in the output; render other proper nouns in their established {target_lang} form.
Use established {target_lang} terminology for domain concepts.
Let the runtime decide which drafted tokens are committed.
""".strip()

def make_structured_prefix_variant(*, variant_id: str, description: str) -> TranslationVariant:
    return TranslationVariant(
        variant_id=variant_id,
        description=description,
        max_history_utterances=0,
        system_prompt_template=STRUCTURED_PREFIX_SYSTEM_PROMPT,
        paper_context_instruction_template=(
            "When [Paper context] is provided, use it only to resolve "
            "terminology, names, and acronyms already grounded in the current "
            "{source_lang} ASR prefix. Do not add facts from the paper context "
            "that are absent from the source prefix."
        ),
        preserve_frozen_prefix=True,
        include_structured_scaffolding=False,
        history_as_chat_turns=True,
    )


ALIGNATT_PREFIX_TRANSLATION_VARIANT = make_structured_prefix_variant(
    variant_id="alignatt_prefix",
    description=(
        "Use an accepted target-language prefix as assistant prefill and let "
        "runtime AlignAtt decide how much new target text is safe to emit."
    ),
)

TRANSLATION_VARIANTS = {
    ALIGNATT_PREFIX_TRANSLATION_VARIANT.variant_id: ALIGNATT_PREFIX_TRANSLATION_VARIANT,
}

DEFAULT_TRANSLATION_VARIANT_ID = ALIGNATT_PREFIX_TRANSLATION_VARIANT.variant_id
FOUNDATIONAL_TRANSLATION_VARIANT_ID = ALIGNATT_PREFIX_TRANSLATION_VARIANT.variant_id
