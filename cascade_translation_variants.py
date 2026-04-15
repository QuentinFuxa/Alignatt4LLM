from __future__ import annotations

from dataclasses import dataclass

from cascade_source_frontier import SourceAccessibilityFrontier


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
        if self.system_prompt_template and self.uses_structured_scaffolding:
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
            return (assistant_prefill + candidate).rstrip(" \n")
        return candidate.strip()

    def _render_structured_user_message(
        self,
        *,
        source_text: str,
        context_block: str,
        is_partial: bool,
        assistant_prefix_seeded: bool,
    ) -> tuple[str, tuple[int, int]]:
        task_lines = [
            "Output natural German, not English word order.",
            "Return only one German continuation, with no explanations or alternatives.",
        ]
        if assistant_prefix_seeded:
            task_lines.append(
                "The assistant message already contains the accepted German prefix from the beginning of the current sentence. Continue directly after it and never restart or rewrite that prefix."
            )
        else:
            task_lines.append(
                "Start the German sentence from the beginning of the current sentence in the assistant continuation."
            )
        if is_partial:
            task_lines.append(
                "If the English prefix is still unfolding, stop at the last complete German word that is already safe to emit now."
            )
        else:
            task_lines.append(
                "The sentence is complete enough to finish naturally while preserving the accepted prefix."
            )

        rendered_task = "\n".join(f"- {line}" for line in task_lines)
        source_header = "[Current English ASR prefix]\n"
        content = (
            "[Instruction]\n"
            f"{rendered_task}\n\n"
            "[Confirmed earlier sentence pairs]\n"
            f"{context_block}\n\n"
            f"{source_header}"
            f"{source_text}"
        )
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


STREAMING_ASR_NOISE_RULE = "The input comes from streaming ASR and may contain recognition noise."
RETURN_CURRENT_SEGMENT_RULE = "Return only the translation of the current source segment."
BASELINE_CONTEXT_RULE = "Use any provided context only to keep terminology consistent."
CONTEXT_DISCOURSE_RULE = (
    "Use any provided context only to keep terminology and discourse consistent."
)
TERMINOLOGY_GUARD_RULES = (
    "Keep quoted paper titles, model names, dataset names, and product names in the source language unless a standard German form is obvious.",
    "Preserve technical constraints and discourse markers faithfully; do not paraphrase away details.",
)
BASELINE_PARTIAL_SEGMENT_RULE = (
    "If the segment is incomplete, translate only the portion that is already clear."
)
GUARDED_PARTIAL_SEGMENT_RULE = (
    "If the segment is incomplete, translate only the portion that is already clear and keep the wording easy to continue later."
)
PUNCTUATION_STABLE_SEGMENT_RULE = "The current source segment is punctuation-stable."

STRUCTURED_PREFIX_SYSTEM_PROMPT = """
You translate live incremental {source_lang} ASR into {target_lang}.
Produce the longest {target_lang} prefix from the beginning of the current sentence that is already safe to emit now.
Use natural {target_lang} word order and preserve names, titles, and technical terms when they are already clear.
If an accepted {target_lang} prefix is already present in the assistant message, keep it exactly and continue directly after it.
Use earlier confirmed sentence pairs only for continuity and terminology.
""".strip()

PREFIX_CONTINUATION_EXAMPLES = (
    PrefixContinuationExample(
        source="because I have seen",
        output="weil ich",
        is_partial=True,
    ),
    PrefixContinuationExample(
        source="because I have seen him",
        output="weil ich ihn gesehen habe",
        is_partial=False,
    ),
    PrefixContinuationExample(
        source="I'm here to introduce",
        output="Ich bin hier, um",
        is_partial=True,
    ),
    PrefixContinuationExample(
        source="I'm here to introduce our paper.",
        output="Ich bin hier, um unser Paper vorzustellen.",
        is_partial=False,
    ),
)


BASELINE_TRANSLATION_VARIANT = TranslationVariant(
    variant_id="baseline",
    description="Translate each segment independently with no previous-utterance context.",
    max_history_utterances=0,
    prompt_rules=(
        STREAMING_ASR_NOISE_RULE,
        BASELINE_CONTEXT_RULE,
        RETURN_CURRENT_SEGMENT_RULE,
    ),
    partial_segment_rule=BASELINE_PARTIAL_SEGMENT_RULE,
    stable_segment_rule=PUNCTUATION_STABLE_SEGMENT_RULE,
)

PROMPT_ONLY_TERMINOLOGY_GUARD_TRANSLATION_VARIANT = TranslationVariant(
    variant_id="prompt_only_terminology_guard",
    description=(
        "Apply terminology and technical-constraint guardrails without previous-utterance context."
    ),
    max_history_utterances=0,
    prompt_rules=(
        STREAMING_ASR_NOISE_RULE,
        BASELINE_CONTEXT_RULE,
        *TERMINOLOGY_GUARD_RULES,
        RETURN_CURRENT_SEGMENT_RULE,
    ),
    partial_segment_rule=GUARDED_PARTIAL_SEGMENT_RULE,
    stable_segment_rule=PUNCTUATION_STABLE_SEGMENT_RULE,
)


def make_structured_prefix_variant(*, variant_id: str, description: str) -> TranslationVariant:
    return TranslationVariant(
        variant_id=variant_id,
        description=description,
        max_history_utterances=0,
        system_prompt_template=STRUCTURED_PREFIX_SYSTEM_PROMPT,
        examples=PREFIX_CONTINUATION_EXAMPLES,
        preserve_frozen_prefix=True,
        include_structured_scaffolding=True,
    )


ALIGNATT_PREFIX_TRANSLATION_VARIANT = make_structured_prefix_variant(
    variant_id="alignatt_prefix",
    description=(
        "Use an accepted German prefix as assistant prefill and let runtime AlignAtt decide how much new target text is safe to emit."
    ),
)

PROMPT_ONLY_PARTIAL_ANCHOR_TRANSLATION_VARIANT = make_structured_prefix_variant(
    variant_id="prompt_only_partial_anchor",
    description=(
        "Legacy alias of the accepted-prefix variant kept for comparability with earlier experiments."
    ),
)

CONTEXT1_TERMINOLOGY_GUARD_TRANSLATION_VARIANT = TranslationVariant(
    variant_id="context1_terminology_guard",
    description=(
        "Reuse one previous committed utterance and guard quoted titles plus technical terms."
    ),
    max_history_utterances=1,
    prompt_rules=(
        STREAMING_ASR_NOISE_RULE,
        CONTEXT_DISCOURSE_RULE,
        *TERMINOLOGY_GUARD_RULES,
        RETURN_CURRENT_SEGMENT_RULE,
        "Use the context as reference only and never repeat earlier sentences.",
    ),
    partial_segment_rule=GUARDED_PARTIAL_SEGMENT_RULE,
    stable_segment_rule=PUNCTUATION_STABLE_SEGMENT_RULE,
)

TRANSLATION_VARIANTS = {
    BASELINE_TRANSLATION_VARIANT.variant_id: BASELINE_TRANSLATION_VARIANT,
    PROMPT_ONLY_TERMINOLOGY_GUARD_TRANSLATION_VARIANT.variant_id: PROMPT_ONLY_TERMINOLOGY_GUARD_TRANSLATION_VARIANT,
    ALIGNATT_PREFIX_TRANSLATION_VARIANT.variant_id: ALIGNATT_PREFIX_TRANSLATION_VARIANT,
    PROMPT_ONLY_PARTIAL_ANCHOR_TRANSLATION_VARIANT.variant_id: PROMPT_ONLY_PARTIAL_ANCHOR_TRANSLATION_VARIANT,
    CONTEXT1_TERMINOLOGY_GUARD_TRANSLATION_VARIANT.variant_id: CONTEXT1_TERMINOLOGY_GUARD_TRANSLATION_VARIANT,
}

DEFAULT_TRANSLATION_VARIANT_ID = BASELINE_TRANSLATION_VARIANT.variant_id
FOUNDATIONAL_TRANSLATION_VARIANT_ID = ALIGNATT_PREFIX_TRANSLATION_VARIANT.variant_id
