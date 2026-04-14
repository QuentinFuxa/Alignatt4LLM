from __future__ import annotations

from dataclasses import dataclass


UNCERTAINTY_MARKER = "<unused0>"


@dataclass(frozen=True)
class PrefixBoundaryExample:
    source: str
    frozen_prefix: str
    output: str


@dataclass(frozen=True)
class RenderedTranslationPrompt:
    messages: list[dict[str, str]]
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
    examples: tuple[PrefixBoundaryExample, ...] = ()
    uncertainty_marker: str | None = None
    preserve_frozen_prefix: bool = False

    @property
    def uses_structured_messages(self) -> bool:
        return self.system_prompt_template is not None

    @property
    def uses_uncertainty_boundary(self) -> bool:
        return self.uncertainty_marker is not None

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
        source_history: list[str],
        translation_history: list[str],
        is_partial: bool,
        assistant_prefill: str = "",
    ) -> RenderedTranslationPrompt:
        if not self.uses_structured_messages:
            return RenderedTranslationPrompt(
                messages=[
                    {
                        "role": "user",
                        "content": self.render_prompt(
                            source_lang=source_lang,
                            target_lang=target_lang,
                            text=text,
                            source_history=source_history,
                            translation_history=translation_history,
                            is_partial=is_partial,
                        ),
                    }
                ]
            )

        messages: list[dict[str, str]] = []
        if self.system_prompt_template and not self.preserve_frozen_prefix:
            messages.append(
                {
                    "role": "system",
                    "content": self.system_prompt_template.format(
                        source_lang=source_lang,
                        target_lang=target_lang,
                        uncertainty_marker=self.uncertainty_marker or "",
                    ).strip(),
                }
            )
        if not self.preserve_frozen_prefix:
            for example in self.examples:
                example_is_partial = bool(
                    self.uncertainty_marker and self.uncertainty_marker in example.output
                )
                messages.append(
                    {
                        "role": "user",
                        "content": self._render_structured_user_message(
                            source_text=example.source,
                            context_block="(none)",
                            is_partial=example_is_partial,
                            assistant_prefix_seeded=False,
                        ),
                    }
                )
                messages.append({"role": "assistant", "content": example.output})

        messages.append(
            {
                "role": "user",
                "content": self._render_structured_user_message(
                    source_text=text,
                    context_block=self._render_context_block(
                        source_history=source_history,
                        translation_history=translation_history,
                    ),
                    is_partial=is_partial,
                    assistant_prefix_seeded=bool(assistant_prefill.strip()),
                ),
            }
        )
        continue_final_message = bool(self.preserve_frozen_prefix and assistant_prefill.strip())
        if continue_final_message:
            messages.append({"role": "assistant", "content": assistant_prefill})
        return RenderedTranslationPrompt(
            messages=messages,
            continue_final_message=continue_final_message,
            assistant_prefill=assistant_prefill,
        )

    def normalize_output(
        self,
        *,
        generated_text: str,
        assistant_prefill: str,
        is_partial: bool,
    ) -> tuple[str, bool]:
        candidate = generated_text
        boundary_seen = False
        if self.uncertainty_marker and self.uncertainty_marker in candidate:
            candidate = candidate.split(self.uncertainty_marker, 1)[0].rstrip()
            boundary_seen = True
        candidate = candidate.rstrip(" \n")
        if assistant_prefill:
            return (assistant_prefill + candidate).rstrip(" \n"), boundary_seen
        return candidate.strip(), boundary_seen

    def _render_structured_user_message(
        self,
        *,
        source_text: str,
        context_block: str,
        is_partial: bool,
        assistant_prefix_seeded: bool,
    ) -> str:
        task_lines = [
            "Output natural German, not English word order.",
            "Return only one German continuation, with no explanations or alternatives.",
        ]
        if assistant_prefix_seeded:
            task_lines.append(
                "The assistant message is already prefilled with the full German output emitted at the previous update from the beginning of the current sentence, so continue directly after it without rewriting or restarting it."
            )
        else:
            task_lines.append(
                "Start the German sentence from the beginning of the current sentence in the assistant continuation."
            )
        if is_partial and self.uncertainty_marker:
            task_lines.append(
                f"If the next German position is still unstable, write {self.uncertainty_marker} exactly there and stop."
            )
        else:
            task_lines.append(
                "The sentence is complete enough to finish naturally while preserving the elected prefix."
            )

        rendered_task = "\n".join(f"- {line}" for line in task_lines)
        return (
            "[Instruction]\n"
            f"{rendered_task}\n\n"
            "[Confirmed earlier sentence pairs]\n"
            f"{context_block}\n\n"
            "[Current English ASR prefix]\n"
            f"{source_text}"
        )

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
Produce the longest {target_lang} prefix from the beginning of the current sentence that is already stable.
Use natural {target_lang} word order and preserve names, titles, and technical terms when they are already clear.
If a frozen {target_lang} prefix is provided, keep it exactly as the beginning of your answer.
If the next {target_lang} position is still unstable, write {uncertainty_marker} exactly there and stop.
Use earlier confirmed sentence pairs only for continuity and terminology.
"""

PREFIX_BOUNDARY_EXAMPLES = (
    PrefixBoundaryExample(
        source="because I have seen",
        frozen_prefix="",
        output=f"weil ich {UNCERTAINTY_MARKER}",
    ),
    PrefixBoundaryExample(
        source="because I have seen him",
        frozen_prefix="weil ich",
        output="weil ich ihn gesehen habe",
    ),
    PrefixBoundaryExample(
        source="I'm here to introduce",
        frozen_prefix="",
        output=f"Ich bin hier, um {UNCERTAINTY_MARKER}",
    ),
    PrefixBoundaryExample(
        source="I'm here to introduce our paper.",
        frozen_prefix="Ich bin hier, um",
        output="Ich bin hier, um unser Paper vorzustellen.",
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

PROMPT_ONLY_PARTIAL_ANCHOR_TRANSLATION_VARIANT = TranslationVariant(
    variant_id="prompt_only_partial_anchor",
    description=(
        "Carry forward an elected German prefix and mark the first unstable position with <unused0> instead of guessing ahead."
    ),
    max_history_utterances=0,
    system_prompt_template=STRUCTURED_PREFIX_SYSTEM_PROMPT,
    examples=PREFIX_BOUNDARY_EXAMPLES,
    uncertainty_marker=UNCERTAINTY_MARKER,
    preserve_frozen_prefix=True,
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
    PROMPT_ONLY_PARTIAL_ANCHOR_TRANSLATION_VARIANT.variant_id: PROMPT_ONLY_PARTIAL_ANCHOR_TRANSLATION_VARIANT,
    CONTEXT1_TERMINOLOGY_GUARD_TRANSLATION_VARIANT.variant_id: CONTEXT1_TERMINOLOGY_GUARD_TRANSLATION_VARIANT,
}

DEFAULT_TRANSLATION_VARIANT_ID = BASELINE_TRANSLATION_VARIANT.variant_id
