from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TranslationVariant:
    variant_id: str
    description: str
    max_history_utterances: int
    prompt_rules: tuple[str, ...]
    partial_segment_rule: str
    stable_segment_rule: str

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
    CONTEXT1_TERMINOLOGY_GUARD_TRANSLATION_VARIANT.variant_id: CONTEXT1_TERMINOLOGY_GUARD_TRANSLATION_VARIANT,
}

DEFAULT_TRANSLATION_VARIANT_ID = BASELINE_TRANSLATION_VARIANT.variant_id
