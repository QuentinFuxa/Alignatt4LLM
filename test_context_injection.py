"""Tests for context_injection/ — paper artifact shape, BM25 retrieval, and
runtime prompt-span preservation when a [Paper context] block is injected.

These tests run without GPU and without any model load. They pin the
invariants we care about for the IWSLT extra-context mechanism:

1. Artifact parsing is deterministic and produces the documented schema.
2. BM25 retrieval depends only on the query + artifact and is stable.
3. Injecting a [Paper context] block into the MT prompt keeps the
   ``source_text_char_span_in_user_message`` pointing at the real source
   substring, so ``PromptSourceMap`` / AlignAtt still work.
4. With ``paper_context_mode='off'`` the renderer is bit-identical to the
   pre-context behaviour (no rendering drift, no accidental prompt bloat).
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import pytest

from cascade_translation_variants import (
    ALIGNATT_PREFIX_TRANSLATION_VARIANT,
    RenderedTranslationPrompt,
)
from context_injection import (
    CONTEXT_MODE_OFF,
    CONTEXT_MODE_RETRIEVED_CHUNKS,
    CONTEXT_MODE_TITLE_ABSTRACT,
    CONTEXT_MODE_TITLE_AND_CHUNKS,
    PAPER_CONTEXT_HEADER,
    BM25Index,
    PaperArtifact,
    PaperChunk,
    PaperContextSelector,
    build_retrieval_query,
    parse_markdown_body,
)


SAMPLE_MARKDOWN = """\
Attention as a Guide for Simultaneous Speech Translation

Sara Papi, University of Trento
Matteo Negri
Marco Turchi

Abstract

We present AlignAtt, a simultaneous speech translation policy that uses the
attention mass of a translation decoder to decide when a prefix of the
hypothesis is safe to emit. AlignAtt makes no assumption about the source
encoder and generalises to multiple language directions.

1 Introduction

Simultaneous speech translation, or SimulST, is the problem of translating
streaming spoken input into fluent target-language text with low latency. A
good SimulST policy balances wait time against quality.

In this work we build AlignAtt on top of the transformer decoder. The policy
operates on attention weights collected at the most recent decoder step.

2 Method

AlignAtt computes an attention-weighted source-position score for each
hypothesis token and accepts the longest prefix whose attention falls
strictly inside the already-seen source prefix. This converts the commit
problem into a prefix-matching problem over decoder attention.

The margin parameter controls how far from the audio frontier a token must
be before it is safely committed. A larger margin trades latency for
stability.

References

[1] Ma et al.
"""


def test_parse_markdown_body_schema():
    artifact = parse_markdown_body(SAMPLE_MARKDOWN, paper_id="alignatt")
    assert artifact.paper_id == "alignatt"
    assert "Attention as a Guide" in artifact.title
    assert "Sara Papi" in artifact.authors
    assert "AlignAtt" in artifact.abstract
    assert artifact.chunks, "expected at least one body chunk"
    # References section must be dropped so retrieval is not polluted by
    # bibliography tokens.
    for chunk in artifact.chunks:
        assert "[1] Ma et al" not in chunk.text
    # Schema round-trips through JSON.
    data = artifact.to_dict()
    restored = PaperArtifact.from_dict(data)
    assert restored == artifact


def test_parse_markdown_body_is_deterministic():
    a = parse_markdown_body(SAMPLE_MARKDOWN, paper_id="alignatt")
    b = parse_markdown_body(SAMPLE_MARKDOWN, paper_id="alignatt")
    assert a == b


def test_parse_markdown_body_strips_pymupdf_image_markers():
    """Regression for the "==> Bild [430 x 186] <==" leak observed in retrieved
    chunks of OiqEWDVtWk (DECISIONS.md). The parser must strip pymupdf4llm
    figure placeholders with a generic regex so they can never reach the
    runtime MT prompt and leak into the translation.
    """

    noisy_markdown = (
        "A Good Paper\n\nAuthor Name\n\nAbstract\n\nShort abstract here.\n\n"
        "1 Introduction\n\nThis is real content. ==> Bild [430 x 186] <== "
        "More real content ==> [Figure 2] [123 x 45] <== and a tail. "
        "![alt](image.png) trailing text.\n"
    )
    artifact = parse_markdown_body(noisy_markdown, paper_id="t")
    combined = " ".join(c.text for c in artifact.chunks)
    assert "Bild" not in combined
    assert "==>" not in combined
    assert "image.png" not in combined
    assert "real content" in combined


def _make_sample_artifact() -> PaperArtifact:
    return PaperArtifact(
        paper_id="t",
        title="Attention as a Guide for Simultaneous Speech Translation",
        authors="Sara Papi, Matteo Negri, Marco Turchi",
        abstract="We propose AlignAtt, a simultaneous speech translation policy.",
        chunks=(
            PaperChunk(
                chunk_id="c0000",
                text=(
                    "AlignAtt uses the attention mass of the decoder to decide "
                    "when a hypothesis prefix is safe to emit, and generalises "
                    "to multiple language directions."
                ),
                section="Method",
            ),
            PaperChunk(
                chunk_id="c0001",
                text=(
                    "Latency budgets in SimulST are traditionally measured by "
                    "average lagging metrics such as AL and LAAL."
                ),
                section="Background",
            ),
            PaperChunk(
                chunk_id="c0002",
                text=(
                    "We evaluate AlignAtt on MuST-C English to German and "
                    "observe competitive BLEU at substantially lower latency."
                ),
                section="Experiments",
            ),
        ),
    )


def _make_budget_artifact() -> PaperArtifact:
    return PaperArtifact(
        paper_id="budget",
        title=(
            "A Very Long Paper Title That Should Be Truncated Cleanly When "
            "The Budget Is Tight"
        ),
        authors="A. Author",
        abstract=(
            "This abstract is intentionally verbose so the renderer has to "
            "truncate it on a word boundary while keeping the ellipsis inside "
            "the configured character budget."
        ),
        chunks=(
            PaperChunk(
                chunk_id="c0000",
                text=(
                    "The retrieved chunk is also intentionally long so the "
                    "renderer has to truncate the first selected chunk while "
                    "keeping the ellipsis counted inside the budget."
                ),
                section="Method",
            ),
            PaperChunk(
                chunk_id="c0001",
                text="A short backup chunk that should usually be dropped.",
                section="Results",
            ),
        ),
    )


def test_bm25_retrieval_is_deterministic_and_relevant():
    artifact = _make_sample_artifact()
    selector = PaperContextSelector.from_artifact(artifact)
    block_a = selector.select(
        mode=CONTEXT_MODE_RETRIEVED_CHUNKS,
        query="attention decoder prefix safe to emit",
        top_k=2,
        max_chars=2000,
    )
    block_b = selector.select(
        mode=CONTEXT_MODE_RETRIEVED_CHUNKS,
        query="attention decoder prefix safe to emit",
        top_k=2,
        max_chars=2000,
    )
    # Determinism.
    assert block_a.text == block_b.text
    # The method chunk mentions "attention", "decoder", "emit": top-1 should
    # be c0000.
    assert block_a.used_chunk_ids[0] == "c0000"
    assert PAPER_CONTEXT_HEADER in block_a.text


def test_bm25_score_orders_by_query_overlap():
    artifact = _make_sample_artifact()
    index = BM25Index.build(artifact.chunks)
    scored = index.score("MuST-C English German BLEU latency", top_k=3)
    assert scored, "expected at least one retrieved chunk"
    # Evaluation chunk should outrank Background and Method here.
    assert scored[0].chunk_id == "c0002"


def test_context_mode_off_returns_empty_block():
    artifact = _make_sample_artifact()
    selector = PaperContextSelector.from_artifact(artifact)
    block = selector.select(mode=CONTEXT_MODE_OFF, query="anything", top_k=3, max_chars=1000)
    assert block.is_empty()
    assert block.text == ""


def test_title_abstract_mode_truncates_title_when_budget_is_tiny():
    artifact = _make_budget_artifact()
    selector = PaperContextSelector.from_artifact(artifact)
    block = selector.select(
        mode=CONTEXT_MODE_TITLE_ABSTRACT,
        query="",
        top_k=0,
        max_chars=28,
    )
    body = block.text.split("\n", 1)[1]
    assert len(body) <= 28
    assert "Title:" in body
    assert body.endswith("...")
    assert "Abstract:" not in body


def test_title_abstract_mode_truncates_abstract_within_budget():
    artifact = _make_budget_artifact()
    selector = PaperContextSelector.from_artifact(artifact)
    block = selector.select(
        mode=CONTEXT_MODE_TITLE_ABSTRACT,
        query="",
        top_k=0,
        max_chars=96,
    )
    body = block.text.split("\n", 1)[1]
    assert len(body) <= 96
    assert body.count("\n") == 1
    assert body.split("\n", 1)[1].endswith("...")


def test_title_and_chunks_mode_counts_separators_in_budget():
    artifact = PaperArtifact(
        paper_id="combo-budget",
        title="Short title",
        authors="A. Author",
        abstract="A medium abstract that still leaves enough room for one chunk.",
        chunks=(
            PaperChunk(
                chunk_id="c0000",
                text=(
                    "This retrieved chunk is intentionally long so the combined "
                    "mode has to budget title, abstract, separator, and chunk "
                    "text together."
                ),
                section="Method",
            ),
        ),
    )
    selector = PaperContextSelector.from_artifact(artifact)
    block = selector.select(
        mode=CONTEXT_MODE_TITLE_AND_CHUNKS,
        query="combined budget chunk",
        top_k=2,
        max_chars=180,
    )
    body = block.text.split("\n", 1)[1]
    assert len(body) <= 180
    assert body.count("\n\n") == 1


def test_retrieved_chunks_mode_truncates_first_chunk_with_ellipsis_in_budget():
    artifact = PaperArtifact(
        paper_id="chunk-budget",
        title="Short title",
        authors="A. Author",
        abstract="",
        chunks=(
            PaperChunk(
                chunk_id="c0000",
                text=(
                    "This retrieved chunk is intentionally long so the first "
                    "selected chunk must be truncated with an ellipsis that "
                    "still fits inside the budget."
                ),
                section="Method",
            ),
        ),
    )
    selector = PaperContextSelector.from_artifact(artifact)
    block = selector.select(
        mode=CONTEXT_MODE_RETRIEVED_CHUNKS,
        query="truncated chunk budget",
        top_k=1,
        max_chars=96,
    )
    body = block.text.split("\n", 1)[1]
    assert len(body) <= 96
    assert "[c0000 | Method]" in body
    assert body.endswith("...")


def test_render_messages_preserves_source_span_with_paper_context():
    """Paper context lives *before* the source header, so the recorded
    `source_text_char_span_in_user_message` must still point at the exact
    current-source text inside the rendered user message.
    """

    variant = ALIGNATT_PREFIX_TRANSLATION_VARIANT
    text = "Simultaneous speech translation, or SimulST, is"
    paper_block = (
        f"{PAPER_CONTEXT_HEADER}\nTitle: Attention as a Guide\n"
        "Abstract: A decoder-attention-based policy for SimulST."
    )
    prompt = variant.render_messages(
        source_lang="English",
        target_lang="German",
        text=text,
        source_frontier=None,
        source_history=[],
        translation_history=[],
        is_partial=True,
        assistant_prefill="",
        paper_context_block=paper_block,
    )
    user_message = prompt.messages[prompt.current_user_message_index]["content"]
    s, e = prompt.source_text_char_span_in_user_message
    assert user_message[s:e] == text
    assert "[Paper context]" in user_message
    assert user_message.index("[Paper context]") < user_message.index(
        "[Current English ASR prefix]"
    )


def test_render_messages_default_matches_no_context_behaviour():
    variant = ALIGNATT_PREFIX_TRANSLATION_VARIANT
    text = "Simultaneous speech translation, or SimulST, is"
    without_kw = variant.render_messages(
        source_lang="English",
        target_lang="German",
        text=text,
        source_frontier=None,
        source_history=[],
        translation_history=[],
        is_partial=False,
        assistant_prefill="",
    )
    with_empty = variant.render_messages(
        source_lang="English",
        target_lang="German",
        text=text,
        source_frontier=None,
        source_history=[],
        translation_history=[],
        is_partial=False,
        assistant_prefill="",
        paper_context_block="",
    )
    assert without_kw.messages == with_empty.messages
    assert (
        without_kw.source_text_char_span_in_user_message
        == with_empty.source_text_char_span_in_user_message
    )


def test_render_messages_rejects_collision_with_source_header():
    variant = ALIGNATT_PREFIX_TRANSLATION_VARIANT
    with pytest.raises(ValueError):
        variant.render_messages(
            source_lang="English",
            target_lang="German",
            text="hello world",
            source_frontier=None,
            source_history=[],
            translation_history=[],
            is_partial=False,
            assistant_prefill="",
            # Smuggling the source header into the paper block is forbidden
            # because `rfind` would mis-locate the real source.
            paper_context_block="[Current English ASR prefix]\nfake",
        )


def test_retrieval_query_builder_respects_history_window():
    history = ["we present AlignAtt", "for simultaneous speech translation"]
    query = build_retrieval_query(
        current_source_prefix="AlignAtt uses decoder attention",
        history_words=[w for item in history for w in item.split()],
        history_window_words=3,
    )
    # Only last 3 history words + current prefix.
    assert query.startswith("simultaneous speech translation")
    assert query.endswith("AlignAtt uses decoder attention")


def test_paper_artifact_roundtrip_via_file(tmp_path: Path):
    artifact = _make_sample_artifact()
    out = tmp_path / "p.json"
    artifact.write_json(out)
    restored = PaperArtifact.read_json(out)
    assert restored == artifact
    # Schema version is embedded so future migrations can check it.
    assert json.loads(out.read_text())["schema_version"].startswith("paper_artifact/")


def test_runtime_config_validates_context_mode_requires_path():
    from cascade_runtime import CascadeRuntimeConfig

    with pytest.raises(ValueError, match="paper_context_path"):
        CascadeRuntimeConfig(paper_context_mode=CONTEXT_MODE_TITLE_ABSTRACT)


def test_runtime_config_accepts_off_without_path():
    from cascade_runtime import CascadeRuntimeConfig

    cfg = CascadeRuntimeConfig()
    assert cfg.paper_context_mode == CONTEXT_MODE_OFF
    assert cfg.paper_context_path is None


def test_shipped_variant_does_not_carry_paper_instruction():
    """Mitigation 1 (DECISIONS.md) was attempted and rolled back because the
    reference-only clause caused Gemma-4-E4B to mode-collapse under
    retrieved_chunks. The shipped variant must *not* emit any paper-context
    instruction; the slot is retained only for future, better experiments.
    """

    variant = ALIGNATT_PREFIX_TRANSLATION_VARIANT
    assert variant.paper_context_instruction_template is None

    prompt_default = variant.render_messages(
        source_lang="English", target_lang="German",
        text="hello", source_frontier=None,
        source_history=[], translation_history=[],
        is_partial=False, assistant_prefill="",
    )
    prompt_with_paper = variant.render_messages(
        source_lang="English", target_lang="German",
        text="hello", source_frontier=None,
        source_history=[], translation_history=[],
        is_partial=False, assistant_prefill="",
        paper_context_block=f"{PAPER_CONTEXT_HEADER}\nTitle: Foo",
    )
    # Same system message whether or not a paper block is injected.
    assert prompt_default.messages[0] == prompt_with_paper.messages[0]


def test_run_context_ablation_plumbs_history_knob_and_summary():
    from types import SimpleNamespace

    import run_context_ablation as rca

    args = SimpleNamespace(
        wav="tmp/clip.wav",
        paper_context_path="data/paper_artifacts/clip.json",
        output_dir="outputs/test",
        chunk_ms=450,
        source="en",
        target="de",
        max_history_utterances=1,
        alignment_backend_name="qwen_forced",
        mt_backend_name="gemma_transformers_alignatt",
        paper_context_top_k=3,
        paper_context_max_chars=1200,
        paper_context_history_window_words=60,
        min_start_seconds=2.0,
        translation_alignatt_min_source_mass=0.0,
    )
    processor_config = rca.build_processor_config(args)
    assert processor_config.max_history_utterances == 1
    summary = rca.build_summary(args=args, load_ms=12.34, per_mode_summaries=[])
    assert summary["max_history_utterances"] == 1


def test_paper_context_instruction_slot_plumbing_when_explicitly_set():
    """Sanity check that the slot works when a variant *does* fill it, even
    though the shipped variant does not. This protects the plumbing for
    future mitigation experiments (target-language block, provenance guard)
    where we may want to reintroduce a system-prompt clause.
    """

    from dataclasses import replace

    variant = replace(
        ALIGNATT_PREFIX_TRANSLATION_VARIANT,
        paper_context_instruction_template="Extra rule for {target_lang} only.",
    )
    prompt_with_paper = variant.render_messages(
        source_lang="English", target_lang="German",
        text="hello", source_frontier=None,
        source_history=[], translation_history=[],
        is_partial=False, assistant_prefill="",
        paper_context_block=f"{PAPER_CONTEXT_HEADER}\nTitle: Foo",
    )
    prompt_default = variant.render_messages(
        source_lang="English", target_lang="German",
        text="hello", source_frontier=None,
        source_history=[], translation_history=[],
        is_partial=False, assistant_prefill="",
    )
    # Instruction fires only when a paper block is present, and substitutes
    # the configured target language.
    assert "Extra rule for German only." in prompt_with_paper.messages[0]["content"]
    assert "Extra rule for" not in prompt_default.messages[0]["content"]
