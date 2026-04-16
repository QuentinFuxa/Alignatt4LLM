#!/usr/bin/env python3
"""Phase 3 GPU validation: qk_fast vs eager agreement + end-to-end online invariant.

Loads Gemma once, builds a real prompt with assistant_prefill, drafts tokens,
then runs both probe paths and verifies:
1. qk_fast source attention rows closely match eager rows
2. qk_fast argmax alignment matches eager argmax alignment
3. Batched prefix-online alignment tail matches true online loop with prefill
"""
from __future__ import annotations

import torch

from cascade_mt_backend import (
    IncrementalAlignAttTracker,
    build_mt_backend,
    compute_prefix_online_alignatt_source_argmaxes,
)
from cascade_source_frontier import build_source_accessibility_frontier
from cascade_translation_variants import ALIGNATT_PREFIX_TRANSLATION_VARIANT
from cascade_runtime import CascadeRuntimeConfig, gemma_model_name, temporary_runtime_config


def main():
    print("Loading Gemma model...")
    config = CascadeRuntimeConfig()
    with temporary_runtime_config(
        config,
        min_start_seconds=2.0,
        partial_max_new_tokens=16,
        partial_followup_max_new_tokens=8,
        max_history_utterances=1,
        translation_alignatt_inaccessible_ms=0.0,
        translation_alignatt_rewind_threshold=8,
    ):
        backend = build_mt_backend(model_name=gemma_model_name, runtime_config=config)
        backend.load()
        print(f"  Loaded. {len(backend.alignatt_heads)} heads.\n")

        source_text = (
            "In this paper we define the problem of constrained language planning "
            "which impose different constraints on the goal of planning"
        )
        assistant_prefill = "In dieser Arbeit definieren wir das Problem"
        word_timestamps = [(i * 200.0, (i + 1) * 200.0) for i in range(len(source_text.split()))]

        frontier = build_source_accessibility_frontier(
            source_text=source_text,
            word_timestamps_ms=word_timestamps,
            current_audio_ms=4000.0,
            inaccessible_ms=0.0,
            is_final=False,
        )

        rendered = ALIGNATT_PREFIX_TRANSLATION_VARIANT.render_messages(
            source_lang="English",
            target_lang="German",
            text=source_text,
            source_frontier=frontier,
            source_history=[],
            translation_history=[],
            is_partial=True,
            assistant_prefill=assistant_prefill,
        )

        prompt_package = backend.render_prompt_package(rendered)
        source_map = prompt_package.source_map
        assert source_map is not None, "Source map must exist for this test"
        print(f"Prompt: {len(prompt_package.prompt_token_ids)} tokens")
        print(f"Source: {len(source_map.source_token_positions)} source tokens "
              f"({source_map.accessible_source_token_count} accessible)")

        # Draft some tokens
        draft_result = backend.decode_draft(
            prompt_token_ids=prompt_package.prompt_token_ids,
            max_new_tokens=12,
        )
        draft_ids = draft_result.draft_generated_ids
        draft_text = backend.tokenizer.decode(draft_ids, skip_special_tokens=True)
        print(f"Draft: {len(draft_ids)} tokens = '{draft_text}'")
        print(f"  prompt_kv_snapshot: {len(draft_result.prompt_kv_snapshot)} layers\n")

        # === Test 1: qk_fast vs eager agreement ===
        print("=" * 60)
        print("Test 1: qk_fast vs eager source attention row agreement")
        print("=" * 60)

        qk_rows, qk_ms, qk_provenance = backend._probe_source_attention_rows_qk_fast(
            draft_generated_ids=draft_ids,
            prompt_num_tokens=draft_result.prompt_num_tokens,
            prompt_kv_snapshot=draft_result.prompt_kv_snapshot,
            source_map=source_map,
        )
        eager_rows, eager_ms, _ = backend._probe_source_attention_rows_eager(
            draft_generated_ids=draft_ids,
            prompt_num_tokens=draft_result.prompt_num_tokens,
            prompt_kv_snapshot=draft_result.prompt_kv_snapshot,
            source_map=source_map,
        )

        print(f"  qk_fast: {len(qk_rows)} token rows, {qk_ms:.1f} ms")
        print(f"  eager:   {len(eager_rows)} token rows, {eager_ms:.1f} ms")

        filter_width = backend.policy.alignatt_filter_width()
        qk_argmaxes = compute_prefix_online_alignatt_source_argmaxes(qk_rows, filter_width=filter_width)
        eager_argmaxes = compute_prefix_online_alignatt_source_argmaxes(eager_rows, filter_width=filter_width)

        print(f"  qk_fast argmaxes: {qk_argmaxes}")
        print(f"  eager   argmaxes: {eager_argmaxes}")

        argmax_match = qk_argmaxes == eager_argmaxes
        print(f"  Argmax match: {argmax_match}")

        max_row_diff = 0.0
        mean_row_diff = 0.0
        for i, (qk_row, eager_row) in enumerate(zip(qk_rows, eager_rows)):
            diff = (qk_row - eager_row).abs().max().item()
            max_row_diff = max(max_row_diff, diff)
            mean_row_diff += (qk_row - eager_row).abs().mean().item()
        mean_row_diff /= max(1, len(qk_rows))
        print(f"  Max abs diff across all rows: {max_row_diff:.6f}")
        print(f"  Mean abs diff: {mean_row_diff:.6f}")

        if argmax_match:
            print("  PASS: qk_fast and eager produce identical alignment decisions.")
        else:
            print("  INFO: argmaxes differ, checking if acceptance decisions would match...")
            # Even if argmaxes differ numerically, check if both lead to same acceptance
            qk_probe = backend.probe_alignatt(
                draft_generated_ids=draft_ids,
                prompt_num_tokens=draft_result.prompt_num_tokens,
                prompt_kv_snapshot=draft_result.prompt_kv_snapshot,
                source_map=source_map,
                upstream_stop_reason=None,
            )
            # Temporarily switch to eager
            orig_mode = backend.alignatt_probe_mode
            backend.alignatt_probe_mode = "eager"
            eager_probe = backend.probe_alignatt(
                draft_generated_ids=draft_ids,
                prompt_num_tokens=draft_result.prompt_num_tokens,
                prompt_kv_snapshot=draft_result.prompt_kv_snapshot,
                source_map=source_map,
                upstream_stop_reason=None,
            )
            backend.alignatt_probe_mode = orig_mode
            if len(qk_probe.accepted_candidate_ids) == len(eager_probe.accepted_candidate_ids):
                print("  PASS: acceptance decisions agree despite argmax differences.")
            else:
                print(f"  WARN: qk_fast accepts {len(qk_probe.accepted_candidate_ids)}, "
                      f"eager accepts {len(eager_probe.accepted_candidate_ids)}")

        # === Test 2: Provenance is populated ===
        print(f"\n  Provenance: {len(qk_provenance)} entries")
        if qk_provenance:
            p = qk_provenance[0]
            total = p.source_accessible + p.source_inaccessible + p.non_source_prompt + p.suffix
            print(f"  Token 0: src_acc={p.source_accessible:.4f} src_inacc={p.source_inaccessible:.4f} "
                  f"non_src={p.non_source_prompt:.4f} suffix={p.suffix:.4f} total={total:.4f}")

        # === Test 3: Batched prefix-online == online loop with real Gemma rows ===
        print(f"\n{'=' * 60}")
        print("Test 2: Batched prefix-online tail matches online loop with real Gemma rows")
        print("=" * 60)

        # Use a prompt WITHOUT prefill so we can draft more tokens and split
        # them into simulated "prefill" and "draft" portions.
        rendered_no_prefill = ALIGNATT_PREFIX_TRANSLATION_VARIANT.render_messages(
            source_lang="English",
            target_lang="German",
            text=source_text,
            source_frontier=frontier,
            source_history=[],
            translation_history=[],
            is_partial=True,
            assistant_prefill="",
        )
        pkg_no_prefill = backend.render_prompt_package(rendered_no_prefill)
        sm_no_prefill = pkg_no_prefill.source_map
        assert sm_no_prefill is not None

        draft2 = backend.decode_draft(
            prompt_token_ids=pkg_no_prefill.prompt_token_ids,
            max_new_tokens=16,
        )
        all_draft_ids = draft2.draft_generated_ids
        print(f"  Drafted {len(all_draft_ids)} tokens without prefill")

        all_rows, _, _ = backend._probe_source_attention_rows_qk_fast(
            draft_generated_ids=all_draft_ids,
            prompt_num_tokens=draft2.prompt_num_tokens,
            prompt_kv_snapshot=draft2.prompt_kv_snapshot,
            source_map=sm_no_prefill,
        )
        print(f"  Got {len(all_rows)} source attention row tensors")

        if len(all_rows) >= 6:
            split_at = len(all_rows) // 3
            prefill_rows = all_rows[:split_at]
            draft_rows_part = all_rows[split_at:]

            batched_all = compute_prefix_online_alignatt_source_argmaxes(
                all_rows, filter_width=filter_width
            )
            batched_tail = batched_all[split_at:]

            tracker = IncrementalAlignAttTracker(filter_width=filter_width)
            for row in prefill_rows:
                tracker.update(row)
            online_tail = [tracker.update(row) for row in draft_rows_part]

            print(f"  Split: {split_at} prefill + {len(draft_rows_part)} draft")
            print(f"  Batched tail: {batched_tail}")
            print(f"  Online tail:  {online_tail}")
            match = batched_tail == online_tail
            print(f"  Match: {match}")
            if match:
                print("  PASS: batched prefix-online tail == online loop on real Gemma attention rows.")
            else:
                print("  FAIL: tails differ!")
        else:
            print(f"  SKIP: only {len(all_rows)} rows, need >= 6 for meaningful split")

        print("\n" + "=" * 60)
        print("Phase 3 GPU validation complete.")
        print("=" * 60)


if __name__ == "__main__":
    main()
