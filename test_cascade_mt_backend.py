import torch
import torch.nn as nn

from cascade_emission import RAW_PASSTHROUGH, apply_emission_policy
from cascade_mt_backend import (
    AlignAttHead,
    AlignAttDecoderPolicy,
    IncrementalAlignAttTracker,
    LayerInputCapture,
    MTBackendResult,
    SelectedLayerInputRecorder,
    compute_alignatt_source_argmaxes,
    compute_prefix_online_alignatt_source_argmaxes,
    extract_source_attention_rows_per_token_from_fast_path,
    extract_source_qk_rows_per_token,
    extract_source_attention_rows_per_token,
    load_alignatt_heads,
    map_attention_head_to_key_value_head,
    multilingual_union_alignatt_heads,
    resolve_prompt_and_suffix_key_states_for_layer,
    shared_kernel_alignatt_heads,
    source_local_position_to_unit_index,
    write_alignatt_heads_file,
    PromptSourceMap,
    PromptSourceUnitSpan,
)
from cascade_source_text import normalize_source_text_for_mt
from cascade_text_surface import normalize_incremental_target_text
from cascade_translation_variants import ALIGNATT_PREFIX_TRANSLATION_VARIANT
from qwen3asr_gemma_cascade_core import (
    PartialTranslationState,
    derive_monotone_partial_acceptance,
    should_run_partial_mt_update,
)


class IdentityNorm(nn.Module):
    def forward(self, values):
        return values


class FakeAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = 2
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.q_norm = IdentityNorm()
        self.k_norm = IdentityNorm()
        self.scaling = 1.0
        self.sliding_window = None
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(4))
            self.k_proj.weight.copy_(torch.eye(4))

    def forward(self, hidden_states=None, position_embeddings=None, **kwargs):
        del kwargs
        return hidden_states, None


class FakeDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = FakeAttentionModule()


class FakeLayerContainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([FakeDecoderLayer() for _ in range(4)])


class FakeHookableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeLayerContainer()


def test_incremental_alignatt_tracker_matches_prefix_recomputation():
    torch.manual_seed(0)

    tracker = IncrementalAlignAttTracker(filter_width=7)
    source_attention_rows_per_token: list[torch.Tensor] = []
    expected_prefix_positions: list[int | None] = []

    for _ in range(8):
        source_attention_rows = torch.rand(4, 9)
        source_attention_rows_per_token.append(source_attention_rows)
        expected_prefix_positions.append(
            compute_alignatt_source_argmaxes(
                source_attention_rows_per_token,
                filter_width=7,
            )[-1]
        )
        assert tracker.update(source_attention_rows) == expected_prefix_positions[-1]

    assert tracker.aligned_source_local_positions == expected_prefix_positions


def test_prefix_online_alignatt_source_argmaxes_matches_tracker_updates():
    torch.manual_seed(1)
    rows_per_token = [torch.rand(3, 7) for _ in range(6)]

    tracker = IncrementalAlignAttTracker(filter_width=5)
    expected = [tracker.update(source_attention_rows) for source_attention_rows in rows_per_token]

    assert compute_prefix_online_alignatt_source_argmaxes(rows_per_token, filter_width=5) == expected


def test_batched_prefix_online_tail_matches_online_loop_with_assistant_prefill():
    """Pin the invariant we rely on when an assistant_prefill is present.

    In a real streaming decode, the prompt already contains an accepted target
    prefix (``assistant_prefill``). The batched prefix-online probe rebuilds
    attention rows for every prefill token AND every drafted token in one
    forward, but only the drafted tail is actionable. This test asserts that
    the tail alignments match those produced by a true online loop that walks
    token-by-token over the same sequence with an ``IncrementalAlignAttTracker``
    that has been warm-started on the prefill rows.

    Concretely: ``batched[prefill_len:] == [tracker.update(row) for row in draft_rows]``
    after ``tracker`` has already consumed the prefill rows. Any drift here
    would mean the probe disagrees with what a strict online decoder would
    observe at those same positions, which would invalidate AlignAtt's
    acceptance reasoning.
    """

    torch.manual_seed(17)
    prefill_rows = [torch.rand(3, 9) for _ in range(4)]
    draft_rows = [torch.rand(3, 9) for _ in range(5)]
    all_rows = prefill_rows + draft_rows
    filter_width = 7

    batched_tail = compute_prefix_online_alignatt_source_argmaxes(
        all_rows, filter_width=filter_width
    )[len(prefill_rows):]

    online_tracker = IncrementalAlignAttTracker(filter_width=filter_width)
    for row in prefill_rows:
        online_tracker.update(row)
    online_tail = [online_tracker.update(row) for row in draft_rows]

    assert batched_tail == online_tail


def test_empty_source_rows_yield_no_alignment_without_breaking_prefill_flow():
    """A zero-width source window must not crash the probe under prefill.

    Edge case seen near ``translation_alignatt_inaccessible_ms`` transitions:
    the accessible source slice can be momentarily empty. In that regime the
    aligned source position is undefined (``None``) and the tracker must treat
    every incoming token as "no alignment evidence yet" rather than raise.
    """

    tracker = IncrementalAlignAttTracker(filter_width=3)
    rows = [torch.zeros(2, 0) for _ in range(3)]
    assert [tracker.update(row) for row in rows] == [None, None, None]


def test_prefix_online_alignatt_source_argmaxes_can_differ_from_full_suffix_global():
    rows_per_token = [
        torch.tensor(
            [[0.49625659, 0.7682218, 0.08847743], [0.13203049, 0.30742282, 0.63407868]],
            dtype=torch.float32,
        ),
        torch.tensor(
            [[0.49009341, 0.89644474, 0.45562798], [0.63230628, 0.34889346, 0.40171731]],
            dtype=torch.float32,
        ),
        torch.tensor(
            [[0.02232575, 0.16885895, 0.29388845], [0.51852179, 0.6976676, 0.8000114]],
            dtype=torch.float32,
        ),
        torch.tensor(
            [[0.16102946, 0.28226858, 0.68160856], [0.91519397, 0.39709991, 0.87415588]],
            dtype=torch.float32,
        ),
    ]

    prefix_online = compute_prefix_online_alignatt_source_argmaxes(rows_per_token, filter_width=1)
    full_suffix_global = compute_alignatt_source_argmaxes(rows_per_token, filter_width=1)

    assert prefix_online == [0, 1, 2, 2]
    assert full_suffix_global == [1, 0, 1, 2]


def test_extract_source_attention_rows_per_token_keeps_each_query_position():
    layer_attentions = {
        3: torch.tensor(
            [
                [
                    [
                        [0.0, 0.1, 0.2, 0.3, 0.4],
                        [0.5, 0.6, 0.7, 0.8, 0.9],
                    ]
                ]
            ],
            dtype=torch.float32,
        )
    }

    rows_per_token = extract_source_attention_rows_per_token(
        layer_attentions_by_layer=layer_attentions,
        alignatt_heads=[AlignAttHead(layer=3, head=0, ts=1.0)],
        source_positions=[1, 3],
    )

    assert len(rows_per_token) == 2
    assert torch.equal(rows_per_token[0], torch.tensor([[0.1, 0.3]]))
    assert torch.equal(rows_per_token[1], torch.tensor([[0.6, 0.8]]))


def test_map_attention_head_to_key_value_head_handles_grouped_query_attention():
    assert map_attention_head_to_key_value_head(0, num_attention_heads=4, num_key_value_heads=2) == 0
    assert map_attention_head_to_key_value_head(1, num_attention_heads=4, num_key_value_heads=2) == 0
    assert map_attention_head_to_key_value_head(2, num_attention_heads=4, num_key_value_heads=2) == 1
    assert map_attention_head_to_key_value_head(3, num_attention_heads=4, num_key_value_heads=2) == 1


def test_selected_layer_input_recorder_captures_kwargs_only_attention_calls():
    model = FakeHookableModel()
    recorder = SelectedLayerInputRecorder(
        model=model,
        alignatt_heads=[AlignAttHead(layer=2, head=0, ts=1.0)],
    )
    hidden_states = torch.randn(1, 3, 4)
    cos = torch.ones(1, 3, 2)
    sin = torch.zeros(1, 3, 2)

    with recorder.capture() as captured:
        model.model.layers[2].self_attn(
            hidden_states=hidden_states,
            position_embeddings=(cos, sin),
        )

    assert sorted(captured.keys()) == [2]
    capture = captured[2]
    assert torch.equal(capture.hidden_states, hidden_states)
    assert capture.position_embeddings is not None
    assert torch.equal(capture.position_embeddings[0], cos)
    assert torch.equal(capture.position_embeddings[1], sin)


def test_extract_source_qk_rows_per_token_reconstructs_prompt_source_scores():
    module = FakeAttentionModule()
    hidden_states = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ]
        ],
        dtype=torch.float32,
    )
    cos = torch.ones(1, 2, 2, dtype=torch.float32)
    sin = torch.zeros(1, 2, 2, dtype=torch.float32)
    layer_inputs = {
        3: LayerInputCapture(
            module=module,
            hidden_states=hidden_states,
            position_embeddings=(cos, sin),
        )
    }
    prompt_kv_snapshot = [
        (
            3,
            torch.tensor(
                [
                    [
                        [[10.0, 0.0], [1.0, 1.0], [0.0, 10.0], [2.0, 0.0]],
                        [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 2.0]],
                    ]
                ],
                dtype=torch.float32,
            ),
            torch.zeros(1, 2, 4, 2, dtype=torch.float32),
            4,
        )
    ]

    rows_per_token = extract_source_qk_rows_per_token(
        layer_inputs_by_layer=layer_inputs,
        prompt_kv_snapshot=prompt_kv_snapshot,
        alignatt_heads=[
            AlignAttHead(layer=3, head=0, ts=1.0),
            AlignAttHead(layer=3, head=1, ts=1.0),
        ],
        source_positions=[1, 3],
    )

    assert len(rows_per_token) == 2
    assert torch.equal(rows_per_token[0], torch.tensor([[3.0, 2.0], [3.0, 8.0]]))
    assert torch.equal(rows_per_token[1], torch.tensor([[11.0, 10.0], [7.0, 16.0]]))


def test_extract_source_attention_rows_per_token_from_fast_path_matches_causal_softmax():
    module = FakeAttentionModule()
    hidden_states = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    cos = torch.ones(1, 2, 2, dtype=torch.float32)
    sin = torch.zeros(1, 2, 2, dtype=torch.float32)
    layer_inputs = {
        3: LayerInputCapture(
            module=module,
            hidden_states=hidden_states,
            position_embeddings=(cos, sin),
        )
    }
    prompt_kv_snapshot = [
        (
            3,
            torch.tensor(
                [
                    [
                        [[1.0, 0.0], [0.0, 1.0]],
                        [[0.0, 0.0], [0.0, 0.0]],
                    ]
                ],
                dtype=torch.float32,
            ),
            torch.zeros(1, 2, 2, 2, dtype=torch.float32),
            2,
        )
    ]

    rows_per_token = extract_source_attention_rows_per_token_from_fast_path(
        layer_inputs_by_layer=layer_inputs,
        prompt_kv_snapshot=prompt_kv_snapshot,
        alignatt_heads=[AlignAttHead(layer=3, head=0, ts=1.0)],
        source_positions=[0, 1],
    )

    expected = torch.tensor(
        [
            [torch.exp(torch.tensor(1.0)) / (2 * torch.exp(torch.tensor(1.0)) + 1.0), 1.0 / (2 * torch.exp(torch.tensor(1.0)) + 1.0)],
            [1.0 / (2.0 + 2.0 * torch.exp(torch.tensor(1.0))), torch.exp(torch.tensor(1.0)) / (2.0 + 2.0 * torch.exp(torch.tensor(1.0)))],
        ],
        dtype=torch.float32,
    )

    assert len(rows_per_token) == 2
    assert torch.allclose(rows_per_token[0], expected[0].unsqueeze(0), atol=1e-6)
    assert torch.allclose(rows_per_token[1], expected[1].unsqueeze(0), atol=1e-6)


def test_resolve_prompt_and_suffix_key_states_for_layer_prefers_runtime_cache_for_full_attention():
    module = FakeAttentionModule()
    capture = LayerInputCapture(
        module=module,
        hidden_states=torch.zeros(1, 2, 4),
        position_embeddings=None,
    )
    prompt_key_cache = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]]]],
        dtype=torch.float32,
    )
    runtime_key_cache = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]]]],
        dtype=torch.float32,
    )

    resolved_prompt_keys, resolved_suffix_keys = resolve_prompt_and_suffix_key_states_for_layer(
        layer_idx=3,
        capture=capture,
        prompt_key_cache_by_layer={3: prompt_key_cache},
        runtime_key_cache_by_layer={3: runtime_key_cache},
        runtime_shared_key_cache_by_layer={},
    )

    assert torch.equal(resolved_prompt_keys, prompt_key_cache)
    assert torch.equal(resolved_suffix_keys, runtime_key_cache[:, :, 2:, :])


def test_resolve_prompt_and_suffix_key_states_for_layer_uses_shared_runtime_cache_for_shared_kv_layers():
    module = FakeAttentionModule()
    module.is_kv_shared_layer = True
    module.kv_shared_layer_index = 3
    capture = LayerInputCapture(
        module=module,
        hidden_states=torch.zeros(1, 2, 4),
        position_embeddings=None,
    )
    prompt_key_cache = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]]]],
        dtype=torch.float32,
    )
    shared_runtime_key_cache = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0], [3.0, 0.0], [0.0, 3.0]]]],
        dtype=torch.float32,
    )

    resolved_prompt_keys, resolved_suffix_keys = resolve_prompt_and_suffix_key_states_for_layer(
        layer_idx=5,
        capture=capture,
        prompt_key_cache_by_layer={3: prompt_key_cache},
        runtime_key_cache_by_layer={},
        runtime_shared_key_cache_by_layer={3: shared_runtime_key_cache},
    )

    assert torch.equal(resolved_prompt_keys, prompt_key_cache)
    assert torch.equal(resolved_suffix_keys, shared_runtime_key_cache[:, :, 2:, :])


def test_source_local_position_to_unit_index_maps_prompt_token_back_to_source_unit():
    source_map = PromptSourceMap(
        source_text="alpha beta",
        source_token_positions=(10, 11, 12, 13),
        source_unit_spans=(
            PromptSourceUnitSpan(
                unit_index=0,
                text="alpha",
                prompt_token_positions=(10, 11),
                is_accessible=True,
                start_ms=0.0,
                end_ms=100.0,
            ),
            PromptSourceUnitSpan(
                unit_index=1,
                text="beta",
                prompt_token_positions=(12, 13),
                is_accessible=False,
                start_ms=100.0,
                end_ms=200.0,
            ),
        ),
        accessible_source_token_count=2,
        accessible_unit_count=1,
        total_unit_count=2,
        current_audio_ms=180.0,
        inaccessible_ms=50.0,
        is_final=False,
    )

    assert source_local_position_to_unit_index(source_map, 0) == 0
    assert source_local_position_to_unit_index(source_map, 2) == 1
    assert source_local_position_to_unit_index(source_map, 7) is None


def test_partial_prompt_contract_keeps_runtime_as_acceptance_authority():
    rendered = ALIGNATT_PREFIX_TRANSLATION_VARIANT.render_messages(
        source_lang="English",
        target_lang="German",
        text="because I have seen",
        source_frontier=None,
        source_history=[],
        translation_history=[],
        is_partial=True,
        assistant_prefill="weil ich",
    )

    system_prompt = rendered.messages[0]["content"]
    current_user_message = rendered.messages[1]["content"]

    assert len(rendered.messages) == 3
    assert "let the runtime decide which drafted tokens are committed." in system_prompt
    assert "safe to emit now" not in system_prompt
    assert "[Current English ASR prefix]\nbecause I have seen" == current_user_message
    assert rendered.messages[-1] == {"role": "assistant", "content": "weil ich"}


def test_monotone_acceptance_requires_full_semantic_prefix_ids_with_prefill():
    previous_state = PartialTranslationState(
        source_prefix="because I have",
        accepted_target="weil ich",
        accepted_token_ids=(10, 11),
    )

    full_prefix_result = MTBackendResult(
        draft_text="weil ich ihn",
        acceptance_text="weil ich ihn",
        accepted_token_ids=(10, 11, 12),
    )
    accepted_text, accepted_ids = derive_monotone_partial_acceptance(
        previous_state=previous_state,
        source_prefix="because I have seen him",
        result=full_prefix_result,
    )

    assert accepted_text == "weil ich ihn"
    assert accepted_ids == (10, 11, 12)

    suffix_only_result = MTBackendResult(
        draft_text="weil ich ihn",
        acceptance_text="weil ich ihn",
        accepted_token_ids=(12,),
    )
    fallback_text, fallback_ids = derive_monotone_partial_acceptance(
        previous_state=previous_state,
        source_prefix="because I have seen him",
        result=suffix_only_result,
    )

    assert fallback_text == "weil ich"
    assert fallback_ids == (10, 11)


def test_partial_mt_scheduler_skips_until_frontier_or_stall_requires_probe():
    previous_state = PartialTranslationState(
        source_prefix="because I have",
        source_accessible_unit_count=3,
        last_mt_audio_seconds=5.0,
    )

    should_run, reason = should_run_partial_mt_update(
        previous_state=previous_state,
        source_prefix="because I have seen",
        accessible_unit_count=3,
        current_audio_seconds_value=5.6,
        stall_seconds=1.2,
    )
    assert not should_run
    assert reason == "frontier_not_advanced"

    should_run, reason = should_run_partial_mt_update(
        previous_state=previous_state,
        source_prefix="because I have seen",
        accessible_unit_count=3,
        current_audio_seconds_value=6.3,
        stall_seconds=1.2,
    )
    assert should_run
    assert reason == "stall_probe"

    should_run, reason = should_run_partial_mt_update(
        previous_state=previous_state,
        source_prefix="because I have seen him",
        accessible_unit_count=4,
        current_audio_seconds_value=5.1,
        stall_seconds=1.2,
    )
    assert should_run
    assert reason == "accessible_frontier_advanced"


def test_partial_mt_scheduler_waits_for_previously_blocked_source_unit():
    previous_state = PartialTranslationState(
        source_prefix="because I have",
        source_accessible_unit_count=3,
        last_mt_audio_seconds=5.0,
        blocked_source_unit_index=4,
    )

    should_run, reason = should_run_partial_mt_update(
        previous_state=previous_state,
        source_prefix="because I have seen",
        accessible_unit_count=4,
        current_audio_seconds_value=5.6,
        stall_seconds=1.2,
    )
    assert not should_run
    assert reason == "blocked_frontier_not_reached"

    should_run, reason = should_run_partial_mt_update(
        previous_state=previous_state,
        source_prefix="because I have seen",
        accessible_unit_count=4,
        current_audio_seconds_value=6.3,
        stall_seconds=1.2,
    )
    assert should_run
    assert reason == "stall_probe"

    should_run, reason = should_run_partial_mt_update(
        previous_state=previous_state,
        source_prefix="because I have seen him",
        accessible_unit_count=5,
        current_audio_seconds_value=5.6,
        stall_seconds=1.2,
    )
    assert should_run
    assert reason == "accessible_frontier_advanced"


def test_emit_policy_normalizes_streaming_surface_artifacts():
    emitted, action = apply_emission_policy(
        RAW_PASSTHROUGH,
        "",
        (
            "In diesem Papier definieren wir das Problem der eingeschränkten Sprachplanung. "
            "die unterschiedliche Einschränkungen auf das Planungsziel auferlegen."
        ),
        max_tail_rewrite_words=14,
        is_final=False,
    )

    assert action == RAW_PASSTHROUGH
    assert emitted.endswith("Planungsziel auferlegen.")


def test_source_normalizer_normalizes_spacing_without_hardcoded_phrases():
    normalized = normalize_source_text_for_mt(
        "Hello  ,   world !  This is a test ."
    )

    assert normalized.text == "Hello, world! This is a test."


def test_alignatt_rewind_keeps_safe_prefix_up_to_offending_token():
    """AlignAtt must keep the prefix already aligned to the accessible source
    and only reject the offending tail.

    The Whisper-style behaviour is: when the current aligned source position
    jumps backward past a configured threshold relative to the prefix's last
    stable alignment, that one suffix token is unsafe. Everything strictly
    before it is provenance-consistent with the accessible source and stays
    accepted. Rejecting the whole suffix would make `accepted_target` shrink
    on every attention glitch and defeat the monotone-acceptance contract.
    """

    from types import SimpleNamespace

    runtime_config = SimpleNamespace(translation_alignatt_rewind_threshold=2)
    policy = AlignAttDecoderPolicy(tokenizer=None, runtime_config=runtime_config)

    draft_token_ids = [101, 102, 103, 104, 105]
    # Tokens 0..2 climb monotonically through the accessible source; token 3
    # jumps back far enough to trigger rewind; token 4 would be downstream.
    aligned_source_local_positions = [0, 3, 6, 1, 7]
    accessible_source_token_count = 10

    accepted_candidate_ids: list[int] = []
    unsafe_reason: str | None = None
    unsafe_target_token_index: int | None = None
    last_aligned: int | None = None

    for token_index, (token_id, aligned) in enumerate(
        zip(draft_token_ids, aligned_source_local_positions)
    ):
        (
            unsafe_reason,
            _,
            _,
            _,
        ) = policy.should_stop_in_loop(
            current_source_local_position=aligned,
            last_aligned_source_local_position=last_aligned,
            accessible_source_token_count=accessible_source_token_count,
        )
        if unsafe_reason in {"rewind", "source_frontier"}:
            unsafe_target_token_index = token_index
            break
        accepted_candidate_ids.append(token_id)
        if aligned is not None:
            last_aligned = aligned

    assert unsafe_reason == "rewind"
    assert unsafe_target_token_index == 3
    # Safe prefix is kept: tokens before the offending one survive the rewind.
    assert accepted_candidate_ids == [101, 102, 103]


def test_alignatt_tolerates_local_reorder_within_rewind_threshold():
    """Local reorderings near the frontier must not trigger a rewind.

    Motivation (Phase 3): the monotone-acceptance contract is compatible with
    short-range reordering — e.g. German verb clusters moving a few source
    positions backward during decoding — as long as the backward jump stays
    within ``translation_alignatt_rewind_threshold``. If every tiny dip reset
    the accepted prefix, AlignAtt would shrink ``accepted_target`` whenever
    the draft briefly re-attends to an earlier source unit, which is the exact
    failure mode we explicitly do NOT want.
    """

    from types import SimpleNamespace

    runtime_config = SimpleNamespace(translation_alignatt_rewind_threshold=3)
    policy = AlignAttDecoderPolicy(tokenizer=None, runtime_config=runtime_config)

    aligned_source_local_positions = [0, 3, 5, 4, 6]  # 5->4 is a 1-step dip
    accessible_source_token_count = 10

    accepted: list[int] = []
    last_aligned: int | None = None
    for token_index, aligned in enumerate(aligned_source_local_positions):
        unsafe_reason, _, _, _ = policy.should_stop_in_loop(
            current_source_local_position=aligned,
            last_aligned_source_local_position=last_aligned,
            accessible_source_token_count=accessible_source_token_count,
        )
        assert unsafe_reason is None, f"unexpected stop at token {token_index}: {unsafe_reason}"
        accepted.append(token_index)
        last_aligned = aligned

    assert accepted == [0, 1, 2, 3, 4]


def test_write_and_load_alignatt_heads_file_round_trips(tmp_path):
    heads = [
        AlignAttHead(layer=11, head=3, ts=0.852),
        AlignAttHead(layer=6, head=5, ts=0.783),
        AlignAttHead(layer=17, head=3, ts=0.747),
    ]
    path = tmp_path / "shared_kernel_heads.json"

    write_alignatt_heads_file(
        heads,
        path,
        direction="shared_kernel",
        extra_metadata={"regime": "shared_kernel", "source_directions": ["en-de", "en-it", "en-zh"]},
    )

    loaded = load_alignatt_heads(str(path), top_k=len(heads))
    assert loaded == heads

    truncated = load_alignatt_heads(str(path), top_k=2)
    assert truncated == heads[:2]


class _FakeTokenizer:
    """Minimal tokenizer double used to exercise stability-unit trimming."""

    def __init__(self, token_strings):
        self._tokens = list(token_strings)

    def convert_ids_to_tokens(self, ids):
        return [self._tokens[int(token_id)] for token_id in ids]


def test_token_starts_stability_unit_recognises_space_and_cjk_boundaries():
    assert AlignAttDecoderPolicy.token_starts_stability_unit("▁weil")
    assert AlignAttDecoderPolicy.token_starts_stability_unit("Ġhello")
    assert AlignAttDecoderPolicy.token_starts_stability_unit("<0x0A>")
    # Each Han character is its own stability unit even without leading space.
    assert AlignAttDecoderPolicy.token_starts_stability_unit("因")
    assert AlignAttDecoderPolicy.token_starts_stability_unit("▁世")
    # Japanese kana share the no-whitespace convention.
    assert AlignAttDecoderPolicy.token_starts_stability_unit("あ")
    # Pure intra-word continuations (BPE fragments of Latin-script words) do not.
    assert not AlignAttDecoderPolicy.token_starts_stability_unit("ation")
    assert not AlignAttDecoderPolicy.token_starts_stability_unit("")


def _make_policy(tokens):
    from types import SimpleNamespace

    return AlignAttDecoderPolicy(
        tokenizer=_FakeTokenizer(tokens),
        runtime_config=SimpleNamespace(),
    )


def test_trim_to_last_stability_unit_drops_trailing_incomplete_word_for_space_language():
    tokens = ["▁weil", "▁ich", "▁ihn", "▁ge", "sehen"]
    policy = _make_policy(tokens)
    trimmed = policy.trim_to_last_stability_unit(list(range(len(tokens))))
    # The last stability unit is the compound "▁ge sehen"; trimming drops it.
    assert trimmed == [0, 1, 2]


def test_trim_to_last_stability_unit_keeps_prefix_characters_for_chinese_script():
    # Mirrors the failure mode described in PLAN.md: 因为我看见了他 would
    # previously collapse to an empty list because no token starts with ▁.
    tokens = ["因", "为", "我", "看", "见", "了", "他"]
    policy = _make_policy(tokens)
    trimmed = policy.trim_to_last_stability_unit(list(range(len(tokens))))
    assert trimmed == [0, 1, 2, 3, 4, 5]


def test_structured_prompt_context_block_is_language_agnostic():
    rendered = ALIGNATT_PREFIX_TRANSLATION_VARIANT.render_messages(
        source_lang="English",
        target_lang="Italian",
        text="because I have seen",
        source_frontier=None,
        source_history=["hello world"],
        translation_history=["ciao mondo"],
        is_partial=True,
        assistant_prefill="perché l'ho",
    )

    current_user_message = rendered.messages[1]["content"]
    assert "English: hello world" in current_user_message
    assert "Italian: ciao mondo" in current_user_message
    assert "German:" not in current_user_message


def test_split_target_emission_units_splits_by_whitespace_for_latin_targets():
    from cascade_text_surface import split_target_emission_units

    assert split_target_emission_units(
        "weil ich ihn gesehen habe", target_lang_code="de"
    ) == ["weil", "ich", "ihn", "gesehen", "habe"]
    assert split_target_emission_units(
        "perché l'ho visto", target_lang_code="it"
    ) == ["perché", "l'ho", "visto"]


def test_split_target_emission_units_is_char_level_for_chinese():
    from cascade_text_surface import split_target_emission_units

    # Matches OmniSTEval's char_level=True resegmentation for zh. No empty
    # strings from spacing artefacts; each Han character is one unit.
    assert split_target_emission_units(
        "因为 我看见了他", target_lang_code="zh"
    ) == ["因", "为", "我", "看", "见", "了", "他"]


def test_split_target_emission_units_nfkc_normalises_before_char_split_for_zh():
    """Char-level unit count must survive OmniSTEval's NFKC normalization.

    The evaluator runs ``unicodedata.normalize("NFKC", prediction)`` before
    splitting into characters. If our in-flight unit count disagrees with the
    post-NFKC count, ``load_hypothesis_jsonl`` fails the
    ``len(units) == len(cu_values)`` assertion and the whole zh evaluation
    run is unusable. Locking the invariant here keeps the char-level
    inference/evaluation contract bit-identical on either side.
    """
    import unicodedata

    from cascade_artifacts import InferenceArtifacts
    from cascade_text_surface import split_target_emission_units

    # Halfwidth/fullwidth digits and kana are the canonical NFKC test: NFKC
    # decomposes "①" into multiple codepoints and folds fullwidth forms to
    # halfwidth. Plain ``list(text)`` would undercount the eventual NFKC
    # characters by these deltas.
    text = "① ②\u3000ＡＢ因为"  # includes fullwidth digits, fullwidth space, fullwidth latin
    units = split_target_emission_units(text, target_lang_code="zh")
    rebuilt = "".join(units)
    assert rebuilt == unicodedata.normalize("NFKC", rebuilt)
    assert len(units) == len(rebuilt)
    assert (
        len(rebuilt)
        == len(unicodedata.normalize("NFKC", text).replace(" ", "").replace("\u3000", ""))
    )

    # End-to-end: the hypothesis record serialised for char-level targets
    # must have ``prediction`` bytes that equal the NFKC-normalised join of
    # units, so that evaluator and runtime count units identically.
    artifacts = InferenceArtifacts(
        wav_path="x.wav",
        chunk_ms=450,
        translation_variant="alignatt_prefix",
        source_language="English",
        target_language="Chinese",
        latency_unit="word",
        audio_duration_ms=1000.0,
        final_asr_text="x",
        final_translation_text=text,
        translation_word_delays_ms=[100.0] * len(units),
        translation_word_elapsed_ms=[100.0] * len(units),
        updates=[],
        runtime_config={},
        target_language_code="zh",
    )
    record = artifacts.hypothesis_record()
    assert len(record["prediction"]) == len(record["delays"])
    assert record["prediction"] == unicodedata.normalize("NFKC", record["prediction"])


def test_freeze_major_tail_rewrites_guards_chinese_at_character_granularity():
    """The tail-rewrite window must count Han characters for zh targets.

    Without a target-language-aware splitter this test would collapse both
    translations into a single "word" each, the rewrite window check would
    become vacuous, and the runtime would emit a 10-char-wide rewrite as if it
    were a no-op surface edit. That would contradict the monotone acceptance
    contract for en->zh serving.
    """
    from cascade_emission import FREEZE_MAJOR_TAIL_REWRITES, apply_emission_policy

    previous = "因为我看见了他走进房间坐下喝茶"  # 14 characters
    raw = "于是我听见了他走进房间坐下喝茶"  # differs in the first 2 characters

    emitted, action = apply_emission_policy(
        FREEZE_MAJOR_TAIL_REWRITES,
        previous,
        raw,
        max_tail_rewrite_words=3,
        is_final=False,
        target_lang_code="zh",
    )

    # Rewrite reaches far beyond the 3-character tail window, so it must be
    # frozen to the previously emitted Chinese surface, not silently accepted.
    assert action == "frozen_major_tail_rewrite"
    assert emitted == previous


def test_freeze_major_tail_rewrites_allows_local_chinese_tail_edit():
    from cascade_emission import FREEZE_MAJOR_TAIL_REWRITES, apply_emission_policy

    previous = "因为我看见了他走进"
    raw = "因为我看见了他走来"  # differs only in the last character

    emitted, action = apply_emission_policy(
        FREEZE_MAJOR_TAIL_REWRITES,
        previous,
        raw,
        max_tail_rewrite_words=3,
        is_final=False,
        target_lang_code="zh",
    )

    assert action == "raw_passthrough"
    assert emitted == raw


def test_register_translation_words_aligns_delays_with_characters_for_chinese():
    from cascade_emission import register_translation_words

    delays: list[float] = []
    # First emission adds one character: "因".
    new_units = register_translation_words(
        "", "因", 100.0, delays, target_lang_code="zh"
    )
    assert new_units == ["因"]
    assert delays == [100.0]

    # Second emission extends to "因为我"; only the new characters get stamps.
    new_units = register_translation_words(
        "因", "因为我", 250.0, delays, target_lang_code="zh"
    )
    assert new_units == ["为", "我"]
    assert delays == [100.0, 250.0, 250.0]


def test_alignatt_heads_path_resolves_to_shipped_files_for_every_supported_direction():
    """Every configured target language must have a real heads file on disk.

    Dropping a new ``LANGUAGE_NAME_TO_CODE`` entry without shipping its
    ``translation_heads_*.json`` would silently fall back to en-de (because
    the MT backend calls ``load_alignatt_heads`` with that missing path at
    runtime) and poison every downstream multilingual comparison. Locking
    this invariant here means that failure surfaces at test time, not during
    a GPU sweep.
    """
    import json
    from pathlib import Path

    from qwen3asr_gemma_cascade_core import LANGUAGE_NAME_TO_CODE, alignatt_heads_path_for

    source_lang = "English"
    for target_label in ("German", "Italian", "Chinese"):
        assert target_label in LANGUAGE_NAME_TO_CODE, target_label
        heads_path = Path(alignatt_heads_path_for(source_lang, target_label))
        assert heads_path.exists(), heads_path
        payload = json.loads(heads_path.read_text(encoding="utf-8"))
        assert payload.get("token_alignment_heads"), heads_path
        # The expected direction string is the one the offline head-detector
        # writes. Catches accidental mismatches between filename and content.
        expected_direction = (
            f"{LANGUAGE_NAME_TO_CODE[source_lang]}-{LANGUAGE_NAME_TO_CODE[target_label]}"
        )
        assert payload.get("direction") == expected_direction, (heads_path, payload.get("direction"))


def test_shared_kernel_alignatt_heads_intersects_directions_and_averages_scores():
    heads_by_direction = {
        "en-de": [
            AlignAttHead(layer=11, head=3, ts=0.80),
            AlignAttHead(layer=6, head=5, ts=0.70),
            AlignAttHead(layer=9, head=0, ts=0.65),
        ],
        "en-it": [
            AlignAttHead(layer=11, head=3, ts=0.78),
            AlignAttHead(layer=6, head=5, ts=0.74),
            AlignAttHead(layer=2, head=1, ts=0.60),
        ],
        "en-zh": [
            AlignAttHead(layer=11, head=3, ts=0.82),
            AlignAttHead(layer=6, head=5, ts=0.66),
            AlignAttHead(layer=9, head=0, ts=0.64),
        ],
    }

    kernel = shared_kernel_alignatt_heads(heads_by_direction)

    identities = [(h.layer, h.head) for h in kernel]
    assert identities == [(11, 3), (6, 5)]
    # (11,3) mean is higher than (6,5) mean, so it must come first.
    assert kernel[0].ts > kernel[1].ts


def test_shared_kernel_alignatt_heads_returns_empty_when_no_full_overlap():
    heads_by_direction = {
        "en-de": [AlignAttHead(layer=1, head=0, ts=0.9)],
        "en-zh": [AlignAttHead(layer=2, head=0, ts=0.9)],
    }

    assert shared_kernel_alignatt_heads(heads_by_direction) == []


def test_multilingual_union_alignatt_heads_ranks_by_mean_ts_and_caps_budget():
    heads_by_direction = {
        "en-de": [
            AlignAttHead(layer=11, head=3, ts=0.80),
            AlignAttHead(layer=9, head=0, ts=0.50),
        ],
        "en-it": [
            AlignAttHead(layer=11, head=3, ts=0.80),
            AlignAttHead(layer=2, head=1, ts=0.60),
        ],
    }

    union = multilingual_union_alignatt_heads(heads_by_direction, max_heads=2)

    assert [(h.layer, h.head) for h in union] == [(11, 3), (2, 1)]
    # The shared head averages to 0.80, not 0.40, because we use mean not sum.
    assert abs(union[0].ts - 0.80) < 1e-9


def test_structured_prompt_header_tracks_source_language_label():
    rendered = ALIGNATT_PREFIX_TRANSLATION_VARIANT.render_messages(
        source_lang="English",
        target_lang="Chinese",
        text="because I have seen him",
        source_frontier=None,
        source_history=[],
        translation_history=[],
        is_partial=True,
        assistant_prefill="因为我",
    )

    current_user_message = rendered.messages[1]["content"]
    assert "[Current English ASR prefix]" in current_user_message
    assert rendered.messages[-1] == {"role": "assistant", "content": "因为我"}
