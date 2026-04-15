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
    map_attention_head_to_key_value_head,
    resolve_prompt_and_suffix_key_states_for_layer,
    source_local_position_to_unit_index,
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
