"""Pure-Python invariants for the Gemma audio alignment helpers.

These tests do not load any model; they lock the data-flow rules that
make the rest of the pipeline correct. We want:

- ``detect_audio_span`` actually finds a contiguous audio-token span
- ``audio_position_to_end_seconds`` respects the known 40 ms calibration
- ``_enforce_monotone`` is idempotent and non-decreasing
- ``split_text_into_word_spans`` matches Qwen's punctuation-stripping
  word-unit convention so the downstream cascade treats the two
  backends identically
- ``aggregate_token_timings_to_words`` turns per-token end-times into
  per-word end-times without introducing non-monotone regressions
"""

from __future__ import annotations


def test_detect_audio_span_returns_contiguous_range():
    from gemma_alignment_probe import detect_audio_span

    ids = [1, 2, 3, 258881, 258881, 258881, 7]
    span = detect_audio_span(ids, audio_token_id=258881, audio_ms_per_token=40.0)
    assert span is not None
    assert span.prompt_start == 3
    assert span.prompt_end == 6
    assert span.length == 3
    assert span.ms_per_token == 40.0


def test_detect_audio_span_returns_none_when_absent():
    from gemma_alignment_probe import detect_audio_span

    assert detect_audio_span([1, 2, 3], audio_token_id=258881, audio_ms_per_token=40.0) is None


def test_audio_position_uses_40ms_calibration():
    from gemma_alignment_probe import audio_position_to_end_seconds

    assert audio_position_to_end_seconds(0, ms_per_token=40.0, audio_duration_s=30.0) == 0.04
    assert audio_position_to_end_seconds(24, ms_per_token=40.0, audio_duration_s=30.0) == 1.0
    # Clamps to audio duration.
    assert audio_position_to_end_seconds(1000, ms_per_token=40.0, audio_duration_s=3.5) == 3.5
    assert audio_position_to_end_seconds(None, ms_per_token=40.0, audio_duration_s=1.0) is None


def test_enforce_monotone_projects_to_running_max():
    from gemma_alignment_probe import _enforce_monotone

    assert _enforce_monotone([0.1, 0.2, 0.15, 0.3, 0.25]) == [0.1, 0.2, 0.2, 0.3, 0.3]
    assert _enforce_monotone([0.0, None, 0.3, None, 0.2]) == [0.0, None, 0.3, None, 0.3]


def test_split_text_into_word_spans_strips_trailing_punctuation():
    from gemma_alignment_probe import split_text_into_word_spans

    spans = split_text_into_word_spans("Hello, world!")
    assert [span[2] for span in spans] == ["Hello", "world"]

    spans2 = split_text_into_word_spans("(hello)")
    assert [span[2] for span in spans2] == ["hello"]


def test_aggregate_token_timings_preserves_monotonicity():
    _TOKEN_TABLE = {
        10: "Hel",
        11: "lo",
        12: ",",
        13: " wor",
        14: "ld",
        15: "!",
    }

    class StubTokenizer:
        def decode(self, ids, skip_special_tokens=False):
            return "".join(_TOKEN_TABLE[i] for i in ids)

    generated_ids = [10, 11, 12, 13, 14, 15]
    # Per-token end times (monotone).
    token_ends = [0.12, 0.20, 0.20, 0.50, 0.80, 0.80]

    from gemma_alignment_probe import aggregate_token_timings_to_words

    words = aggregate_token_timings_to_words(
        "Hello, world!",
        generated_ids=generated_ids,
        tokenizer=StubTokenizer(),
        token_end_times_s=token_ends,
        audio_duration_s=1.0,
    )
    # Two words: "Hello", "world"
    assert [w.text for w in words] == ["Hello", "world"]
    # Each word's end time is monotone non-decreasing.
    assert words[0].end_time <= words[1].end_time
    assert words[0].end_time <= 0.20 + 1e-9
    assert words[1].end_time <= 0.80 + 1e-9


def test_monotonicity_score_rewards_forward_progress():
    from gemma_alignment_probe import monotonicity_score

    assert monotonicity_score([0, 1, 2, 3]) == 1.0
    assert monotonicity_score([0, 0, 0, 0]) == 1.0
    # One backward jump across three transitions -> 2/3.
    assert abs(monotonicity_score([0, 1, 0, 2]) - (2.0 / 3.0)) < 1e-9
    assert monotonicity_score([None, None]) == 0.0


def test_alignment_result_observer_tokens_roundtrip():
    from alignment_backend import (
        AlignAttObserverToken,
        AlignAttProvenanceBreakdown,
        AlignmentResult,
        WordAlignment,
    )
    from run_alignment_single_audio import deserialize_alignment, serialize_alignment

    result = AlignmentResult(
        text="hello",
        words=(WordAlignment(text="hello", start_time=0.1, end_time=0.2),),
        audio_duration_s=0.5,
        observer_tokens=(
            AlignAttObserverToken(
                token_id=42,
                token_str="hello",
                aligned_source_position=3,
                source_accessible_mass=0.75,
                blocked_source_local_position=4,
                blocked_source_unit_index=2,
                provenance=AlignAttProvenanceBreakdown(
                    source_accessible=0.75,
                    source_inaccessible=0.05,
                    non_source_prompt=0.10,
                    suffix=0.10,
                ),
            ),
        ),
        diagnostics={"backend": "gemma_onepass_qk_fast"},
    )

    roundtrip = deserialize_alignment(serialize_alignment(result))

    assert roundtrip.text == "hello"
    assert len(roundtrip.observer_tokens) == 1
    assert roundtrip.observer_tokens[0].token_id == 42
    assert roundtrip.observer_tokens[0].aligned_source_position == 3
    assert roundtrip.observer_tokens[0].source_accessible_mass == 0.75
    assert roundtrip.observer_tokens[0].provenance is not None
    assert roundtrip.observer_tokens[0].provenance.source_inaccessible == 0.05


def test_reconstruct_vllm_audio_attention_rows_matches_raw_qk_scores():
    import numpy as np
    from cascade_mt_backend import AlignAttHead
    from gemma_vllm_alignment_backend import reconstruct_vllm_audio_attention_rows

    rows, diagnostics = reconstruct_vllm_audio_attention_rows(
        {
            "layer_captures": {
                4: {
                    "selected_heads": [7],
                    "prompt_audio_k": np.asarray(
                        [[[1.0, 0.0], [0.0, 1.0]]],
                        dtype=np.float32,
                    ),
                    "decode_q": np.asarray(
                        [[[2.0, 5.0]]],
                        dtype=np.float32,
                    ),
                    "scaling": 1.0,
                }
            }
        },
        alignatt_heads=(AlignAttHead(layer=4, head=7, ts=0.0),),
    )

    assert diagnostics["effective_head_count"] == 1
    assert diagnostics["generated_token_count"] == 1
    assert len(rows) == 1
    assert tuple(rows[0].shape) == (1, 2)
    assert rows[0][0].tolist() == [2.0, 5.0]


def test_reconstruct_vllm_audio_attention_rows_reports_prompt_only_capture_gap():
    import numpy as np
    from cascade_mt_backend import AlignAttHead
    from gemma_vllm_alignment_backend import reconstruct_vllm_audio_attention_rows

    rows, diagnostics = reconstruct_vllm_audio_attention_rows(
        {
            "layer_captures": {
                4: {
                    "selected_heads": [7],
                    "prompt_audio_k": np.asarray(
                        [[[1.0, 0.0], [0.0, 1.0]]],
                        dtype=np.float32,
                    ),
                    "decode_q": None,
                    "scaling": 1.0,
                }
            }
        },
        alignatt_heads=(AlignAttHead(layer=4, head=7, ts=0.0),),
    )

    assert rows == []
    assert diagnostics["captured_layer_count"] == 1
    assert diagnostics["effective_head_count"] == 0
    assert diagnostics["generated_token_count"] == 0
    assert diagnostics["audio_span_length"] == 2
    assert diagnostics["missing_heads"] == [{"layer": 4, "head": 7}]


def test_prompt_observer_cache_entry_requires_complete_prompt_capture():
    import numpy as np
    from gemma_vllm_alignment_backend import _build_prompt_observer_cache_entry

    assert (
        _build_prompt_observer_cache_entry(
            {
                "prompt_length": 10,
                "audio_prompt_positions": [6, 7],
                "layer_captures": {
                    0: {
                        "prompt_audio_k": None,
                        "prompt_audio_missing_positions": [0, 1],
                    }
                },
            }
        )
        is None
    )

    entry = _build_prompt_observer_cache_entry(
        {
            "prompt_length": 10,
            "audio_prompt_positions": [6, 7],
            "layer_captures": {
                0: {
                    "prompt_audio_k": np.asarray([[[1.0], [2.0]]], dtype=np.float32),
                    "prompt_audio_missing_positions": [],
                }
            },
        }
    )

    assert entry is not None
    assert entry.prompt_length == 10
    assert entry.audio_prompt_positions == (6, 7)
    assert tuple(entry.layer_prompt_audio_k[0].shape) == (1, 2, 1)


def test_prompt_observer_cache_hydrates_missing_prompt_keys():
    import numpy as np
    from gemma_vllm_alignment_backend import (
        _build_prompt_observer_cache_entry,
        _hydrate_capture_payload_from_prompt_observer_cache,
    )

    cache_entry = _build_prompt_observer_cache_entry(
        {
            "prompt_length": 10,
            "audio_prompt_positions": [6, 7],
            "layer_captures": {
                0: {
                    "prompt_audio_k": np.asarray(
                        [[[1.0, 0.0], [0.0, 1.0]]],
                        dtype=np.float32,
                    ),
                    "prompt_audio_missing_positions": [],
                }
            },
        }
    )
    assert cache_entry is not None

    hydrated, diagnostics = _hydrate_capture_payload_from_prompt_observer_cache(
        {
            "prompt_length": 10,
            "audio_prompt_positions": [6, 7],
            "layer_captures": {
                0: {
                    "selected_heads": [7],
                    "prompt_audio_k": None,
                    "prompt_audio_missing_positions": [0, 1],
                    "decode_q": np.asarray([[[2.0, 5.0]]], dtype=np.float32),
                    "scaling": 1.0,
                    "head_dim": 2,
                }
            },
            "debug": {
                "layer_stats": {
                    "0": {
                        "selected_head_count": 1,
                        "prompt_audio_capture_count": 0,
                        "decode_q_count": 1,
                        "missing_prompt_audio_count": 2,
                    }
                }
            },
        },
        cache_entry=cache_entry,
    )

    assert hydrated is not None
    assert diagnostics == {
        "hit": True,
        "restored_layer_count": 1,
        "complete_after_restore": True,
    }
    assert tuple(hydrated["layer_captures"][0]["prompt_audio_k"].shape) == (1, 2, 2)
    assert hydrated["layer_captures"][0]["prompt_audio_missing_positions"] == []
    assert hydrated["debug"]["layer_stats"]["0"]["prompt_audio_capture_count"] == 2
    assert hydrated["debug"]["layer_stats"]["0"]["missing_prompt_audio_count"] == 0


def test_prompt_observer_cache_ignores_signature_mismatch():
    import numpy as np
    from gemma_vllm_alignment_backend import (
        _build_prompt_observer_cache_entry,
        _hydrate_capture_payload_from_prompt_observer_cache,
    )

    cache_entry = _build_prompt_observer_cache_entry(
        {
            "prompt_length": 10,
            "audio_prompt_positions": [6, 7],
            "layer_captures": {
                0: {
                    "prompt_audio_k": np.asarray([[[1.0], [2.0]]], dtype=np.float32),
                    "prompt_audio_missing_positions": [],
                }
            },
        }
    )
    assert cache_entry is not None

    hydrated, diagnostics = _hydrate_capture_payload_from_prompt_observer_cache(
        {
            "prompt_length": 11,
            "audio_prompt_positions": [6, 7],
            "layer_captures": {
                0: {
                    "prompt_audio_k": None,
                    "prompt_audio_missing_positions": [0, 1],
                }
            },
        },
        cache_entry=cache_entry,
    )

    assert hydrated is not None
    assert diagnostics == {
        "hit": False,
        "restored_layer_count": 0,
        "complete_after_restore": False,
    }
    assert hydrated["layer_captures"][0]["prompt_audio_k"] is None


def test_fetch_audio_qk_observer_payload_preserves_debug_state_and_clears_model():
    import numpy as np
    from gemma_vllm_alignment_backend import (
        _fetch_audio_qk_observer_from_model,
        _install_audio_qk_observer_on_model,
    )

    class FakeAttention:
        def __init__(self):
            self.forward = lambda *args, **kwargs: None
            self.scaling = 0.5
            self.head_dim = 2

    class FakeLayer:
        def __init__(self):
            self.self_attn = FakeAttention()

    class FakeInnerModel:
        def __init__(self):
            self.layers = [FakeLayer()]

    class FakeLanguageModel:
        def __init__(self):
            self.model = FakeInnerModel()

    class FakeModel:
        def __init__(self):
            self.language_model = FakeLanguageModel()

    model = FakeModel()
    install = _install_audio_qk_observer_on_model(
        model,
        selected_heads=[{"layer": 0, "head": 1}],
        audio_prompt_positions=[10, 11],
        prompt_length=12,
        patch_mode="postload_instance",
    )

    assert install == {
        "layer_count": 1,
        "audio_prompt_length": 2,
        "patch_mode": "postload_instance",
    }

    state = model._alignatt_audio_qk_state
    state["debug"] = {
        "forward_call_count": 7,
        "prompt_forward_call_count": 2,
        "decode_forward_call_count": 5,
        "positions_sample": [[0, 1, 2], [12]],
        "prompt_positions_sample": [[0, 1, 2]],
        "decode_positions_sample": [[12], [13]],
    }
    layer_state = state["layer_captures"][0]
    layer_state["prompt_audio_k_by_position"][10] = np.asarray(
        [[1.0, 0.0]],
        dtype=np.float32,
    )
    layer_state["prompt_audio_k_by_position"][11] = np.asarray(
        [[0.0, 1.0]],
        dtype=np.float32,
    )
    layer_state["decode_q"].append(
        np.asarray([[3.0, 4.0]], dtype=np.float32)
    )

    payload = _fetch_audio_qk_observer_from_model(model)

    assert payload is not None
    assert payload["debug"]["forward_call_count"] == 7
    assert payload["debug"]["prompt_forward_call_count"] == 2
    assert payload["debug"]["decode_forward_call_count"] == 5
    assert payload["debug"]["positions_sample"] == [[0, 1, 2], [12]]
    assert payload["debug"]["layer_stats"] == {
        "0": {
            "selected_head_count": 1,
            "prompt_audio_capture_count": 2,
            "decode_q_count": 1,
            "missing_prompt_audio_count": 0,
        }
    }
    assert tuple(payload["layer_captures"][0]["prompt_audio_k"].shape) == (1, 2, 2)
    assert tuple(payload["layer_captures"][0]["decode_q"].shape) == (1, 1, 2)
    assert model._alignatt_audio_qk_state is None
    assert model.language_model.model.layers[0].self_attn._alignatt_audio_qk_state is None
    assert model.language_model.model.layers[0].self_attn._alignatt_audio_qk_layer_idx is None


def test_tensor_buffer_observer_roundtrip_preserves_prompt_and_decode_buffers():
    import torch
    from gemma_vllm_alignment_backend import (
        _configure_audio_qk_tensor_observer_on_model,
        _fetch_audio_qk_tensor_observer_from_model,
        _prepare_audio_qk_tensor_observer_on_model,
    )

    class FakeProj:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class FakeAttention:
        def __init__(self):
            self.qkv_proj = FakeProj()
            self.num_heads = 4
            self.num_kv_heads = 2
            self.head_dim = 2
            self.scaling = 0.5

    class FakeLayer:
        def __init__(self):
            self.self_attn = FakeAttention()

    class FakeInnerModel:
        def __init__(self):
            self.layers = [FakeLayer()]

    class FakeLanguageModel:
        def __init__(self):
            self.model = FakeInnerModel()

    class FakeModel:
        def __init__(self):
            self.language_model = FakeLanguageModel()

    model = FakeModel()
    configure = _configure_audio_qk_tensor_observer_on_model(
        model,
        selected_heads=[{"layer": 0, "head": 1}],
        max_audio_tokens=4,
        max_decode_tokens=3,
    )
    assert configure == {
        "layer_count": 1,
        "max_audio_tokens": 4,
        "max_decode_tokens": 3,
        "storage_mode": "tensor_buffers",
    }

    prepare = _prepare_audio_qk_tensor_observer_on_model(
        model,
        prompt_length=10,
        audio_prompt_start=6,
        audio_prompt_length=4,
    )
    assert prepare == {
        "prompt_length": 10,
        "audio_prompt_start": 6,
        "audio_prompt_length": 4,
        "storage_mode": "tensor_buffers",
    }

    state = model._alignatt_audio_qk_state
    assert state == {
        "storage_mode": "tensor_buffers",
        "layer_indices": (0,),
    }
    observer = model.language_model.model.layers[0].self_attn._alignatt_audio_qk_tensor_observer
    assert set(dict(observer.named_buffers())) >= {
        "prompt_length_tensor",
        "audio_prompt_start_tensor",
        "audio_prompt_length_tensor",
        "forward_call_count_tensor",
        "prompt_audio_k_buffer",
        "decode_q_buffer",
    }
    assert observer.prompt_length_tensor.item() == 10
    assert observer.audio_prompt_start_tensor.item() == 6
    assert observer.audio_prompt_length_tensor.item() == 4
    observer.forward_call_count_tensor.fill_(9)
    observer.prompt_forward_call_count_tensor.fill_(3)
    observer.decode_forward_call_count_tensor.fill_(6)
    observer.prompt_audio_k_buffer[0, :4, :] = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]],
        dtype=torch.float32,
    )
    observer.prompt_written_buffer[:4] = True
    observer.decode_q_buffer[:2, 0, :] = torch.tensor(
        [[3.0, 4.0], [5.0, 6.0]],
        dtype=torch.float32,
    )
    observer.decode_written_buffer[:2] = True

    payload = _fetch_audio_qk_tensor_observer_from_model(model)

    assert payload is not None
    assert payload["prompt_length"] == 10
    assert payload["audio_prompt_positions"] == [6, 7, 8, 9]
    assert payload["debug"]["storage_mode"] == "tensor_buffers"
    assert payload["debug"]["forward_call_count"] == 9
    assert payload["debug"]["decode_forward_call_count"] == 6
    assert payload["debug"]["layer_stats"] == {
        "0": {
            "selected_head_count": 1,
            "prompt_audio_capture_count": 4,
            "decode_q_count": 2,
            "missing_prompt_audio_count": 0,
        }
    }
    assert tuple(payload["layer_captures"][0]["prompt_audio_k"].shape) == (1, 4, 2)
    assert tuple(payload["layer_captures"][0]["decode_q"].shape) == (2, 1, 2)


def test_tensor_buffer_observer_configuration_reuses_existing_module_when_compatible():
    import torch
    from gemma_vllm_alignment_backend import _configure_audio_qk_tensor_observer_on_model

    class FakeProj:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class FakeAttention:
        def __init__(self):
            self.qkv_proj = FakeProj()
            self.num_heads = 4
            self.num_kv_heads = 2
            self.head_dim = 2
            self.scaling = 0.5

    class FakeLayer:
        def __init__(self):
            self.self_attn = FakeAttention()

    class FakeInnerModel:
        def __init__(self):
            self.layers = [FakeLayer()]

    class FakeLanguageModel:
        def __init__(self):
            self.model = FakeInnerModel()

    class FakeModel:
        def __init__(self):
            self.language_model = FakeLanguageModel()

    model = FakeModel()
    _configure_audio_qk_tensor_observer_on_model(
        model,
        selected_heads=[{"layer": 0, "head": 1}],
        max_audio_tokens=4,
        max_decode_tokens=3,
    )
    observer = model.language_model.model.layers[0].self_attn._alignatt_audio_qk_tensor_observer

    _configure_audio_qk_tensor_observer_on_model(
        model,
        selected_heads=[{"layer": 0, "head": 1}],
        max_audio_tokens=4,
        max_decode_tokens=3,
    )
    reused = model.language_model.model.layers[0].self_attn._alignatt_audio_qk_tensor_observer

    assert reused is observer


def test_tensor_buffer_capture_ignores_out_of_span_positions_without_edge_corruption():
    import torch
    from gemma_vllm_alignment_backend import (
        _capture_audio_qk_into_tensor_buffers,
        _configure_audio_qk_tensor_observer_on_model,
        _prepare_audio_qk_tensor_observer_on_model,
    )

    class FakeProj:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class FakeAttention:
        def __init__(self):
            self.qkv_proj = FakeProj()
            self.num_heads = 4
            self.num_kv_heads = 2
            self.head_dim = 2
            self.scaling = 1.0

    class FakeLayer:
        def __init__(self):
            self.self_attn = FakeAttention()

    class FakeInnerModel:
        def __init__(self):
            self.layers = [FakeLayer()]

    class FakeLanguageModel:
        def __init__(self):
            self.model = FakeInnerModel()

    class FakeModel:
        def __init__(self):
            self.language_model = FakeLanguageModel()

    model = FakeModel()
    _configure_audio_qk_tensor_observer_on_model(
        model,
        selected_heads=[{"layer": 0, "head": 0}],
        max_audio_tokens=4,
        max_decode_tokens=2,
    )
    _prepare_audio_qk_tensor_observer_on_model(
        model,
        prompt_length=10,
        audio_prompt_start=6,
        audio_prompt_length=4,
    )

    attn = model.language_model.model.layers[0].self_attn
    positions = torch.tensor([5, 6, 8, 10, 12], dtype=torch.int64)
    q = torch.tensor(
        [
            [90.0, 91.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [10.0, 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [20.0, 21.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [30.0, 31.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [99.0, 98.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    k = torch.tensor(
        [
            [70.0, 71.0, 0.0, 0.0],
            [1.0, 2.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0],
            [80.0, 81.0, 0.0, 0.0],
            [88.0, 89.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    _capture_audio_qk_into_tensor_buffers(attn, positions, q, k)

    observer = attn._alignatt_audio_qk_tensor_observer
    assert observer.forward_call_count_tensor.item() == 1
    assert observer.prompt_forward_call_count_tensor.item() == 1
    assert observer.decode_forward_call_count_tensor.item() == 1
    assert observer.prompt_written_buffer.tolist() == [True, False, True, False]
    assert observer.decode_written_buffer.tolist() == [True, False]
    assert observer.prompt_audio_k_buffer[0, 0, :].tolist() == [1.0, 2.0]
    assert observer.prompt_audio_k_buffer[0, 2, :].tolist() == [3.0, 4.0]
    assert observer.decode_q_buffer[0, 0, :].tolist() == [30.0, 31.0]
    assert observer.decode_q_buffer[1, 0, :].tolist() == [0.0, 0.0]


def test_gemma_vllm_backend_builds_explicit_compilation_config():
    from pathlib import Path
    from types import SimpleNamespace

    from gemma_vllm_alignment_backend import GemmaVLLMAttentionAlignmentBackend

    backend = GemmaVLLMAttentionAlignmentBackend(
        model_name="stub-model",
        runtime_config=SimpleNamespace(
            gemma_vllm_compilation_mode="vllm_compile",
            gemma_vllm_cudagraph_mode="none",
            gemma_vllm_compile_cache_dir="tmp/test-vllm-cache",
            gemma_vllm_disable_compile_cache=True,
        ),
    )

    config = backend._build_compilation_config()

    assert config == {
        "mode": "vllm_compile",
        "cudagraph_mode": "none",
        "cache_dir": str(Path("tmp/test-vllm-cache").resolve()),
        "inductor_compile_config": {"force_disable_caches": True},
    }


def test_gemma_vllm_backend_builds_no_compilation_config_by_default():
    from types import SimpleNamespace

    from gemma_vllm_alignment_backend import GemmaVLLMAttentionAlignmentBackend

    backend = GemmaVLLMAttentionAlignmentBackend(
        model_name="stub-model",
        runtime_config=SimpleNamespace(),
    )

    assert backend._build_compilation_config() is None


def test_gemma_vllm_backend_disables_prefix_caching_by_default_for_observer_path():
    from types import SimpleNamespace

    from gemma_vllm_alignment_backend import GemmaVLLMAttentionAlignmentBackend

    backend = GemmaVLLMAttentionAlignmentBackend(
        model_name="stub-model",
        runtime_config=SimpleNamespace(),
    )

    assert backend.enable_prefix_caching is False


def test_gemma_vllm_backend_allows_explicit_prefix_caching_override():
    from types import SimpleNamespace

    from gemma_vllm_alignment_backend import GemmaVLLMAttentionAlignmentBackend

    backend = GemmaVLLMAttentionAlignmentBackend(
        model_name="stub-model",
        runtime_config=SimpleNamespace(gemma_vllm_enable_prefix_caching=True),
    )

    assert backend.enable_prefix_caching is True


def test_gemma_vllm_inspect_cli_accepts_compilation_overrides():
    from run_alignment_single_audio import build_cli

    parser = build_cli()
    args = parser.parse_args(
        [
            "gemma_vllm_inspect",
            "--wav",
            "tmp/alignatt_smoke18.wav",
            "--output",
            "tmp/out.json",
            "--vllm-executor-backend",
            "uni",
            "--vllm-patch-mode",
            "preload_class",
            "--no-vllm-enforce-eager",
            "--no-vllm-enable-prefix-caching",
            "--vllm-compilation-mode",
            "vllm_compile",
            "--vllm-cudagraph-mode",
            "none",
            "--vllm-compile-cache-dir",
            "tmp/cache",
            "--vllm-disable-compile-cache",
        ]
    )

    assert args.cmd == "gemma_vllm_inspect"
    assert args.vllm_executor_backend == "uni"
    assert args.vllm_patch_mode == "preload_class"
    assert args.vllm_enforce_eager is False
    assert args.vllm_enable_prefix_caching is False
    assert args.vllm_compilation_mode == "vllm_compile"
    assert args.vllm_cudagraph_mode == "none"
    assert args.vllm_compile_cache_dir == "tmp/cache"
    assert args.vllm_disable_compile_cache is True


def test_audio_too_long_raises_with_explicit_error():
    """Long-audio guard must fail loudly (PLAN.md Phase 5)."""
    import numpy as np
    from gemma_alignment_probe import (
        GemmaAttentionAlignmentBackend,
        GemmaAudioTooLongError,
    )

    # Build a backend without loading the model; only the guard is exercised.
    backend = GemmaAttentionAlignmentBackend.__new__(GemmaAttentionAlignmentBackend)
    backend.max_audio_seconds = 30.0
    audio = np.zeros(31 * 16000, dtype=np.float32)  # 31 s

    raised = False
    try:
        backend._enforce_audio_cap(audio, sample_rate=16000)
    except GemmaAudioTooLongError as exc:
        raised = True
        assert "31" in str(exc) and "30" in str(exc)
    assert raised, "audio past cap must raise GemmaAudioTooLongError, not silently truncate"

    # In-cap audio returns the duration without raising.
    short = np.zeros(5 * 16000, dtype=np.float32)
    assert abs(backend._enforce_audio_cap(short, sample_rate=16000) - 5.0) < 1e-6


def test_qk_fast_prefix_slicing_preserves_audio_features():
    import torch
    from gemma_alignment_probe import GemmaAttentionAlignmentBackend

    inputs = {
        "input_ids": torch.tensor([[11, 12, 13, 14]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        "audio_features": torch.randn(1, 80, 16),
        "non_tensor": "keep",
    }
    sliced = GemmaAttentionAlignmentBackend._slice_inputs_to_prefix(inputs, 2)

    assert sliced["input_ids"].tolist() == [[11, 12]]
    assert sliced["attention_mask"].tolist() == [[1, 1]]
    assert tuple(sliced["audio_features"].shape) == (1, 80, 16)
    assert sliced["non_tensor"] == "keep"


def test_gemma_onepass_backend_uses_public_runtime_id():
    from gemma_alignment_probe import GemmaAttentionAlignmentBackend

    assert GemmaAttentionAlignmentBackend.name == "gemma_onepass_qk_fast"


def test_gemma_vllm_backend_reset_caches_clears_prompt_observer_cache():
    from types import SimpleNamespace

    from gemma_vllm_alignment_backend import (
        GemmaVLLMAttentionAlignmentBackend,
        _PromptObserverCacheEntry,
    )

    backend = GemmaVLLMAttentionAlignmentBackend(
        model_name="stub-model",
        runtime_config=SimpleNamespace(),
    )
    backend._prompt_observer_cache["test"] = _PromptObserverCacheEntry(
        prompt_length=10,
        audio_prompt_positions=(6, 7),
        layer_prompt_audio_k={},
    )
    backend._last_generated_token_ids = [1, 2, 3]
    assert len(backend._prompt_observer_cache) == 1
    assert backend._last_generated_token_ids is not None

    backend.reset_caches()

    assert len(backend._prompt_observer_cache) == 0
    assert backend._last_generated_token_ids is None


def test_gemma_vllm_backend_decode_drift_reports_divergence():
    from types import SimpleNamespace

    from gemma_vllm_alignment_backend import GemmaVLLMAttentionAlignmentBackend

    backend = GemmaVLLMAttentionAlignmentBackend(
        model_name="stub-model",
        runtime_config=SimpleNamespace(),
    )
    # First call: no previous run.
    assert backend._compute_decode_drift([10, 20, 30]) is None

    # Identical second run.
    backend._last_generated_token_ids = [10, 20, 30]
    drift = backend._compute_decode_drift([10, 20, 30])
    assert drift is not None
    assert drift["identical"] is True
    assert drift["first_divergence_index"] is None

    # Divergent run.
    drift = backend._compute_decode_drift([10, 20, 99])
    assert drift is not None
    assert drift["identical"] is False
    assert drift["first_divergence_index"] == 2

    # Length change.
    drift = backend._compute_decode_drift([10, 20, 30, 40])
    assert drift is not None
    assert drift["identical"] is False
    assert drift["first_divergence_index"] == 3
    assert drift["prev_token_count"] == 3
    assert drift["current_token_count"] == 4


def test_gemma_vllm_inspect_cli_accepts_repeat_flag():
    from run_alignment_single_audio import build_cli

    parser = build_cli()
    args = parser.parse_args(
        [
            "gemma_vllm_inspect",
            "--wav", "tmp/alignatt_smoke18.wav",
            "--output", "tmp/out.json",
            "--repeat", "3",
        ]
    )
    assert args.repeat == 3


def test_seam_comparison_cli_parses_correctly():
    from run_alignment_single_audio import build_cli

    parser = build_cli()
    args = parser.parse_args(
        [
            "seam_comparison",
            "--wav", "tmp/alignatt_smoke18.wav",
            "--heads-path", "assets/heads.json",
            "--top-k", "4",
            "--output-dir", "tmp/seam_out",
        ]
    )
    assert args.cmd == "seam_comparison"
    assert args.wav == "tmp/alignatt_smoke18.wav"
    assert args.heads_path == "assets/heads.json"
    assert args.top_k == 4
    assert args.output_dir == "tmp/seam_out"


def test_gemma_vllm_backend_is_valid_alignment_backend_name():
    from cascade_runtime import (
        CascadeRuntimeConfig,
        STABLE_ALIGNMENT_BACKEND_NAMES,
        VALID_ALIGNMENT_BACKEND_NAMES,
    )

    assert "gemma_vllm_qk_fast" in VALID_ALIGNMENT_BACKEND_NAMES
    assert "gemma_vllm_qk_fast" not in STABLE_ALIGNMENT_BACKEND_NAMES
    # Config accepts the experimental name without raising.
    config = CascadeRuntimeConfig(alignment_backend_name="gemma_vllm_qk_fast")
    assert config.alignment_backend_name == "gemma_vllm_qk_fast"


def test_gemma_vllm_backend_runtime_config_defaults_match_validated_seam():
    from cascade_runtime import CascadeRuntimeConfig

    config = CascadeRuntimeConfig(alignment_backend_name="gemma_vllm_qk_fast")
    # Defaults must match the validated cudagraph=full seam (PLAN.md).
    assert config.gemma_vllm_enforce_eager is False
    assert config.gemma_vllm_enable_prefix_caching is False
    assert config.gemma_vllm_cudagraph_mode == "full"


def test_build_alignment_backend_dispatches_to_gemma_vllm():
    """Runtime dispatcher must produce a correctly-configured vLLM backend
    when alignment_backend_name='gemma_vllm_qk_fast'. Does not call load()."""
    from cascade_runtime import CascadeRuntimeConfig, build_alignment_backend
    from gemma_vllm_alignment_backend import GemmaVLLMAttentionAlignmentBackend

    config = CascadeRuntimeConfig(alignment_backend_name="gemma_vllm_qk_fast")
    backend = build_alignment_backend(config)
    assert isinstance(backend, GemmaVLLMAttentionAlignmentBackend)
    assert backend.name == "gemma_vllm_qk_fast"
    assert backend.enforce_eager is False
    assert backend.enable_prefix_caching is False
    assert backend.cudagraph_mode == "full"
    assert backend.worker_mode == "custom_tensor"


def _run_all() -> None:
    failures = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception as exc:  # pragma: no cover - surface every failure
                failures.append((name, exc))
            else:
                print(f"ok  {name}")
    if failures:
        print("")
        for name, exc in failures:
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    _run_all()
